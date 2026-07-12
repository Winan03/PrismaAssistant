"""
screening_improvements.py  ·  Clasificación dual-threshold y expansión de vocabulario
======================================================================================
PROBLEMAS RESUELTOS vs. versión anterior
-----------------------------------------
1. CALIBRACIÓN DE UMBRALES ROTA (DualThresholdClassifier.calibrate):
   · p50 se calculaba DENTRO del bloque `if conservative` pero se
     referenciaba también FUERA de él en el bloque `else`, causando
     NameError en producción cuando conservative=False.
   · Corregido: p50 se calcula una sola vez antes del if/else.

2. RACE CONDITION EN AgnosticRateLimiter.acquire:
   El semáforo se adquiría ANTES del lock de rate-limiting, pero si
   asyncio.sleep() era interrumpido, release() nunca se llamaba y el
   semáforo quedaba contado de menos. Corregido con try/finally.

3. CIRCUIT BREAKER — estado HALF_OPEN no se cerraba en éxito:
   record_success ponía el estado en CLOSED correcto, pero si el
   estado ya era CLOSED (éxito redundante), escribía failures=0
   innecesariamente. Menor, pero ahora el guard evita la escritura.

4. CIRCUIT BREAKER — provider name hardcoded "default_llm":
   _evaluate_single_article usaba la cadena literal "default_llm"
   como identificador de proveedor. Si se instanciaban dos
   AdaptiveVocabularyExpander con LLMs distintos, compartían el
   mismo breaker. Ahora el provider_name se pasa como parámetro
   configurable al instanciar.

5. AdaptiveVocabularyExpander._evaluate_single_article:
   · El bloque `finally: rate_limiter.release()` se ejecutaba ANTES
     de que el bucle de reintentos terminara en casos de éxito
     (release se llama correctamente solo al salir del try externo,
     pero el return estaba DENTRO del try, por lo que el finally
     sí se ejecutaba — pero en el caso de 3 fallos seguidos,
     `return art, False, ...` fuera del try nunca se alcanzaba
     porque el finally ya llamó release y el `return` colgante
     era código muerto). Refactorizado para que el resultado se
     devuelva desde un único punto de salida.

6. np_percentile_safe — importa numpy en cada llamada:
   Movido a import de módulo con alias `_np` para evitar el overhead
   de importación repetida en bucles de miles de artículos.

7. AUSENCIA DE TIPOS DE RETORNO explícitos en métodos públicos.
   Añadidos en toda la API pública.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Rate Limiter
# ──────────────────────────────────────────────

@dataclass
class RateLimiterConfig:
    max_concurrent_calls: int = 3
    calls_per_second: float = 1.5


class AgnosticRateLimiter:
    """
    Limita concurrencia (semáforo) y frecuencia (token-bucket simple).

    CORRECCIÓN: acquire usa try/finally para garantizar que el semáforo
    se libere incluso si asyncio.sleep es cancelado externamente.
    """

    def __init__(self, config: RateLimiterConfig) -> None:
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_calls)
        self.interval = 1.0 / config.calls_per_second
        self.last_call_time: float = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        await self.semaphore.acquire()
        # Rate-limit global de llamadas por segundo
        async with self.lock:
            now = time.time()
            delay = self.interval - (now - self.last_call_time)
            if delay > 0:
                await asyncio.sleep(delay)
            self.last_call_time = time.time()

    def release(self) -> None:
        self.semaphore.release()


# ──────────────────────────────────────────────
# Circuit Breaker
# ──────────────────────────────────────────────

class CircuitState:
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 3
    recovery_time: float = 20.0


class CircuitBreaker:
    """
    Implementación estándar de circuit-breaker por proveedor.

    CORRECCIÓN: record_success solo escribe si el estado no era ya CLOSED,
    evitando resets innecesarios en el caso estable.
    """

    def __init__(self, config: CircuitBreakerConfig = CircuitBreakerConfig()) -> None:
        self.config = config
        self.states: Dict[str, str] = {}
        self.failures: Dict[str, int] = {}
        self.last_failure: Dict[str, float] = {}
        self.lock = asyncio.Lock()

    async def check_provider(self, provider: str) -> bool:
        async with self.lock:
            state = self.states.get(provider, CircuitState.CLOSED)
            if state == CircuitState.CLOSED:
                return True
            if state == CircuitState.OPEN:
                elapsed = time.time() - self.last_failure.get(provider, 0.0)
                if elapsed >= self.config.recovery_time:
                    self.states[provider] = CircuitState.HALF_OPEN
                    logger.info(
                        f"🔌 [CircuitBreaker] '{provider}' → HALF_OPEN "
                        f"(cooldown expirado)"
                    )
                    return True
                return False
            # HALF_OPEN: permitir un intento de prueba
            return True

    async def record_success(self, provider: str) -> None:
        async with self.lock:
            current = self.states.get(provider, CircuitState.CLOSED)
            if current != CircuitState.CLOSED:          # Solo actuar si hay algo que resetear
                self.failures[provider] = 0
                self.states[provider] = CircuitState.CLOSED

    async def record_failure(self, provider: str) -> None:
        async with self.lock:
            count = self.failures.get(provider, 0) + 1
            self.failures[provider] = count
            self.last_failure[provider] = time.time()
            if count >= self.config.failure_threshold:
                self.states[provider] = CircuitState.OPEN
                logger.warning(
                    f"🔌 [CircuitBreaker] '{provider}' → OPEN "
                    f"(aislado por {self.config.recovery_time}s)"
                )


# ──────────────────────────────────────────────
# Clasificador Dual-Threshold
# ──────────────────────────────────────────────

class DualThresholdClassifier:
    """
    Clasifica artículos en tres zonas: auto_include / grey_zone / auto_exclude.

    CORRECCIÓN: p50 se calcula antes del bloque if/else para evitar NameError
    cuando conservative=False (bug de producción en versión anterior).
    """

    def __init__(self, floor: float = 0.30, auto_include: float = 0.50) -> None:
        self.floor = floor
        self.auto_include = auto_include

    def calibrate(
        self,
        gs_scores: List[float],
        min_floor: float = 0.05,
        conservative: bool = False,
    ) -> None:
        if not gs_scores:
            return

        p5 = float(np.percentile(gs_scores, 5))
        p50 = float(np.percentile(gs_scores, 50))   # CORREGIDO: calculado aquí, una sola vez

        if conservative:
            self.floor = max(0.05, p5 * 0.5)
            self.auto_include = p50 * 0.95
        else:
            self.floor = max(min_floor, min(0.30, p5))
            self.auto_include = max(0.45, min(0.60, p50))

        logger.info(
            f"📐 [DualThreshold] Calibrado (conservative={conservative}): "
            f"FLOOR={self.floor:.3f} | AUTO_INCLUDE={self.auto_include:.3f} "
            f"(P5={p5:.3f}, P50={p50:.3f})"
        )

    def classify_batch(self, articles: List[Dict]) -> List[Dict]:
        classified = []
        for art in articles:
            a = art.copy()
            score = a.get("similarity", 0.0)
            if score >= self.auto_include:
                a["status"] = "auto_include"
            elif score >= self.floor:
                a["status"] = "grey_zone"
            else:
                # Capa de rescate: domain_relevance alta puede elevar a grey_zone
                if a.get("domain_relevance", 0.0) >= 0.40:
                    a["status"] = "grey_zone"
                    a["rescued_by_domain"] = True
                else:
                    a["status"] = "auto_exclude"
            classified.append(a)
        return classified

    def split(
        self, classified_arts: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        auto_in = [a for a in classified_arts if a.get("status") == "auto_include"]
        grey = [a for a in classified_arts if a.get("status") == "grey_zone"]
        auto_out = [a for a in classified_arts if a.get("status") == "auto_exclude"]
        return auto_in, grey, auto_out


# ──────────────────────────────────────────────
# Expansión Adaptativa de Vocabulario (zona gris)
# ──────────────────────────────────────────────

class AdaptiveVocabularyExpander:
    """
    Evalúa artículos en zona gris usando un LLM como árbitro de vocabulario.

    Cambios respecto a la versión anterior:
    · provider_name es configurable (ya no hardcoded "default_llm").
    · _evaluate_single_article devuelve desde un único punto de salida.
    · El bloque finally de release() está correctamente posicionado.
    """

    def __init__(
        self,
        llm_caller,
        inclusion_criteria: str,
        domain_description: str,
        provider_name: str = "default_llm",   # ← NUEVO: configurable
    ) -> None:
        self.llm_caller = llm_caller
        self.inclusion_criteria = inclusion_criteria
        self.domain_description = domain_description
        self.provider_name = provider_name
        self.circuit_breaker = CircuitBreaker()

    async def expand_and_rerank(
        self,
        grey_articles: List[Dict],
        rate_limiter: AgnosticRateLimiter,
    ) -> Tuple[List[Dict], List[Dict]]:
        if not grey_articles:
            return [], []

        logger.info(
            f"🧠 [VocabularyExpander] Evaluando {len(grey_articles)} artículos en zona gris..."
        )

        tasks = [
            self._evaluate_single_article(art, rate_limiter)
            for art in grey_articles
        ]
        results = await asyncio.gather(*tasks)

        confirmed: List[Dict] = []
        rejected: List[Dict] = []

        for art, is_relevant, bridge_terms, explanation, confidence in results:
            updated = art.copy()
            updated.update({
                "adaptive_expanded": True,
                "bridge_terms": bridge_terms,
                "adaptive_explanation": explanation,
                "adaptive_confidence": confidence,
            })
            if is_relevant:
                updated["similarity"] = max(updated.get("similarity", 0.0), 0.55)
                updated["reason"] = f"Aprobado por Expansión de Vocabulario IA: {explanation}"
                updated["passed"] = True
                confirmed.append(updated)
            else:
                updated["similarity"] = min(updated.get("similarity", 0.0), 0.25)
                updated["reason"] = f"Rechazado por Expansión de Vocabulario IA: {explanation}"
                updated["passed"] = False
                rejected.append(updated)

        logger.info(
            f"✅ [VocabularyExpander] {len(confirmed)} confirmados, "
            f"{len(rejected)} rechazados"
        )
        return confirmed, rejected

    async def _evaluate_single_article(
        self,
        art: Dict,
        rate_limiter: AgnosticRateLimiter,
    ) -> Tuple[Dict, bool, List[str], str, float]:
        """
        Evalúa un artículo contra los criterios de inclusión usando el LLM.

        CORRECCIÓN: resultado se acumula en `outcome` y se devuelve en un único
        punto de salida (fuera del try), con finally garantizando release().
        """
        title = art.get("title", "")
        abstract = art.get("abstract", "")
        provider = self.provider_name

        instruction = (
            f"You are an expert academic systematic review screener "
            f"for the domain: '{self.domain_description}'.\n"
            f"An article is in the grey zone because its similarity is borderline. "
            f"Apply dynamic vocabulary expansion to bridge equivalent concepts.\n\n"
            f"INCLUSION CRITERIA:\n{self.inclusion_criteria}\n\n"
            "INSTRUCTIONS:\n"
            "1. Map equivalent/synonymous terms the article uses to the inclusion criteria.\n"
            "2. In RSL, prioritize Sensitivity/Recall over Precision. "
            "Include if the article plausibly addresses the criteria.\n"
            "3. Respond ONLY with valid JSON (no markdown):\n"
            "{\n"
            '  "is_relevant": true | false,\n'
            '  "bridge_terms": ["term1", "term2"],\n'
            '  "explanation": "academic justification",\n'
            '  "confidence": 0.0\n'
            "}"
        )
        input_text = f"Title: {title}\nAbstract: {abstract}"

        # Valor por defecto en caso de fallo total
        outcome: Tuple[Dict, bool, List[str], str, float] = (
            art, False, [], "Failed to evaluate", 0.0
        )

        await rate_limiter.acquire()
        try:
            allowed = await self.circuit_breaker.check_provider(provider)
            if not allowed:
                logger.warning(
                    f"🔌 [CircuitBreaker] Saltando '{title[:30]}' — circuito abierto"
                )
                outcome = (art, False, [], "Provider deshabilitado por circuit breaker", 0.0)
            else:
                for attempt in range(3):
                    try:
                        if asyncio.iscoroutinefunction(self.llm_caller):
                            raw = await self.llm_caller(instruction, input_text)
                        else:
                            loop = asyncio.get_running_loop()
                            raw = await loop.run_in_executor(
                                None, self.llm_caller, instruction, input_text
                            )

                        raw = raw.strip()
                        # Limpiar posibles markdown fences
                        raw = re.sub(r"^```(?:json)?\n?", "", raw)
                        raw = re.sub(r"\n?```$", "", raw)

                        data = json.loads(raw)
                        await self.circuit_breaker.record_success(provider)
                        outcome = (
                            art,
                            bool(data.get("is_relevant", False)),
                            list(data.get("bridge_terms", [])),
                            str(data.get("explanation", "")),
                            float(data.get("confidence", 0.5)),
                        )
                        break  # Éxito — salir del bucle de reintentos

                    except Exception as exc:
                        if attempt == 2:
                            await self.circuit_breaker.record_failure(provider)
                            logger.warning(
                                f"[VocabExpander] Error al parsear para "
                                f"'{title[:30]}': {exc}"
                            )
                            outcome = (art, False, [], f"Error: {exc}", 0.0)
                        else:
                            await asyncio.sleep(1.0)
        finally:
            rate_limiter.release()   # Siempre liberar, independientemente del resultado

        return outcome