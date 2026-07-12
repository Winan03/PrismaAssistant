"""
cross_encoder_reranker.py - Re-ranking Académico con LLM Fallback Chain
========================================================================
REEMPLAZA el Cross-Encoder clásico (ms-marco) por un pipeline de juicio
académico via LLM con fallback automático entre proveedores disponibles.

Por qué se abandonó el Cross-Encoder clásico:
  ❌ ms-marco entrenado en queries web (Bing/Google), NO en papers académicos
  ❌ CPU-only: 50 pares × ~300ms = ~15s por búsqueda (inaceptable)
  ❌ Sin soporte multilingüe real (3+ idiomas)
  ❌ Scores no calibrados entre modelos, combinación lineal 40/60 arbitraria

Nueva arquitectura — Fallback Chain (prioridad por disponibilidad):
  1️⃣  Ollama local        — sin límite de tokens, privado, gratuito
  2️⃣  Gemini 2.5 Flash    — 200 RPD gratis, rotación de hasta 5 keys
  3️⃣  Groq / LLaMA 3.3   — muy rápido, 100 RPD gratis
  4️⃣  Cerebras            — rotación de hasta 3 keys
  5️⃣  Fallback pasivo     — retorna similarity previa sin modificar

La interfaz pública (rerank_with_cross_encoder, is_available) es
IDÉNTICA al módulo original para no romper imports en el resto del proyecto.

Variables .env ya existentes que se usan (sin agregar nada nuevo):
  USE_OLLAMA_EMBEDDING, OLLAMA_EMBEDDING_BASE_URL, OLLAMA_MODEL
  GEMINI_API_KEY … GEMINI_API_KEY_5, GEMINI_MODEL
  GROQ_API_KEY
  CEREBRAS_API_KEY … CEREBRAS_API_KEY_3

Variables opcionales nuevas (con defaults seguros si no existen):
  LLM_RERANKER_TOP_N=20
  LLM_RERANKER_CONCURRENCY=5
  LLM_RERANKER_TIMEOUT=30
  LLM_RERANKER_TEXT_FIELDS=title,abstract,keywords
  LLM_RERANKER_MAX_ABSTRACT_TOKENS=300
  LLM_RERANKER_AUDIT_MODE=True
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lectura de .env / config.py (compatible con lo que ya tienes)
# ---------------------------------------------------------------------------

def _env(key: str, default: str = "") -> str:
    """Lee variable de entorno. Intenta config.py primero, luego os.environ."""
    try:
        import config as _c
        val = getattr(_c, key, None)
        if val is not None:
            return str(val).strip().strip('"').strip("'")
    except ImportError:
        pass
    return os.getenv(key, default).strip().strip('"').strip("'")


def _env_bool(key: str, default: bool = False) -> bool:
    return _env(key, str(default)).lower() in ("true", "1", "yes")


def _env_int(key: str, default: int = 0) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_list(key: str, default: str = "") -> List[str]:
    return [x.strip() for x in _env(key, default).split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Configuración (frozen = reproducible dentro de una corrida RSL)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RerankerConfig:
    """Todo leído desde .env. Frozen para garantizar reproducibilidad RSL."""

    # ── Ollama ──────────────────────────────────────────────────────────────
    use_ollama: bool         = field(default_factory=lambda: _env_bool("USE_OLLAMA_EMBEDDING", True))
    ollama_base_url: str     = field(default_factory=lambda: _env("OLLAMA_EMBEDDING_BASE_URL", "http://localhost:11434"))
    # Usa OLLAMA_MODEL para reranking; si no está configurado, Ollama se salta
    # (nomic-embed-text es embedding-only, no sirve para chat/reranking)
    ollama_rerank_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", ""))

    # ── Gemini ──────────────────────────────────────────────────────────────
    gemini_keys: Tuple[str, ...] = field(default_factory=lambda: tuple(filter(None, [
        _env("GEMINI_API_KEY"),
        _env("GEMINI_API_KEY_2"),
        _env("GEMINI_API_KEY_3"),
        _env("GEMINI_API_KEY_4"),
        _env("GEMINI_API_KEY_5"),
    ])))
    gemini_model: str        = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.5-flash"))

    # ── Groq ────────────────────────────────────────────────────────────────
    groq_key: str            = field(default_factory=lambda: _env("GROQ_API_KEY"))
    groq_model: str          = field(default_factory=lambda: "llama-3.3-70b-versatile")

    # ── Cerebras ────────────────────────────────────────────────────────────
    cerebras_keys: Tuple[str, ...] = field(default_factory=lambda: tuple(filter(None, [
        _env("CEREBRAS_API_KEY"),
        _env("CEREBRAS_API_KEY_2"),
        _env("CEREBRAS_API_KEY_3"),
    ])))
    cerebras_model: str      = field(default_factory=lambda: "llama-3.3-70b")

    # ── Parámetros del reranker ─────────────────────────────────────────────
    top_n: int               = field(default_factory=lambda: _env_int("LLM_RERANKER_TOP_N", 20))
    concurrency: int         = field(default_factory=lambda: _env_int("LLM_RERANKER_CONCURRENCY", 5))
    timeout_seconds: float   = field(default_factory=lambda: float(_env("LLM_RERANKER_TIMEOUT", "30")))
    text_fields: Tuple[str, ...] = field(default_factory=lambda: tuple(
        _env_list("LLM_RERANKER_TEXT_FIELDS", "title,abstract,keywords")
    ))
    max_abstract_tokens: int = field(default_factory=lambda: _env_int("LLM_RERANKER_MAX_ABSTRACT_TOKENS", 300))
    audit_mode: bool         = field(default_factory=lambda: _env_bool("LLM_RERANKER_AUDIT_MODE", True))


_cfg = RerankerConfig()


# ---------------------------------------------------------------------------
# Rotador de API keys
# ---------------------------------------------------------------------------

class _KeyRotator:
    """Rota entre múltiples keys del mismo proveedor cuando una devuelve 429."""

    def __init__(self, keys: Tuple[str, ...]):
        self._keys = list(keys)
        self._idx = 0

    def current(self) -> Optional[str]:
        return self._keys[self._idx % len(self._keys)] if self._keys else None

    def rotate(self) -> Optional[str]:
        if not self._keys:
            return None
        self._idx = (self._idx + 1) % len(self._keys)
        return self.current()

    def available(self) -> bool:
        return bool(self._keys)


_gemini_rotator   = _KeyRotator(_cfg.gemini_keys)
_cerebras_rotator = _KeyRotator(_cfg.cerebras_keys)


# ---------------------------------------------------------------------------
# Prompt del sistema (domain-agnostic, académico, multilingüe)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a rigorous academic research assistant specialized in systematic literature reviews (SLR).

Your ONLY task: evaluate whether a scientific article is relevant to a research query for inclusion in a systematic review.

Evaluate across 4 dimensions:
1. THEMATIC ALIGNMENT (0-1): Does the article directly address the research question?
2. METHODOLOGICAL RIGOR (0-1): Does it present empirical data, experiments, or validated methods?
3. KNOWLEDGE CONTRIBUTION (0-1): Does it add new findings beyond summarizing others?
4. SLR INCLUSION WORTHINESS (0-1): Would a rigorous reviewer include it in a systematic review on this topic?

STRICT RULES:
- Be completely domain-agnostic (applies equally to medicine, CS, social sciences, engineering, etc.)
- Respond ONLY with valid JSON. Zero text outside the JSON object.
- The "reason" field MUST be in the SAME LANGUAGE as the research query.
- Score 0.0 = completely irrelevant; 1.0 = perfectly relevant core paper.

Required JSON format (no other text):
{
  "score": <float 0.0-1.0>,
  "relevance_type": "<core|peripheral|methodological|background|irrelevant>",
  "dimensions": {
    "thematic": <float 0.0-1.0>,
    "methodological": <float 0.0-1.0>,
    "contribution": <float 0.0-1.0>,
    "slr_worthy": <float 0.0-1.0>
  },
  "reason": "<1-2 sentences in the query language explaining the score>"
}"""


# ---------------------------------------------------------------------------
# Construcción de texto (domain-agnostic, truncado por tokens estimados)
# ---------------------------------------------------------------------------

def _build_user_prompt(query: str, article: Dict, cfg: RerankerConfig) -> str:
    parts: List[str] = []
    token_count = 0

    for fname in cfg.text_fields:
        val = article.get(fname)
        if not val or not isinstance(val, str):
            continue
        val = val.strip()
        tokens = val.split()
        remaining = cfg.max_abstract_tokens - token_count
        if remaining <= 0:
            break
        if len(tokens) <= remaining:
            parts.append(f"{fname.upper()}: {val}")
            token_count += len(tokens)
        else:
            parts.append(f"{fname.upper()}: {' '.join(tokens[:remaining])} [...]")
            break

    doc_text = "\n".join(parts) if parts else "(no text available)"
    return (
        f"RESEARCH QUERY:\n{query}\n\n"
        f"ARTICLE TO EVALUATE:\n{doc_text}\n\n"
        "Respond with JSON only."
    )


# ---------------------------------------------------------------------------
# Parser de respuesta (robusto ante JSON imperfecto)
# ---------------------------------------------------------------------------

def _parse_response(raw: str, idx: int, prior: float = 0.5) -> Dict[str, Any]:
    """Extrae y valida el JSON de la respuesta del LLM."""
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', raw, re.DOTALL)
    if not match:
        logger.debug(f"Art #{idx}: no JSON encontrado → '{raw[:80]}'")
        return _fallback_result(prior)
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return _fallback_result(prior)

    try:
        score = max(0.0, min(1.0, float(parsed.get("score", 0.5))))
    except (TypeError, ValueError):
        score = 0.5

    # Score compuesto desde dimensiones (más robusto que score único)
    dims = parsed.get("dimensions", {})
    if all(k in dims for k in ("thematic", "methodological", "contribution", "slr_worthy")):
        try:
            composed = (
                0.35 * float(dims["thematic"]) +
                0.25 * float(dims["slr_worthy"]) +
                0.20 * float(dims["methodological"]) +
                0.20 * float(dims["contribution"])
            )
            score = round(0.5 * score + 0.5 * max(0.0, min(1.0, composed)), 6)
        except (TypeError, ValueError):
            pass

    return {
        "score":          score,
        "relevance_type": parsed.get("relevance_type", "unknown"),
        "dimensions":     dims,
        "reason":         parsed.get("reason", ""),
        "_error":         False,
    }


def _fallback_result(prior: float = 0.5) -> Dict[str, Any]:
    return {
        "score":          prior,
        "relevance_type": "fallback",
        "dimensions":     {},
        "reason":         "LLM evaluation failed, using prior score",
        "_error":         True,
    }


# ---------------------------------------------------------------------------
# Clientes async por proveedor
# ---------------------------------------------------------------------------

async def _try_ollama(
    client: httpx.AsyncClient, query: str, article: Dict,
    cfg: RerankerConfig, idx: int,
) -> Optional[Dict[str, Any]]:
    """1️⃣ Ollama local — sin límite de tokens, privado, siempre preferido."""
    model = cfg.ollama_rerank_model
    if not cfg.use_ollama or not model:
        return None

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_prompt(query, article, cfg)},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 256},
    }
    try:
        r = await client.post(f"{cfg.ollama_base_url}/api/chat",
                              json=payload, timeout=cfg.timeout_seconds)
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "")
        result = _parse_response(content, idx)
        if not result["_error"]:
            result["_provider"] = "ollama"
        return result
    except httpx.ConnectError:
        logger.debug("Ollama no disponible → siguiente proveedor.")
        return None
    except Exception as exc:
        logger.debug(f"Ollama error #{idx}: {exc}")
        return None


async def _try_gemini(
    client: httpx.AsyncClient, query: str, article: Dict,
    cfg: RerankerConfig, idx: int,
) -> Optional[Dict[str, Any]]:
    """2️⃣ Gemini — 200 RPD gratis, rotación automática de keys."""
    key = _gemini_rotator.current()
    if not key:
        return None

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models"
        f"/{cfg.gemini_model}:generateContent"
    )
    payload = {
        "system_instruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": _build_user_prompt(query, article, cfg)}]}],
        "generationConfig": {
            "temperature":      0.0,
            "maxOutputTokens":  256,
            "responseMimeType": "application/json",
        },
    }
    try:
        r = await client.post(url, json=payload,
                              params={"key": key}, timeout=cfg.timeout_seconds)
        if r.status_code == 429:
            _gemini_rotator.rotate()
            return None
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        result = _parse_response(text, idx)
        if not result["_error"]:
            result["_provider"] = "gemini"
        return result
    except Exception as exc:
        logger.debug(f"Gemini error #{idx}: {exc}")
        return None


async def _try_groq(
    client: httpx.AsyncClient, query: str, article: Dict,
    cfg: RerankerConfig, idx: int,
) -> Optional[Dict[str, Any]]:
    """3️⃣ Groq — LLaMA 3.3 70B, muy rápido."""
    if not cfg.groq_key:
        return None

    payload = {
        "model":            cfg.groq_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_prompt(query, article, cfg)},
        ],
        "temperature":       0.0,
        "max_tokens":        256,
        "response_format":  {"type": "json_object"},
    }
    try:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {cfg.groq_key}"},
            timeout=cfg.timeout_seconds,
        )
        if r.status_code == 429:
            return None
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        result = _parse_response(content, idx)
        if not result["_error"]:
            result["_provider"] = "groq"
        return result
    except Exception as exc:
        logger.debug(f"Groq error #{idx}: {exc}")
        return None


async def _try_cerebras(
    client: httpx.AsyncClient, query: str, article: Dict,
    cfg: RerankerConfig, idx: int,
) -> Optional[Dict[str, Any]]:
    """4️⃣ Cerebras — rotación automática de hasta 3 keys."""
    key = _cerebras_rotator.current()
    if not key:
        return None

    payload = {
        "model":       cfg.cerebras_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_prompt(query, article, cfg)},
        ],
        "temperature": 0.0,
        "max_tokens":  256,
    }
    try:
        r = await client.post(
            "https://api.cerebras.ai/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {key}"},
            timeout=cfg.timeout_seconds,
        )
        if r.status_code == 429:
            _cerebras_rotator.rotate()
            return None
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        result = _parse_response(content, idx)
        if not result["_error"]:
            result["_provider"] = "cerebras"
        return result
    except Exception as exc:
        logger.debug(f"Cerebras error #{idx}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Orquestador de un artículo: Ollama → Gemini → Groq → Cerebras → fallback
# ---------------------------------------------------------------------------

async def _score_one(
    client: httpx.AsyncClient,
    query: str,
    article: Dict,
    cfg: RerankerConfig,
    semaphore: asyncio.Semaphore,
    idx: int,
) -> Dict[str, Any]:
    """Evalúa un artículo recorriendo la cadena de fallback."""
    async with semaphore:
        prior = float(article.get("similarity", 0.5))

        for caller in (_try_ollama, _try_gemini, _try_groq, _try_cerebras):
            result = await caller(client, query, article, cfg, idx)
            if result and not result.get("_error"):
                return result

        logger.warning(f"⚠️  Art #{idx}: todos los proveedores fallaron → fallback pasivo ({prior:.3f})")
        r = _fallback_result(prior)
        r["_provider"] = "fallback"
        return r


async def _evaluate_batch(
    articles: List[Dict], query: str, cfg: RerankerConfig,
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(cfg.concurrency)
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(*[
            _score_one(client, query, art, cfg, sem, i)
            for i, art in enumerate(articles)
        ])


# ---------------------------------------------------------------------------
# Auditoría estructurada
# ---------------------------------------------------------------------------

@dataclass
class _Audit:
    n_evaluated: int = 0
    n_skipped: int = 0
    n_errors: int = 0
    elapsed: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    providers: Dict[str, int] = field(default_factory=dict)
    rel_types: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def log(self) -> None:
        logger.info(
            f"\n{'='*60}\n"
            f"📊 LLM Reranker Audit\n"
            f"{'='*60}\n"
            f"  Evaluados : {self.n_evaluated}  |  Omitidos: {self.n_skipped}  |  Errores: {self.n_errors}\n"
            f"  Tiempo    : {self.elapsed:.1f}s ({self.elapsed/max(self.n_evaluated,1):.1f}s/art)\n"
            f"  Scores    : mean={self.mean:.4f}  std={self.std:.4f}  "
            f"P25={self.p25:.4f}  P50={self.p50:.4f}  P75={self.p75:.4f}\n"
            f"  Proveedores : {self.providers}\n"
            f"  Relevancia  : {self.rel_types}\n"
            f"{'='*60}"
        )
        for w in self.warnings:
            logger.warning(f"  ⚠️  {w}")

    def check(self, scores: np.ndarray) -> None:
        if float(np.std(scores)) < 0.05:
            self.warnings.append("Scores muy concentrados (std<0.05): revisar prompt o modelo.")
        if float(np.mean(scores)) > 0.85:
            self.warnings.append("Score medio muy alto (>0.85): query posiblemente demasiado genérica.")
        fb = self.providers.get("fallback", 0)
        if fb / max(self.n_evaluated, 1) > 0.4:
            self.warnings.append(
                f"{fb} artículos usaron fallback pasivo. "
                "Verificar Ollama corriendo y API keys en .env."
            )


# ---------------------------------------------------------------------------
# API PÚBLICA — firma idéntica al módulo original
# ---------------------------------------------------------------------------

def rerank_with_cross_encoder(
    candidates: List[Dict],
    query: str,
    top_n: int = None,
    batch_size: int = None,
) -> List[Dict]:
    """
    Re-rankea candidatos usando LLM como juez académico.

    FIRMA IDÉNTICA al módulo original — no rompe ningún import existente.

    Cadena de proveedores (automática según disponibilidad):
      Ollama local → Gemini → Groq → Cerebras → fallback pasivo

    Args:
        candidates:  Artículos pre-filtrados por Fase 1 (BM25 + embeddings)
        query:       Query de investigación (cualquier idioma)
        top_n:       Candidatos a evaluar con LLM (default: LLM_RERANKER_TOP_N=20)
        batch_size:  Controla concurrencia async (default: LLM_RERANKER_CONCURRENCY=5)

    Returns:
        Lista completa reordenada. Campos añadidos a cada artículo evaluado:
          cross_encoder_score  — score 0-1 (nombre original, backward compatible)
          llm_relevance        — core | peripheral | methodological | background | irrelevant
          llm_reason           — justificación en el idioma de la query
          llm_dimensions       — subscores por dimensión académica
          llm_provider         — proveedor que generó el score
          similarity           — actualizado con score LLM (campo unificado downstream)
    """
    effective_top_n   = top_n    or _cfg.top_n
    effective_concurr = batch_size or _cfg.concurrency

    if not candidates:
        return candidates
    if not query or not query.strip():
        logger.warning("⚠️ Reranker: query vacía → retornando sin re-ranking.")
        return candidates

    to_rerank = candidates[:effective_top_n]
    rest      = candidates[effective_top_n:]

    # Override de concurrencia si difiere de la config global
    run_cfg = _cfg
    if effective_concurr != _cfg.concurrency:
        import copy
        run_cfg = copy.copy(_cfg)
        object.__setattr__(run_cfg, "concurrency", effective_concurr)

    logger.info(
        f"🧠 LLM Reranker: {len(to_rerank)} artículos | "
        f"cadena: Ollama→Gemini→Groq→Cerebras | "
        f"concurrencia={run_cfg.concurrency}"
    )

    t0 = time.perf_counter()
    results: List[Dict[str, Any]] = asyncio.run(
        _evaluate_batch(to_rerank, query, run_cfg)
    )
    elapsed = time.perf_counter() - t0

    scores: List[float] = []
    providers: Dict[str, int] = {}
    rel_types: Dict[str, int] = {}
    n_errors = 0

    for art, res in zip(to_rerank, results):
        score    = res["score"]
        provider = res.get("_provider", "fallback")
        rel      = res.get("relevance_type", "unknown")

        scores.append(score)
        providers[provider] = providers.get(provider, 0) + 1
        rel_types[rel]      = rel_types.get(rel, 0) + 1
        if res.get("_error"):
            n_errors += 1

        # Campos backward-compatible + nuevos
        art["cross_encoder_score"] = round(score, 6)
        art["llm_relevance"]       = rel
        art["llm_reason"]          = res.get("reason", "")
        art["llm_dimensions"]      = res.get("dimensions", {})
        art["llm_provider"]        = provider
        art["similarity"]          = round(score, 6)

    to_rerank.sort(key=lambda x: x.get("cross_encoder_score", 0.0), reverse=True)

    scores_arr = np.array(scores, dtype=float)
    audit = _Audit(
        n_evaluated=len(to_rerank),
        n_skipped=len(rest),
        n_errors=n_errors,
        elapsed=elapsed,
        mean=float(np.mean(scores_arr)),
        std=float(np.std(scores_arr)),
        p25=float(np.percentile(scores_arr, 25)),
        p50=float(np.percentile(scores_arr, 50)),
        p75=float(np.percentile(scores_arr, 75)),
        providers=providers,
        rel_types=rel_types,
    )
    audit.check(scores_arr)
    if _cfg.audit_mode:
        audit.log()

    return to_rerank + rest


def is_available() -> bool:
    """
    Verifica disponibilidad. Firma idéntica al módulo original.
    Retorna True si al menos un proveedor está configurado.
    """
    available = (
        (_cfg.use_ollama and bool(_cfg.ollama_rerank_model))
        or _gemini_rotator.available()
        or bool(_cfg.groq_key)
        or _cerebras_rotator.available()
    )
    if not available:
        logger.warning(
            "⚠️ LLM Reranker: ningún proveedor activo. "
            "Configura OLLAMA_MODEL, GEMINI_API_KEY, GROQ_API_KEY o CEREBRAS_API_KEY en .env"
        )
    return available


def export_scores_for_audit(articles: List[Dict]) -> List[Dict]:
    """
    Exporta campos de scoring para auditoría independiente de la RSL.
    Permite a revisores externos verificar criterios de inclusión/exclusión.
    """
    fields = (
        "id", "title", "year", "doi",
        "cross_encoder_score", "llm_relevance",
        "llm_reason", "llm_dimensions", "llm_provider", "similarity",
    )
    return [{f: art.get(f) for f in fields} for art in articles]