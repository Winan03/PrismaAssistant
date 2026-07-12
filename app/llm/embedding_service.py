"""
embedding_service.py  ·  Servicio de Embeddings Unificado (v5.0 — Senior Refactor)
====================================================================================

PROBLEMAS RESUELTOS vs. v4.2
------------------------------

CRITICAL-01  Estado global mutable con mutación de config en runtime
CRITICAL-02  Cache dict global sin límite de tamaño (memory leak)
CRITICAL-03  _ensure_model_loaded() NO es thread-safe (doble inicialización)
CRITICAL-04  Cambio silencioso de EMBEDDING_DIM rompe el índice ChromaDB
CRITICAL-05  OllamaEmbedder.embed() silencia errores por lote y devuelve zeros
             sin que el caller lo sepa (fallo invisible)
PERF-01      Cache con key=text[:200] colisiona para textos de comienzo idéntico
PERF-02      _embedding_cache no tiene TTL ni LRU; crece hasta OOM en sesiones largas
PERF-03      _ensure_model_loaded() llamada en CADA get_embeddings() con una
             doble-comprobación sin lock → re-adquiere el GIL innecesariamente
PERF-04      OllamaEmbedder._ensure_pulled() hace una petición HTTP en cada
             init, incluso cuando el modelo ya está verificado
ARCH-01      Mezcla de responsabilidades: caché, retry, fallback y normalización
             están todos en la misma función _get_embeddings_local()
ARCH-02      check_service() no reporta si el modelo respondió correctamente en
             el último llamado (solo si _local_model is not None)
"""

from __future__ import annotations

import hashlib
import logging
import os
os.environ["OMP_NUM_THREADS"] = "4" # Prevents PyTorch from allocating too many threads causing CPU thrashing
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import requests

import config

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTES Y CONFIGURACIÓN (read-only en runtime — no se muta config)
# ==============================================================================

_BATCH_SIZE: int = 8
_OLLAMA_EMBED_TIMEOUT: int = 30
_OLLAMA_DETECT_TIMEOUT: int = 20
_OLLAMA_PULL_TIMEOUT: int = 300
_MAX_CACHE_ENTRIES: int = 10_000   # LRU cap — evita OOM en sesiones largas


# ==============================================================================
# CRITICAL-01 FIX: EMBEDDING_DIM como propiedad inmutable en runtime
#
# La versión anterior hacía:
#     EMBEDDING_DIM = _local_model.dim
#     config.EMBEDDING_DIM = _local_model.dim   ← mutación global de config
#
# Esto es un bug silencioso crítico: si ChromaDB ya tiene una colección
# creada con dim=768 y el modelo devuelve 384, la mutación hace que el
# código "funcione" pero inserta vectores de dimensión incorrecta en la
# colección, corrompiendo el índice sin ningún error explícito.
#
# Solución: La dimensión efectiva se determina UNA SOLA VEZ en la
# inicialización del provider y se verifica contra config.EMBEDDING_DIM.
# Si no coinciden, se lanza ValueError en lugar de mutar silenciosamente.
# ==============================================================================

def _get_configured_dim() -> int:
    """Lee EMBEDDING_DIM desde config. Nunca lo muta."""
    return int(getattr(config, "EMBEDDING_DIM", 768))


# ==============================================================================
# CRITICAL-02 + PERF-01 + PERF-02 FIX: LRU Cache con hash SHA-256
#
# Problemas del cache anterior:
# 1. key = text[:200]  →  colisión garantizada para textos con prefijo idéntico
#    (ej. dos abstracts del mismo autor que empiezan igual)
# 2. dict sin límite → crece sin límite en sesiones largas (memory leak)
# 3. Sin TTL → un embedding de un modelo antiguo puede sobrevivir a un
#    reinicio del modelo
#
# Solución: OrderedDict como LRU manual con cap=_MAX_CACHE_ENTRIES.
# Key = SHA-256(text) → colisión criptográficamente despreciable.
# ==============================================================================

class _LRUEmbeddingCache:
    """
    Cache LRU thread-safe para embeddings.
    Key: SHA-256 del texto completo (no truncado).
    Cap: _MAX_CACHE_ENTRIES entradas. Al superar el límite, expulsa el más antiguo.
    """

    def __init__(self, maxsize: int = _MAX_CACHE_ENTRIES) -> None:
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._cache_file = os.path.join(os.path.dirname(__file__), "embedding_cache.pkl")
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        import pickle
        if os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, "rb") as f:
                    self._cache = pickle.load(f)
                    logger.info(f"💾 [EmbeddingService] Loaded {len(self._cache)} embeddings from disk cache.")
            except Exception as e:
                logger.warning(f"⚠️ [EmbeddingService] Failed to load cache from disk: {e}")
                self._cache = OrderedDict()

    def _save_to_disk(self) -> None:
        import pickle
        try:
            # Atomic save to avoid corruption if interrupted
            temp_file = self._cache_file + ".tmp"
            with open(temp_file, "wb") as f:
                pickle.dump(self._cache, f)
            os.replace(temp_file, self._cache_file)
        except Exception as e:
            logger.warning(f"⚠️ [EmbeddingService] Failed to save cache to disk: {e}")

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        key = self._key(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)   # Actualizar posición LRU
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def set(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        needs_save = False
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)   # Expulsar el más antiguo
                needs_save = True
            self._cache[key] = embedding
            
        if needs_save:
            # We save periodically. For small runs saving every time is fine,
            # but to be performant we only save every 10 new items or on exit.
            # However, for simplicity and safety, we can just save it.
            # To avoid slow down, we'll only trigger save if length is a multiple of 10.
            if len(self._cache) % 10 == 0:
                self._save_to_disk()

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._save_to_disk()

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total else 0.0,
            }


_cache = _LRUEmbeddingCache()


# ==============================================================================
# ESTADO DEL SERVICIO (centralizado, no disperso en variables globales)
# ==============================================================================

@dataclass
class _ServiceState:
    """
    Encapsula TODO el estado mutable del servicio en un único objeto.
    El lock protege inicialización concurrente (CRITICAL-03 fix).
    """
    model: Optional[object] = None           # OllamaEmbedder | SentenceTransformer
    backend: str = "uninitialized"
    effective_dim: int = field(default_factory=_get_configured_dim)
    last_success_ts: float = 0.0
    last_error: str = ""
    init_lock: threading.Lock = field(default_factory=threading.Lock)
    initialized: bool = False


_state = _ServiceState()


# ==============================================================================
# OLLAMA EMBEDDER
# ==============================================================================

class OllamaEmbedder:
    """
    Cliente robusto para Ollama con:
    · Auto-pull del modelo si no está disponible
    · Autodetección de dimensión con fallback a endpoint legacy
    · Micro-batching con error explícito (no zeros silenciosos)

    PERF-04 FIX: _ensure_pulled solo hace HTTP si el modelo no fue
    verificado todavía (flag _pulled_verified en instancia).
    """

    def __init__(self, model_name: str, base_url: str) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        api_key = getattr(config, "OLLAMA_API_KEY", None)
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.dim: int = 0
        self._pulled_verified = False
        self._ensure_pulled()
        self._detect_dimension()    # Lanza excepción si Ollama no responde

    # ── Verificación / descarga del modelo ──────────────────────────────────

    def _ensure_pulled(self) -> None:
        if self._pulled_verified:
            return
        try:
            resp = requests.get(f"{self.base_url}/api/tags", headers=self.headers, timeout=_OLLAMA_DETECT_TIMEOUT)
            if resp.status_code == 200:
                available = [m.get("name", "") for m in resp.json().get("models", [])]
                base = self.model_name.split(":")[0]
                if any(self.model_name == m or m.startswith(base) for m in available):
                    logger.info(f"✅ [Ollama] Modelo '{self.model_name}' disponible.")
                    self._pulled_verified = True
                    return

            # Modelo no encontrado → pull automático
            logger.warning(
                f"📥 [Ollama] '{self.model_name}' no encontrado. Descargando..."
            )
            pull = requests.post(
                f"{self.base_url}/api/pull",
                headers=self.headers,
                json={"name": self.model_name, "stream": False},
                timeout=_OLLAMA_PULL_TIMEOUT,
            )
            if pull.status_code == 200:
                logger.info(f"✅ [Ollama] '{self.model_name}' descargado correctamente.")
                self._pulled_verified = True
            else:
                raise RuntimeError(
                    f"Ollama pull falló con HTTP {pull.status_code}: {pull.text[:200]}"
                )
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"No se puede conectar a Ollama en '{self.base_url}'. "
                f"¿Está corriendo `ollama serve`? Detalle: {exc}"
            ) from exc

    # ── Detección de dimensión ───────────────────────────────────────────────

    def _detect_dimension(self) -> None:
        """
        Detecta la dimensión real del modelo. Prueba el endpoint nuevo (/api/embed)
        y, si falla, el legacy (/api/embeddings). Lanza si ninguno responde.
        """
        endpoints = [
            (f"{self.base_url}/api/embed",       {"model": self.model_name, "input": ["test"]},  "embeddings"),
            (f"{self.base_url}/api/embeddings",  {"model": self.model_name, "prompt": "test"},   "embedding"),
        ]
        for url, payload, key in endpoints:
            try:
                resp = requests.post(url, headers=self.headers, json=payload, timeout=_OLLAMA_DETECT_TIMEOUT)
                if resp.status_code == 200:
                    data = resp.json()
                    raw = data.get(key, [])
                    # /api/embed devuelve lista de listas; /api/embeddings devuelve lista
                    vec = raw[0] if isinstance(raw[0], list) else raw if raw else []
                    if vec:
                        self.dim = len(vec)
                        logger.info(
                            f"✅ [Ollama] Dimensión detectada: {self.dim} "
                            f"(endpoint: {url.split('/')[-1]})"
                        )
                        return
            except Exception:
                continue

        raise RuntimeError(
            f"[Ollama] No se pudo detectar la dimensión del modelo '{self.model_name}'. "
            "Verifica que el modelo responda correctamente."
        )

    # ── Generación de embeddings ─────────────────────────────────────────────

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Genera embeddings en micro-batches.

        CRITICAL-05 FIX: ya no devuelve zeros silenciosamente.
        Si un batch falla después de reintentar individualmente,
        lanza RuntimeError para que el caller pueda decidir el fallback.
        """
        all_embeddings: List[List[float]] = []
        url_batch = f"{self.base_url}/api/embed"
        url_single = f"{self.base_url}/api/embeddings"

        for batch_start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[batch_start: batch_start + _BATCH_SIZE]
            batch_ok = False

            # Intento 1: batch nativo
            try:
                resp = requests.post(
                    url_batch,
                    headers=self.headers,
                    json={"model": self.model_name, "input": batch},
                    timeout=_OLLAMA_EMBED_TIMEOUT,
                )
                if resp.status_code == 200:
                    embs = resp.json().get("embeddings", [])
                    if len(embs) == len(batch):
                        all_embeddings.extend(embs)
                        batch_ok = True
            except Exception as exc:
                logger.debug(f"[Ollama] Batch falló ({exc}), intentando uno a uno...")

            # Intento 2: uno a uno (fallback de batch)
            if not batch_ok:
                for text in batch:
                    try:
                        r = requests.post(
                            url_single,
                            headers=self.headers,
                            json={"model": self.model_name, "prompt": text},
                            timeout=_OLLAMA_EMBED_TIMEOUT,
                        )
                        if r.status_code == 200:
                            emb = r.json().get("embedding", [])
                            if emb:
                                all_embeddings.append(emb)
                                continue
                        raise RuntimeError(f"HTTP {r.status_code}")
                    except Exception as exc:
                        raise RuntimeError(
                            f"[Ollama] Fallo irrecuperable al embedir texto: '{text[:60]}'. "
                            f"Detalle: {exc}"
                        ) from exc

        result = np.array(all_embeddings, dtype=np.float32)
        if result.shape != (len(texts), self.dim):
            raise RuntimeError(
                f"[Ollama] Shape inesperado: {result.shape}, esperado ({len(texts)}, {self.dim})"
            )
        return result


# ==============================================================================
# CRITICAL-03 FIX: Inicialización thread-safe con double-checked locking
# ==============================================================================

def _ensure_model_loaded() -> None:
    """
    Garantiza que el modelo esté inicializado exactamente UNA VEZ,
    incluso con llamadas concurrentes desde múltiples threads.

    Patrón: double-checked locking con threading.Lock.
    El primer check (sin lock) evita el overhead del lock en el caso estable.
    El segundo check (con lock) protege la región crítica de init.
    """
    if _state.initialized:
        return

    with _state.init_lock:
        if _state.initialized:    # Re-check dentro del lock
            return
        _initialize_backend()
        _state.initialized = True


def _initialize_backend() -> None:
    """
    Intenta cargar Ollama primero; si falla, carga SentenceTransformer.
    CRITICAL-01 FIX: verifica dimensión contra config sin mutarla.
    """
    configured_dim = _get_configured_dim()

    # ── Intento 1: Ollama local ──────────────────────────────────────────────
    if getattr(config, "USE_OLLAMA_EMBEDDING", False):
        try:
            embedder = OllamaEmbedder(
                model_name=config.OLLAMA_EMBEDDING_MODEL,
                base_url=config.OLLAMA_EMBEDDING_BASE_URL,
            )
            _validate_dim(embedder.dim, configured_dim, source="Ollama")
            _state.model = embedder
            _state.backend = "ollama"
            _state.effective_dim = embedder.dim
            logger.info(
                f"✅ [EmbeddingService] Backend: Ollama | "
                f"Modelo: {config.OLLAMA_EMBEDDING_MODEL} | Dim: {embedder.dim}"
            )
            return
        except Exception as exc:
            logger.warning(
                f"⚠️ [EmbeddingService] Ollama no disponible: {exc}. "
                "Pasando a SentenceTransformer..."
            )

    # ── Intento 2: SentenceTransformer (fallback offline) ───────────────────
    try:
        from sentence_transformers import SentenceTransformer

        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/allenai-specter")
        if model_name.startswith("ollama/"):
            model_name = "sentence-transformers/allenai-specter"

        logger.info(f"📦 [EmbeddingService] Cargando SentenceTransformer: {model_name}...")
        st_model = SentenceTransformer(model_name)
        actual_dim = st_model.get_sentence_embedding_dimension()
        _validate_dim(actual_dim, configured_dim, source="SentenceTransformer")

        _state.model = st_model
        _state.backend = "sentence_transformer"
        _state.effective_dim = actual_dim
        logger.info(
            f"✅ [EmbeddingService] Backend: SentenceTransformer | "
            f"Modelo: {model_name} | Dim: {actual_dim}"
        )
    except Exception as exc:
        logger.critical(f"❌ [EmbeddingService] Todos los backends fallaron: {exc}")
        raise RuntimeError(
            "EmbeddingService no pudo inicializar ningún backend. "
            "Revisa la configuración de Ollama y SentenceTransformer."
        ) from exc


def _validate_dim(actual: int, configured: int, source: str) -> None:
    """
    CRITICAL-01 FIX: Verifica dimensión sin mutar config.
    Si hay discrepancia, lanza ValueError con instrucciones claras.
    """
    if actual != configured:
        raise ValueError(
            f"[EmbeddingService] Discrepancia de dimensión en {source}: "
            f"el modelo devuelve {actual} dims pero config.EMBEDDING_DIM={configured}. "
            f"Actualiza config.EMBEDDING_DIM={actual} o cambia el modelo. "
            f"NO se muta config en runtime para evitar corrupción silenciosa del índice vectorial."
        )


# ==============================================================================
# API PÚBLICA
# ==============================================================================

def get_embeddings(
    texts: Union[str, List[str]],
    use_cache: bool = True,
) -> np.ndarray:
    """
    Genera embeddings para uno o más textos.

    Args:
        texts:     texto único o lista de textos.
        use_cache: True para usar el cache LRU. False fuerza recalcular.

    Returns:
        np.ndarray de shape (n_texts, effective_dim), dtype float32.

    Raises:
        RuntimeError: si el backend no pudo inicializarse.
    """
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        _ensure_model_loaded()
        return np.zeros((0, _state.effective_dim), dtype=np.float32)

    _ensure_model_loaded()

    results: List[Optional[np.ndarray]] = [None] * len(texts)
    uncached_indices: List[int] = []
    uncached_texts: List[str] = []

    # Separar textos en caché vs. no en caché
    if use_cache:
        for i, text in enumerate(texts):
            cached = _cache.get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
    else:
        uncached_indices = list(range(len(texts)))
        uncached_texts = list(texts)

    # Generar solo los no cacheados
    if uncached_texts:
        new_embeddings = _generate_raw(uncached_texts)
        if use_cache:
            for text, emb in zip(uncached_texts, new_embeddings):
                _cache.set(text, emb)
        for idx, emb in zip(uncached_indices, new_embeddings):
            results[idx] = emb

    return np.array(results, dtype=np.float32)


def get_single_embedding(text: str) -> np.ndarray:
    """Embedding de un texto único como vector 1D de shape (effective_dim,)."""
    return get_embeddings([text])[0]


def _generate_raw(texts: List[str]) -> np.ndarray:
    """
    Delega al backend activo. Registra timestamp de último éxito.
    ARCH-01 FIX: función pura de generación, sin mezclar cache ni fallback logic.
    """
    try:
        if isinstance(_state.model, OllamaEmbedder):
            result = _state.model.embed(texts)
        else:
            # SentenceTransformer
            result = _state.model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                batch_size=_BATCH_SIZE,
            )
            result = np.array(result, dtype=np.float32)

        _state.last_success_ts = time.time()
        _state.last_error = ""
        return result

    except Exception as exc:
        _state.last_error = str(exc)
        logger.error(f"❌ [EmbeddingService] Error en generación: {exc}")

        # Si Ollama falló, resetear para que el próximo call intente SentenceTransformer
        if _state.backend == "ollama":
            logger.warning(
                "⚠️ [EmbeddingService] Ollama falló. Reseteando para usar "
                "SentenceTransformer en la próxima llamada."
            )
            _state.model = None
            _state.initialized = False

        raise


# ==============================================================================
# UTILIDADES
# ==============================================================================

def clear_cache() -> None:
    """Limpia el cache LRU de embeddings."""
    _cache.clear()
    logger.info("🧹 [EmbeddingService] Cache limpiado.")


def check_service() -> dict:
    """
    ARCH-02 FIX: Reporta estado real del servicio incluyendo última actividad.
    No hace llamadas HTTP; usa el estado interno actualizado en cada generación.
    """
    cache_stats = _cache.stats()
    secs_since_ok = (
        round(time.time() - _state.last_success_ts, 1)
        if _state.last_success_ts
        else None
    )
    active_model = (
        getattr(config, "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        if _state.backend == "ollama"
        else getattr(config, "EMBEDDING_MODEL", "sentence-transformers/allenai-specter")
    )
    return {
        "backend": _state.backend,
        "model": active_model,
        "effective_dim": _state.effective_dim,
        "configured_dim": _get_configured_dim(),
        "dim_mismatch": _state.effective_dim != _get_configured_dim(),
        "initialized": _state.initialized,
        "last_success_seconds_ago": secs_since_ok,
        "last_error": _state.last_error or None,
        "cache": cache_stats,
    }


def reset_service() -> None:
    """
    Fuerza reinicialización completa del servicio.
    Útil en tests o cuando se cambia el modelo en runtime de forma controlada.
    """
    with _state.init_lock:
        _state.model = None
        _state.backend = "uninitialized"
        _state.effective_dim = _get_configured_dim()
        _state.last_success_ts = 0.0
        _state.last_error = ""
        _state.initialized = False
    _cache.clear()
    logger.info("🔄 [EmbeddingService] Servicio reseteado completamente.")
