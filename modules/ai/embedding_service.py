"""
embedding_service.py - Servicio de Embeddings Unificado (v4.0)

Estrategia para produccion (local-first):
  1. PRIMARIO: SentenceTransformer local (all-mpnet-base-v2)
     - En HF Spaces Docker: descarga ~420MB la primera vez, luego queda en cache
     - En produccion con 16GB RAM: funciona perfectamente
  2. APIS opcionales: Voyage AI / HuggingFace (si se configuran y el local falla)

768 dims -> ChromaDB compatible.
"""

import logging
import time
import numpy as np
from typing import List, Union
import config

# ==============================================================================
# CONFIGURACION
# ==============================================================================

EMBEDDING_DIM = 768
BATCH_SIZE = 64  # v17.1: Subido de 20 → 64 para reducir overhead de bucle CPU (3x menos iteraciones)

# Cache en memoria para evitar llamadas repetidas al mismo texto
_embedding_cache: dict = {}

# Modelo local (singleton)
_local_model = None

# ==============================================================================
# FUNCION PRINCIPAL
# ==============================================================================

def get_embeddings(texts: Union[str, List[str]], use_cache: bool = True) -> np.ndarray:
    """
    Genera embeddings para uno o mas textos.
    Returns: np.ndarray shape (n_texts, 768).
    """
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return np.zeros((0, EMBEDDING_DIM))

    # Revisar cache
    results = [None] * len(texts)
    uncached_indices, uncached_texts = [], []

    if use_cache:
        for i, text in enumerate(texts):
            key = text[:200]
            if key in _embedding_cache:
                results[i] = _embedding_cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
    else:
        uncached_indices = list(range(len(texts)))
        uncached_texts = list(texts)

    if not uncached_texts:
        return np.array(results)

    # Generar embeddings
    new_embeddings = _get_embeddings_local(uncached_texts)

    if new_embeddings is None:
        logging.error("Embedding service: modelo local fallo. Devolviendo zeros.")
        new_embeddings = np.zeros((len(uncached_texts), EMBEDDING_DIM))

    # Poblar cache
    if use_cache:
        for i, emb in zip(uncached_indices, new_embeddings):
            _embedding_cache[texts[i][:200]] = emb

    for i, emb in zip(uncached_indices, new_embeddings):
        results[i] = emb

    return np.array(results)


def get_single_embedding(text: str) -> np.ndarray:
    """Embedding de un texto unico como vector 1D (768,)."""
    return get_embeddings([text])[0]


# ==============================================================================
# MODELO LOCAL: SentenceTransformer (all-mpnet-base-v2)
# ==============================================================================

def _get_embeddings_local(texts: List[str]) -> np.ndarray:
    """
    Genera embeddings usando SentenceTransformer local.
    En HF Spaces: el modelo se descarga de HuggingFace Hub una sola vez (~420MB)
    y queda en cache. Con 16GB RAM del free tier no hay problema.
    """
    global _local_model

    try:
        from sentence_transformers import SentenceTransformer

        if _local_model is None:
            logging.info(f"Cargando modelo de embeddings: {config.EMBEDDING_MODEL}...")
            _local_model = SentenceTransformer(config.EMBEDDING_MODEL)
            logging.info(f"Modelo cargado: {config.EMBEDDING_MODEL} (768 dims)")

        embeddings = _local_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=BATCH_SIZE
        )
        return np.array(embeddings, dtype=np.float32)

    except Exception as e:
        logging.error(f"Error en modelo local de embeddings: {e}")
        return None


# ==============================================================================
# UTILIDADES
# ==============================================================================

def clear_cache():
    """Limpia el cache de embeddings."""
    global _embedding_cache
    _embedding_cache.clear()
    logging.info("Cache de embeddings limpiado.")


def check_service() -> dict:
    """Retorna el estado del servicio."""
    return {
        "backend": "local",
        "model": config.EMBEDDING_MODEL,
        "dims": EMBEDDING_DIM,
        "cached_texts": len(_embedding_cache),
        "model_loaded": _local_model is not None
    }
