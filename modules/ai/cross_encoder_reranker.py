"""
cross_encoder_reranker.py - Re-ranking de Alta Precisión con Cross-Encoder
==========================================================================
Implementa la Etapa 2 del pipeline de búsqueda híbrida:

  Fase 1 (Recuperación rápida): BM25 + Embeddings → top-50 candidatos
  Fase 2 (Re-ranking preciso):  Cross-Encoder evalúa query + artículo JUNTOS

El Cross-Encoder es significativamente más preciso que los Bi-encoders
(embeddings) porque procesa la query y el documento en la MISMA red neuronal
de atención, capturando dependencias cruzadas entre ambos textos.

Desventaja: Computacionalmente más pesado (~1s por artículo en CPU).
Por eso solo se aplica sobre los top-50 candidatos del Fase 1.

Modelo: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~22MB descargado
  - Entrenado en MS-MARCO (passage ranking)
  - Optimizado para relevancia académica/factual
  - Rápido en CPU (~100ms por par)
"""

import logging
import time
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Singleton del modelo para evitar recargas
_cross_encoder_model = None
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_cross_encoder():
    """Carga el modelo Cross-Encoder en singleton (lazy loading)."""
    global _cross_encoder_model
    if _cross_encoder_model is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"🔄 Cargando Cross-Encoder: {CROSS_ENCODER_MODEL}...")
            t0 = time.time()
            _cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL)
            logger.info(f"✅ Cross-Encoder cargado en {time.time()-t0:.1f}s")
        except Exception as e:
            logger.error(f"❌ No se pudo cargar Cross-Encoder: {e}")
            _cross_encoder_model = None
    return _cross_encoder_model


def rerank_with_cross_encoder(
    candidates: List[Dict],
    query: str,
    top_n: int = 50,
    batch_size: int = 32,
) -> List[Dict]:
    """
    Re-rankea los candidatos usando Cross-Encoder (query, documento) conjuntos.

    Args:
        candidates: Lista de artículos pre-filtrados por BM25+Embeddings
        query:      Pregunta de investigación original (en el idioma del investigador)
        top_n:      Solo re-rankea los top N candidatos (default: 50)
        batch_size: Tamaño de lote para inferencia en CPU

    Returns:
        Lista completa de candidatos reordenada:
          - Los top_n evaluados por Cross-Encoder van primero (mejor score primero)
          - El resto (sin evaluar) va al final en su orden original
    """
    model = _get_cross_encoder()
    if model is None:
        logger.warning("⚠️ Cross-Encoder no disponible. Retornando candidatos sin re-ranking.")
        return candidates

    if not candidates:
        return candidates

    # Solo evaluar los top_n (los demás ya son menos relevantes)
    to_rerank   = candidates[:top_n]
    rest        = candidates[top_n:]

    # Preparar pares (query, documento) para el Cross-Encoder
    pairs = []
    for art in to_rerank:
        title    = art.get('title', '') or ''
        abstract = art.get('abstract', '') or ''
        # Texto representativo: título + primeros 400 chars del abstract
        doc_text = f"{title}. {abstract[:400]}"
        pairs.append((query, doc_text))

    logger.info(f"🎯 Cross-Encoder: evaluando {len(pairs)} pares (query × artículo)...")
    t0 = time.time()

    try:
        # Cross-Encoder retorna logits crudos (no calibrados)
        # Mayor logit = más relevante para la query
        raw_scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        # Normalizar a [0, 1] usando sigmoid (transforma logits a probabilidades)
        # sigmoid(x) = 1 / (1 + e^(-x))
        ce_scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores)))

        elapsed = time.time() - t0
        logger.info(
            f"✅ Cross-Encoder completado en {elapsed:.1f}s | "
            f"Score promedio: {ce_scores.mean():.3f} | "
            f"Score máximo: {ce_scores.max():.3f}"
        )

        # Anotar score en cada artículo
        for i, art in enumerate(to_rerank):
            art['cross_encoder_score'] = float(round(ce_scores[i], 4))

        # Reordenar por Cross-Encoder score (descendente)
        to_rerank.sort(key=lambda x: x.get('cross_encoder_score', 0), reverse=True)

        # Actualizar 'similarity' con el score del cross-encoder como señal dominante
        # Combinamos: 40% score anterior (embeddings+fuzzy) + 60% cross-encoder
        for art in to_rerank:
            prev_sim = art.get('similarity', 0.5)
            ce_score = art.get('cross_encoder_score', 0.5)
            art['similarity'] = round(0.40 * prev_sim + 0.60 * ce_score, 4)

        # Registrar cuántos artículos superaron threshold de calidad
        high_quality = sum(1 for a in to_rerank if a.get('cross_encoder_score', 0) >= 0.5)
        logger.info(f"   📊 Artículos con CE score ≥ 0.5 (relevantes): {high_quality}/{len(to_rerank)}")

    except Exception as e:
        logger.error(f"❌ Error en Cross-Encoder predict: {e}. Retornando orden original.")
        return candidates

    # Combinar: re-rankeados primero, luego el resto sin evaluar
    return to_rerank + rest


def is_available() -> bool:
    """Verifica si el Cross-Encoder está disponible para usar."""
    try:
        from sentence_transformers import CrossEncoder
        return True
    except ImportError:
        return False
