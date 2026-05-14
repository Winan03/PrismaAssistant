"""
bm25_retriever.py - Recuperación Léxica BM25 + Reciprocal Rank Fusion (RRF)
============================================================================
Implementa el componente de Sparse Retrieval de la búsqueda híbrida.

Flujo:
  1. BM25Okapi indexa los textos del corpus (títulos + abstracts)
  2. Para cada query semántica, obtiene un ranking léxico BM25
  3. RRF fusiona el ranking BM25 con el ranking de embeddings
     → Artículos bien posicionados en AMBOS rankings suben significativamente
     → Falsos positivos (solo semánticos) bajan porque no contienen la terminología exacta

Referencias:
  - Robertson & Zaragoza (2009): "The Probabilistic Relevance Framework: BM25 and Beyond"
  - Cormack et al. (2009): "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
"""

import logging
import re
import unicodedata
from typing import List, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTES RRF
# ============================================================
RRF_K = 60  # Constante de suavizado RRF (Cormack 2009 recomienda 60)


# ============================================================
# TOKENIZADOR CIENTÍFICO
# ============================================================

def _tokenize_scientific(text: str) -> List[str]:
    """
    Tokenizador optimizado para textos académicos.
    - Conserva acrónimos (LLM, SAST, NLP)
    - Descompone CamelCase (CodeBERT → ['code', 'bert'])
    - Elimina stopwords en EN + ES
    - Conserva términos compuestos con guión (fine-tuning → ['fine', 'tuning', 'finetuning'])
    """
    if not text:
        return []

    STOPWORDS_EN = {
        'the', 'a', 'an', 'and', 'or', 'not', 'in', 'on', 'of', 'to', 'for',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do',
        'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'this', 'that', 'these', 'those', 'with', 'from', 'by', 'at', 'as',
        'its', 'we', 'our', 'their', 'also', 'can', 'show', 'paper', 'work',
        'propose', 'present', 'results', 'based', 'using', 'used', 'use',
        'method', 'approach', 'study', 'research', 'novel', 'new', 'improve',
    }
    STOPWORDS_ES = {
        'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'en', 'con',
        'por', 'para', 'que', 'es', 'son', 'se', 'al', 'su', 'sus',
    }
    STOPWORDS = STOPWORDS_EN | STOPWORDS_ES

    # Normalizar acentos
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = text.lower()

    tokens = []

    # 1. Preservar acrónimos completos antes de fragmentar
    acronyms = re.findall(r'\b[a-z]{2,6}\b', text)

    # 2. Dividir por caracteres no alfanuméricos (excepto guión)
    words = re.split(r'[^\w\-]+', text)

    for word in words:
        word = word.strip('-').strip()
        if not word or len(word) < 2:
            continue

        # Expandir palabras con guión: fine-tuning → [fine, tuning, finetuning]
        if '-' in word:
            parts = word.split('-')
            tokens.extend(p for p in parts if len(p) >= 2 and p not in STOPWORDS)
            combined = ''.join(parts)
            if len(combined) >= 4:
                tokens.append(combined)
            continue

        if word not in STOPWORDS and len(word) >= 2:
            tokens.append(word)

    return tokens


# ============================================================
# BM25 RETRIEVER
# ============================================================

class BM25Retriever:
    """
    Encapsula un índice BM25Okapi sobre un corpus de textos.
    Thread-safe para su uso en el pipeline de screening.
    """

    def __init__(self, texts: List[str]):
        """
        Args:
            texts: Lista de strings (título + abstract de cada artículo).
                   El índice i en 'texts' corresponde al índice i en 'articles'.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank_bm25 no está instalado. Ejecuta: pip install rank-bm25"
            )

        self.n_docs = len(texts)
        tokenized_corpus = [_tokenize_scientific(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"🗂️ BM25 indexado: {self.n_docs} documentos")

    def get_scores(self, query: str) -> np.ndarray:
        """
        Retorna scores BM25 para todos los documentos dado un query.
        Shape: (n_docs,) — mayor score = más relevante léxicamente.
        """
        tokens = _tokenize_scientific(query)
        if not tokens:
            return np.zeros(self.n_docs)
        scores = self.bm25.get_scores(tokens)
        return np.array(scores, dtype=np.float32)

    def get_multi_query_scores(self, queries: List[str]) -> np.ndarray:
        """
        Combina scores BM25 de múltiples queries.
        Estrategia: Max(70%) + Mean(30%) — igual que el pipeline de embeddings.

        Returns: np.ndarray shape (n_docs,)
        """
        if not queries:
            return np.zeros(self.n_docs)

        all_scores = np.array([self.get_scores(q) for q in queries])  # (n_queries, n_docs)
        max_scores  = np.max(all_scores, axis=0)
        mean_scores = np.mean(all_scores, axis=0)
        combined = (max_scores * 0.7) + (mean_scores * 0.3)
        return combined


# ============================================================
# RECIPROCAL RANK FUSION (RRF)
# ============================================================

def reciprocal_rank_fusion(
    embedding_scores: np.ndarray,
    bm25_scores: np.ndarray,
    weight_embedding: float = 0.6,
    weight_bm25: float = 0.4,
    k: int = RRF_K,
) -> np.ndarray:
    """
    Fusiona rankings de embeddings y BM25 usando Reciprocal Rank Fusion.

    Fórmula RRF: score_rrf(d) = Σ weight_i / (k + rank_i(d))

    Ventaja sobre suma ponderada de scores:
    - Agnóstico a la escala de cada sistema (BM25 y coseno tienen escalas distintas)
    - Penaliza fuertemente artículos que solo aparecen en UN ranking
    - Favorece artículos que aparecen en AMBOS rankings

    Args:
        embedding_scores: Scores de similitud coseno normalizados (n_docs,)
        bm25_scores:      Scores BM25 crudos (n_docs,)
        weight_embedding: Peso del ranking de embeddings en RRF
        weight_bm25:      Peso del ranking BM25 en RRF
        k:                Constante de suavizado RRF (default: 60)

    Returns:
        np.ndarray: Scores RRF fusionados, normalizados a [0, 1]
    """
    n = len(embedding_scores)
    if n == 0:
        return np.array([])

    # Calcular rankings (1-indexed, mayor score → menor rank)
    emb_ranks  = np.argsort(np.argsort(-embedding_scores)) + 1  # Rank 1 = mejor
    bm25_ranks = np.argsort(np.argsort(-bm25_scores)) + 1

    # RRF ponderado
    rrf_scores = (
        weight_embedding / (k + emb_ranks.astype(float)) +
        weight_bm25      / (k + bm25_ranks.astype(float))
    )

    # Normalizar a [0, 1]
    rrf_min, rrf_max = rrf_scores.min(), rrf_scores.max()
    if rrf_max > rrf_min:
        rrf_scores = (rrf_scores - rrf_min) / (rrf_max - rrf_min)
    else:
        rrf_scores = np.full(n, 0.5)

    logger.debug(
        f"🔀 RRF: min={rrf_scores.min():.3f}, max={rrf_scores.max():.3f}, "
        f"mean={rrf_scores.mean():.3f}"
    )
    return rrf_scores.astype(np.float32)


# ============================================================
# FUNCIÓN DE ALTO NIVEL (usada por screening.py)
# ============================================================

def compute_hybrid_scores(
    texts: List[str],
    embedding_scores: np.ndarray,
    semantic_queries: List[str],
    weight_embedding: float = 0.6,
    weight_bm25: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pipeline completo BM25 + RRF para integración en screen_articles().

    Args:
        texts:            Lista de "título. abstract" de cada artículo
        embedding_scores: Scores de embeddings ya calculados (n_docs,)
        semantic_queries: Queries semánticas expandidas por LLM
        weight_embedding: Peso para embeddings en RRF
        weight_bm25:      Peso para BM25 en RRF

    Returns:
        Tuple:
          - rrf_scores (np.ndarray): Score híbrido final (n_docs,)
          - bm25_raw   (np.ndarray): Score BM25 crudo para logging (n_docs,)
    """
    try:
        retriever = BM25Retriever(texts)
        bm25_raw  = retriever.get_multi_query_scores(semantic_queries)

        rrf_scores = reciprocal_rank_fusion(
            embedding_scores=embedding_scores,
            bm25_scores=bm25_raw,
            weight_embedding=weight_embedding,
            weight_bm25=weight_bm25,
        )

        n_bm25_nonzero = np.sum(bm25_raw > 0)
        logger.info(
            f"🔀 Hybrid BM25+RRF completado | "
            f"Docs con match léxico: {n_bm25_nonzero}/{len(texts)} | "
            f"RRF mean: {rrf_scores.mean():.3f}"
        )
        return rrf_scores, bm25_raw

    except ImportError as e:
        logger.warning(f"⚠️ BM25 no disponible ({e}). Usando solo embeddings.")
        return embedding_scores, np.zeros(len(texts))
    except Exception as e:
        logger.error(f"❌ Error en BM25 híbrido: {e}. Fallback a embeddings puros.")
        return embedding_scores, np.zeros(len(texts))
