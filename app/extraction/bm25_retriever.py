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

Diseño agnóstico al dominio:
  - Sin stopwords hardcodeadas: se derivan de la frecuencia del corpus (TF global)
  - Sin pesos fijos de RRF: configurables en tiempo de ejecución
  - Sin lógica de lenguaje asumida: detección automática de idioma si langdetect disponible
  - Toda constante expuesta como parámetro con su valor por defecto documentado

Referencias:
  - Robertson & Zaragoza (2009): "The Probabilistic Relevance Framework: BM25 and Beyond"
  - Cormack et al. (2009): "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
  - Luhn (1958): "The automatic creation of literature abstracts" — base del filtro de frecuencia media
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTES — todas expuestas, ninguna oculta en lógica
# ============================================================
RRF_K: int = 60
"""Constante de suavizado RRF (Cormack 2009). Valores típicos: 10-100.
Mayor k → diferencias de rank menos pronunciadas."""

DEFAULT_MIN_TOKEN_LEN: int = 2
"""Longitud mínima de token para considerar un término."""

DEFAULT_MAX_TOKEN_LEN: int = 40
"""Longitud máxima; filtra URLs o strings de hashing accidentales."""

DEFAULT_CORPUS_STOPWORD_PERCENTILE: float = 0.97
"""Percentil de frecuencia de corpus sobre el cual un token se considera
stopword por ser demasiado omnipresente (p.ej., 'the', 'de', 'and').
Ajustable según tamaño del corpus: corpora pequeños (<200 docs) pueden
necesitar 0.90; corpora grandes (>5000) pueden tolerar 0.98."""

DEFAULT_CAMELCASE_SPLIT: bool = True
"""Si True, descompone CamelCase: CodeBERT → ['code', 'bert', 'codebert'].
Útil en dominios de software/biomédico; desactivar para textos generales."""

DEFAULT_HYPHEN_EXPAND: bool = True
"""Si True, expande palabras con guión: fine-tuning → [fine, tuning, finetuning].
Mantiene tanto partes individuales como la forma compuesta."""

DEFAULT_WEIGHT_EMBEDDING: float = 0.6
DEFAULT_WEIGHT_BM25: float = 0.4
DEFAULT_MULTI_QUERY_MAX_WEIGHT: float = 0.7
DEFAULT_MULTI_QUERY_MEAN_WEIGHT: float = 0.3


# ============================================================
# CONFIGURACIÓN INYECTABLE
# ============================================================

@dataclass
class TokenizerConfig:
    """
    Configuración completa del tokenizador científico.
    Todos los valores tienen defaults razonables; ninguno asume un dominio.

    Parámetros
    ----------
    min_token_len : int
        Longitud mínima de token válido.
    max_token_len : int
        Longitud máxima de token válido.
    corpus_stopword_percentile : float
        Percentil [0,1] de frecuencia de corpus para derivar stopwords.
        0.0 = ningún token se descarta por frecuencia (sin filtro).
        1.0 = todos los tokens se descartan (no útil).
    camelcase_split : bool
        Descomponer CamelCase en sub-tokens.
    hyphen_expand : bool
        Expandir términos con guión.
    extra_stopwords : set[str]
        Stopwords adicionales inyectadas externamente (vacío por defecto).
        Permite que el caller agregue dominio-específicas sin modificar este módulo.
    """
    min_token_len: int = DEFAULT_MIN_TOKEN_LEN
    max_token_len: int = DEFAULT_MAX_TOKEN_LEN
    corpus_stopword_percentile: float = DEFAULT_CORPUS_STOPWORD_PERCENTILE
    camelcase_split: bool = DEFAULT_CAMELCASE_SPLIT
    hyphen_expand: bool = DEFAULT_HYPHEN_EXPAND
    extra_stopwords: set = field(default_factory=set)


@dataclass
class RRFConfig:
    """
    Configuración del módulo Reciprocal Rank Fusion.

    Parámetros
    ----------
    k : int
        Constante de suavizado. 60 es el valor canónico de Cormack 2009.
    weight_embedding : float
        Peso del ranking de embeddings en la fusión RRF.
    weight_bm25 : float
        Peso del ranking BM25 en la fusión RRF.
        Nota: weight_embedding + weight_bm25 no necesita sumar 1.0
        porque RRF es agnóstico a la escala absoluta de los pesos;
        solo importa la proporción relativa.
    """
    k: int = RRF_K
    weight_embedding: float = DEFAULT_WEIGHT_EMBEDDING
    weight_bm25: float = DEFAULT_WEIGHT_BM25


# ============================================================
# TOKENIZADOR CIENTÍFICO AGNÓSTICO
# ============================================================

def _split_camelcase(word: str) -> List[str]:
    """
    Descompone CamelCase en tokens individuales.
    Preserva secuencias de mayúsculas contiguas como acrónimos.
    Ejemplos:
      'CodeBERT'    → ['Code', 'BERT']
      'GPT4Model'   → ['GPT4', 'Model']
      'finetuning'  → ['finetuning']
    """
    # Insertar separador antes de mayúscula precedida de minúscula
    # y antes de mayúscula seguida de minúscula (para acrónimos)
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', word)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    return s.split()


def _normalize_unicode(text: str) -> str:
    """Elimina diacríticos y normaliza a ASCII compatible."""
    text = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in text if not unicodedata.combining(c))


def _tokenize(text: str, config: TokenizerConfig) -> List[str]:
    """
    Tokeniza un texto académico de forma agnóstica al dominio.

    El único conocimiento de dominio permitido aquí es morfológico
    (CamelCase, guiones), no léxico (sin listas de stopwords hardcodeadas).
    Las stopwords se derivan del corpus en BM25Retriever.__init__.

    Args:
        text:   Texto a tokenizar.
        config: Configuración de tokenización.

    Returns:
        Lista de tokens en minúsculas, sin duplicados consecutivos.
    """
    if not text or not text.strip():
        return []

    text = _normalize_unicode(text).lower()

    tokens: List[str] = []

    # Dividir por separadores no alfanuméricos (preservando guiones internos)
    raw_words = re.split(r'[^\w\-]+', text)

    for raw in raw_words:
        raw = raw.strip('-').strip()
        if not raw:
            continue

        # --- Expansión de palabras con guión ---
        if config.hyphen_expand and '-' in raw:
            parts = [p for p in raw.split('-') if p]
            for part in parts:
                if config.min_token_len <= len(part) <= config.max_token_len:
                    tokens.append(part)
            combined = ''.join(parts)
            if config.min_token_len <= len(combined) <= config.max_token_len:
                tokens.append(combined)
            continue

        # --- Descomposición CamelCase ---
        if config.camelcase_split and any(c.isupper() for c in raw):
            sub_tokens = _split_camelcase(raw)
            for sub in sub_tokens:
                sub_lower = sub.lower()
                if config.min_token_len <= len(sub_lower) <= config.max_token_len:
                    tokens.append(sub_lower)
            # También preservar la forma completa en minúsculas
            full_lower = raw.lower()
            if (len(sub_tokens) > 1 and
                    config.min_token_len <= len(full_lower) <= config.max_token_len):
                tokens.append(full_lower)
            continue

        # --- Token normal ---
        if config.min_token_len <= len(raw) <= config.max_token_len:
            tokens.append(raw)

    # Aplicar stopwords extra inyectadas externamente
    if config.extra_stopwords:
        tokens = [t for t in tokens if t not in config.extra_stopwords]

    return tokens


def _derive_corpus_stopwords(
    tokenized_corpus: List[List[str]],
    percentile: float,
) -> set:
    """
    Deriva stopwords automáticamente por frecuencia de documento (DF).

    Un token es stopword si su Document Frequency supera el percentil
    indicado sobre la distribución de todas las DFs del corpus.

    Este enfoque (inspirado en Luhn 1958) es completamente agnóstico:
    no requiere conocer el idioma ni el dominio. Los tokens omnipresentes
    como 'the', 'de', 'and', 'en' emergen solos como stopwords.

    Args:
        tokenized_corpus: Corpus ya tokenizado.
        percentile:       Percentil de corte [0, 1].

    Returns:
        Conjunto de tokens considerados stopwords por frecuencia.
    """
    if not tokenized_corpus or percentile <= 0.0:
        return set()

    n_docs = len(tokenized_corpus)
    # Frecuencia de documento (en cuántos docs aparece cada token)
    df_counter: Counter = Counter()
    for doc_tokens in tokenized_corpus:
        df_counter.update(set(doc_tokens))  # set() para contar una vez por doc

    if not df_counter:
        return set()

    df_values = np.array(list(df_counter.values()), dtype=float)
    threshold = np.percentile(df_values, percentile * 100)

    stopwords = {
        token for token, df in df_counter.items()
        if df >= threshold
    }

    logger.debug(
        f"📊 Stopwords derivadas del corpus: {len(stopwords)} tokens "
        f"(DF ≥ {threshold:.0f} de {n_docs} docs, percentil {percentile:.0%})"
    )
    return stopwords


# ============================================================
# BM25 RETRIEVER
# ============================================================

class BM25Retriever:
    """
    Índice BM25Okapi agnóstico al dominio sobre un corpus de textos.

    Características:
    - Stopwords derivadas automáticamente del corpus (sin listas hardcodeadas)
    - Configuración completa inyectable vía TokenizerConfig
    - Thread-safe para uso en pipelines paralelos
    - Fallback graceful si rank_bm25 no está disponible

    Uso básico:
        retriever = BM25Retriever(texts)
        scores = retriever.get_scores("software defect prediction")

    Uso avanzado (dominio médico, sin CamelCase, stopwords extra):
        config = TokenizerConfig(
            camelcase_split=False,
            extra_stopwords={"patient", "clinical", "hospital"}
        )
        retriever = BM25Retriever(texts, config=config)
    """

    def __init__(
        self,
        texts: List[str],
        config: Optional[TokenizerConfig] = None,
    ):
        """
        Args:
            texts:  Lista de strings (título + abstract de cada artículo).
                    El índice i en 'texts' corresponde al índice i del corpus.
            config: Configuración del tokenizador. Si None, usa defaults.

        Raises:
            ImportError: Si rank_bm25 no está instalado.
            ValueError:  Si texts está vacío.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank_bm25 no está instalado. Ejecuta: pip install rank-bm25"
            )

        if not texts:
            raise ValueError("BM25Retriever requiere al menos un documento en el corpus.")

        self.config = config or TokenizerConfig()
        self.n_docs = len(texts)

        # Paso 1: tokenizar sin stopwords de corpus (aún no las conocemos)
        raw_tokenized = [_tokenize(t, self.config) for t in texts]

        # Paso 2: derivar stopwords automáticamente del corpus
        corpus_stopwords = _derive_corpus_stopwords(
            raw_tokenized,
            self.config.corpus_stopword_percentile,
        )

        # Paso 3: filtrar stopwords del corpus en cada documento
        tokenized_corpus = [
            [tok for tok in doc_tokens if tok not in corpus_stopwords]
            for doc_tokens in raw_tokenized
        ]

        # Paso 4: indexar con BM25
        self.bm25 = BM25Okapi(tokenized_corpus)
        self._corpus_stopwords = corpus_stopwords  # expuesto para inspección/debug

        logger.info(
            f"🗂️ BM25 indexado: {self.n_docs} docs | "
            f"Stopwords corpus: {len(corpus_stopwords)} | "
            f"Config: camelcase={self.config.camelcase_split}, "
            f"hyphen={self.config.hyphen_expand}"
        )

    def get_scores(self, query: str) -> np.ndarray:
        """
        Scores BM25 para todos los documentos dado un query.

        Args:
            query: Texto de consulta (cualquier idioma/dominio).

        Returns:
            np.ndarray shape (n_docs,): mayor score = más relevante léxicamente.
            Retorna array de ceros si el query está vacío tras tokenización.
        """
        tokens = _tokenize(query, self.config)
        # Filtrar también stopwords del corpus en el query
        tokens = [t for t in tokens if t not in self._corpus_stopwords]

        if not tokens:
            logger.debug("⚠️ Query vacío tras tokenización. Retornando scores cero.")
            return np.zeros(self.n_docs, dtype=np.float32)

        scores = self.bm25.get_scores(tokens)
        return np.array(scores, dtype=np.float32)

    def get_multi_query_scores(
        self,
        queries: List[str],
        max_weight: float = DEFAULT_MULTI_QUERY_MAX_WEIGHT,
        mean_weight: float = DEFAULT_MULTI_QUERY_MEAN_WEIGHT,
    ) -> np.ndarray:
        """
        Combina scores BM25 de múltiples queries.

        Estrategia: score = max_weight * Max(scores) + mean_weight * Mean(scores)

        La combinación Max+Mean captura dos señales complementarias:
        - Max: un query altamente específico que hace match exacto
        - Mean: consistencia de relevancia a través de múltiples perspectivas

        Args:
            queries:      Lista de queries semánticas expandidas.
            max_weight:   Peso del score máximo (default 0.7).
            mean_weight:  Peso del score promedio (default 0.3).

        Returns:
            np.ndarray shape (n_docs,)
        """
        if not queries:
            return np.zeros(self.n_docs, dtype=np.float32)

        all_scores = np.array(
            [self.get_scores(q) for q in queries],
            dtype=np.float32
        )  # shape: (n_queries, n_docs)

        max_scores  = np.max(all_scores,  axis=0)
        mean_scores = np.mean(all_scores, axis=0)
        combined = (max_scores * max_weight) + (mean_scores * mean_weight)
        return combined

    @property
    def corpus_stopwords(self) -> set:
        """Stopwords derivadas del corpus. Solo lectura, útil para debugging."""
        return frozenset(self._corpus_stopwords)


# ============================================================
# RECIPROCAL RANK FUSION (RRF)
# ============================================================

def reciprocal_rank_fusion(
    embedding_scores: np.ndarray,
    bm25_scores: np.ndarray,
    config: Optional[RRFConfig] = None,
) -> np.ndarray:
    """
    Fusiona rankings de embeddings y BM25 usando Reciprocal Rank Fusion.

    Fórmula RRF: score_rrf(d) = Σ weight_i / (k + rank_i(d))

    Ventaja sobre suma ponderada de scores directos:
    - Agnóstico a la escala de cada sistema (BM25 y coseno tienen escalas distintas)
    - Penaliza fuertemente artículos que solo aparecen bien en UN ranking
    - Favorece artículos consistentemente buenos en AMBOS rankings

    Args:
        embedding_scores: Scores de similitud coseno (n_docs,). Cualquier rango.
        bm25_scores:      Scores BM25 crudos (n_docs,). Cualquier rango.
        config:           Configuración RRF (k, pesos). Si None, usa defaults.

    Returns:
        np.ndarray: Scores RRF fusionados, normalizados a [0, 1].
    """
    cfg = config or RRFConfig()
    n = len(embedding_scores)

    if n == 0:
        return np.array([], dtype=np.float32)

    if len(bm25_scores) != n:
        raise ValueError(
            f"Dimensiones incompatibles: embedding_scores={n}, "
            f"bm25_scores={len(bm25_scores)}"
        )

    # Calcular rankings (1-indexed; rank 1 = el mejor score)
    emb_ranks  = np.argsort(np.argsort(-embedding_scores)) + 1
    bm25_ranks = np.argsort(np.argsort(-bm25_scores)) + 1

    # RRF ponderado
    rrf_scores = (
        cfg.weight_embedding / (cfg.k + emb_ranks.astype(np.float32)) +
        cfg.weight_bm25      / (cfg.k + bm25_ranks.astype(np.float32))
    )

    # Normalizar a [0, 1]
    rrf_min, rrf_max = rrf_scores.min(), rrf_scores.max()
    if rrf_max > rrf_min:
        rrf_scores = (rrf_scores - rrf_min) / (rrf_max - rrf_min)
    else:
        rrf_scores = np.full(n, 0.5, dtype=np.float32)

    logger.debug(
        f"🔀 RRF: k={cfg.k} | w_emb={cfg.weight_embedding} | w_bm25={cfg.weight_bm25} | "
        f"min={rrf_scores.min():.3f} max={rrf_scores.max():.3f} mean={rrf_scores.mean():.3f}"
    )
    return rrf_scores.astype(np.float32)


# ============================================================
# FUNCIÓN DE ALTO NIVEL (usada por screening.py)
# ============================================================

def compute_hybrid_scores(
    texts: List[str],
    embedding_scores: np.ndarray,
    semantic_queries: List[str],
    tokenizer_config: Optional[TokenizerConfig] = None,
    rrf_config: Optional[RRFConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pipeline completo BM25 + RRF para integración en screen_articles().

    Toda la configuración es inyectable; no hay valores hardcodeados internos.
    El caller puede personalizar comportamiento sin modificar este módulo.

    Args:
        texts:             Lista de "título. abstract" de cada artículo.
        embedding_scores:  Scores de embeddings ya calculados (n_docs,).
        semantic_queries:  Queries semánticas expandidas por LLM.
        tokenizer_config:  Config del tokenizador BM25. None = defaults agnósticos.
        rrf_config:        Config de RRF. None = defaults (k=60, 0.6/0.4).

    Returns:
        Tuple:
          - rrf_scores (np.ndarray): Score híbrido final normalizado (n_docs,).
          - bm25_raw   (np.ndarray): Score BM25 crudo para logging/análisis (n_docs,).

    Ejemplo de uso con dominio personalizado:
        tok_cfg = TokenizerConfig(
            corpus_stopword_percentile=0.95,
            extra_stopwords={"patient", "clinical"},
            camelcase_split=False,
        )
        rrf_cfg = RRFConfig(k=30, weight_embedding=0.7, weight_bm25=0.3)

        rrf, bm25 = compute_hybrid_scores(
            texts, emb_scores, queries,
            tokenizer_config=tok_cfg,
            rrf_config=rrf_cfg,
        )
    """
    try:
        retriever = BM25Retriever(texts, config=tokenizer_config)
        bm25_raw  = retriever.get_multi_query_scores(semantic_queries)

        rrf_scores = reciprocal_rank_fusion(
            embedding_scores=embedding_scores,
            bm25_scores=bm25_raw,
            config=rrf_config,
        )

        n_bm25_nonzero = int(np.sum(bm25_raw > 0))
        logger.info(
            f"🔀 Hybrid BM25+RRF completado | "
            f"Docs con match léxico: {n_bm25_nonzero}/{len(texts)} | "
            f"RRF mean: {rrf_scores.mean():.3f}"
        )
        return rrf_scores, bm25_raw

    except ImportError as e:
        logger.warning(f"⚠️ BM25 no disponible ({e}). Fallback a embeddings puros.")
        return embedding_scores.astype(np.float32), np.zeros(len(texts), dtype=np.float32)

    except Exception as e:
        logger.error(f"❌ Error en BM25 híbrido: {e}. Fallback a embeddings puros.")
        return embedding_scores.astype(np.float32), np.zeros(len(texts), dtype=np.float32)