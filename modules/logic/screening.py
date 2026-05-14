import logging
import re
import time
import json
import numpy as np
from typing import List, Dict, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity

import requests
import config
from modules.ai.embedding_service import get_embeddings, get_single_embedding
from utils.query_expander import expand_query_with_llm, extract_english_terms, get_exclusion_terms_with_llm, expand_query_with_synonyms
from modules.core.search_engine import detect_search_domain
from modules.logic.metadata_filter import apply_hard_filters, parse_exclusion_criteria_for_hard_filter
from utils.bm25_retriever import compute_hybrid_scores
from modules.ai.cross_encoder_reranker import rerank_with_cross_encoder

# ============================================================
# CONFIGURACION DEL MODELO (UNIFICADO VIA embedding_service)
# ============================================================
MODEL_NAME = config.EMBEDDING_MODEL

def get_embedding(text: str) -> np.ndarray:
    """Atajo para obtener embedding de un texto único."""
    return get_single_embedding(text)


# ============================================================
# EXTRACCION DE KEYWORDS DE DOMINIO
# ============================================================

# Palabras metodologicas que NO distinguen dominio (Genéricas)
METHODOLOGY_TERMS = {
    'machine learning', 'deep learning', 'artificial intelligence', 'ai',
    'neural network', 'model', 'algorithm', 'classification', 'prediction',
    'analysis', 'method', 'approach', 'technique', 'system',
    'framework', 'evaluation', 'comparison', 'review', 'survey', 'study',
    'transformer', 'attention', 'embedding', 'training', 'dataset',
    'accuracy', 'precision', 'recall', 'f1', 'performance', 'optimization',
    'hyperparameter', 'tuning', 'preprocessing', 'feature', 'extraction',
    'regression', 'random forest', 'xgboost', 'lstm', 'cnn', 'rnn', 'gru',
    'bayesian', 'probabilistic', 'stochastic', 'ensemble', 'boosting',
    'cross-validation', 'overfitting', 'generalization', 'benchmark',
    'error', 'reduction', 'improvement',
    'automated', 'automatic', 'intelligent', 'smart', 'adaptive',
    'nlp', 'natural language processing', 'text mining', 'sentiment',
    'supervised', 'unsupervised', 'reinforcement', 'transfer learning',
    'fine-tuning', 'pre-trained',
}

# Stopwords en espanol e ingles
STOPWORDS = {
    'cual', 'como', 'que', 'es', 'son', 'las', 'los',
    'del', 'de', 'la', 'el', 'en', 'un', 'una', 'para', 'por', 'con', 'sin',
    'sobre', 'entre', 'mas', 'se', 'al', 'a', 'e', 'i', 'o', 'u', 'y',
    'su', 'sus', 'mi', 'tu', 'frente', 'durante', 'mediante',
    'the', 'an', 'and', 'or', 'not', 'in', 'on', 'of', 'to', 'for',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'this', 'that', 'these', 'those', 'with', 'from', 'by', 'at', 'as',
    'their', 'its', 'what', 'which', 'how', 'when', 'where', 'who', 'whom',
    'medida', 'eficacia', 'reducen', 'margen', 'base',
    'herramientas', 'tradicionales', 'modelos', 'grandes', 'lenguaje',
}

# Traducciones espanol-ingles para terminos de dominio comunes
DOMAIN_TRANSLATIONS = {
    'codigo fuente': 'source code',
    'codigo': 'code',
    'vulnerabilidades': 'vulnerability',
    'vulnerabilidad': 'vulnerability',
    'analisis estatico': 'static analysis',
    'falsos positivos': 'false positive',
    'deteccion de vulnerabilidades': 'vulnerability detection',
    'seguridad de software': 'software security',
    'revision de codigo': 'code review',
    'eventos deportivos': 'sports events',
    'deportivos': 'sports',
    'deportes': 'sports',
    'predicciones deportivas': 'sports prediction',
    'variables temporales': 'time series temporal',
    'enfermedades cardiovasculares': 'cardiovascular disease',
    'salud mental': 'mental health',
    'educacion superior': 'higher education',
    'energia renovable': 'renewable energy',
    'internet de las cosas': 'internet of things iot',
    'cadena de suministro': 'supply chain',
    'redes neuronales': 'neural network',
    'inteligencia artificial': 'artificial intelligence',
    'aprendizaje profundo': 'deep learning',
    'aprendizaje automatico': 'machine learning',
}


# Eliminadas listas hardcoded (v6.1 - Dinamismo Puro)
# Ahora los términos de exclusión se generan vía LLM específicamente para cada pregunta de investigación.

def _extract_comparison_poles(question: str) -> List[List[str]]:
    """Extrae dinámicamente los dos polos de una comparación (A vs B) (v7.2)."""
    if not question: return []
    
    prompt = f"""Research Question: "{question}"

What are the TWO main concepts being compared? List synonyms (technical terms in English) for each.
Force ALL terms to be in ENGLISH even if the question is in another language.

Respond with ONLY this JSON format:
{{"poles": [["concept1", "synonym1a", "synonym1b"], ["concept2", "synonym2a", "synonym2b"]]}}"""
    
    try:
        # Usamos Groq directamente para forzar JSON mode y temperatura 0
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {config.GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "You are a research engineer. Extract comparison poles as JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 150,
                "response_format": {"type": "json_object"}
            },
            timeout=10
        )
        
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            data = json.loads(content)
            poles = data.get("poles", [])
            if poles and len(poles) >= 2:
                logging.info(f"⚖️ Polos de comparación extraídos: {poles[0][0]} vs {poles[1][0]}")
                return poles
    except Exception as e:
        logging.warning(f"⚠️ No se pudieron extraer polos: {e}")
    
    return []


def extract_domain_keywords(question: str) -> Set[str]:
    """
    Extrae keywords DE DOMINIO de la pregunta de investigacion.
    Excluye palabras metodologicas (ML, AI, etc.) compartidas entre dominios.
    """
    import unicodedata
    
    def strip_accents(text):
        """Quita acentos: código -> codigo, análisis -> analisis"""
        nfkd = unicodedata.normalize('NFKD', text)
        return ''.join(c for c in nfkd if not unicodedata.combining(c))
    
    domain_keywords = set()
    q_lower = strip_accents(question.lower())

    # 1. Traducir frases compuestas espanol-ingles
    for esp, eng in DOMAIN_TRANSLATIONS.items():
        if esp in q_lower:
            for word in eng.split():
                if word.lower() not in METHODOLOGY_TERMS and len(word) > 2:
                    domain_keywords.add(word.lower())

    # 2. Extraer acronimos (SAST, LLM, IoT, CNN, etc.)
    acronyms = re.findall(r'\b([A-Z]{2,6}s?)\b', question)
    skip_acronyms = {'AND', 'OR', 'NOT', 'THE', 'FOR', 'DE', 'LA', 'LOS', 'EN', 'DEL', 'UNA'}
    for a in acronyms:
        clean = a.rstrip('s').lower()
        if clean.upper() not in skip_acronyms and clean not in METHODOLOGY_TERMS:
            domain_keywords.add(clean)

    # 3. Extraer palabras en ingles directas (de texto mixto espanol/ingles)
    english_words = re.findall(r'\b([a-z]{3,})\b', q_lower)
    for w in english_words:
        if w not in STOPWORDS and w not in METHODOLOGY_TERMS and len(w) > 2:
            spanish_common = {'como', 'para', 'pero', 'puede', 'porque', 'tiene',
                            'cada', 'otro', 'otra', 'entre', 'desde',
                            'hasta', 'sin', 'mejor', 'peor', 'mayor',
                            'menor', 'parte', 'forma', 'modo', 'tipo', 'caso',
                            'manera', 'vez', 'tiempo', 'punto', 'lado',
                            'nivel', 'dentro', 'fuera', 'antes'}
            if w not in spanish_common:
                domain_keywords.add(w)

    # 4. Extraer texto entre comillas o parentesis
    quoted = re.findall(r'["\u00ab]([^"\u00bb]+)["\u00bb]', question)
    for q in quoted:
        for word in q.lower().split():
            if word not in STOPWORDS and word not in METHODOLOGY_TERMS and len(word) > 2:
                domain_keywords.add(word)

    # 5. Limpiar
    domain_keywords.discard('')

    logging.info(f"🏷️ Keywords de dominio extraidos: {sorted(domain_keywords)}")
    return domain_keywords


def compute_domain_relevance(article: Dict, domain_keywords: Set[str], 
                              semantic_queries: List[str] = None) -> float:
    """
    Calcula relevancia de dominio usando:
    1. Frases completas de las queries LLM (n-gramas 2-4 palabras).
    2. Keywords individuales como fallback.
    """
    title = (article.get('title', '') or '').lower()
    abstract = (article.get('abstract', '') or '').lower()
    text = f"{title}. {abstract}"

    if not text.strip():
        return 0.0

    # PRIORIDAD 1: N-gramas de las queries semánticas
    if semantic_queries:
        phrase_matches = set()
        for query in semantic_queries:
            words = query.lower().split()
            # Extraer bigramas a tetragramas
            for n in range(2, min(5, len(words) + 1)):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i:i+n])
                    if phrase in text:
                        phrase_matches.add(phrase)
        
        count = len(phrase_matches)
        if count >= 3: return 1.0
        elif count == 2: return 0.75
        elif count == 1: return 0.50

    # FALLBACK: Keywords individuales
    matches = sum(1 for kw in domain_keywords if kw in text)
    if matches == 0: return 0.0
    elif matches == 1: return 0.4
    elif matches == 2: return 0.7
    else: return 1.0


def normalize_scores_robust(scores: np.ndarray) -> np.ndarray:
    """
    Normaliza scores usando percentiles (5-95) para ignorar outliers.
    Evita la distorsión causada por un solo artículo líder atípico.
    """
    if len(scores) == 0:
        return scores
    
    p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
    
    if p95 == p5:
        return np.full_like(scores, 0.5)
        
    # Clipping y Min-Max
    clipped = np.clip(scores, p5, p95)
    normalized = (clipped - p5) / (p95 - p5)
    return normalized

def compute_keyword_boost(article: Dict, english_terms: List[str], 
                          semantic_queries: List[str] = None,
                          comparison_poles: List[List[str]] = None) -> float:
    """
    Boost/Malus basado en qué tan bien el paper coincide con el ÁNGULO
    específico de la RQ, no solo con los términos sueltos. (v7.0)
    """
    if not english_terms:
        return 0.0
        
    title = str(article.get('title', '')).lower()
    abstract = str(article.get('abstract', '')).lower()
    content = f"{title}. {abstract}"
    
    # --- BOOST: términos de la RQ presentes ---
    matches = sum(1 for t in english_terms if t.lower() in content)
    coverage = matches / len(english_terms)
    critical_terms = english_terms[:3]
    critical_matches = sum(1 for t in critical_terms if t.lower() in content)
    critical_coverage = critical_matches / len(critical_terms) if critical_terms else 0
    base_boost = (coverage * 0.4) + (critical_coverage * 0.6)

    # --- MALUS CONTEXTUAL (Agnóstico): Contraste por trigramas ---
    if semantic_queries:
        trigram_hits = 0
        for q in semantic_queries:
            words = q.lower().split()
            for i in range(len(words) - 2):
                trigram = " ".join(words[i:i+3])
                if trigram in content:
                    trigram_hits += 1
        
        if matches >= 2 and trigram_hits == 0:
            base_boost *= 0.5

    # --- MALUS POR COMPARACIÓN AUSENTE (v7.0 Agnóstico) ---
    if comparison_poles and len(comparison_poles) >= 2:
        pole_a_present = any(term.lower() in content for term in comparison_poles[0])
        pole_b_present = any(term.lower() in content for term in comparison_poles[1])
        
        # Si falta CUALQUIERA de los dos polos, penalizar (Estudio no comparativo)
        if not pole_a_present or not pole_b_present:
            base_boost *= 0.6 # Malus del 40%
            
    return base_boost

def get_adaptive_threshold(scores: List[float], target_n: int = 50, min_threshold: float = 0.40) -> float:
    """
    Calcula un umbral adaptativo para obtener ~target_n artículos.
    Nunca baja del min_threshold (suelo absoluto de calidad).
    """
    if not scores:
        return min_threshold
    
    sorted_scores = sorted(scores, reverse=True)
    adaptive = sorted_scores[min(target_n - 1, len(sorted_scores) - 1)]
    
    # El umbral adaptativo no puede ser menor al suelo absoluto
    final_thresh = max(adaptive, min_threshold)
    logging.info(f"🎯 Umbral Adaptativo: {final_thresh:.2f} (Target N={target_n}, Suelo={min_threshold})")
    return final_thresh


# ============================================================
# LOGICA FUZZY MAMDANI (v1.0) - Sin dependencias externas
# ============================================================

def _trap(x: float, a: float, b: float, c: float, d: float) -> float:
    """Función de pertenencia trapezoidal: sube de a->b, plano b->c, baja c->d."""
    if x <= a or x >= d:
        return 0.0
    elif x <= b:
        return (x - a) / (b - a) if b > a else 1.0
    elif x <= c:
        return 1.0
    else:
        return (d - x) / (d - c) if d > c else 1.0

def _tri(x: float, a: float, b: float, c: float) -> float:
    """Función de pertenencia triangular: sube a->b, baja b->c."""
    if x <= a or x >= c:
        return 0.0
    elif x <= b:
        return (x - a) / (b - a) if b > a else 1.0
    else:
        return (c - x) / (c - b) if c > b else 1.0


def compute_fuzzy_score(semantic_sim: float, domain_rel: float, kw_boost_raw: float) -> float:
    """
    Inferencia Fuzzy Mamdani sobre 3 variables de entrada.
    Reduce falsos positivos: artículos con alta similitud semántica
    pero baja relevancia de dominio reciben un score final más bajo.

    Variables de entrada:
      - semantic_sim  : similitud coseno normalizada [0, 1]
      - domain_rel    : relevancia de dominio [0, 1]
      - kw_boost_raw  : boost de keywords (antes de escalar, [0, ~0.4])

    Salida: score difuso de relevancia [0, 1]
    """
    # Normalizar kw_boost al rango [0,1] (máx estimado ~0.20 según pipeline)
    kw = min(kw_boost_raw / 0.20, 1.0)

    # ── FUNCIONES DE PERTENENCIA: Similitud Semántica ──
    sim_low    = _trap(semantic_sim,  0.0,  0.0,  0.25, 0.45)
    sim_medium = _tri( semantic_sim,  0.30, 0.55, 0.75)
    sim_high   = _trap(semantic_sim,  0.60, 0.80, 1.0,  1.0)

    # ── FUNCIONES DE PERTENENCIA: Relevancia de Dominio ──
    dom_low    = _trap(domain_rel,    0.0,  0.0,  0.25, 0.45)
    dom_high   = _trap(domain_rel,    0.35, 0.60, 1.0,  1.0)

    # ── FUNCIONES DE PERTENENCIA: Keyword Boost ──
    kw_low     = _trap(kw,            0.0,  0.0,  0.20, 0.40)
    kw_high    = _trap(kw,            0.30, 0.55, 1.0,  1.0)

    # ── REGLAS MAMDANI (activación = min de antecedentes) ──
    # Cada regla produce: (activación, valor_crisp_de_salida)
    rules = [
        # R1: sim_alta + dom_alta + kw_alta  → Muy Relevante
        (min(sim_high, dom_high, kw_high),   0.92),
        # R2: sim_alta + dom_alta             → Relevante
        (min(sim_high, dom_high),             0.78),
        # R3: sim_alta + dom_baja             → Medio (falso positivo vectorial)
        (min(sim_high, dom_low),              0.50),
        # R4: sim_media + dom_alta + kw_alta  → Relevante
        (min(sim_medium, dom_high, kw_high),  0.75),
        # R5: sim_media + dom_alta            → Medio-alto
        (min(sim_medium, dom_high),           0.58),
        # R6: sim_media + dom_baja            → Bajo
        (min(sim_medium, dom_low),            0.30),
        # R7: sim_baja                        → No relevante
        (sim_low,                             0.08),
    ]

    # ── DEFUZZIFICACIÓN: Centroide Ponderado ──
    numerator   = sum(act * val for act, val in rules)
    denominator = sum(act       for act, _   in rules)

    if denominator < 1e-9:
        return semantic_sim  # fallback: devolver similitud raw

    return numerator / denominator

def screen_articles(articles: List[Dict], query: str, threshold: float = 0.70,
                    max_results: int = 50, original_question: str = "",
                    inclusion_criteria: str = "", exclusion_criteria: str = "") -> List[Dict]:
    """
    Pipeline híbrido de screening para Revisiones Sistemáticas PRISMA.

    ARQUITECTURA DE 5 CAPAS:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Capa 0: Hard Filter por metadatos (pre-vectorial)
            → Excluye artículos sin abstract, fuera de rango temporal, etc.
    Capa 1: BM25 léxico + Embeddings semánticos → Fusión RRF
            → BM25 castiga artículos sin la terminología exacta de la RQ
            → RRF favorece artículos que aparecen bien en AMBOS rankings
    Capa 2: Expansión de consulta con sinónimos LLM (WordNet moderno)
            → Enriquece la consulta BM25 con sinónimos científicos del dominio
    Capa 3: Fuzzy Mamdani (similitud semántica × relevancia dominio × keywords)
    Capa 4: Cross-Encoder (re-ranking preciso sobre top-50 candidatos)
    Capa 5: LLM-as-a-Judge estricto (Cerebras) con justificación obligatoria
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    if not articles:
        return []

    # ============================================================
    # CAPA 0: HARD FILTER POR METADATOS (Pre-Vectorial)
    # ============================================================
    min_year, extra_excl_terms = parse_exclusion_criteria_for_hard_filter(exclusion_criteria)
    articles, hard_filter_report = apply_hard_filters(
        articles,
        min_year=min_year,
        min_abstract_length=80,   # Excluir artículos sin abstract útil
        extra_exclusion_terms=extra_excl_terms,
    )
    if not articles:
        logging.warning("⚠️ Hard Filter eliminó todos los artículos. Verifica los criterios de exclusión.")
        return []


    # 1. Extraer keywords de dominio de la pregunta original
    domain_keywords = extract_domain_keywords(original_question or query)

    # 2. Preparar textos (Titulo + Abstract) - Compatibilidad MPNet
    texts = [f"{art.get('title', '') or ''}. {art.get('abstract', '') or ''}" for art in articles]

    logging.info(f"🚀 Screening semantico de {len(texts)} articulos...")

    # Default por seguridad (evita NameError si hay excepción previa)
    RAW_SCORE_FLOOR = 0.40
    
    try:
        # 1. Obtener queries de screening en INGLES (Cierre de Gap Semántico)
        original_q = original_question or query

        # ── Enriquecimiento vectorial con criterios I/E del investigador ──
        enriched_query = original_q
        inc_lines: List[str] = []
        manual_exclusion_terms: List[str] = []

        if inclusion_criteria and inclusion_criteria.strip():
            inc_lines = [l.strip() for l in inclusion_criteria.strip().splitlines() if l.strip()]
            if inc_lines:
                enriched_query = original_q + ". " + " AND ".join(inc_lines)
                logging.info(f"✅ [Criterios I] {len(inc_lines)} criterios de inclusión detectados")
        if exclusion_criteria and exclusion_criteria.strip():
            exc_raw = [l.strip() for l in exclusion_criteria.strip().splitlines() if l.strip()]
            manual_exclusion_terms = [line.lower() for line in exc_raw if line]
            if manual_exclusion_terms:
                logging.info(f"🚫 [Criterios E] {len(manual_exclusion_terms)} criterios de exclusión detectados")


        # ── CAPA 2: Expansión con Sinónimos LLM (WordNet moderno) ──
        # Genera sinónimos científicos ANTES de calcular BM25 para enriquecer la búsqueda
        corpus_sample = [f"{a.get('title','')}. {a.get('abstract','')[:150]}" for a in articles[:10]]
        synonym_data = expand_query_with_synonyms(original_q, corpus_sample=corpus_sample)
        synonym_terms   = synonym_data.get('flat_terms', [])
        synonym_queries = synonym_data.get('expanded_queries', [])
        logging.info(f"🌐 [Synonyms] {len(synonym_terms)} términos expandidos para BM25")

        semantic_queries = expand_query_with_llm(enriched_query)
        # Enriquecer queries semánticas con las queries de sinónimos
        all_bm25_queries = list(set(semantic_queries + synonym_queries))

        english_terms = extract_english_terms(enriched_query)
        # Enriquecer english_terms con sinónimos para keyword boost
        english_terms = list(set(english_terms + synonym_terms))[:20]
        
        exclusion_terms = get_exclusion_terms_with_llm(original_q)
        # Combinar exclusiones: LLM + manuales del investigador
        exclusion_terms = list(set(exclusion_terms + manual_exclusion_terms))
        comparison_poles = _extract_comparison_poles(original_q)
        
        if comparison_poles:
            poles_str = " vs ".join([p[0] for p in comparison_poles])
            logging.info(f"⚖️ Polos de comparación detectados: {poles_str}")

        logging.info(f"🌎 Screening con {len(semantic_queries)} queries semánticas + {len(synonym_queries)} queries de sinónimos")

        # ── CAPA 1A: Scoring de Embeddings ──
        corpus_embeddings = get_embeddings(texts)  # shape: (n_articles, 384)
        
        all_query_scores = []
        for q in semantic_queries:
            q_emb = get_single_embedding(q)
            q_scores = cosine_similarity([q_emb], corpus_embeddings)[0]
            all_query_scores.append(q_scores)
            
        all_query_scores = np.array(all_query_scores)  # (n_queries, n_articles)
        
        max_scores = np.max(all_query_scores, axis=0)
        mean_scores = np.mean(all_query_scores, axis=0)
        embedding_raw_scores = (max_scores * 0.7) + (mean_scores * 0.3)

        # ── CAPA 1B: BM25 Léxico + RRF Fusion ──
        rrf_scores, bm25_raw_scores = compute_hybrid_scores(
            texts=texts,
            embedding_scores=embedding_raw_scores,
            semantic_queries=all_bm25_queries,
            weight_embedding=0.6,
            weight_bm25=0.4,
        )
        # El score final de la Capa 1 es el RRF fusionado
        final_raw_scores = rrf_scores

        # 🧠 DETECTAR DOMINIO para validación suave
        domain_results = detect_search_domain(original_question or query)
        detected_domain = domain_results['id']

        # 3. Normalización Robusta (Percentiles P5-P95)
        normalized_scores = normalize_scores_robust(final_raw_scores)
        
        max_raw = np.max(final_raw_scores)
        
        # --- UMBRAL MiNIMO DE CALIDAD (muy permisivo) ---
        # Solo excluye articulos absolutamente irrelevantes (raw<0.20).
        # El ranking real lo hace el hybrid_score (fuzzy+normalized).
        # Esto garantiza siempre llegar a max_results=100.
        corpus_size = len(articles)
        RAW_SCORE_FLOOR = 0.20   # Piso minimo de calidad, no filtro de ranking
            
        logging.info(f"📊 Líder Raw: {max_raw:.3f} | RAW FLOOR mínimo: {RAW_SCORE_FLOOR} (corpus={corpus_size})")
        
        # Capa 1: Gatekeeper Dinámico

        for i, art in enumerate(articles):
            raw_val = float(final_raw_scores[i])
            rel_val = float(normalized_scores[i])
            
            content_lower = f"{art.get('title', '')} {art.get('abstract', '')}".lower()

            # --- CAPA 1: GATEKEEPER DINÁMICO (Malus por términos de exclusión LLM) ---
            exclusion_malus = 0.0
            for term in exclusion_terms:
                if term.lower() in content_lower:
                    exclusion_malus -= 0.15

            # --- CAPA 2: KEYWORD BOOST CON COMPARACIÓN (v7.0) ---
            kw_boost = compute_keyword_boost(art, english_terms, semantic_queries, comparison_poles) * 0.20

            # --- CAPA 3: LOGICA FUZZY MAMDANI (v1.0) ---
            domain_rel_val = compute_domain_relevance(art, domain_keywords, semantic_queries)
            fuzzy_val = compute_fuzzy_score(raw_val, domain_rel_val, kw_boost)

            # Score híbrido: 60% fuzzy (multidimensional) + 40% similitud normalizada (fidelidad vectorial)
            # El exclusion_malus se aplica sobre el score combinado
            hybrid_score = (fuzzy_val * 0.6) + (rel_val * 0.4) + exclusion_malus

            # Asegurar rango [0, 1]
            final_score = min(max(hybrid_score, 0.0), 1.0)

            art['similarity']      = final_score
            art['raw_similarity']  = raw_val
            art['domain_relevance'] = domain_rel_val
            art['fuzzy_score']     = fuzzy_val

            if i < 5:
                logging.info(
                    f"🔷 [FUZZY] Art '{art.get('title', '')[:38]}...': "
                    f"raw={raw_val:.3f}, dom={domain_rel_val:.2f}, kw={kw_boost:.2f} "
                    f"→ fuzzy={fuzzy_val:.3f}, hybrid={final_score:.3f}"
                )

    except Exception as e:
        logging.error(f"❌ Error critico en screening: {e}")
        for i, art in enumerate(articles):
            art['similarity'] = 1.0 - (i / len(articles))
            art['raw_similarity'] = art['similarity']
            art['domain_relevance'] = 1.0

    # ============================================================
    # ESTRATEGIA DE SELECCION V6 - FUZZY-FIRST + CROSS-ENCODER
    # ============================================================

    # 1. Excluir los absolutamente irrelevantes (raw < 0.20)
    eligible = [a for a in articles if a.get('raw_similarity', 0) >= RAW_SCORE_FLOOR]
    
    # 2. Ordenar por hybrid_score (ranking fuzzy-first)
    eligible.sort(key=lambda x: x.get('similarity', 0), reverse=True)

    raw_above_floor = len(eligible)
    logging.info(f"🔍 Candidatos (raw≥{RAW_SCORE_FLOOR}): {raw_above_floor} | Pre-CE top-{max_results}")

    # 3. Priorizar artículos con URL válida dentro del top-N
    with_url    = [a for a in eligible if a.get('url') and len(str(a.get('url'))) > 10]
    without_url = [a for a in eligible if not (a.get('url') and len(str(a.get('url'))) > 10)]

    # Tomar top-200 para Cross-Encoder (necesita un pool mayor)
    ce_pool_size = min(200, len(eligible))
    final_selection = with_url[:ce_pool_size]
    if len(final_selection) < ce_pool_size:
        needed = ce_pool_size - len(final_selection)
        final_selection.extend(without_url[:needed])

    final_selection.sort(key=lambda x: x.get('similarity', 0), reverse=True)

    # ── CAPA 4: CROSS-ENCODER RE-RANKING (top-50 candidatos) ──
    # Procesa query + artículo JUNTOS en una red de atención cruzada.
    # Mucho más preciso que bi-encoders para detectar relevancia real.
    if original_question:
        logging.info(f"🎯 Cross-Encoder: re-rankeando top-50 de {len(final_selection)} candidatos...")
        final_selection = rerank_with_cross_encoder(
            candidates=final_selection,
            query=original_question,
            top_n=50,
            batch_size=16,
        )

    # Recortar al tamaño objetivo después del Cross-Encoder
    final_selection = final_selection[:max_results * 2]  # Pool 2x para el AI-Judge

    # ============================================================
    # AI ABSTRACT RE-RANKER (v1.0)
    # Evalua abstracts en zona ambigua con IA para eliminar falsos positivos
    # ============================================================
    if original_question and config.CEREBRAS_API_KEYS:
        final_selection = ai_multicriteria_score(
            candidates=final_selection,
            original_question=original_question,
            target_n=max_results,
            candidate_pool=min(len(final_selection), 200),
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
        )

    # ============================================================
    # REPORTE DE SELECCION V3
    # ============================================================
    if final_selection:
        count_with_url = sum(1 for art in final_selection if art.get('url'))
        count_without_url = len(final_selection) - count_with_url
        avg_similarity = np.mean([art['similarity'] for art in final_selection])
        avg_domain = np.mean([art.get('domain_relevance', 0) for art in final_selection])
        ai_evaluated = sum(1 for art in final_selection if art.get('ai_evaluated', False))

        logging.info(
            f"\n    ✅ SCREENING V4 (FUZZY+AI) COMPLETADO:\n"
            f"       📄 Total seleccionados: {len(final_selection)}\n"
            f"       🔗 Con URL/PDF: {count_with_url}\n"
            f"       ❌ Sin URL: {count_without_url}\n"
            f"       🤖 Evaluados por IA: {ai_evaluated}\n"
            f"       📊 Similitud promedio: {avg_similarity*100:.1f}%\n"
            f"       🏷️ Relevancia de dominio promedio: {avg_domain*100:.1f}%\n"
            f"       🎯 Rango similitud: {min(art['similarity'] for art in final_selection)*100:.1f}% - {max(art['similarity'] for art in final_selection)*100:.1f}%\n"
        )
    else:
        logging.warning("⚠️ SCREENING V4: Ningún artículo superó los umbrales de relevancia")

    return final_selection[:max_results]

# ============================================================
# MODO CRITERIOS: AI SCREENING DIRECTO (v1.0)
# Evaluacion de TODOS los articulos contra criterios I/E con IA
# ============================================================

def ai_criteria_screening_full(
    articles: List[Dict],
    question: str,
    inc_lines: List[str],
    exc_lines: List[str],
    target_n: int = 100,
    max_workers: int = 8,
) -> List[Dict]:
    """
    Evalua TODOS los articulos directamente contra criterios I/E con Cerebras.
    Reemplaza al pipeline fuzzy/embeddings cuando el investigador define criterios.

    Logica:
    - include=True  si cumple TODOS los criterios de inclusion
                    Y no cumple NINGUNO de exclusion
    - include=False en cualquier otro caso
    - Siempre retorna target_n articulos:
        primero los calificados (ordenados por score),
        luego relleno si son necesarios para llegar a 100
    """
    api_key = config.CEREBRAS_API_KEYS[0]

    criteria_block = ""
    if inc_lines:
        criteria_block += "INCLUSION CRITERIA (article MUST meet ALL):\n"
        criteria_block += "\n".join(f"- {c}" for c in inc_lines) + "\n\n"
    if exc_lines:
        criteria_block += "EXCLUSION CRITERIA (article is EXCLUDED if it meets ANY):\n"
        criteria_block += "\n".join(f"- {c}" for c in exc_lines) + "\n\n"

    logging.info(f"🔬 [Criteria Screen] Evaluando {len(articles)} articulos con Cerebras...")

    def evaluate_article(art: Dict):
        title    = art.get('title', '')[:150]
        abstract = art.get('abstract', '')[:500]
        prompt = (
            f'You are a strict academic screener for a PRISMA systematic literature review.\n'
            f'Research question: "{question}"\n\n'
            f'{criteria_block}'
            f'Article title: "{title}"\n'
            f'Abstract: "{abstract}"\n\n'
            'EVALUATION PROTOCOL (LLM-as-a-Judge):\n'
            '1. For EACH inclusion criterion, find a SPECIFIC PHRASE in the abstract that confirms it.\n'
            '   If you cannot quote a specific phrase → the criterion is NOT met → include=false.\n'
            '2. For EACH exclusion criterion, check if the abstract matches ANY of them.\n'
            '   If it matches even ONE → include=false.\n'
            '3. "score" reflects how STRONGLY the article meets ALL criteria (1-10).\n\n'
            'STRICT RULES:\n'
            '- include=true ONLY if ALL inclusion criteria are confirmed by explicit text in the abstract.\n'
            '- If the abstract is ambiguous or vague, set include=false (benefit of the doubt goes to exclusion).\n'
            '- Reviews, surveys, meta-analyses, and state-of-the-art papers score 1-3.\n\n'
            'Respond ONLY with valid JSON: {"include": true, "score": 8, "reason": "one-line justification"}'
        )
        try:
            resp = requests.post(
                config.CEREBRAS_ENDPOINT,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": config.CEREBRAS_MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 80, "temperature": 0.0},
                timeout=15
            )
            content = resp.json()["choices"][0]["message"]["content"].strip()
            # Regex: acepta cualquier orden de keys
            m1 = re.search(r'"include"\s*:\s*(true|false)', content)
            m2 = re.search(r'"score"\s*:\s*(\d+)', content)
            if m1 and m2:
                return m1.group(1) == "true", float(m2.group(1))
            data = json.loads(content)
            return bool(data.get("include", True)), float(data.get("score", 5))
        except Exception as e:
            logging.debug(f"[Criteria] Error '{title[:35]}': {e}")
            return True, 5.0  # Error de red: incluir con score neutro

    # Evaluar todos concurrentemente
    eval_results: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_article, art): i for i, art in enumerate(articles)}
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                inc, score = future.result()
            except Exception:
                inc, score = True, 5.0
            eval_results[idx] = (inc, score)
            done += 1
            if done % 100 == 0:
                logging.info(f"   🔄 Progreso: {done}/{len(articles)} evaluados...")

    # Clasificar y anotar
    qualified, fill_pool = [], []
    for i, art in enumerate(articles):
        inc, score = eval_results.get(i, (True, 5.0))
        art['ai_relevance_score'] = round(score, 1)
        art['ai_evaluated']       = True
        art['criteria_passed']    = inc
        art['similarity']         = round(score / 10.0, 4)  # % que ve el usuario
        (qualified if inc else fill_pool).append(art)

    qualified.sort(key=lambda x: x.get('similarity', 0), reverse=True)
    fill_pool.sort(key=lambda x: x.get('similarity', 0), reverse=True)

    # Garantizar target_n: calificados primero, relleno si son pocos
    result = qualified[:target_n]
    if len(result) < target_n:
        needed = target_n - len(result)
        result.extend(fill_pool[:needed])
        logging.info(
            f"   ⚠️ Solo {len(qualified)} cumplen criterios; "
            f"se agregan {min(needed, len(fill_pool))} de relleno para llegar a {target_n}"
        )

    logging.info(
        f"✅ [Criteria Screen] Evaluados: {len(articles)} | "
        f"Cumplen criterios: {len(qualified)} | Descartados: {len(fill_pool)} | "
        f"Mostrando: {len(result)}"
    )
    return result


# ============================================================
# EMBUDO ETAPA 2: MULTI-CRITERION SCORER (v1.0)
# Scoring 3D sobre top-200 pre-filtrados por embeddings
# ============================================================

# Palabras clave que indican que un articulo ES un survey/review
_REVIEW_SIGNALS = {
    "systematic review", "literature review", "scoping review",
    "mapping study", "meta-analysis", "state of the art",
    "survey of", "we surveyed", "this paper reviews",
    "this paper summarizes", "a review of", "review paper",
    "bibliometric",
}

def ai_multicriteria_score(
    candidates: List[Dict],
    original_question: str,
    target_n: int = 100,
    candidate_pool: int = 200,
    max_workers: int = 8,
    inclusion_criteria: str = "",
    exclusion_criteria: str = "",
) -> List[Dict]:
    """
    Etapa 2 del embudo: scoring multi-dimensional sobre top-200.

    3 dimensiones (Cerebras llama3.1-8b fast scoring):
    1. inclusion_fit  (1-10): cumple criterios de inclusion del investigador
    2. study_type     (1-10): experimental/aplicado? (surveys -> 1-3)
    3. exclusion_match(1-10): viola criterios de exclusion? (mayor=peor)

    Score final = 0.25*embed + 0.45*(inclusion_fit/10) + 0.30*(study_type/10)

    Hard exclusion:
    - study_type <= 3  (es un survey/review/estado del arte)
    - exclusion_match >= 6 (viola criterio de exclusion del investigador)
    - inclusion_fit < 4 (claramente no cumple criterios)

    Siempre retorna target_n articulos:
    primero los calificados (ordenados por score combinado),
    luego relleno de la zona gris si son necesarios para llegar a 100.
    """
    pool = candidates[:candidate_pool]
    if not pool:
        return candidates[:target_n]

    api_key = config.CEREBRAS_API_KEYS[0]

    # Construir bloques de criterios
    inc_block = ""
    exc_block = ""
    inc_lines_parsed = [l.strip() for l in inclusion_criteria.strip().splitlines() if l.strip()] if inclusion_criteria else []
    exc_lines_parsed = [l.strip() for l in exclusion_criteria.strip().splitlines() if l.strip()] if exclusion_criteria else []
    if inc_lines_parsed:
        inc_block = "INCLUSION CRITERIA:\n" + "\n".join(f"- {c}" for c in inc_lines_parsed)
    if exc_lines_parsed:
        exc_block = "EXCLUSION CRITERIA:\n" + "\n".join(f"- {c}" for c in exc_lines_parsed)

    provider_label = "Cerebras 8B (3D scorer)"
    logging.info(f"🤖 [MCS] {provider_label} | {len(pool)} candidatos | 3 dimensiones")

    def score_article(art: Dict) -> tuple:
        title    = art.get('title', '')[:150]
        abstract = art.get('abstract', '')[:500]

        # Pre-filtro rapido: detectar reviews por texto sin llamar a la IA
        abstract_lower = abstract.lower()
        is_review_fast = any(sig in abstract_lower for sig in _REVIEW_SIGNALS)

        prompt = (
            f'Research question: "{original_question}"\n'
            + (f'\n{inc_block}\n' if inc_block else '') +
            (f'{exc_block}\n' if exc_block else '') +
            f'\nTitle: "{title}"\nAbstract: "{abstract}"\n\n'
            'Score this article on 3 dimensions (1-10 each):\n'
            '"i" (inclusion_fit): Does abstract clearly meet ALL inclusion criteria?\n'
            '   10=clearly meets all, 5=partially, 1=fails or unclear\n'
            '   If no inclusion criteria given, score how relevant to the research question.\n'
            '"s" (study_type): Is this an EXPERIMENTAL/APPLIED study with real results?\n'
            '   9-10=has methodology+metrics+numerical results\n'
            '   6-8=applied framework or tool evaluation\n'
            '   3-5=theoretical or preliminary\n'
            '   1-3=survey, review, mapping, state-of-the-art, meta-analysis\n'
            '   AUTO-RULE: Contains "review", "survey", "mapping", "state of the art" => score 1-3\n'
            '"e" (exclusion_match): Does it match ANY exclusion criterion?\n'
            '   1=matches none, 5=partially, 10=clearly excluded\n\n'
            'ONLY valid JSON: {"i": N, "s": N, "e": N}'
        )
        try:
            resp = requests.post(
                config.CEREBRAS_ENDPOINT,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": config.CEREBRAS_MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 25, "temperature": 0.0},
                timeout=15
            )
            content = resp.json()["choices"][0]["message"]["content"].strip()
            mi = re.search(r'"i"\s*:\s*(\d+)', content)
            ms = re.search(r'"s"\s*:\s*(\d+)', content)
            me = re.search(r'"e"\s*:\s*(\d+)', content)
            i_score = float(mi.group(1)) if mi else 5.0
            s_score = float(ms.group(1)) if ms else 5.0
            e_score = float(me.group(1)) if me else 3.0
            # Override study_type si el pre-filtro detecto review
            if is_review_fast:
                s_score = min(s_score, 3.0)
            return i_score, s_score, e_score
        except Exception as ex:
            logging.debug(f"[MCS] Error '{title[:35]}': {ex}")
            s_fallback = 3.0 if is_review_fast else 5.0
            return 5.0, s_fallback, 3.0

    # Evaluar todos concurrentemente
    mcs_results: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(score_article, art): i for i, art in enumerate(pool)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                mcs_results[idx] = future.result()
            except Exception:
                mcs_results[idx] = (5.0, 5.0, 3.0)

    # Aplicar scores y separar calificados de relleno
    qualified, fill_pool = [], []
    for i, art in enumerate(pool):
        i_score, s_score, e_score = mcs_results.get(i, (5.0, 5.0, 3.0))

        embed_sim   = art.get('similarity', 0.5)  # score original de embeddings
        combined    = 0.25 * embed_sim + 0.45 * (i_score / 10.0) + 0.30 * (s_score / 10.0)

        art['ai_relevance_score'] = round(i_score, 1)
        art['ai_study_type']      = round(s_score, 1)
        art['ai_excl_match']      = round(e_score, 1)
        art['ai_evaluated']       = True
        art['similarity']         = round(combined, 4)  # % display = score combinado

        # Hard exclusion
        hard_exclude = (
            s_score <= 3.0     # Es un survey/review
            or e_score >= 6.0  # Viola criterio de exclusion
            or i_score < 4.0   # No cumple criterios de inclusion
        )
        (fill_pool if hard_exclude else qualified).append(art)

    qualified.sort(key=lambda x: x.get('similarity', 0), reverse=True)
    fill_pool.sort(key=lambda x: x.get('similarity', 0), reverse=True)

    # Garantizar target_n: calificados primero, relleno si son pocos
    result = qualified[:target_n]
    if len(result) < target_n:
        needed = target_n - len(result)
        result.extend(fill_pool[:needed])
        logging.info(f"   📦 Relleno: {min(needed, len(fill_pool))} arts adicionales para completar {target_n}")

    avg_combined = np.mean([a.get('similarity', 0) for a in result]) if result else 0
    logging.info(
        f"✅ [MCS] Evaluados: {len(pool)} | Calificados: {len(qualified)} | "
        f"Descartados (hard): {len(fill_pool)} | Total: {len(result)} | "
        f"Score combinado prom: {avg_combined*100:.1f}%"
    )
    return result



def ai_rerank_abstracts(
    candidates: List[Dict],
    original_question: str,
    target_n: int = 100,
    candidate_pool: int = 200,
    soft_cutoff: float = 6.0,    # Umbral "preferido" — artículos sobre 6 van primero
    max_workers: int = 4,
    inclusion_criteria: str = "",
    exclusion_criteria: str = "",
) -> List[Dict]:
    """
    Re-rankea candidatos con IA como árbitro principal.
    SIEMPRE devuelve exactamente target_n (100) artículos.

    v3.1 — clave: NUNCA filtra por debajo de target_n:
    - Pool top-200 se evalúa con Groq 70B (o Cerebras como fallback)
    - Los criterios I/E del investigador guían el scoring
    - Resultado ordenado: primero los ≥ soft_cutoff (mejor calidad),
      luego los < soft_cutoff como relleno si son necesarios para llegar a 100
    - Score display = ai_score / 10 (% que ve el usuario = opinión de la IA)

    Flujo PRISMA correcto:
      Fuzzy/Embeddings → top-200 (pre-filtro veloz)
      Groq 70B        → re-ordena los 200 por relevancia real
      Sistema         → muestra top-100 al investigador para revisión manual
    """
    pool = candidates[:candidate_pool]
    if not pool:
        return candidates[:target_n]

    # Elegir proveedor: preferir Groq 70B (mucho más inteligente que 8B)
    use_groq = bool(config.GROQ_API_KEY)
    api_key  = config.GROQ_API_KEY if use_groq else (config.CEREBRAS_API_KEYS[0] if config.CEREBRAS_API_KEYS else None)
    if not api_key:
        logging.warning("[AI Rerank] Sin clave disponible, omitiendo re-ranking")
        return candidates[:target_n]

    endpoint = config.GROQ_ENDPOINT if use_groq else config.CEREBRAS_ENDPOINT
    model    = config.GROQ_MODEL    if use_groq else config.CEREBRAS_MODEL
    provider = "Groq 70B" if use_groq else "Cerebras 8B"

    logging.info(f"🤖 [AI Rerank v3.1 | {provider}] Evaluando {len(pool)} candidatos → garantiza {target_n} arts...")

    # Construir bloque de criterios del investigador
    criteria_block = ""
    inc_lines = [l.strip() for l in inclusion_criteria.strip().splitlines() if l.strip()] if inclusion_criteria else []
    exc_lines = [l.strip() for l in exclusion_criteria.strip().splitlines() if l.strip()] if exclusion_criteria else []
    if inc_lines:
        criteria_block += "\nINCLUSION CRITERIA (article MUST meet ALL):\n" + "\n".join(f"- {c}" for c in inc_lines)
    if exc_lines:
        criteria_block += "\nEXCLUSION CRITERIA (score 1-3 if ANY criterion is met):\n" + "\n".join(f"- {c}" for c in exc_lines)

    def score_article(art: Dict) -> float:
        title    = art.get('title', '')[:150]
        abstract = art.get('abstract', '')[:500]
        prompt = (
            f'You are a systematic literature review expert.\n'
            f'Research question: "{original_question}"'
            + (criteria_block if criteria_block else '') +
            f'\n\nArticle title: "{title}"\n'
            f'Abstract: "{abstract}"\n\n'
            'Score relevance 1-10 (be STRICT and ACADEMIC):\n'
            '- 9-10: Directly and empirically addresses the research question with clear methods and results\n'
            '- 7-8: Clearly relevant, presents specific findings or framework on the topic\n'
            '- 5-6: Tangentially related or lacks empirical rigor for this specific question\n'
            '- 1-4: Wrong domain, off-topic, or does not address the research question\n'
            + (f'\nApply exclusion criteria strictly: if ANY exclusion criterion is met, score 1-3.\n' if exc_lines else '') +
            '\nRespond ONLY with valid JSON: {"score": N}'
        )
        try:
            resp = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 20, "temperature": 0.0},
                timeout=20
            )
            if resp.status_code == 429:
                time.sleep(3)
                raise Exception("Rate limited")
            content = resp.json()["choices"][0]["message"]["content"].strip()
            match = re.search(r'\{\s*"score"\s*:\s*(\d+)\s*\}', content)
            if match:
                return float(match.group(1))
            return float(json.loads(content).get("score", 5))
        except Exception as e:
            logging.debug(f"[AI Rerank] Error '{title[:35]}': {e}")
            return 5.0

    # Evaluar con batches + rate limiting para Groq (30 req/min free tier)
    ai_raw_scores: dict = {}
    pool_items  = list(enumerate(pool))
    batch_size  = max_workers

    for batch_start in range(0, len(pool_items), batch_size):
        batch = pool_items[batch_start: batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(score_article, art): idx for idx, art in batch}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    ai_raw_scores[idx] = future.result()
                except Exception:
                    ai_raw_scores[idx] = 5.0
        # Pausa entre batches para respetar rate limit de Groq
        if use_groq and batch_start + batch_size < len(pool_items):
            time.sleep(2.5)

    # Anotar score en cada artículo
    for i, art in enumerate(pool):
        ai_score = ai_raw_scores.get(i, 5.0)
        art['ai_relevance_score'] = round(ai_score, 1)
        art['ai_evaluated']       = True
        art['similarity']         = round(ai_score / 10.0, 4)  # % display = opinión IA

    # Separar: primero los "buenos" (≥ soft_cutoff), luego el resto como relleno
    good_arts = sorted([a for a in pool if a.get('ai_relevance_score', 0) >= soft_cutoff],
                       key=lambda x: x.get('similarity', 0), reverse=True)
    fill_arts = sorted([a for a in pool if a.get('ai_relevance_score', 0) < soft_cutoff],
                       key=lambda x: x.get('similarity', 0), reverse=True)

    # GARANTIZA target_n artículos: buenos primero, relleno si necesario
    result = good_arts[:target_n]
    if len(result) < target_n:
        needed = target_n - len(result)
        result.extend(fill_arts[:needed])
        logging.info(f"   📦 Relleno: {min(needed, len(fill_arts))} artículos adicionales para completar {target_n}")

    ai_avg = np.mean(list(ai_raw_scores.values())) if ai_raw_scores else 0
    logging.info(
        f"✅ [AI Rerank v3.1 | {provider}] Evaluados: {len(pool)} | "
        f"Calificados (≥{soft_cutoff}): {len(good_arts)} | "
        f"Total final: {len(result)} | Score IA promedio: {ai_avg:.1f}/10"
    )
    return result
