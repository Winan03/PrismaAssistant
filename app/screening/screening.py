import logging
import re
import time
import json
import numpy as np
from typing import List, Dict, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity

import requests
import config
from app.llm.embedding_service import get_embeddings, get_single_embedding
from app.domain.query_expander import expand_query_with_llm, extract_english_terms, extract_english_terms_with_llm, get_exclusion_terms_with_llm, expand_query_with_synonyms
from app.screening.metadata_filter import apply_hard_filters, parse_exclusion_criteria_for_hard_filter
from app.extraction.bm25_retriever import compute_hybrid_scores
from app.llm.cross_encoder_reranker import rerank_with_cross_encoder
from app.llm.ai_model import LocalModel

# ============================================================
# FILTRO DE TERMINOS SEGUROS PARA BM25
# Bloquea acrónimos cortos y ambiguos que envenenan el índice léxico
# Ejemplos problemáticos: 'IA', 'ST', 'AI' → coinciden con todo
# ============================================================
_MIN_TERM_LENGTH = 3   # Mínimo de caracteres para entrar al BM25

def _filter_safe_terms(terms: List[str]) -> List[str]:
    """
    Elimina términos cortos y ambiguos que envenenan el índice BM25.
    Reglas:
      1. Ignorar si len <= 2 chars
      2. Ignorar si es todo mayúsculas con <= 3 chars (acrónimo genérico)
    """
    result = []
    for t in terms:
        t_clean = t.strip()
        if not t_clean:
            continue
        if len(t_clean) < _MIN_TERM_LENGTH:
            continue  # Muy corto → ruido
        if t_clean.isupper() and len(t_clean) <= 3:
            continue  # Acrónimo genérico desconocido (ALL-CAPS ≤ 3 chars)
        result.append(t_clean)
    return result


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

# Las listas hardcodeadas han sido eliminadas para mantener la herramienta 100% agnóstica.
# Ahora se usa un LLM dinámicamente para filtrar stopwords y extraer términos.
# para mantener la herramienta 100% agnóstica. Ahora se usa un LLM dinámicamente.

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
        content = LocalModel.get_instance().generate(
            instruction=prompt,
            max_tokens=150,
            system_prompt="You are a research engineer. Extract comparison poles as JSON only."
        ).strip()
        
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n", "", content)
            content = re.sub(r"\n```$", "", content)
            
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

    # 1. Traducir y extraer términos clave de dominio en inglés dinámicamente usando LLM
    llm_terms = extract_english_terms_with_llm(question)
    for term in llm_terms:
        for word in term.split():
            if len(word) > 2:
                domain_keywords.add(word.lower())

    logging.info(f"🏷️ Keywords de dominio extraidos (via LLM): {sorted(domain_keywords)}")
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
        # RELAJADO: En RSLs incipientes, muchos estudios no tienen grupo de control tradicional.
        # Se reduce el malus de 0.6 a 0.85 para no destruir el score de estudios exploratorios.
        if not pole_a_present or not pole_b_present:
            base_boost *= 0.85 # Malus del 15% en lugar del 40%
            
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
    if x < a or x > d:
        return 0.0
    elif x <= b:
        return (x - a) / (b - a) if b > a else 1.0
    elif x <= c:
        return 1.0
    else:
        return (d - x) / (d - c) if d > c else 1.0

def _tri(x: float, a: float, b: float, c: float) -> float:
    """Función de pertenencia triangular: sube a->b, baja b->c."""
    if x < a or x > c:
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
    # Asegurar rango [0.0, 1.0]
    semantic_sim = min(max(semantic_sim, 0.0), 1.0)

    # Normalizar kw_boost al rango [0,1] (máx estimado ~0.20 según pipeline)
    kw = min(max(kw_boost_raw / 0.20, 0.0), 1.0)
    domain_rel = min(max(domain_rel, 0.0), 1.0)

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

    if denominator == 0.0:
        logging.error(f"❌ ZERO DENOMINATOR in compute_fuzzy_score! semantic_sim={semantic_sim}, domain_rel={domain_rel}, kw={kw}")
        denominator = 1e-9

    fuzzy_val = numerator / denominator

    # Piso dinámico para evitar penalización excesiva por baja similitud vectorial
    # cuando la relevancia de dominio es alta.
    # Cuando domain_rel == 1.0, el piso de fuzzy_score es exactamente 0.30.
    if domain_rel > 0.0:
        min_floor = 0.08 + (0.30 - 0.08) * domain_rel
        fuzzy_val = max(fuzzy_val, min_floor)

    return fuzzy_val

def screen_articles(articles: List[Dict], query: str, threshold: float = 0.70,
                    max_results: int = 50, original_question: str = "",
                    inclusion_criteria: str = "", exclusion_criteria: str = "",
                    conservative: bool = False) -> Tuple[List[Dict], List[Dict]]:
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
    Capa 5: LLM-as-a-Judge (Cerebras) con justificación obligatoria
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    conservative=True → Modo RSL de alta sensibilidad (Oami et al. 2024):
      - RAW_SCORE_FLOOR = 0.0 (todos los artículos reciben score)
      - LLM-Judge: en caso de duda → INCLUIR (minimiza Falsos Negativos)
      - Hard exclusion thresholds: relajados (FP aceptables, FN no)
    """

    if not articles:
        return [], []

    full_ranking = max_results <= 0

    # ============================================================
    # CAPA 0: HARD FILTER POR METADATOS (Pre-Vectorial)
    # ============================================================
    from app.domain.query_expander import get_exclusion_terms_with_llm, get_venue_blocklist_with_llm

    min_year, extra_excl_terms = parse_exclusion_criteria_for_hard_filter(exclusion_criteria)
    
    # Generar exclusiones de dominio dinámicas desde la RQ (cacheadas en .cache/)
    llm_exclusion_terms = get_exclusion_terms_with_llm(original_question or query)
    llm_venue_blocklist = get_venue_blocklist_with_llm(original_question or query)
    all_exclusion_terms = list(set((extra_excl_terms or []) + (llm_exclusion_terms or [])))

    articles, excluded_metadata, hard_filter_report = apply_hard_filters(
        articles,
        min_year=min_year,
        min_abstract_length=80,   # Excluir artículos sin abstract útil
        extra_exclusion_terms=all_exclusion_terms,
        excluded_venues=llm_venue_blocklist,
    )
    if not articles:
        logging.warning("⚠️ Hard Filter eliminó todos los artículos. Verifica los criterios de exclusión.")
        return [], excluded_metadata


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

        english_terms = extract_english_terms_with_llm(enriched_query)
        # ── Filtrar acrónimos ambiguos ANTES de BM25 (Bug Fix) ──
        # Acrónimos como 'IA', 'ST', 'AI' indexan todo el corpus y anulan BM25
        english_terms = _filter_safe_terms(list(set(english_terms + synonym_terms)))[:20]
        synonym_terms  = _filter_safe_terms(synonym_terms)  # También para keyword boost
        if synonym_terms != synonym_data.get('flat_terms', []):
            blocked = set(synonym_data.get('flat_terms', [])) - set(synonym_terms)
            if blocked:
                logging.info(f"🚫 [BM25 Filter] Acrónimos bloqueados del índice: {blocked}")
        
        exclusion_terms = get_exclusion_terms_with_llm(original_q)
        # Combinar exclusiones: LLM + manuales del investigador
        exclusion_terms = list(set(exclusion_terms + manual_exclusion_terms))
        comparison_poles = _extract_comparison_poles(original_q)
        
        if comparison_poles:
            poles_str = " vs ".join([p[0] for p in comparison_poles])
            logging.info(f"⚖️ Polos de comparación detectados: {poles_str}")

        logging.info(f"🌎 Screening con {len(semantic_queries)} queries semánticas + {len(synonym_queries)} queries de sinónimos")

        # ── CAPA 1A: Pre-Filtro Léxico BM25 (Ultra Rápido) ──
        try:
            from app.extraction.bm25_retriever import BM25Retriever, reciprocal_rank_fusion
            retriever = BM25Retriever(texts)
            bm25_raw_scores = retriever.get_multi_query_scores(all_bm25_queries)
        except Exception as e:
            logging.error(f"Error en BM25: {e}")
            bm25_raw_scores = np.zeros(len(texts))

        # Seleccionar el Top-1500 léxico para computar embeddings pesados
        MAX_EMBEDDING_POOL = min(1500, len(texts))
        top_indices = np.argsort(-bm25_raw_scores)[:MAX_EMBEDDING_POOL]
        top_texts = [texts[i] for i in top_indices]

        # ── CAPA 1B: Scoring de Embeddings (SOLO PARA EL TOP) ──
        logging.info(f"⚡ Computando embeddings SOLO para el Top-{MAX_EMBEDDING_POOL} léxico (ahorro de CPU del {(1 - MAX_EMBEDDING_POOL/len(texts))*100:.1f}%)...")
        corpus_embeddings = get_embeddings(top_texts)  # shape: (n_top, 384)
        
        all_query_scores = []
        for q in semantic_queries:
            q_emb = get_single_embedding(q)
            q_scores = cosine_similarity([q_emb], corpus_embeddings)[0]
            all_query_scores.append(q_scores)
            
        all_query_scores = np.array(all_query_scores)  # (n_queries, n_top)
        max_scores = np.max(all_query_scores, axis=0)
        mean_scores = np.mean(all_query_scores, axis=0)
        top_embedding_raw_scores = (max_scores * 0.7) + (mean_scores * 0.3)

        # Reconstruir el array completo de scores de embeddings (los descartados tienen 0)
        embedding_raw_scores = np.zeros(len(texts))
        for idx, original_idx in enumerate(top_indices):
            embedding_raw_scores[original_idx] = top_embedding_raw_scores[idx]

        # ── CAPA 1C: RRF Fusion ──
        try:
            rrf_scores = reciprocal_rank_fusion(
                embedding_scores=embedding_raw_scores,
                bm25_scores=bm25_raw_scores,
                weight_embedding=0.6,
                weight_bm25=0.4,
            )
        except Exception as e:
            logging.error(f"Error en RRF: {e}")
            rrf_scores = embedding_raw_scores

        # El score final de la Capa 1 es el RRF fusionado
        final_raw_scores = rrf_scores

        # 🧠 DETECTAR DOMINIO para validación suave
        domain_results = detect_search_domain(original_question or query)
        detected_domain = domain_results['id']

        # 3. Normalización Robusta (Percentiles P5-P95)
        normalized_scores = normalize_scores_robust(final_raw_scores)
        
        max_raw = np.max(final_raw_scores)
        
        # --- UMBRAL MINIMO DE CALIDAD ---
        # conservative=True → floor=0.0 (evalúa TODOS los artículos, modo RSL)
        # conservative=False → floor=0.20 (modo producción normal)
        corpus_size = len(articles)
        RAW_SCORE_FLOOR = 0.0 if conservative else 0.20
            
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
            fuzzy_val = compute_fuzzy_score(rel_val, domain_rel_val, kw_boost)

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
    target_results = len(eligible) if full_ranking else max(1, max_results)
    ranking_label = "ranking completo" if full_ranking else f"top-{target_results}"
    logging.info(f"🔍 Candidatos (raw≥{RAW_SCORE_FLOOR}): {raw_above_floor} | Pre-CE {ranking_label}")

    # 3. Priorizar artículos con URL válida dentro del top-N
    with_url    = [a for a in eligible if a.get('url') and len(str(a.get('url'))) > 10]
    without_url = [a for a in eligible if not (a.get('url') and len(str(a.get('url'))) > 10)]

    # En ranking completo se conserva todo el orden; el reranker solo recalibra la cabeza.
    ce_pool_size = len(eligible) if full_ranking else min(200, len(eligible))
    final_selection = with_url[:ce_pool_size]
    if len(final_selection) < ce_pool_size:
        needed = ce_pool_size - len(final_selection)
        final_selection.extend(without_url[:needed])

    final_selection.sort(key=lambda x: x.get('similarity', 0), reverse=True)

    # ── CAPA 4: CROSS-ENCODER RE-RANKING (top-50 candidatos) ──
    # IMPORTANTE: ms-marco-MiniLM espera una query CORTA EN INGLÉS.
    # Se usa la cadena de conceptos generada por el LLM en main.py.
    ce_query = query
    if ce_query:
        import string
        import nltk
        from nltk.corpus import stopwords
        
        # 1. Cargamos los stopwords en inglés
        try:
            stop_words_ingles = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            stop_words_ingles = set(stopwords.words('english'))
        
        # 2. Quitamos signos de puntuación y pasamos a minúsculas
        texto_sin_puntuacion = ce_query.translate(str.maketrans('', '', string.punctuation)).lower()
        
        # 3. Filtramos las palabras: nos quedamos solo con las que NO son stopwords
        ce_words = [
            palabra for palabra in texto_sin_puntuacion.split() 
            if palabra not in stop_words_ingles
        ]
        
        if len(ce_words) < 4 and 'english_terms' in locals() and english_terms:
            extra_texto = " ".join([t for t in english_terms if t.lower() not in texto_sin_puntuacion]).translate(str.maketrans('', '', string.punctuation)).lower()
            extra_words = [w for w in extra_texto.split() if w not in stop_words_ingles][:3]
            ce_words.extend(extra_words)
            
        if len(ce_words) > 75:  # Límite de seguridad
            ce_words = ce_words[:75]
            
        # 4. Unimos las palabras clave en un solo string
        ce_query = " ".join(ce_words)
            
        logging.info(f"🎯 Cross-Encoder: re-rankeando top-50 de {len(final_selection)} candidatos...")
        logging.info(f"   CE query (Filtrada NLP): '{ce_query}'")

        # --- CAPA 4.5: FILTRO DE RUIDO SEMÁNTICO PRE-CE (Agnóstico) ---
        # El filtrado de ruido queda delegado al contrato y a los criterios de la pregunta.

        final_selection = rerank_with_cross_encoder(
            candidates=final_selection,
            query=ce_query,
            top_n=50,
            batch_size=16,
        )

    # Recortar solo si se solicito un limite positivo.
    if not full_ranking:
        final_selection = final_selection[:target_results * 2]  # Pool 2x para el AI-Judge

    # ============================================================
    # AI ABSTRACT RE-RANKER (v1.0)
    # Evalua abstracts en zona ambigua con IA para eliminar falsos positivos
    # ============================================================
    if original_question and config.CEREBRAS_API_KEYS:
        final_selection = ai_multicriteria_score(
            candidates=final_selection,
            original_question=original_question,
            target_n=target_results,
            candidate_pool=len(final_selection) if full_ranking else min(len(final_selection), 200),
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
        )

    # ============================================================
    # OPTIMIZACION AVANZADA DE SCREENING (INFERENCIA DUAL Y ZONA GRIS)
    # ============================================================
    try:
        from app.screening.screening_improvements import (
            DualThresholdClassifier, AdaptiveVocabularyExpander,
            AgnosticRateLimiter, RateLimiterConfig
        )
        from app.llm.ai_model import LocalModel

        # 1. Definir callable del LLM
        def llm_caller(instruction: str, input_text: str = "") -> str:
            return LocalModel.get_instance().generate(instruction, input_text)

        # 2. Calibrar y clasificar
        gs_scores = [a.get("similarity", 0.0) for a in final_selection]
        
        classifier = DualThresholdClassifier()
        classifier.calibrate(gs_scores, conservative=conservative)

        classified = classifier.classify_batch(final_selection)
        auto_in, grey, auto_out = classifier.split(classified)

        logging.info(f"   [Classifier] Auto-Include: {len(auto_in)} | Grey-Zone: {len(grey)} | Auto-Exclude: {len(auto_out)}")

        # Log rescued articles
        rescued_count = sum(1 for art in grey if art.get("rescued_by_domain", False))
        if rescued_count > 0:
            logging.info(f"   [Rescue] Rescatados {rescued_count} artículos por relevancia de dominio.")

        # 3. Configurar Rate Limiter y Expander
        rate_cfg = RateLimiterConfig(max_concurrent_calls=3, calls_per_second=1.5)
        rate_limiter = AgnosticRateLimiter(rate_cfg)

        expander = AdaptiveVocabularyExpander(
            llm_caller=llm_caller,
            inclusion_criteria=inclusion_criteria,
            domain_description=(original_question or query or "general academic literature")[:200]
        )

        # 4. Evaluar la zona gris de forma asíncrona
        async def run_expansion():
            return await expander.expand_and_rerank(grey, rate_limiter)

        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        try:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                grey_confirmed, grey_rejected = executor.submit(lambda: asyncio.run(run_expansion())).result()
        except RuntimeError:
            grey_confirmed, grey_rejected = asyncio.run(run_expansion())

        # 5. Re-consolidar artículos finalistas
        final_screened_arts = []
        for a in auto_in:
            a_copy = a.copy()
            a_copy["passed"] = True
            final_screened_arts.append(a_copy)
            
        for a in grey_confirmed:
            final_screened_arts.append(a)
            
        for a in grey_rejected:
            final_screened_arts.append(a)
            
        for a in auto_out:
            a_copy = a.copy()
            a_copy["passed"] = False
            final_screened_arts.append(a_copy)

        # Filtrar solo los aprobados (passed == True) para mantener el formato original
        final_selection = [a for a in final_screened_arts if a.get("passed", False)]
        
        logging.info(f"   [Consolidate] Total articulos aprobados final consolidado: {len(final_selection)}")
    except Exception as e_imp:
        logging.warning(f"⚠️ Error al ejecutar optimizaciones avanzadas en screen_articles: {e_imp}")

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

    if full_ranking:
        return final_selection, excluded_metadata
    return final_selection[:target_results], excluded_metadata

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

    def evaluate_article(art: Dict, idx: int):
        title    = art.get('title', '')[:300]
        abstract = art.get('abstract', '')[:3000]
        prompt = (
            f'You are a conservative academic screener for a PRISMA systematic literature review.\n'
            f'Research question: "{question}"\n\n'
            f'{criteria_block}'
            f'Article title: "{title}"\n'
            f'Abstract: "{abstract}"\n\n'
            'EVALUATION PROTOCOL (LLM-as-a-Judge — CONSERVATIVE RSL MODE):\n'
            '1. For EACH inclusion criterion, check if the abstract COULD plausibly address it.\n'
            '   In RSL, it is worse to miss a relevant article than to include a borderline one.\n'
            '2. For EACH exclusion criterion, it must CLEARLY and EXPLICITLY match to exclude.\n'
            '   Do NOT exclude based on vague or partial similarity to an exclusion criterion.\n'
            '3. "score" reflects how STRONGLY the article meets ALL criteria (1-10).\n\n'
            'CONSERVATIVE RULES (Oami et al. 2024 protocol):\n'
            '- include=true if the article is PLAUSIBLY relevant, even if not 100% confirmed.\n'
            '- If the abstract is ambiguous or vague, set include=TRUE (benefit of the doubt goes to INCLUSION).\n'
            '- Only include=false when the article CLEARLY does not match the research question.\n'
            '- Reviews, surveys, meta-analyses score 1-3 on score, but may still have include=true.\n\n'
            'Respond ONLY with valid JSON: {"include": true, "score": 8, "reason": "one-line justification"}'
        )
        results = []
        for run_idx in range(3):
            try:
                content = LocalModel.get_instance().generate(
                    instruction=prompt,
                    max_tokens=80,
                    system_prompt="You are a conservative academic screener. Respond ONLY with a valid raw JSON object matching the requested schema."
                ).strip()
                
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\n", "", content)
                    content = re.sub(r"\n```$", "", content)
                
                data = json.loads(content)
                include_val = data.get("include")
                score_val = data.get("score")
                reason_val = data.get("reason", "")
                
                if include_val is not None and score_val is not None:
                    results.append((bool(include_val), float(score_val), reason_val))
                else:
                    cat = int(data.get("category", 0))
                    if cat == 1:
                        results.append((True, 10.0, reason_val))
                    elif cat == 2:
                        results.append((True, 7.0, reason_val))
                    else:
                        results.append((False, 1.0, reason_val))
            except Exception as e:
                logging.debug(f"[Criteria] Error in run {run_idx+1}/3 for '{title[:35]}': {e}")
                
        if not results:
            return True, 5.0  # Error de red/parsing: incluir con score neutro
            
        # Mayoría de votos
        trues = sum(1 for r in results if r[0])
        final_inc = trues >= 2 if len(results) >= 2 else results[0][0]
        final_score = sum(r[1] for r in results) / len(results)
        
        return final_inc, final_score

    # Evaluar todos concurrentemente
    eval_results: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_article, art, i): i for i, art in enumerate(articles)}
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

def classify_criteria_by_detectability(inclusion_criteria: str, exclusion_criteria: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Clasifica dinámicamente los criterios de inclusión y exclusión en:
    1. abstract_level: Altamente detectables en abstracts (señales léxicas/conceptos directos).
    2. full_text_level: Requieren texto completo / PDF (detalles de diseño, algoritmos, metodologías).
    
    100% Agnóstico al dominio.
    """
    default_res = {
        "inclusion": {
            "abstract_level": [],
            "full_text_level": []
        },
        "exclusion": {
            "abstract_level": [],
            "full_text_level": []
        }
    }
    
    inc_lines = [line.strip() for line in inclusion_criteria.splitlines() if line.strip()] if inclusion_criteria else []
    exc_lines = [line.strip() for line in exclusion_criteria.splitlines() if line.strip()] if exclusion_criteria else []
    
    if not inc_lines and not exc_lines:
        return default_res
        
    prompt = f"""You are a systematic literature review assistant.
We have user-defined inclusion and exclusion criteria for a PRISMA systematic review.
Since abstracts (typically 200 words) are too short, some criteria can be evaluated with high confidence just from the abstract (e.g., target population, primary technology medium, publication type), while other criteria require reading the full text / PDF (e.g., specific software architectures, complex algorithms, subtle pedagogical study designs, clinical vs pedagogical focus, precise outcomes).

Classify EACH of the following criteria into exactly ONE of two categories:
1. "abstract_level": Highly detectable in abstracts. (Simple lexical/conceptual signals, basic population, basic tech medium, review/survey detection).
2. "full_text_level": Requires full text/PDF. (Deep technical details, exact methodologies, complex outcomes, architectural nuances, subtle distinctions).

Criteria to classify:
INCLUSION:
{chr(10).join(f"- {line}" for line in inc_lines)}

EXCLUSION:
{chr(10).join(f"- {line}" for line in exc_lines)}

Respond ONLY with a valid JSON object in this format (no markdown code blocks, just raw JSON):
{{
  "inclusion": {{
    "abstract_level": ["criterio1", "criterio2"],
    "full_text_level": ["criterio3"]
  }},
  "exclusion": {{
    "abstract_level": ["criterio1"],
    "full_text_level": ["criterio2"]
  }}
}}
"""
    try:
        content = LocalModel.get_instance().generate(
            instruction=prompt,
            max_tokens=400,
            system_prompt="You are a systematic literature review assistant. Respond ONLY with a valid raw JSON object matching the requested schema."
        ).strip()
        
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n", "", content)
            content = re.sub(r"\n```$", "", content)
        data = json.loads(content)
        
        result = default_res.copy()
        if "inclusion" in data:
            result["inclusion"]["abstract_level"] = data["inclusion"].get("abstract_level", [])
            result["inclusion"]["full_text_level"] = data["inclusion"].get("full_text_level", [])
        if "exclusion" in data:
            result["exclusion"]["abstract_level"] = data["exclusion"].get("abstract_level", [])
            result["exclusion"]["full_text_level"] = data["exclusion"].get("full_text_level", [])
            
        logging.info(f"📋 Classified criteria: {result}")
        return result
    except Exception as e:
        logging.warning(f"⚠️ Error classifying criteria, using fallback: {e}")
        return {
            "inclusion": {"abstract_level": inc_lines, "full_text_level": []},
            "exclusion": {"abstract_level": exc_lines, "full_text_level": []}
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
    Etapa 2 del embudo: scoring multi-dimensional y predictivo sobre top-200.
    Implementa una arquitectura de screening de 3 capas con inferencia estructurada de criterios.
    """
    full_ranking = target_n <= 0 or target_n >= len(candidates)
    effective_target = len(candidates) if full_ranking else max(1, target_n)
    effective_pool = len(candidates) if candidate_pool <= 0 else min(candidate_pool, len(candidates))
    pool = candidates[:effective_pool]
    remainder = candidates[effective_pool:]
    if not pool:
        return candidates if full_ranking else candidates[:effective_target]

    api_key = config.CEREBRAS_API_KEYS[0] if (hasattr(config, "CEREBRAS_API_KEYS") and config.CEREBRAS_API_KEYS) else (config.GROQ_API_KEY if hasattr(config, "GROQ_API_KEY") else None)
    if not api_key:
        logging.warning("⚠️ Sin clave de API disponible para screening predictivo. Omitiendo etapa.")
        return candidates if full_ranking else candidates[:effective_target]

    # 1. Clasificación dinámica de criterios en abstract-level y full-text-level
    classified_criteria = classify_criteria_by_detectability(inclusion_criteria, exclusion_criteria)
    
    inc_abstract = classified_criteria["inclusion"]["abstract_level"]
    inc_full_text = classified_criteria["inclusion"]["full_text_level"]
    exc_abstract = classified_criteria["exclusion"]["abstract_level"]
    exc_full_text = classified_criteria["exclusion"]["full_text_level"]

    provider_label = "Cerebras 8B (Predictive Scorer)"
    logging.info(f"🤖 [MCS-3Layers] {provider_label} | {len(pool)} candidatos | Evaluación Predictiva")

    def score_article(art: Dict, idx: int) -> tuple:
        title    = art.get('title', '')[:300]
        abstract = art.get('abstract', '')[:3000]

        # Pre-filtro rapido: detectar reviews por texto sin llamar a la IA
        abstract_lower = abstract.lower()
        is_review_fast = any(sig in abstract_lower for sig in _REVIEW_SIGNALS)

        # Construir prompt guiado por capas
        prompt = (
            f'You are an expert academic screener for a PRISMA systematic literature review.\n'
            f'Research question: "{original_question}"\n\n'
            'We are in the ABSTRACT SCREENING phase. We have dynamically classified our eligibility criteria into:\n'
            '1) ABSTRACT-LEVEL DETECTABLE CRITERIA (Highly visible in abstracts)\n'
            '2) FULL-TEXT REQUIRED CRITERIA (Deep technical/pedagogical/methodological details likely not fully declared in a short abstract)\n\n'
        )
        
        if inc_abstract:
            prompt += "INCLUSION CRITERIA (Abstract-level - MUST BE PLAUSIBLY MET):\n" + "\n".join(f"- {c}" for c in inc_abstract) + "\n\n"
        if inc_full_text:
            prompt += "INCLUSION CRITERIA (Full-text level - DO NOT reject if absent or vague in abstract):\n" + "\n".join(f"- {c}" for c in inc_full_text) + "\n\n"
        if exc_abstract:
            prompt += "EXCLUSION CRITERIA (Abstract-level - Reject immediately if clearly and explicitly met):\n" + "\n".join(f"- {c}" for c in exc_abstract) + "\n\n"
        if exc_full_text:
            prompt += "EXCLUSION CRITERIA (Full-text level - DO NOT reject if abstract is vague; only reject if it explicitly confirms violation):\n" + "\n".join(f"- {c}" for c in exc_full_text) + "\n\n"

        prompt += (
            f'Article Title: "{title}"\n'
            f'Abstract: "{abstract}"\n\n'
            'EVALUATION RULES (CONSERVATIVE PRISMA PROTOCOL):\n'
            '0. MANDATORY 3-POINT PICO GATEKEEPER PRE-CHECK:\n'
            '   Before performing a detailed grade, you MUST check if the abstract violates any of the core PICO dimensions of our research question:\n'
            '   A) TARGET POPULATION check: Does the abstract explicitly target or plausibly include the research question\'s target population? If it explicitly studies an entirely different demographic, species, condition, or age range, reject immediately.\n'
            '   B) TARGET INTERVENTION check: Does the abstract target the correct intervention, exposure, technology, system, or method specified by the research question? If it is a completely unrelated field or intervention class, reject immediately.\n'
            '   C) CONTEXT/SETTING check: Does the abstract study the phenomenon in the context or setting required by the research question? If it is situated in a clearly incompatible setting, reject immediately.\n'
            '   If ANY of these 3 Gatekeeper checks are explicitly violated, you MUST classify the article as "category": 0 (Red - Reject) and set "requires_pdf": false.\n\n'
            '1. Evaluate if the abstract satisfies the ABSTRACT-LEVEL inclusion and exclusion criteria.\n'
            '   - If the abstract CLEARLY violates any abstract-level exclusion criterion, or CLEARLY lacks the abstract-level inclusion criteria, classify as "category": 0 (Red - Reject).\n'
            '2. Evaluate the FULL-TEXT criteria:\n'
            '   - If the abstract meets the abstract-level criteria but the abstract does NOT explicitly mention or is vague about the FULL-TEXT criteria, do NOT reject it! Set "category": 2 (Yellow - Doubt/Requires PDF) and set "requires_pdf": true.\n'
            '   - Under the "Benefit of the Doubt" rule, borderline, vague, or complex articles MUST be kept in Category 2 for full-text PDF review. Recall is more important than precision at this stage.\n'
            '3. Classify as "category": 1 (Green - Perfect) only if the abstract explicitly and clearly satisfies ALL criteria (both abstract-level and full-text level).\n\n'
            'Respond ONLY with a valid JSON object in this format (no markdown blocks, just raw JSON):\n'
            '{\n'
            '  "category": 1 | 2 | 0,\n'
            '  "reason": "brief general justification",\n'
            '  "criteria_breakdown": {\n'
            '    "Name of Criterion": {\n'
            '      "status": "met" | "unclear" | "violated",\n'
            '      "inference": "brief explanation of what was found or inferred from the abstract",\n'
            '      "confidence": 0.0 to 1.0\n'
            '    }\n'
            '  },\n'
            '  "requires_pdf": true | false\n'
            '}'
        )
        
        try:
            content = LocalModel.get_instance().generate(
                instruction=prompt,
                max_tokens=500,
                system_prompt="You are an expert academic screener. Respond ONLY with a valid raw JSON object matching the requested schema."
            ).strip()
            
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n", "", content)
                content = re.sub(r"\n```$", "", content)
            
            data = json.loads(content)
            cat = int(data.get("category", 0))
            reason = data.get("reason", "Evaluación completada")
            requires_pdf = bool(data.get("requires_pdf", False))
            criteria_breakdown = data.get("criteria_breakdown", {})
            
            # Map Category to scores
            if cat == 1: # Green
                i_score = 10.0
                s_score = 10.0
                e_score = 1.0
            elif cat == 2: # Yellow
                i_score = 7.5
                s_score = 7.5
                e_score = 1.0
                requires_pdf = True # Force PDF since it's category 2
            else: # Red
                i_score = 1.0
                s_score = 1.0
                e_score = 10.0

            # Override study_type si el pre-filtro detecto review
            if is_review_fast:
                s_score = min(s_score, 3.0)
                
            return i_score, s_score, e_score, reason, requires_pdf, criteria_breakdown
            
        except Exception as ex:
            logging.debug(f"[MCS] Error en '{title[:35]}': {ex}")
            s_fallback = 3.0 if is_review_fast else 5.0
            return 5.0, s_fallback, 3.0, f"Error en evaluación: {ex}", True, {}

    # Evaluar todos concurrentemente
    mcs_results: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(score_article, art, i): i for i, art in enumerate(pool)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                mcs_results[idx] = future.result()
            except Exception as e:
                mcs_results[idx] = (5.0, 5.0, 3.0, f"Error de evaluación: {e}", True, {})

    # Aplicar scores y separar calificados de relleno
    qualified, fill_pool = [], []
    for i, art in enumerate(pool):
        result = mcs_results.get(i, (5.0, 5.0, 3.0, "Sin evaluar", True, {}))
        
        # Unpack seguro soportando retrocompatibilidad de longitud
        if len(result) == 3:
            i_score, s_score, e_score = result
            reason = "Sin evaluar"
            requires_pdf = True
            criteria_breakdown = {}
        elif len(result) == 4:
            i_score, s_score, e_score, reason = result
            requires_pdf = True
            criteria_breakdown = {}
        elif len(result) == 5:
            i_score, s_score, e_score, reason, requires_pdf = result
            criteria_breakdown = {}
        else:
            i_score, s_score, e_score, reason, requires_pdf, criteria_breakdown = result

        embed_sim   = art.get('similarity', 0.5)  # score original de embeddings
        combined    = 0.25 * embed_sim + 0.45 * (i_score / 10.0) + 0.30 * (s_score / 10.0)

        art['ai_relevance_score'] = round(i_score, 1)
        art['ai_study_type']      = round(s_score, 1)
        art['ai_excl_match']      = round(e_score, 1)
        art['ai_reason']          = reason
        art['ai_evaluated']       = True
        art['requires_pdf_verification'] = requires_pdf
        art['criteria_breakdown'] = criteria_breakdown
        art['similarity']         = round(combined, 4)  # % display = score combinado

        # Hard exclusion: Descartar SOLO los Rojos (cat=0 => i_score=1.0)
        # Los Amarillos (cat=2 => i_score=7.5) pasarán la prueba para manual/PDF review
        hard_exclude = (i_score < 2.0)
        
        (fill_pool if hard_exclude else qualified).append(art)

    qualified.sort(key=lambda x: x.get('similarity', 0), reverse=True)
    fill_pool.sort(key=lambda x: x.get('similarity', 0), reverse=True)

    # Garantizar target_n: calificados primero, relleno si son pocos
    result = qualified[:effective_target]
    if len(result) < effective_target:
        needed = effective_target - len(result)
        result.extend(fill_pool[:needed])
        if len(result) < effective_target and remainder:
            result.extend(remainder[:effective_target - len(result)])
        logging.info(f"   📦 Relleno: {min(needed, len(fill_pool) + len(remainder))} arts adicionales para completar {effective_target}")

    avg_combined = np.mean([a.get('similarity', 0) for a in result]) if result else 0
    logging.info(
        f"✅ [MCS-3Layers] Evaluados: {len(pool)} | Calificados: {len(qualified)} | "
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
      Sistema         → conserva el ranking completo para revisión manual y WSS@95
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

    def score_article(art: Dict, idx: int) -> float:
        title    = art.get('title', '')[:300]
        abstract = art.get('abstract', '')[:3000]
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
            content = LocalModel.get_instance().generate(
                instruction=prompt,
                max_tokens=20,
                system_prompt="You are an expert systematic literature review expert. Respond ONLY with a valid raw JSON object matching the requested schema."
            ).strip()
            
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n", "", content)
                content = re.sub(r"\n```$", "", content)
                
            match = re.search(r'\{\s*"score"\s*:\s*(\d+)\s*\}', content)
            if match:
                return float(match.group(1))
            return float(json.loads(content).get("score", 5))
            
        except Exception as e:
            logging.debug(f"[AI Rerank] Error in '{title[:35]}': {e}")
            return 5.0

    # Evaluar con batches + rate limiting para Groq (30 req/min free tier)
    ai_raw_scores: dict = {}
    pool_items  = list(enumerate(pool))
    batch_size  = max_workers

    for batch_start in range(0, len(pool_items), batch_size):
        batch = pool_items[batch_start: batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(score_article, art, idx): idx for idx, art in batch}
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

    # Identificar excluidos (los que no están en el resultado final)
    # Incluimos los excluidos por metadatos (excluded_metadata) y los que no pasaron el corte de la IA
    result_ids = {id(a) for a in result}
    excluded_ai = [a for a in pool if id(a) not in result_ids]
    
    # Marcar razón de exclusión para los de la IA si no la tienen
    for art in excluded_ai:
        if not art.get('exclusion_reason'):
            score = art.get('ai_relevance_score', 0)
            art['exclusion_reason'] = f"Baja relevancia (Score IA: {score}/10)"
            art['_exclusion_reason'] = art['exclusion_reason'] # Compatibilidad con reporte PDF

    excluded_all = excluded_metadata + excluded_ai
    
    return result, excluded_all
