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
from utils.query_expander import expand_query_with_llm, extract_english_terms, get_exclusion_terms_with_llm
from modules.core.search_engine import detect_search_domain

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
    Filtra articulos usando SPECTER2 + filtro de relevancia de dominio.

    ESTRATEGIA:
    1. Calcula similitud semantica (SPECTER2)
    2. Calcula relevancia de dominio (keywords de la pregunta)
    3. Score ajustado = similitud - penalizacion_dominio
    4. Filtra con umbral minimo
    5. Prioriza articulos CON URL/PDF
    """
    if not articles:
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

        # ── Enriquecimiento vectorial con criterios I/E del investigador (Opción A) ──
        # Los criterios de inclusión se concatenan a la query para orientar el embedding.
        # Los criterios de exclusión se agregan a la lista de términos de exclusión existentes.
        enriched_query = original_q
        manual_exclusion_terms = []
        if inclusion_criteria and inclusion_criteria.strip():
            # Cada línea es un criterio; se concatenan como contexto adicional
            inc_lines = [l.strip() for l in inclusion_criteria.strip().splitlines() if l.strip()]
            if inc_lines:
                enriched_query = original_q + ". " + " AND ".join(inc_lines)
                logging.info(f"✅ [Criterios I] Enriqueciendo query con {len(inc_lines)} criterios de inclusión")
        if exclusion_criteria and exclusion_criteria.strip():
            exc_lines = [l.strip() for l in exclusion_criteria.strip().splitlines() if l.strip()]
            manual_exclusion_terms = [line.lower() for line in exc_lines if line]
            if manual_exclusion_terms:
                logging.info(f"🚫 [Criterios E] {len(manual_exclusion_terms)} criterios de exclusión manuales aplicados")

        semantic_queries = expand_query_with_llm(enriched_query)
        english_terms = extract_english_terms(enriched_query)
        
        exclusion_terms = get_exclusion_terms_with_llm(original_q)
        # Combinar exclusiones: LLM + manuales del investigador
        exclusion_terms = list(set(exclusion_terms + manual_exclusion_terms))
        comparison_poles = _extract_comparison_poles(original_q)
        
        if comparison_poles:
            poles_str = " vs ".join([p[0] for p in comparison_poles])
            logging.info(f"⚖️ Polos de comparación detectados: {poles_str}")

        logging.info(f"🌎 Screening con {len(semantic_queries)} queries y {len(exclusion_terms)} filtros de exclusión...")

        # 2. Scoring Multi-Query con numpy (sin PyTorch)
        # Encode todos los abstracts una vez
        corpus_embeddings = get_embeddings(texts)  # shape: (n_articles, 768)
        
        # Calcular scores por cada query
        all_query_scores = []
        for q in semantic_queries:
            q_emb = get_single_embedding(q)  # shape: (768,)
            q_scores = cosine_similarity([q_emb], corpus_embeddings)[0]  # (n_articles,)
            all_query_scores.append(q_scores)
            
        all_query_scores = np.array(all_query_scores) # (n_queries, n_articles)
        
        # Combinación Probabilística: Max (70%) + Mean (30%)
        max_scores = np.max(all_query_scores, axis=0)
        mean_scores = np.mean(all_query_scores, axis=0)
        final_raw_scores = (max_scores * 0.7) + (mean_scores * 0.3)

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
    # ESTRATEGIA DE SELECCION V5 - FUZZY-FIRST RANKING
    # ============================================================
    # El hybrid_score (60% fuzzy + 40% normalizado) es el criterio principal.
    # El RAW_SCORE_FLOOR (0.20) solo excluye articulos absolutamente off-topic.
    # Esto garantiza siempre llegar a max_results=100.

    # 1. Excluir solo los absolutamente irrelevantes (raw < 0.20)
    eligible = [a for a in articles if a.get('raw_similarity', 0) >= RAW_SCORE_FLOOR]
    
    # 2. Ordenar por hybrid_score (ranking fuzzy-first)
    eligible.sort(key=lambda x: x.get('similarity', 0), reverse=True)

    raw_above_floor = len(eligible)
    logging.info(f"🔍 Candidatos (raw≥{RAW_SCORE_FLOOR}): {len(articles)} → {raw_above_floor} | Seleccionando Top-{max_results} por hybrid_score")

    # 3. Priorizar artículos con URL válida dentro del top-N
    with_url    = [a for a in eligible if a.get('url') and len(str(a.get('url'))) > 10]
    without_url = [a for a in eligible if not (a.get('url') and len(str(a.get('url'))) > 10)]

    # Tomar top-N combinando con_url primero, luego sin_url para completar
    final_selection = with_url[:max_results]
    if len(final_selection) < max_results:
        needed = max_results - len(final_selection)
        final_selection.extend(without_url[:needed])

    # Mantener orden por hybrid_score tras combinar
    final_selection.sort(key=lambda x: x.get('similarity', 0), reverse=True)

    # ============================================================
    # AI ABSTRACT RE-RANKER (v1.0)
    # Evalua abstracts en zona ambigua con IA para eliminar falsos positivos
    # ============================================================
    if original_question and config.CEREBRAS_API_KEYS:
        final_selection = ai_rerank_abstracts(
            candidates=final_selection,
            original_question=original_question,
            target_n=max_results,
            candidate_pool=min(len(final_selection), 150),
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
# AI ABSTRACT RE-RANKER (v2.0 - TOTAL EVALUATION + HARD CUTOFF)
# ============================================================

def ai_rerank_abstracts(
    candidates: List[Dict],
    original_question: str,
    target_n: int = 100,
    candidate_pool: int = 150,
    min_ai_score: float = 5.0,   # Corte duro: articulos con AI < 5/10 son EXCLUIDOS
    max_workers: int = 8,
) -> List[Dict]:
    """
    Re-rankea candidatos usando IA como arbitro principal de relevancia.

    v2.0 vs v1.0:
    - TODOS los articulos son evaluados (antes solo la zona ambigua)
    - Corte duro: articulos con AI score < min_ai_score son excluidos del pool
    - Pesos AI-first: 65% AI + 35% fuzzy (antes 50/50)
    - Prompt mejorado: penaliza explicitamente dominios cruzados

    Score final = 0.65 * (ai_score/10) + 0.35 * fuzzy_hybrid
    """
    pool = candidates[:candidate_pool]
    if not pool:
        return candidates[:target_n]

    api_key = config.CEREBRAS_API_KEYS[0] if config.CEREBRAS_API_KEYS else None
    if not api_key:
        logging.warning("⚠️ [AI Rerank] Sin clave Cerebras disponible, omitiendo re-ranking")
        return candidates[:target_n]

    logging.info(f"🤖 [AI Rerank v2] Evaluando TODOS los {len(pool)} candidatos con IA...")

    def score_article(art: Dict) -> float:
        """Puntua 1-10 segun relevancia directa. Penaliza dominios cruzados."""
        title    = art.get('title', '')[:150]
        abstract = art.get('abstract', '')[:500]

        prompt = (
            f'Research question: "{original_question}"\n\n'
            f'Title: "{title}"\n'
            f'Abstract: "{abstract}"\n\n'
            'Does this article SPECIFICALLY and DIRECTLY address the research question?\n'
            'Score 1-10:\n'
            '- 8-10: Directly studies this exact topic with empirical/experimental results\n'
            '- 5-7: Clearly relevant, main topic overlaps significantly\n'
            '- 1-4: Off-topic, wrong domain, or only tangentially related\n\n'
            'CRITICAL RULES:\n'
            '- If the article is about medicine, biology, agriculture, education, physics, '
            'automotive, or any field OTHER than software/computer science: score 1-2\n'
            '- If the article uses AI but applies it to a DIFFERENT DOMAIN than the research '
            'question: score 1-3\n'
            '- Only score >= 5 if the article is DIRECTLY about the research question topic\n\n'
            'Respond ONLY with valid JSON: {"score": N}'
        )
        try:
            resp = requests.post(
                config.CEREBRAS_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": config.CEREBRAS_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 20,
                    "temperature": 0.0
                },
                timeout=15
            )
            content = resp.json()["choices"][0]["message"]["content"].strip()
            match = re.search(r'\{\s*"score"\s*:\s*(\d+)\s*\}', content)
            if match:
                return float(match.group(1))
            data = json.loads(content)
            return float(data.get("score", 5))
        except Exception as e:
            logging.debug(f"🔹 [AI Rerank] Error: '{title[:35]}': {e}")
            return 5.0  # Score neutro si falla (no excluir por error de red)

    # Evaluar TODOS los articulos con workers concurrentes
    ai_raw_scores: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(score_article, art): i for i, art in enumerate(pool)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                ai_raw_scores[idx] = future.result()
            except Exception:
                ai_raw_scores[idx] = 5.0

    # Aplicar scores y corte duro
    qualified = []
    excluded  = 0
    for i, art in enumerate(pool):
        ai_score = ai_raw_scores.get(i, 5.0)
        art['ai_relevance_score'] = round(ai_score, 1)
        art['ai_evaluated']       = True

        if ai_score < min_ai_score:
            excluded += 1
            continue  # EXCLUIR del pool

        fuzzy_hybrid = art.get('similarity', 0.5)
        # AI-first: 65% IA + 35% fuzzy
        art['similarity'] = round((ai_score / 10.0) * 0.65 + fuzzy_hybrid * 0.35, 4)
        qualified.append(art)

    qualified.sort(key=lambda x: x.get('similarity', 0), reverse=True)

    ai_avg = np.mean(list(ai_raw_scores.values())) if ai_raw_scores else 0
    logging.info(
        f"✅ [AI Rerank v2] Completado | Evaluados: {len(pool)} | "
        f"Excluidos (AI<{min_ai_score}): {excluded} | "
        f"Calificados: {len(qualified)} | Score IA promedio: {ai_avg:.1f}/10"
    )

    return qualified[:target_n]

