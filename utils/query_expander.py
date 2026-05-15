"""
Query Expander - VERSIÓN MULTI-PROVIDER + AGNÓSTICO AL DOMINIO
Genera ecuaciones booleanas EN INGLÉS basadas en los conceptos de la pregunta.
Si el LLM falla, extrae términos técnicos y genera queries en inglés.
"""
import requests
import config
import logging
import json
import re
import os
import hashlib
from typing import List, Optional, Dict
from modules.ai.ai_model import generate_text

# Cache
CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(question: str) -> str:
    # v6: Queries semánticas cortas (no booleanas pesadas)
    key = f"semantic_v6_{question.strip().lower()}"
    return hashlib.md5(key.encode()).hexdigest()

def load_from_cache(question: str) -> Optional[List[str]]:
    cache_key = get_cache_key(question)
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)["terms"]
        except: pass
    return None

def save_to_cache(question: str, terms: List[str]):
    try:
        with open(os.path.join(CACHE_DIR, f"{get_cache_key(question)}.json"), 'w', encoding='utf-8') as f:
            json.dump({"question": question, "terms": terms}, f)
    except: pass


# ============================================================
# PROMPT AGNÓSTICO (compartido entre providers)
# ============================================================

def _build_prompt(question: str) -> str:
    return f"""Act as an expert systematic review librarian. Create 4-6 SHORT SEMANTIC SEARCH QUERIES IN ENGLISH for finding highly relevant academic papers.

Research Question: "{question}"

STRATEGY:
- Query 1-2: Very specific (3-5 key terms, directly matching the research question)
- Query 3-4: Moderately specific (combine main concepts with alternative terms)
- Query 5-6: Broader but still focused (wider synonyms, but keep the core topic)

CRITICAL RULES:
1. Output ONLY valid JSON: {{"queries": ["query1", "query2", ...]}}
2. ALL queries MUST be in ENGLISH
3. DO NOT use boolean operators (AND, OR, NOT) — just write natural keyword phrases
4. Each query should be 3-7 words of specific technical terms
5. Focus on the EXACT topic, not broad categories
6. Include specific tools, methods, or technologies mentioned in the question
7. Include acronyms where appropriate (LLM, SAST, CNN, etc.)

EXAMPLES:
- CS/Security question → ["LLM vulnerability detection source code", "large language model static analysis security", "GPT code vulnerability SAST comparison", "AI code review false positive reduction"]
- Health/AI question → ["deep learning cardiovascular diagnosis", "CNN heart disease detection ECG", "machine learning cardiac risk prediction"]

JSON Output:"""

def _build_api_prompt(question: str) -> str:
    return f"""You are a systematic review librarian specialized in academic search engineering.

Research Question: "{question}"

STRATEGY: Generate 6 ULTRA-SHORT, BROAD semantic search queries (2-4 words maximum) to cast a massive dragnet over academic literature databases (PubMed, IEEE, OpenAlex).
Do NOT combine all concepts into one long query. Break the research question into independent pairs of broad keywords.

EXAMPLES of good dragnet queries (from various random domains):
- Domain A: "blockchain supply chain"
- Domain A: "distributed ledger logistics"
- Domain B: "myocardial infarction mortality"
- Domain B: "heart attack survival"
- Domain C: "quantum computing cryptography"

RULES:
- FOCUS ONLY ON THE 2 STRONGEST PILLARS of the research question (e.g. "target population" + "core technology").
- IGNORE highly specific modifiers, edge constraints, or comparative elements (e.g. "ecological environments", "fixed timing", "traditional systems") which will be filtered locally later.
- Each query MUST BE MAXIMUM 3 or 4 WORDS. Do not write long sentences.
- Output ONLY valid JSON: {{"queries": ["query1", "query2", ...]}}
- ALL queries MUST be in ENGLISH, even if the research question is in another language.
- NO boolean operators (AND/OR), no quotes, plain text only.
- USE natural phrasing or adjacent technical keywords.

JSON Output:"""


def _build_system_msg() -> str:
    return "You are a search query generator for systematic reviews. ALL queries MUST be in ENGLISH. Extract concepts from the user's question, translate if needed, and create boolean queries. Output valid JSON only."


# ============================================================
# MULTI-PROVIDER QUERY GENERATION
# ============================================================

def expand_query_with_llm(question: str) -> List[str]:
    """Queries largas para screening/embedding."""
    cached = load_from_cache(question)
    if cached: return cached

    prompt = _build_prompt(question)
    system_msg = _build_system_msg()

    logging.info("🤖 [Query Gen] Solicitando queries al LLM Router...")
    queries = _call_llm_json(prompt, system_msg, return_dict=False)
    if queries:
        save_to_cache(question, queries)
        return queries

    return generate_fallback_queries(question)

def get_exclusion_terms_with_llm(question: str) -> List[str]:
    """Identifica temas que suelen causar ruido para esta RQ específica (v7.8)."""
    prompt = f"""Identify 6-8 SHORT terms or topics (1-2 words max) that are SEMANTIC DISTRACTORS for this research question.
A distractor is a topic that shares SOME keywords but has a completely DIFFERENT context, domain, or goal.

Research Question: "{question}"

For example, if the research is about X applied to Y, what other completely unrelated fields (Z) also use X? Output those unrelated fields.
DO NOT output terms that are relevant to the research question.

Output ONLY valid JSON: {{"exclusions": ["term1", "term2", ... ]}}
Terms MUST be 1-2 words only for strict keyword matching."""
    
    system_msg = "You are a research filter specialist. Identify out-of-context noise terms."
    
    logging.info("🤖 [Exclusions] Solicitando términos distractores al LLM Router...")
    res = _call_llm_json(prompt, system_msg, return_dict=False)
    if res: return res
        
    return ["clinical", "patient", "medical"] # Fallback mínimo universal

def generate_api_queries_with_llm(question: str) -> List[str]:
    """Queries CORTAS de alta intersección para APIs (Recall)."""
    # Cache distinguible por prefijo en get_cache_key (v7 for multi-perspective)
    # v7.8: Fresh cache after physical cleanup
    cache_key = hashlib.md5(f"api_v7_8_{question.strip().lower()}".encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)["queries"]
        except: pass

    prompt = _build_api_prompt(question)
    system_msg = _build_system_msg()

    logging.info("🤖 [API Queries] Solicitando queries al LLM Router...")
    queries = _call_llm_json(prompt, system_msg, return_dict=False)
    if queries:
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({"question": question, "queries": queries}, f)
        except: pass
        return queries

    return generate_fallback_queries(question)


def _call_llm_json(prompt: str, system_msg: str, return_dict: bool = False):
    """Llama al LLM usando el router multi-provider y parsea el JSON."""
    # Añadimos instrucción para que devuelva JSON
    instruction = prompt + "\n\nCRITICAL: You MUST output ONLY valid JSON format. Do not use markdown blocks like ```json."
    
    response = generate_text(instruction=instruction, input_text="", max_tokens=1024, system_prompt=system_msg)
    if "Error de generación" in response or "⚠️" in response:
        logging.warning(f"⚠️ [Query Gen] Error del LLM Router: {response}")
        return None
        
    # Limpiar posibles bloques markdown
    clean_resp = response.replace("```json", "").replace("```", "").strip()
    
    try:
        data = json.loads(clean_resp)
        if return_dict:
            return data
        
        # Para functions que esperan List[str] (queries o exclusions)
        queries = data.get("queries", data.get("exclusions", []))
        clean = [q for q in queries if q and len(q) > 2]
        if clean:
            return clean
            
    except json.JSONDecodeError:
        # Intentar extraer JSON con regex
        match = re.search(r'\{.*\}', clean_resp, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if return_dict:
                    return data
                queries = data.get("queries", data.get("exclusions", []))
                return [q for q in queries if q and len(q) > 2]
            except:
                pass
                
    return None


# ============================================================
# FALLBACK INTELIGENTE (sin LLM)
# ============================================================

def generate_fallback_queries(question: str) -> List[str]:
    """
    Genera queries EN INGLÉS desde los términos de la pregunta.
    Prefiere acrónimos y términos técnicos en inglés.
    """
    logging.info("🔧 Generando queries desde términos técnicos (Fallback sin LLM)...")
    
    terms = extract_english_terms(question)
    
    if len(terms) < 2:
        # Última línea: limpiar pregunta de stopwords y usar directamente
        clean = _clean_question_for_search(question)
        return [clean] if clean else ["systematic review"]
    
    # Query 1: Todos los conceptos principales (específico)
    q1 = " AND ".join(f'"{t}"' if ' ' in t else f'"{t}"' for t in terms[:3])
    
    # Query 2: Solo 2 conceptos principales (más amplio)
    q2 = " AND ".join(f'"{t}"' if ' ' in t else f'"{t}"' for t in terms[:2])
    
    # Query 3: Primer y tercer concepto (cobertura diferente)
    if len(terms) >= 3:
        q3 = f'"{terms[0]}" AND "{terms[2]}"'
        queries = [q1, q2, q3]
    else:
        queries = [q1, q2]
    
    for i, q in enumerate(queries):
        logging.info(f"   📋 Fallback Query {i+1}: {q[:100]}...")
    
    return queries


def extract_english_terms(question: str) -> List[str]:
    """
    Extrae términos técnicos EN INGLÉS de la pregunta.
    Prioriza: acrónimos > términos en paréntesis explicativos > sustantivos técnicos.
    """
    terms = []
    seen = set()
    
    def _add(t: str):
        key = t.lower().strip()
        if key and key not in seen and len(key) > 1:
            seen.add(key)
            terms.append(t.strip())
    
    # 1. Acrónimos (SIEMPRE son en inglés): LLMs, SAST, CNN, IoT, NLP
    acronyms = re.findall(r'\b([A-Z][A-Za-z]*[A-Z]+[a-z]*)\b', question)  # LLMs, SAST
    acronyms += re.findall(r'\b([A-Z]{2,6}s?)\b', question)  # LLM, SAST, CNNs
    skip = {'AND', 'OR', 'NOT', 'THE', 'FOR', 'DE', 'LA', 'LOS', 'EN', 'DEL', 'UNA'}
    for a in acronyms:
        clean_a = a.rstrip('s')  # LLMs -> LLM
        if clean_a.upper() not in skip:
            _add(a)
    
    # 2. Texto entre paréntesis que parece inglés/técnico
    parens = re.findall(r'\(([^)]+)\)', question)
    for p in parens:
        p = p.strip()
        # Si es un acrónimo solo, ya lo tenemos
        if re.match(r'^[A-Z]{2,6}s?$', p):
            continue
        # Si contiene texto en inglés (letters only, no Spanish articles)
        if re.search(r'[a-z]{3,}', p) and not re.search(r'\b(los|las|del|una|para|con)\b', p.lower()):
            _add(p)
    
    # 3. Términos entre comillas
    quoted = re.findall(r'"([^"]+)"', question)
    for q in quoted:
        _add(q)
    
    # 4. Palabras en inglés que aparecen directamente en la pregunta
    english_words_in_text = re.findall(r'\b([a-z]{2,}(?:\s+[a-z]{2,}){0,2})\b', question)
    for w in english_words_in_text:
        if w.lower() not in seen and _is_english_technical(w):
            _add(w)
    
    logging.info(f"   🔑 Términos técnicos fallback (EN): {terms[:8]}")
    return terms[:8]

def extract_english_terms_with_llm(question: str) -> List[str]:
    """
    Extrae dinámicamente los términos técnicos y los traduce al inglés usando el LLM.
    Mantiene la aplicación agnóstica al dominio sin diccionarios hardcodeados.
    """
    cache_key = hashlib.md5(f"eng_terms_v1_{question.strip().lower()}".encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            return cached.get("terms", [])
        except: pass

    prompt = f"""Identify the core technical concepts and keywords from the following research question and translate them into English.
DO NOT include generic stopwords or common verbs.
Research Question: "{question}"

Output ONLY a JSON object with this format:
{{"terms": ["english term 1", "english term 2", "english term 3"]}}"""

    system_msg = "You are a technical translator for academic search. Output valid JSON only."
    
    logging.info("🤖 [English Terms] Solicitando términos técnicos en inglés al LLM Router...")
    terms = _call_llm_json(prompt, system_msg, return_dict=False)
    
    if terms:
        # Limpiar y asegurar que sean válidos
        clean_terms = [t for t in terms if isinstance(t, str) and len(t) > 2]
        if clean_terms:
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump({"question": question, "terms": clean_terms}, f)
            except: pass
            logging.info(f"   🔑 Términos técnicos extraídos por LLM (EN): {clean_terms}")
            return clean_terms

    # Fallback si el LLM falla
    return extract_english_terms(question)


def _is_english_technical(word: str) -> bool:
    """Detecta si una palabra/frase es un término técnico en inglés."""
    tech_terms = {
        'machine learning', 'deep learning', 'artificial intelligence',
        'neural network', 'natural language processing', 'computer vision',
        'static analysis', 'dynamic analysis', 'code review',
        'vulnerability', 'security', 'false positive', 'false negative',
        'source code', 'software', 'framework', 'algorithm',
        'classification', 'detection', 'prediction', 'regression',
        'transformer', 'attention', 'embedding', 'fine-tuning',
        'prompt engineering', 'code generation', 'benchmark',
        'systematic review', 'meta-analysis', 'screening',
    }
    return word.lower() in tech_terms


def _clean_question_for_search(question: str) -> str:
    """Limpia la pregunta para usarla como query de búsqueda."""
    # Remover signos de puntuación españoles
    clean = re.sub(r'[¿?¡!,;:.]', ' ', question)
    # Remover stopwords comunes en español
    stopwords = {
        'cuál', 'cual', 'cómo', 'como', 'qué', 'que', 'es', 'son', 'las', 'los',
        'del', 'de', 'la', 'el', 'en', 'un', 'una', 'para', 'por', 'con', 'sin',
        'sobre', 'entre', 'más', 'mas', 'se', 'al', 'a', 'e', 'i', 'o', 'u', 'y',
        'su', 'sus', 'mi', 'tu', 'frente', 'durante', 'mediante', 'través',
    }
    words = [w for w in clean.split() if w.lower() not in stopwords and len(w) > 1]
    return " ".join(words[:10])


# ============================================================
# EXPANSIÓN SEMÁNTICA CON SINÓNIMOS (equivalente a WordNet + Corpus)
# ============================================================

def expand_query_with_synonyms(
    question: str,
    corpus_sample: Optional[List[str]] = None,
) -> Dict:
    """
    Genera sinónimos, acrónimos y jerga científica equivalente para los
    términos clave de la RQ. Equivalente al WordNet + Corpus Semántico
    recomendado por el profesor.

    Estrategia:
      1. Extrae los conceptos clave de la RQ
      2. Pide al LLM generar variantes científicas en inglés
      3. Opcionalmente, adapta al vocabulario del corpus recibido
      4. Retorna términos enriquecidos para inyectar en BM25

    Args:
        question:      Pregunta de investigación (cualquier idioma)
        corpus_sample: Muestra de abstracts del corpus (opcional, para
                       adaptar el vocabulario al dominio específico)

    Returns:
        Dict con:
          - "synonyms":  términos sinónimos por concepto clave
          - "flat_terms": lista plana de todos los términos para BM25
          - "expanded_queries": queries enriquecidas con sinónimos
    """
    # Cache con prefijo distinto para no mezclar con expand_query_with_llm
    cache_key_raw = f"synonyms_v2_{question.strip().lower()}"
    cache_key = hashlib.md5(cache_key_raw.encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"syn_{cache_key}.json")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            logging.info(f"🔑 [Synonyms] Cargado de caché: {len(cached.get('flat_terms', []))} términos")
            return cached
        except Exception:
            pass

    # Construir corpus hint si se proporcionó muestra
    corpus_hint = ""
    if corpus_sample:
        # Tomar una muestra representativa (max 5 abstracts, 200 chars c/u)
        samples = [s[:200] for s in corpus_sample[:5] if s]
        if samples:
            corpus_hint = (
                "\n\nCORPUS SAMPLE (use these to adapt terminology to the domain):\n"
                + "\n---\n".join(samples)
            )

    system_msg = (
        "You are a systematic review librarian and domain expert. "
        "Generate a comprehensive synonym expansion for academic search. "
        "Output ONLY valid JSON, no explanations."
    )

    user_prompt = f"""Research Question: "{question}"{corpus_hint}

EXTRACT ONLY THE 2 TO 4 MOST FUNDAMENTAL CORE PILLARS of the research question (e.g., target population, main technology, primary outcome).
IGNORE highly specific edge constraints, ecological variables, or secondary contexts.

For each of these CORE PILLARS, generate:
1. The main concept in English
2. All scientific synonyms, acronyms, and equivalent terms used in academic papers
3. Specific sub-types or variants relevant to this domain

RULES:
- Extract up to 4 concepts. Do not extract every detail.
- ALL terms must be in English
- Include domain-specific acronyms (e.g., SAST, LLM, NLP, EHR)
- Include both formal terms and common abbreviations
- Focus on terms that would appear in academic paper titles/abstracts
- Each term should be 1-4 words maximum

Output ONLY this JSON format:
{{
  "concepts": [
    {{
      "main": "main concept name",
      "synonyms": ["synonym1", "synonym2", "acronym1", "variant1"]
    }}
  ],
  "cross_terms": ["term that combines 2+ concepts", "compound term2"]
}}"""

    logging.info("🤖 [Synonyms] Solicitando sinónimos al LLM Router...")
    result = _call_llm_json(user_prompt, system_msg, return_dict=True)

    # Procesar resultado del LLM
    if result and isinstance(result, dict) and "concepts" in result:
        concepts = result.get("concepts", [])
        cross_terms = result.get("cross_terms", [])

        # Construir lista plana de todos los términos
        flat_terms = []
        synonyms_by_concept = {}
        for concept in concepts:
            main = concept.get("main", "")
            syns = concept.get("synonyms", [])
            if main:
                flat_terms.append(main)
                synonyms_by_concept[main] = syns
            flat_terms.extend(syns)

        flat_terms.extend(cross_terms)

        # Deduplicar y limpiar
        seen = set()
        flat_clean = []
        for t in flat_terms:
            t_clean = str(t).strip().lower()
            if t_clean and t_clean not in seen and len(t_clean) >= 2:
                seen.add(t_clean)
                flat_clean.append(str(t).strip())

        # Construir queries enriquecidas (combinando concepto principal + sinónimos)
        expanded_queries = []
        for concept in concepts:
            main = concept.get("main", "")
            syns = concept.get("synonyms", [])[:3]  # Top 3 sinónimos
            if main and syns:
                # Query: "main synonym1 synonym2"
                expanded_queries.append(f"{main} {' '.join(syns)}")
        expanded_queries.extend(cross_terms[:3])

        output = {
            "synonyms": synonyms_by_concept,
            "flat_terms": flat_clean,
            "expanded_queries": expanded_queries,
        }

        # Guardar en caché
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False)
        except Exception:
            pass

        logging.info(
            f"🌐 [Synonyms] Expansión completada: {len(flat_clean)} términos únicos | "
            f"{len(synonyms_by_concept)} conceptos | {len(expanded_queries)} queries enriquecidas"
        )
        return output

    # ── Fallback: construir sinónimos estructurados básicos ──
    # Cuando el LLM falla, construimos un dict mínimo de sinónimos
    # agrupado por concepto para que concept_presence_filter pueda funcionar.
    # Se filtran acrónimos cortos para no envenenar BM25.
    logging.warning("⚠️ [Synonyms] LLM falló, usando fallback de términos técnicos")
    fallback_terms_raw = extract_english_terms(question)

    # Filtrar acrónimos problemáticos del fallback
    _BAD_ACRONYMS = {'ia', 'st', 'ai', 'qa', 'ml', 'dl', 'se', 'it', 'is', 'rq', 'nn', 'cv'}
    fallback_terms = [
        t for t in fallback_terms_raw
        if len(t.strip()) >= 3
        and t.strip().lower() not in _BAD_ACRONYMS
        and not (t.strip().isupper() and len(t.strip()) <= 3)
    ]
    logging.info(f"   🔑 Términos técnicos (EN, filtrados): {fallback_terms}")

    # Construir synomyms agrupados: cada término de 3+ palabras es un "concepto"
    # Los términos cortos son "sinónimos" del primer concepto largo que encontremos
    synonyms_structured: Dict[str, List[str]] = {}
    long_terms   = [t for t in fallback_terms if len(t.split()) >= 2]
    short_terms  = [t for t in fallback_terms if len(t.split()) == 1]

    if long_terms:
        for lt in long_terms:
            synonyms_structured[lt] = []
        # Distribuir los términos cortos como sinónimos del primer concepto largo
        if short_terms and long_terms:
            synonyms_structured[long_terms[0]].extend(short_terms)
    elif short_terms:
        # Solo hay términos cortos: usar el más largo como concepto principal
        primary = max(short_terms, key=len)
        synonyms_structured[primary] = [t for t in short_terms if t != primary]

    return {
        "synonyms": synonyms_structured,   # ← Ahora tiene estructura, no vacío
        "flat_terms": fallback_terms,
        "expanded_queries": fallback_terms[:3],
    }


def expand_query(question: str, max_terms: int = 12) -> List[str]:
    """Función principal: Multi-provider LLM con fallback inteligente en inglés."""
    return expand_query_with_llm(question)