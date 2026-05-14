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

CRITICAL FIRST STEP — Determine the RESEARCH DIRECTION:
A) "AI/ML APPLIED TO Software Testing" (using AI to automate, improve, or assist testing)
B) "Testing/Validation OF AI/ML Systems" (how to test or verify AI models themselves)
C) Both directions are equally central to the research question

Select the direction(s) that match the research question, then generate queries ONLY for that direction.

STRATEGY: Generate 6 SHORT search queries (5-7 words each):
1. THE METHOD/TOOL: Specific AI technique used (e.g., "LLM-based test generation").
2. THE APPLICATION: The specific testing problem WITHOUT the AI method (e.g., "regression test selection automation").
3. THE OUTCOME: The measured result or metric (e.g., "test coverage improvement AI").
4. ALTERNATIVES: Synonyms or adjacent concepts that researchers use in papers.

RULES:
- Each query MUST combine at least 2 concepts.
- Output ONLY valid JSON: {{"queries": ["query1", "query2", ...]}}
- ALL queries MUST be in ENGLISH.
- NO boolean operators, no quotes, plain text only.
- USE directional connectors: "for", "using", "applied to", "to automate"
  Direction A example: "machine learning for software test case generation"
  Direction B example: "testing machine learning model robustness validation"

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

    # Fallback chain
    for try_func in [_try_github_models, _try_groq, _try_huggingface]:
        queries = try_func(prompt, system_msg)
        if queries:
            save_to_cache(question, queries)
            return queries

    return generate_fallback_queries(question)

def get_exclusion_terms_with_llm(question: str) -> List[str]:
    """Identifica temas que suelen causar ruido para esta RQ específica (v7.8)."""
    prompt = f"""Identify 6-8 SHORT terms or topics (1-2 words max) that are SEMANTIC DISTRACTORS for this research question.
A distractor is a topic that shares keywords (LLM, Security, etc.) but has a completely DIFFERENT GOAL.

Research Question: "{question}"

Examples of GOOD (Short) noise terms:
- "watermarking" (if the goal is code audit)
- "jailbreak" (if the goal is vulnerability detection)
- "healthcare" (if the goal is software engineering)
- "plagiarism" (if the goal is code generation)

Output ONLY valid JSON: {{"exclusions": ["term1", "term2", ... ]}}
Terms MUST be 1-2 words only for strict keyword matching."""
    
    system_msg = "You are a research filter specialist. Identify out-of-context noise terms."
    
    # Intento rápido con Groq o GitHub (fallback liviano)
    for try_func in [_try_groq, _try_github_models]:
        res = try_func(prompt, system_msg)
        if res: return res # Reutilizamos _parse_llm_response que ahora busca "exclusions"
        
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

    # Fallback chain
    for try_func in [_try_github_models, _try_groq, _try_huggingface]:
        queries = try_func(prompt, system_msg)
        if queries:
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump({"question": question, "queries": queries}, f)
            except: pass
            return queries

    return generate_fallback_queries(question)


def _parse_llm_response(content: str) -> List[str]:
    """Parsea la respuesta JSON del LLM y extrae queries."""
    try:
        data = json.loads(content)
        # Buscar en queries o exclusions
        queries = data.get("queries", data.get("exclusions", []))
        clean = [q for q in queries if q and len(q) > 2]
        if clean:
            return clean
    except json.JSONDecodeError:
        # Intentar extraer JSON de la respuesta
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                queries = data.get("queries", [])
                return [q for q in queries if q and len(q) > 10]
            except:
                pass
    return []


def _try_github_models(prompt: str, system_msg: str) -> List[str]:
    """Intenta generar queries con GitHub Models (Grok-3)."""
    try:
        if not getattr(config, 'GITHUB_MODELS_TOKEN', None):
            return []
        
        logging.info("🤖 [Query Gen] Intentando con GitHub Models (Grok-3)...")
        response = requests.post(
            f"{config.GITHUB_MODELS_ENDPOINT}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.GITHUB_MODELS_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "model": config.PROMPT_GENERATION_MODEL,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 600,
                "response_format": {"type": "json_object"}
            },
            timeout=25
        )
        
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            queries = _parse_llm_response(content)
            if queries:
                logging.info(f"   ✅ GitHub Models: {len(queries)} queries generadas")
                return queries
        else:
            logging.warning(f"   ⚠️ GitHub Models respondió {response.status_code}")
    except Exception as e:
        logging.warning(f"   ⚠️ GitHub Models falló: {e}")
    return []


def _try_groq(prompt: str, system_msg: str) -> List[str]:
    """Intenta generar queries con Groq."""
    try:
        groq_key = getattr(config, 'GROQ_API_KEY', None)
        if not groq_key:
            return []
        
        logging.info("🤖 [Query Gen] Intentando con Groq...")
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 600,
                "response_format": {"type": "json_object"}
            },
            timeout=25
        )
        
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            queries = _parse_llm_response(content)
            if queries:
                logging.info(f"   ✅ Groq: {len(queries)} queries generadas")
                return queries
        else:
            logging.warning(f"   ⚠️ Groq respondió {response.status_code}")
    except Exception as e:
        logging.warning(f"   ⚠️ Groq falló: {e}")
    return []


def _try_huggingface(prompt: str, system_msg: str) -> List[str]:
    """Intenta generar queries con HuggingFace Router."""
    try:
        hf_token = getattr(config, 'HUGGINGFACE_TOKEN', None)
        if not hf_token:
            return []
        
        logging.info("🤖 [Query Gen] Intentando con HuggingFace...")
        response = requests.post(
            "https://router.huggingface.co/novita/v3/openai/chat/completions",
            headers={
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/json"
            },
            json={
                "model": "Qwen/Qwen2.5-72B-Instruct",
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 600
            },
            timeout=30
        )
        
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            queries = _parse_llm_response(content)
            if queries:
                logging.info(f"   ✅ HuggingFace: {len(queries)} queries generadas")
                return queries
        else:
            logging.warning(f"   ⚠️ HuggingFace respondió {response.status_code}")
    except Exception as e:
        logging.warning(f"   ⚠️ HuggingFace falló: {e}")
    return []


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
    
    # 4. Detectar frases técnicas en inglés dentro de texto español
    # Patrones comunes: "Modelos de Lenguaje Grande" → "Large Language Model"
    english_technical = {
        'modelos de lenguaje': 'Large Language Model',
        'modelo de lenguaje': 'Large Language Model',
        'aprendizaje profundo': 'deep learning',
        'aprendizaje automático': 'machine learning',
        'aprendizaje de máquina': 'machine learning',
        'inteligencia artificial': 'artificial intelligence',
        'redes neuronales': 'neural network',
        'red neuronal': 'neural network',
        'procesamiento de lenguaje natural': 'natural language processing',
        'análisis estático': 'static analysis',
        'análisis de código': 'code analysis',
        'código fuente': 'source code',
        'vulnerabilidades': 'vulnerability',
        'detección de vulnerabilidades': 'vulnerability detection',
        'falsos positivos': 'false positive',
        'falso positivo': 'false positive',
        'seguridad informática': 'cybersecurity',
        'seguridad de software': 'software security',
        'revisión de código': 'code review',
        'enfermedades cardiovasculares': 'cardiovascular disease',
        'enfermedad cardiovascular': 'cardiovascular disease',
        'insuficiencia cardíaca': 'heart failure',
        'fibrilación auricular': 'atrial fibrillation',
        'diagnóstico temprano': 'early diagnosis',
        'detección temprana': 'early detection',
        'diabetes': 'diabetes',
        'cáncer': 'cancer',
        'salud mental': 'mental health',
        'educación superior': 'higher education',
        'rendimiento académico': 'academic performance',
        'cambio climático': 'climate change',
        'energía renovable': 'renewable energy',
        'internet de las cosas': 'Internet of Things',
        'computación en la nube': 'cloud computing',
        'cadena de suministro': 'supply chain',
        'experiencia del usuario': 'user experience',
        'interfaz de usuario': 'user interface',
        'base de datos': 'database',
        'minería de datos': 'data mining',
        'ciencia de datos': 'data science',
        'gemelo digital': 'digital twin',
        'gemelos digitales': 'digital twin',
        'realidad aumentada': 'augmented reality',
        'realidad virtual': 'virtual reality',
    }
    
    q_lower = question.lower()
    for esp, eng in english_technical.items():
        if esp in q_lower and eng.lower() not in seen:
            _add(eng)
    
    # 5. Palabras en inglés que aparecen directamente en la pregunta
    english_words_in_text = re.findall(r'\b([a-z]{2,}(?:\s+[a-z]{2,}){0,2})\b', question)
    for w in english_words_in_text:
        if w.lower() not in seen and _is_english_technical(w):
            _add(w)
    
    logging.info(f"   🔑 Términos técnicos (EN): {terms[:8]}")
    return terms[:8]


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
    cache_key_raw = f"synonyms_v1_{question.strip().lower()}"
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

For each KEY CONCEPT in the research question, generate:
1. The main concept in English
2. All scientific synonyms, acronyms, and equivalent terms used in academic papers
3. Specific sub-types or variants relevant to this domain

RULES:
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

    result = None
    for try_func in [_try_github_models, _try_groq, _try_huggingface]:
        raw = try_func(user_prompt, system_msg)
        if raw:
            result = raw
            break

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

    # Fallback: usar extract_english_terms como base
    logging.warning("⚠️ [Synonyms] LLM falló, usando fallback de términos técnicos")
    fallback_terms = extract_english_terms(question)
    return {
        "synonyms": {},
        "flat_terms": fallback_terms,
        "expanded_queries": fallback_terms[:3],
    }


def expand_query(question: str, max_terms: int = 12) -> List[str]:
    """Función principal: Multi-provider LLM con fallback inteligente en inglés."""
    return expand_query_with_llm(question)