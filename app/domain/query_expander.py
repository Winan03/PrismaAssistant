"""
Query Expander - VERSIÃ“N MULTI-PROVIDER + AGNÃ“STICO AL DOMINIO
Genera ecuaciones booleanas EN INGLÃ‰S basadas en los conceptos de la pregunta.
Si el LLM falla, extrae tÃ©rminos tÃ©cnicos y genera queries en inglÃ©s.
"""
import requests
import config
import logging
import json
import re
import os
import hashlib
from typing import List, Optional, Dict
from modules.ai.ai_model import generate_text, generate_text_with_ollama_model

# Cache
CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_VALIDATION_VERSION = "v3"

_BROAD_SYNONYM_TERMS = set(getattr(config, "BROAD_SYNONYM_TERMS", ()))
_GENERIC_EQUIVALENCE_HEAD_TERMS = set(getattr(config, "GENERIC_EQUIVALENCE_HEAD_TERMS", ()))

def get_cache_key(question: str) -> str:
    # v6: Queries semÃ¡nticas cortas (no booleanas pesadas)
    key = f"semantic_v6_{question.strip().lower()}"
    return hashlib.md5(key.encode()).hexdigest()

def load_from_cache(question: str) -> Optional[List[str]]:
    cache_key = get_cache_key(question)
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            if _is_validated_cache(cached, "semantic_queries") and _validate_query_list(cached.get("terms", [])):
                return cached["terms"]
            _discard_cache_file(cache_path, "semantic_queries cache is stale or invalid")
        except Exception:
            _discard_cache_file(cache_path, "semantic_queries cache could not be read")
    return None

def save_to_cache(question: str, terms: List[str]):
    if not _validate_query_list(terms):
        logging.warning("[Cache] Semantic queries not cached: invalid structure")
        return
    if not _llm_cache_validator("semantic_queries", question, {"terms": terms}):
        logging.warning("[Cache] Semantic queries not cached: semantic validation failed")
        return
    payload = _with_cache_meta({"question": question, "terms": terms}, "semantic_queries")
    try:
        with open(os.path.join(CACHE_DIR, f"{get_cache_key(question)}.json"), 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass


def _discard_cache_file(cache_path: str, reason: str) -> None:
    try:
        os.remove(cache_path)
        logging.info("[Cache] Removed %s: %s", os.path.basename(cache_path), reason)
    except OSError:
        logging.warning("[Cache] Ignoring %s: %s", os.path.basename(cache_path), reason)


def _with_cache_meta(payload: Dict, kind: str) -> Dict:
    enriched = payload.copy()
    enriched["_cache_meta"] = {
        "kind": kind,
        "validation_version": CACHE_VALIDATION_VERSION,
        "validated": True,
    }
    return enriched


def _strip_cache_meta(payload: Dict) -> Dict:
    return {k: v for k, v in payload.items() if k != "_cache_meta"}


def _is_validated_cache(payload: object, kind: str) -> bool:
    if not isinstance(payload, dict):
        return False
    meta = payload.get("_cache_meta")
    return (
        isinstance(meta, dict)
        and meta.get("kind") == kind
        and meta.get("validation_version") == CACHE_VALIDATION_VERSION
        and meta.get("validated") is True
    )


_PICO_CATEGORY_CUES = {
    "C": (
        "comparator", "comparison", "control", "baseline", "placebo",
        "standard", "usual", "conventional", "alternative", "non-adaptive",
        "fixed", "versus", "vs",
    ),
    "O": (
        "outcome", "effect", "impact", "measure", "measurement", "metric",
        "score", "rate", "frequency", "duration", "change", "response",
        "improvement", "reduction", "increase", "decrease",
    ),
    "I": (
        "intervention", "treatment", "program", "protocol", "method",
        "approach", "model", "system", "tool", "device", "platform",
        "application", "software", "technique", "technology", "exposure",
        "agent", "strategy",
    ),
    "P": (
        "population", "participant", "participants", "patient", "patients",
        "subject", "subjects", "user", "users", "cohort", "sample",
        "learner", "learners",
    ),
}


def _has_category_cue(text: str, cues: tuple) -> bool:
    return any(re.search(rf"(?<![a-z0-9]){re.escape(cue)}(?![a-z0-9])", text) for cue in cues)


def infer_pico_category(term: str) -> str:
    text = str(term or "").lower()
    for category in ("C", "O", "I", "P"):
        if _has_category_cue(text, _PICO_CATEGORY_CUES[category]):
            return category
    return "P"


def _meaningful_tokens(term: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(term or "").lower())
        if len(token) > 1
    ]


def _is_broad_synonym(main: str, synonym: str, category: str) -> bool:
    normalized = " ".join(_meaningful_tokens(synonym))
    if not normalized:
        return True
    if normalized in _BROAD_SYNONYM_TERMS:
        return True
    main_tokens = _meaningful_tokens(main)
    synonym_tokens = _meaningful_tokens(synonym)
    if category in {"I", "C", "O"} and len(main_tokens) >= 2 and len(synonym_tokens) == 1:
        return True
    if category in {"I", "C", "O"}:
        if len(main_tokens) >= 2 and len(synonym_tokens) >= 2 and main_tokens[-1] == synonym_tokens[-1]:
            main_modifiers = set(main_tokens[:-1]) - _GENERIC_EQUIVALENCE_HEAD_TERMS
            synonym_modifiers = set(synonym_tokens[:-1]) - _GENERIC_EQUIVALENCE_HEAD_TERMS
            if main_modifiers and synonym_modifiers and not main_modifiers.intersection(synonym_modifiers):
                return True
        main_specific = [token for token in main_tokens if token not in _GENERIC_EQUIVALENCE_HEAD_TERMS]
        if main_specific and not set(main_specific).intersection(synonym_tokens):
            return True
    return False


def _validate_term_list(value: object, required: bool = False) -> bool:
    if not isinstance(value, list):
        return False
    clean_terms = [str(item).strip() for item in value if str(item).strip()]
    if required and not clean_terms:
        return False
    return all(1 <= len(term.split()) <= 8 for term in clean_terms)


def _validate_query_list(value: object) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(item, str) and 2 <= len(item.split()) <= 12 for item in value)


def _validate_pico_structure(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    if not _validate_term_list(payload.get("P"), required=True):
        return False
    if not _validate_term_list(payload.get("I"), required=True):
        return False
    for key in ("C", "O"):
        if key in payload and not _validate_term_list(payload.get(key), required=False):
            return False
    semantic_queries = payload.get("semantic_queries", [])
    return not semantic_queries or _validate_query_list(semantic_queries)


def _validate_atom_groups(value: object) -> bool:
    if not isinstance(value, list) or not value:
        return False
    required_count = 0
    for item in value:
        if not isinstance(item, dict):
            return False
        name = str(item.get("name") or item.get("id") or "").strip()
        terms = item.get("terms")
        category = str(item.get("category") or "").upper()
        if not name or category not in {"P", "I", "C", "O"}:
            return False
        if not _validate_term_list(terms, required=True):
            return False
        if bool(item.get("required", False)):
            required_count += 1
    return required_count >= 2


def _sanitize_atom_groups(value: object) -> List[Dict]:
    if not isinstance(value, list):
        return []
    cleaned: List[Dict] = []
    seen = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("id") or "").strip()
        category = str(item.get("category") or infer_pico_category(name)).upper()
        if not name or category not in {"P", "I", "C", "O"}:
            continue
        raw_terms = item.get("terms", [])
        if not isinstance(raw_terms, list):
            continue
        terms: List[str] = []
        for term in raw_terms:
            text = re.sub(r"\s+", " ", str(term or "").strip())
            key = text.lower()
            if not text or key in seen or len(text.split()) > 8:
                continue
            seen.add(key)
            terms.append(text)
        if not terms:
            continue
        cleaned.append({
            "name": name,
            "category": category,
            "required": bool(item.get("required", False)),
            "terms": terms,
        })
    return cleaned


def _validate_synonym_structure(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    synonyms = payload.get("synonyms")
    categories = payload.get("categories")
    if not isinstance(synonyms, dict) or not synonyms:
        return False
    if categories is not None and not isinstance(categories, dict):
        return False
    concept_categories = set()
    for concept, terms in synonyms.items():
        if not str(concept).strip() or not _validate_term_list(terms, required=False):
            return False
        category = str((categories or {}).get(concept) or infer_pico_category(str(concept))).upper()
        if category not in {"P", "I", "C", "O"}:
            return False
        concept_categories.add(category)

    atom_groups = payload.get("atom_groups")
    if not _validate_atom_groups(atom_groups):
        return False
    atom_categories = {
        str(item.get("category") or "").upper()
        for item in atom_groups
        if isinstance(item, dict)
    }
    required_categories = {
        str(item.get("category") or "").upper()
        for item in atom_groups
        if isinstance(item, dict) and bool(item.get("required", False))
    }
    if {"P", "I"}.intersection(concept_categories) - required_categories:
        return False
    if concept_categories - atom_categories:
        return False
    return True


def _build_fallback_synonym_payload_from_pico(question: str) -> Dict:
    """Build a conservative, domain-agnostic filter artifact from validated PICO."""
    pico = generate_api_queries_with_llm(question)
    synonyms: Dict[str, List[str]] = {}
    categories: Dict[str, str] = {}
    flat_terms: List[str] = []
    atom_groups: List[Dict] = []

    for category in ("P", "I", "C", "O"):
        terms = [
            re.sub(r"\s+", " ", str(term or "").strip())
            for term in pico.get(category, [])
            if str(term or "").strip()
        ]
        if not terms:
            continue

        for term in terms:
            synonyms[term] = []
            categories[term] = category
            flat_terms.append(term)

        atom_groups.append({
            "name": f"{category} evidence",
            "category": category,
            "required": category in {"P", "I"},
            "terms": terms,
        })

    seen = set()
    flat_clean = []
    for term in flat_terms:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            flat_clean.append(term)

    return {
        "synonyms": synonyms,
        "categories": categories,
        "atom_groups": atom_groups,
        "flat_terms": flat_clean,
        "expanded_queries": list(dict.fromkeys([
            query for query in pico.get("semantic_queries", []) if isinstance(query, str)
        ]))[:6],
        "_validation_status": "fallback_from_pico",
    }


def _llm_cache_validator(kind: str, question: str, payload: Dict) -> bool:
    if not bool(getattr(config, "CACHE_LLM_VALIDATION_ENABLED", True)):
        return True

    prompt = f"""Validate whether this generated artifact should be cached for future systematic-review runs.

Research question:
"{question}"

Artifact kind: {kind}
Artifact JSON:
{json.dumps(payload, ensure_ascii=False)[:5000]}

Validation rules:
- The artifact must be aligned with the exact research question.
- Population, intervention, outcome, context, and comparator terms must stay in their intended roles.
- Do not accept broader adjacent concepts as substitutes for a specific required concept.
- For synonyms, every synonym must be equivalent enough to help retrieval/filtering, not merely topically related.
- For synonyms, composite interventions must be represented as required atom_groups, not flattened into broad synonyms.
- For semantic queries, each query must combine the core population and intervention unless the artifact kind is not query-related.

Return ONLY valid JSON:
{{"valid": true, "reason": "brief reason"}}"""

    try:
        result = _call_llm_json(
            prompt,
            "You are a strict cache quality validator. Output valid JSON only.",
            return_dict=True,
        )
    except Exception as exc:
        logging.warning("[Cache] LLM validator failed for %s: %s", kind, exc)
        return False

    if not isinstance(result, dict):
        return False
    valid = str(result.get("valid", "")).strip().lower()
    accepted = result.get("valid") is True or valid == "true"
    if not accepted:
        logging.warning("[Cache] %s rejected by validator: %s", kind, result.get("reason", "no reason"))
    return accepted


# ============================================================
# PROMPT AGNÃ“STICO (compartido entre providers)
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
3. DO NOT use boolean operators (AND, OR, NOT) â€” just write natural keyword phrases
4. Each query should be 3-7 words of specific technical terms
5. Focus on the EXACT topic, not broad categories
6. Include specific tools, methods, or technologies mentioned in the question
7. Include acronyms where appropriate (LLM, SAST, CNN, etc.)


JSON Output:"""

def _build_api_prompt(question: str) -> str:
    return f"""You are a systematic review librarian specialized in academic search engineering.

Research Question: "{question}"

TASK:
Extract the core PICO components from the research question, AND generate 4-6 broad semantic dragnet queries.

1. P (Population): Target population only.
2. I (Intervention): Core technologies/interventions.
3. C (Context/Comparison): Explicit setting/context constraints and comparison systems if any.
4. O (Outcome): Expected outcomes.
5. semantic_queries: Generate 6 short semantic search queries combining P and I, optionally adding highly distinctive O or C terms.

RULES:
- Output ONLY valid JSON in this exact structure:
{{
  "P": ["population term 1", "population term 2"],
  "I": ["intervention term 1", "intervention term 2"],
  "C": [],
  "O": ["outcome term 1"],
  "semantic_queries": ["query 1", "query 2", "query 3"]
}}
- ALL terms and queries MUST be in ENGLISH.
- DO NOT use boolean operators (AND/OR), no quotes.
- Do not omit explicit setting, environment, domain, or contextual constraints from C.
- Keep P and I specific; do not replace them with broader adjacent populations or technologies.
- Use terminology likely to appear in academic titles/abstracts; do not create unnatural literal compounds.
- If a concept combines an object and a method/technology, keep them as separate searchable concepts unless the compound is established in the literature.
- semantic_queries must include at least one P term and one I term. Avoid single-concept generic queries.

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

    logging.info("ðŸ¤– [Query Gen] Solicitando queries al LLM Router...")
    queries = _call_llm_json(prompt, system_msg, return_dict=False)
    if queries:
        save_to_cache(question, queries)
        return queries

    return generate_fallback_queries(question)

def get_exclusion_terms_with_llm(question: str) -> List[str]:
    """Identifica temas que suelen causar ruido para esta RQ especÃ­fica (v7.8)."""
    prompt = f"""Identify 6-8 SHORT terms or topics (1-2 words max) that are SEMANTIC DISTRACTORS for this research question.
A distractor is a topic that shares SOME keywords but has a completely DIFFERENT context, domain, or goal.

Research Question: "{question}"

For example, if the research is about X applied to Y, what other completely unrelated fields (Z) also use X? Output those unrelated fields.
DO NOT output terms that are relevant to the research question.

Output ONLY valid JSON: {{"exclusions": ["term1", "term2", ... ]}}
Terms MUST be 1-2 words only for strict keyword matching."""
    
    system_msg = "You are a research filter specialist. Identify out-of-context noise terms."
    
    logging.info("ðŸ¤– [Exclusions] Solicitando tÃ©rminos distractores al LLM Router...")
    res = _call_llm_json(prompt, system_msg, return_dict=False)
    if res: return res
        
    return ["clinical", "patient", "medical"] # Fallback mÃ­nimo universal

def generate_api_queries_with_llm(question: str) -> dict:
    """Genera componentes PICO estructurados y queries semÃ¡nticas para two_phase_search."""
    # Cache distinguible por prefijo en get_cache_key (v7.8: PICO dict cache)
    cache_key = hashlib.md5(f"pico_v10_tiered_{question.strip().lower()}".encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            if _is_validated_cache(cached, "pico") and _validate_pico_structure(cached):
                clean_cached = _strip_cache_meta(cached)
                logging.info("[PICO] Validated cache loaded")
                return clean_cached
            _discard_cache_file(cache_path, "PICO cache is stale or invalid")
        except Exception:
            _discard_cache_file(cache_path, "PICO cache could not be read")

    prompt = _build_api_prompt(question)
    system_msg = _build_system_msg()

    logging.info("ðŸ¤– [API Queries] Solicitando PICO al LLM Router...")
    # return_dict=True para obtener el dict completo
    result = _call_llm_json(prompt, system_msg, return_dict=True)

    if isinstance(result, dict) and result:
        # Si el LLM devolviÃ³ "queries" pero no las claves PICO directas, mapear
        if "queries" in result and not any(k in result for k in ["P", "I", "C", "O"]):
            queries = result["queries"]
            result = {
                "P": queries[:2] if len(queries) >= 2 else queries,
                "I": queries[2:4] if len(queries) >= 4 else queries[2:],
                "C": [],
                "O": queries[4:] if len(queries) >= 4 else [],
                "semantic_queries": queries
            }

        # Asegurar claves PICO mÃ­nimas
        for k in ["P", "I", "C", "O"]:
            if k not in result:
                result[k] = []
        if "semantic_queries" not in result:
            # Reconstruir semantic_queries de P e I
            p_terms = result.get("P", [])
            i_terms = result.get("I", [])
            result["semantic_queries"] = [f"{p} {i}" for p in p_terms for i in i_terms][:6]

        if _validate_pico_structure(result) and _llm_cache_validator("pico", question, result):
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(_with_cache_meta(result, "pico"), f, ensure_ascii=False)
            except Exception:
                pass
            return result

        _discard_cache_file(cache_path, "PICO result failed validation")
        logging.warning("[PICO] LLM result rejected; using technical-term fallback")

    # Fallback inteligente: construir un PICO mÃ­nimo desde la lista de queries fallback
    fallback_list = generate_fallback_queries(question)
    return {
        "P": fallback_list[:2],
        "I": fallback_list[2:4],
        "C": [],
        "O": fallback_list[4:],
        "semantic_queries": fallback_list
    }


def _call_llm_json(prompt: str, system_msg: str, return_dict: bool = False):
    """Llama al LLM usando el router multi-provider y parsea el JSON."""
    # AÃ±adimos instrucciÃ³n para que devuelva JSON
    instruction = prompt + "\n\nCRITICAL: You MUST output ONLY valid JSON format. Do not use markdown blocks like ```json."
    
    planner_model = getattr(config, "OLLAMA_MODEL_PLANNER", "")
    if planner_model:
        response = generate_text_with_ollama_model(
            instruction=instruction,
            model_name=planner_model,
            input_text="",
            max_tokens=1024,
            system_prompt=system_msg,
        )
    else:
        response = generate_text(instruction=instruction, input_text="", max_tokens=1024, system_prompt=system_msg)
    if "Error de generaciÃ³n" in response or "âš ï¸" in response:
        logging.warning(f"âš ï¸ [Query Gen] Error del LLM Router: {response}")
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
    Genera queries EN INGLÃ‰S desde los tÃ©rminos de la pregunta.
    Prefiere acrÃ³nimos y tÃ©rminos tÃ©cnicos en inglÃ©s.
    """
    logging.info("ðŸ”§ Generando queries desde tÃ©rminos tÃ©cnicos (Fallback sin LLM)...")
    
    terms = extract_english_terms(question)
    
    if len(terms) < 2:
        # Ãšltima lÃ­nea: limpiar pregunta de stopwords y usar directamente
        clean = _clean_question_for_search(question)
        return [clean] if clean else ["systematic review"]
    
    # Query 1: Todos los conceptos principales (especÃ­fico)
    q1 = " AND ".join(f'"{t}"' if ' ' in t else f'"{t}"' for t in terms[:3])
    
    # Query 2: Solo 2 conceptos principales (mÃ¡s amplio)
    q2 = " AND ".join(f'"{t}"' if ' ' in t else f'"{t}"' for t in terms[:2])
    
    # Query 3: Primer y tercer concepto (cobertura diferente)
    if len(terms) >= 3:
        q3 = f'"{terms[0]}" AND "{terms[2]}"'
        queries = [q1, q2, q3]
    else:
        queries = [q1, q2]
    
    for i, q in enumerate(queries):
        logging.info(f"   ðŸ“‹ Fallback Query {i+1}: {q[:100]}...")
    
    return queries


def extract_english_terms(question: str) -> List[str]:
    """
    Extrae tÃ©rminos tÃ©cnicos EN INGLÃ‰S de la pregunta.
    Prioriza: acrÃ³nimos > tÃ©rminos en parÃ©ntesis explicativos > sustantivos tÃ©cnicos.
    """
    terms = []
    seen = set()
    
    def _add(t: str):
        key = t.lower().strip()
        if key and key not in seen and len(key) > 1:
            seen.add(key)
            terms.append(t.strip())
    
    # 1. AcrÃ³nimos (SIEMPRE son en inglÃ©s): LLMs, SAST, CNN, IoT, NLP
    acronyms = re.findall(r'\b([A-Z][A-Za-z]*[A-Z]+[a-z]*)\b', question)  # LLMs, SAST
    acronyms += re.findall(r'\b([A-Z]{2,6}s?)\b', question)  # LLM, SAST, CNNs
    skip = {'AND', 'OR', 'NOT', 'THE', 'FOR', 'DE', 'LA', 'LOS', 'EN', 'DEL', 'UNA'}
    for a in acronyms:
        clean_a = a.rstrip('s')  # LLMs -> LLM
        if clean_a.upper() not in skip:
            _add(a)
    
    # 2. Texto entre parÃ©ntesis que parece inglÃ©s/tÃ©cnico
    parens = re.findall(r'\(([^)]+)\)', question)
    for p in parens:
        p = p.strip()
        # Si es un acrÃ³nimo solo, ya lo tenemos
        if re.match(r'^[A-Z]{2,6}s?$', p):
            continue
        # Si contiene texto en inglÃ©s (letters only, no Spanish articles)
        if re.search(r'[a-z]{3,}', p) and not re.search(r'\b(los|las|del|una|para|con)\b', p.lower()):
            _add(p)
    
    # 3. TÃ©rminos entre comillas
    quoted = re.findall(r'"([^"]+)"', question)
    for q in quoted:
        _add(q)
    
    # 4. Palabras en inglÃ©s que aparecen directamente en la pregunta
    english_words_in_text = re.findall(r'\b([a-z]{2,}(?:\s+[a-z]{2,}){0,2})\b', question)
    for w in english_words_in_text:
        if w.lower() not in seen and _is_english_technical(w):
            _add(w)
    
    logging.info(f"   ðŸ”‘ TÃ©rminos tÃ©cnicos fallback (EN): {terms[:8]}")
    return terms[:8]

def extract_english_terms_with_llm(question: str) -> List[str]:
    """
    Extrae dinÃ¡micamente los tÃ©rminos tÃ©cnicos y los traduce al inglÃ©s usando el LLM.
    Mantiene la aplicaciÃ³n agnÃ³stica al dominio sin diccionarios hardcodeados.
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
    
    logging.info("ðŸ¤– [English Terms] Solicitando tÃ©rminos tÃ©cnicos en inglÃ©s al LLM Router...")
    terms = _call_llm_json(prompt, system_msg, return_dict=False)
    
    if terms:
        # Limpiar y asegurar que sean vÃ¡lidos
        clean_terms = [t for t in terms if isinstance(t, str) and len(t) > 2]
        if clean_terms:
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump({"question": question, "terms": clean_terms}, f)
            except: pass
            logging.info(f"   ðŸ”‘ TÃ©rminos tÃ©cnicos extraÃ­dos por LLM (EN): {clean_terms}")
            return clean_terms

    # Fallback si el LLM falla
    return extract_english_terms(question)


def _is_english_technical(word: str) -> bool:
    """Detecta si una palabra/frase es un tÃ©rmino tÃ©cnico en inglÃ©s."""
    clean = re.sub(r"\s+", " ", str(word or "").strip().lower())
    if len(clean) < 3 or clean in TOKEN_STOPWORDS:
        return False
    if not re.search(r"[a-z]", clean):
        return False
    if len(clean.split()) >= 2:
        return True
    return bool(re.search(r"(tion|ment|ing|ics|ity|ism|ive|al|ic|ous|ware|graph|model|method|system)$", clean))


def _clean_question_for_search(question: str) -> str:
    """Limpia la pregunta para usarla como query de bÃºsqueda."""
    # Remover signos de puntuaciÃ³n espaÃ±oles
    clean = re.sub(r'[Â¿?Â¡!,;:.]', ' ', question)
    # Remover stopwords comunes en espaÃ±ol
    stopwords = {
        'cuÃ¡l', 'cual', 'cÃ³mo', 'como', 'quÃ©', 'que', 'es', 'son', 'las', 'los',
        'del', 'de', 'la', 'el', 'en', 'un', 'una', 'para', 'por', 'con', 'sin',
        'sobre', 'entre', 'mÃ¡s', 'mas', 'se', 'al', 'a', 'e', 'i', 'o', 'u', 'y',
        'su', 'sus', 'mi', 'tu', 'frente', 'durante', 'mediante', 'travÃ©s',
    }
    words = [w for w in clean.split() if w.lower() not in stopwords and len(w) > 1]
    return " ".join(words[:10])


# ============================================================
# EXPANSIÃ“N SEMÃNTICA CON SINÃ“NIMOS (equivalente a WordNet + Corpus)
# ============================================================

def expand_query_with_synonyms(
    question: str,
    corpus_sample: Optional[List[str]] = None,
) -> Dict:
    """
    Genera sinÃ³nimos, acrÃ³nimos y jerga cientÃ­fica equivalente para los
    tÃ©rminos clave de la RQ. Equivalente al WordNet + Corpus SemÃ¡ntico
    recomendado por el profesor.

    Estrategia:
      1. Extrae los conceptos clave de la RQ
      2. Pide al LLM generar variantes cientÃ­ficas en inglÃ©s
      3. Opcionalmente, adapta al vocabulario del corpus recibido
      4. Retorna tÃ©rminos enriquecidos para inyectar en BM25

    Args:
        question:      Pregunta de investigaciÃ³n (cualquier idioma)
        corpus_sample: Muestra de abstracts del corpus (opcional, para
                       adaptar el vocabulario al dominio especÃ­fico)

    Returns:
        Dict con:
          - "synonyms":  tÃ©rminos sinÃ³nimos por concepto clave
          - "flat_terms": lista plana de todos los tÃ©rminos para BM25
          - "expanded_queries": queries enriquecidas con sinÃ³nimos
    """
    # Cache con prefijo distinto para no mezclar con expand_query_with_llm
    cache_key_raw = f"synonyms_v7_validated_atoms_{question.strip().lower()}"
    cache_key = hashlib.md5(cache_key_raw.encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"syn_{cache_key}.json")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            logging.info(f"ðŸ”‘ [Synonyms] Cargado de cachÃ©: {len(cached.get('flat_terms', []))} tÃ©rminos")
            if _is_validated_cache(cached, "synonyms") and _validate_synonym_structure(cached):
                clean_cached = _strip_cache_meta(cached)
                logging.info("[Synonyms] Validated cache loaded: %d terms", len(clean_cached.get("flat_terms", [])))
                return clean_cached
            _discard_cache_file(cache_path, "synonyms cache is stale or invalid")
        except Exception:
            _discard_cache_file(cache_path, "synonyms cache could not be read")

    # Construir corpus hint si se proporcionÃ³ muestra
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

EXTRACT THE 3 TO 6 MOST IMPORTANT ELIGIBILITY PILLARS of the research question
(target population, main intervention/technology, primary outcome, and explicit context/comparator when present).

Also produce atom_groups: small mandatory/optional evidence atoms used for a strict pre-screen.
If an intervention is composite, do NOT flatten it into one broad synonym. Split it into required atoms
such as population, physical intervention/agent, core technology, distinctive technical modality, outcome,
context, or comparator as implied by the research question.

For each of these CORE PILLARS, generate:
1. The main concept in English
2. All scientific synonyms, acronyms, and equivalent terms used in academic papers
3. Specific sub-types or variants relevant to this domain
4. A PICO category: P, I, C, or O

RULES:
- Extract up to 6 concepts. Do not extract every detail, but keep explicit intervention and outcome constraints.
- ALL terms must be in English
- Include domain-specific acronyms when they are present or clearly implied by the question.
- Include both formal terms and common abbreviations
- Focus on terms that would appear in academic paper titles/abstracts
- Each term should be 1-4 words maximum
- Synonyms must be true retrieval equivalents, not broader neighboring topics.
- atom_groups terms may be broader than synonyms only when they represent one required atom.
- At least population and the central intervention atoms must be required=true.

Output ONLY this JSON format:
{{
  "concepts": [
    {{
      "main": "main concept name",
      "category": "P, I, C, or O",
      "synonyms": ["synonym1", "synonym2", "acronym1", "variant1"]
    }}
  ],
  "atom_groups": [
    {{
      "name": "atom name",
      "category": "P, I, C, or O",
      "required": true,
      "terms": ["term1", "term2"]
    }}
  ],
  "cross_terms": ["term that combines 2+ concepts", "compound term2"]
}}"""

    logging.info("ðŸ¤– [Synonyms] Solicitando sinÃ³nimos al LLM Router...")
    result = _call_llm_json(user_prompt, system_msg, return_dict=True)

    # Procesar resultado del LLM
    if result and isinstance(result, dict) and "concepts" in result:
        concepts = result.get("concepts", [])
        cross_terms = result.get("cross_terms", [])
        atom_groups = _sanitize_atom_groups(result.get("atom_groups", []))

        # Construir lista plana de todos los tÃ©rminos
        flat_terms = []
        synonyms_by_concept = {}
        categories_by_concept = {}
        query_synonyms_by_concept = {}
        for concept in concepts:
            main = concept.get("main", "")
            category = str(concept.get("category") or infer_pico_category(main)).upper()
            syns = [
                str(s).strip()
                for s in concept.get("synonyms", [])
                if str(s).strip() and not _is_broad_synonym(main, str(s), category)
            ]
            if main:
                flat_terms.append(main)
                synonyms_by_concept[main] = syns
                categories_by_concept[main] = category
                query_synonyms_by_concept[main] = syns[:3]
            flat_terms.extend(syns)

        if not atom_groups:
            atom_groups = [
                {
                    "name": main,
                    "category": category,
                    "required": category in {"P", "I"},
                    "terms": [main, *query_synonyms_by_concept.get(main, [])],
                }
                for main, category in categories_by_concept.items()
                if main and category in {"P", "I", "C", "O"}
            ]

        flat_terms.extend(cross_terms)

        # Deduplicar y limpiar
        seen = set()
        flat_clean = []
        for t in flat_terms:
            t_clean = str(t).strip().lower()
            if t_clean and t_clean not in seen and len(t_clean) >= 2:
                seen.add(t_clean)
                flat_clean.append(str(t).strip())

        # Construir queries enriquecidas (combinando concepto principal + sinÃ³nimos)
        expanded_queries = []
        for concept in concepts:
            main = concept.get("main", "")
            syns = concept.get("synonyms", [])[:3]  # Top 3 sinÃ³nimos
            syns = query_synonyms_by_concept.get(main, syns)
            if main and syns:
                # Query: "main synonym1 synonym2"
                expanded_queries.append(f"{main} {' '.join(syns)}")
        expanded_queries.extend(cross_terms[:3])

        output = {
            "synonyms": synonyms_by_concept,
            "categories": categories_by_concept,
            "atom_groups": atom_groups,
            "flat_terms": flat_clean,
            "expanded_queries": expanded_queries,
            "_validation_status": "candidate",
        }

        if not _validate_synonym_structure(output):
            logging.warning("[Synonyms] LLM artifact rejected locally; using validated PICO fallback")
            return _build_fallback_synonym_payload_from_pico(question)
        if not _llm_cache_validator("synonyms", question, output):
            logging.warning("[Synonyms] LLM artifact rejected by validator; using validated PICO fallback")
            return _build_fallback_synonym_payload_from_pico(question)

        output["_validation_status"] = "validated"
        output_to_cache = _with_cache_meta(output, "synonyms")

        # Guardar en cachÃ©
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(output_to_cache, f, ensure_ascii=False)
        except Exception:
            pass

        logging.info(
            f"ðŸŒ [Synonyms] ExpansiÃ³n completada: {len(flat_clean)} tÃ©rminos Ãºnicos | "
            f"{len(synonyms_by_concept)} conceptos | {len(expanded_queries)} queries enriquecidas"
        )
        return output

    # â”€â”€ Fallback: construir sinÃ³nimos estructurados bÃ¡sicos â”€â”€
    # Cuando el LLM falla, construimos un dict mÃ­nimo de sinÃ³nimos
    # agrupado por concepto para que concept_presence_filter pueda funcionar.
    # Se filtran acrÃ³nimos cortos para no envenenar BM25.
    logging.warning("âš ï¸ [Synonyms] LLM fallÃ³, usando fallback de tÃ©rminos tÃ©cnicos")
    fallback_terms_raw = extract_english_terms(question)

    # Filtrar acrÃ³nimos problemÃ¡ticos del fallback
    _BAD_ACRONYMS = {'ia', 'st', 'ai', 'qa', 'ml', 'dl', 'se', 'it', 'is', 'rq', 'nn', 'cv'}
    fallback_terms = [
        t for t in fallback_terms_raw
        if len(t.strip()) >= 3
        and t.strip().lower() not in _BAD_ACRONYMS
        and not (t.strip().isupper() and len(t.strip()) <= 3)
    ]
    logging.info(f"   ðŸ”‘ TÃ©rminos tÃ©cnicos (EN, filtrados): {fallback_terms}")

    # Construir synomyms agrupados: cada tÃ©rmino de 3+ palabras es un "concepto"
    # Los tÃ©rminos cortos son "sinÃ³nimos" del primer concepto largo que encontremos
    synonyms_structured: Dict[str, List[str]] = {}
    long_terms   = [t for t in fallback_terms if len(t.split()) >= 2]
    short_terms  = [t for t in fallback_terms if len(t.split()) == 1]

    if long_terms:
        for lt in long_terms:
            synonyms_structured[lt] = []
        # Distribuir los tÃ©rminos cortos como sinÃ³nimos del primer concepto largo
        if short_terms and long_terms:
            synonyms_structured[long_terms[0]].extend(short_terms)
    elif short_terms:
        # Solo hay tÃ©rminos cortos: usar el mÃ¡s largo como concepto principal
        primary = max(short_terms, key=len)
        synonyms_structured[primary] = [t for t in short_terms if t != primary]

    return {
        "synonyms": synonyms_structured,   # â† Ahora tiene estructura, no vacÃ­o
        "categories": {term: infer_pico_category(term) for term in synonyms_structured},
        "atom_groups": [
            {
                "name": term,
                "category": infer_pico_category(term),
                "required": infer_pico_category(term) in {"P", "I"},
                "terms": [term, *terms],
            }
            for term, terms in synonyms_structured.items()
        ],
        "flat_terms": fallback_terms,
        "expanded_queries": fallback_terms[:3],
        "_validation_status": "technical_term_fallback",
    }


def expand_query(question: str, max_terms: int = 12) -> List[str]:
    """FunciÃ³n principal: Multi-provider LLM con fallback inteligente en inglÃ©s."""
    return expand_query_with_llm(question)
