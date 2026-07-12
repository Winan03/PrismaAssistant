"""Eligibility contract and atom-aware retrieval helpers.

This module keeps the review logic domain-agnostic. The LLM extracts the
question-specific atoms; code only validates structure, ranks coverage, and
prevents broad near-matches from becoming direct evidence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import config
from app.llm.ai_model import generate_text_with_ollama_model
from app.domain.query_expander import generate_api_queries_with_llm

logger = logging.getLogger(__name__)

CACHE_DIR = ".cache"
CONTRACT_CACHE_VERSION = "eligibility_contract_v13"
VALID_ATOM_CATEGORIES = {"P", "I", "O", "C", "CTX", "SD"}
TOKEN_STOPWORDS = {
    "a", "an", "and", "as", "at", "based", "by", "for", "from", "in", "into",
    "of", "on", "or", "the", "to", "using", "with", "within",
}
TEMPORAL_METADATA_RE = re.compile(
    r"\b("
    r"publication\s+(date|year)|published\s+(between|from|in)|"
    r"date\s+range|year\s+range|publication\s+period|"
    r"fecha|periodo|a(?:n|\u00f1)o|202[0-9]|201[0-9]"
    r")\b",
    re.IGNORECASE,
)
BROAD_AI_RE = re.compile(
    r"\b(machine learning|deep learning|aprendizaje autom[a\u00e1]tico|"
    r"artificial intelligence|ai|ml|dl)\b",
    re.IGNORECASE,
)
SPECIFIC_AI_METHOD_RE = re.compile(
    r"\b(ensemble|bagging|boosting|stacking|voting|random forest|xgboost|"
    r"support vector machine|svm|neural network|graph neural network|codebert|"
    r"lstm|bilstm|transformer|feature selection|hyperparameter|optimization|"
    r"optimisation|metaheuristic|swarm|genetic algorithm)\b",
    re.IGNORECASE,
)


def _cache_key(question: str, inclusion_criteria: str, exclusion_criteria: str) -> str:
    raw = "|".join([
        CONTRACT_CACHE_VERSION,
        question.strip().lower(),
        inclusion_criteria.strip().lower(),
        exclusion_criteria.strip().lower(),
    ])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _cache_path(question: str, inclusion_criteria: str, exclusion_criteria: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{_cache_key(question, inclusion_criteria, exclusion_criteria)}.json")


def _clean_json_response(content: str) -> str:
    text = str(content or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0).strip() if match else text


def _normalise_term(term: object) -> str:
    text = re.sub(r"\s+", " ", str(term or "").strip())
    return text[:120]


def _normalise_query(query: object) -> str:
    text = re.sub(r"\s+", " ", str(query or "").strip())
    return text[:320]


def _unique_terms(terms: Iterable[object], limit: int = 10) -> List[str]:
    seen = set()
    clean: List[str] = []
    for term in terms:
        value = _normalise_term(term)
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        clean.append(value)
        if len(clean) >= limit:
            break
    return clean


def _unique_queries(queries: Iterable[object], limit: int = 10) -> List[str]:
    seen = set()
    clean: List[str] = []
    for query in queries:
        value = _normalise_query(query)
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        clean.append(value)
        if len(clean) >= limit:
            break
    return clean


def _tokenise(text: object) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) > 1 and token not in TOKEN_STOPWORDS
    ]


def _query_override_key(query: str) -> str:
    return " ".join(
        token
        for token in re.findall(r"[a-z0-9]+", str(query or "").lower())
        if len(token) > 1
    )


def _atom_query_terms(atom: Dict[str, Any], limit: int = 10) -> List[str]:
    return _unique_terms([
        *(atom.get("query_terms") or []),
        *(atom.get("evidence_terms") or []),
        atom.get("label"),
    ], limit=limit)


def _atom_text(atom: Dict[str, Any]) -> str:
    return " ".join(str(value or "") for value in [
        atom.get("label"),
        *(atom.get("query_terms") or []),
        *(atom.get("evidence_terms") or []),
        *(atom.get("terms") or []),
    ])


def _is_temporal_metadata_atom(atom: Dict[str, Any]) -> bool:
    return bool(TEMPORAL_METADATA_RE.search(_atom_text(atom)))


def _is_broad_ai_atom(atom: Dict[str, Any]) -> bool:
    return bool(BROAD_AI_RE.search(_atom_text(atom)))


def _is_specific_ai_method_atom(atom: Dict[str, Any]) -> bool:
    text = _atom_text(atom)
    return bool(SPECIFIC_AI_METHOD_RE.search(text)) and not _is_broad_ai_atom(atom)


def _query_matches_atom(query: str, atom: Dict[str, Any]) -> bool:
    text = re.sub(r"\s+", " ", str(query or "").lower())
    for term in _atom_query_terms(atom, limit=14):
        for variant in _term_variants(term):
            if _contains_term(text, variant):
                return True
    return False


def _query_matches_any(query: str, atoms: List[Dict[str, Any]]) -> bool:
    return any(_query_matches_atom(query, atom) for atom in atoms)


def _query_matches_all(query: str, atoms: List[Dict[str, Any]]) -> bool:
    return all(_query_matches_atom(query, atom) for atom in atoms)


def _required_atoms_by_role(contract: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    required = [
        atom for atom in contract.get("required_atoms", [])
        if isinstance(atom, dict) and bool(atom.get("required")) and not _is_temporal_metadata_atom(atom)
    ]
    p_atoms = [atom for atom in required if str(atom.get("category") or "").upper() == "P"]
    i_atoms = [atom for atom in required if str(atom.get("category") or "").upper() == "I"]
    central_i = [atom for atom in i_atoms if bool(atom.get("prefilter_required"))]
    if not central_i and i_atoms:
        central_i = i_atoms[:1]
    technical_i = [atom for atom in i_atoms if atom not in central_i]
    return p_atoms, central_i, technical_i


def _query_contract_tier(query: str, contract: Dict[str, Any]) -> Optional[str]:
    p_atoms, central_i, technical_i = _required_atoms_by_role(contract)
    if p_atoms and not _query_matches_any(query, p_atoms):
        return None
    if central_i and not _query_matches_any(query, central_i):
        return None
    if technical_i and _query_matches_all(query, technical_i):
        return "exact"
    if technical_i and _query_matches_any(query, technical_i):
        return "focused"
    if technical_i:
        return None
    return "exact"


def _quote_query_term(term: str) -> str:
    clean = re.sub(r"\s+", " ", str(term or "").strip())
    if not clean:
        return ""
    if re.search(r"\s", clean):
        return f'"{clean}"'
    return clean


def _first_terms(atom: Dict[str, Any], limit: int = 2) -> List[str]:
    return [
        term for term in _atom_query_terms(atom, limit=8)
        if len(_tokenise(term)) >= 1
    ][:limit]


def _cartesian_term_choices(atoms: List[Dict[str, Any]], per_atom_limit: int = 2) -> List[List[str]]:
    choices: List[List[str]] = [[]]
    for atom in atoms:
        terms = _first_terms(atom, limit=per_atom_limit)
        if not terms:
            continue
        choices = [current + [term] for current in choices for term in terms]
    return choices


def _build_contract_repair_queries(contract: Dict[str, Any], limit: int = 12) -> Tuple[List[str], List[str]]:
    p_atoms, central_i, technical_i = _required_atoms_by_role(contract)
    if not p_atoms or not central_i:
        return [], []

    p_terms = _first_terms(p_atoms[0], limit=2)
    central_terms = [
        term
        for atom in central_i
        for term in _first_terms(atom, limit=3)
    ][:4]
    technical_choices = _cartesian_term_choices(technical_i, per_atom_limit=2)

    direct: List[str] = []
    if technical_i and technical_choices:
        for p_term in p_terms:
            for central_term in central_terms:
                for tech_terms in technical_choices:
                    query_terms = [_quote_query_term(p_term), _quote_query_term(central_term)]
                    query_terms.extend(_quote_query_term(term) for term in tech_terms)
                    direct.append(" AND ".join(term for term in query_terms if term))
    else:
        for p_term in p_terms:
            for central_term in central_terms:
                direct.append(" AND ".join(filter(None, [_quote_query_term(p_term), _quote_query_term(central_term)])))

    focused: List[str] = []
    for tech_atom in technical_i:
        for p_term in p_terms[:1]:
            for central_term in central_terms[:2]:
                for tech_term in _first_terms(tech_atom, limit=2):
                    focused.append(" AND ".join(filter(None, [
                        _quote_query_term(p_term),
                        _quote_query_term(central_term),
                        _quote_query_term(tech_term),
                    ])))

    return _unique_queries(direct, limit=limit), _unique_queries(focused, limit=limit)


def _validate_contract_queries(
    contract: Dict[str, Any],
    direct_queries: List[str],
    focused_queries: List[str],
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    valid_direct: List[str] = []
    valid_focused: List[str] = []
    rejected: List[Dict[str, str]] = []

    for query in direct_queries:
        tier = _query_contract_tier(query, contract)
        if tier == "exact":
            valid_direct.append(query)
        elif tier == "focused":
            valid_focused.append(query)
            rejected.append({"query": query, "from": "direct", "action": "demoted_to_focused"})
        else:
            rejected.append({"query": query, "from": "direct", "action": "rejected_missing_required_atoms"})

    for query in focused_queries:
        tier = _query_contract_tier(query, contract)
        if tier in {"exact", "focused"}:
            valid_focused.append(query)
        else:
            rejected.append({"query": query, "from": "focused", "action": "rejected_missing_required_atoms"})

    repair_direct, repair_focused = _build_contract_repair_queries(contract)
    output_direct = valid_direct if valid_direct else repair_direct
    output_focused = [*valid_focused, *repair_focused]
    validation = {
        "direct_input": len(direct_queries),
        "focused_input": len(focused_queries),
        "direct_valid": len(valid_direct),
        "focused_valid": len(valid_focused),
        "direct_repair": 0 if valid_direct else len(repair_direct),
        "focused_repair": len(repair_focused),
        "rejected": rejected[:20],
    }
    return (
        _unique_queries(output_direct, limit=18),
        _unique_queries(output_focused, limit=24),
        validation,
    )


def _looks_composite(text: str) -> bool:
    value = f" {str(text or '').lower()} "
    return any(marker in value for marker in (" and ", " plus ", " with ", " / ", ";", "(", ")"))


def _split_composite_label(label: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(label or "").strip())
    if not text:
        return []

    parenthetical = re.findall(r"\(([^)]+)\)", text)
    prefix = re.sub(r"\([^)]*\)", "", text).strip(" ,;:-")
    parts: List[str] = [prefix] if prefix else []
    for chunk in parenthetical:
        parts.extend(re.split(r"\s+(?:and|plus|or)\s+|[,;/]", chunk))

    clean_parts = []
    for part in parts:
        clean = re.sub(r"\s+", " ", part).strip(" ,;:-")
        if len(_tokenise(clean)) >= 2 or _looks_like_atomic_component(clean):
            clean_parts.append(clean)
    return _unique_terms(clean_parts, limit=6)


def _looks_like_atomic_component(value: str) -> bool:
    clean = re.sub(r"\s+", " ", str(value or "").strip())
    if not clean:
        return False
    compact = re.sub(r"[^A-Za-z0-9]", "", clean)
    if clean.isupper() and 2 <= len(compact) <= 12:
        return True
    if any(char.isdigit() for char in clean):
        return True
    return len(_tokenise(clean)) == 1 and len(compact) >= 4


def _initialism(tokens: Iterable[str]) -> str:
    return "".join(token[:1] for token in tokens if token)


def _terms_for_component(component: str, terms: List[str]) -> List[str]:
    component_tokens = set(_tokenise(component))
    selected: List[str] = []
    for term in terms:
        term_tokens = _tokenise(term)
        term_token_set = set(term_tokens)
        component_initialism = _initialism(term_tokens)
        if (
            component_tokens
            and term_tokens
            and (
                component_tokens.intersection(term_token_set)
                or component_initialism in component_tokens
                or any(component_initialism.startswith(token) for token in component_tokens if len(token) >= 2)
            )
        ):
            selected.append(term)
    return _unique_terms([component, *selected], limit=8)


def _normalise_category(category: object) -> str:
    value = str(category or "").strip().upper()
    if value in {"CONTEXT", "SETTING", "ENVIRONMENT"}:
        return "CTX"
    if value in {"DESIGN", "STUDY_DESIGN", "METHOD"}:
        return "SD"
    return value if value in VALID_ATOM_CATEGORIES else ""


def _normalise_atom(raw: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    label = _normalise_term(raw.get("label") or raw.get("name") or raw.get("requirement"))
    category = _normalise_category(raw.get("category"))
    if not label or not category:
        return None

    atom_id = _normalise_term(raw.get("id")) or f"{category}{index + 1}"
    query_terms = _unique_terms(raw.get("query_terms") or raw.get("terms") or [])
    evidence_terms = _unique_terms(raw.get("evidence_terms") or raw.get("terms") or [])
    terms = _unique_terms([label, *query_terms, *evidence_terms], limit=12)

    atom = {
        "id": atom_id,
        "category": category,
        "label": label,
        "required": bool(raw.get("required", True)),
        "must_be_explicit": bool(raw.get("must_be_explicit", True)),
        "prefilter_required": bool(raw.get("prefilter_required", category in {"P", "I"})),
        "query_terms": query_terms or terms[:4],
        "evidence_terms": evidence_terms or terms[:6],
        "terms": terms,
        "not_acceptable": _unique_terms(raw.get("not_acceptable") or [], limit=8),
    }
    if _is_temporal_metadata_atom(atom):
        logger.info("[EligibilityContract] Ignoring metadata atom handled by filters: %s", label)
        return None
    return atom


def _validate_contract(contract: object) -> bool:
    if not isinstance(contract, dict):
        return False
    atoms = contract.get("required_atoms") or contract.get("atoms")
    if not isinstance(atoms, list):
        return False
    normalised = [
        _normalise_atom(atom, idx)
        for idx, atom in enumerate(atoms)
        if isinstance(atom, dict)
    ]
    normalised = [atom for atom in normalised if atom]
    required_atoms = [atom for atom in normalised if atom.get("required")]
    
    if not required_atoms:
        logger.warning("[EligibilityContract] Validation failed: No required atoms found in contract.")
        return False
    return True


def _normalise_contract(payload: Dict[str, Any], question: str) -> Dict[str, Any]:
    atoms_raw = payload.get("required_atoms") or payload.get("atoms") or []
    atoms = [
        atom
        for idx, raw in enumerate(atoms_raw)
        if isinstance(raw, dict)
        for atom in [_normalise_atom(raw, idx)]
        if atom
    ]

    query_plan = payload.get("query_plan") if isinstance(payload.get("query_plan"), dict) else {}
    direct = _unique_queries(query_plan.get("direct") or payload.get("direct_queries") or [], limit=10)
    focused = _unique_queries(query_plan.get("focused") or payload.get("focused_queries") or [], limit=14)
    scout = _unique_queries(query_plan.get("scout") or payload.get("scout_queries") or [], limit=8)

    contract = {
        "version": CONTRACT_CACHE_VERSION,
        "research_question": question,
        "required_atoms": atoms,
        "query_plan": {
            "direct": direct,
            "focused": focused,
            "scout": scout,
        },
        "exclusion_hints": _unique_terms(payload.get("exclusion_hints") or [], limit=20),
        "background_hints": _unique_terms(payload.get("background_hints") or [], limit=20),
        "_validation_status": "llm_contract",
    }
    return _repair_contract(contract)


def _repair_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    atoms = [
        atom for atom in list(contract.get("required_atoms") or [])
        if isinstance(atom, dict) and not _is_temporal_metadata_atom(atom)
    ]
    repaired: List[Dict[str, Any]] = []
    required_i = [
        atom for atom in atoms
        if atom.get("category") == "I" and atom.get("required")
    ]

    for atom in atoms:
        if (
            atom.get("category") == "I"
            and atom.get("required")
            and _looks_composite(str(atom.get("label") or ""))
        ):
            components = _split_composite_label(str(atom.get("label") or ""))
            if len(components) >= 2:
                source_terms = _unique_terms([
                    *(atom.get("query_terms") or []),
                    *(atom.get("evidence_terms") or []),
                    *(atom.get("terms") or []),
                ], limit=30)
                for idx, component in enumerate(components, start=1):
                    component_terms = _terms_for_component(component, source_terms)
                    split_atom = atom.copy()
                    split_atom["id"] = f"{atom.get('id')}_{idx}"
                    split_atom["label"] = component
                    split_atom["query_terms"] = component_terms[:6]
                    split_atom["evidence_terms"] = component_terms[:8]
                    split_atom["terms"] = component_terms[:10]
                    split_atom["prefilter_required"] = idx == 1
                    repaired.append(split_atom)
                continue
        repaired.append(atom)

    question_text = str(contract.get("research_question") or "")
    has_broad_ai_required = any(
        atom.get("category") == "I"
        and bool(atom.get("required"))
        and _is_broad_ai_atom(atom)
        for atom in repaired
    )
    if BROAD_AI_RE.search(question_text) and not has_broad_ai_required:
        repaired.append({
            "id": "I_broad_ai",
            "category": "I",
            "label": "Machine Learning / Deep Learning",
            "required": True,
            "must_be_explicit": True,
            "prefilter_required": True,
            "query_terms": ["machine learning", "deep learning", "artificial intelligence"],
            "evidence_terms": [
                "machine learning",
                "deep learning",
                "neural network",
                "classification algorithm",
                "supervised learning",
            ],
            "terms": [
                "Machine Learning / Deep Learning",
                "machine learning",
                "deep learning",
                "artificial intelligence",
            ],
            "not_acceptable": [],
        })
        has_broad_ai_required = True

    i_seen = 0
    for atom in repaired:
        category = str(atom.get("category") or "").upper()
        if has_broad_ai_required and category == "I" and _is_specific_ai_method_atom(atom):
            atom["required"] = False
            atom["prefilter_required"] = False
            atom["role_hint"] = "optional_method_example"
            continue
        if category == "P" and atom.get("required"):
            atom["prefilter_required"] = True
        elif category == "I" and atom.get("required"):
            i_seen += 1
            atom["prefilter_required"] = bool(atom.get("prefilter_required")) and i_seen <= 2
        else:
            atom["prefilter_required"] = False

    contract["required_atoms"] = repaired
    return contract


def _fallback_contract(question: str) -> Dict[str, Any]:
    pico = generate_api_queries_with_llm(question)
    atoms: List[Dict[str, Any]] = []
    idx = 1
    for category in ("P", "I", "O", "C"):
        for term in pico.get(category, []) if isinstance(pico, dict) else []:
            clean = _normalise_term(term)
            if not clean:
                continue
            atoms.append({
                "id": f"{category}{idx}",
                "category": category,
                "label": clean,
                "required": category in {"P", "I", "O", "C"},
                "must_be_explicit": True,
                "prefilter_required": category in {"P", "I"},
                "query_terms": [clean],
                "evidence_terms": [clean],
                "terms": [clean],
                "not_acceptable": [],
            })
            idx += 1

    queries = [
        query
        for query in (pico.get("semantic_queries", []) if isinstance(pico, dict) else [])
        if isinstance(query, str) and query.strip()
    ]
    return _repair_contract({
        "version": CONTRACT_CACHE_VERSION,
        "research_question": question,
        "required_atoms": atoms,
        "query_plan": {
            "direct": queries[:6],
            "focused": queries[6:18],
            "scout": [],
        },
        "exclusion_hints": [],
        "background_hints": [],
        "_validation_status": "fallback_from_pico",
    })


def generate_eligibility_contract(
    question: str,
    inclusion_criteria: str = "",
    exclusion_criteria: str = "",
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Generate a question-specific eligibility contract."""
    if not bool(getattr(config, "ELIGIBILITY_CONTRACT_ENABLED", True)):
        return _fallback_contract(question)

    path = _cache_path(question, inclusion_criteria, exclusion_criteria)
    if use_cache and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                cached = json.load(handle)
            if cached.get("version") == CONTRACT_CACHE_VERSION and _validate_contract(cached):
                logger.info("[EligibilityContract] Validated cache loaded")
                return cached
        except Exception:
            logger.warning("[EligibilityContract] Ignoring unreadable cache")

    prompt = f"""You are a senior systematic-review methodologist and search strategist.
Create a domain-agnostic eligibility contract for this review question.

Research question:
"{question}"

Inclusion criteria:
{inclusion_criteria or "[none provided]"}

Exclusion criteria:
{exclusion_criteria or "[none provided]"}

Rules:
- Split composite requirements into independent atoms.
- If the intervention contains multiple capabilities joined by "and", "with",
  parentheses, or commas, create separate required I atoms for each capability.
- CRITICAL RULE FOR required=true: Mark an atom required=true ONLY when it is explicitly and strictly demanded by the Inclusion Criteria text. If a concept is mentioned in the Research Question but is NOT explicitly demanded in the Inclusion Criteria text, it MUST be marked required=false. The Inclusion Criteria are the ultimate operational rules for what is included. For example, if the question mentions a broad context (e.g. "software testing" or "clinical trials") but the inclusion criteria only ask for a specific subset (e.g. "software defect prediction" or "breast cancer trials"), do not mark the broad context itself as a separate required atom (mark it required=false or omit it) as this would falsely exclude papers that study the specific subset but do not explicitly mention the broad context name.
- If no Inclusion Criteria are provided (e.g., in the initial search phase), mark ONLY the main Population and the main Intervention from the Research Question as required=true (e.g., if the question is "AI in software testing automation", require "Software Testing" and "AI", but mark secondary outcomes/contexts like "automation" or "efficiency" as required=false) to ensure high recall during database retrieval.
- Broaden intervention and technology terms: For any required technology, intervention, or methodology atoms, both query_terms and evidence_terms must include general terms of the field as well as common specific synonyms, sub-techniques, and related methods. For example, if the intervention is AI/ML, include general terms ("machine learning", "artificial intelligence") and specific common families ("neural network", "ensemble", "random forest", "support vector machine", "SVM", "classification", "clustering", "GNN", "extreme learning machine", "optimization"); if the intervention is a medical treatment, include general terms and common specific drug names, subclasses, or delivery methods, so that articles studying specific instances or variants of the intervention are not missed.
- Do not create required atoms for publication year, publication date, or time windows; those are handled by metadata filters before screening.
- If the question asks which techniques, models, or treatments are effective, do not make individual example techniques required unless the criteria explicitly require exactly that specific technique.
- Mark prefilter_required=true only for population and the central intervention
  identity atoms that should be visible in title/abstract search.
- For prefilter_required identity atoms, query_terms must include common
  title/abstract aliases that preserve the same concrete requirement, including
  wording used by adjacent literatures. Do not use single-word generic terms
  unless the term is itself a standard controlled expression.
- evidence_terms may include broader textual signals than query_terms, but each
  term must still support the same atom during screening.
- Keep population, intervention, outcome, comparator, context, and study design in their own roles.
- Do not replace a specific required atom with a broader neighboring topic.
- query_plan.direct must target direct evidence and combine population plus intervention.
- query_plan.focused may be broader but still combines population plus intervention.
- query_plan.scout is only for snowballing/background discovery and may omit
  one required atom to improve recall.
- Use academic title/abstract search wording, not unnatural literal compounds.

Return ONLY raw JSON:
{{
  "required_atoms": [
    {{
      "id": "short_id",
      "category": "P|I|O|C|CTX|SD",
      "label": "specific requirement",
      "required": true,
      "must_be_explicit": true,
      "prefilter_required": true,
      "query_terms": ["3-10 search terms"],
      "evidence_terms": ["2-8 evidence terms"],
      "not_acceptable": ["broader substitutes that must not satisfy this atom"]
    }}
  ],
  "query_plan": {{
    "direct": ["up to 10 high-precision queries"],
    "focused": ["up to 14 focused recall queries"],
    "scout": ["up to 8 snowballing/background queries"]
  }},
  "exclusion_hints": ["generic reasons likely to exclude papers"],
  "background_hints": ["near-miss themes useful only for theory/background"]
}}"""

    try:
        response = generate_text_with_ollama_model(
            instruction=prompt,
            model_name=getattr(config, "OLLAMA_MODEL_PLANNER", ""),
            max_tokens=2400,
            system_prompt="You output strict JSON for systematic-review search planning.",
        )
        parsed = json.loads(_clean_json_response(response))
        contract = _normalise_contract(parsed, question)
        if _validate_contract(contract):
            try:
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(contract, handle, ensure_ascii=False, indent=2)
            except Exception:
                logger.warning("[EligibilityContract] Could not write cache")
            logger.info(
                "[EligibilityContract] %d atoms | direct=%d focused=%d scout=%d",
                len(contract.get("required_atoms", [])),
                len(contract.get("query_plan", {}).get("direct", [])),
                len(contract.get("query_plan", {}).get("focused", [])),
                len(contract.get("query_plan", {}).get("scout", [])),
            )
            return contract
        logger.warning("[EligibilityContract] LLM contract rejected; using fallback")
    except Exception as exc:
        logger.warning("[EligibilityContract] Generation failed: %s", exc)

    return _fallback_contract(question)


def merge_contract_into_pico(pico: Dict[str, Any], contract: Dict[str, Any]) -> Dict[str, Any]:
    """Add contract atoms and planned queries to a PICO-like framework."""
    merged: Dict[str, Any] = {
        key: list(value) if isinstance(value, list) else value
        for key, value in (pico or {}).items()
    }
    for key in ("P", "I", "O", "C"):
        merged.setdefault(key, [])

    category_map = {"P": "P", "I": "I", "O": "O", "C": "C", "CTX": "C"}
    for atom in contract.get("required_atoms", []):
        if not isinstance(atom, dict):
            continue
        if _is_temporal_metadata_atom(atom):
            continue
        target = category_map.get(str(atom.get("category") or "").upper())
        if not target:
            continue
        query_terms = atom.get("query_terms") or []
        terms = _unique_terms([*query_terms, atom.get("label")], limit=5)
        merged[target] = _unique_terms([*merged.get(target, []), *terms], limit=12)

    plan = contract.get("query_plan") if isinstance(contract.get("query_plan"), dict) else {}
    direct_queries = _unique_queries(plan.get("direct") or [], limit=12)
    focused_queries = _unique_queries(plan.get("focused") or [], limit=18)
    scout_queries = _unique_queries(plan.get("scout") or [], limit=12)
    direct_queries, focused_queries, validation = _validate_contract_queries(
        contract,
        direct_queries,
        focused_queries,
    )
    logger.info(
        "[ContractQueries] direct=%s/%s focused=%s/%s repair=%s+%s rejected=%s",
        validation.get("direct_valid", 0),
        validation.get("direct_input", 0),
        validation.get("focused_valid", 0),
        validation.get("focused_input", 0),
        validation.get("direct_repair", 0),
        validation.get("focused_repair", 0),
        len(validation.get("rejected", []) or []),
    )
    _p_atoms, central_i, technical_i = _required_atoms_by_role(contract)
    planned_queries = _unique_queries([
        *direct_queries,
        *focused_queries,
    ], limit=24)
    merged["semantic_queries"] = _unique_queries([
        *planned_queries,
        *(merged.get("semantic_queries") or []),
    ], limit=30)
    merged["_semantic_scout_queries"] = scout_queries
    merged["_semantic_query_tier_overrides"] = {
        **{_query_override_key(query): "exact" for query in direct_queries},
        **{_query_override_key(query): "focused" for query in focused_queries},
        **{_query_override_key(query): "scout" for query in scout_queries},
    }
    merged["_contract_technical_i_groups"] = [
        _atom_query_terms(atom, limit=12)
        for atom in technical_i
        if _atom_query_terms(atom, limit=12)
    ]
    merged["_contract_central_i_groups"] = [
        _atom_query_terms(atom, limit=12)
        for atom in central_i
        if _atom_query_terms(atom, limit=12)
    ]
    merged["_contract_query_validation"] = validation
    return merged


def contract_to_synonym_payload(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a contract into the atom_groups format used by ConceptFilter."""
    synonyms: Dict[str, List[str]] = {}
    categories: Dict[str, str] = {}
    atom_groups: List[Dict[str, Any]] = []
    flat_terms: List[str] = []
    category_map = {"P": "P", "I": "I", "O": "O", "C": "C", "CTX": "C"}
    p_atoms, central_i, technical_i = _required_atoms_by_role(contract)
    central_ids = {str(atom.get("id") or atom.get("label") or "") for atom in central_i}
    technical_ids = {str(atom.get("id") or atom.get("label") or "") for atom in technical_i}

    for atom in contract.get("required_atoms", []):
        if not isinstance(atom, dict):
            continue
        if _is_temporal_metadata_atom(atom):
            continue
        raw_category = str(atom.get("category") or "").upper()
        category = category_map.get(raw_category)
        if not category:
            continue
        terms = _unique_terms([atom.get("label"), *(atom.get("evidence_terms") or []), *(atom.get("query_terms") or [])], limit=12)
        terms = _expand_recall_aliases_for_atom(atom, terms)
        if not terms:
            continue
        concept = terms[0]
        synonyms[concept] = terms[1:]
        categories[concept] = category
        flat_terms.extend(terms)
        prefilter_required = (
            bool(atom.get("prefilter_required"))
            if "prefilter_required" in atom
            else bool(atom.get("required")) and raw_category in {"P", "I"}
        )
        atom_groups.append({
            "id": atom.get("id") or concept,
            "name": atom.get("label") or concept,
            "category": category,
            "required": prefilter_required,
            "role": (
                "central_intervention"
                if str(atom.get("id") or atom.get("label") or "") in central_ids
                else "technical_intervention"
                if str(atom.get("id") or atom.get("label") or "") in technical_ids
                else "population"
                if raw_category == "P"
                else raw_category.lower()
            ),
            "terms": terms,
        })

    plan = contract.get("query_plan") if isinstance(contract.get("query_plan"), dict) else {}
    return {
        "synonyms": synonyms,
        "categories": categories,
        "atom_groups": atom_groups,
        "flat_terms": _unique_terms(flat_terms, limit=80),
        "expanded_queries": _unique_queries([
            *(plan.get("direct") or []),
            *(plan.get("focused") or []),
        ], limit=18),
        "prefilter_rules": {
            "population_atoms": [atom.get("label") for atom in p_atoms if atom.get("label")],
            "central_intervention_atoms": [atom.get("label") for atom in central_i if atom.get("label")],
            "technical_intervention_atoms": [atom.get("label") for atom in technical_i if atom.get("label")],
            "min_technical_intervention_atoms": 1 if technical_i else 0,
        },
        "_validation_status": "eligibility_contract",
    }


def _expand_recall_aliases_for_atom(atom: Dict[str, Any], terms: List[str]) -> List[str]:
    text = " ".join([str(atom.get("label") or ""), *terms]).lower()
    aliases: List[str] = []
    if re.search(r"\b(machine learning|deep learning|aprendizaje autom[aá]tico|ml|dl)\b", text):
        aliases.extend([
            "machine learning",
            "deep learning",
            "neural network",
            "neural networks",
            "artificial neural network",
            "classification algorithm",
            "classifier",
            "supervised learning",
            "ensemble learning",
            "random forest",
            "support vector machine",
            "decision tree",
            "extreme learning machine",
            "graph neural network",
            "codebert",
            "lstm",
            "bilstm",
            "transfer learning",
            "reinforcement learning",
            "feature selection",
        ])
    if re.search(r"\b(software defect|software fault|software bug|defect prediction|fault prediction|bug prediction)\b", text):
        aliases.extend([
            "software defect prediction",
            "software fault prediction",
            "software bug prediction",
            "defect prediction",
            "fault prediction",
            "bug prediction",
            "defect density prediction",
            "cross-project defect prediction",
            "just-in-time software defect prediction",
            "source code defect prediction",
            "defect-prone modules",
            "fault-proneness prediction",
        ])
    return _unique_terms([*terms, *aliases], limit=30)


def _term_variants(term: str) -> List[str]:
    clean = re.sub(r"\s+", " ", str(term or "").strip().lower())
    if not clean:
        return []
    variants = {clean, clean.replace("-", " "), clean.replace(" ", "-")}
    return sorted(variants, key=len, reverse=True)


def _contains_term(text: str, term: str) -> bool:
    term = str(term or "").strip().lower()
    if not term:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _atom_terms(atom: Dict[str, Any]) -> List[str]:
    return _unique_terms([
        atom.get("label"),
        *(atom.get("evidence_terms") or []),
        *(atom.get("query_terms") or []),
        *(atom.get("terms") or []),
    ], limit=16)


def compute_atom_coverage(article: Dict[str, Any], contract: Dict[str, Any]) -> Dict[str, Any]:
    """Lexically estimate atom coverage for ranking and audit, not final inclusion."""
    title = str(article.get("title") or "")
    abstract = str(article.get("abstract") or "")
    text = f"{title}\n{abstract}".lower()
    required_atoms = [
        atom for atom in contract.get("required_atoms", [])
        if isinstance(atom, dict) and bool(atom.get("required")) and not _is_temporal_metadata_atom(atom)
    ]
    matched: List[str] = []
    missing: List[str] = []
    by_category: Dict[str, int] = {}

    for atom in required_atoms:
        atom_label = str(atom.get("label") or atom.get("id") or "")
        terms = [variant for term in _atom_terms(atom) for variant in _term_variants(term)]
        is_match = any(_contains_term(text, term) for term in terms)
        if is_match:
            matched.append(atom_label)
            category = str(atom.get("category") or "").upper()
            by_category[category] = by_category.get(category, 0) + 1
        else:
            missing.append(atom_label)

    total = len(required_atoms)
    score = (len(matched) / total) if total else 0.0
    return {
        "score": round(score, 4),
        "matched": matched,
        "missing": missing,
        "required_total": total,
        "required_matched": len(matched),
        "by_category": by_category,
    }


def rank_articles_by_contract(
    articles: List[Dict[str, Any]],
    contract: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Attach atom coverage and rank articles before expensive PDF/LLM stages."""
    if not articles or not isinstance(contract, dict) or not contract.get("required_atoms"):
        return articles

    enriched: List[Dict[str, Any]] = []
    for article in articles:
        coverage = compute_atom_coverage(article, contract)
        item = article.copy()
        item["_atom_coverage_score"] = coverage["score"]
        item["_atom_coverage_matched"] = coverage["matched"]
        item["_atom_coverage_missing"] = coverage["missing"]
        item["_atom_coverage_required_total"] = coverage["required_total"]
        item["_atom_coverage_required_matched"] = coverage["required_matched"]
        enriched.append(item)

    def _rank_key(article: Dict[str, Any]) -> Tuple[float, float, int]:
        return (
            float(article.get("_atom_coverage_score") or 0.0),
            float(article.get("relevance_score") or 0.0),
            int(article.get("citations") or 0),
        )

    ranked = sorted(enriched, key=_rank_key, reverse=True)
    logger.info("[AtomCoverageRanker] Ranked %d articles by eligibility atoms", len(ranked))
    return ranked


def required_atom_labels(contract: Dict[str, Any]) -> List[str]:
    return [
        str(atom.get("label") or atom.get("id") or "")
        for atom in contract.get("required_atoms", [])
        if isinstance(atom, dict) and bool(atom.get("required")) and not _is_temporal_metadata_atom(atom)
    ]
