"""Adaptive, domain-agnostic retrieval helpers.

The functions in this module improve recall without changing screening rules:
they mine literal vocabulary from retrieved abstracts and use citation graph
neighbors only as additional candidates for the existing strict pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

import config
from app.llm.ai_model import generate_text_with_ollama_model

logger = logging.getLogger(__name__)

_CATEGORY_TO_FRAMEWORK = {"P": "P", "I": "I", "O": "O", "C": "C", "CTX": "C"}
_TOKEN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "based", "be", "by", "for", "from",
    "in", "into", "is", "of", "on", "or", "the", "to", "using", "with",
    "within",
}


@dataclass
class AdaptiveLexiconReport:
    observed_terms: int = 0
    accepted_terms: Dict[str, List[str]] = field(default_factory=dict)
    rejected_terms: Dict[str, str] = field(default_factory=dict)
    generated_queries: List[str] = field(default_factory=list)
    llm_used: bool = False


@dataclass
class GraphRecoveryReport:
    attempted: bool = False
    seed_count: int = 0
    recovered_count: int = 0
    reason: str = ""


def _normalise_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _tokenise(value: object) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(token) > 1 and token not in _TOKEN_STOPWORDS
    ]


def _article_text(article: Dict[str, Any]) -> str:
    return _normalise_text(f"{article.get('title') or ''} {article.get('abstract') or ''}")


def _term_pattern(term: str) -> str:
    clean = re.escape(str(term or "").strip().lower())
    return rf"(?<![a-z0-9]){clean}(?![a-z0-9])"


def _contains_literal(corpus_text: str, term: str) -> bool:
    clean = _normalise_text(term).lower()
    if not clean:
        return False
    return re.search(_term_pattern(clean), corpus_text) is not None


def _unique(values: Iterable[object], limit: int = 20) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        clean = _normalise_text(value)
        key = clean.lower()
        if not clean or key in seen:
            continue
        seen.add(key)
        out.append(clean)
        if len(out) >= limit:
            break
    return out


def _bounded(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _low_signal_term(term: str) -> bool:
    tokens = _tokenise(term)
    if not tokens:
        return True
    if len(tokens) == 1 and len(tokens[0]) < 4 and not _looks_like_compact_identifier(term):
        return True
    broad = set(getattr(config, "BROAD_ENRICHMENT_STOPWORDS", set()))
    return " ".join(tokens) in broad or all(token in broad for token in tokens)


def _looks_like_compact_identifier(term: str) -> bool:
    clean = _normalise_text(term).strip()
    compact = re.sub(r"[^A-Za-z0-9]", "", clean)
    if any(char.isdigit() for char in clean):
        return True
    if any(marker in clean for marker in ("-", "/", "+")):
        return True
    return clean.isupper() and 2 <= len(compact) <= 12


def _empirical_term_is_query_specific(term: str) -> bool:
    tokens = _tokenise(term)
    if len(tokens) >= 2:
        return True
    return bool(tokens) and _looks_like_compact_identifier(term)


def _extract_observed_terms(articles: List[Dict[str, Any]]) -> List[str]:
    max_abstracts = _bounded(
        getattr(config, "ADAPTIVE_LEXICON_MAX_ABSTRACTS", 80),
        default=80,
        minimum=10,
        maximum=250,
    )
    max_terms = _bounded(
        getattr(config, "ADAPTIVE_LEXICON_MAX_TERMS", 160),
        default=160,
        minimum=30,
        maximum=400,
    )
    texts = [
        _article_text(article)
        for article in articles[:max_abstracts]
        if _article_text(article)
    ]
    if not texts:
        return []

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 4),
            max_features=max_terms,
            min_df=1,
            stop_words="english",
        )
        matrix = vectorizer.fit_transform(texts)
        names = vectorizer.get_feature_names_out()
        scores = matrix.mean(axis=0).A1
        ranked = [
            term
            for term, _score in sorted(zip(names, scores), key=lambda item: item[1], reverse=True)
            if not _low_signal_term(term)
        ]
        return _unique(ranked, limit=max_terms)
    except Exception as exc:
        logger.warning("[AdaptiveLexicon] TF-IDF failed: %s", exc)
        return []


def _contract_atoms(contract: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        atom
        for atom in contract.get("required_atoms", [])
        if isinstance(atom, dict) and atom.get("required")
    ]


def _fallback_term_mapping(observed_terms: List[str], atoms: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    mapped: Dict[str, List[str]] = {}
    for atom in atoms:
        atom_id = str(atom.get("id") or atom.get("label") or "")
        atom_tokens = set(_tokenise(" ".join([
            str(atom.get("label") or ""),
            " ".join(atom.get("query_terms") or []),
            " ".join(atom.get("evidence_terms") or []),
        ])))
        if not atom_id or not atom_tokens:
            continue
        matches = []
        for term in observed_terms:
            term_tokens = set(_tokenise(term))
            if term_tokens and atom_tokens.intersection(term_tokens):
                matches.append(term)
        if matches:
            mapped[atom_id] = _unique(matches, limit=6)
    return mapped


def _parse_llm_mapping(content: str) -> Dict[str, List[str]]:
    text = str(content or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    parsed = json.loads(text)
    raw_atoms = parsed.get("atoms") if isinstance(parsed, dict) else None
    if not isinstance(raw_atoms, list):
        return {}
    mapping: Dict[str, List[str]] = {}
    for item in raw_atoms:
        if not isinstance(item, dict):
            continue
        atom_id = str(item.get("id") or "").strip()
        terms = item.get("terms") or []
        if atom_id and isinstance(terms, list):
            mapping[atom_id] = _unique(terms, limit=8)
    return mapping


def _llm_term_mapping(
    question: str,
    observed_terms: List[str],
    atoms: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    atom_block = [
        {
            "id": atom.get("id"),
            "category": atom.get("category"),
            "label": atom.get("label"),
            "query_terms": atom.get("query_terms") or [],
        }
        for atom in atoms
    ]
    prompt = f"""You are improving search recall for a systematic review.

Research question:
{question}

Required atoms:
{json.dumps(atom_block, ensure_ascii=False)}

Observed n-grams from real titles/abstracts:
{json.dumps(observed_terms, ensure_ascii=False)}

Task:
Map observed n-grams to the atom they can help retrieve.

Rules:
- Every returned term must be copied exactly from the observed n-gram list.
- Do not invent synonyms.
- Do not include generic terms that do not identify the atom.
- It is acceptable to return an empty list for an atom.
- Output only JSON.

Output:
{{"atoms":[{{"id":"atom id","terms":["literal observed term"]}}]}}"""
    response = generate_text_with_ollama_model(
        instruction=prompt,
        model_name=getattr(config, "OLLAMA_MODEL_PLANNER", ""),
        max_tokens=1400,
        system_prompt="You output strict JSON for empirical query expansion.",
    )
    return _parse_llm_mapping(response)


def mine_empirical_lexicon(
    question: str,
    articles: List[Dict[str, Any]],
    contract: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, List[str]], AdaptiveLexiconReport]:
    """Extract literal corpus terms mapped to contract atoms."""
    report = AdaptiveLexiconReport()
    if not articles or not isinstance(contract, dict):
        return {}, report

    atoms = _contract_atoms(contract)
    if not atoms:
        return {}, report

    observed_terms = _extract_observed_terms(articles)
    report.observed_terms = len(observed_terms)
    if not observed_terms:
        return {}, report

    mapping: Dict[str, List[str]] = {}
    if bool(getattr(config, "ADAPTIVE_LEXICON_LLM_ENABLED", True)):
        try:
            mapping = _llm_term_mapping(question, observed_terms, atoms)
            report.llm_used = True
        except Exception as exc:
            logger.warning("[AdaptiveLexicon] LLM mapping failed: %s", exc)

    if not mapping:
        mapping = _fallback_term_mapping(observed_terms, atoms)

    observed_lookup = {term.lower(): term for term in observed_terms}
    corpus_text = "\n".join(_article_text(article).lower() for article in articles)
    accepted: Dict[str, List[str]] = {}
    known_atom_ids = {str(atom.get("id") or atom.get("label") or "") for atom in atoms}
    per_atom = _bounded(
        getattr(config, "ADAPTIVE_LEXICON_TERMS_PER_ATOM", 5),
        default=5,
        minimum=1,
        maximum=12,
    )

    for atom_id, terms in mapping.items():
        if atom_id not in known_atom_ids:
            continue
        for term in terms:
            clean = _normalise_text(term)
            key = clean.lower()
            if not clean or _low_signal_term(clean):
                report.rejected_terms[clean] = "low_signal"
                continue
            if not _empirical_term_is_query_specific(clean):
                report.rejected_terms[clean] = "low_specificity"
                continue
            if key not in observed_lookup and not _contains_literal(corpus_text, clean):
                report.rejected_terms[clean] = "not_observed"
                continue
            accepted.setdefault(atom_id, [])
            if clean not in accepted[atom_id] and len(accepted[atom_id]) < per_atom:
                accepted[atom_id].append(observed_lookup.get(key, clean))

    report.accepted_terms = accepted
    return accepted, report


def _terms_for_atom(atom: Dict[str, Any], empirical_terms: Dict[str, List[str]]) -> List[str]:
    atom_id = str(atom.get("id") or atom.get("label") or "")
    return _unique([
        *(empirical_terms.get(atom_id) or []),
        *(atom.get("query_terms") or []),
        atom.get("label"),
    ], limit=6)


def _query_key(query: str) -> str:
    return " ".join(_tokenise(query))


def _build_empirical_queries(
    contract: Dict[str, Any],
    empirical_terms: Dict[str, List[str]],
) -> Tuple[List[str], Dict[str, str]]:
    atoms = _contract_atoms(contract)
    p_atoms = [atom for atom in atoms if str(atom.get("category") or "").upper() == "P"]
    i_atoms = [atom for atom in atoms if str(atom.get("category") or "").upper() == "I"]
    central_i = [atom for atom in i_atoms if bool(atom.get("prefilter_required"))] or i_atoms[:1]
    technical_i = [atom for atom in i_atoms if atom not in central_i]
    if not p_atoms or not central_i:
        return [], {}

    max_queries = _bounded(
        getattr(config, "ADAPTIVE_LEXICON_MAX_QUERIES", 12),
        default=12,
        minimum=2,
        maximum=30,
    )
    queries: List[str] = []
    tiers: Dict[str, str] = {}

    p_terms = _terms_for_atom(p_atoms[0], empirical_terms)[:3]
    central_terms = [
        term
        for atom in central_i
        for term in _terms_for_atom(atom, empirical_terms)[:3]
    ][:4]
    technical_terms = [
        _terms_for_atom(atom, empirical_terms)[:3]
        for atom in technical_i
    ]
    technical_terms = [terms for terms in technical_terms if terms]

    for p_term in p_terms:
        for central_term in central_terms:
            if not p_term or not central_term:
                continue
            if technical_terms:
                for tech_terms in technical_terms:
                    for tech_term in tech_terms:
                        query = _normalise_text(f"{p_term} {central_term} {tech_term}")
                        queries.append(query)
                        tiers[_query_key(query)] = "focused"
                        if len(queries) >= max_queries:
                            return _unique(queries, limit=max_queries), tiers
            query = _normalise_text(f"{p_term} {central_term}")
            queries.append(query)
            tiers[_query_key(query)] = "core"
            if len(queries) >= max_queries:
                return _unique(queries, limit=max_queries), tiers

    return _unique(queries, limit=max_queries), tiers


def augment_framework_with_empirical_lexicon(
    framework: Dict[str, Any],
    question: str,
    articles: List[Dict[str, Any]],
    contract: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Add literal, observed search vocabulary to a PICO-like framework."""
    if not bool(getattr(config, "ADAPTIVE_LEXICAL_EXPANSION_ENABLED", True)):
        return framework, asdict(AdaptiveLexiconReport())

    empirical_terms, report = mine_empirical_lexicon(question, articles, contract)
    if not empirical_terms or not isinstance(contract, dict):
        return framework, asdict(report)

    augmented = {
        key: list(value) if isinstance(value, list) else value
        for key, value in framework.items()
    }
    for atom in _contract_atoms(contract):
        atom_id = str(atom.get("id") or atom.get("label") or "")
        terms = empirical_terms.get(atom_id) or []
        target = _CATEGORY_TO_FRAMEWORK.get(str(atom.get("category") or "").upper())
        if target and terms:
            augmented[target] = _unique([*(augmented.get(target) or []), *terms], limit=18)

    queries, tier_overrides = _build_empirical_queries(contract, empirical_terms)
    if queries:
        augmented["semantic_queries"] = _unique([
            *(augmented.get("semantic_queries") or []),
            *queries,
        ], limit=42)
        existing_overrides = augmented.get("_semantic_query_tier_overrides") or {}
        if not isinstance(existing_overrides, dict):
            existing_overrides = {}
        augmented["_semantic_query_tier_overrides"] = {
            **existing_overrides,
            **tier_overrides,
        }
        report.generated_queries = queries

    logger.info(
        "[AdaptiveLexicon] observed=%d accepted_atoms=%d queries=%d",
        report.observed_terms,
        len(report.accepted_terms),
        len(report.generated_queries),
    )
    return augmented, asdict(report)


def recover_articles_from_near_misses(
    near_misses: List[Dict[str, Any]],
    question: str,
    contract: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run bounded citation-graph recovery from the strongest near-misses."""
    report = GraphRecoveryReport(attempted=True)
    if not bool(getattr(config, "ADAPTIVE_GRAPH_RECOVERY_ENABLED", True)):
        report.reason = "disabled"
        return [], asdict(report)
    if not near_misses:
        report.reason = "no_near_misses"
        return [], asdict(report)

    try:
        from app.core.search_engine import search_semantic_scholar_graph_neighbors
        from app.domain.eligibility_contract import rank_articles_by_contract

        ranked = rank_articles_by_contract(near_misses, contract or {})
        max_seeds = _bounded(
            getattr(config, "ADAPTIVE_GRAPH_RECOVERY_MAX_SEEDS", 6),
            default=6,
            minimum=1,
            maximum=20,
        )
        seeds = ranked[:max_seeds]
        report.seed_count = len(seeds)
        recovered = search_semantic_scholar_graph_neighbors(
            seeds,
            question=question,
            max_articles=_bounded(
                getattr(config, "ADAPTIVE_GRAPH_RECOVERY_MAX_ARTICLES", 120),
                default=120,
                minimum=10,
                maximum=500,
            ),
        )
        report.recovered_count = len(recovered)
        if not recovered:
            report.reason = "no_graph_neighbors"
        return recovered, asdict(report)
    except Exception as exc:
        report.reason = f"error: {exc}"
        logger.warning("[AdaptiveGraph] Recovery failed: %s", exc)
        return [], asdict(report)
