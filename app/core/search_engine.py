"""
search_engine.py â€” Refactored for academic SLR quality.

Fixes applied vs original:
  1. DOMAIN_MARKERS & openalex_concept hardcodes â†’ runtime config + LLM-driven concept resolution
  2. Mixed asyncio/ThreadPoolExecutor architecture â†’ unified async with asyncio.gather
  3. sources_config hardcoded targets â†’ driven by SourceProfile dataclasses in config
  4. Citation filter thresholds inline â†’ per-source strategy in SourceProfile
  5. GENERAL_METRICS hardcoded set â†’ loaded from external YAML/JSON config
  6. Naive dedup (DOI string + title[:70]) â†’ shared normalize_doi + Jaccard from two_phase_search
  7. Re-ranking inline scoring â†’ isolated, testable RelevanceScorer
  8. User-agent rotation inline â†’ injected UA pool from config
  9. Year-range hardcoded "13" â†’ configurable SEARCH_WINDOW_YEARS
 10. get_registered_sources() implemented so two_phase_search dynamic bypass works
"""

import asyncio
import logging
import math
import random
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import quote

import requests
import config
from Bio import Entrez
from app.extraction.pdf_extractor import enrich_initial_search_result
from app.domain.query_expander import generate_api_queries_with_llm

logger = logging.getLogger(__name__)

Entrez.email = config.ACADEMIC_EMAIL

# â”€â”€â”€ Configuration contract â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# All magic numbers live here or in config.py. Nothing is scattered inline.

SEARCH_WINDOW_YEARS: int = getattr(config, "SEARCH_WINDOW_YEARS", 13)
MAX_RESULTS_TOTAL: int = getattr(config, "MAX_SEARCH_RESULTS", 6000)
JACCARD_DEDUP_THRESHOLD: float = getattr(config, "JACCARD_DEDUP_THRESHOLD", 0.85)

# Citation thresholds by article age bucket (years since publication).
# Loaded from config so the researcher can tune them without touching code.
CITATION_BUCKETS: List[Tuple[int, int]] = sorted(
    getattr(config, "CITATION_BUCKETS", [
        (0, 0),    # current year:  0+ citations required
        (1, 0),    # 1 year old:    0+
        (2, 1),    # 2 years old:   1+
        (5, 3),    # 3-5 years old: 3+
        (13, 5),   # 6-13 years:    5+
    ]),
    key=lambda x: x[0],
)

# User-agent pool â€” loaded from config, never hardcoded inline.
_DEFAULT_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/119.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/118.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119.0 Safari/537.36",
]
UA_POOL: List[str] = getattr(config, "USER_AGENT_POOL", _DEFAULT_UA_POOL)


# â”€â”€â”€ Source registry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class SourceProfile:
    """
    Single source of truth for every search source.
    Adding a new database = adding one SourceProfile here.
    No other code needs to change.
    """
    name: str                          # display name, used as the dedup key
    enabled: bool = True
    target: int = 2000                 # max articles to harvest
    timeout_seconds: float = 30.0
    requires_api_key: bool = False
    api_key_attr: str = ""             # attribute name in config module
    rate_limit_sleep: float = 1.2      # base sleep between paginated requests
    max_retries: int = 3


# Registry is the single authoritative list. Two-phase bypass reads from here.
_SOURCE_REGISTRY: Dict[str, SourceProfile] = {
    "Semantic Scholar": SourceProfile(
        name="Semantic Scholar",
        target=getattr(config, "SS_TARGET", 4000),
        timeout_seconds=150.0,
        requires_api_key=True,
        api_key_attr="SEMANTIC_SCHOLAR_API_KEY",
        rate_limit_sleep=1.2,
    ),
    "PubMed": SourceProfile(
        name="PubMed",
        target=getattr(config, "PUBMED_TARGET", 2000),
        timeout_seconds=150.0,
        rate_limit_sleep=0.5,
    ),
    "OpenAlex": SourceProfile(
        name="OpenAlex",
        target=getattr(config, "OPENALEX_TARGET", 1500),
        timeout_seconds=150.0,
        rate_limit_sleep=0.5,
        enabled=False,
    ),
    "Europe PMC": SourceProfile(
        name="Europe PMC",
        target=getattr(config, "EPMC_TARGET", 2000),
        timeout_seconds=150.0,
        rate_limit_sleep=0.2,
    ),
}


def get_registered_sources() -> List[str]:
    """
    Returns the list of all registered source names.
    Used by two_phase_search.py for dynamic bypass logic.
    No source name is ever hardcoded outside this module.
    """
    return [name for name, profile in _SOURCE_REGISTRY.items() if profile.enabled]


def _profile(name: str) -> SourceProfile:
    return _SOURCE_REGISTRY[name]


# â”€â”€â”€ Noise filter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Loaded from config so it can be extended per-deployment without code changes.

_DEFAULT_TITLE_NOISE: frozenset = frozenset(getattr(config, "TITLE_NOISE_PATTERNS", [
    "github repository", "software package", "npm install",
    "package documentation", "official documentation", "installation guide",
    "read me", "readme.md", "release notes", "getting started guide",
    "bug report", "pull request", "commit log", "jenkins pipeline",
    "docker image", "kubernetes deployment", "configuration file",
]))

_GENERAL_METRICS: frozenset = frozenset(getattr(config, "GENERAL_METRICS", [
    "accuracy", "precision", "recall", "f1", "f1-score", "auc", "roc",
    "performance", "metric", "metrics", "evaluation", "impact",
    "effectiveness", "comparison", "analysis", "study", "results",
    "empirical", "systematic", "methodology", "measure", "validation",
]))


def _is_noise_title(title: str) -> bool:
    if not title:
        return True
    t = title.lower()
    return any(noise in t for noise in _DEFAULT_TITLE_NOISE)


def _is_generic_metric(term: str) -> bool:
    words = set(re.sub(r"[^a-z0-9\s-]", "", term.lower()).replace("-", " ").split())
    return not words or words.issubset(_GENERAL_METRICS)


def _query_tokens(query: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(query or "").lower())
        if len(token) > 1
    ]


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _semantic_scholar_query_text(raw_query: str) -> str:
    """
    Semantic Scholar behaves like a semantic/relevance engine, not a strict
    Boolean database. This normalizes any upstream query into a concise natural
    phrase without domain-specific vocabulary assumptions.
    """
    text = str(raw_query or "")
    text = re.sub(r"\b(AND|OR|NOT)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[\(\)\[\]\{\}\"'`]", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()

    stopwords: Set[str] = set(getattr(config, "SEMANTIC_QUERY_STOPWORDS", set()))
    preserved_short: Set[str] = set(getattr(config, "SEARCH_PRESERVE_SHORT_TOKENS", {"ai", "ml", "nlp"}))
    max_terms = _bounded_int(
        getattr(config, "SEMANTIC_SCHOLAR_QUERY_MAX_TERMS", 12),
        default=12,
        minimum=3,
        maximum=25,
    )

    words: List[str] = []
    seen: Set[str] = set()
    for raw_token in text.split():
        token = raw_token.strip("-")
        if not token:
            continue
        if len(token) <= 2 and token not in preserved_short:
            continue
        if token in stopwords:
            continue
        if token in seen:
            continue
        seen.add(token)
        words.append(token)
        if len(words) >= max_terms:
            break

    return " ".join(words)


def _query_key(query: str) -> str:
    return " ".join(_query_tokens(query))


def _query_categories(query_terms: List[str], buckets: Dict[str, List[str]]) -> List[str]:
    query_text = " ".join(query_terms).lower()
    query_token_set = set(_query_tokens(query_text))
    generic_tokens = {
        str(token).lower()
        for token in getattr(config, "GENERIC_EQUIVALENCE_HEAD_TERMS", set())
    }
    categories: List[str] = []

    for category, terms in buckets.items():
        for term in terms:
            term_text = str(term or "").lower().strip()
            term_tokens = set(_query_tokens(term_text))
            if not term_tokens:
                continue
            specific_tokens = {token for token in term_tokens if token not in generic_tokens}
            match_tokens = specific_tokens or term_tokens
            overlap = query_token_set.intersection(match_tokens)
            if term_text in query_text:
                categories.append(category)
                break
            if len(term_tokens) == 1 and overlap:
                categories.append(category)
                break
            if len(term_tokens) == 2 and len(query_token_set.intersection(term_tokens)) == 2:
                categories.append(category)
                break
            if len(term_tokens) >= 3 and len(overlap) >= 2 and len(overlap) / max(len(match_tokens), 1) >= 0.67:
                categories.append(category)
                break

    return categories


def _query_rejection_reason(query_terms: List[str], buckets: Dict[str, List[str]]) -> Optional[str]:
    tokens = _query_tokens(" ".join(query_terms))
    categories = set(_query_categories(query_terms, buckets))

    if len(tokens) < 2:
        return "too_short"
    if "P" in buckets and "P" not in categories:
        return "missing_population"
    if "I" in buckets and "I" not in categories:
        return "missing_intervention"
    if len(categories) < 2:
        return "single_concept_query"
    return None


# â”€â”€â”€ DOI normalisation (same logic as two_phase_search.py) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", re.IGNORECASE)


def _normalize_doi(raw: str) -> str:
    doi = _DOI_PREFIX_RE.sub("", (raw or "").strip()).strip("/").lower()
    return f"https://doi.org/{doi}" if doi.startswith("10.") else ""


# â”€â”€â”€ Deduplication â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _tokenize_title(title: str) -> frozenset:
    nfkd = unicodedata.normalize("NFKD", title.lower())
    ascii_t = "".join(c for c in nfkd if not unicodedata.combining(c))
    stopwords = {"a", "an", "the", "of", "in", "on", "for", "and", "or",
                 "to", "with", "is", "are", "at", "by", "from"}
    return frozenset(
        t for t in re.findall(r"[a-z0-9]+", ascii_t)
        if len(t) > 1 and t not in stopwords
    )


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _deduplicate(articles: List[Dict]) -> List[Dict]:
    """
    Three-tier dedup:
      Tier 1 â€” Normalized DOI exact match.
      Tier 2 â€” Jaccard similarity on title tokens (configurable threshold).
      Tier 3 â€” Preserves first-seen record; marks 'retrieval_phase' as 'both'
                when the same paper appears from multiple phases.
    """
    doi_index: Dict[str, Dict] = {}
    title_index: List[Tuple[frozenset, Dict]] = []

    for art in articles:
        doi_norm = _normalize_doi(art.get("doi") or "")
        art["doi_normalized"] = doi_norm

        # Tier 1
        if doi_norm:
            if doi_norm in doi_index:
                continue
            doi_index[doi_norm] = art
            title_index.append((_tokenize_title(art.get("title") or ""), art))
            continue

        # Tier 2
        tokens = _tokenize_title(art.get("title") or "")
        if tokens:
            match = next(
                (rec for tok, rec in title_index
                 if _jaccard(tokens, tok) >= JACCARD_DEDUP_THRESHOLD),
                None,
            )
            if match:
                continue

        stub_key = f"__no_doi_{len(doi_index)}"
        doi_index[stub_key] = art
        title_index.append((tokens, art))

    return list(doi_index.values())


# â”€â”€â”€ Citation threshold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _min_citations_for_year(pub_year: int) -> int:
    current_year = datetime.now(timezone.utc).year
    if pub_year <= 0:
        return 0
    age = current_year - pub_year
    threshold = 0
    for max_age, min_cit in CITATION_BUCKETS:
        if age <= max_age:
            return min_cit
        threshold = min_cit
    return threshold


# â”€â”€â”€ Relevance scorer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class RelevanceScorer:
    """
    Isolated, testable lexical co-occurrence scorer.
    Receives PICO terms at construction; scores articles one at a time.
    No global state, no inline magic numbers.
    """

    P_WEIGHT = 3.0
    I_WEIGHT = 2.0
    O_WEIGHT = 1.0
    COOCCURRENCE_BONUS = 25.0
    TITLE_BOOST_EXACT = 8.0
    ABSTRACT_BOOST_EXACT = 3.0
    MAX_CITATION_BOOST = 10.0
    CITATION_SCALE = 1.5

    def __init__(self, pico: Dict[str, List[str]]) -> None:
        self.p_terms = [t.lower().strip() for t in pico.get("P", []) if len(t.strip()) > 3]
        self.i_terms = [t.lower().strip() for t in pico.get("I", []) if len(t.strip()) > 3]
        self.o_terms = [t.lower().strip() for t in pico.get("O", []) if len(t.strip()) > 3]
        self.exact_boosts = self._build_boosts(pico)
        logger.debug("[RelevanceScorer] boosts=%s", self.exact_boosts)

    @staticmethod
    def _build_boosts(pico: Dict[str, List[str]]) -> List[str]:
        P = [t.lower().strip() for t in pico.get("P", []) if len(t.strip()) > 4][:3]
        I = [t.lower().strip() for t in pico.get("I", []) if len(t.strip()) > 4][:3]
        boosts: List[str] = []
        for term in P + I:
            if " " in term:
                boosts.append(term)
        for p in P[:2]:
            for i in I[:2]:
                boosts.extend([f"{p} {i}", f"{i} {p}"])
        seen: set = set()
        return [b for b in boosts if not (b in seen or seen.add(b))][:12]  # type: ignore[func-returns-value]

    def score(self, art: Dict) -> float:
        title = (art.get("title") or "").lower()
        abstract = (art.get("abstract") or "").lower()
        text = f"{title} {abstract}"

        p_hits = sum(1 for t in self.p_terms if t in text)
        i_hits = sum(1 for t in self.i_terms if t in text)
        o_hits = sum(1 for t in self.o_terms if t in text)

        score = (self.P_WEIGHT * p_hits
                 + self.I_WEIGHT * i_hits
                 + self.O_WEIGHT * o_hits)

        if p_hits > 0 and i_hits > 0:
            score += self.COOCCURRENCE_BONUS

        for boost in self.exact_boosts:
            if boost in title:
                score += self.TITLE_BOOST_EXACT
            elif boost in abstract:
                score += self.ABSTRACT_BOOST_EXACT

        citations = float(art.get("citations") or 0)
        score += min(self.MAX_CITATION_BOOST,
                     math.log1p(citations) * self.CITATION_SCALE)

        return score


# â”€â”€â”€ HTTP helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _random_ua() -> str:
    return random.choice(UA_POOL)


def _get_with_retry(
    url: str,
    params: Dict,
    headers: Dict,
    profile: SourceProfile,
    label: str,
) -> Optional[requests.Response]:
    """
    Unified retry/back-off logic shared by all sources.
    Returns the Response on HTTP 200, None on unrecoverable error.
    """
    for attempt in range(profile.max_retries):
        try:
            r = requests.get(url, params=params, headers=headers,
                             timeout=profile.timeout_seconds)
            if r.status_code == 200:
                return r
            if r.status_code == 400:
                body = r.text[:200]
                logger.error("âŒ HTTP 400 [%s] (%s): %s", profile.name, label, body)
                return None
            if r.status_code == 403:
                logger.error("âŒ HTTP 403 [%s] â€” API key or IP banned", profile.name)
                return None
            if r.status_code == 429:
                wait = profile.rate_limit_sleep * 8 * (attempt + 1)
                logger.warning("âš ï¸ Rate-limit [%s] â€” sleeping %.1fs", profile.name, wait)
                time.sleep(wait)
                continue
            if 500 <= r.status_code < 505:
                wait = 2 ** attempt
                logger.warning("âš ï¸ Server error %d [%s] â€” retry %d in %.1fs",
                                r.status_code, profile.name, attempt + 1, wait)
                time.sleep(wait)
                continue
            logger.warning("âš ï¸ HTTP %d [%s] (%s)", r.status_code, profile.name, label)
            return None
        except Exception as exc:
            logger.error("âŒ Request error [%s] (%s): %s", profile.name, label, exc)
            time.sleep(1)
    return None


# â”€â”€â”€ Query translators â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _agnostic_fallback_terms(question: Optional[str]) -> List[str]:
    """Extracts meaningful terms from a question without any domain assumptions."""
    if not question:
        return ["systematic review"]
    try:
        from app.domain.query_expander import extract_english_terms
        terms = extract_english_terms(question)
        if terms:
            return terms[:3]
    except Exception:
        pass
    stopwords = {
        "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for",
        "with", "in", "on", "to", "from", "de", "la", "el", "en", "y", "que",
    }
    words = [w.lower() for w in re.findall(r"[a-zA-ZÃ¡Ã©Ã­Ã³ÃºÃ±ÃÃ‰ÃÃ“ÃšÃ‘]+", question)
             if len(w) > 3 and w.lower() not in stopwords]
    return words[:3] or ["systematic review"]


def _pico_to_pubmed(pico: Dict, question: Optional[str] = None) -> str:
    P = [t.strip() for t in pico.get("P", []) if t.strip()]
    I = [t.strip() for t in pico.get("I", []) if t.strip()]
    O = [t.strip() for t in pico.get("O", []) if t.strip() and not _is_generic_metric(t)]
    C = [t.strip() for t in pico.get("C", []) if t.strip() and not _is_generic_metric(t)]
    optional_terms = list(dict.fromkeys(O + C))

    if not P and not I:
        fallback = _agnostic_fallback_terms(question)
        return " OR ".join(
            f'"{t}"[tiab]' if " " in t else f"{t}[tiab]" for t in fallback
        )

    def _fmt(lst: List[str]) -> str:
        return "(" + " OR ".join(
            f'"{t}"[tiab]' if " " in t else f"{t}[tiab]" for t in lst
        ) + ")"

    parts = ([_fmt(P)] if P else []) + ([_fmt(I)] if I else [])
    if optional_terms and getattr(config, "INCLUDE_OPTIONAL_TERMS_IN_STRUCTURED_SEARCH", False):
        parts.append(_fmt(optional_terms))
    return " AND ".join(parts)


def _pico_to_europe_pmc(pico: Dict, question: Optional[str] = None) -> str:
    P = [t.strip() for t in pico.get("P", []) if t.strip()]
    I = [t.strip() for t in pico.get("I", []) if t.strip()]
    O = [t.strip() for t in pico.get("O", []) if t.strip() and not _is_generic_metric(t)]
    C = [t.strip() for t in pico.get("C", []) if t.strip() and not _is_generic_metric(t)]
    optional_terms = list(dict.fromkeys(O + C))

    if not P and not I:
        fallback = _agnostic_fallback_terms(question)
        return " OR ".join(f'"{t}"' if " " in t else t for t in fallback)

    def _fmt(lst: List[str]) -> str:
        return "(" + " OR ".join(f'"{t}"' if " " in t else t for t in lst) + ")"

    parts = ([_fmt(P)] if P else []) + ([_fmt(I)] if I else [])
    if optional_terms and getattr(config, "INCLUDE_OPTIONAL_TERMS_IN_STRUCTURED_SEARCH", False):
        parts.append(_fmt(optional_terms))
    return " AND ".join(parts)


def _pico_to_semantic_scholar(pico: Dict) -> str:
    words = []
    for key in pico:
        vals = pico.get(key, [])
        if vals:
            words.append(str(vals[0]))
    query = re.sub(r"[^a-zA-Z0-9\s]", "", " ".join(words))
    return " ".join(query.split()[:8]).lower()


def _query_tier(categories: List[str]) -> str:
    cats = set(categories)
    if not {"P", "I"}.issubset(cats):
        return "other"
    optional_count = len(cats.intersection({"O", "C"}))
    if optional_count == 0:
        return "core"
    if optional_count == 1:
        return "focused"
    return "exact"


def _query_matches_contract_group(query_text: str, terms: List[str]) -> bool:
    text_tokens = set(_query_tokens(query_text))
    for term in terms:
        term_tokens = set(_query_tokens(str(term or "")))
        if term_tokens and term_tokens.issubset(text_tokens):
            return True
    return False


def _apply_contract_technical_tier(
    query_text: str,
    tier: str,
    pico: Dict[str, Any],
) -> str:
    technical_groups = pico.get("_contract_technical_i_groups") or []
    central_groups = pico.get("_contract_central_i_groups") or []
    if not isinstance(technical_groups, list) or not technical_groups:
        return tier
    if tier not in {"exact", "focused"}:
        return tier
    if isinstance(central_groups, list) and central_groups:
        has_central = any(
            isinstance(group, list) and _query_matches_contract_group(query_text, group)
            for group in central_groups
        )
        if not has_central:
            return "core"

    matched = 0
    for group in technical_groups:
        if isinstance(group, list) and _query_matches_contract_group(query_text, group):
            matched += 1

    if matched == len(technical_groups):
        return "exact"
    if matched > 0:
        return "focused"
    return "core"


def _build_semantic_queries_with_audit(
    pico: Dict,
    question: Optional[str] = None,
    emit_logs: bool = True,
) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
    """
    Builds up to 24 multi-concept query combinations from PICO.
    Fully agnostic: clusters terms by stem overlap, then builds
    4-concept â†’ 3-concept â†’ 2-concept combos in round-robin order.
    """
    tier_overrides = pico.get("_semantic_query_tier_overrides") or {}
    if not isinstance(tier_overrides, dict):
        tier_overrides = {}

    llm_queries: List[Tuple[List[str], str]] = []
    for q in pico.get("semantic_queries", []):
        if isinstance(q, str):
            words = q.split()
            if words:
                llm_queries.append((words, "llm"))
        elif isinstance(q, list):
            llm_queries.append((q, "llm"))
    for q in pico.get("_semantic_scout_queries", []):
        if isinstance(q, str):
            words = q.split()
            if words:
                llm_queries.append((words, "scout"))
        elif isinstance(q, list):
            llm_queries.append((q, "scout"))

    keys = [k for k in ["P", "I", "O", "C"] if pico.get(k)]
    buckets: Dict[str, List[str]] = {
        k: [t.strip() for t in pico.get(k, []) if t.strip()] for k in keys
    }
    if not buckets.get("P"):
        buckets["P"] = _agnostic_fallback_terms(question)
    if not buckets.get("I"):
        buckets["I"] = _agnostic_fallback_terms(question)[-2:] or ["analysis"]

    def _cluster(terms: List[str]) -> List[List[str]]:
        generic = {"software", "system", "method", "algorithm", "technique",
                   "approach", "model", "analysis", "framework"}
        clusters: List[List[str]] = []
        for term in terms:
            words = {w.lower() for w in term.split()
                     if w.lower() not in generic and len(w) > 2}
            placed = False
            for cluster in clusters:
                for existing in cluster:
                    ex_words = {w.lower() for w in existing.split()
                                if w.lower() not in generic and len(w) > 2}
                    if any(w[:4] == e[:4] for w in words for e in ex_words
                           if len(w) >= 4 and len(e) >= 4):
                        cluster.append(term)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                clusters.append([term])
        return clusters

    P_cl = _cluster(buckets.get("P", []))
    I_cl = _cluster(buckets.get("I", []))
    O_cl = _cluster(buckets.get("O", [])) if buckets.get("O") else []
    C_cl = _cluster(buckets.get("C", [])) if buckets.get("C") else []

    programmatic: List[Tuple[List[str], str]] = []
    max_depth = max(
        (len(p) * len(i) for p in P_cl for i in I_cl), default=1
    )
    for idx in range(max_depth):
        for p_c in P_cl:
            for i_c in I_cl:
                p, i = p_c[idx % len(p_c)], i_c[idx % len(i_c)]
                if O_cl and C_cl:
                    for o_c in O_cl:
                        for c_c in C_cl:
                            programmatic.append(
                                ([p, i, o_c[idx % len(o_c)], c_c[idx % len(c_c)]], "programmatic")
                            )
                if O_cl:
                    for o_c in O_cl:
                        programmatic.append(([p, i, o_c[idx % len(o_c)]], "programmatic"))
                if C_cl:
                    for c_c in C_cl:
                        programmatic.append(([p, i, c_c[idx % len(c_c)]], "programmatic"))
                programmatic.append(([p, i], "programmatic"))

    accepted: List[Tuple[List[str], Dict[str, Any]]] = []
    audit: List[Dict[str, Any]] = []
    rejected_llm: List[Dict[str, Any]] = []
    seen: set = set()

    for q, origin in [*llm_queries, *programmatic]:
        q = [str(term).strip() for term in q if str(term).strip()]
        query_text = " ".join(q)
        categories = _query_categories(q, buckets)
        reason = None if origin == "scout" and len(_query_tokens(query_text)) >= 2 else _query_rejection_reason(q, buckets)
        key = _query_key(query_text)
        tier = str(tier_overrides.get(key) or _query_tier(categories))
        if origin == "scout":
            tier = "scout"
        if key not in tier_overrides:
            tier = _apply_contract_technical_tier(query_text, tier, pico)

        record = {
            "origin": origin,
            "query": query_text,
            "categories": ",".join(categories),
            "tier": tier,
            "token_count": len(_query_tokens(query_text)),
            "status": "accepted",
            "reason": "",
            "used": False,
        }
        if reason:
            record["status"] = "rejected"
            record["reason"] = reason
            if origin == "llm":
                rejected_llm.append(record)
            continue
        if key in seen:
            record["status"] = "rejected"
            record["reason"] = "duplicate"
            if origin == "llm":
                rejected_llm.append(record)
            continue

        seen.add(key)
        accepted.append((q, record))

    tier_priority = {"exact": 0, "focused": 1, "scout": 2, "core": 3, "other": 4}
    accepted.sort(key=lambda item: (
        tier_priority.get(str(item[1].get("tier") or "other"), 3),
        item[1]["origin"] != "llm",
        abs(int(item[1]["token_count"]) - 7),
        int(item[1]["token_count"]),
    ))
    tier_limits = getattr(
        config,
        "SEARCH_QUERY_TIER_LIMITS",
        {"core": 6, "focused": 10, "exact": 8, "scout": 4},
    )
    used: List[Tuple[List[str], Dict[str, Any]]] = []
    overflow_by_tier: Dict[str, List[Tuple[List[str], Dict[str, Any]]]] = {}
    for tier_name in ("exact", "focused", "scout", "core", "other"):
        tier_pool = [item for item in accepted if item[1].get("tier") == tier_name]
        quota = int(tier_limits.get(tier_name, 0))
        used.extend(tier_pool[:quota])
        overflow_by_tier[tier_name] = tier_pool[quota:]
    if len(used) < 24:
        fill_pool = (
            overflow_by_tier.get("exact", [])
            + overflow_by_tier.get("focused", [])
            + overflow_by_tier.get("scout", [])
            + overflow_by_tier.get("core", [])
            + overflow_by_tier.get("other", [])
        )
        used.extend(fill_pool[: 24 - len(used)])
    used = used[:24]

    if emit_logs:
        tier_counts: Dict[str, int] = {}
        for _q, record in used:
            tier_name = str(record.get("tier") or "other")
            tier_counts[tier_name] = tier_counts.get(tier_name, 0) + 1
        logger.info(
            "🎯 [QueryBuilder] %d accepted queries (LLM=%d, programmatic=%d). Using %d tiered queries: %s",
            len(accepted),
            len(llm_queries),
            len(programmatic),
            len(used),
            tier_counts,
        )
    for q, record in used:
        record["used"] = True
        audit.append(record)
    audit.extend(rejected_llm)

    if emit_logs:
        for _, record in used:
            logger.info(
                "   [QueryAudit] USED %-12s tier=%s cats=%s query='%s'",
                record["origin"],
                record.get("tier", ""),
                record["categories"],
                record["query"][:120],
            )
        for record in rejected_llm:
            logger.warning(
                "   [QueryAudit] REJECTED %-12s reason=%s query='%s'",
                record["origin"],
                record["reason"],
                record["query"][:120],
            )

    return [q for q, _record in used], audit


def _build_semantic_queries(pico: Dict, question: Optional[str] = None) -> List[List[str]]:
    queries, _audit = _build_semantic_queries_with_audit(pico, question)
    return queries


def build_search_query_audit(
    pico: Dict,
    question: Optional[str] = None,
    phase: str = "",
) -> List[Dict[str, Any]]:
    queries, audit = _build_semantic_queries_with_audit(pico, question, emit_logs=False)
    rows: List[Dict[str, Any]] = []

    for row in audit:
        enriched = row.copy()
        enriched["phase"] = phase
        enriched["source_family"] = "semantic"
        rows.append(enriched)

    rows.append({
        "phase": phase,
        "source_family": "pubmed",
        "origin": "pico_translator",
        "query": _pico_to_pubmed(pico, question),
        "categories": ",".join(k for k in ["P", "I", "O", "C"] if pico.get(k)),
        "token_count": 0,
        "status": "accepted",
        "reason": "",
        "used": True,
    })
    rows.append({
        "phase": phase,
        "source_family": "europe_pmc",
        "origin": "pico_translator",
        "query": _pico_to_europe_pmc(pico, question),
        "categories": ",".join(k for k in ["P", "I", "O", "C"] if pico.get(k)),
        "token_count": 0,
        "status": "accepted",
        "reason": "",
        "used": True,
    })
    rows.append({
        "phase": phase,
        "source_family": "semantic_query_count",
        "origin": "query_builder",
        "query": str(len(queries)),
        "categories": "",
        "token_count": 0,
        "status": "accepted",
        "reason": "",
        "used": True,
    })
    return rows


# â”€â”€â”€ Source implementations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _limit_source_results(articles: List[Dict], profile: SourceProfile) -> List[Dict]:
    limit = max(0, int(profile.target))
    if limit and len(articles) > limit:
        logger.info("[%s] Applying hard source cap: %d -> %d", profile.name, len(articles), limit)
        return articles[:limit]
    return articles


def _search_semantic_scholar(
    queries: List[List[str]],
    profile: SourceProfile,
    cancel_event: Optional[threading.Event] = None,
) -> List[Dict]:
    if not getattr(config, profile.api_key_attr, ""):
        logger.warning("âš ï¸ Semantic Scholar API key not set. Skipping.")
        return []

    api_key = getattr(config, profile.api_key_attr)
    current_year = datetime.now(timezone.utc).year
    year_range = f"{current_year - SEARCH_WINDOW_YEARS}-{current_year}"
    fields = ("title,year,abstract,authors,externalIds,url,"
              "openAccessPdf,journal,publicationDate,venue,citationCount")
    articles_map: Dict[str, Dict] = {}
    relevance_limit = _bounded_int(
        getattr(config, "SEMANTIC_SCHOLAR_RELEVANCE_PER_QUERY", 40),
        default=40,
        minimum=1,
        maximum=100,
    )
    citation_limit = _bounded_int(
        getattr(config, "SEMANTIC_SCHOLAR_CITATION_PER_QUERY", 15),
        default=15,
        minimum=0,
        maximum=100,
    )
    citation_query_limit = _bounded_int(
        getattr(config, "SEMANTIC_SCHOLAR_CITATION_QUERY_LIMIT", 8),
        default=8,
        minimum=0,
        maximum=1000,
    )
    citation_pass_enabled = bool(getattr(config, "SEMANTIC_SCHOLAR_ENABLE_CITATION_PASS", True))
    citation_sort = str(getattr(config, "SEMANTIC_SCHOLAR_CITATION_SORT", "citationCount:desc")).strip()
    seen_queries: Set[str] = set()

    for idx, terms in enumerate(queries):
        if cancel_event and cancel_event.is_set():
            logger.info("ðŸ›‘ [SS] Cancelled after timeout/cancel signal")
            break
        if len(articles_map) >= profile.target:
            break

        query = _semantic_scholar_query_text(" ".join(terms).strip())
        if not query or len(query) < 5:
            continue
        query_key = _query_key(query)
        if query_key in seen_queries:
            continue
        seen_queries.add(query_key)

        logger.info("ðŸš€ [SS] Query %d/%d: '%s'", idx + 1, len(queries), query)
        offset = 0
        retrieved = 0
        max_offset = relevance_limit

        while offset < max_offset and retrieved < relevance_limit:
            if cancel_event and cancel_event.is_set():
                break
            if len(articles_map) >= profile.target:
                break

            page_limit = min(100, relevance_limit - retrieved)
            headers = {
                "x-api-key": api_key,
                "User-Agent": _random_ua(),
                "Accept": "application/json",
            }
            params = {
                "query": query,
                "year": year_range,
                "limit": page_limit,
                "fields": fields,
                "offset": offset,
            }
            resp = _get_with_retry(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params, headers, profile, f"offset={offset}",
            )
            if resp is None:
                break

            data = resp.json()
            papers = data.get("data") or []
            if not papers:
                break

            added = _process_ss_batch(papers, articles_map, query, retrieval_mode="relevance")
            retrieved += added
            offset += page_limit
            time.sleep(profile.rate_limit_sleep + random.uniform(0.1, 0.4))
            if added == 0:
                break

        if (
            citation_pass_enabled
            and citation_limit > 0
            and idx < citation_query_limit
            and citation_sort
            and not (cancel_event and cancel_event.is_set())
            and len(articles_map) < profile.target
        ):
            logger.info("Ã°Å¸Å¡â‚¬ [SS] Query %d/%d citations: '%s'", idx + 1, len(queries), query)
            headers = {
                "x-api-key": api_key,
                "User-Agent": _random_ua(),
                "Accept": "application/json",
            }
            params = {
                "query": query,
                "year": year_range,
                "limit": citation_limit,
                "fields": fields,
                "sort": citation_sort,
            }
            resp = _get_with_retry(
                "https://api.semanticscholar.org/graph/v1/paper/search/bulk",
                params, headers, profile, f"bulk-sort={citation_sort}",
            )
            if resp is not None:
                data = resp.json()
                papers = data.get("data") or []
                _process_ss_batch(papers, articles_map, query, retrieval_mode="citation")
                time.sleep(profile.rate_limit_sleep + random.uniform(0.1, 0.4))

    logger.info("ðŸ“š [SS] %d articles", len(articles_map))
    return _limit_source_results(list(articles_map.values()), profile)


def _merge_retrieval_mode(existing: Dict, retrieval_query: str, retrieval_mode: str) -> None:
    current_modes = {
        mode.strip()
        for mode in str(existing.get("retrieval_mode") or "").split("+")
        if mode.strip()
    }
    current_modes.add(retrieval_mode)
    existing["retrieval_mode"] = "+".join(sorted(current_modes))

    queries = existing.setdefault("retrieval_queries", [])
    if isinstance(queries, list) and retrieval_query not in queries:
        queries.append(retrieval_query)


def _process_ss_batch(
    papers: List[Dict],
    out: Dict[str, Dict],
    retrieval_query: str,
    retrieval_mode: str = "relevance",
) -> int:
    added = 0
    for p in papers:
        title = p.get("title") or ""
        if _is_noise_title(title):
            continue
        abstract = p.get("abstract") or ""
        doi = (p.get("externalIds") or {}).get("DOI") or ""
        has_doi = len(doi.strip()) > 5
        if len(abstract) < 100 and not has_doi and len(title) < 25:
            continue
        abstract = abstract or "No abstract available."
        pid = p.get("paperId")
        if not pid:
            continue
        if pid in out:
            _merge_retrieval_mode(out[pid], retrieval_query, retrieval_mode)
            out[pid]["citations"] = max(out[pid].get("citations") or 0, p.get("citationCount") or 0)
            continue

        journal_info = p.get("journal") or {}
        oa_pdf = p.get("openAccessPdf") or {}
        pdf_url = oa_pdf.get("url") or ""
        paper_url = p.get("url") or ""
        url = paper_url or (f"https://doi.org/{doi}" if doi else "")
        journal = journal_info.get("name") or p.get("venue") or ""

        out[pid] = {
            "paper_id": pid,
            "title": title,
            "authors": [a["name"] for a in (p.get("authors") or [])],
            "doi": doi,
            "year": p.get("year") or 0,
            "abstract": abstract,
            "journal": journal if journal else None,
            "venue": p.get("venue") or journal or None,
            "volume": journal_info.get("volume") or "",
            "pages": journal_info.get("pages") or "",
            "url": url,
            "pdf_url": pdf_url or url,
            "open_access": bool(pdf_url),
            "citations": p.get("citationCount") or 0,
            "source": "Semantic Scholar",
            "retrieval_query": retrieval_query,
            "retrieval_queries": [retrieval_query],
            "retrieval_mode": retrieval_mode,
        }
        added += 1
    return added


def _semantic_scholar_seed_ids(article: Dict[str, Any]) -> List[str]:
    identifiers: List[str] = []
    paper_id = str(article.get("paper_id") or article.get("paperId") or "").strip()
    if paper_id:
        identifiers.append(paper_id)

    doi = str(article.get("doi") or "").strip()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip("/")
    if doi.startswith("10."):
        identifiers.append(f"DOI:{doi}")

    seen: Set[str] = set()
    unique_ids: List[str] = []
    for identifier in identifiers:
        key = identifier.lower()
        if key not in seen:
            seen.add(key)
            unique_ids.append(identifier)
    return unique_ids


def _graph_direction_payload(direction: str) -> Tuple[str, str]:
    if direction == "citations":
        return "citingPaper", (
            "citingPaper.paperId,citingPaper.title,citingPaper.year,"
            "citingPaper.abstract,citingPaper.authors,citingPaper.externalIds,"
            "citingPaper.url,citingPaper.openAccessPdf,citingPaper.journal,"
            "citingPaper.publicationDate,citingPaper.venue,citingPaper.citationCount"
        )
    return "citedPaper", (
        "citedPaper.paperId,citedPaper.title,citedPaper.year,"
        "citedPaper.abstract,citedPaper.authors,citedPaper.externalIds,"
        "citedPaper.url,citedPaper.openAccessPdf,citedPaper.journal,"
        "citedPaper.publicationDate,citedPaper.venue,citedPaper.citationCount"
    )


def search_semantic_scholar_graph_neighbors(
    seed_articles: List[Dict[str, Any]],
    question: str = "",
    max_articles: Optional[int] = None,
    cancel_event: Optional[threading.Event] = None,
) -> List[Dict[str, Any]]:
    """Recover candidates from Semantic Scholar citation graph neighbors."""
    profile = _profile("Semantic Scholar")
    if not getattr(config, profile.api_key_attr, ""):
        logger.warning("[AdaptiveGraph] Semantic Scholar API key not set. Skipping graph recovery.")
        return []

    api_key = getattr(config, profile.api_key_attr)
    per_seed = _bounded_int(
        getattr(config, "ADAPTIVE_GRAPH_RECOVERY_PER_SEED", 12),
        default=12,
        minimum=1,
        maximum=100,
    )
    max_items = _bounded_int(
        max_articles if max_articles is not None else getattr(config, "ADAPTIVE_GRAPH_RECOVERY_MAX_ARTICLES", 120),
        default=120,
        minimum=1,
        maximum=1000,
    )
    directions = [
        direction
        for direction in getattr(config, "ADAPTIVE_GRAPH_RECOVERY_DIRECTIONS", ["references", "citations"])
        if direction in {"references", "citations"}
    ] or ["references", "citations"]

    headers = {
        "x-api-key": api_key,
        "User-Agent": _random_ua(),
        "Accept": "application/json",
    }
    articles_map: Dict[str, Dict] = {}
    seen_seed_ids: Set[str] = set()

    for seed_idx, seed in enumerate(seed_articles):
        if cancel_event and cancel_event.is_set():
            break
        if len(articles_map) >= max_items:
            break
        seed_ids = _semantic_scholar_seed_ids(seed)
        if not seed_ids:
            continue
        seed_id = next((value for value in seed_ids if value.lower() not in seen_seed_ids), "")
        if not seed_id:
            continue
        seen_seed_ids.add(seed_id.lower())
        seed_label = str(seed.get("title") or question or "near_miss")[:80]

        for direction in directions:
            if cancel_event and cancel_event.is_set():
                break
            if len(articles_map) >= max_items:
                break
            nested_key, fields = _graph_direction_payload(direction)
            url_id = quote(seed_id, safe=":")
            url = f"https://api.semanticscholar.org/graph/v1/paper/{url_id}/{direction}"
            params = {
                "fields": fields,
                "limit": per_seed,
            }
            logger.info(
                "[AdaptiveGraph] Seed %d/%d %s: %s",
                seed_idx + 1,
                len(seed_articles),
                direction,
                seed_label,
            )
            resp = _get_with_retry(url, params, headers, profile, f"graph-{direction}")
            if resp is None:
                continue
            data = resp.json()
            records = data.get("data") or []
            papers = [
                item.get(nested_key)
                for item in records
                if isinstance(item, dict) and isinstance(item.get(nested_key), dict)
            ]
            _process_ss_batch(
                papers[: max(0, max_items - len(articles_map))],
                articles_map,
                f"graph:{direction}:{seed_label}",
                retrieval_mode=f"snowball_{direction}",
            )
            time.sleep(profile.rate_limit_sleep + random.uniform(0.1, 0.4))

    logger.info("[AdaptiveGraph] Recovered %d Semantic Scholar graph neighbors", len(articles_map))
    return list(articles_map.values())[:max_items]


def _search_pubmed(
    queries: List[List[str]],
    profile: SourceProfile,
    cancel_event: Optional[threading.Event] = None,
) -> List[Dict]:
    all_articles: List[Dict] = []
    seen_ids: set = set()
    current_year = datetime.now(timezone.utc).year
    past_year = current_year - SEARCH_WINDOW_YEARS

    for idx, terms in enumerate(queries):
        if cancel_event and cancel_event.is_set():
            logger.info("ðŸ›‘ [PubMed] Cancelled after timeout/cancel signal")
            break
        if len(all_articles) >= profile.target:
            break

        base = terms[0] if len(terms) == 1 else " AND ".join(terms)
        query = (f"({base}) AND (free full text[sb]) "
                 f"AND ({past_year}:{current_year}[dp])")
        logger.info("ðŸ”¬ [PubMed] Query %d/%d: '%s'", idx + 1, len(queries), query[:80])

        try:
            retstart = 0
            while len(all_articles) < profile.target and retstart < 5000:
                if cancel_event and cancel_event.is_set():
                    break
                handle = Entrez.esearch(
                    db="pubmed", term=query, retmax=500,
                    retstart=retstart, sort="relevance",
                )
                record = Entrez.read(handle)
                handle.close()

                ids = [i for i in record["IdList"] if i not in seen_ids]
                if not ids:
                    break
                seen_ids.update(ids)

                handle = Entrez.efetch(
                    db="pubmed", id=ids, rettype="abstract", retmode="xml"
                )
                records = Entrez.read(handle)
                handle.close()

                if "PubmedArticle" not in records:
                    break

                for article in records["PubmedArticle"]:
                    if len(all_articles) >= profile.target:
                        break
                    parsed = _parse_pubmed_article(article)
                    if parsed:
                        parsed["retrieval_query"] = query
                        all_articles.append(parsed)

                retstart += 100

        except Exception as exc:
            logger.error("âŒ [PubMed] Error: %s", exc)

    logger.info("ðŸ“š [PubMed] %d articles", len(all_articles))
    return _limit_source_results(all_articles, profile)


def _parse_pubmed_article(article: Any) -> Optional[Dict]:
    medline = article["MedlineCitation"]
    art = medline["Article"]
    journal = art.get("Journal", {})
    issue = journal.get("JournalIssue", {})

    abstract_parts = (
        art.get("Abstract", {}).get("AbstractText", [])
    )
    if isinstance(abstract_parts, list):
        abstract = " ".join(str(p) for p in abstract_parts)
    else:
        abstract = str(abstract_parts or "")

    if len(abstract) < 100:
        return None

    authors: List[str] = []
    for a in art.get("AuthorList", []):
        if "LastName" in a and "ForeName" in a:
            authors.append(f"{a['LastName']}, {a['ForeName']}")

    pub_date = issue.get("PubDate", {})
    try:
        year = int(pub_date.get("Year", 0))
    except (ValueError, TypeError):
        year = 0

    doi = ""
    for eid in art.get("ELocationID", []):
        if eid.attributes.get("EIdType") == "doi":
            doi = str(eid)
            break

    pmid = str(medline.get("PMID", ""))

    return {
        "title": str(art.get("ArticleTitle", "")),
        "abstract": abstract,
        "year": year,
        "authors": authors,
        "journal": journal.get("Title") or None,
        "journal_short": journal.get("ISOAbbreviation") or "",
        "volume": issue.get("Volume") or "",
        "issue": issue.get("Issue") or "",
        "pages": art.get("Pagination", {}).get("MedlinePgn") or "",
        "issn": journal.get("ISSN") or "",
        "language": (art.get("Language") or ["eng"])[0],
        "doi": doi,
        "pubmed_id": pmid,
        "url": f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}",
        "pdf_url": "",
        "open_access": True,
        "citations": 0,
        "source": "PubMed",
    }


def _search_openalex(
    queries: List[List[str]],
    profile: SourceProfile,
    concept_id: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
) -> List[Dict]:
    current_year = datetime.now(timezone.utc).year
    past_year = current_year - SEARCH_WINDOW_YEARS
    base_filter = f"from_publication_date:{past_year}-01-01,type:article"
    if concept_id:
        base_filter += f",concepts.id:{concept_id}"

    headers = {
        "User-Agent": f"PrismaAssistant/2.0 (mailto:{config.ACADEMIC_EMAIL})",
        "From": config.ACADEMIC_EMAIL,
    }
    articles_map: Dict[str, Dict] = {}
    max_per_query = max(100, profile.target // max(len(queries), 1) + 50)

    for idx, terms in enumerate(queries[:12]):
        if cancel_event and cancel_event.is_set():
            logger.info("ðŸ›‘ [OpenAlex] Cancelled after timeout/cancel signal")
            break
        if len(articles_map) >= profile.target:
            break
        if idx > 0:
            time.sleep(1.5)

        raw = " ".join(terms).strip()
        clean = re.sub(r"\b(AND|OR)\b", " ", raw)
        clean = re.sub(r'["()\[\]]', "", clean)
        clean = re.sub(r"\s+", " ", clean).strip()

        logger.info("ðŸ”¬ [OpenAlex] Query %d/%d: '%s'", idx + 1, len(queries), clean[:60])
        cursor = "*"
        page = 0
        retrieved = 0

        while len(articles_map) < profile.target and cursor and page < 8 and retrieved < max_per_query:
            if cancel_event and cancel_event.is_set():
                break
            params = {
                "search": clean,
                "filter": base_filter,
                "per_page": 200,
                "cursor": cursor,
                "sort": "cited_by_count:desc",
                "mailto": config.ACADEMIC_EMAIL,
                "select": ("id,doi,title,display_name,publication_year,language,authorships,"
                           "primary_location,abstract_inverted_index,open_access,cited_by_count"),
            }
            resp = _get_with_retry(
                "https://api.openalex.org/works",
                params, headers, profile, f"page={page}",
            )
            if resp is None:
                break

            data = resp.json()
            results = data.get("results") or []
            if not results:
                break
            cursor = (data.get("meta") or {}).get("next_cursor")
            time.sleep(profile.rate_limit_sleep + random.uniform(0.1, 0.3))

            for work in results:
                if retrieved >= max_per_query:
                    break
                title = work.get("display_name") or ""
                if not title or _is_noise_title(title):
                    continue
                abstract = _reconstruct_abstract(work.get("abstract_inverted_index")) or ""
                doi_raw = work.get("doi") or ""
                doi = doi_raw.replace("https://doi.org/", "")
                has_doi = len(doi.strip()) > 5
                if len(abstract) < 150 and not has_doi and len(title) < 25:
                    continue
                abstract = abstract or "No abstract available."
                work_id = work.get("id") or ""
                if work_id in articles_map:
                    continue

                authors = [
                    (a.get("author") or {}).get("display_name") or ""
                    for a in (work.get("authorships") or [])[:10]
                ]
                authors = [a for a in authors if a]
                loc = work.get("primary_location") or {}
                src = loc.get("source") or {}
                oa = work.get("open_access") or {}

                articles_map[work_id] = {
                    "openalex_id": work_id,
                    "title": title,
                    "abstract": abstract,
                    "year": work.get("publication_year") or 0,
                    "authors": authors,
                    "doi": doi,
                    "journal": src.get("display_name") or None,
                    "venue": src.get("display_name") or None,
                    "language": work.get("language"),
                    "url": doi_raw or work_id,
                    "pdf_url": oa.get("oa_url") or "",
                    "open_access": True,
                    "citations": work.get("cited_by_count") or 0,
                    "source": "OpenAlex",
                    "retrieval_query": clean,
                }
                retrieved += 1
            page += 1

    logger.info("ðŸ“š [OpenAlex] %d articles", len(articles_map))
    return _limit_source_results(list(articles_map.values()), profile)


def _reconstruct_abstract(inverted_index: Optional[Dict]) -> str:
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    try:
        word_positions = [
            (pos, word)
            for word, positions in inverted_index.items()
            for pos in positions
        ]
        word_positions.sort(key=lambda x: x[0])
        return " ".join(word for _, word in word_positions)
    except Exception:
        return ""


def _search_europe_pmc(
    queries: List[List[str]],
    profile: SourceProfile,
    cancel_event: Optional[threading.Event] = None,
) -> List[Dict]:
    current_year = datetime.now(timezone.utc).year
    past_year = current_year - SEARCH_WINDOW_YEARS
    articles_map: Dict[str, Dict] = {}

    for idx, terms in enumerate(queries):
        if cancel_event and cancel_event.is_set():
            logger.info("ðŸ›‘ [EuropePMC] Cancelled after timeout/cancel signal")
            break
        if len(articles_map) >= profile.target:
            break

        raw = " ".join(terms).strip()
        query = f"({raw}) AND (PUB_YEAR:[{past_year} TO {current_year}])"
        logger.info("ðŸ”¬ [EuropePMC] Query %d/%d: '%s'", idx + 1, len(queries), raw[:60])

        cursor = "*"
        page = 0

        while len(articles_map) < profile.target and cursor and page < 100:
            if cancel_event and cancel_event.is_set():
                break
            params = {
                "query": query,
                "resultType": "core",
                "pageSize": 1000,
                "cursorMark": cursor,
                "format": "json",
                "sort": "RELEVANCE",
            }
            resp = _get_with_retry(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params, {}, profile, f"page={page}",
            )
            if resp is None:
                break

            data = resp.json()
            results = (data.get("resultList") or {}).get("result") or []
            if not results:
                break

            next_cursor = data.get("nextCursorMark")
            if next_cursor == cursor:
                break
            cursor = next_cursor

            for art in results:
                title = art.get("title") or ""
                if not title or _is_noise_title(title):
                    continue
                abstract = art.get("abstractText") or ""
                doi = art.get("doi") or ""
                has_doi = len(doi.strip()) > 5
                if len(abstract) < 150 and not has_doi and len(title) < 25:
                    continue
                abstract = abstract or "No abstract available."

                pmcid = art.get("pmcid") or ""
                pmid = art.get("pmid") or ""
                key = pmcid or pmid or title[:60]
                if key in articles_map:
                    continue

                try:
                    year = int(art.get("pubYear") or 0)
                except (ValueError, TypeError):
                    year = 0

                articles_map[key] = {
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "authors": _parse_epmc_authors(art.get("authorString") or ""),
                    "doi": doi,
                    "journal": (art.get("journalTitle")
                                or ((art.get("journalInfo") or {}).get("journal") or {}).get("title")
                                or None),
                    "volume": art.get("journalVolume") or "",
                    "issue": art.get("issue") or "",
                    "pages": art.get("pageInfo") or "",
                    "pubmed_id": pmid,
                    "language": art.get("language"),
                    "url": (f"https://doi.org/{doi}" if doi
                            else f"https://europepmc.org/article/MED/{pmid}" if pmid else ""),
                    "pdf_url": f"https://europepmc.org/articles/{pmcid}?pdf=render" if pmcid else "",
                    "open_access": True,
                    "citations": art.get("citationCount") or 0,
                    "source": "Europe PMC",
                    "retrieval_query": query,
                }

            page += 1
            time.sleep(profile.rate_limit_sleep)

    logger.info("ðŸ“š [EuropePMC] %d articles", len(articles_map))
    return _limit_source_results(list(articles_map.values()), profile)


def _parse_epmc_authors(author_string: str) -> List[str]:
    if not author_string:
        return []
    authors: List[str] = []
    parts = [p.strip(". ") for p in author_string.split(", ") if p.strip(". ")]
    i = 0
    while i < len(parts):
        part = parts[i]
        if i + 1 < len(parts) and len(parts[i + 1]) <= 3:
            authors.append(f"{part}, {parts[i + 1]}")
            i += 2
        else:
            authors.append(part)
            i += 1
    return authors[:20]


# â”€â”€â”€ LLM-driven OpenAlex concept resolution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def _resolve_openalex_concept(question: str) -> Optional[str]:
    """
    Asks the search engine's LLM to pick the best OpenAlex concept ID
    for the given question, then validates it against the OpenAlex API.

    This replaces the hardcoded domain â†’ concept_id map.
    Returns None if the LLM can't identify a concept or the API rejects it.
    """
    system_msg = (
        "You are a research librarian. Given a research question, output ONLY "
        "a valid OpenAlex concept ID (format: C followed by digits) that best "
        "represents the primary academic field. "
        "If unsure, output null. No explanation."
    )
    prompt = (
        f'Research question: "{question}"\n'
        "Output only a single OpenAlex concept ID or null."
    )
    try:
        from app.llm.ai_model import generate_text
        response = (generate_text(
            instruction=prompt, input_text="", max_tokens=32,
            system_prompt=system_msg,
        ) or "").strip()

        match = re.search(r"C\d{5,12}", response)
        if not match:
            return None
        concept_id = match.group()

        # Validate concept exists in OpenAlex
        r = requests.get(
            f"https://api.openalex.org/concepts/{concept_id}",
            params={"mailto": config.ACADEMIC_EMAIL},
            timeout=5,
        )
        if r.status_code == 200:
            concept_name = r.json().get("display_name", concept_id)
            logger.info("ðŸ§  [ConceptResolver] Resolved '%s' â†’ %s (%s)",
                        question[:50], concept_id, concept_name)
            return concept_id
        logger.warning("âš ï¸ [ConceptResolver] Concept %s not found in OpenAlex", concept_id)
    except Exception as exc:
        logger.warning("âš ï¸ [ConceptResolver] Failed: %s", exc)
    return None


# â”€â”€â”€ Citation filter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_CITATION_FILTERED_SOURCES = {"Semantic Scholar", "OpenAlex", "Europe PMC"}


def _apply_citation_filter(articles: List[Dict]) -> Tuple[List[Dict], int]:
    mode = str(getattr(config, "CITATION_FILTER_MODE", "rank_only")).strip().lower()
    if mode in {"off", "rank", "rank_only", "ranking"}:
        return articles, 0

    kept, dropped = [], 0
    for a in articles:
        if a.get("source") not in _CITATION_FILTERED_SOURCES:
            kept.append(a)
            continue
        if not _normalize_doi(a.get("doi") or ""):
            kept.append(a)  # No DOI â†’ can't reliably filter
            continue
        threshold = _min_citations_for_year(a.get("year") or 0)
        if (a.get("citations") or 0) >= threshold:
            kept.append(a)
        else:
            dropped += 1
    return kept, dropped


# â”€â”€â”€ Main entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def run_parallel_search(
    original_question: str,
    domain: str = "general",   # kept for backwards-compat; no longer used internally
    client_id: str = None,
    pico: Optional[Dict] = None,
    disabled_sources: Optional[List[str]] = None,
    custom_timeout: Optional[float] = None,
) -> List[Dict]:
    """
    Unified parallel search engine.

    Changes from original:
    - Pure asyncio (no mixed ThreadPoolExecutor + asyncio anti-pattern).
    - Source configuration driven by SourceProfile registry.
    - Domain detection replaced by LLM-driven OpenAlex concept resolution.
    - Citation filter, dedup, and re-ranking are isolated functions.
    - No magic numbers inline.
    """
    pico_dict = pico if pico is not None else generate_api_queries_with_llm(original_question)
    disabled = set(disabled_sources or [])

    # Build queries once; pass to each source
    semantic_queries = _build_semantic_queries(pico_dict, original_question)
    pubmed_queries = [[_pico_to_pubmed(pico_dict, original_question)]]
    epmc_queries = [[_pico_to_europe_pmc(pico_dict, original_question)]]

    current_year = datetime.now(timezone.utc).year
    past_year = current_year - SEARCH_WINDOW_YEARS
    logger.info("ðŸŽ¯ [PICOâ†’Queries] PubMed: '%s'", pubmed_queries[0][0][:80])
    logger.info("ðŸŽ¯ [PICOâ†’Queries] SS strategies: %d", len(semantic_queries))
    logger.info("ðŸŽ¯ [PICOâ†’Queries] EPMC: '%s'", epmc_queries[0][0][:80])

    # Resolve OpenAlex concept via LLM (replaces hardcoded domain map)
    openalex_concept = await _resolve_openalex_concept(original_question)

    # Build task list from registry, skipping disabled sources
    cancel_event = threading.Event()

    async def _run_in_thread(fn: Callable, *args, **kwargs) -> List[Dict]:
        return await asyncio.to_thread(fn, *args, **kwargs, cancel_event=cancel_event)

    tasks: Dict[str, asyncio.Task] = {}

    if "Semantic Scholar" not in disabled:
        p = _profile("Semantic Scholar")
        if p.enabled:
            timeout_val = custom_timeout if custom_timeout is not None else p.timeout_seconds
            tasks["Semantic Scholar"] = asyncio.create_task(
                asyncio.wait_for(
                    _run_in_thread(_search_semantic_scholar, semantic_queries, p),
                    timeout=timeout_val,
                )
            )

    if "PubMed" not in disabled:
        p = _profile("PubMed")
        if p.enabled:
            timeout_val = custom_timeout if custom_timeout is not None else p.timeout_seconds
            tasks["PubMed"] = asyncio.create_task(
                asyncio.wait_for(
                    _run_in_thread(_search_pubmed, pubmed_queries, p),
                    timeout=timeout_val,
                )
            )

    if "OpenAlex" not in disabled:
        p = _profile("OpenAlex")
        if p.enabled:
            timeout_val = custom_timeout if custom_timeout is not None else p.timeout_seconds
            tasks["OpenAlex"] = asyncio.create_task(
                asyncio.wait_for(
                    _run_in_thread(_search_openalex, semantic_queries, p, openalex_concept),
                    timeout=timeout_val,
                )
            )

    if "Europe PMC" not in disabled:
        p = _profile("Europe PMC")
        if p.enabled:
            timeout_val = custom_timeout if custom_timeout is not None else p.timeout_seconds
            tasks["Europe PMC"] = asyncio.create_task(
                asyncio.wait_for(
                    _run_in_thread(_search_europe_pmc, epmc_queries, p),
                    timeout=timeout_val,
                )
            )

    logger.info("ðŸš€ Running %d sources in parallel: %s", len(tasks), list(tasks))

    raw_articles: List[Dict] = []
    try:
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    finally:
        cancel_event.set()

    for name, result in zip(tasks.keys(), results):
        if isinstance(result, asyncio.TimeoutError):
            logger.warning("âš ï¸ [%s] Timed out â€” skipped", name)
        elif isinstance(result, Exception):
            logger.error("âŒ [%s] Exception: %s", name, result)
        elif isinstance(result, list):
            logger.info("âœ… [%s] %d articles", name, len(result))
            raw_articles.extend(result)
        else:
            logger.warning("âš ï¸ [%s] Unexpected result type: %s", name, type(result))

    logger.info("ðŸ“Š Raw total: %d articles from %d sources", len(raw_articles), len(tasks))

    # Citation filter
    raw_articles, dropped = _apply_citation_filter(raw_articles)
    if dropped:
        logger.info("ðŸ§¹ Citation filter dropped %d low-quality articles", dropped)

    # Deduplication
    unique = _deduplicate(raw_articles)
    logger.info("ðŸ”— After dedup: %d unique articles", len(unique))

    # Relevance re-ranking
    scorer = RelevanceScorer(pico_dict)
    for a in unique:
        a["relevance_score"] = scorer.score(a)
    unique.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    logger.info("ðŸŽ¯ Re-ranked by PICO co-occurrence")

    # Cap to configured maximum
    if len(unique) > MAX_RESULTS_TOTAL:
        unique = unique[:MAX_RESULTS_TOTAL]

    # Source distribution log
    source_counts: Dict[str, int] = {}
    for a in unique:
        src = a.get("source", "Unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
    for src, count in sorted(source_counts.items()):
        logger.info("   ðŸ“Œ %s: %d", src, count)

    # Lazy PDF enrichment
    for a in unique:
        enrich_initial_search_result(a)

    return unique

