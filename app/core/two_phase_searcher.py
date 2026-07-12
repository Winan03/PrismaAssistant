"""
two_phase_search.py — Refactored for academic SLR quality.

Fixes applied:
  1. Domain-agnostic LLM prompt (no hardcoded examples)
  2. Source monitoring derived from runtime config, not hardcoded list
  3. Robust DOI normalisation + Jaccard fuzzy dedup
  4. Cache keyed on prompt hash + question (TTL + version-safe)
  5. ArticleRecord TypedDict enforces full provenance contract
  6. Explicit BootstrapResult returned to caller (no silent fallback)
"""

import logging
import json
import re
import os
import hashlib
import unicodedata
from typing import Any, List, Dict, Optional, TypedDict, Literal
from dataclasses import dataclass, field
from datetime import datetime, timezone

import config
from sklearn.feature_extraction.text import TfidfVectorizer

from app.llm.ai_model import generate_text
from app.core.search_engine import (
    build_search_query_audit,
    run_parallel_search,
    get_registered_sources,
)
from app.core.adaptive_retrieval import augment_framework_with_empirical_lexicon
from app.domain.eligibility_contract import merge_contract_into_pico
from app.domain.query_expander import generate_api_queries_with_llm

logger = logging.getLogger(__name__)

# ─── Cache ────────────────────────────────────────────────────────────────────

CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Increment this whenever the classification prompt changes.
# Old cache files with a different PROMPT_VERSION are automatically stale.
PROMPT_VERSION = "v4_validated"

# Seconds before a cache entry is considered stale (7 days).
CACHE_TTL_SECONDS = 60 * 60 * 24 * 7

_BROAD_ENRICHMENT_STOPWORDS = frozenset(getattr(config, "BROAD_ENRICHMENT_STOPWORDS", ()))


def _dedupe_terms(terms: List[str]) -> List[str]:
    """Preserve order while removing empty or duplicate search terms."""
    seen = set()
    clean_terms: List[str] = []
    for term in terms:
        text = re.sub(r"\s+", " ", str(term or "").strip())
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            clean_terms.append(text)
    return clean_terms


def _meaningful_tokens(term: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(term or "").lower())
        if len(token) > 1
    ]


def _is_low_signal_enrichment(term: str) -> bool:
    """Reject corpus-mined terms that broaden Phase C without carrying a concept."""
    tokens = _meaningful_tokens(term)
    if not tokens:
        return True
    normalized = " ".join(tokens)
    if normalized in _BROAD_ENRICHMENT_STOPWORDS:
        return True
    if len(tokens) == 1 and tokens[0] in _BROAD_ENRICHMENT_STOPWORDS:
        return True
    if len(tokens) == 1 and len(tokens[0]) < 7:
        return True
    return False


def _build_enriched_framework(
    original_components: Dict[str, List[str]],
    classified_components: Dict[str, List[str]],
    framework_keys: List[str],
    anchor_keys: List[str],
) -> Dict[str, List[str]]:
    """
    Phase B may discover useful outcome/context terms, but it must not broaden
    mandatory search axes. P/I are anchored to the original RQ decomposition.
    """
    enriched: Dict[str, List[str]] = {}
    dropped: Dict[str, List[str]] = {}
    anchor_set = set(anchor_keys)

    for key in framework_keys:
        base_terms = _dedupe_terms(original_components.get(key, []))
        if key in anchor_set:
            extra_terms = _dedupe_terms(classified_components.get(key, []))
            if extra_terms:
                dropped[key] = extra_terms
            enriched[key] = base_terms
            continue

        accepted_extras: List[str] = []
        for term in _dedupe_terms(classified_components.get(key, [])):
            if _is_low_signal_enrichment(term):
                dropped.setdefault(key, []).append(term)
                continue
            accepted_extras.append(term)
        enriched[key] = _dedupe_terms(base_terms + accepted_extras)

    if original_components.get("semantic_queries"):
        enriched["semantic_queries"] = _dedupe_terms(original_components.get("semantic_queries", []))
    for key, value in original_components.items():
        if str(key).startswith("_"):
            enriched[key] = value

    if dropped:
        logger.info("🧹 [Phase B→C] Dropped unsafe enrichment terms: %s", dropped)

    return enriched


def _sanitize_initial_framework(pico: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Remove low-signal initial PICO terms before they create broad API queries."""
    clean_pico: Dict[str, List[str]] = {}
    for key, terms in pico.items():
        if str(key).startswith("_"):
            clean_pico[key] = terms
            continue
        if key == "semantic_queries":
            clean_pico[key] = _dedupe_terms(terms)
            continue
        clean_terms = []
        for term in _dedupe_terms(terms):
            if key in {"C", "O"} and _is_low_signal_enrichment(term):
                logger.info("🧹 [PICO] Dropped low-signal %s term: %s", key, term)
                continue
            clean_terms.append(term)
        clean_pico[key] = clean_terms
    return clean_pico


def _build_initial_framework(original_question: str, eligibility_contract: Optional[Dict] = None) -> Dict:
    """Build the retrieval framework, preferring the audited eligibility contract."""
    if eligibility_contract:
        contract_framework = merge_contract_into_pico({}, eligibility_contract)
        contract_framework = _sanitize_initial_framework(contract_framework)
        if contract_framework:
            logger.info("[Two-Phase] Using eligibility contract as primary search framework")
            return contract_framework

    pico = generate_api_queries_with_llm(original_question)
    if isinstance(pico, dict):
        return _sanitize_initial_framework(pico)
    return pico


def _build_phase_a_framework(pico: Dict[str, List[str]], framework_keys: List[str]) -> Dict:
    """Keep Phase A broad while preserving contract-planned direct queries."""
    broad_keys = framework_keys[:2]
    broad_components = {k: pico.get(k, []) for k in broad_keys}
    null_components = {k: [] for k in framework_keys if k not in broad_keys}
    broad_pico: Dict = {**broad_components, **null_components}
    if "semantic_queries" in pico:
        broad_pico["semantic_queries"] = pico["semantic_queries"]
    for key in pico:
        if str(key).startswith("_"):
            broad_pico[key] = pico[key]
    return broad_pico


def _sanitize_classified_components(
    classified_components: Dict[str, List[str]],
    framework_keys: List[str],
    anchor_keys: List[str],
) -> Dict[str, List[str]]:
    """Keep cached Phase B enrichment narrow and role-safe."""
    clean: Dict[str, List[str]] = {key: [] for key in framework_keys}
    dropped: Dict[str, List[str]] = {}
    anchor_set = set(anchor_keys)

    for key, values in classified_components.items():
        if key not in clean or not isinstance(values, list):
            continue
        for term in _dedupe_terms(values):
            if key in anchor_set:
                dropped.setdefault(key, []).append(term)
                continue
            if _is_low_signal_enrichment(term):
                dropped.setdefault(key, []).append(term)
                continue
            clean[key].append(term)
        clean[key] = _dedupe_terms(clean[key])

    if dropped:
        logger.info("[Phase B] Dropped unsafe cached/classified terms: %s", dropped)

    return clean


# ─── Data contracts ───────────────────────────────────────────────────────────

class ArticleRecord(TypedDict):
    """
    Canonical record for every article that passes through the pipeline.
    All fields are mandatory; consumers must not assume extra keys.
    """
    title: str
    abstract: str
    doi: str                                   # raw DOI as returned by source
    doi_normalized: str                        # https://doi.org/<suffix>, lowercase
    source: str                                # e.g. "PubMed", "OpenAlex"
    retrieval_phase: Literal["seed", "enriched", "both"]
    retrieval_query: str                       # exact query string used
    retrieval_date: str                        # ISO-8601 UTC
    url: Optional[str]
    year: Optional[int]
    authors: Optional[List[str]]
    journal: Optional[str]
    venue: Optional[str]
    pdf_url: Optional[str]
    open_access: Optional[bool]
    citations: Optional[int]
    language: Optional[str]
    paper_id: Optional[str]
    openalex_id: Optional[str]
    retrieval_mode: Optional[str]
    retrieval_queries: Optional[List[str]]


@dataclass
class BootstrapResult:
    """
    Return value of two_phase_search.
    Carries the corpus AND full methodological metadata for audit/reproduction.
    """
    corpus: List[ArticleRecord]
    pico_original: Dict[str, List[str]]
    pico_enriched: Dict[str, List[str]]
    bootstrap_executed: bool
    seed_count: int
    enriched_count: int
    final_count: int
    disabled_sources: List[str]
    cache_hit: bool
    query_audit: List[Dict] = field(default_factory=list)
    adaptive_lexicon_report: Dict[str, Any] = field(default_factory=dict)
    search_date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ─── DOI normalisation ────────────────────────────────────────────────────────

_DOI_PREFIX_RE = re.compile(
    r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", re.IGNORECASE
)

def normalize_doi(raw: str) -> str:
    """
    Collapse all known DOI URL/prefix variants to a canonical
    'https://doi.org/<suffix>' lowercase form.

    Examples
    --------
    '10.1145/1234'                  -> 'https://doi.org/10.1145/1234'
    'https://doi.org/10.1145/1234'  -> 'https://doi.org/10.1145/1234'
    'doi:10.1145/1234'              -> 'https://doi.org/10.1145/1234'
    'HTTP://DX.DOI.ORG/10.1145/...' -> 'https://doi.org/10.1145/...'
    ''                              -> ''
    """
    doi = (raw or "").strip()
    if not doi:
        return ""
    doi = _DOI_PREFIX_RE.sub("", doi).strip("/").lower()
    if not doi.startswith("10."):
        return ""  # not a valid DOI suffix
    return f"https://doi.org/{doi}"


# ─── Deduplication ────────────────────────────────────────────────────────────

def _tokenize_title(title: str) -> set:
    """
    Lowercase, strip accents, remove non-alpha, split to tokens.
    Used for Jaccard similarity dedup when DOI is absent.
    """
    nfkd = unicodedata.normalize("NFKD", title.lower())
    ascii_title = "".join(c for c in nfkd if not unicodedata.combining(c))
    tokens = re.findall(r"[a-z0-9]+", ascii_title)
    # Remove single-char tokens and stop-words that add noise
    stopwords = {"a", "an", "the", "of", "in", "on", "for", "and", "or",
                 "to", "with", "is", "are", "at", "by", "from"}
    return {t for t in tokens if len(t) > 1 and t not in stopwords}


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


JACCARD_THRESHOLD = 0.85  # titles this similar are considered the same paper


def deduplicate(
    seed: List[Dict],
    enriched: List[Dict],
) -> List[ArticleRecord]:
    """
    Merge seed + enriched corpora with three-tier dedup strategy:

    Tier 1 — Normalized DOI exact match.
    Tier 2 — Jaccard similarity on title tokens (threshold 0.85).
    Tier 3 — If article appears in both phases, mark retrieval_phase='both'.

    Returns a list of ArticleRecord with full provenance.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    def _to_record(art: Dict, phase: Literal["seed", "enriched"]) -> ArticleRecord:
        return ArticleRecord(
            title=art.get("title") or "",
            abstract=art.get("abstract") or "",
            doi=art.get("doi") or "",
            doi_normalized=normalize_doi(art.get("doi") or ""),
            source=art.get("source") or "unknown",
            retrieval_phase=phase,
            retrieval_query=art.get("retrieval_query") or "",
            retrieval_date=art.get("retrieval_date") or now_iso,
            url=art.get("url"),
            year=art.get("year"),
            authors=art.get("authors"),
            journal=art.get("journal"),
            venue=art.get("venue"),
            pdf_url=art.get("pdf_url"),
            open_access=art.get("open_access"),
            citations=art.get("citations"),
            language=art.get("language"),
            paper_id=art.get("paper_id") or art.get("paperId"),
            openalex_id=art.get("openalex_id") or art.get("openalexId"),
            retrieval_mode=art.get("retrieval_mode"),
            retrieval_queries=art.get("retrieval_queries"),
        )

    # Build maps: doi_norm -> record, title_tokens -> record
    doi_map: Dict[str, ArticleRecord] = {}
    title_map: List[tuple] = []  # list of (token_set, record)

    def _insert(art: Dict, phase: Literal["seed", "enriched"]) -> None:
        record = _to_record(art, phase)

        # Tier 1: DOI match
        if record["doi_normalized"]:
            existing = doi_map.get(record["doi_normalized"])
            if existing:
                if existing["retrieval_phase"] != phase:
                    existing["retrieval_phase"] = "both"
                return
            doi_map[record["doi_normalized"]] = record
            title_map.append((_tokenize_title(record["title"]), record))
            return

        # Tier 2: Jaccard match (only for DOI-less records)
        tokens = _tokenize_title(record["title"])
        if tokens:
            for existing_tokens, existing_record in title_map:
                if _jaccard(tokens, existing_tokens) >= JACCARD_THRESHOLD:
                    if existing_record["retrieval_phase"] != phase:
                        existing_record["retrieval_phase"] = "both"
                    return

        # No match found — it's a new unique record
        doi_map[f"__no_doi_{len(doi_map)}"] = record
        title_map.append((tokens, record))

    for art in seed:
        _insert(art, "seed")
    for art in enriched:
        _insert(art, "enriched")

    return list(doi_map.values())


# ─── Cache helpers ────────────────────────────────────────────────────────────

def _cache_key(question: str, prompt_template: str) -> str:
    """
    Key = MD5( PROMPT_VERSION + question + prompt_template ).
    Changing the prompt or the version string automatically invalidates
    all existing cache entries without manual cleanup.
    """
    raw = f"{PROMPT_VERSION}|{question.strip().lower()}|{prompt_template}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_load(key: str) -> Optional[Dict]:
    path = os.path.join(CACHE_DIR, f"boot_{key}.json")
    if not os.path.exists(path):
        return None
    try:
        mtime = os.path.getmtime(path)
        age = datetime.now(timezone.utc).timestamp() - mtime
        if age > CACHE_TTL_SECONDS:
            os.remove(path)
            logger.info("🗑 [Cache] Entrada expirada eliminada: %s", key)
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("⚠️ [Cache] Error al leer %s: %s", key, e)
        return None


def _cache_save(key: str, data: Dict) -> None:
    path = os.path.join(CACHE_DIR, f"boot_{key}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        logger.warning("⚠️ [Cache] Error al guardar %s: %s", key, e)


# ─── Domain-agnostic LLM classification prompt ────────────────────────────────

def _build_classification_prompt(
    original_question: str,
    framework: str,
    original_components: Dict[str, List[str]],
    top_terms: List[str],
) -> str:
    """
    Build a fully domain-agnostic prompt.

    The framework label (PICO, SPIDER, PECO, etc.) and its components
    are derived at runtime from the actual question — never hardcoded here.
    No domain-specific examples are injected.
    """
    components_block = "\n".join(
        f"- {k}: {v}" for k, v in original_components.items()
    )
    component_keys = list(original_components.keys())

    return f"""You are a systematic review librarian with expertise in evidence synthesis.

Research Question:
"{original_question}"

Search Framework: {framework}
Known components already identified:
{components_block}

Below are the {len(top_terms)} most statistically significant n-grams (unigrams, bigrams, trigrams)
extracted via TF-IDF from the titles and abstracts of a seed corpus retrieved for this question:

{top_terms}

Task:
From the n-grams above, identify terms that belong to the MISSING or UNDER-SPECIFIED
components of the {framework} framework for this specific research question.
The components that still need enrichment are: {component_keys}.

Rules:
- Only use terms that appear in or closely match the provided n-gram list.
- Adapt them to academic database query format (e.g. MeSH-style, controlled vocabulary).
- Do NOT invent terms outside the provided list.
- Do NOT include terms already covered by the known components above.
- Output ONLY a valid JSON object mapping each {framework} component key to a list of strings.
- No markdown, no explanation, no preamble.

Output format:
{{
  {', '.join(f'"{k}": []' for k in component_keys)}
}}"""


# ─── Main pipeline ────────────────────────────────────────────────────────────

async def two_phase_search(
    original_question: str,
    client_id: str = None,
    framework: str = "PICO",
    tfidf_max_features: int = 150,
    tfidf_top_n: int = 120,
    jaccard_threshold: float = JACCARD_THRESHOLD,
    eligibility_contract: Optional[Dict] = None,
) -> BootstrapResult:
    """
    Two-Phase Search with TF-IDF Bootstrapping for academic SLR.

    Phase A — Broad search with high-recall components only (P + I).
    Phase B — Extract real vocabulary (TF-IDF n-grams) and classify
               missing framework components via LLM (domain-agnostic).
    Phase C — Parallel enriched search with the bootstrapped framework.

    Returns a BootstrapResult with full methodological provenance.
    Raises ValueError if the question or PICO generation fails validation.
    """
    logger.info("🌊 [Two-Phase] Starting Phase A — broad search")

    # ── 1. Generate initial framework components ──────────────────────────────
    pico = _build_initial_framework(original_question, eligibility_contract)

    if not isinstance(pico, dict) or not pico:
        raise ValueError(
            f"[Two-Phase] PICO/framework generation returned an invalid "
            f"structure: {pico!r}. Check generate_api_queries_with_llm."
        )

    # High-recall components for Phase A: first two keys of the framework.
    # Works for PICO (P+I), SPIDER (S+PI), PECO (P+E), etc.
    framework_keys = [
        k for k in pico.keys()
        if k != "semantic_queries" and not str(k).startswith("_")
    ]
    broad_keys = framework_keys[:2]
    broad_components = {k: pico.get(k, []) for k in broad_keys}
    broad_pico = _build_phase_a_framework(pico, framework_keys)
    query_audit = build_search_query_audit(broad_pico, original_question, phase="phase_a")

    logger.info("🔍 [Phase A] Components: %s", broad_components)
    seed_corpus = await run_parallel_search(
        original_question, client_id=client_id, pico=broad_pico
    )

    if not seed_corpus:
        logger.warning("⚠️ [Phase A] No articles returned. Falling back to full framework.")
        # Return a BootstrapResult that clearly signals bootstrap did NOT run.
        return BootstrapResult(
            corpus=[],
            pico_original=pico,
            pico_enriched=pico,
            bootstrap_executed=False,
            seed_count=0,
            enriched_count=0,
            final_count=0,
            disabled_sources=[],
            cache_hit=False,
            query_audit=query_audit,
            adaptive_lexicon_report={},
        )

    logger.info("✅ [Phase A] %d seed articles retrieved.", len(seed_corpus))

    # ── 2. Dynamic source monitoring — derived from runtime config ────────────
    # get_registered_sources() returns the list of source names from the
    # search engine's own configuration. No names are hardcoded here.
    registered_sources = get_registered_sources()
    sources_in_seed = {art.get("source") for art in seed_corpus if art.get("source")}
    disabled_sources = [
        s for s in registered_sources if s not in sources_in_seed
    ]
    if disabled_sources:
        logger.warning(
            "⚠️ [Dynamic Bypass] Sources absent from Phase A (will skip in C): %s",
            disabled_sources,
        )

    # ── 3. Phase B — TF-IDF vocabulary extraction ────────────────────────────
    logger.info("🧪 [Phase B] Extracting n-grams via TF-IDF...")
    texts = [
        f"{a.get('title', '')} {a.get('abstract', '')}".strip()
        for a in seed_corpus
        if (a.get("title") or a.get("abstract"))
    ]

    top_terms: List[str] = []
    if texts:
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=tfidf_max_features,
                stop_words="english",
                min_df=max(1, min(2, len(texts))),
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            mean_scores = tfidf_matrix.mean(axis=0).A1
            term_scores = sorted(
                zip(feature_names, mean_scores), key=lambda x: x[1], reverse=True
            )
            top_terms = [t for t, _ in term_scores[:tfidf_top_n]]
            logger.info("📊 [Phase B] %d significant n-grams extracted.", len(top_terms))
        except Exception as e:
            logger.error("❌ [Phase B] TF-IDF failed: %s", e)

    # ── 4. LLM classification — domain-agnostic, cache-safe ──────────────────
    llm_classified: Dict[str, List[str]] = {k: [] for k in framework_keys}
    cache_hit = False

    if top_terms:
        prompt_template = _build_classification_prompt(
            original_question=original_question,
            framework=framework,
            original_components=pico,
            top_terms=top_terms,
        )
        cache_key = _cache_key(original_question, prompt_template)
        cached = _cache_load(cache_key)

        if cached:
            llm_classified = _sanitize_classified_components(cached, framework_keys, broad_keys)
            cache_hit = True
            logger.info("🔑 [Cache] Classification loaded from cache (key=%s).", cache_key)
        else:
            logger.info("🤖 [Phase B] Calling LLM for framework component classification...")
            response = generate_text(
                instruction=prompt_template,
                input_text="",
                max_tokens=1024,
                system_prompt="You are a systematic review librarian. Output only valid JSON.",
            )

            if response and "Error de generación" not in response and "⚠️" not in response:
                clean = response.replace("```json", "").replace("```", "").strip()
                parsed = None
                try:
                    parsed = json.loads(clean)
                except json.JSONDecodeError:
                    match = re.search(r"\{.*\}", clean, re.DOTALL)
                    if match:
                        try:
                            parsed = json.loads(match.group())
                        except json.JSONDecodeError:
                            logger.warning("⚠️ [Phase B] Could not parse LLM JSON after regex fallback.")

                if parsed and isinstance(parsed, dict):
                    # Only accept keys that are valid framework components
                    raw_classified = {
                        k: v for k, v in parsed.items()
                        if k in framework_keys and isinstance(v, list)
                    }
                    llm_classified = _sanitize_classified_components(raw_classified, framework_keys, broad_keys)
                    if any(llm_classified.values()):
                        _cache_save(cache_key, llm_classified)
                    else:
                        logger.info("[Cache] Phase B classification not cached: no valid enrichment terms.")
                else:
                    logger.warning("⚠️ [Phase B] LLM returned unparseable output. Proceeding without enrichment.")

    # ── 5. Enriched PICO + Phase C ────────────────────────────────────────────
    enhanced_pico = _build_enriched_framework(
        original_components=pico,
        classified_components=llm_classified,
        framework_keys=framework_keys,
        anchor_keys=broad_keys,
    )
    enhanced_pico, adaptive_lexicon_report = augment_framework_with_empirical_lexicon(
        enhanced_pico,
        original_question,
        seed_corpus,
        eligibility_contract,
    )

    logger.info("✨ [Phase B→C] Enriched framework components:")
    for k, v in enhanced_pico.items():
        logger.info("   %s: %s", k, v)
    query_audit.extend(build_search_query_audit(enhanced_pico, original_question, phase="phase_c"))

    logger.info("🌊 [Phase C] Starting enriched parallel search...")
    phase_c_timeout = 90.0 if len(seed_corpus) > 50 else 150.0
    logger.info("⏱️ [Phase C] Dynamic timeout set to %.1fs (seed size: %d)", phase_c_timeout, len(seed_corpus))
    enriched_corpus = await run_parallel_search(
        original_question,
        client_id=client_id,
        pico=enhanced_pico,
        disabled_sources=[],  # Dejar que todas las fuentes intenten en C (Fase C)
        custom_timeout=phase_c_timeout,
    )
    logger.info("✅ [Phase C] %d articles retrieved.", len(enriched_corpus))

    # ── 6. Deduplicate with full provenance ───────────────────────────────────
    final_corpus = deduplicate(seed_corpus, enriched_corpus)
    logger.info(
        "📚 [Dedup] seed=%d enriched=%d → final=%d unique articles.",
        len(seed_corpus),
        len(enriched_corpus),
        len(final_corpus),
    )

    return BootstrapResult(
        corpus=final_corpus,
        pico_original=pico,
        pico_enriched=enhanced_pico,
        bootstrap_executed=True,
        seed_count=len(seed_corpus),
        enriched_count=len(enriched_corpus),
        final_count=len(final_corpus),
        disabled_sources=disabled_sources,
        cache_hit=cache_hit,
        query_audit=query_audit,
        adaptive_lexicon_report=adaptive_lexicon_report,
    )
