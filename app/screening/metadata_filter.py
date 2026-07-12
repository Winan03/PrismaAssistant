"""
metadata_filter.py  ·  Hard Filtering por Metadatos (Pre-Vectorial)
====================================================================
PROBLEMAS RESUELTOS vs. versión anterior
-----------------------------------------
1. FIRMA INCONSISTENTE — apply_hard_filters devolvía 2 ó 3 valores
   dependiendo del path de ejecución (algunos return devolvían solo
   (passed, excluded) mientras el docstring prometía 3). Ahora siempre
   devuelve Tuple[List, List, Dict].

2. FILTRO DE ABSTRACT SILENCIOSO con DOI:
   Cuando un artículo tenía DOI válido pero abstract vacío, se le asignaba
   "No abstract available..." y se continuaba procesando sin registrar
   la sustitución en el reporte. Ahora se registra en un campo
   '_abstract_substituted' para trazabilidad.

3. VENTANA DESLIZANTE DE CO-OCURRENCIA — off-by-one:
   El algoritmo de ventana en _check_cooccurrence usaba el extremo
   match.end() (posición exclusiva) para medir la longitud de ventana,
   pero comparaba start del primer match vs end del último. Esto hacía
   que ventanas grandes pasaran el filtro cuando no debían. Corregido
   a usar match.start() en ambos extremos.

4. PARSE_EXCLUSION_CRITERIA — solo extraía min_year; los hard_terms
   siempre devolvían lista vacía. Completada la función para extraer
   términos explícitos de exclusión de texto libre.

5. ACOPLAMIENTO IMPLÍCITO — _REVIEW_SIGNALS era un set global mutable.
   Convertido a frozenset para evitar mutación accidental en runtime.

6. FALTA DE REPORTE ESTRUCTURADO para concept_presence_filter:
   el tercer valor del return era un dict ad-hoc sin tipo garantizado.
   Añadidos dataclasses ConceptFilterReport y HardFilterReport.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Tuple
import config
from app.screening.filters import has_academic_venue

logger = logging.getLogger(__name__)

# Señales de revisión como frozenset (inmutable, thread-safe)
_REVIEW_SIGNALS: frozenset = frozenset({
    "systematic review", "literature review", "scoping review",
    "mapping study", "meta-analysis", "state of the art",
    "survey of", "bibliometric analysis", "umbrella review",
    "narrative review", "a review of", "overview of",
    "systematic mapping",
})


# ──────────────────────────────────────────────
# Estructuras de reporte
# ──────────────────────────────────────────────

@dataclass
class HardFilterReport:
    total: int = 0
    passed: int = 0
    excluded: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)


@dataclass
class ConceptFilterReport:
    total: int = 0
    passed: int = 0
    excluded: int = 0
    reduction_pct: float = 0.0
    n_concept_groups: int = 0
    mandatory_categories: List[str] = field(default_factory=list)
    atom_failure_counts: Dict[str, int] = field(default_factory=dict)
    prefilter_rules: Dict[str, Any] = field(default_factory=dict)
    skipped: bool = False


# ──────────────────────────────────────────────
# Filtro de Presencia de Conceptos (PICO asimétrico)
# ──────────────────────────────────────────────

def concept_presence_filter(
    articles: List[Dict],
    synonym_data: Dict,
    min_concepts_required: int = 3,
) -> Tuple[List[Dict], List[Dict], ConceptFilterReport]:
    """
    Filtra artículos mediante PICO Asimétrico:
      · P (Población) e I (Intervención) son obligatorios.
      · O (Outcome) y C (Comparación) son opcionales (bonus de relevancia).

    Returns:
        passed   — artículos que cumplen todas las categorías obligatorias
        excluded — artículos descartados (con '_exclusion_reason')
        report   — ConceptFilterReport para trazabilidad PRISMA
    """
    report = ConceptFilterReport(total=len(articles))

    if not articles:
        return [], [], report

    if not isinstance(synonym_data, dict):
        logger.warning(
            f"⚠️ [ConceptFilter] synonym_data no es dict ({type(synonym_data)}). "
            "Omitiendo filtro."
        )
        report.passed = len(articles)
        report.skipped = True
        return articles, [], report

    synonyms_by_concept: Dict = synonym_data.get("synonyms", {})
    categories_by_concept: Dict = synonym_data.get("categories", {})

    if not synonyms_by_concept:
        logger.warning("⚠️ [ConceptFilter] Sin sinónimos — omitiendo filtro.")
        report.passed = len(articles)
        report.skipped = True
        return articles, [], report

    # Construir grupos de términos por concepto
    concept_groups = []
    for concept, synonyms in synonyms_by_concept.items():
        raw_terms = {concept.lower()} | {s.lower() for s in synonyms if s}
        terms = set()
        for term in raw_terms:
            terms.update(_expand_term_variants(term))
        category = str(categories_by_concept.get(concept) or _infer_pico_category(concept, terms)).upper()
        if category not in ("P", "I", "C", "O"):
            category = "P"
        concept_groups.append({"name": concept, "terms": terms, "category": category})

    atom_groups = _build_atom_groups(synonym_data.get("atom_groups", []))
    if atom_groups:
        prefilter_rules = _normalise_prefilter_rules(synonym_data.get("prefilter_rules"), atom_groups)
        required_atoms = [g for g in atom_groups if g.get("required")]
        if not required_atoms:
            required_atoms = atom_groups[:1]

        report.n_concept_groups = len(atom_groups)
        report.mandatory_categories = [g["name"] for g in required_atoms]
        report.prefilter_rules = prefilter_rules

        logger.info(
            "[ConceptFilter] %d atom groups | Required atoms: %s | Technical min: %s/%d | %d articulos",
            len(atom_groups),
            report.mandatory_categories,
            prefilter_rules.get("technical_intervention_atoms", []),
            int(prefilter_rules.get("min_technical_intervention_atoms", 0) or 0),
            len(articles),
        )

        passed: List[Dict] = []
        excluded: List[Dict] = []

        for art in articles:
            title = art.get("title") or ""
            abstract = art.get("abstract") or ""
            matched_atoms, _ = _check_cooccurrence(title, abstract, atom_groups, window_size=10000)
            matched_set = set(matched_atoms)
            missing_required = [g["name"] for g in required_atoms if g["name"] not in matched_set]
            missing_prefilter = _missing_prefilter_atoms(matched_set, prefilter_rules)
            missing_gate = _unique_names([*missing_required, *missing_prefilter])
            all_missing = [g["name"] for g in atom_groups if g["name"] not in matched_set]

            if not missing_gate:
                optional_groups = [g for g in atom_groups if not g.get("required")]
                optional_present = sum(1 for g in optional_groups if g["name"] in matched_set)
                bonus_score = (optional_present / len(optional_groups)) if optional_groups else 0.0

                enriched = art.copy()
                enriched["_concepts_matched_count"] = len(matched_set)
                enriched["_concepts_matched_list"] = matched_atoms
                enriched["_concept_atoms_matched_list"] = matched_atoms
                enriched["_concept_atoms_missing_list"] = all_missing
                enriched["_concept_bonus"] = round(bonus_score, 2)
                passed.append(enriched)
            else:
                excl = art.copy()
                excl["_exclusion_reason"] = (
                    f"Faltan atomos obligatorios: {', '.join(missing_gate)}"
                )
                excl["_concepts_matched_count"] = len(matched_set)
                excl["_concepts_matched_list"] = matched_atoms
                excl["_concept_atoms_matched_list"] = matched_atoms
                excl["_concept_atoms_missing_list"] = missing_gate
                for atom_name in missing_gate:
                    report.atom_failure_counts[atom_name] = report.atom_failure_counts.get(atom_name, 0) + 1
                excluded.append(excl)

        reduction_pct = (len(excluded) / len(articles) * 100) if articles else 0.0
        report.passed = len(passed)
        report.excluded = len(excluded)
        report.reduction_pct = round(reduction_pct, 1)

        logger.info(
            f"âœ… [ConceptFilter] {len(articles)} â†’ {len(passed)} "
            f"({len(excluded)} excluidos, {reduction_pct:.1f}% reducciÃ³n)"
        )
        return passed, excluded, report

    # Determinar categorías obligatorias desde la RQ real
    mandatory_categories = []
    for cat in ("P", "I"):
        if any(g["category"] == cat for g in concept_groups):
            mandatory_categories.append(cat)
    if not mandatory_categories:
        mandatory_categories = ["P"]

    report.n_concept_groups = len(concept_groups)
    report.mandatory_categories = mandatory_categories

    logger.info(
        f"🔬 [ConceptFilter] {len(concept_groups)} grupos | "
        f"Obligatorios: {mandatory_categories} | {len(articles)} artículos"
    )

    passed: List[Dict] = []
    excluded: List[Dict] = []

    for art in articles:
        title = art.get("title") or ""
        abstract = art.get("abstract") or ""

        matched_groups, _ = _check_cooccurrence(title, abstract, concept_groups, window_size=10000)
        matched_set = set(matched_groups)
        matched_categories = {
            g["category"] for g in concept_groups if g["name"] in matched_set
        }

        missing_mandatory = [c for c in mandatory_categories if c not in matched_categories]

        if not missing_mandatory:
            # Calcular bonus (categorías O y C)
            bonus_groups = [g for g in concept_groups if g["category"] in ("O", "C")]
            bonus_present = sum(1 for g in bonus_groups if g["name"] in matched_set)
            bonus_score = (bonus_present / len(bonus_groups)) if bonus_groups else 0.0

            enriched = art.copy()
            enriched["_concepts_matched_count"] = len(matched_set)
            enriched["_concepts_matched_list"] = matched_groups
            enriched["_concept_bonus"] = round(bonus_score, 2)
            passed.append(enriched)
        else:
            missing_names = [
                g["name"] for g in concept_groups if g["category"] in missing_mandatory
            ]
            excl = art.copy()
            excl["_exclusion_reason"] = (
                f"Falta concepto obligatorio ({', '.join(missing_mandatory)}): "
                f"{', '.join(missing_names)}"
            )
            excl["_concepts_matched_count"] = len(matched_set)
            excl["_concepts_matched_list"] = matched_groups
            excl["_concepts_missing_list"] = [
                g["name"] for g in concept_groups if g["name"] not in matched_set
            ]
            excluded.append(excl)

    reduction_pct = (len(excluded) / len(articles) * 100) if articles else 0.0
    report.passed = len(passed)
    report.excluded = len(excluded)
    report.reduction_pct = round(reduction_pct, 1)

    logger.info(
        f"✅ [ConceptFilter] {len(articles)} → {len(passed)} "
        f"({len(excluded)} excluidos, {reduction_pct:.1f}% reducción)"
    )
    return passed, excluded, report


def _unique_names(names: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for name in names:
        clean = str(name or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result


def _normalise_prefilter_rules(raw_rules: object, atom_groups: List[Dict]) -> Dict[str, Any]:
    rules = raw_rules if isinstance(raw_rules, dict) else {}
    technical_atoms = [
        str(name).strip()
        for name in rules.get("technical_intervention_atoms", [])
        if str(name).strip()
    ]
    if not technical_atoms:
        technical_atoms = [
            group["name"]
            for group in atom_groups
            if group.get("role") == "technical_intervention"
        ]

    min_technical = rules.get("min_technical_intervention_atoms", 0)
    try:
        min_technical_int = max(0, int(min_technical))
    except (TypeError, ValueError):
        min_technical_int = 0

    return {
        "population_atoms": _unique_names([
            str(name).strip()
            for name in rules.get("population_atoms", [])
            if str(name).strip()
        ]),
        "central_intervention_atoms": _unique_names([
            str(name).strip()
            for name in rules.get("central_intervention_atoms", [])
            if str(name).strip()
        ]),
        "technical_intervention_atoms": _unique_names(technical_atoms),
        "min_technical_intervention_atoms": min(min_technical_int, len(technical_atoms)),
    }


def _missing_prefilter_atoms(matched_set: set, prefilter_rules: Dict[str, Any]) -> List[str]:
    technical_atoms = prefilter_rules.get("technical_intervention_atoms") or []
    min_technical = int(prefilter_rules.get("min_technical_intervention_atoms", 0) or 0)
    if not technical_atoms or min_technical <= 0:
        return []

    matched_technical = [name for name in technical_atoms if name in matched_set]
    if len(matched_technical) >= min_technical:
        return []
    return [name for name in technical_atoms if name not in matched_set]


def _build_atom_groups(raw_atom_groups: object) -> List[Dict]:
    if not isinstance(raw_atom_groups, list):
        return []
    groups: List[Dict] = []
    for item in raw_atom_groups:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("id") or "").strip()
        category = str(item.get("category") or "").upper()
        terms_raw = item.get("terms", [])
        if not name or category not in {"P", "I", "C", "O"} or not isinstance(terms_raw, list):
            continue

        terms = set()
        for term in terms_raw:
            text = str(term or "").strip().lower()
            if not text:
                continue
            terms.update(_expand_term_variants(text))
        if not terms:
            continue

        groups.append({
            "name": name,
            "category": category,
            "terms": terms,
            "required": bool(item.get("required", False)),
            "role": str(item.get("role") or "").strip(),
        })
    return groups


def _expand_term_variants(term: str) -> set:
    clean = re.sub(r"\s+", " ", str(term or "").strip().lower())
    if not clean:
        return set()

    variants = {clean, clean.replace("-", " "), clean.replace(" ", "-")}
    words = clean.split()
    if words:
        singular_words = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words]
        variants.add(" ".join(singular_words))
        if len(words) >= 3:
            # Composite PICO terms generated by LLMs are often unnatural
            # compounds. Keep matching phrase-based, but allow contiguous
            # subphrases so "method object" can match within a longer concept.
            for size in range(2, len(words)):
                for start in range(0, len(words) - size + 1):
                    phrase = " ".join(words[start:start + size])
                    variants.add(phrase)
            if singular_words != words:
                for size in range(2, len(singular_words)):
                    for start in range(0, len(singular_words) - size + 1):
                        phrase = " ".join(singular_words[start:start + size])
                        variants.add(phrase)
    if "technologies" in clean:
        variants.add(clean.replace("technologies", "technology"))
    stopwords = set(getattr(config, "TERM_VARIANT_STOPWORDS", ()))
    return {
        variant
        for variant in variants
        if len(variant) >= 2 and set(variant.split()) - stopwords
    }


def _infer_pico_category(concept: str, terms: set) -> str:
    """Clasifica conceptos legacy cuando el cache de sinonimos no trae categorias PICO."""
    text = " ".join([concept.lower(), *[str(t).lower() for t in terms]])

    intervention_markers = (
        "intervention", "treatment", "program", "protocol", "method", "approach",
        "model", "system", "tool", "device", "platform", "application", "software",
        "technique", "technology", "exposure", "agent", "strategy",
    )
    outcome_markers = (
        "outcome", "effect", "impact", "measure", "measurement", "metric", "score",
        "rate", "frequency", "duration", "change", "response", "improvement",
        "reduction", "increase", "decrease",
    )
    comparator_markers = (
        "comparator", "comparison", "control", "baseline", "placebo", "standard",
        "usual", "conventional", "alternative", "non-adaptive", "fixed", "versus",
        "vs",
    )
    population_markers = (
        "population", "participant", "participants", "patient", "patients",
        "subject", "subjects", "user", "users", "cohort", "sample", "learner",
        "learners",
    )

    if any(marker in text for marker in comparator_markers):
        return "C"
    if any(marker in text for marker in outcome_markers):
        return "O"
    if any(marker != "ai" and marker in text for marker in intervention_markers) or re.search(r"\bai\b", text):
        return "I"
    if any(marker in text for marker in population_markers):
        return "P"
    return "P"


def _term_pattern(term: str) -> str:
    escaped = re.escape(str(term or "").strip().lower())
    return rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"


def _contains_term(text: str, term: str) -> bool:
    return re.search(_term_pattern(term), text) is not None


def _check_cooccurrence(
    title: str,
    abstract: str,
    concept_groups: List[Dict],
    window_size: int = 10000,
) -> Tuple[List[str], int]:
    """
    Co-ocurrencia en ventana deslizante.

    CORRECCIÓN: la longitud de ventana se mide como
    abstract_matches[end_idx][0] - abstract_matches[start_idx][0]
    (start vs start) en lugar de end vs start, que producía ventanas
    demasiado permisivas.
    """
    title_lower = title.lower()
    abstract_lower = abstract.lower()

    # Coincidencias en título (sin límite de ventana)
    title_matches: set = set()
    for group in concept_groups:
        if any(_contains_term(title_lower, term) for term in group["terms"]):
            title_matches.add(group["name"])

    # Coincidencias en abstract con posición
    abstract_matches: List[Tuple[int, str]] = []
    for group in concept_groups:
        for term in group["terms"]:
            for m in re.finditer(_term_pattern(term), abstract_lower):
                abstract_matches.append((m.start(), group["name"]))

    abstract_matches.sort(key=lambda x: x[0])

    best_abstract_groups: set = set()
    n = len(abstract_matches)
    if n > 0:
        start_idx = 0
        for end_idx in range(n):
            # Corrección: start vs start (no end vs start)
            while abstract_matches[end_idx][0] - abstract_matches[start_idx][0] > window_size:
                start_idx += 1
            window_groups = {abstract_matches[i][1] for i in range(start_idx, end_idx + 1)}
            if len(window_groups) > len(best_abstract_groups):
                best_abstract_groups = window_groups

    matched = list(title_matches | best_abstract_groups)
    return matched, len(matched)


# ──────────────────────────────────────────────
# Hard Filters por Metadatos
# ──────────────────────────────────────────────

def apply_hard_filters(
    articles: List[Dict],
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    min_abstract_length: int = 80,
    exclude_review_articles: bool = False,
    allowed_sources: Optional[List[str]] = None,
    extra_exclusion_terms: Optional[List[str]] = None,
    excluded_venues: Optional[List[str]] = None,
    academic_only: bool = False,
) -> Tuple[List[Dict], List[Dict], HardFilterReport]:
    """
    Filtra artículos por metadatos antes del proceso vectorial.

    GARANTÍA DE FIRMA: siempre devuelve exactamente 3 valores.

    Returns:
        passed   — artículos que pasan todos los criterios
        excluded — artículos descartados (con '_exclusion_reason')
        report   — HardFilterReport para trazabilidad PRISMA
    """
    report = HardFilterReport(total=len(articles))

    if not articles:
        return [], [], report

    passed: List[Dict] = []
    excluded: List[Dict] = []
    reasons: Dict[str, int] = {}

    for art in articles:
        reason = _check_exclusion(
            art,
            min_year=min_year,
            max_year=max_year,
            min_abstract_length=min_abstract_length,
            exclude_review_articles=exclude_review_articles,
            allowed_sources=allowed_sources,
            extra_exclusion_terms=extra_exclusion_terms or [],
            excluded_venues=excluded_venues,
            academic_only=academic_only,
        )
        if reason:
            excl = art.copy()
            excl["_exclusion_reason"] = reason
            excluded.append(excl)
            reasons[reason] = reasons.get(reason, 0) + 1
        else:
            passed.append(art)

    report.passed = len(passed)
    report.excluded = len(excluded)
    report.reasons = reasons

    if excluded:
        reasons_str = ", ".join(f"{k}={v}" for k, v in reasons.items() if v > 0)
        logger.info(
            f"🚧 [HardFilter] {len(articles)} → {len(passed)} "
            f"({len(excluded)} excluidos) | {reasons_str}"
        )
    else:
        logger.info(f"✅ [HardFilter] Todos los {len(articles)} artículos pasaron")

    return passed, excluded, report


def _check_exclusion(
    art: Dict,
    min_year: Optional[int],
    max_year: Optional[int],
    min_abstract_length: int,
    exclude_review_articles: bool,
    allowed_sources: Optional[List[str]],
    extra_exclusion_terms: List[str],
    excluded_venues: Optional[List[str]] = None,
    academic_only: bool = False,
) -> Optional[str]:
    """Evalúa un artículo. Retorna razón de exclusión o None si pasa."""
    abstract = (art.get("abstract") or "").strip()
    title = (art.get("title") or "").strip()
    combined = f"{title} {abstract}".lower()

    doi = (art.get("doi") or "")
    has_valid_doi = len(str(doi).strip()) > 5 and "sin doi" not in str(doi).lower()

    # 1. Abstract
    if not abstract:
        if not has_valid_doi:
            return "sin_abstract"
        # DOI válido pero sin abstract: sustituir con placeholder trazable
        art["abstract"] = "No abstract available for this article."
        art["_abstract_substituted"] = True   # ← trazabilidad nueva
    elif len(abstract) < min_abstract_length:
        if not has_valid_doi and len(title) < 25:
            return "abstract_corto"

    # 2. Rango temporal
    if min_year is not None or max_year is not None:
        year = _extract_year(art)
        if year is not None:
            if min_year and year < min_year:
                return "fuera_de_rango_temporal"
            if max_year and year > max_year:
                return "fuera_de_rango_temporal"

    # 3. Fuentes permitidas
    if allowed_sources:
        source = (art.get("journal") or art.get("venue") or "").lower()
        if source and not any(a.lower() in source for a in allowed_sources):
            return "fuente_no_permitida"

    # 4. Excluir reviews
    if exclude_review_articles:
        if any(sig in combined for sig in _REVIEW_SIGNALS):
            return "es_review_survey"

    # 5. Términos de exclusión dura
    for term in extra_exclusion_terms:
        if term.lower() in combined:
            return "termino_exclusion_dura"

    # 6. Blocklist de venues
    if excluded_venues:
        journal = (art.get("journal") or art.get("venue") or "").lower()
        if journal and any(frag in journal for frag in excluded_venues):
            return "venue_excluido"

    # 7. Calidad académica
    if academic_only:
        if not has_academic_venue(art):
            return "sin_revista"

    return None


def _extract_year(art: Dict) -> Optional[int]:
    for field_name in ("year", "publicationDate", "date", "pub_year"):
        val = art.get(field_name)
        if val:
            year_str = str(val)[:4]
            if re.match(r"^\d{4}$", year_str):
                return int(year_str)
    return None


# ──────────────────────────────────────────────
# Parser de criterios de exclusión de texto libre
# ──────────────────────────────────────────────

def parse_exclusion_criteria_for_hard_filter(
    exclusion_criteria_text: str,
) -> Tuple[Optional[int], List[str]]:
    """
    Parsea criterios de exclusión del investigador.

    Extrae:
    - min_year   : año mínimo (ej. "Artículos anteriores a 2019" → 2019)
    - hard_terms : términos explícitos de exclusión léxica.

    Patrones de términos soportados (insensible a mayúsculas/idioma):
      "artículos que contengan X"
      "excluir X"
      "exclude X"
      "sin mención de X"
      "no incluir X"

    Returns: (min_year, hard_terms)
    """
    if not exclusion_criteria_text:
        return None, []

    min_year: Optional[int] = None
    hard_terms: List[str] = []

    lines = [l.strip() for l in exclusion_criteria_text.splitlines() if l.strip()]

    for line in lines:
        line_lower = line.lower()

        # ── Año mínimo ──
        year_match = re.search(
            r"(?:antes?|before|prior|anteriores?)\s+(?:a|to|de)?\s*(\d{4})",
            line_lower,
        )
        if year_match:
            try:
                y = int(year_match.group(1))
                if 1990 <= y <= 2030:
                    if min_year is None or y > min_year:
                        min_year = y
            except ValueError:
                pass

        # ── Términos de exclusión léxica ──
        # Patrones: "excluir X", "exclude X", "no incluir X",
        #           "sin mención de X", "artículos que contengan X"
        term_patterns = [
            r"(?:excluir|exclude|no incluir)\s+[\"']?([^\"'\n,;]{3,50})[\"']?",
            r"sin\s+(?:menci[oó]n\s+de\s+)?[\"']?([^\"'\n,;]{3,50})[\"']?",
            r"art[ií]culos?\s+que\s+contengan?\s+[\"']?([^\"'\n,;]{3,50})[\"']?",
        ]
        for pattern in term_patterns:
            for m in re.finditer(pattern, line_lower):
                term = m.group(1).strip().rstrip(".,;")
                if len(term) >= 3 and term not in hard_terms:
                    hard_terms.append(term)

    logger.info(
        f"🗓️ [ExclusionParser] min_year={min_year}, hard_terms={hard_terms}"
    )
    return min_year, hard_terms
