"""
metadata_filter.py - Hard Filtering por Metadatos (Pre-Vectorial)
=================================================================
Aplica filtros ESTRICTOS sobre los metadatos del artículo ANTES de
cualquier cálculo vectorial o de embeddings.

Ventaja: Reduce drásticamente la superficie donde pueden ocurrir
falsos positivos, porque excluye artículos que incumplen criterios
objetivos (año, tipo de documento, longitud de abstract) sin necesidad
de embeddings ni LLM.
"""

import logging
import re
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_REVIEW_SIGNALS = {
    "systematic review", "literature review", "scoping review",
    "mapping study", "meta-analysis", "state of the art",
    "survey of", "bibliometric analysis", "umbrella review",
    "narrative review", "a review of", "overview of",
    "systematic mapping",
}


# ============================================================
# FILTRO DE PRESENCIA DE CONCEPTOS (Post-API, Pre-ChromaDB)
# ============================================================

def concept_presence_filter(
    articles: List[Dict],
    synonym_data: Dict,
    min_concepts_required: int = 2,
) -> Tuple[List[Dict], Dict]:
    """
    Filtra artículos que NO contienen términos de los conceptos clave en su
    título+abstract. Se aplica DESPUÉS de las APIs y ANTES de ChromaDB.

    Estrategia (sugerida por el profesor):
      Si un abstract no contiene al menos 1 sinónimo de CADA concepto clave
      → el artículo es descartado antes de llegar al embedding.

    Esto reduce 878 artículos a los ~300-400 que realmente tocan el tema,
    evitando contaminar ChromaDB con ruido.

    Args:
        articles:              Lista de artículos del corpus crudo (post-API)
        synonym_data:          Output de expand_query_with_synonyms()
                               {"synonyms": {concepto: [sinónimos]}, "flat_terms": [...]}
        min_concepts_required: Mínimo de conceptos que deben estar presentes (default: 2)
                               Con 3 conceptos en la RQ:
                                 - 2 = filtro moderado (cualquier 2 de los 3)
                                 - 3 = filtro estricto (obligatoriamente los 3)

    Returns:
        Tuple:
          - filtered_articles (List[Dict]): Artículos que pasaron
          - report (Dict):                  Estadísticas del filtrado
    """
    if not articles:
        return [], {"total": 0, "passed": 0, "excluded": 0}

    synonyms_by_concept = synonym_data.get("synonyms", {})

    if not synonyms_by_concept:
        # Sin datos de sinónimos: pasar todos sin filtrar
        logger.warning("⚠️ [Concept Filter] Sin datos de sinónimos — omitiendo filtro de presencia")
        return articles, {"total": len(articles), "passed": len(articles), "excluded": 0, "skipped": True}

    # Construir grupos de términos por concepto (concepto + sus sinónimos)
    concept_groups = []
    for concept, synonyms in synonyms_by_concept.items():
        group_terms = set()
        group_terms.add(concept.lower())
        group_terms.update(s.lower() for s in synonyms if s)
        concept_groups.append({
            "name": concept,
            "terms": group_terms
        })

    n_concepts = len(concept_groups)
    effective_min = min(min_concepts_required, n_concepts)  # No pedir más de lo que hay

    logger.info(
        f"🔬 [Concept Filter] {n_concepts} grupos conceptuales | "
        f"Mínimo requerido: {effective_min}/{n_concepts} | "
        f"Evaluando {len(articles)} artículos..."
    )

    passed = []
    excluded_count = 0

    for art in articles:
        title    = (art.get('title')    or '').lower()
        abstract = (art.get('abstract') or '').lower()
        combined = f"{title} {abstract}"

        # Contar cuántos conceptos están presentes
        concepts_present = 0
        for group in concept_groups:
            for term in group["terms"]:
                if term in combined:
                    concepts_present += 1
                    break  # Solo necesitamos 1 término del concepto

        if concepts_present >= effective_min:
            art['_concepts_matched'] = concepts_present
            passed.append(art)
        else:
            excluded_count += 1

    reduction_pct = (excluded_count / len(articles) * 100) if articles else 0
    report = {
        "total": len(articles),
        "passed": len(passed),
        "excluded": excluded_count,
        "reduction_pct": round(reduction_pct, 1),
        "n_concept_groups": n_concepts,
        "min_required": effective_min,
    }

    logger.info(
        f"✅ [Concept Filter] {len(articles)} → {len(passed)} artículos "
        f"({excluded_count} excluidos, {reduction_pct:.1f}% reducción) | "
        f"Conceptos detectados/requeridos: {effective_min}/{n_concepts}"
    )
    return passed, report


def apply_hard_filters(
    articles: List[Dict],
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    min_abstract_length: int = 80,
    exclude_review_articles: bool = False,
    allowed_sources: Optional[List[str]] = None,
    extra_exclusion_terms: Optional[List[str]] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Filtra artículos por metadatos antes del proceso vectorial.

    Returns:
        Tuple:
          - filtered_articles (List[Dict]): Artículos que pasaron
          - report (Dict):                  Reporte de exclusiones
    """
    if not articles:
        return [], {"total": 0, "passed": 0, "excluded": 0, "reasons": {}}

    passed = []
    excluded_reasons: Dict[str, int] = {
        "sin_abstract": 0,
        "abstract_corto": 0,
        "fuera_de_rango_temporal": 0,
        "fuente_no_permitida": 0,
        "es_review_survey": 0,
        "termino_exclusion_dura": 0,
    }

    for art in articles:
        reason = _check_exclusion(
            art,
            min_year=min_year,
            max_year=max_year,
            min_abstract_length=min_abstract_length,
            exclude_review_articles=exclude_review_articles,
            allowed_sources=allowed_sources,
            extra_exclusion_terms=extra_exclusion_terms or [],
        )
        if reason:
            excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1
        else:
            passed.append(art)

    n_excluded = len(articles) - len(passed)
    report = {
        "total": len(articles),
        "passed": len(passed),
        "excluded": n_excluded,
        "reasons": excluded_reasons,
    }

    if n_excluded > 0:
        reasons_str = ", ".join(f"{k}={v}" for k, v in excluded_reasons.items() if v > 0)
        logger.info(
            f"🚧 [Hard Filter] {len(articles)} → {len(passed)} artículos "
            f"({n_excluded} excluidos) | {reasons_str}"
        )
    else:
        logger.info(f"✅ [Hard Filter] Todos los {len(articles)} artículos pasaron")

    return passed, report


def _check_exclusion(
    art: Dict,
    min_year: Optional[int],
    max_year: Optional[int],
    min_abstract_length: int,
    exclude_review_articles: bool,
    allowed_sources: Optional[List[str]],
    extra_exclusion_terms: List[str],
) -> Optional[str]:
    """Evalúa un artículo. Retorna razón de exclusión o None si pasa."""
    abstract = (art.get('abstract') or '').strip()
    title    = (art.get('title')    or '').strip()
    combined = f"{title} {abstract}".lower()

    # 1. Abstract ausente o demasiado corto
    if not abstract:
        return "sin_abstract"
    if len(abstract) < min_abstract_length:
        return "abstract_corto"

    # 2. Filtro temporal
    if min_year is not None or max_year is not None:
        year = _extract_year(art)
        if year is not None:
            if min_year and year < min_year:
                return "fuera_de_rango_temporal"
            if max_year and year > max_year:
                return "fuera_de_rango_temporal"

    # 3. Fuentes permitidas
    if allowed_sources:
        source = (art.get('journal') or art.get('venue') or '').lower()
        if source and not any(a.lower() in source for a in allowed_sources):
            return "fuente_no_permitida"

    # 4. Excluir reviews (opcional)
    if exclude_review_articles:
        if any(sig in combined for sig in _REVIEW_SIGNALS):
            return "es_review_survey"

    # 5. Términos de exclusión dura del investigador
    for term in extra_exclusion_terms:
        if term.lower() in combined:
            return "termino_exclusion_dura"

    return None


def _extract_year(art: Dict) -> Optional[int]:
    """Extrae el año de publicación de distintos campos posibles."""
    for field in ['year', 'publicationDate', 'date', 'pub_year']:
        val = art.get(field)
        if val:
            year_str = str(val)[:4]
            if re.match(r'^\d{4}$', year_str):
                return int(year_str)
    return None


def parse_exclusion_criteria_for_hard_filter(
    exclusion_criteria_text: str
) -> Tuple[Optional[int], List[str]]:
    """
    Parsea criterios de exclusión del investigador para extraer:
    - min_year: año mínimo (ej: "Artículos anteriores a 2019" → 2019)
    - hard_terms: términos para exclusión literal

    Returns: (min_year, hard_terms)
    """
    min_year = None
    hard_terms: List[str] = []

    if not exclusion_criteria_text:
        return None, []

    lines = [l.strip() for l in exclusion_criteria_text.splitlines() if l.strip()]

    for line in lines:
        line_lower = line.lower()
        # Detectar restricción de año
        year_match = re.search(
            r'(?:antes?|before|prior|anteriores?)\s+(?:a|to|de)?\s*(\d{4})', line_lower
        )
        if year_match:
            try:
                y = int(year_match.group(1))
                if 1990 <= y <= 2030:
                    if min_year is None or y > min_year:
                        min_year = y
            except ValueError:
                pass

    logger.info(f"🗓️ Hard Filter parseado: min_year={min_year}, hard_terms={hard_terms}")
    return min_year, hard_terms
