"""
eval_screening_v2.py
====================
Evaluación de calidad semántica para cribado RSL (Revisión Sistemática de Literatura).
Diseñado bajo criterios PRISMA 2020 — prioriza sensibilidad (recall) en la fase de
identificación/screening, con reporte explícito de precisión y utilidad diferencial
por zona del documento (título vs. abstract).

Métricas implementadas
----------------------
  - Recall@N        : fracción de ítems obligatorios recuperados en los N primeros
  - Precision@N     : fracción de los N primeros que son relevantes (según must_include)
  - F-beta@N        : media armónica ponderada; beta>1 prioriza recall (beta=2 es el default)
  - Noise Ratio@N   : fracción de ítems de exclusión presentes en los N primeros
  - Title Hit Rate  : qué porcentaje de los hits se identificaron solo por título
                      (proxy de calidad del ranking del motor de búsqueda)
  - Coverage Curve  : recall en cortes progresivos y en el ranking completo
  - WSS@95          : corte donde se alcanza el 95% de los estudios relevantes

Notas metodológicas
-------------------
  1. Matching: se usan dos niveles — exacto (substring normalizado) y difuso
     (ratio ≥ FUZZY_THRESHOLD). El nivel difuso captura variantes tipográficas
     y abreviaturas frecuentes en bases como Scopus/WoS.
  2. El score principal NO combina recall y ruido en una sola cifra; se reportan
     por separado para que el investigador decida el umbral según su protocolo.
  3. La función es pura (sin side-effects de logging dentro del cálculo) para
     facilitar su uso en pipelines automatizados o tests unitarios.
  4. El formato de salida incluye un bloque JSON machine-readable para integración
     con gestores de referencias o dashboards.

Dependencias: Python ≥ 3.9, thefuzz (pip install thefuzz), pandas (opcional)
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

# thefuzz es preferible a fuzzywuzzy (mantenida activamente, misma API)
try:
    from thefuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DEL INVESTIGADOR
# ---------------------------------------------------------------------------
# 🚨 INSTRUCCIÓN: Editar este bloque antes de cada sesión de evaluación.
#    Cada entrada en must_include/must_exclude puede ser:
#      - Un fragmento del título (ej. "machine learning readmission")
#      - Un DOI parcial (ej. "10.1016/j.jbi.2021")
#      - Un apellido + año del primer autor (ej. "obermeyer 2019")
#    Cuanto más corto y distintivo, menos falsos positivos.
#
#    RECOMENDACIÓN: Incluir al menos 10 ítems en must_include para que
#    el recall sea estadísticamente interpretable (IC 95% < ±15 pp).
# ---------------------------------------------------------------------------

GROUND_TRUTH: dict[str, list[str]] = {
    # Papers que DEBEN aparecer (semillas de calibración / gold standard)
    "must_include": [
        # Ejemplos — reemplazar con los del tema del investigador:
        # "attention is all you need",
        # "bert pre-training deep bidirectional",
        # "obermeyer 2019 dissecting racial bias",
    ],
    # Términos o papers que NO deben aparecer (ruido temático confirmado)
    "must_exclude": [
        # Ejemplos:
        # "computer vision object detection",
        # "reinforcement learning atari",
    ],
}

# Umbral para coincidencia difusa (0–100). 85 es conservador; bajar a 75
# si los títulos del corpus tienen muchas abreviaturas o idiomas mixtos.
FUZZY_THRESHOLD: int = 85

# Cortes base; evaluate_results agrega siempre el corte final del ranking completo.
COVERAGE_CUTOFFS: list[int] = [10, 20, 50, 100]

# Beta para F-beta score (beta=2 → recall vale el doble que precisión,
# alineado con la fase de screening PRISMA donde los falsos negativos
# son más costosos que los falsos positivos)
FBETA: float = 2.0


# ---------------------------------------------------------------------------
# TIPOS DE DATOS
# ---------------------------------------------------------------------------

@dataclass
class ArticleText:
    """Representación normalizada de un artículo para búsqueda."""
    title: str = ""
    abstract: str = ""
    combined: str = ""

    @classmethod
    def from_dict(cls, record: dict) -> "ArticleText":
        title = str(record.get("title", "")).lower().strip()
        abstract = str(record.get("abstract", "")).lower().strip()
        return cls(title=title, abstract=abstract, combined=f"{title} {abstract}")


@dataclass
class EvaluationResult:
    """Resultado estructurado de la evaluación — serializable a JSON."""
    n_articles: int = 0
    # Recall
    recall_at_n: dict[int, float] = field(default_factory=dict)
    found_in_ranking: list[str] = field(default_factory=list)
    missing_in_ranking: list[str] = field(default_factory=list)
    found_at_100: list[str] = field(default_factory=list)
    missing_at_100: list[str] = field(default_factory=list)
    # Precisión y F-beta
    precision_at_n: dict[int, float] = field(default_factory=dict)
    fbeta_at_n: dict[int, float] = field(default_factory=dict)
    # Ruido
    noise_ratio_at_n: dict[int, float] = field(default_factory=dict)
    noise_present_in_ranking: list[str] = field(default_factory=list)
    noise_present_at_100: list[str] = field(default_factory=list)
    cutoff_at_95: Optional[int] = None
    wss_at_95: Optional[float] = None
    # Diagnóstico de título vs. abstract
    title_hit_rate: float = 0.0   # fracción de hits detectados solo por título
    abstract_only_hits: list[str] = field(default_factory=list)
    # Matching difuso
    fuzzy_hits: list[str] = field(default_factory=list)
    # Metadatos
    fuzzy_available: bool = False
    fuzzy_threshold: int = FUZZY_THRESHOLD


# ---------------------------------------------------------------------------
# FUNCIONES DE MATCHING
# ---------------------------------------------------------------------------

def _exact_hit(query: str, text: str) -> bool:
    """Coincidencia exacta por substring (normalizado a minúsculas)."""
    return query.lower() in text


def _fuzzy_hit(query: str, text: str, threshold: int) -> bool:
    """Coincidencia difusa usando token_set_ratio (robusto a reordenamiento)."""
    if not FUZZY_AVAILABLE:
        return False
    return fuzz.token_set_ratio(query.lower(), text) >= threshold


def _matches(query: str, text: str, threshold: int) -> bool:
    return _exact_hit(query, text) or _fuzzy_hit(query, text, threshold)


def _ranking_cutoffs(n_articles: int, requested: list[int]) -> list[int]:
    cutoffs = sorted({c for c in requested if 0 < c <= n_articles})
    if n_articles > 0 and n_articles not in cutoffs:
        cutoffs.append(n_articles)
    return cutoffs


# ---------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ---------------------------------------------------------------------------

def evaluate_results(
    articles: list[dict],
    ground_truth: Optional[dict[str, list[str]]] = None,
    fuzzy_threshold: int = FUZZY_THRESHOLD,
    coverage_cutoffs: list[int] = COVERAGE_CUTOFFS,
    fbeta: float = FBETA,
) -> EvaluationResult:
    """
    Evalúa una lista de artículos contra el Ground Truth del investigador.

    Parameters
    ----------
    articles        : Lista de dicts con claves 'title' y 'abstract'.
                      El orden importa: se asume que están rankeados por
                      relevancia descendente (como devuelve la mayoría de APIs).
    ground_truth    : Diccionario con 'must_include' y 'must_exclude'.
                      Si es None se usa GROUND_TRUTH global.
    fuzzy_threshold : Umbral de similitud difusa (0–100).
    coverage_cutoffs: Cortes N para calcular Recall@N, Precision@N, etc.
    fbeta           : Parámetro beta para el F-beta score.

    Returns
    -------
    EvaluationResult con todas las métricas calculadas.
    """
    gt = ground_truth if ground_truth is not None else GROUND_TRUTH
    must_include = gt.get("must_include", [])
    must_exclude = gt.get("must_exclude", [])

    result = EvaluationResult(
        n_articles=len(articles),
        fuzzy_available=FUZZY_AVAILABLE,
        fuzzy_threshold=fuzzy_threshold,
    )

    if not articles:
        logger.warning("No hay artículos para evaluar.")
        return result

    if not must_include and not must_exclude:
        logger.info(
            "GROUND_TRUTH vacío. Configure must_include y must_exclude "
            "antes de ejecutar la evaluación."
        )
        return result

    # Normalizar textos una sola vez
    parsed = [ArticleText.from_dict(a) for a in articles]

    # --- Recall y diagnóstico título vs. abstract ---
    title_only_hits: list[str] = []
    abstract_only_hits: list[str] = []
    fuzzy_hits: list[str] = []
    found_in_ranking: list[str] = []

    for term in must_include:
        # Buscar en todo el ranking, no solo en un Top-N fijo.
        for art in parsed:
            hit_title    = _matches(term, art.title,    fuzzy_threshold)
            hit_abstract = _matches(term, art.abstract, fuzzy_threshold)
            if hit_title or hit_abstract:
                found_in_ranking.append(term)
                # Diagnóstico: ¿era detectable solo por título?
                if hit_title and not _exact_hit(term, art.abstract):
                    title_only_hits.append(term)
                elif not hit_title:
                    abstract_only_hits.append(term)
                # ¿Fue un hit difuso (no exacto)?
                if (not _exact_hit(term, art.title) and
                        not _exact_hit(term, art.abstract) and
                        FUZZY_AVAILABLE):
                    fuzzy_hits.append(term)
                break  # encontrado, pasar al siguiente término

    result.found_in_ranking = found_in_ranking
    result.missing_in_ranking = [t for t in must_include if t not in found_in_ranking]
    result.found_at_100 = found_in_ranking
    result.missing_at_100 = result.missing_in_ranking
    result.abstract_only_hits = abstract_only_hits
    result.fuzzy_hits = fuzzy_hits
    result.title_hit_rate = (
        len(title_only_hits) / len(found_in_ranking)
        if found_in_ranking else 0.0
    )

    # --- Ruido en ranking completo ---
    noise_in_ranking: list[str] = []
    for excl in must_exclude:
        if any(_matches(excl, art.combined, fuzzy_threshold) for art in parsed):
            noise_in_ranking.append(excl)
    result.noise_present_in_ranking = noise_in_ranking
    result.noise_present_at_100 = noise_in_ranking

    # --- Curva de cobertura: Recall@N, Precision@N, F-beta@N, Noise@N ---
    n_must = len(must_include)
    n_excl = len(must_exclude)
    cutoffs = _ranking_cutoffs(len(parsed), coverage_cutoffs)

    for cutoff in cutoffs:
        n = min(cutoff, len(parsed))
        subset = parsed[:n]
        combined_subset = [art.combined for art in subset]

        hits_n = [
            t for t in must_include
            if any(_matches(t, txt, fuzzy_threshold) for txt in combined_subset)
        ]
        noise_n = [
            e for e in must_exclude
            if any(_matches(e, txt, fuzzy_threshold) for txt in combined_subset)
        ]

        recall_n    = len(hits_n) / n_must if n_must > 0 else 0.0
        precision_n = len(hits_n) / n      if n      > 0 else 0.0
        noise_n_ratio = len(noise_n) / n_excl if n_excl > 0 else 0.0

        # F-beta: ((1+beta²) * P * R) / (beta² * P + R)
        beta2 = fbeta ** 2
        denom = beta2 * precision_n + recall_n
        fbeta_n = ((1 + beta2) * precision_n * recall_n / denom) if denom > 0 else 0.0

        result.recall_at_n[cutoff]    = round(recall_n,    4)
        result.precision_at_n[cutoff] = round(precision_n, 4)
        result.fbeta_at_n[cutoff]     = round(fbeta_n,     4)
        result.noise_ratio_at_n[cutoff] = round(noise_n_ratio, 4)

    if n_must > 0:
        found_prefix: set[str] = set()
        for idx, art in enumerate(parsed, start=1):
            for term in must_include:
                if term not in found_prefix and _matches(term, art.combined, fuzzy_threshold):
                    found_prefix.add(term)
            if len(found_prefix) / n_must >= 0.95:
                result.cutoff_at_95 = idx
                result.wss_at_95 = round(max(0.0, 0.95 - (idx / len(parsed))), 4)
                break

    # --- Logging legible ---
    _log_report(result, must_include, must_exclude, fbeta)

    return result


# ---------------------------------------------------------------------------
# REPORTE
# ---------------------------------------------------------------------------

def _log_report(
    r: EvaluationResult,
    must_include: list[str],
    must_exclude: list[str],
    fbeta: float,
) -> None:
    """Imprime el reporte de evaluación en formato legible para el investigador."""
    sep = "=" * 60
    logger.info(sep)
    logger.info("EVALUACIÓN RSL — CRIBADO SEMÁNTICO")
    logger.info(f"Artículos evaluados : {r.n_articles}")
    logger.info(f"Ítems obligatorios  : {len(must_include)}")
    logger.info(f"Ítems de exclusión  : {len(must_exclude)}")
    logger.info(f"Matching difuso     : {'activo (thefuzz)' if r.fuzzy_available else 'no disponible — instalar thefuzz'}")
    logger.info(sep)

    logger.info("CURVA DE COBERTURA")
    header = f"  {'Corte':>6}  {'Recall':>8}  {'Precisión':>10}  {'F{:.0f}-score'.format(fbeta):>10}  {'Ruido':>7}"
    logger.info(header)
    logger.info("  " + "-" * 50)
    for n in sorted(r.recall_at_n):
        logger.info(
            f"  {n:>6}  "
            f"{r.recall_at_n[n]*100:>7.1f}%  "
            f"{r.precision_at_n[n]*100:>9.1f}%  "
            f"{r.fbeta_at_n[n]*100:>9.1f}%  "
            f"{r.noise_ratio_at_n[n]*100:>6.1f}%"
        )

    logger.info(sep)
    full_recall = (len(r.found_in_ranking) / len(must_include)) if must_include else 0.0
    logger.info(f"RECALL ranking completo : {len(r.found_in_ranking)}/{len(must_include)} ({full_recall*100:.1f}%)")
    if r.missing_in_ranking:
        logger.info(f"  Faltantes  : {r.missing_in_ranking}")
    if r.fuzzy_hits:
        logger.info(f"  Hits difusos (umbral {r.fuzzy_threshold}): {r.fuzzy_hits}")

    logger.info(sep)
    logger.info("DIAGNÓSTICO TÍTULO vs. ABSTRACT")
    logger.info(f"  Detectables solo por título  : {r.title_hit_rate*100:.1f}% de los hits")
    if r.abstract_only_hits:
        logger.info(f"  Solo en abstract (riesgo de pérdida si se filtra por título): {r.abstract_only_hits}")

    logger.info(sep)
    full_noise = (len(r.noise_present_in_ranking) / len(must_exclude)) if must_exclude else 0.0
    logger.info(f"RUIDO ranking completo : {len(r.noise_present_in_ranking)}/{len(must_exclude)} ({full_noise*100:.1f}%)")
    if r.noise_present_in_ranking:
        logger.info(f"  Presentes  : {r.noise_present_in_ranking}")
    if r.cutoff_at_95 is not None:
        logger.info(f"WSS@95 : corte={r.cutoff_at_95}/{r.n_articles} | ahorro={r.wss_at_95*100:.1f}%")

    logger.info(sep)
    logger.info("INTERPRETACIÓN RÁPIDA (PRISMA 2020)")
    if full_recall >= 0.90:
        logger.info("  ✓ Recall ≥ 90% — estrategia de búsqueda aceptable para RSL.")
    elif full_recall >= 0.75:
        logger.info("  ⚠ Recall 75–89% — revisar query o ampliar bases de datos.")
    else:
        logger.info("  ✗ Recall < 75% — estrategia de búsqueda insuficiente. Rediseñar query.")
    logger.info(sep)


def results_to_json(result: EvaluationResult) -> str:
    """Serializa el resultado a JSON para integración con pipelines externos."""
    return json.dumps(asdict(result), indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd

    if len(sys.argv) > 1:
        path = sys.argv[1]
        df = pd.read_csv(path)
        articles = df.to_dict("records")
        result = evaluate_results(articles)
        # Opcional: guardar JSON con métricas
        out_json = path.replace(".csv", "_eval.json")
        with open(out_json, "w", encoding="utf-8") as f:
            f.write(results_to_json(result))
        logger.info(f"Métricas guardadas en: {out_json}")
    else:
        print("Uso: python eval_screening_v2.py <ruta_al_csv_de_resultados>")
        print("     El CSV debe tener columnas 'title' y 'abstract'.")
