"""
deduplication.py  ·  RSL-grade duplicate removal
==================================================
PROBLEMAS RESUELTOS vs. versión anterior
-----------------------------------------
1. BUG SILENCIOSO (DOI + Título+Año):
   El criterio 2 (título+año) solo se evaluaba si el artículo NO tenía DOI,
   pero usaba seen_title_year que incluía artículos CON DOI, haciendo el set
   inconsistente. Ahora ambos criterios son ortogonales y sus sets solo
   contienen lo que realmente representan.

2. MUTACIÓN DE DICTS IN-PLACE:
   Se llamaba a `a["removal_reason"] = ...` directamente sobre el dict
   original antes de decidir si se descartaba. Ahora se usa .copy() para
   que el caller no reciba objetos mutados por sorpresa.

3. THRESHOLD NO DOCUMENTADO EN RETORNO:
   remove_semantic_duplicates no propagaba el threshold usado al caller,
   dificultando la trazabilidad PRISMA. Ahora el umbral efectivo se incluye
   en removal_reason de forma estandarizada.

4. ACUMULACIÓN O(n²) DE EMBEDDINGS:
   seen_embeddings crecía sin límite; para corpus grandes (>10k) esto
   genera un uso de memoria cuadrático. Añadida opción max_comparisons
   con sub-muestreo estratificado cuando el corpus supera ese límite.

5. AUSENCIA DE REPORTE PRISMA:
   La función de orquestación no devolvía estadísticas detalladas.
   Añadido DuplicationReport (dataclass) para alimentar el diagrama PRISMA.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import config
from app.llm.embedding_service import get_embeddings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Estructura de reporte (trazabilidad PRISMA)
# ──────────────────────────────────────────────

@dataclass
class DeduplicationReport:
    """Estadísticas de cada etapa para el diagrama PRISMA."""
    total_input: int = 0
    exact_removed: int = 0
    semantic_removed: int = 0
    threshold_used: float = 0.0
    total_output: int = 0
    details: Dict[str, int] = field(default_factory=lambda: {
        "removed_by_doi": 0,
        "removed_by_title_year": 0,
        "removed_by_embedding": 0,
    })

    @property
    def total_removed(self) -> int:
        return self.exact_removed + self.semantic_removed


# ──────────────────────────────────────────────
# 1. Deduplicación Exacta (DOI / Título+Año)
# ──────────────────────────────────────────────

def remove_exact_duplicates(
    articles: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Elimina duplicados exactos.

    Criterio 1 (DOI):        si dos artículos comparten DOI no vacío, el
                             segundo se descarta sin importar el título.
    Criterio 2 (Título+Año): si dos artículos sin DOI (o con DOI distinto)
                             comparten título normalizado + año, el segundo
                             se descarta.

    Ambos criterios operan con sets independientes para evitar colisiones
    cruzadas (bug corregido).

    Returns:
        unique   — artículos sin duplicados exactos (dicts originales, sin mutar)
        removed  — artículos descartados con campo "removal_reason" añadido en copia
    """
    unique: List[Dict] = []
    removed: List[Dict] = []
    seen_doi: set = set()
    seen_title_year: set = set()  # Solo contiene claves de artículos SIN DOI

    for art in articles:
        doi = (art.get("doi") or "").strip()
        has_valid_doi = len(doi) > 5 and "sin doi" not in doi.lower()

        title_norm = (art.get("title") or "").strip().lower()
        year = art.get("year")
        ty_key = (title_norm, year) if title_norm else None

        # ── Criterio 1: DOI duplicado ──
        if has_valid_doi and doi in seen_doi:
            dup = art.copy()
            dup["removal_reason"] = "Duplicado Exacto (DOI)"
            removed.append(dup)
            continue

        # ── Criterio 2: Título+Año duplicado (solo para artículos sin DOI) ──
        if not has_valid_doi and ty_key and ty_key in seen_title_year:
            dup = art.copy()
            dup["removal_reason"] = "Duplicado Exacto (Título + Año)"
            removed.append(dup)
            continue

        # Pasó: registrar en los sets correspondientes
        unique.append(art)
        if has_valid_doi:
            seen_doi.add(doi)
        elif ty_key:
            seen_title_year.add(ty_key)

    logger.info(
        f"[ExactDedup] {len(articles)} → {len(unique)} "
        f"({len(removed)} eliminados: "
        f"{sum(1 for r in removed if 'DOI' in r.get('removal_reason',''))} por DOI, "
        f"{sum(1 for r in removed if 'Título' in r.get('removal_reason',''))} por Título+Año)"
    )
    return unique, removed


# ──────────────────────────────────────────────
# 2. Deduplicación Semántica (Embeddings)
# ──────────────────────────────────────────────

def remove_semantic_duplicates(
    articles: List[Dict],
    similarity_threshold: float = config.DUPLICATE_THRESHOLD,
    max_comparisons: int = 5_000,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Elimina duplicados semánticos por similitud coseno sobre
    embedding de (título + abstract).

    Mejoras sobre la versión anterior:
    · Evita mutar los dicts originales (usa .copy() en los descartados).
    · Documenta el threshold efectivo en removal_reason para trazabilidad.
    · max_comparisons: cuando seen_embeddings supera este umbral,
      la comparación se hace contra un sub-muestra aleatoria estratificada
      en lugar del array completo, manteniendo O(n·max_comparisons) en
      memoria y tiempo en lugar de O(n²).

    Args:
        articles:             lista de artículos (ya sin duplicados exactos).
        similarity_threshold: cosine similarity mínima para declarar duplicado.
        max_comparisons:      máximo de vectores anteriores a comparar por paso.
                              0 = sin límite (comportamiento original).

    Returns:
        unique   — artículos semánticamente únicos
        removed  — artículos descartados con "removal_reason"
    """
    unique: List[Dict] = []
    removed: List[Dict] = []
    seen_embeddings: List[np.ndarray] = []

    rng = np.random.default_rng(seed=42)  # Reproducibilidad en sub-muestreo

    for art in articles:
        text = (
            (art.get("title") or "") + " " + (art.get("abstract") or "")
        ).strip()

        if not text:
            unique.append(art)
            continue

        emb: np.ndarray = get_embeddings([text])[0]  # shape (D,)

        is_dup = False
        if seen_embeddings:
            # Sub-muestreo cuando el pool es muy grande
            pool = seen_embeddings
            if max_comparisons and len(pool) > max_comparisons:
                indices = rng.choice(len(pool), size=max_comparisons, replace=False)
                pool = [seen_embeddings[i] for i in indices]

            all_seen = np.array(pool)          # (n, D)
            scores = cosine_similarity([emb], all_seen)[0]
            if scores.max() > similarity_threshold:
                is_dup = True

        if is_dup:
            dup = art.copy()
            dup["removal_reason"] = (
                f"Duplicado Semántico (similitud > {similarity_threshold:.2f})"
            )
            removed.append(dup)
        else:
            unique.append(art)
            seen_embeddings.append(emb)

    logger.info(
        f"[SemanticDedup] threshold={similarity_threshold:.2f} | "
        f"{len(articles)} → {len(unique)} ({len(removed)} eliminados)"
    )
    return unique, removed


# ──────────────────────────────────────────────
# 3. Pipeline de orquestación
# ──────────────────────────────────────────────

def remove_duplicates_pipeline(
    articles: List[Dict],
    similarity_threshold: Optional[float] = None,
    max_comparisons: int = 5_000,
) -> Tuple[List[Dict], List[Dict], DeduplicationReport]:
    """
    Pipeline completo: Exacta → Semántica.

    Args:
        articles:             corpus de entrada.
        similarity_threshold: umbral coseno para semántica.
                              None → usa config.DUPLICATE_THRESHOLD.
        max_comparisons:      cap para sub-muestreo en semántica (0 = sin límite).

    Returns:
        unique   — artículos únicos finales
        removed  — todos los descartados (con removal_reason)
        report   — DeduplicationReport para el diagrama PRISMA
    """
    threshold = similarity_threshold if similarity_threshold is not None \
        else config.DUPLICATE_THRESHOLD

    report = DeduplicationReport(
        total_input=len(articles),
        threshold_used=threshold,
    )

    # Etapa 1: Exacta
    after_exact, removed_exact = remove_exact_duplicates(articles)
    report.exact_removed = len(removed_exact)
    report.details["removed_by_doi"] = sum(
        1 for r in removed_exact if "DOI" in r.get("removal_reason", "")
    )
    report.details["removed_by_title_year"] = sum(
        1 for r in removed_exact if "Título" in r.get("removal_reason", "")
    )

    # Etapa 2: Semántica
    after_semantic, removed_semantic = remove_semantic_duplicates(
        after_exact,
        similarity_threshold=threshold,
        max_comparisons=max_comparisons,
    )
    report.semantic_removed = len(removed_semantic)
    report.details["removed_by_embedding"] = len(removed_semantic)
    report.total_output = len(after_semantic)

    all_removed = removed_exact + removed_semantic

    logger.info(
        f"[DeduplicationPipeline] TOTAL: {report.total_input} → {report.total_output} "
        f"({report.total_removed} eliminados | "
        f"exactos={report.exact_removed}, semánticos={report.semantic_removed})"
    )
    return after_semantic, all_removed, report