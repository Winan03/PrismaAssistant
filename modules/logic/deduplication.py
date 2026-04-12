from sklearn.metrics.pairwise import cosine_similarity
import config
from typing import List, Dict, Tuple
import numpy as np
from modules.ai.embedding_service import get_embeddings

# ============================================
# 1. Deduplicación Exacta (DOI)
# ============================================
def remove_exact_duplicates(articles: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Elimina duplicados exactos por DOI o Title + Year.
    Devuelve: (artículos sin duplicados exactos, duplicados exactos)
    """
    unique = []
    removed = []
    seen_doi = set()
    seen_title_year = set()

    for a in articles:
        doi = a.get("doi")
        title_year_key = (a.get("title", "").strip().lower(), a.get("year"))

        # Criterio 1: DOI
        if doi and doi in seen_doi:
            a["removal_reason"] = "Duplicado Exacto (DOI)"
            removed.append(a)
            continue
        
        # Criterio 2: Título y Año (para artículos sin DOI)
        if title_year_key[0] and title_year_key in seen_title_year:
            if not doi or not doi.strip():
                 a["removal_reason"] = "Duplicado Exacto (Título + Año)"
                 removed.append(a)
                 continue
        
        # Si pasa, lo agregamos y actualizamos las listas de vistos
        unique.append(a)
        if doi:
            seen_doi.add(doi)
        if title_year_key[0]:
            seen_title_year.add(title_year_key)

    return unique, removed

# ============================================
# 2. Deduplicación Semántica (Embeddings)
# ============================================
def remove_semantic_duplicates(articles: List[Dict], similarity_threshold: float = config.DUPLICATE_THRESHOLD) -> Tuple[List[Dict], List[Dict]]:
    """
    Elimina duplicados semánticos por similaridad de embedding (título + abstract).
    Usa embedding_service (HF API o modelo local como fallback).
    Devuelve: (artículos semánticamente únicos, duplicados semánticos)
    """
    unique = []
    removed = []
    seen_embeddings: List[np.ndarray] = []  # Lista de vectores numpy

    for a in articles:
        text = a["title"] + " " + a.get("abstract", "")
        if not text.strip():
            unique.append(a)
            continue

        emb = get_embeddings([text])[0]  # shape (768,)
        is_dup = False

        if seen_embeddings:
            # Comparar contra todos los vistos: shape (n_seen, 768)
            all_seen = np.array(seen_embeddings)
            scores = cosine_similarity([emb], all_seen)[0]  # (n_seen,)
            if scores.max() > similarity_threshold:
                is_dup = True

        if not is_dup:
            unique.append(a)
            seen_embeddings.append(emb)
        else:
            a["removal_reason"] = f"Duplicado Semántico (> {similarity_threshold})"
            removed.append(a)

    return unique, removed

# ============================================
# FUNCIÓN DE ORQUESTACIÓN (Si decides mantenerla)
# ============================================
def remove_duplicates_pipeline(articles: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Aplica la deduplicación completa: Exacta -> Semántica"""
    
    # 1. Deduplicación Exacta
    articles_after_exact, removed_exact = remove_exact_duplicates(articles)
    
    # 2. Deduplicación Semántica
    articles_after_semantic, removed_semantic = remove_semantic_duplicates(
        articles_after_exact, 
        similarity_threshold=config.DUPLICATE_THRESHOLD
    )
    
    # 3. Consolidar removidos
    all_removed = removed_exact + removed_semantic
    
    return articles_after_semantic, all_removed