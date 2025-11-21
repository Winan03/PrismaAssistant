from sentence_transformers import SentenceTransformer, util
import config
from typing import List, Dict, Tuple
import torch

# Cargar modelo una sola vez (como ya lo tienes)
model = SentenceTransformer(config.EMBEDDING_MODEL) 

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
    Devuelve: (artículos semánticamente únicos, duplicados semánticos)
    """
    unique = []
    removed = []
    seen_emb = [] # Lista de tensores

    for a in articles:
        text = a["title"] + " " + a.get("abstract", "")
        if not text.strip():
            unique.append(a)
            continue

        # Convertir a tensor
        emb = model.encode(text, convert_to_tensor=True) 
        is_dup = False
        
        # Comparar el embedding actual con los embeddings únicos ya vistos
        if seen_emb:
            # CORRECCIÓN: Usar torch.cat() en lugar de util.cat()
            # Y util.cos_sim acepta directamente la lista de tensores.
            
            # Concatenar todos los tensores vistos en un solo tensor para comparación en lote
            all_seen_embeddings = torch.cat(seen_emb)
            
            # util.cos_sim calcula la similaridad entre un vector (emb) y el conjunto (all_seen_embeddings)
            # Retorna una lista de scores. Usamos .max() para ver el score más alto.
            cosine_scores = util.cos_sim(emb, all_seen_embeddings)[0]
            
            if cosine_scores.max() > similarity_threshold:
                is_dup = True
        
        if not is_dup:
            unique.append(a)
            # Añadir el embedding a la lista de tensores vistos, listo para concatenación
            seen_emb.append(emb.unsqueeze(0)) 
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