"""
Filtros PRISMA con manejo robusto de valores None
"""
import logging

def apply_filters(
    articles, 
    start_year=None, 
    end_year=None, 
    doc_type=None, 
    open_access=False, 
    language=None,
    journals=None,
    quartiles=None
):
    """
    Aplica filtros de exclusión según criterios PRISMA.
    
    Args:
        articles: Lista de artículos a filtrar
        start_year: Año mínimo de publicación
        end_year: Año máximo de publicación
        doc_type: Tipo de documento (DESACTIVADO - no funcional)
        open_access: Si True, solo artículos de acceso abierto
        language: Código de idioma (en, es, pt, etc.)
        journals: Lista de journals específicos a incluir
        quartiles: Lista de cuartiles (Q1, Q2, Q3, Q4) - DESACTIVADO por falta de metadata
        
    Returns:
        Lista de artículos filtrados
    """
    filtered = articles
    initial_count = len(filtered)
    
    logging.info(f"🔍 Aplicando filtros a {initial_count} artículos...")

    # 1. Filtro por año de inicio
    if start_year:
        filtered = [
            a for a in filtered 
            if (a.get("year") or 0) >= start_year
        ]
        logging.info(f"   ✅ Año inicio >= {start_year}: {len(filtered)} artículos")
    
    # 2. Filtro por año final
    if end_year:
        filtered = [
            a for a in filtered 
            if (a.get("year") or 0) <= end_year
        ]
        logging.info(f"   ✅ Año final <= {end_year}: {len(filtered)} artículos")
    
    # 3. Filtro por journals específicos
    if journals and len(journals) > 0:
        filtered = [
            a for a in filtered
            if a.get("journal", "").strip() in journals
        ]
        logging.info(f"   ✅ Journals seleccionados ({len(journals)}): {len(filtered)} artículos")
    
    # 4. Filtro por acceso abierto
    if open_access:
        filtered = [
            a for a in filtered 
            if is_open_access(a)
        ]
        logging.info(f"   ✅ Open Access: {len(filtered)} artículos")
    
    # 5. Filtro por idioma (heurística basada en abstract)
    if language:
        filtered = [
            a for a in filtered
            if detect_language(a) == language
        ]
        logging.info(f"   ✅ Idioma '{language}': {len(filtered)} artículos")
    
    # 6. Filtro por tipo de documento (DESACTIVADO)
    # Razón: Metadata inconsistente entre APIs
    if doc_type:
        logging.warning(f"   ⚠️ Filtro 'doc_type' desactivado (metadata no disponible)")
        pass
    
    # 7. Filtro por cuartiles (DESACTIVADO)
    # Razón: Semantic Scholar/PubMed no incluyen cuartil en metadata
    if quartiles and len(quartiles) > 0:
        logging.warning(f"   ⚠️ Filtro 'quartiles' desactivado (requiere API externa de Scimago)")
        pass
    
    excluded = initial_count - len(filtered)
    logging.info(f"📊 Filtros aplicados: {len(filtered)} artículos restantes ({excluded} excluidos)")
    
    return filtered


def is_open_access(article):
    """
    Determina si un artículo es de acceso abierto (heurística).
    
    Estrategia:
    - URL contiene "open", "arxiv", "plos", "biorxiv"
    - DOI empieza con prefijos de editoriales OA conocidas
    - Campo "isOpenAccess" si está disponible
    """
    # 1. Verificar campo explícito (Semantic Scholar)
    if article.get("isOpenAccess"):
        return True
    
    # 2. Verificar URL
    url = article.get("url", "").lower()
    oa_indicators = ["open", "arxiv", "plos", "biorxiv", "medrxiv", "ssrn", "researchgate"]
    if any(indicator in url for indicator in oa_indicators):
        return True
    
    # 3. Verificar DOI de editoriales OA conocidas
    doi = article.get("doi", "").lower()
    oa_doi_prefixes = [
        "10.1371",  # PLOS
        "10.3389",  # Frontiers
        "10.1186",  # BioMed Central
        "10.1038",  # Nature (algunos OA)
        "10.7554",  # eLife
    ]
    if any(doi.startswith(prefix) for prefix in oa_doi_prefixes):
        return True
    
    return False


def detect_language(article):
    """
    Detecta el idioma de un artículo (heurística basada en abstract).
    
    Returns:
        'en', 'es', 'pt', 'fr' o None
    """
    abstract = article.get("abstract", "").lower()
    title = article.get("title", "").lower()
    text = f"{title} {abstract}"
    
    if not text.strip():
        return None
    
    # Palabras clave por idioma
    english_words = ["the", "and", "of", "in", "to", "with", "for", "this", "that", "was"]
    spanish_words = ["de", "la", "el", "en", "los", "las", "con", "por", "para", "que"]
    portuguese_words = ["da", "do", "em", "para", "com", "uma", "dos", "das", "pela", "pelo"]
    french_words = ["le", "de", "et", "la", "un", "une", "des", "les", "dans", "pour"]
    
    # Contar coincidencias
    en_count = sum(1 for word in english_words if f" {word} " in text)
    es_count = sum(1 for word in spanish_words if f" {word} " in text)
    pt_count = sum(1 for word in portuguese_words if f" {word} " in text)
    fr_count = sum(1 for word in french_words if f" {word} " in text)
    
    # Determinar idioma dominante
    counts = {
        'en': en_count,
        'es': es_count,
        'pt': pt_count,
        'fr': fr_count
    }
    
    max_lang = max(counts, key=counts.get)
    
    # Solo retornar si hay al menos 3 coincidencias
    if counts[max_lang] >= 3:
        return max_lang
    
    return None