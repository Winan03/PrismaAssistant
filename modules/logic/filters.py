import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_truly_open_access(article: Dict) -> bool:
    """
    Verifica si un artículo es Open Access usando múltiples señales.
    Estrategia 'Permisiva': Si el buscador dice que es OA, le creemos.
    """
    # 1. Flag explícito puesto por search_engine.py (CRUCIAL)
    if article.get('open_access') is True:
        return True
        
    # 2. Tiene URL directa al PDF (Prueba definitiva)
    if article.get('pdf_url') and len(str(article.get('pdf_url', ''))) > 10:
        return True
        
    # 3. Estructura de Semantic Scholar
    if article.get('openAccessPdf'):
        return True
        
    # 4. Fuente ArXiv o PMC (Siempre son OA)
    url = str(article.get('url', '')).lower()
    source = str(article.get('source', '')).lower()
    if 'arxiv' in url or 'arxiv' in source:
        return True
    if 'pmc' in url:
        return True

    return False

def detect_language(article: Dict) -> Optional[str]:
    """Heurística simple para detectar idioma si no viene en metadata"""
    text = (article.get('title', '') + " " + article.get('abstract', '')).lower()
    if not text.strip(): return None
    
    common_words = {
        'en': ['the', 'and', 'with', 'for', 'study'],
        'es': ['el', 'la', 'con', 'para', 'estudio'],
        'pt': ['o', 'a', 'com', 'para', 'estudo']
    }
    
    scores = {lang: sum(1 for w in words if f" {w} " in text) for lang, words in common_words.items()}
    return max(scores, key=scores.get) if any(scores.values()) else 'en'

def apply_filters(
    articles: List[Dict],
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    open_access: bool = False,
    language: Optional[str] = None,
    journals: Optional[List[str]] = None,
    quartiles: Optional[List[str]] = None
) -> List[Dict]:
    """
    Aplica filtros físicos a la lista de artículos.
    """
    if not articles:
        return []
        
    filtered = articles
    initial_count = len(filtered)

    # 1. Filtro de Año (Robusto ante valores no numéricos)
    def get_year(art):
        try: return int(art.get('year', 0))
        except: return 0

    if start_year:
        filtered = [a for a in filtered if get_year(a) >= start_year]
    if end_year:
        filtered = [a for a in filtered if get_year(a) <= end_year]
    
    logging.info(f"   ✅ Año {start_year}-{end_year}: {len(filtered)} artículos")

    # 2. Filtro Open Access (Lógica Mejorada)
    if open_access:
        # Usamos la nueva función permisiva
        oa_filtered = [a for a in filtered if is_truly_open_access(a)]
        
        # Safety Check: Si el filtro mata todo, avisamos
        if len(oa_filtered) == 0 and len(filtered) > 0:
            logging.warning("⚠️ ALERTA: El filtro OA eliminó todos los artículos. Verifica search_engine.py.")
        
        filtered = oa_filtered
        logging.info(f"   ✅ Open Access: {len(filtered)} artículos")

    # 3. Filtro Idioma
    if language and language != 'all':
        filtered = [a for a in filtered if detect_language(a) == language]

    # 4. Filtro Revistas
    if journals:
        target_journals = {j.lower().strip() for j in journals}
        filtered = [
            a for a in filtered 
            if a.get('journal') and str(a.get('journal')).lower().strip() in target_journals
        ]

    logging.info(f"📊 Filtros aplicados: {len(filtered)} restantes (de {initial_count})")
    return filtered