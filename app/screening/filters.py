import json
import logging
import re
from collections import Counter
from typing import List, Dict, Optional, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_MISSING_JOURNAL_VALUES = {
    "",
    "unknown",
    "none",
    "null",
    "n/d",
    "na",
    "sin revista",
    "sin revista / otros",
    "revista no disponible",
}

_PREPRINT_VENUE_MARKERS = (
    "arxiv",
    "biorxiv",
    "medrxiv",
    "ssrn",
    "preprint",
    "research square",
)

_LANGUAGE_ALIASES = {
    "en": "en", "eng": "en", "english": "en",
    "es": "es", "spa": "es", "esp": "es", "spanish": "es", "espanol": "es", "español": "es",
    "pt": "pt", "por": "pt", "portuguese": "pt", "portugues": "pt", "português": "pt",
    "fr": "fr", "fre": "fr", "fra": "fr", "french": "fr",
    "de": "de", "ger": "de", "deu": "de", "german": "de",
    "it": "it", "ita": "it", "italian": "it",
    "zh": "zh", "chi": "zh", "zho": "zh", "chinese": "zh",
}

_LANGUAGE_META = {
    "en": {"name": "English", "flag": "EN"},
    "es": {"name": "Español", "flag": "ES"},
    "pt": {"name": "Português", "flag": "PT"},
    "fr": {"name": "Français", "flag": "FR"},
    "de": {"name": "Deutsch", "flag": "DE"},
    "it": {"name": "Italiano", "flag": "IT"},
    "zh": {"name": "Chinese", "flag": "ZH"},
    "und": {"name": "No determinado", "flag": "ND"},
}

_LANGUAGE_STOPWORDS = {
    "en": {
        "the", "and", "with", "for", "study", "studies", "children", "child",
        "results", "method", "methods", "using", "between", "from", "this",
        "that", "were", "was", "are", "have", "has", "among", "into",
    },
    "es": {
        "que", "los", "las", "con", "para", "estudio", "estudios", "niños",
        "ninos", "infantil", "educacion", "educación", "resultados", "metodos",
        "métodos", "entre", "este", "esta", "son", "como", "una", "del",
    },
    "pt": {
        "que", "com", "para", "estudo", "estudos", "criancas", "crianças",
        "infantil", "educacao", "educação", "resultados", "metodos", "métodos",
        "entre", "este", "esta", "sao", "são", "uma", "dos", "das", "pela", "pelo",
    },
}

def get_journal_name(article: Dict) -> str:
    """Devuelve la revista/sede normalizada para filtros y conteos."""
    return " ".join(str(article.get("journal") or article.get("venue") or "").split())

def normalize_journal_name(value: str) -> str:
    return " ".join(str(value or "").split()).casefold()

def normalize_language_code(value: object) -> Optional[str]:
    if isinstance(value, (list, tuple, set)):
        for item in value:
            code = normalize_language_code(item)
            if code:
                return code
        return None
    if isinstance(value, dict):
        for key in ("code", "name", "language"):
            code = normalize_language_code(value.get(key))
            if code:
                return code
        return None

    raw = str(value or "").strip().lower()
    if not raw:
        return None
    raw = raw.replace("_", "-").split("-")[0]
    return _LANGUAGE_ALIASES.get(raw, raw if len(raw) == 2 else None)

def detect_language_with_source(article: Dict) -> tuple[str, str]:
    for field_name in ("language", "lang", "publication_language", "language_code"):
        code = normalize_language_code(article.get(field_name))
        if code:
            return code, "metadata"

    text = f"{article.get('title', '')} {article.get('abstract', '')}".lower()
    words = set(re.findall(r"[a-záéíóúñüàèìòùâêîôûçãõ]+", text))
    if not words:
        return "und", "unknown"

    scores = {code: len(words & stopwords) for code, stopwords in _LANGUAGE_STOPWORDS.items()}
    best_code, best_score = max(scores.items(), key=lambda item: item[1])
    second_score = max((score for code, score in scores.items() if code != best_code), default=0)
    if best_score < 2 or (second_score and best_score < second_score + 2):
        return "und", "unknown"
    return best_code, "inferred"

def summarize_languages(articles: List[Dict]) -> List[Dict]:
    counts: Counter = Counter()
    metadata_counts: Counter = Counter()
    inferred_counts: Counter = Counter()

    for article in articles:
        code, source = detect_language_with_source(article)
        counts[code] += 1
        if source == "metadata":
            metadata_counts[code] += 1
        elif source == "inferred":
            inferred_counts[code] += 1

    summaries: List[Dict] = []
    for code, count in counts.most_common():
        meta = _LANGUAGE_META.get(code, {"name": code.upper(), "flag": code.upper()})
        metadata_count = metadata_counts[code]
        inferred_count = inferred_counts[code]
        summaries.append({
            "code": code,
            "name": meta["name"],
            "flag": meta["flag"],
            "count": count,
            "metadata_count": metadata_count,
            "inferred_count": inferred_count,
            "unknown_count": count - metadata_count - inferred_count,
            "note": f"metadata: {metadata_count}; inferido: {inferred_count}",
        })
    return summaries

def parse_journal_filters(raw_journals: Optional[Union[str, List[str]]]) -> List[str]:
    """Acepta JSON o campos repetidos; no rompe nombres con coma."""
    if not raw_journals:
        return []

    raw_values = [raw_journals] if isinstance(raw_journals, str) else raw_journals
    parsed: List[str] = []

    for raw in raw_values:
        text = str(raw or "").strip()
        if not text:
            continue
        if text.startswith("["):
            try:
                values = json.loads(text)
                if isinstance(values, list):
                    parsed.extend(str(value).strip() for value in values if str(value).strip())
                    continue
            except (json.JSONDecodeError, TypeError):
                logging.warning("No se pudo parsear la lista JSON de revistas seleccionadas.")
        parsed.append(text)

    unique: List[str] = []
    seen = set()
    for journal in parsed:
        key = normalize_journal_name(journal)
        if key and key not in seen:
            seen.add(key)
            unique.append(" ".join(journal.split()))
    return unique

def has_academic_venue(article: Dict) -> bool:
    year = article.get("year", 0)
    try:
        year = int(year)
    except (TypeError, ValueError):
        year = 0

    journal_name = get_journal_name(article)
    journal_key = normalize_journal_name(journal_name)
    if year == 0 or journal_key in _MISSING_JOURNAL_VALUES:
        return False
    return not any(marker in journal_key for marker in _PREPRINT_VENUE_MARKERS)

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
    """Usa metadata de idioma; solo infiere desde texto cuando la API no lo trae."""
    code, _source = detect_language_with_source(article)
    return code

def apply_filters(
    articles: List[Dict],
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    open_access: bool = False,
    language: Optional[str] = None,
    journals: Optional[List[str]] = None,
    quartiles: Optional[List[str]] = None,
    academic_only: bool = False
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

    # 3. Filtro Calidad Académica (Solo con revista)
    if academic_only:
        def is_academic(a: Dict) -> bool:
            year = a.get('year', 0)
            try:
                year = int(year)
            except:
                year = 0

            # UX: No excluir si la API no provee el campo journal/venue.
            # Solo excluir si el campo existe pero está vacío/unknown.
            journal_missing = a.get('journal') is None and a.get('venue') is None
            journal_val = (a.get('journal') or a.get('venue') or '').strip()
            explicit_empty = (not journal_val) or journal_val.lower() in (
                'unknown', 'revista no disponible', 'sin revista'
            )
            if journal_missing:
                explicit_empty = False

            return not (explicit_empty or year == 0)
            
        filtered = [a for a in filtered if is_academic(a)]
        filtered = [a for a in filtered if has_academic_venue(a)]
        logging.info(f"   ✅ Calidad Académica (Solo con revista): {len(filtered)} artículos")

    # 3. Filtro Idioma
    if language and language != 'all':
        filtered = [a for a in filtered if detect_language(a) == language]

    # 4. Filtro Revistas
    if journals:
        target_journals = {normalize_journal_name(j) for j in journals}
        filtered = [
            a for a in filtered 
            if normalize_journal_name(get_journal_name(a)) in target_journals
        ]

    logging.info(f"📊 Filtros aplicados: {len(filtered)} restantes (de {initial_count})")
    return filtered
