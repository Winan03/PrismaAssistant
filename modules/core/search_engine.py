import time
import requests
import config
import logging
import re
from typing import List, Dict, Optional
from utils.query_expander import generate_api_queries_with_llm
from Bio import Entrez
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.pdf_extractor import enrich_initial_search_result

Entrez.email = "prisma-assistant@upao.edu.pe"

# ============================================================
# 🧠 DETECCIÓN DE DOMINIO Y SELECCIÓN INTELIGENTE DE FUENTES
# ============================================================

# Palabras clave por dominio para detección y anclaje
DOMAIN_MARKERS = {
    'cs': {
        'keywords': {'software', 'code', 'programming', 'algorithm', 'llm', 'sast', 'vulnerability', 'ciberseguridad', 'computación'},
        'anchors': 'software computer computer-science'
    },
    'medical': {
        'keywords': {'clinical', 'patient', 'hospital', 'therapy', 'medical', 'paciente', 'clínico', 'tratamiento'},
        'anchors': 'medicine clinical-study'
    },
    'law': {
        'keywords': {'ley', 'derecho', 'jurídico', 'legal', 'law', 'jurisprudencia', 'constitucional', 'court'},
        'anchors': 'law legal-study'
    },
    'architecture': {
        'keywords': {'architecture', 'urbanismo', 'construcción', 'building', 'design', 'structure', 'arquitectura', 'habitat'},
        'anchors': 'architecture urban-planning'
    }
}

def detect_search_domain(question: str) -> dict:
    """
    Detecta el dominio de forma dinámica y devuelve anclas de precisión.
    Retorna: {'id': 'cs'|'medical'|'law'|..., 'anchors': 'str', 'score': int}
    """
    q_lower = question.lower()
    best_domain = 'general'
    max_score = 0
    anchors = ""
    
    for dom_id, config in DOMAIN_MARKERS.items():
        score = sum(2 if kw in q_lower else 0 for kw in config['keywords'])
        if score > max_score:
            max_score = score
            best_domain = dom_id
            anchors = config['anchors']
    
    logging.info(f"🧠 Dominio Universal: {best_domain.upper()} (score={max_score})")
    return {'id': best_domain, 'anchors': anchors, 'score': max_score}

# ============================================================
# 🧠 GESTIÓN DE CONSULTAS (ESTRATEGIA MULTI-QUERY)
# ============================================================

def generate_smart_variations(terms: List[str]) -> List[List[str]]:
    """
    Implementa la estrategia 'Divide y Vencerás' para maximizar el Recall.
    AGNÓSTICO AL DOMINIO: usa los términos proporcionados por el usuario/LLM.
    """
    # Si los términos ya contienen operadores booleanos (queries del LLM), usarlos directamente
    if any((" AND " in t or " OR " in t) for t in terms):
        logging.info("⚡ Detectadas Estrategias Multi-Query avanzadas (Grok).")
        return [[t] for t in terms]

    # Fallback: crear variaciones a partir de los términos del usuario
    logging.info("🔧 Generando variaciones desde términos del usuario...")
    
    if len(terms) == 0:
        return [["artificial intelligence"]]
    
    if len(terms) == 1:
        return [[terms[0]]]
    
    # Estrategia 1: Todos los términos juntos
    variations = [terms[:4]]  # Max 4 términos por query
    
    # Estrategia 2: Pares de términos (para más cobertura)
    if len(terms) >= 2:
        variations.append([terms[0], terms[1]])
    if len(terms) >= 3:
        variations.append([terms[0], terms[2]])
    if len(terms) >= 4:
        variations.append([terms[1], terms[3]])
    
    return variations

# ============================================================
# ✅ SEMANTIC SCHOLAR (BÚSQUEDA SEMÁNTICA ENFOCADA)
# ============================================================

def _clean_for_semantic_search(query: str) -> str:
    """Limpia operadores booleanos y formato para búsqueda semántica."""
    clean = query
    clean = re.sub(r'\b(AND|OR|NOT)\b', ' ', clean)
    clean = clean.replace('(', ' ').replace(')', ' ').replace('"', '')
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

def _truncate_query_for_api(query: str, max_words: int = 10) -> str:
    """
    Semantic Scholar API rechaza queries muy largas (HTTP 400).
    Trunca a las primeras max_words palabras más significativas.
    """
    words = query.split()
    if len(words) <= max_words:
        return query
    
    generic = {'and', 'or', 'the', 'for', 'with', 'using', 'based', 'on', 'in', 'of', 'a'}
    result = []
    for w in words:
        if len(result) >= max_words:
            break
        if w.lower() not in generic or len(result) < 4:
            result.append(w)
    
    truncated = " ".join(result)
    logging.debug(f"   ✂️ Query truncada: '{query[:50]}...' → '{truncated}'")
    return truncated


# Función _generate_focused_queries eliminada por redundancia (usar query_expander.py)


def _clean_question_for_search(question: str) -> str:
    """Limpia pregunta en español para usar como query."""
    clean = re.sub(r'[¿?¡!,;:.]', ' ', question)
    stopwords = {
        'cuál', 'cual', 'cómo', 'como', 'qué', 'que', 'es', 'son', 'las', 'los',
        'del', 'de', 'la', 'el', 'en', 'un', 'una', 'para', 'por', 'con', 'sin',
        'sobre', 'entre', 'más', 'mas', 'se', 'al', 'a', 'e', 'i', 'o', 'u', 'y',
        'su', 'sus', 'mi', 'tu', 'frente', 'durante', 'mediante', 'través',
    }
    words = [w for w in clean.split() if w.lower() not in stopwords and len(w) > 1]
    return " ".join(words[:10])


def search_semantic_scholar_oa(term_variations: List[List[str]], target: int = 250, 
                                fields_of_study: Optional[str] = None,
                                original_question: Optional[str] = None) -> List[Dict]:
    if not config.SEMANTIC_SCHOLAR_API_KEY: return []

    articles_map = {} 
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": config.SEMANTIC_SCHOLAR_API_KEY}
    fields = "title,year,abstract,authors,externalIds,url,openAccessPdf,journal,publicationDate,venue"

    def _safe_request(params, stage_label):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                r = requests.get(url, headers=headers, params=params, timeout=20)
                if r.status_code == 200:
                    return r.json()
                elif r.status_code == 429:
                    wait = 2 * (attempt + 1)
                    logging.warning(f"⚠️ Rate limit SS ({stage_label}), esperando {wait}s...")
                    time.sleep(wait)
                elif 500 <= r.status_code <= 504:
                    wait = 2 ** attempt
                    logging.warning(f"⚠️ Semantic Scholar Error {r.status_code} ({stage_label}), reintento {attempt+1}/{max_retries} en {wait}s...")
                    time.sleep(wait)
                else:
                    logging.warning(f"⚠️ Semantic Scholar HTTP {r.status_code} ({stage_label})")
                    break
            except Exception as e:
                logging.error(f"❌ Semantic Scholar Error ({stage_label}): {e}")
                time.sleep(1)
        return None

    # === ÚNICA FASE: Búsqueda con las estrategias del Query Expander ===
    # El Recall y la Precisión ahora residen en la inteligencia del LLM expander
    for i, terms_group in enumerate(term_variations):
        if len(articles_map) >= target: break
        
        raw_query = " ".join(terms_group).replace('  ', ' ').strip()
        query = _clean_for_semantic_search(raw_query)
        query = _truncate_query_for_api(query, max_words=10)
        
        logging.info(f"🎯 Semantic Scholar (Estrategia {i+1}/{len(term_variations)}): '{query}'")
        if i == 0 and fields_of_study:
            logging.info(f"   🔬 Filtro de dominio: fieldsOfStudy={fields_of_study}")

        offset = 0
        while len(articles_map) < target and offset < 300: 
            params = {
                "query": query,
                "limit": 100,
                "fields": fields,
                "openAccessPdf": "true",
                "offset": offset
            }
            if fields_of_study:
                params["fieldsOfStudy"] = fields_of_study

            data = _safe_request(params, f"Fase 2-{i+1}")
            if data and data.get('data'):
                added = _process_ss_results(data['data'], articles_map)
                offset += 100
                if added == 0: break 
            else:
                break

    logging.info(f"📚 Semantic Scholar total: {len(articles_map)} artículos")
    return list(articles_map.values())


def _process_ss_results(papers: list, articles_map: dict) -> int:
    """Procesa resultados de Semantic Scholar. Acepta papers con o sin PDF abierto."""
    added_count = 0
    for paper in papers:
        abstract = paper.get('abstract')
        if not abstract or len(abstract) < 100: continue

        paper_id = paper.get('paperId')
        if paper_id not in articles_map:
            journal_info = paper.get('journal') or {}
            venue = paper.get('venue') or ""
            
            # PDF abierto es preferido pero no obligatorio
            oa_pdf = paper.get('openAccessPdf') or {}
            pdf_url = oa_pdf.get('url', '')
            paper_url = paper.get('url', '')
            doi = paper.get('externalIds', {}).get('DOI', '')
            
            # Construir URL del artículo (preferir DOI si no hay PDF)
            article_url = paper_url
            if not article_url and doi:
                article_url = f"https://doi.org/{doi}"
            
            articles_map[paper_id] = {
                "title": paper.get('title', ''),
                "authors": [a['name'] for a in paper.get('authors', [])],
                "doi": doi,
                "year": paper.get('year') or 0,
                "abstract": abstract,
                "journal": journal_info.get('name', venue),
                "volume": journal_info.get('volume', ''),
                "issue": journal_info.get('pages', '').split('-')[0] if '-' in journal_info.get('pages', '') else "",
                "pages": journal_info.get('pages', ''),
                "url": article_url,
                "pdf_url": pdf_url if pdf_url else article_url,
                "open_access": bool(pdf_url),
                "source": "Semantic Scholar"
            }
            added_count += 1
    return added_count

# ============================================================
# ✅ PUBMED (META AJUSTADA: 200) - ¡AHORA CON FULL METADATA!
# ============================================================

def search_pubmed_oa(term_variations: List[List[str]], target: int = 200) -> List[Dict]:
    all_articles = []
    seen_ids = set()
    
    for i, terms_group in enumerate(term_variations):
        if len(all_articles) >= target: break
        
        base_query = terms_group[0] if len(terms_group) == 1 else " AND ".join(terms_group)
        query = f"({base_query}) AND (free full text[sb]) AND (2018:2025[dp])"
        
        logging.info(f"🔬 PubMed (Estrategia {i+1}): '{query[:60]}...'")
        
        try:
            retstart = 0
            while len(all_articles) < target and retstart < 400:
                handle = Entrez.esearch(db="pubmed", term=query, retmax=100, retstart=retstart, sort="relevance")
                record = Entrez.read(handle)
                handle.close()
                
                ids = [i for i in record["IdList"] if i not in seen_ids]
                if not ids: break 
                seen_ids.update(ids)
                
                handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
                records = Entrez.read(handle)
                handle.close()
                
                if 'PubmedArticle' not in records: break

                for article in records['PubmedArticle']:
                    if len(all_articles) >= target: break
                    
                    medline = article['MedlineCitation']
                    article_data = medline['Article']
                    journal_data = article_data.get('Journal', {})
                    journal_issue = journal_data.get('JournalIssue', {})
                    
                    # Abstract
                    abstract_text = ""
                    if 'Abstract' in article_data and 'AbstractText' in article_data['Abstract']:
                        abs_parts = article_data['Abstract']['AbstractText']
                        if isinstance(abs_parts, list):
                            abstract_text = " ".join([str(part) for part in abs_parts])
                        else:
                            abstract_text = str(abs_parts)
                    
                    if len(abstract_text) < 100: continue

                    title = article_data.get('ArticleTitle', '')
                    
                    # Autores
                    authors = []
                    if 'AuthorList' in article_data:
                        for a in article_data['AuthorList']:
                            if 'LastName' in a and 'ForeName' in a:
                                authors.append(f"{a['LastName']}, {a['ForeName']}")
                    
                    # Año
                    pub_date = journal_issue.get('PubDate', {})
                    try:
                        year = int(pub_date.get('Year', '0'))
                    except:
                        year = 2024
                    
                    # Identificadores (DOI, PMID)
                    doi = ""
                    pmid = medline.get('PMID', '')
                    if 'ELocationID' in article_data:
                        for eid in article_data['ELocationID']:
                            if eid.attributes.get('EIdType') == 'doi':
                                doi = str(eid)

                    # 🔥 EXTRACCIÓN PROFUNDA DE METADATOS EXTRA 🔥
                    volume = journal_issue.get('Volume', '')
                    issue = journal_issue.get('Issue', '')
                    pages = article_data.get('Pagination', {}).get('MedlinePgn', '')
                    issn = journal_data.get('ISSN', '')
                    journal_abbr = journal_data.get('ISOAbbreviation', '')
                    language = article_data.get('Language', ['eng'])[0]
                    
                    all_articles.append({
                        "title": title,
                        "abstract": abstract_text,
                        "year": year,
                        "authors": authors,
                        "journal": journal_data.get('Title', ''),
                        "journal_short": journal_abbr, # Nuevo
                        "volume": volume, # Nuevo
                        "issue": issue,   # Nuevo (Number)
                        "pages": pages,   # Nuevo
                        "issn": issn,     # Nuevo
                        "language": language, # Nuevo
                        "doi": doi, 
                        "pubmed_id": str(pmid), # Nuevo (útil para URL)
                        "source": "PubMed",
                        "open_access": True,
                        "pdf_url": "" # Se llenará luego con normalize
                    })

                retstart += 100
        except Exception as e:
            logging.error(f"❌ PubMed error: {e}")
            
    return all_articles

# ============================================================
# ✅ OPENALEX (META: 200) - ARTÍCULOS PEER-REVIEWED
# ============================================================

def search_openalex(term_variations: List[List[str]], target: int = 200, concept_id: Optional[str] = None) -> List[Dict]:
    """
    Busca artículos Open Access en OpenAlex (250M+ obras indexadas).
    """
    articles_map = {}
    email = config.ACADEMIC_EMAIL
    
    # Filtro base
    base_filter = "is_oa:true,from_publication_date:2018-01-01,type:article"
    if concept_id:
        base_filter += f",concepts.id:{concept_id}"
        logging.info(f"   🔬 OpenAlex: Filtrando por concepto {concept_id}")

    for i, terms_group in enumerate(term_variations):
        if len(articles_map) >= target: break
        
        query = " ".join(terms_group).replace('  ', ' ').strip()
        # Limpiar comillas y operadores booleanos para la API de OpenAlex
        clean_query = query.replace('"', '').replace('(', '').replace(')', '')
        clean_query = re.sub(r'\b(AND|OR)\b', ' ', clean_query).strip()
        clean_query = re.sub(r'\s+', ' ', clean_query)
        
        logging.info(f"🔬 OpenAlex (Estrategia {i+1}): '{clean_query[:60]}...'")
        
        cursor = "*"  # Paginación con cursor
        page_count = 0
        
        while len(articles_map) < target and cursor and page_count < 5:
            params = {
                "search": clean_query,
                "filter": base_filter,
                "per_page": 50,
                "cursor": cursor,
                "mailto": email,
                "select": "id,doi,title,display_name,publication_year,authorships,primary_location,abstract_inverted_index,open_access"
            }
            
            try:
                r = requests.get("https://api.openalex.org/works", params=params, timeout=20)
                if r.status_code == 200:
                    data = r.json()
                    results = data.get('results', [])
                    if not results: break
                    
                    cursor = data.get('meta', {}).get('next_cursor')
                    
                    for work in results:
                        # Reconstruir abstract desde inverted_index
                        abstract = _reconstruct_abstract(work.get('abstract_inverted_index'))
                        if not abstract or len(abstract) < 150: continue
                        
                        doi_raw = work.get('doi', '') or ''
                        doi = doi_raw.replace('https://doi.org/', '')
                        
                        title = work.get('display_name', '')
                        if not title: continue
                        
                        # Deduplicación interna
                        work_id = work.get('id', '')
                        if work_id in articles_map: continue
                        
                        # Extraer autores
                        authors = []
                        for authorship in work.get('authorships', [])[:10]:
                            author = authorship.get('author', {})
                            name = author.get('display_name', '')
                            if name:
                                authors.append(name)
                        
                        # Extraer info de ubicación primaria (journal)
                        primary_loc = work.get('primary_location', {}) or {}
                        source = primary_loc.get('source', {}) or {}
                        
                        # URL del PDF
                        pdf_url = ""
                        oa_info = work.get('open_access', {}) or {}
                        pdf_url = oa_info.get('oa_url', '') or ''
                        
                        articles_map[work_id] = {
                            "title": title,
                            "abstract": abstract,
                            "year": work.get('publication_year') or 0,
                            "authors": authors,
                            "doi": doi,
                            "journal": source.get('display_name', ''),
                            "volume": "",  # OpenAlex no da volume en búsqueda
                            "issue": "",
                            "pages": "",
                            "url": doi_raw if doi_raw else work_id,
                            "pdf_url": pdf_url,
                            "open_access": True,
                            "source": "OpenAlex"
                        }
                    
                    page_count += 1
                    if not cursor: break
                    time.sleep(0.15)  # Respetar rate limit
                    
                elif r.status_code == 429:
                    logging.warning("⚠️ OpenAlex rate limit, esperando 2s...")
                    time.sleep(2)
                else:
                    logging.error(f"❌ OpenAlex HTTP {r.status_code}")
                    break
            except Exception as e:
                logging.error(f"❌ OpenAlex error: {e}")
                break
    
    logging.info(f"📚 OpenAlex: {len(articles_map)} artículos encontrados")
    return list(articles_map.values())


def _reconstruct_abstract(inverted_index: dict) -> str:
    """
    Reconstruye el texto del abstract desde el formato inverted_index de OpenAlex.
    OpenAlex almacena abstracts como {palabra: [posiciones]} para ahorrar espacio.
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    
    try:
        # Crear lista de (posición, palabra)
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        
        # Ordenar por posición y unir
        word_positions.sort(key=lambda x: x[0])
        return " ".join(word for _, word in word_positions)
    except Exception:
        return ""


# ============================================================
# ✅ EUROPE PMC (META: 200) - LITERATURA BIOMÉDICA AMPLIADA
# ============================================================

def search_europe_pmc(term_variations: List[List[str]], target: int = 200) -> List[Dict]:
    """
    Busca artículos Open Access en Europe PMC (versión ampliada de PubMed).
    No requiere API key. Devuelve metadata rica en JSON.
    """
    articles_map = {}
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    
    for i, terms_group in enumerate(term_variations):
        if len(articles_map) >= target: break
        
        raw_query = " ".join(terms_group).replace('  ', ' ').strip()
        # Construir query con filtros Europe PMC
        query = f'({raw_query}) AND (OPEN_ACCESS:y) AND (PUB_YEAR:[2018 TO 2026])'
        
        logging.info(f"🔬 Europe PMC (Estrategia {i+1}): '{raw_query[:60]}...'")
        
        cursor_mark = "*"
        page_count = 0
        
        while len(articles_map) < target and cursor_mark and page_count < 5:
            params = {
                "query": query,
                "resultType": "core",  # Metadata completa incluyendo abstract
                "pageSize": 50,
                "cursorMark": cursor_mark,
                "format": "json",
                "sort": "RELEVANCE"
            }
            
            try:
                r = requests.get(base_url, params=params, timeout=20)
                if r.status_code == 200:
                    data = r.json()
                    results = data.get('resultList', {}).get('result', [])
                    if not results: break
                    
                    next_cursor = data.get('nextCursorMark')
                    if next_cursor == cursor_mark:
                        break  # No más resultados
                    cursor_mark = next_cursor
                    
                    for article in results:
                        abstract = article.get('abstractText', '')
                        if not abstract or len(abstract) < 150: continue
                        
                        title = article.get('title', '')
                        if not title: continue
                        
                        # Crear key única
                        pmcid = article.get('pmcid', '')
                        pmid = article.get('pmid', '')
                        article_key = pmcid or pmid or title[:60]
                        if article_key in articles_map: continue
                        
                        # Extraer autores
                        authors = []
                        author_string = article.get('authorString', '')
                        if author_string:
                            # Europe PMC devuelve "Apellido AB, Apellido CD, ..."
                            authors = [a.strip() for a in author_string.split(',') if a.strip()]
                            # Reagrupar de a pares (cada autor tiene nombre + iniciales)
                            authors = _parse_epmc_authors(author_string)
                        
                        # DOI
                        doi = article.get('doi', '') or ''
                        
                        # Año
                        try:
                            year = int(article.get('pubYear', '0'))
                        except (ValueError, TypeError):
                            year = 0
                        
                        # URL del PDF (construir desde PMCID)
                        pdf_url = ""
                        if pmcid:
                            pdf_url = f"https://europepmc.org/articles/{pmcid}?pdf=render"
                        
                        # Páginas
                        pages = article.get('pageInfo', '') or ''
                        
                        articles_map[article_key] = {
                            "title": title,
                            "abstract": abstract,
                            "year": year,
                            "authors": authors,
                            "doi": doi,
                            "journal": article.get('journalTitle', '') or article.get('journalInfo', {}).get('journal', {}).get('title', ''),
                            "volume": article.get('journalVolume', ''),
                            "issue": article.get('issue', ''),
                            "pages": pages,
                            "pubmed_id": pmid,
                            "url": f"https://doi.org/{doi}" if doi else f"https://europepmc.org/article/MED/{pmid}" if pmid else "",
                            "pdf_url": pdf_url,
                            "open_access": True,
                            "source": "Europe PMC"
                        }
                    
                    page_count += 1
                    time.sleep(0.2)  # Cortesía
                    
                else:
                    logging.error(f"❌ Europe PMC HTTP {r.status_code}")
                    break
            except Exception as e:
                logging.error(f"❌ Europe PMC error: {e}")
                break
    
    logging.info(f"📚 Europe PMC: {len(articles_map)} artículos encontrados")
    return list(articles_map.values())


def _parse_epmc_authors(author_string: str) -> List[str]:
    """
    Parsea el string de autores de Europe PMC.
    Formato input: "García AB, López CD, Smith EF"
    Formato output: ["García AB", "López CD", "Smith EF"]
    """
    if not author_string:
        return []
    
    authors = []
    # Separar por ", " pero considerar que ", " también separa nombre de iniciales
    # Europe PMC usa formato: "Apellido Iniciales, Apellido Iniciales"
    parts = author_string.split(', ')
    
    i = 0
    while i < len(parts):
        part = parts[i].strip('. ')
        if not part:
            i += 1
            continue
        
        # Si la siguiente parte es solo iniciales (<=3 chars), combinar
        if i + 1 < len(parts) and len(parts[i+1].strip('. ')) <= 3:
            authors.append(f"{part}, {parts[i+1].strip()}")
            i += 2
        else:
            authors.append(part)
            i += 1
    
    return authors[:20]  # Limitar a 20 autores


# Eliminada _derive_api_queries (v6): La inteligencia se delegó al LLM en query_expander.generate_api_queries_with_llm

# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def search_articles(query_terms: List[str], max_results: int = 600, 
                    original_question: str = "") -> tuple:
    """
    Búsqueda INTELIGENTE con selección de fuentes por dominio.
    Retorna: (artículos, tiempo_búsqueda, queries_usadas_por_fuente)
    """
    start_time = time.perf_counter()
    all_articles = []

    # 1. Obtener queries del LLM (Sincronizado con el screening)
    
    # v6: El LLM genera las queries de API directamente preservando la intersección
    api_queries = generate_api_queries_with_llm(original_question or " ".join(query_terms))
    variations = [[q] for q in api_queries]
    
    if not variations:
        variations = generate_smart_variations(query_terms)

    # 🧠 DETECTAR DOMINIO para elegir fuentes
    domain_results = detect_search_domain(original_question) if original_question else {'id': 'general', 'anchors': '', 'score': 0}
    domain = domain_results['id']
    domain_injection = domain_results['anchors']
    
    openalex_concept = None

    target_per_source = max(1000, max_results // 4)

    if domain == 'cs':
        openalex_concept = "C41008148" # Computer Science
        target_cs = max(1500, max_results // 2)
        sources_config = {
            'Semantic Scholar': {'target': target_cs, 'fieldsOfStudy': 'Computer Science'},
            'OpenAlex': {'target': target_cs},
        }
    elif domain == 'medical':
        openalex_concept = "C71924100" # Medicine
        sources_config = {
            'Semantic Scholar': {'target': target_per_source, 'fieldsOfStudy': 'Medicine'},
            'PubMed': {'target': target_per_source},
            'OpenAlex': {'target': target_per_source},
            'Europe PMC': {'target': target_per_source},
        }
    elif domain == 'law':
        openalex_concept = "C199539241" # Law
        target_law = max(1500, max_results // 2)
        sources_config = {
            'Semantic Scholar': {'target': target_law},
            'OpenAlex': {'target': target_law},
        }
    elif domain == 'architecture':
        openalex_concept = "C13184196" # Architecture
        target_arch = max(1500, max_results // 2)
        sources_config = {
            'Semantic Scholar': {'target': target_arch},
            'OpenAlex': {'target': target_arch},
        }
    else:
        sources_config = {
            'Semantic Scholar': {'target': target_per_source},
            'PubMed': {'target': target_per_source},
            'OpenAlex': {'target': target_per_source},
            'Europe PMC': {'target': target_per_source},
        }

    # Eliminada inyección manual de términos por dominio (causa ruido en APIs)
    # El anclaje ahora se hace a nivel de screening semántico y n-gramas.
    
    # Diccionario para capturar las queries usadas por cada fuente
    search_queries_used = {src: [] for src in sources_config}
    
    active_sources = list(sources_config.keys())
    logging.info(f"🚀 Búsqueda en {len(active_sources)} fuentes: {', '.join(active_sources)} (Meta: {max_results})")
    
    # Capturar queries para logging
    for terms_group in variations:
        query = " ".join(terms_group).replace('  ', ' ').strip()
        clean_query = query.replace('"', '').replace('(', '').replace(')', '')
        clean_query = re.sub(r'\b(AND|OR)\b', ' ', clean_query).strip()
        clean_query = re.sub(r'\s+', ' ', clean_query)
        
        if 'Semantic Scholar' in search_queries_used:
            if query not in search_queries_used['Semantic Scholar']:
                search_queries_used['Semantic Scholar'].append(query)
        if 'PubMed' in search_queries_used:
            base_query = terms_group[0] if len(terms_group) == 1 else " AND ".join(terms_group)
            pubmed_query = f"({base_query}) AND (free full text[sb]) AND (2018:2026[dp])"
            if pubmed_query not in search_queries_used['PubMed']:
                search_queries_used['PubMed'].append(pubmed_query)
        if 'OpenAlex' in search_queries_used:
            if clean_query not in search_queries_used['OpenAlex']:
                search_queries_used['OpenAlex'].append(clean_query)
        if 'Europe PMC' in search_queries_used:
            raw_query = " ".join(terms_group).replace('  ', ' ').strip()
            epmc_query = f"({raw_query}) AND (OPEN_ACCESS:y) AND (PUB_YEAR:[2018 TO 2026])"
            if epmc_query not in search_queries_used['Europe PMC']:
                search_queries_used['Europe PMC'].append(epmc_query)
    
    if original_question:
        if 'Semantic Scholar' in search_queries_used:
            search_queries_used['Semantic Scholar'].insert(0, f"[SEMANTIC] {original_question[:100]}")

    # Ejecutar fuentes activas en paralelo
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        if 'Semantic Scholar' in sources_config:
            ss_config = sources_config['Semantic Scholar']
            futures.append(executor.submit(
                search_semantic_scholar_oa, variations, 
                ss_config['target'],
                fields_of_study=ss_config.get('fieldsOfStudy'),
                original_question=original_question
            ))
        
        if 'PubMed' in sources_config:
            futures.append(executor.submit(
                search_pubmed_oa, variations, sources_config['PubMed']['target']
            ))
        
        if 'OpenAlex' in sources_config:
            futures.append(executor.submit(
                search_openalex, variations, 
                sources_config['OpenAlex']['target'],
                concept_id=openalex_concept
            ))
        
        if 'Europe PMC' in sources_config:
            futures.append(executor.submit(
                search_europe_pmc, variations, sources_config['Europe PMC']['target']
            ))
        
        for f in as_completed(futures):
            try:
                all_articles.extend(f.result())
            except Exception as e:
                logging.error(f"Error hilo búsqueda: {e}")

    logging.info(f"📊 Total raw antes de deduplicación: {len(all_articles)}")

    # Deduplicación mejorada (DOI > título normalizado)
    unique = {}
    for a in all_articles:
        doi = a.get('doi', '').lower().strip()
        title_key = "".join(e for e in a.get('title', '').lower() if e.isalnum())[:60]
        key = doi if len(doi) > 5 else title_key
        if key and key not in unique:
            unique[key] = a
            
    final_list = list(unique.values())
    
    if len(final_list) > max_results:
        final_list = final_list[:max_results]
        
    logging.info(f"📚 Total únicos finales: {len(final_list)}")

    # Log de distribución por fuente
    source_counts = {}
    for a in final_list:
        src = a.get('source', 'Unknown')
        source_counts[src] = source_counts.get(src, 0) + 1
    for src, count in sorted(source_counts.items()):
        logging.info(f"   📌 {src}: {count} artículos")

    # Enriquecimiento PDF "Lazy"
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(enrich_initial_search_result, a) for a in final_list]
        for f in as_completed(futures): f.result()
            
    return final_list, time.perf_counter() - start_time, search_queries_used, len(all_articles)

