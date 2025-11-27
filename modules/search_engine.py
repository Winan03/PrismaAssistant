import time
import requests
import config
import logging
import re
from typing import List, Dict
from Bio import Entrez
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules.pdf_extractor import enrich_initial_search_result

Entrez.email = "prisma-assistant@upao.edu.pe"

# ============================================================
# 🧠 GESTIÓN DE CONSULTAS (ESTRATEGIA MULTI-QUERY)
# ============================================================

def generate_smart_variations(terms: List[str]) -> List[List[str]]:
    """
    Implementa la estrategia 'Divide y Vencerás' para maximizar el Recall.
    """
    if any((" AND " in t or " OR " in t) for t in terms):
        logging.info("⚡ Detectadas Estrategias Multi-Query avanzadas (Grok).")
        return [[t] for t in terms]

    # Bloques de construcción (Fallback)
    ai_broad = '("artificial intelligence" OR "machine learning" OR "deep learning")'
    cvd_broad = '("cardiovascular disease" OR "heart disease")'
    goal_early = '("early detection" OR "diagnosis")'
    
    return [
        [ai_broad, cvd_broad, goal_early], 
        [ai_broad, '("heart failure" OR "atrial fibrillation")', goal_early], 
        [ai_broad, cvd_broad, '("AUC" OR "accuracy")']
    ]

# ============================================================
# ✅ SEMANTIC SCHOLAR (META AJUSTADA: 250)
# ============================================================

def search_semantic_scholar_oa(term_variations: List[List[str]], target: int = 250) -> List[Dict]:
    if not config.SEMANTIC_SCHOLAR_API_KEY: return []

    articles_map = {} 
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": config.SEMANTIC_SCHOLAR_API_KEY}
    # 🔥 SOLICITAMOS MÁS CAMPOS AQUÍ: publicationDate, venue, etc.
    fields = "title,year,abstract,authors,externalIds,url,openAccessPdf,journal,publicationDate,venue"

    for i, terms_group in enumerate(term_variations):
        if len(articles_map) >= target: break
        
        query = " ".join(terms_group).replace('  ', ' ').strip()
        logging.info(f"🔬 Semantic Scholar (Estrategia {i+1}): '{query[:60]}...'")

        offset = 0
        while len(articles_map) < target and offset < 500: 
            params = {
                "query": query,
                "limit": 100,
                "fields": fields,
                "openAccessPdf": "true", 
                "offset": offset
            }

            try:
                r = requests.get(url, headers=headers, params=params, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    if not data.get('data'): break 
                    
                    added_count = 0
                    for paper in data['data']:
                        if not paper.get('openAccessPdf') or not paper.get('openAccessPdf', {}).get('url'): continue
                        
                        abstract = paper.get('abstract')
                        if not abstract or len(abstract) < 150: continue

                        paper_id = paper.get('paperId')
                        if paper_id not in articles_map:
                            # Extracción robusta de metadatos de revista
                            journal_info = paper.get('journal') or {}
                            venue = paper.get('venue') or ""
                            
                            articles_map[paper_id] = {
                                "title": paper.get('title', ''),
                                "authors": [a['name'] for a in paper.get('authors', [])],
                                "doi": paper.get('externalIds', {}).get('DOI', ''),
                                "year": paper.get('year') or 0,
                                "abstract": abstract,
                                # Datos enriquecidos para BibTeX
                                "journal": journal_info.get('name', venue),
                                "volume": journal_info.get('volume', ''),
                                "issue": journal_info.get('pages', '').split('-')[0] if '-' in journal_info.get('pages', '') else "", # A veces pages trae issue implícito, mejor dejar vacío si no es claro
                                "pages": journal_info.get('pages', ''),
                                "url": paper.get('url', ''),
                                "pdf_url": paper['openAccessPdf']['url'],
                                "open_access": True,
                                "source": "Semantic Scholar"
                            }
                            added_count += 1
                    
                    offset += 100
                    if added_count == 0: break 
                else: break
            except Exception as e:
                logging.error(f"❌ Semantic Scholar error: {e}")
                break

    return list(articles_map.values())

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
# ✅ ARXIV (META AJUSTADA: 100)
# ============================================================

def search_arxiv_light(terms_list: List[str], max_results: int = 100) -> List[Dict]:
    articles = []
    import urllib.parse
    import xml.etree.ElementTree as ET
    
    if isinstance(terms_list[0], str) and "OR" not in terms_list[0]:
         query_str = " AND ".join(terms_list[:3])
    else:
         query_str = " AND ".join(terms_list)

    url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query_str)}&start=0&max_results={max_results}"
    
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            root = ET.fromstring(r.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns).text.replace('\n', ' ')
                summary = entry.find('atom:summary', ns).text.replace('\n', ' ')
                pub = entry.find('atom:published', ns).text[:4]
                pdf = entry.find('atom:link[@title="pdf"]', ns)
                doi_elem = entry.find('arxiv:doi', ns)
                
                # Categoría primaria (como "serie")
                category = entry.find('atom:category', ns)
                primary_cat = category.attrib['term'] if category is not None else ""

                articles.append({
                    "title": title,
                    "abstract": summary,
                    "year": int(pub),
                    "source": "arXiv",
                    "open_access": True,
                    "pdf_url": pdf.attrib['href'] if pdf is not None else "",
                    "journal": "arXiv Preprint",
                    "doi": doi_elem.text if doi_elem is not None else "", # Nuevo: Extrae DOI si existe
                    "series": primary_cat, # Nuevo: Categoría arXiv como serie
                    "authors": [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
                })
    except Exception as e:
        logging.error(f"❌ ArXiv error: {e}")

    return articles

# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def search_articles(query_terms: List[str], max_results: int = 500) -> tuple:
    """
    Meta Total Ajustada: 500-600 candidatos raw -> ~450 únicos.
    """
    start_time = time.perf_counter()
    variations = generate_smart_variations(query_terms)
    all_articles = []

    logging.info(f"🚀 Búsqueda Masiva Optimizada (Meta: {max_results})")

    with ThreadPoolExecutor(max_workers=3) as executor:
        f1 = executor.submit(search_semantic_scholar_oa, variations, 250)
        f2 = executor.submit(search_pubmed_oa, variations, 200)
        arxiv_query = variations[0] if variations else query_terms
        f3 = executor.submit(search_arxiv_light, arxiv_query, 100)
        
        for f in as_completed([f1, f2, f3]):
            try:
                all_articles.extend(f.result())
            except Exception as e:
                logging.error(f"Error hilo búsqueda: {e}")

    # Deduplicación
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

    # Enriquecimiento PDF "Lazy"
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(enrich_initial_search_result, a) for a in final_list]
        for f in as_completed(futures): f.result()
            
    return final_list, time.perf_counter() - start_time