"""
Búsqueda Multi-API OPTIMIZADA PARA VELOCIDAD Y ENRIQUECIMIENTO DE TEXTO COMPLETO
"""
import time
import requests
from semanticscholar import SemanticScholar
import config
import logging
from typing import List, Dict
from Bio import Entrez
from concurrent.futures import ThreadPoolExecutor, as_completed

# IMPORTACIÓN NECESARIA para enriquecer el artículo
from modules.pdf_extractor import enrich_article_with_full_text 

# Configurar email para PubMed
Entrez.email = "prisma-assistant@upao.edu.pe"


def search_semantic_scholar_fast(terms: List[str], max_results: int = 200) -> List[Dict]:
    """
    Búsqueda RÁPIDA en Semantic Scholar, capturando la URL del PDF (si Open Access).
    """
    if not config.SEMANTIC_SCHOLAR_API_KEY:
        logging.warning("⚠️ Semantic Scholar API Key no configurada")
        return []
    
    articles = []
    
    try:
        sch = SemanticScholar(api_key=config.SEMANTIC_SCHOLAR_API_KEY)
        query = " ".join(terms[:5])
        
        logging.info(f"🔬 Semantic Scholar: 1 query rápida con 100 resultados...")
        logging.info(f"   Query: '{query[:80]}...'")
        
        # Semantic Scholar pagina cada 100, así que solo pedimos 100
        results = sch.search_paper(query, limit=100)
        
        seen_titles = set()
        
        for paper in results:
            if len(articles) >= max_results:
                break
            
            if not paper.title or paper.title in seen_titles:
                continue
            
            seen_titles.add(paper.title)
            
            year = paper.year if paper.year else 0
            
            doi = paper.externalIds.get('DOI', '') if paper.externalIds and isinstance(paper.externalIds, dict) else ""
            
            journal_name = ""
            if paper.journal:
                if hasattr(paper.journal, 'name'):
                    journal_name = str(paper.journal.name) if paper.journal.name else ""
                elif isinstance(paper.journal, str):
                    journal_name = paper.journal
            
            # CAPTURA DE PDF URL (CRÍTICO)
            pdf_url = ""
            if paper.openAccessPdf and paper.openAccessPdf.get('url'):
                 pdf_url = paper.openAccessPdf['url']
                 
            articles.append({
                "title": paper.title,
                "authors": [str(a.name) for a in paper.authors] if paper.authors else [],
                "doi": doi,
                "year": year,
                "abstract": paper.abstract or "",
                "journal": journal_name,
                "url": paper.url or "",
                "pdf_url": pdf_url, # <-- CAMPO AÑADIDO
                "source": "Semantic Scholar"
            })
        
        logging.info(f"   ✅ {len(articles)} artículos en 1 query")
        
    except Exception as e:
        logging.error(f"❌ Semantic Scholar error: {e}")
    
    return articles


def search_pubmed_fast(terms: List[str], max_results: int = 150) -> List[Dict]:
    """
    Búsqueda con queries booleanas BALANCEADAS en PubMed.
    """
    articles = []
    
    try:
        llm_core = []
        for t in terms:
            t_lower = t.lower()
            if any(kw in t_lower for kw in ['language model', 'llm', 'gpt', 'chatgpt', 'transformer']):
                if 'language model' in t_lower:
                    llm_core.append('large language model')
                elif 'gpt' in t_lower or 'chatgpt' in t_lower:
                    llm_core.append('GPT')
                else:
                    llm_core.append(t)
        
        if not llm_core:
            llm_core = ['large language model', 'GPT', 'artificial intelligence']
        
        med_broad = ['clinical', 'diagnosis', 'diagnostic', 'medical', 'healthcare']
        
        llm_query = " OR ".join([f'"{t}"[Title/Abstract]' for t in llm_core[:3]])
        med_query = " OR ".join([f'{t}[Title/Abstract]' for t in med_broad[:5]])
        
        query = f"({llm_query}) AND ({med_query})"
        
        logging.info(f"🔬 PubMed: Query balanceada...")
        logging.info(f"   Query: {query[:200]}...")
        
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        id_list = record["IdList"]
        
        if not id_list:
            logging.warning("⚠️ PubMed: Sin resultados con query AND, probando OR...")
            query_fallback = f"({llm_query}) {med_broad[0]}[Title/Abstract]"
            handle = Entrez.esearch(db="pubmed", term=query_fallback, retmax=max_results, sort="relevance")
            record = Entrez.read(handle)
            handle.close()
            id_list = record["IdList"]
            
            if not id_list:
                return []
        
        logging.info(f"   Encontrados {len(id_list)} IDs, obteniendo detalles...")
        
        for i in range(0, len(id_list), 100):
            batch_ids = id_list[i:i+100]
            
            handle = Entrez.efetch(db="pubmed", id=batch_ids, rettype="medline", retmode="text")
            records = handle.read()
            handle.close()
            
            current_record = {}
            for line in records.split("\n"):
                if line.startswith("PMID- "):
                    if current_record and "title" in current_record:
                        articles.append({
                            "title": current_record.get("title", ""),
                            "authors": current_record.get("authors", []),
                            "doi": current_record.get("doi", ""),
                            "year": current_record.get("year", 0),
                            "abstract": current_record.get("abstract", ""),
                            "journal": current_record.get("journal", ""),
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{current_record.get('pmid', '')}/",
                            "source": "PubMed",
                            "pdf_url": "" # PubMed rara vez da la URL directa al PDF
                        })
                    current_record = {"pmid": line.split("- ")[1].strip()}
                
                elif line.startswith("TI  - "):
                    current_record["title"] = line.split("- ", 1)[1].strip()
                elif line.startswith("AU  - "):
                    if "authors" not in current_record: current_record["authors"] = []
                    current_record["authors"].append(line.split("- ")[1].strip())
                elif line.startswith("DP  - "):
                    try:
                        current_record["year"] = int(line.split("- ")[1].split()[0])
                    except: pass
                elif line.startswith("AB  - "):
                    current_record["abstract"] = line.split("- ", 1)[1].strip()
                elif line.startswith("TA  - "):
                    current_record["journal"] = line.split("- ")[1].strip()
                elif line.startswith("AID - ") and "[doi]" in line:
                    current_record["doi"] = line.split("- ")[1].split("[")[0].strip()
            
            if current_record and "title" in current_record:
                articles.append({
                    "title": current_record.get("title", ""),
                    "authors": current_record.get("authors", []),
                    "doi": current_record.get("doi", ""),
                    "year": current_record.get("year", 0),
                    "abstract": current_record.get("abstract", ""),
                    "journal": current_record.get("journal", ""),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{current_record.get('pmid', '')}/",
                    "source": "PubMed",
                    "pdf_url": ""
                })
            
            time.sleep(0.3)
        
        logging.info(f"   ✅ {len(articles)} artículos")
        
    except Exception as e:
        logging.error(f"❌ PubMed error: {e}")
    
    return articles


def search_arxiv_fast(terms: List[str], max_results: int = 50) -> List[Dict]:
    """
    Búsqueda RÁPIDA en arXiv.
    """
    articles = []
    
    try:
        import urllib.parse
        core_terms = " AND ".join([f'"{t}"' for t in terms[:3]])
        query = f"{core_terms}"
        encoded_query = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={max_results}&sortBy=relevance"
        
        logging.info(f"🔬 arXiv: 1 query rápida con {max_results} resultados...")
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', namespace)
            
            for entry in entries:
                title = entry.find('atom:title', namespace)
                summary = entry.find('atom:summary', namespace)
                published = entry.find('atom:published', namespace)
                link = entry.find('atom:id', namespace)
                pdf_link = entry.find('atom:link[@title="pdf"]', namespace) # Capturar PDF
                
                authors = [author.find('atom:name', namespace).text for author in entry.findall('atom:author', namespace) if author.find('atom:name', namespace) is not None]
                
                year = int(published.text[:4]) if published is not None else 0
                
                articles.append({
                    "title": title.text.strip() if title is not None else "",
                    "authors": authors,
                    "doi": "",
                    "year": year,
                    "abstract": summary.text.strip() if summary is not None else "",
                    "journal": "arXiv preprint",
                    "url": link.text if link is not None else "",
                    "pdf_url": pdf_link.attrib['href'] if pdf_link is not None else "", # <-- CAMPO AÑADIDO
                    "source": "arXiv"
                })
            
            logging.info(f"   ✅ {len(articles)} artículos")
        
    except Exception as e:
        logging.error(f"❌ arXiv error: {e}")
    
    return articles


def search_articles(query_terms: List[str], max_results: int = 400) -> tuple:
    """
    Búsqueda PARALELA ultra-rápida y enriquecimiento de PDF.
    """
    start = time.perf_counter()
    
    logging.info(f"🔍 Búsqueda RÁPIDA con {len(query_terms)} términos")
    logging.info(f"   Términos: {', '.join(query_terms[:5])}...")
    
    ss_max = 100
    pm_max = 100
    ax_max = 50
    all_articles = []
    
    # 1. Ejecutar las 3 búsquedas en paralelo
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_ss = executor.submit(search_semantic_scholar_fast, query_terms, ss_max)
        future_pm = executor.submit(search_pubmed_fast, query_terms, pm_max)
        future_ax = executor.submit(search_arxiv_fast, query_terms, ax_max)
        
        for future in as_completed([future_ss, future_pm, future_ax]):
            try:
                articles = future.result()
                all_articles.extend(articles)
            except Exception as e:
                logging.error(f"❌ Error en búsqueda paralela: {e}")
    
    # 2. Eliminar duplicados por título
    seen_titles = set()
    unique_articles = []
    for article in all_articles:
        title_lower = article["title"].lower().strip()
        if title_lower and title_lower not in seen_titles:
            seen_titles.add(title_lower)
            unique_articles.append(article)
    
    # 3. EXTRACCIÓN PARALELA DE PDF (ENRIQUECIMIENTO)
    articles_to_enrich = [a for a in unique_articles if a.get('pdf_url')]
    logging.info(f"📄 Iniciando enriquecimiento de {len(articles_to_enrich)} artículos con texto completo...")
    
    start_enrich = time.perf_counter()
    enriched_map = {} # Usaremos un mapa para actualizar la lista original

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_article = {
            executor.submit(enrich_article_with_full_text, article): article
            for article in articles_to_enrich
        }
        for future in as_completed(future_to_article):
            enriched_article = future.result()
            # Guardar el artículo enriquecido por su título para una actualización fácil
            enriched_map[enriched_article["title"].lower().strip()] = enriched_article
    
    t_enrich = time.perf_counter() - start_enrich
    logging.info(f"✅ Enriquecimiento completado en {t_enrich:.2f}s.")
    
    # 4. ACTUALIZAR la lista unique_articles con el texto completo
    final_articles = []
    for article in unique_articles:
        title_key = article["title"].lower().strip()
        
        if title_key in enriched_map:
             # Usar el artículo enriquecido (que ahora tiene 'full_text')
             final_articles.append(enriched_map[title_key])
        else:
             # Usar el artículo original (solo con abstract)
             final_articles.append(article) 
             
    # 5. RESUMEN FINAL
    total_time = time.perf_counter() - start
    
    sources_count = {}
    for a in final_articles:
        source = a.get("source", "Unknown")
        sources_count[source] = sources_count.get(source, 0) + 1
    
    logging.info(f"📊 RESUMEN BÚSQUEDA RÁPIDA:")
    for source, count in sources_count.items():
        logging.info(f"   - {source}: {count}")
    logging.info(f"   - Total único (con enriquecimiento): {len(final_articles)} en {total_time:.2f}s")
    logging.info(f"   ⚡ Velocidad: {len(final_articles)/total_time:.1f} artículos/segundo")
    
    return final_articles, total_time