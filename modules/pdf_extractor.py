import requests
import logging
from typing import Dict
from io import BytesIO
import pdfplumber

# Configuración de límites
MAX_PDF_SIZE = 20 * 1024 * 1024  # 20MB
TIMEOUT = 10  # Reducido a 10s para ser más ágil en modo JIT
MAX_PAGES = 30  
MAX_CHARS = 80000 

def extract_text_from_pdf_url(pdf_url: str) -> str:
    """Descarga y extrae texto de un PDF de forma robusta y rápida."""
    if not pdf_url: return ""
    
    try:
        # Headers rotativos simples
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(pdf_url, stream=True, timeout=TIMEOUT, headers=headers)
        
        if response.status_code != 200: 
            logging.warning(f"⚠️ PDF no disponible ({response.status_code}): {pdf_url[:50]}...")
            return ""
        
        # Verificar tamaño
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > MAX_PDF_SIZE:
            return ""

        pdf_content = response.content
        
        text_content = ""
        with pdfplumber.open(BytesIO(pdf_content)) as pdf:
            pages = pdf.pages[:MAX_PAGES]
            text_content = "\n".join([p.extract_text() or "" for p in pages])
            
        clean_text = text_content[:MAX_CHARS].strip()
        return clean_text if clean_text else ""
        
    except Exception as e:
        return ""

def enrich_initial_search_result(article: Dict) -> Dict:
    """FASE 1: Preparación metadata."""
    abstract = article.get('abstract', '') or ''
    
    if len(abstract) > 800:
        article['full_text'] = abstract
        article['full_text_source'] = 'abstract_proxy'
        article['is_pdf_downloaded'] = False
    else:
        article['full_text'] = abstract
        article['full_text_source'] = 'abstract_short'
        article['is_pdf_downloaded'] = False
        
    if article.get('pdf_url') and len(str(article.get('pdf_url'))) > 10:
        article['needs_pdf_download'] = True
    else:
        article['needs_pdf_download'] = False
        
    return article

def download_full_text_lazy(article: Dict) -> Dict:
    """
    FASE 2: Descarga Real (On Demand).
    Se llama desde main.py -> generate_column si hace falta.
    """
    if article.get('is_pdf_downloaded'):
        return article
        
    url = article.get('pdf_url')
    if not url: return article
    
    full_text = extract_text_from_pdf_url(url)
    
    if len(full_text) > 1000:
        article['full_text'] = full_text
        article['full_text_source'] = 'pdf_download'
        article['is_pdf_downloaded'] = True
        article['needs_pdf_download'] = False
    else:
        # Si falla, marcamos que ya lo intentamos para no reintentar infinitamente
        article['is_pdf_downloaded'] = False 
        article['needs_pdf_download'] = False 
        
    return article