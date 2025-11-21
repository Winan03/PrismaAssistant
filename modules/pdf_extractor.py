
import requests
import logging
from typing import Dict
from io import BytesIO
import pdfplumber
import time

logging.basicConfig(level=logging.INFO)

# ✅ LÍMITES AJUSTADOS
MAX_PDF_SIZE = 30 * 1024 * 1024  # 15MB (antes 10MB)
TIMEOUT = 25  # 15s (antes 10s)
MAX_PAGES = 40  # ✅ Lee hasta 15 páginas (antes 5)

def extract_text_from_pdf_url(pdf_url: str) -> str:
    
    if not pdf_url:
        return ""
    
    logging.info(f"💾 Intentando extraer texto de PDF: {pdf_url[:60]}...")

    try:
        # ✅ HEADERS CON USER-AGENT (evita 403 en muchos sitios)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Descarga con headers
        response = requests.get(pdf_url, stream=True, timeout=TIMEOUT, headers=headers)
        
        if response.status_code == 403:
            logging.warning(f"❌ Error HTTP 403 al descargar PDF.")
            return ""
        
        if response.status_code != 200:
            logging.warning(f"❌ Error HTTP {response.status_code} al descargar PDF.")
            return ""

        # Verificar tamaño
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > MAX_PDF_SIZE:
            logging.warning("⚠️ PDF es demasiado grande, omitiendo.")
            return ""

        # Leer contenido
        pdf_content = response.content
        
        # ✅ EXTRACCIÓN MEJORADA
        extracted_text = ""
        with pdfplumber.open(BytesIO(pdf_content)) as pdf:
            total_pages = len(pdf.pages)
            pages_to_read = min(total_pages, MAX_PAGES)
            
            for page_num, page in enumerate(pdf.pages[:pages_to_read], 1):
                page_text = page.extract_text() or ""
                extracted_text += f"\n--- Page {page_num} ---\n{page_text}"
            
            if total_pages > MAX_PAGES:
                logging.info(f"   📄 Extraídas {pages_to_read}/{total_pages} páginas")
        
        # ✅ LÍMITE FINAL: 30,000 caracteres (coherente con database.py)
        final_text = extracted_text[:30000].strip()
        
        if final_text:
            logging.info(f"   ✅ Extraídos {len(final_text)} caracteres del PDF")
        
        return final_text
        
    except requests.exceptions.Timeout:
        logging.error("❌ Tiempo de espera (Timeout) agotado para la descarga.")
        return ""
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Error de red al descargar PDF: {e}")
        return ""
    except Exception as e:
        logging.error(f"❌ Fallo en la extracción del PDF: {e}")
        return ""

def enrich_article_with_full_text(article: Dict) -> Dict:
    """
    Añade el campo 'full_text' al artículo si hay PDF URL.
    ✅ MEJORAS:
    - Verifica si ya tiene full_text (evita re-descargar)
    - Añade campo 'pdf_extraction_status' para debugging
    """
    # ✅ EVITAR RE-DESCARGA
    if article.get('full_text'):
        return article
    
    pdf_url = article.get('pdf_url', '')
    
    if pdf_url:
        full_text = extract_text_from_pdf_url(pdf_url)
        
        if full_text:
            article['full_text'] = full_text
            article['pdf_extraction_status'] = 'success'
        else:
            article['pdf_extraction_status'] = 'failed'
    else:
        article['pdf_extraction_status'] = 'no_pdf_url'
    
    return article