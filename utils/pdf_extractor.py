import re
import hashlib
import time
import requests
import logging
import threading
from typing import Dict, Optional
from io import BytesIO
from pathlib import Path
import pdfplumber

# v15.6: Lock global para serializar peticiones por dominio
_DOMAIN_LOCKS = {}
_LOCK_MAP_LOCK = threading.Lock()

def get_domain_lock(domain: str):
    with _LOCK_MAP_LOCK:
        if domain not in _DOMAIN_LOCKS:
            _DOMAIN_LOCKS[domain] = threading.Lock()
        return _DOMAIN_LOCKS[domain]

# Configuracion de limites (v15.5: Limites extendidos para evitar perdida de datos)
MAX_PDF_SIZE = 100 * 1024 * 1024  # 100MB
TIMEOUT = 30                      # v15.6: Aumentado a 30s
MAX_PAGES = 500  
MAX_CHARS = 1000000               # 1M caracteres

# v15.4: Cache persistente en disco para evitar re-descargas
PDF_CACHE_DIR = Path(__file__).parent.parent / "pdf_text_cache"
PDF_CACHE_DIR.mkdir(exist_ok=True)

# v15.4: Delays por dominio para evitar rate-limiting
DOMAIN_DELAYS = {
    'arxiv.org': 1.5,           # ArXiv: max ~40 req/min
    'aclanthology.org': 1.0,    # ACL Anthology
    'link.springer.com': 1.0,   # Springer
    'kar.kent.ac.uk': 0.5,
    'escholarship.org': 0.5,
}

# v15.2: Unpaywall API para papers bajo paywall
UNPAYWALL_EMAIL = "jnacarinoa1@upao.edu.pe"
UNPAYWALL_API = "https://api.unpaywall.org/v2/{doi}?email={email}"

def _get_cache_key(url: str) -> str:
    """Genera clave unica de cache basada en la URL."""
    return hashlib.md5(url.encode()).hexdigest()

def _read_cache(url: str) -> Optional[str]:
    """Lee texto cacheado si existe. None si no hay cache."""
    cache_file = PDF_CACHE_DIR / f"{_get_cache_key(url)}.txt"
    if cache_file.exists():
        try:
            text = cache_file.read_text(encoding='utf-8')
            if len(text) > 1000:
                logging.info(f"   [Cache] Hit: {url[:50]}")
                return text
        except Exception:
            pass
    return None

def _write_cache(url: str, text: str):
    """Guarda texto extraido en cache."""
    if len(text) < 500: return
    cache_file = PDF_CACHE_DIR / f"{_get_cache_key(url)}.txt"
    try:
        cache_file.write_text(text[:MAX_CHARS], encoding='utf-8')
    except Exception:
        pass


def _extract_doi_from_url(url: str) -> Optional[str]:
    """Extrae el DOI de una URL de paper (ACM, IEEE, DOI.org, arXiv, etc.)."""
    if not url: return None
    patterns = [
        r'10\.\d{4,9}/[^\s&?#]+',  # DOI estandar: 10.XXXX/...
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            doi = m.group(0).rstrip('/.,')
            return doi
    return None

def _get_pdf_via_unpaywall(doi: str) -> Optional[bytes]:
    """
    v15.3: Consulta Unpaywall API para obtener PDF de acceso abierto REAL.
    Filtra URLs de portales paywalled (ej: ACM devuelve su propia URL como 'OA')
    y prioriza repositorios verdaderamente abiertos.
    """
    if not doi: return None

    # Repositorios verdaderamente abiertos (prioridad ALTA)
    TRUSTED_OPEN = [
        'arxiv.org', 'europepmc.org', 'pubmedcentral', 'pmc.ncbi',
        'biorxiv.org', 'medrxiv.org', 'zenodo.org', 'figshare.com',
        'hal.science', 'hal.archives-ouvertes', 'osf.io',
        'researchgate.net', 'semanticscholar.org', 'unpaywall.org',
        'scholar.harvard.edu', 'eprints.', 'dspace.', 'repository.',
        '.ac.uk', '.edu/', 'kar.kent.ac.uk', 'escholarship.org',
        'engrxiv.org', 'preprints.org', 'ssrn.com', 'dergipark',
        'aclanthology.org', 'emnlp', 'naacl', 'acl2', 'openreview.net',
    ]

    # Portales que requieren autenticacion aunque Unpaywall los marque como OA
    PAYWALLED_DOMAINS = [
        'dl.acm.org', 'ieeexplore.ieee.org', 'link.springer.com',
        'www.mdpi.com', 'sciencedirect.com', 'wiley.com', 'tandfonline',
        'nature.com/articles', 'doi.org', 'downloads.hindawi.com',
        'igi-global.com', 'iopscience.iop.org',
    ]

    def is_trusted(url: str) -> bool:
        return any(t in url for t in TRUSTED_OPEN)

    def is_paywalled(url: str) -> bool:
        return any(p in url for p in PAYWALLED_DOMAINS)

    def _try_download(url: str) -> Optional[bytes]:
        if not url or is_paywalled(url):
            return None
        try:
            r = requests.get(
                url, timeout=TIMEOUT + 10,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; PrismaAssistant/1.0; mailto:jnacarinoa1@upao.edu.pe)'},
                allow_redirects=True
            )
            if r.status_code == 200:
                ct = r.headers.get('Content-Type', '').lower()
                if 'html' not in ct and len(r.content) > 5000:
                    return r.content
        except Exception:
            pass
        return None

    try:
        api_url = UNPAYWALL_API.format(doi=doi, email=UNPAYWALL_EMAIL)
        resp = requests.get(api_url, timeout=10)
        if resp.status_code != 200:
            return None

        data = resp.json()
        oa_locations = data.get('oa_locations', [])
        best_oa = data.get('best_oa_location')

        # Recolectar todas las URLs candidatas (pdf y landing page) con su puntuacion
        candidates = []
        for loc in oa_locations:
            pdf_url = loc.get('url_for_pdf')
            landing_url = loc.get('url')
            host = loc.get('host_type', '')
            for url in [pdf_url, landing_url]:
                if not url: continue
                score = 0
                if is_paywalled(url): continue       # Descartar paywalled
                if is_trusted(url): score += 10      # Preferir repositorios abiertos
                if pdf_url and url == pdf_url: score += 5   # Preferir URL directa a PDF
                if host == 'repository': score += 3
                candidates.append((score, url))

        # Ordenar por puntuacion descendente y probar en orden
        candidates.sort(key=lambda x: x[0], reverse=True)

        for score, url in candidates:
            logging.info(f"   [Unpaywall] Intentando OA ({score}pts): {url[:70]}")
            content = _try_download(url)
            if content:
                logging.info(f"   [Unpaywall] Exito con repositorio abierto: {len(content)} bytes")
                return content

        logging.info(f"   [Unpaywall] Sin PDF verdaderamente abierto para DOI: {doi}")
        return None
    except Exception as e:
        logging.debug(f"[Unpaywall] Error: {e}")
        return None


def extract_text_from_pdf_url(pdf_url: str, pdf_content: bytes = None) -> str:
    """Descarga y extrae texto de un PDF de forma robusta (o usa contenido directo)."""
    if not pdf_url and not pdf_content: return ""
    
    # v15.4: Comprobar cache antes de descargar
    if pdf_content is None:
        cached = _read_cache(pdf_url)
        if cached:
            return cached
    
    # v14.1: Si se provee contenido directo, saltar descarga
    if pdf_content is None:
        # Normalizar URL de ArXiv (v14.1)
        if "arxiv.org/abs/" in pdf_url:
            pdf_url = pdf_url.replace("/abs/", "/pdf/")
        
        session = requests.Session()
        
        import random
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Edge/122.0.2365.92'
        ]
        chosen_ua = random.choice(user_agents)
        
        try:
            from urllib.parse import urlparse
            p_url = urlparse(pdf_url)
            domain = p_url.netloc
            base_url = f"{p_url.scheme}://{domain}"

            # v15.6: Serializar peticion por dominio usando Lock
            lock = get_domain_lock(domain)
            with lock:
                # v15.4: Aplicar delay segun dominio para evitar bloqueos
                for d, delay in DOMAIN_DELAYS.items():
                    if d in domain:
                        logging.info(f"   [RateLimit] Esperando {delay}s para {domain}...")
                        time.sleep(delay)
                        break

                # Headers realistas (v14.1)
                headers = {
                    'User-Agent': chosen_ua,
                    'Accept': 'application/pdf,application/x-pdf,*/*',
                    'Accept-Language': 'en-US,en;q=0.9,es;q=0.8',
                    'Connection': 'keep-alive',
                }

                # Algunos sitios bloquean si no hay un Referer de su propia página principal
                if any(x in domain for x in ["mdpi.com", "acm.org", "ieee.org", "sciencedirect.com", "springer.com"]):
                    try:
                        session.get(base_url, timeout=7, headers={'User-Agent': chosen_ua})
                        headers['Referer'] = base_url
                    except:
                        pass

                # Intento 1: Con Session y Headers completos
                # v15.6: Quitamos stream=True para mayor estabilidad en descargas completas
                response = session.get(pdf_url, timeout=TIMEOUT, headers=headers, allow_redirects=True)
            
            if response.status_code != 200: 
                # Intento 2 (Stateless): Sin Session para evitar bloqueos por Cookies/IP persistente
                logging.info(f" 🔄 Reintento apolítico (sin session) para {domain}...")
                # Usar requests directo (limpio) con un UA diferente
                clean_headers = {'User-Agent': random.choice(user_agents), 'Accept': 'application/pdf'}
                response = requests.get(pdf_url, timeout=TIMEOUT + 10, headers=clean_headers, allow_redirects=True)
                
                if response.status_code != 200:
                    logging.warning(f"\u26a0\ufe0f PDF no disponible ({response.status_code}): {pdf_url[:60]}...")
                    # === v15.2: FALLBACK UNPAYWALL ===
                    doi = _extract_doi_from_url(pdf_url)
                    if doi:
                        logging.info(f" [Unpaywall] Buscando version abierta, DOI: {doi}")
                        unpaywall_content = _get_pdf_via_unpaywall(doi)
                        if unpaywall_content:
                            pdf_content = unpaywall_content
                        else:
                            return ""
                    else:
                        return ""
            
            # Verificar si realmente es un PDF (v14.1)
            if pdf_content is None:
                content_type = response.headers.get('Content-Type', '').lower()
                if 'text/html' in content_type:
                    logging.warning(f" \u274c Landing page detectada: {domain}")
                    # === v15.2: FALLBACK UNPAYWALL para landing pages ===
                    doi = _extract_doi_from_url(pdf_url)
                    if doi:
                        logging.info(f" [Unpaywall] Intentando con DOI: {doi}")
                        unpaywall_content = _get_pdf_via_unpaywall(doi)
                        if unpaywall_content:
                            pdf_content = unpaywall_content
                        else:
                            return ""
                    else:
                        return ""
                else:
                    # Verificar tamano
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > MAX_PDF_SIZE:
                        logging.warning(f" \u274c PDF demasiado grande: {content_length} bytes")
                        return ""
                    pdf_content = response.content

        except Exception as e:
            logging.error(f"Error descargando PDF ({pdf_url[:40]}): {e}")
            return ""
    
    # Si pdf_content es None aquí, significa que hubo un error en la descarga o no se pudo obtener.
    # Si se proporcionó directamente, o se descargó con éxito, continuamos.
    if pdf_content is None:
        return ""

    try:
        text_content = ""
        with pdfplumber.open(BytesIO(pdf_content)) as pdf:
            pages = pdf.pages[:MAX_PAGES]
            
            for p in pages:
                # v14.0: Extracción con conciencia de Layout (manejo de columnas)
                # Ordenamos palabras por su posición vertical (top) y luego horizontal (left)
                # Esto ayuda enormemente en papers de doble columna.
                words = p.extract_words(x_tolerance=3, y_tolerance=3)
                
                # Reconstruir líneas basándonos en la posición vertical (top)
                lines = {}
                for w in words:
                    top_val = w.get('top')
                    if top_val is None: continue
                    top = round(float(top_val), 1)
                    if top not in lines:
                        lines[top] = []
                    lines[top].append(w.get('text', ''))
                
                # Unir palabras en líneas y líneas en página
                page_text = "\n".join([" ".join(lines[t]) for t in sorted(lines.keys())])
                
                # v15.6: Fallback si la reconstruccion por layout falla (ej. PDF extraño)
                if len(page_text.strip()) < 100:
                    page_text = p.extract_text() or ""
                
                # v14.0: Agregar tablas si existen para evitar "sopa de números"
                tables = p.extract_tables()
                table_text = ""
                for table in tables:
                    if table:
                        # Convertir tabla a formato legible (Markdown-ish)
                        rows = []
                        for row in table:
                            clean_row = [str(cell).strip().replace("\n", " ") for cell in row if cell]
                            if clean_row:
                                rows.append(" | ".join(clean_row))
                        if rows:
                            table_text += "\n[TABLA DETECTADA]\n" + "\n".join(rows) + "\n[FIN TABLA]\n"
                
                text_content += page_text + "\n" + table_text + "\n\n"
            
        clean_text = text_content[:MAX_CHARS].strip()
        
        # v15.4: Guardar en cache si la extracción fue exitosa
        if len(clean_text) > 1000 and pdf_url:
            _write_cache(pdf_url, clean_text)
            
        return clean_text if clean_text else ""
        
    except Exception as e:
        import traceback
        logging.error(f"❌ Error crítico en extracción/pdfplumber: {str(e)}\n{traceback.format_exc()}")
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

def download_full_text_lazy(article: Dict, force: bool = False) -> Dict:
    """
    FASE 2: Descarga Real (On Demand).
    Se llama desde main.py -> generate_column si hace falta.
    v10.7: Parámetro force agregado.
    """
    text_len = len(str(article.get('full_text', '')))
    if article.get('is_pdf_downloaded') and not force and text_len > 2000:
        return article
        
    url = article.get('pdf_url')
    if not url: return article
    
    logging.info(f" 📥 Descargando PDF: {url[:60]}...")
    full_text = extract_text_from_pdf_url(url)
    
    if len(full_text) > 1000:
        logging.info(f" ✅ PDF procesado ({len(full_text)} chars): {article.get('title')[:40]}...")
        article['full_text'] = full_text
        article['full_text_source'] = 'pdf_download'
        article['is_pdf_downloaded'] = True
        article['needs_pdf_download'] = False
    else:
        logging.warning(f" ❌ Falló descarga o extracción: {article.get('title')[:40]}...")
        # Si falla, marcamos que ya lo intentamos para no reintentar infinitamente
        article['is_pdf_downloaded'] = False 
        article['needs_pdf_download'] = False 
        
    return article