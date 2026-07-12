"""
pdf_fetcher_v2.py
=================
Extractor de PDFs para RSL con cascada de 5 niveles y gestión de cola manual.

La solución al problema "SIN PDF"
-----------------------------------
El código original fallaba silenciosamente y retornaba "" — el investigador
nunca sabía cuántos papers quedaban sin texto hasta llegar a la síntesis.

Este módulo hace lo contrario: agota 5 fuentes automáticas, y solo cuando
todas fallan, escribe el paper en pending_manual.csv con toda la información
necesaria para que el investigador resuelva el lote de una vez, no uno por uno.

Niveles de adquisición
-----------------------
  1. Descarga directa (pdf_url original, con headers realistas + retry)
  2. Unpaywall API → solo repositorios OA confiables (filtra portales paywalled)
  2.5 Sci-Hub (sci-hub.box) → resolución por DOI con alta cobertura (>90%)
  3. Semantic Scholar API (openAccessPdf) — cobertura excelente en CS/ML
  4. Abstract como proxy semántico (degradado pero válido para screening)
  5. Cola manual: escribe pending_manual.csv + procesa inbox manual

Estado persistente (acquisition_state.json)
--------------------------------------------
  'pdf_direct'      — descargado desde pdf_url original
  'pdf_unpaywall'   — obtenido via Unpaywall
  'pdf_scihub'      — obtenido via Sci-Hub
  'pdf_s2'          — obtenido via Semantic Scholar
  'abstract_proxy'  — solo abstract (suficiente para screening inicial)
  'manual_pending'  — en cola para upload manual
  'manual_done'     — upload manual completado
  'failed'          — sin DOI ni abstract recuperable

Workflow manual (para el investigador)
---------------------------------------
  1. Al final del pipeline:  python pdf_fetcher_v2.py --report
  2. Descargar los PDFs faltantes (UI PRISMA, Sci-Hub, biblioteca, etc.)
     Depositarlos en manual_pdfs/ con nombre sugerido del CSV.
  3. Ejecutar:  python pdf_fetcher_v2.py --process-manual
     → procesa el lote completo, actualiza estado, extrae texto.

Dependencias: pip install requests pdfplumber
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import multiprocessing
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

import requests
import pypdfium2 as pdfium

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------

MAX_PDF_SIZE  = 50 * 1024 * 1024
TIMEOUT_SHORT = 12
TIMEOUT_LONG  = 35
STAGE2_FAST_TIMEOUT_SHORT = 5
STAGE2_FAST_TIMEOUT_LONG = 8
STAGE2_SOURCE_WORKERS = 4
MAX_PAGES     = 400
MAX_CHARS     = 800_000
PDF_EXTRACTION_TIMEOUT = int(os.getenv("PDF_EXTRACTION_TIMEOUT", "8"))

BASE_DIR      = Path(__file__).parent
PDF_CACHE_DIR = BASE_DIR / "pdf_text_cache"
STATE_FILE    = BASE_DIR / "acquisition_state.json"
MANUAL_CSV    = BASE_DIR / "pending_manual.csv"
MANUAL_INBOX  = BASE_DIR / "manual_pdfs"

for _d in [PDF_CACHE_DIR, MANUAL_INBOX]:
    _d.mkdir(exist_ok=True)

UNPAYWALL_EMAIL = "jnacarinoa1@upao.edu.pe"

DOMAIN_DELAYS: dict[str, float] = {
    'arxiv.org': 1.5,
    'aclanthology.org': 1.0,
    'link.springer.com': 1.2,
    'kar.kent.ac.uk': 0.8,
    'escholarship.org': 0.8,
    'api.semanticscholar.org': 0.6,
    'api.unpaywall.org': 0.3,
    'sci-hub.box': 1.0,
}

# Repositorios genuinamente abiertos — se descargan sin restricciones
TRUSTED_OA = [
    'arxiv.org', 'europepmc.org', 'pubmedcentral', 'pmc.ncbi',
    'biorxiv.org', 'medrxiv.org', 'zenodo.org', 'hal.science',
    'hal.archives-ouvertes', 'osf.io', 'aclanthology.org',
    'openreview.net', 'eprints.', 'dspace.', 'repository.',
    '.ac.uk', 'escholarship.org', 'engrxiv.org', 'preprints.org',
    'semanticscholar.org', 'figshare.com', 'ssrn.com', 'dergipark',
]

# Portales que exigen autenticación aunque Unpaywall los marque como OA
PAYWALLED = [
    'dl.acm.org', 'ieeexplore.ieee.org', 'link.springer.com',
    'www.mdpi.com', 'sciencedirect.com', 'wiley.com', 'tandfonline',
    'nature.com/articles', 'doi.org', 'downloads.hindawi.com',
    'igi-global.com', 'iopscience.iop.org',
]

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36',
]

# ---------------------------------------------------------------------------
# LOCKS POR DOMINIO
# ---------------------------------------------------------------------------

_DOMAIN_LOCKS: dict[str, threading.Lock] = {}
_LOCK_MAP = threading.Lock()
_STATE_LOCK = threading.Lock()

def _get_lock(domain: str) -> threading.Lock:
    with _LOCK_MAP:
        if domain not in _DOMAIN_LOCKS:
            _DOMAIN_LOCKS[domain] = threading.Lock()
        return _DOMAIN_LOCKS[domain]

# ---------------------------------------------------------------------------
# ESTADO PERSISTENTE
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    with _STATE_LOCK:
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text(encoding='utf-8'))
            except Exception:
                pass
    return {}

def _save_state(state: dict) -> None:
    with _STATE_LOCK:
        current = {}
        if STATE_FILE.exists():
            try:
                current = json.loads(STATE_FILE.read_text(encoding='utf-8'))
            except Exception:
                current = {}
        current.update(state)
        STATE_FILE.write_text(
            json.dumps(current, ensure_ascii=False, indent=2), encoding='utf-8'
        )

def _article_key(article: Dict) -> str:
    """Clave única estable: DOI > url > titulo."""
    doi = article.get('doi') or _extract_doi(article.get('pdf_url', ''))
    if doi:
        return f"doi:{doi.lower()}"
    url = article.get('pdf_url', '')
    if url:
        return f"url:{hashlib.md5(url.encode()).hexdigest()[:12]}"
    title = str(article.get('title', '')).lower().strip()[:80]
    return f"title:{hashlib.md5(title.encode()).hexdigest()[:12]}"

# ---------------------------------------------------------------------------
# CACHÉ DE TEXTO EXTRAÍDO
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    return PDF_CACHE_DIR / f"{hashlib.md5(key.encode()).hexdigest()}.txt"

def _read_cache(key: str) -> Optional[str]:
    p = _cache_path(key)
    if p.exists():
        try:
            t = p.read_text(encoding='utf-8')
            return t if len(t) > 500 else None
        except Exception:
            pass
    return None

def _write_cache(key: str, text: str) -> None:
    if len(text) >= 500:
        try:
            _cache_path(key).write_text(text[:MAX_CHARS], encoding='utf-8')
        except Exception:
            pass

# ---------------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------------

def _extract_doi(url: str) -> Optional[str]:
    if not url:
        return None
    m = re.search(r'10\.\d{4,9}/[^\s&?#]+', url)
    return m.group(0).rstrip('/.,') if m else None

def _rewrite_ieee_url(url: str) -> str:
    """Extrae el arnumber de una URL de IEEE y la reescribe al endpoint directo del PDF."""
    if 'ieee.org' in url or 'xplorestaging.ieee.org' in url:
        m = re.search(r'arnumber=(\d+)', url) or re.search(r'/document/(\d+)', url)
        if m:
            arnumber = m.group(1)
            return f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={arnumber}&ref="
    return url

def _ieee_pdf_candidates(url: str) -> list[str]:
    if 'ieeexplore.ieee.org' not in url and 'xplorestaging.ieee.org' not in url:
        return [url]

    match = re.search(r'arnumber=(\d+)', url) or re.search(r'/document/(\d+)', url)
    if not match:
        return [url]

    arnumber = match.group(1)
    candidates = [
        url,
        f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={arnumber}&ref=",
        f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={arnumber}",
        f"https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber={arnumber}",
    ]
    return list(dict.fromkeys(candidates))

def _is_trusted(url: str) -> bool:
    return any(t in url for t in TRUSTED_OA)

def _is_paywalled(url: str) -> bool:
    # Removed 'ieeexplore.ieee.org' since some papers are Open Access and can be downloaded directly
    paywalled_domains = [
        'dl.acm.org', 'link.springer.com',
        'www.mdpi.com', 'sciencedirect.com', 'wiley.com', 'tandfonline',
        'nature.com/articles', 'doi.org', 'downloads.hindawi.com',
        'igi-global.com', 'iopscience.iop.org',
    ]
    return any(p in url for p in paywalled_domains)

def _domain_wait(domain: str) -> None:
    for d, delay in DOMAIN_DELAYS.items():
        if d in domain:
            time.sleep(delay)
            break

def _is_pdf(content: bytes, content_type: str = '') -> bool:
    """Verifica que el contenido sea un PDF real, no una landing page HTML."""
    if 'text/html' in content_type.lower():
        return False
    if len(content) < 5000:
        return False
    return content[:5] == b'%PDF-'

# ---------------------------------------------------------------------------
# EXTRACCIÓN DE TEXTO (pypdfium2 hiper-rápido en C++)
# ---------------------------------------------------------------------------

def _extract_text(pdf_bytes: bytes) -> str:
    """
    Extrae texto de forma extremadamente rápida sin bloquear el CPU (GIL).
    Reemplaza a pdfplumber para optimizar extracción de 500+ papers.
    """
    if not pdf_bytes or not pdf_bytes.startswith(b'%PDF-'):
        logger.warning("pypdfium2 omitido: el binario no tiene cabecera PDF valida.")
        return ""

    parts = []
    page_errors = 0
    doc = None
    try:
        doc = pdfium.PdfDocument(pdf_bytes)
        num_pages = min(len(doc), MAX_PAGES)
        for i in range(num_pages):
            try:
                page = doc[i]
                text_page = page.get_textpage()
                text = text_page.get_text_bounded()
                if text:
                    parts.append(text)
            except Exception as exc:
                page_errors += 1
                logger.debug("pypdfium2 omitio pagina %d: %s", i + 1, exc)

        text = "\n\n".join(parts)[:MAX_CHARS].strip()
        if page_errors:
            logger.warning(
                "pypdfium2 omitio %d/%d paginas; texto extraido=%d chars.",
                page_errors,
                num_pages,
                len(text),
            )
        return text
    except Exception as e:
        logger.warning("pypdfium2 no pudo abrir o extraer el PDF: %s", e)
        return ""
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


def _extract_selective_sections_worker(pdf_bytes: bytes, min_len: int, queue: multiprocessing.Queue) -> None:
    try:
        queue.put(extract_selective_sections(pdf_bytes, min_len=min_len))
    except Exception as exc:
        logger.debug("Worker de extraccion PDF fallo: %s", exc)
        queue.put("")


def _extract_text_worker(pdf_bytes: bytes, queue: multiprocessing.Queue) -> None:
    try:
        queue.put(_extract_text(pdf_bytes))
    except Exception as exc:
        logger.debug("Worker de texto PDF fallo: %s", exc)
        queue.put("")


def _run_pdf_worker_process(
    pdf_bytes: bytes,
    mode: str,
    timeout_seconds: int,
    min_len: int = 3000,
) -> str:
    if not pdf_bytes or not pdf_bytes.startswith(b"%PDF-"):
        logger.warning("Worker PDF omitido: el binario no tiene cabecera PDF valida.")
        return ""

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as handle:
            handle.write(pdf_bytes)
            temp_path = handle.name

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "utils.pdf_text_worker",
                temp_path,
                mode,
                str(min_len),
            ],
            cwd=str(BASE_DIR.parent.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            logger.warning(
                "Worker PDF fallo (%s): %s",
                mode,
                (completed.stderr or completed.stdout or "")[:180],
            )
            return ""
        return str(completed.stdout or "").strip()
    except subprocess.TimeoutExpired:
        logger.warning("Worker PDF timeout tras %ss en modo %s; se usara fallback.", timeout_seconds, mode)
        return ""
    except Exception as exc:
        logger.warning("Worker PDF error en modo %s: %s", mode, exc)
        return ""
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


def extract_text_with_timeout(
    pdf_bytes: bytes,
    timeout_seconds: int = PDF_EXTRACTION_TIMEOUT,
) -> str:
    if timeout_seconds <= 0:
        return _extract_text(pdf_bytes)
    if os.name == "nt":
        return _run_pdf_worker_process(pdf_bytes, "full", timeout_seconds)

    context = multiprocessing.get_context("spawn" if os.name == "nt" else "fork")
    queue = context.Queue(maxsize=1)
    process = context.Process(target=_extract_text_worker, args=(pdf_bytes, queue))
    process.daemon = True
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(2)
        logger.warning("pypdfium2 full-text timeout tras %ss; se usara fallback.", timeout_seconds)
        return ""

    try:
        return str(queue.get_nowait() or "")
    except Exception:
        return ""


def extract_selective_sections_with_timeout(
    pdf_bytes: bytes,
    min_len: int = 3000,
    timeout_seconds: int = PDF_EXTRACTION_TIMEOUT,
) -> str:
    if timeout_seconds <= 0:
        return extract_selective_sections(pdf_bytes, min_len=min_len)
    if os.name == "nt":
        return _run_pdf_worker_process(pdf_bytes, "selective", timeout_seconds, min_len=min_len)

    context = multiprocessing.get_context("spawn" if os.name == "nt" else "fork")
    queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_extract_selective_sections_worker,
        args=(pdf_bytes, min_len, queue),
    )
    process.daemon = True
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(2)
        logger.warning("pypdfium2 timeout tras %ss; se usara abstract como respaldo.", timeout_seconds)
        return ""

    try:
        return str(queue.get_nowait() or "")
    except Exception:
        return ""


def extract_selective_sections(pdf_bytes: bytes, min_len: int = 3000) -> str:
    """
    Extracción quirúrgica de PDFs académicos para screening.
    Extrae únicamente las secciones clave (Introducción, Metodología, Conclusiones)
    y elimina/recorta la sección de Referencias para optimizar tokens y evitar ruido.
    """
    full_text = _extract_text(pdf_bytes)
    if not full_text:
        return ""
    if len(full_text) < min_len:
        return full_text.strip()
    
    # 1. Eliminar referencias bibliográficas (el 30-40% final del paper)
    # Buscamos patrones comunes de encabezado de referencias en líneas completas
    ref_patterns = [
        r'\n\s*(?:[IVX\d]+\.\s*)?(?:References|Bibliography|Literature Cited|Referencias|Bibliografía)\s*$',
        r'\n\s*(?:[IVX\d]+\.\s*)?(?:References|Bibliography|Literature Cited|Referencias|Bibliografía)\s*\n'
    ]
    
    ref_idx = len(full_text)
    for pat in ref_patterns:
        matches = list(re.finditer(pat, full_text, re.IGNORECASE))
        if matches:
            # Tomamos la última coincidencia razonable (por si acaso el abstract menciona "references")
            for m in reversed(matches):
                # Si está en el último 50% del documento, es muy probable que sea la sección final de referencias
                if m.start() > len(full_text) * 0.5:
                    ref_idx = min(ref_idx, m.start())
                    break
    
    # Recortar hasta la bibliografía
    text_no_refs = full_text[:ref_idx].strip()
    
    # 2. Heurística para extraer secciones: Introduction, Methods, Conclusions
    intro_patterns = [r'\n\s*(?:[IVX\d]+\.\s*)?(?:Introduction|Introducción)\b']
    methods_patterns = [
        r'\n\s*(?:[IVX\d]+\.\s*)?(?:Methods|Methodology|Proposed Method|Proposed Methodology|Experimental Design|System Model|Materials\s+and\s+Methods|Método|Metodología)\b'
    ]
    concl_patterns = [
        r'\n\s*(?:[IVX\d]+\.\s*)?(?:Conclusion|Conclusions|Concluding Remarks|Discussion\s+and\s+Conclusion|Conclusions\s+and\s+Future\s+Work|Conclusiones)\b'
    ]
    
    def find_first_match(patterns, text) -> int:
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.start()
        return -1
        
    intro_pos = find_first_match(intro_patterns, text_no_refs)
    methods_pos = find_first_match(methods_patterns, text_no_refs)
    concl_pos = find_first_match(concl_patterns, text_no_refs)
    
    sections = []
    
    # Introducción
    if intro_pos != -1:
        intro_end = methods_pos if methods_pos > intro_pos else (intro_pos + 8000)
        intro_text = text_no_refs[intro_pos:intro_end].strip()
        sections.append("=== INTRODUCTION ===\n" + intro_text)
    else:
        sections.append("=== INTRODUCTION (FALLBACK) ===\n" + text_no_refs[:6000])
        
    # Metodología
    if methods_pos != -1:
        methods_end = concl_pos if concl_pos > methods_pos else (methods_pos + 12000)
        methods_text = text_no_refs[methods_pos:methods_end].strip()
        sections.append("=== METHODOLOGY ===\n" + methods_text)
        
    # Conclusiones
    if concl_pos != -1:
        concl_text = text_no_refs[concl_pos:].strip()
        sections.append("=== CONCLUSIONS ===\n" + concl_text)
    elif len(text_no_refs) > 12000:
        sections.append("=== CONCLUSIONS (FALLBACK) ===\n" + text_no_refs[-4000:])
        
    final_extracted = "\n\n".join(sections)
    return final_extracted[:30000].strip() # Límite superior de seguridad (~6000-8000 palabras)


def extract_selective_sections_from_text(full_text: str, min_len: int = 3000) -> str:
    """
    Extracción quirúrgica directamente desde texto completo ya extraído/recuperado.
    Elimina referencias bibliográficas y selecciona secciones clave.
    """
    if not full_text:
        return ""
    if len(full_text) < min_len:
        return full_text.strip()
    
    # 1. Eliminar referencias bibliográficas (el 30-40% final del paper)
    ref_patterns = [
        r'\n\s*(?:[IVX\d]+\.\s*)?(?:References|Bibliography|Literature Cited|Referencias|Bibliografía)\s*$',
        r'\n\s*(?:[IVX\d]+\.\s*)?(?:References|Bibliography|Literature Cited|Referencias|Bibliografía)\s*\n'
    ]
    
    ref_idx = len(full_text)
    for pat in ref_patterns:
        matches = list(re.finditer(pat, full_text, re.IGNORECASE))
        if matches:
            for m in reversed(matches):
                if m.start() > len(full_text) * 0.5:
                    ref_idx = min(ref_idx, m.start())
                    break
    
    # Recortar hasta la bibliografía
    text_no_refs = full_text[:ref_idx].strip()
    
    # 2. Heurística para extraer secciones: Introduction, Methods, Conclusions
    intro_patterns = [r'\n\s*(?:[IVX\d]+\.\s*)?(?:Introduction|Introducción)\b']
    methods_patterns = [
        r'\n\s*(?:[IVX\d]+\.\s*)?(?:Methods|Methodology|Proposed Method|Proposed Methodology|Experimental Design|System Model|Materials\s+and\s+Methods|Método|Metodología)\b'
    ]
    concl_patterns = [
        r'\n\s*(?:[IVX\d]+\.\s*)?(?:Conclusion|Conclusions|Concluding Remarks|Discussion\s+and\s+Conclusion|Conclusions\s+and\s+Future\s+Work|Conclusiones)\b'
    ]
    
    def find_first_match(patterns, text) -> int:
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.start()
        return -1
        
    intro_pos = find_first_match(intro_patterns, text_no_refs)
    methods_pos = find_first_match(methods_patterns, text_no_refs)
    concl_pos = find_first_match(concl_patterns, text_no_refs)
    
    sections = []
    
    # Introducción
    if intro_pos != -1:
        intro_end = methods_pos if methods_pos > intro_pos else (intro_pos + 8000)
        intro_text = text_no_refs[intro_pos:intro_end].strip()
        sections.append("=== INTRODUCTION ===\n" + intro_text)
    else:
        sections.append("=== INTRODUCTION (FALLBACK) ===\n" + text_no_refs[:6000])
        
    # Metodología
    if methods_pos != -1:
        methods_end = concl_pos if concl_pos > methods_pos else (methods_pos + 12000)
        methods_text = text_no_refs[methods_pos:methods_end].strip()
        sections.append("=== METHODOLOGY ===\n" + methods_text)
        
    # Conclusiones
    if concl_pos != -1:
        concl_text = text_no_refs[concl_pos:].strip()
        sections.append("=== CONCLUSIONS ===\n" + concl_text)
    elif len(text_no_refs) > 12000:
        sections.append("=== CONCLUSIONS (FALLBACK) ===\n" + text_no_refs[-4000:])
        
    final_extracted = "\n\n".join(sections)
    return final_extracted[:30000].strip()



# ---------------------------------------------------------------------------
# NIVEL 1: DESCARGA DIRECTA
# ---------------------------------------------------------------------------

def _level1_direct(
    url: str,
    timeout_short: int = TIMEOUT_SHORT,
    timeout_long: int = TIMEOUT_LONG,
) -> Optional[bytes]:
    if not url:
        return None
    if "arxiv.org/abs/" in url:
        url = url.replace("/abs/", "/pdf/")

    candidate_urls = _ieee_pdf_candidates(url)
    parsed = urlparse(url)
    domain = parsed.netloc
    ua = random.choice(USER_AGENTS)
    headers = {
        'User-Agent': ua,
        'Accept': 'application/pdf,application/x-pdf,*/*',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    with _get_lock(domain):
        _domain_wait(domain)
        session = requests.Session()

        # Simular visita previa en portales que requieren Referer
        if any(x in domain for x in ["mdpi.com", "acm.org", "ieee.org", "sciencedirect.com", "springer.com"]):
            try:
                session.get(f"{parsed.scheme}://{domain}", timeout=timeout_short,
                            headers={'User-Agent': ua})
                headers['Referer'] = f"{parsed.scheme}://{domain}"
            except Exception:
                pass

        # Intento 1: con sesión
        for candidate_url in candidate_urls:
            try:
                logger.info(f"[L1] Intentando directo: {candidate_url[:90]}")
                r = session.get(candidate_url, timeout=timeout_long, headers=headers, allow_redirects=True)
                if r.status_code == 200 and _is_pdf(r.content, r.headers.get('Content-Type', '')):
                    logger.info(f"[L1] Directo OK: {len(r.content)}B")
                    return r.content
            except Exception as exc:
                logger.debug(f"[L1] Directo fallo: {exc}")

        # Intento 2: stateless (sin cookies acumuladas)
        for candidate_url in candidate_urls:
            try:
                r2 = requests.get(
                    candidate_url, timeout=timeout_long,
                    headers={'User-Agent': random.choice(USER_AGENTS), 'Accept': 'application/pdf,application/x-pdf,*/*'},
                    allow_redirects=True
                )
                if r2.status_code == 200 and _is_pdf(r2.content, r2.headers.get('Content-Type', '')):
                    logger.info(f"[L1] Stateless OK: {len(r2.content)}B")
                    return r2.content
            except Exception as exc:
                logger.debug(f"[L1] Stateless fallo: {exc}")

    return None

# ---------------------------------------------------------------------------
# NIVEL 2: UNPAYWALL
# ---------------------------------------------------------------------------

def _level2_unpaywall(
    doi: str,
    timeout_short: int = TIMEOUT_SHORT,
    timeout_long: int = TIMEOUT_LONG,
) -> Optional[bytes]:
    if not doi:
        return None
    try:
        resp = requests.get(
            f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}",
            timeout=timeout_short
        )
        if resp.status_code != 200:
            return None

        candidates: list[tuple[int, str]] = []
        for loc in resp.json().get('oa_locations', []):
            for candidate_url in [loc.get('url_for_pdf'), loc.get('url')]:
                if not candidate_url or _is_paywalled(candidate_url):
                    continue
                score = (10 if _is_trusted(candidate_url) else 0) + \
                        (5 if candidate_url == loc.get('url_for_pdf') else 0) + \
                        (3 if loc.get('host_type') == 'repository' else 0)
                candidates.append((score, candidate_url))

        for _, candidate_url in sorted(candidates, reverse=True):
            for pdf_candidate_url in _ieee_pdf_candidates(_rewrite_ieee_url(candidate_url)):
                try:
                    # IEEE requires Accept and sometimes Referer to allow PDF download
                    headers = {'User-Agent': random.choice(USER_AGENTS)}
                    if 'ieeexplore.ieee.org' in pdf_candidate_url:
                        headers['Accept'] = 'application/pdf,application/x-pdf,*/*'
                        m = re.search(r'arnumber=(\d+)', pdf_candidate_url)
                        if m:
                            headers['Referer'] = f"https://ieeexplore.ieee.org/document/{m.group(1)}"

                    r = requests.get(
                        pdf_candidate_url, timeout=timeout_long,
                        headers=headers,
                        allow_redirects=True
                    )
                    if r.status_code == 200 and _is_pdf(r.content, r.headers.get('Content-Type', '')):
                        logger.info(f"[L2] Unpaywall OK: {pdf_candidate_url[:60]}")
                        return r.content
                except Exception:
                    continue
    except Exception as e:
        logger.debug(f"[L2] Error: {e}")
    return None

# ---------------------------------------------------------------------------
# NIVEL 2.5: SCI-HUB (sci-hub.box)
# ---------------------------------------------------------------------------

def _level_scihub(
    doi: str,
    timeout_long: int = TIMEOUT_LONG,
) -> Optional[bytes]:
    """
    Sci-Hub resuelve PDFs por DOI con cobertura >90% para artículos publicados.
    Parsea la página HTML de sci-hub.box para extraer el URL del PDF embebido.
    """
    if not doi:
        return None

    SCIHUB_MIRRORS = ['sci-hub.box']

    for mirror in SCIHUB_MIRRORS:
        try:
            with _get_lock(mirror):
                _domain_wait(mirror)
                ua = random.choice(USER_AGENTS)
                resp = requests.get(
                    f"https://{mirror}/{doi}",
                    headers={'User-Agent': ua, 'Accept': 'text/html,application/xhtml+xml,*/*'},
                    timeout=timeout_long,
                    allow_redirects=True
                )

                if resp.status_code != 200:
                    continue

                # Si la respuesta directa ya es un PDF
                ct = resp.headers.get('Content-Type', '')
                if _is_pdf(resp.content, ct):
                    logger.info(f"[L2.5] Sci-Hub PDF directo: {doi}")
                    return resp.content

                # Parsear HTML para encontrar el URL del PDF embebido
                html = resp.text

                # sci-hub.box usa <object type="application/pdf" data="/storage/.../*.pdf#...">
                # Orden de prioridad basado en inspección real del DOM:
                pdf_url = None
                patterns = [
                    # Patrón principal: <object type="application/pdf" data="...">
                    r'<object[^>]+?type\s*=\s*["\']application/pdf["\'][^>]+?data\s*=\s*["\']([^"\'#>]+)',
                    r'<object[^>]+?data\s*=\s*["\']([^"\'#>]+\.pdf)[^"\']*["\']',
                    # Fallbacks para otros mirrors de Sci-Hub:
                    r'<iframe[^>]+?src\s*=\s*["\']([^"\'>]+\.pdf[^"\']*)["\']',
                    r'<iframe[^>]+?id\s*=\s*["\']pdf["\'][^>]+?src\s*=\s*["\']([^"\'>]+)["\']',
                    r'<embed[^>]+?src\s*=\s*["\']([^"\'>]+)["\'][^>]+?type\s*=\s*["\']application/pdf',
                    r'<embed[^>]+?src\s*=\s*["\']([^"\'>]+\.pdf[^"\']*)["\']',
                    # Download link pattern (botón de descarga)
                    r'class\s*=\s*["\']download["\'][^>]*>\s*<a[^>]+?href\s*=\s*["\']([^"\']+)["\']',
                    r'<a[^>]+?href\s*=\s*["\']([^"\']+\.pdf[^"\']*)["\'][^>]*>\s*(?:download|save|descargar)',
                ]

                for pat in patterns:
                    m = re.search(pat, html, re.IGNORECASE)
                    if m:
                        pdf_url = m.group(1).strip()
                        break

                if not pdf_url:
                    logger.debug(f"[L2.5] Sci-Hub: no PDF URL found in HTML for {doi}")
                    continue

                # Normalizar URL relativa
                if pdf_url.startswith('//'):
                    pdf_url = 'https:' + pdf_url
                elif pdf_url.startswith('/'):
                    pdf_url = f'https://{mirror}' + pdf_url

                # Descargar el PDF
                _domain_wait(mirror)
                r2 = requests.get(
                    pdf_url, timeout=timeout_long,
                    headers={'User-Agent': ua, 'Referer': f'https://{mirror}/{doi}'},
                    allow_redirects=True
                )
                if r2.status_code == 200 and _is_pdf(r2.content, r2.headers.get('Content-Type', '')):
                    logger.info(f"[L2.5] Sci-Hub OK: {len(r2.content)}B for {doi}")
                    return r2.content

        except Exception as e:
            logger.debug(f"[L2.5] Sci-Hub error ({mirror}): {e}")
            continue

    return None


# ---------------------------------------------------------------------------
# NIVEL 3: SEMANTIC SCHOLAR API
# ---------------------------------------------------------------------------

def _level3_semantic_scholar(
    doi: str,
    title: str,
    timeout_short: int = TIMEOUT_SHORT,
    timeout_long: int = TIMEOUT_LONG,
) -> Optional[bytes]:
    """
    Semantic Scholar ofrece openAccessPdf para una fracción muy alta de papers
    en CS, ML y medicina. No requiere API key para consultas básicas.
    """
    with _get_lock('api.semanticscholar.org'):
        _domain_wait('api.semanticscholar.org')
        headers = {'User-Agent': f'RSLAssistant/2.0 (mailto:{UNPAYWALL_EMAIL})'}
        paper = None

        if doi:
            try:
                r = requests.get(
                    f"https://api.semanticscholar.org/graph/v1/paper/{doi}",
                    params={'fields': 'openAccessPdf'},
                    headers=headers, timeout=timeout_short
                )
                if r.status_code == 200:
                    paper = r.json()
            except Exception:
                pass

        # Búsqueda por título si no se encontró por DOI
        if not paper and title:
            try:
                r = requests.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={'query': title[:150], 'fields': 'openAccessPdf', 'limit': 3},
                    headers=headers, timeout=timeout_short
                )
                if r.status_code == 200:
                    data = r.json().get('data', [])
                    if data:
                        paper = data[0]
            except Exception:
                pass

        if not paper:
            return None

        oa = paper.get('openAccessPdf')
        if not oa:
            return None
        pdf_url = oa.get('url', '')
        if not pdf_url or _is_paywalled(pdf_url):
            return None

        try:
            r = requests.get(
                pdf_url, timeout=timeout_long,
                headers={'User-Agent': random.choice(USER_AGENTS)},
                allow_redirects=True
            )
            if r.status_code == 200 and _is_pdf(r.content, r.headers.get('Content-Type', '')):
                logger.info(f"[L3] SemanticScholar OK: {pdf_url[:60]}")
                return r.content
        except Exception:
            pass

    return None

# ---------------------------------------------------------------------------
# NIVEL 3.5: CROSSREF TITLE SEARCH (Especial para IEEE)
# ---------------------------------------------------------------------------

def _level3_5_crossref_title(
    title: str,
    timeout_short: int = TIMEOUT_SHORT,
    timeout_long: int = TIMEOUT_LONG,
) -> Optional[bytes]:
    """
    Busca por título en Crossref y extrae el enlace de text-mining.
    Muy útil para artículos Open Access de IEEE que fallan en otros niveles.
    """
    if not title or len(title) < 10:
        return None
    
    with _get_lock('api.crossref.org'):
        _domain_wait('api.crossref.org')
        try:
            import urllib.parse
            url = f"https://api.crossref.org/works?query.title={urllib.parse.quote(title)}&select=link&rows=2"
            r = requests.get(url, timeout=timeout_short, headers={'mailto': UNPAYWALL_EMAIL, 'User-Agent': 'PrismaAssistant/1.0'})
            if r.status_code == 200:
                data = r.json()
                items = data.get('message', {}).get('items', [])
                for item in items:
                    links = item.get('link', [])
                    for link in links:
                        # Buscamos enlaces de similitud o text-mining que suelen ser PDFs directos
                        if link.get('intended-application') in ('text-mining', 'similarity-checking'):
                            pdf_url = link.get('URL')
                            if pdf_url and not _is_paywalled(pdf_url):
                                r2 = requests.get(
                                    pdf_url, timeout=timeout_long,
                                    headers={'User-Agent': random.choice(USER_AGENTS)},
                                    allow_redirects=True
                                )
                                if r2.status_code == 200 and _is_pdf(r2.content, r2.headers.get('Content-Type', '')):
                                    logger.info(f"[L3.5] Crossref OK: {pdf_url[:60]}")
                                    return r2.content
        except Exception as e:
            logger.debug(f"[L3.5] Crossref error para '{title[:30]}': {e}")
    return None

# ---------------------------------------------------------------------------
# COLA MANUAL: escritura y procesamiento
# ---------------------------------------------------------------------------

def _queue_manual(article: Dict, state: dict, key: str) -> None:
    """
    Registra el paper en pending_manual.csv con nombre de archivo sugerido.
    El investigador descarga el PDF, lo deposita en manual_pdfs/ y luego
    ejecuta --process-manual para procesar el lote completo.
    """
    doi = article.get('doi') or _extract_doi(article.get('pdf_url', ''))
    doi_safe = re.sub(r'[^\w.]', '_', doi or '')[:50]
    suggested = f"{doi_safe}.pdf" if doi_safe else f"manual_{key[-8:]}.pdf"

    state[key].update({
        'doi': doi,
        'pdf_url': article.get('pdf_url', ''),
        'source_url': article.get('url', '') or article.get('landing_url', ''),
        'year': str(article.get('year', '')),
        'authors': str(article.get('authors', ''))[:80],
        'suggested_filename': suggested,
        'ts': time.strftime('%Y-%m-%d %H:%M'),
    })

    # Evitar duplicados en el CSV
    existing: set[str] = set()
    if MANUAL_CSV.exists():
        try:
            with MANUAL_CSV.open(encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if row.get('doi'):
                        existing.add(row['doi'])
        except Exception:
            pass

    if doi and doi in existing:
        return

    write_header = not MANUAL_CSV.exists()
    with MANUAL_CSV.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'doi', 'title', 'year', 'authors',
            'pdf_url', 'source_url', 'suggested_filename'
        ])
        if write_header:
            writer.writeheader()
        writer.writerow({
            'doi': doi or '',
            'title': state[key].get('title', ''),
            'year': state[key].get('year', ''),
            'authors': state[key].get('authors', ''),
            'pdf_url': state[key].get('pdf_url', ''),
            'source_url': state[key].get('source_url', ''),
            'suggested_filename': suggested,
        })

    logger.warning(f"[MANUAL] En cola → {suggested}: {state[key].get('title', '')[:55]}")


def process_manual_inbox(state: Optional[dict] = None) -> dict:
    """
    Procesa todos los PDFs depositados en manual_pdfs/.

    El investigador nombra cada archivo como indica la columna
    'suggested_filename' de pending_manual.csv (contiene el DOI en el nombre).
    Esta función empareja por DOI en el nombre del archivo.

    Uso: python pdf_fetcher_v2.py --process-manual
    """
    if state is None:
        state = _load_state()

    results: dict = {'processed': 0, 'errors': []}
    pdf_files = list(MANUAL_INBOX.glob("*.pdf"))

    if not pdf_files:
        logger.info("[Manual] manual_pdfs/ está vacío.")
        return results

    # Índice DOI → clave de estado
    doi_index: dict[str, str] = {
        v['doi'].lower(): k
        for k, v in state.items()
        if v.get('status') in ('manual_pending', 'abstract_proxy')
        and v.get('doi')
    }

    for pdf_path in pdf_files:
        try:
            pdf_bytes = pdf_path.read_bytes()
            if not _is_pdf(pdf_bytes):
                results['errors'].append(f"{pdf_path.name}: no es PDF válido")
                continue

            text = _extract_text(pdf_bytes)
            if len(text) < 500:
                results['errors'].append(f"{pdf_path.name}: extracción vacía")
                continue

            # Emparejar por DOI embebido en el nombre del archivo
            matched_key = None
            stem = pdf_path.stem.lower()
            for doi_key, state_key in doi_index.items():
                doi_normalized = doi_key.replace('/', '_').replace('.', '_')
                if doi_normalized[:20] in stem or stem[:20] in doi_normalized:
                    matched_key = state_key
                    break

            if matched_key:
                cache_key = f"manual:{matched_key}"
                _write_cache(cache_key, text)
                state[matched_key]['status'] = 'manual_done'
                state[matched_key]['full_text_cache_key'] = cache_key
                state[matched_key]['full_text_source'] = 'manual_upload'
                title_preview = state[matched_key].get('title', '')[:50]
                logger.info(f"[Manual] ✓ Emparejado: {pdf_path.name} → {title_preview}")
                results['processed'] += 1
            else:
                # Sin emparejar: guardar con clave basada en nombre
                cache_key = f"manual_unmatched:{pdf_path.stem}"
                _write_cache(cache_key, text)
                results['errors'].append(
                    f"{pdf_path.name}: texto guardado sin emparejar (revisar DOI en nombre)"
                )

        except Exception as e:
            results['errors'].append(f"{pdf_path.name}: {e}")

    _save_state(state)
    return results


def print_report(state: Optional[dict] = None) -> None:
    """Reporte de estado de adquisición. Llamar antes de iniciar síntesis."""
    if state is None:
        state = _load_state()

    counts = Counter(v.get('status', 'unknown') for v in state.values())
    total = sum(counts.values())

    labels = {
        'pdf_direct':     ('PDF descargado (directo)',      '✓'),
        'pdf_unpaywall':  ('PDF via Unpaywall',             '✓'),
        'pdf_scihub':     ('PDF via Sci-Hub',               '✓'),
        'pdf_s2':         ('PDF via Semantic Scholar',      '✓'),
        'abstract_proxy': ('Abstract proxy (degradado)',    '~'),
        'manual_pending': ('Pendiente upload manual',       '!'),
        'manual_done':    ('Upload manual completado',      '✓'),
        'failed':         ('Sin fuente disponible',         '✗'),
    }

    print("\n" + "=" * 55)
    print("  REPORTE DE ADQUISICIÓN RSL")
    print("=" * 55)
    print(f"  Total artículos registrados: {total}")
    print("-" * 55)
    for status, (label, icon) in labels.items():
        n = counts.get(status, 0)
        if n > 0:
            pct = n / total * 100
            print(f"  {icon} {label:<38} {n:>3} ({pct:.0f}%)")

    pending = counts.get('manual_pending', 0)
    abstract_only = counts.get('abstract_proxy', 0)
    if pending > 0 or abstract_only > 0:
        print("-" * 55)
        if pending > 0:
            print(f"  → {pending} papers en pending_manual.csv")
            print(f"    Depositar PDFs en:  {MANUAL_INBOX}/")
            print(f"    Luego ejecutar:     python pdf_fetcher_v2.py --process-manual")
        if abstract_only > 0:
            print(f"  ~ {abstract_only} papers solo con abstract (síntesis metodológica limitada)")
    print("=" * 55 + "\n")

# ---------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL — REEMPLAZA download_full_text_lazy()
# ---------------------------------------------------------------------------

def _get_doi_from_crossref(title: str) -> Optional[str]:
    """Obtiene el DOI a partir del título utilizando la API pública de Crossref."""
    if not title or len(title) < 10:
        return None
    try:
        import urllib.parse
        url = f"https://api.crossref.org/works?query.title={urllib.parse.quote(title)}&select=DOI,title&rows=1"
        r = requests.get(url, timeout=TIMEOUT_SHORT, headers={'User-Agent': 'PrismaAssistant (mailto:' + UNPAYWALL_EMAIL + ')'})
        if r.status_code == 200:
            data = r.json()
            items = data.get('message', {}).get('items', [])
            if items:
                return items[0].get('DOI')
    except Exception as e:
        logger.debug(f"Error en Crossref para título '{title[:30]}': {e}")
    return None

def _race_stage2_sources(doi: str, title: str) -> tuple[Optional[bytes], Optional[str], Optional[str]]:
    """Lanza las fuentes remotas de Stage 2 en paralelo y retorna el primer PDF valido."""
    tasks: list[tuple[str, str, Callable[[], Optional[bytes]]]] = []
    if doi:
        tasks.extend([
            (
                "pdf_unpaywall",
                f"unpaywall:{doi}",
                lambda: _level2_unpaywall(
                    doi,
                    timeout_short=STAGE2_FAST_TIMEOUT_SHORT,
                    timeout_long=STAGE2_FAST_TIMEOUT_LONG,
                ),
            ),
            (
                "pdf_scihub",
                f"scihub:{doi}",
                lambda: _level_scihub(doi, timeout_long=STAGE2_FAST_TIMEOUT_LONG),
            ),
            (
                "pdf_s2",
                f"s2:{doi}",
                lambda: _level3_semantic_scholar(
                    doi,
                    title,
                    timeout_short=STAGE2_FAST_TIMEOUT_SHORT,
                    timeout_long=STAGE2_FAST_TIMEOUT_LONG,
                ),
            ),
        ])
    if title:
        tasks.append((
            "pdf_crossref",
            f"crossref:{title[:40]}",
            lambda: _level3_5_crossref_title(
                title,
                timeout_short=STAGE2_FAST_TIMEOUT_SHORT,
                timeout_long=STAGE2_FAST_TIMEOUT_LONG,
            ),
        ))

    if not tasks:
        return None, None, None

    executor = ThreadPoolExecutor(max_workers=min(STAGE2_SOURCE_WORKERS, len(tasks)))
    futures: dict[Future, tuple[str, str]] = {
        executor.submit(fn): (source, cache_key)
        for source, cache_key, fn in tasks
    }

    try:
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                source, cache_key = futures[future]
                try:
                    pdf_bytes = future.result()
                except Exception as exc:
                    logger.debug(f"[Stage2] Fuente {source} fallo: {exc}")
                    continue
                if pdf_bytes and _is_pdf(pdf_bytes):
                    for pending_future in pending:
                        pending_future.cancel()
                    return pdf_bytes, source, cache_key
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return None, None, None

def acquire_full_text(article: Dict, force: bool = False) -> Dict:
    """
    Reemplaza a download_full_text_lazy(). Cascada completa de 5 niveles.

    El artículo retornado siempre tiene:
      full_text            : texto extraído (nunca silenciosamente vacío)
      full_text_source     : origen del texto
      acquisition_status   : estado de adquisición (para reporte)
      is_pdf_downloaded    : True si se obtuvo PDF real
    """
    state = _load_state()
    key = _article_key(article)
    existing = state.get(key, {})

    # Si ya está resuelto con PDF real y hay texto, retornar desde caché
    done_statuses = {'pdf_direct', 'pdf_unpaywall', 'pdf_scihub', 'pdf_s2', 'pdf_crossref', 'manual_done'}
    if not force and existing.get('status') in done_statuses:
        cache_key = existing.get('full_text_cache_key', key)
        cached = _read_cache(cache_key)
        if cached:
            article['full_text'] = cached
            article['full_text_source'] = existing['status']
            article['acquisition_status'] = existing['status']
            article['is_pdf_downloaded'] = True
            return article
    if not force and existing.get('status') in {'abstract_proxy', 'manual_pending', 'failed'}:
        abstract = str(article.get('abstract', '') or '')
        article['full_text'] = abstract
        article['full_text_source'] = existing['status']
        article['acquisition_status'] = existing['status']
        article['is_pdf_downloaded'] = False
        if abstract:
            logger.info(f"[Cache] Stage 2 sin PDF previo: {existing['status']} | {article.get('title', '')[:50]}")
            return article
        if existing.get('status') in {'manual_pending', 'failed'}:
            return article

    doi   = article.get('doi') or _extract_doi(article.get('pdf_url', ''))
    title = str(article.get('title', ''))
    
    if not doi and title:
        doi = _get_doi_from_crossref(title)
        if doi:
            logger.info(f"[Crossref] DOI resuelto por título: {doi}")
            article['doi'] = doi

    url   = article.get('pdf_url', '')

    # Inicializar entrada de estado
    state[key] = {'status': 'in_progress', 'title': title[:80]}

    def _resolve(pdf_bytes: bytes, source: str, cache_key: str) -> Optional[Dict]:
        text = extract_text_with_timeout(pdf_bytes)
        if len(text) < 500:
            return None
        _write_cache(cache_key, text)
        article['full_text'] = text
        article['full_text_source'] = source
        article['acquisition_status'] = source
        article['is_pdf_downloaded'] = True
        state[key] = {'status': source, 'title': title[:80],
                      'full_text_cache_key': cache_key}
        _save_state({key: state[key]})
        logger.info(f"[{source.upper()}] ✓ {len(text)} chars: {title[:50]}")
        return article

    # ── Nivel 1: Descarga directa ─────────────────────────────────────────
    if url:
        cached = _read_cache(url)
        if cached and not force:
            logger.info(f"[Cache] Hit: {url[:50]}")
            state[key] = {'status': 'pdf_direct', 'title': title[:80],
                          'full_text_cache_key': url}
            _save_state({key: state[key]})
            article.update({'full_text': cached, 'full_text_source': 'pdf_direct',
                            'acquisition_status': 'pdf_direct', 'is_pdf_downloaded': True})
            return article

        logger.info(f"[L1] Descarga directa: {url[:60]}")
        b = _level1_direct(
            url,
            timeout_short=STAGE2_FAST_TIMEOUT_SHORT,
            timeout_long=STAGE2_FAST_TIMEOUT_LONG,
        )
        if b:
            result = _resolve(b, 'pdf_direct', url)
            if result:
                return result

    skip_remote_sources = False
    if doi or title:
        logger.info(f"[Stage2] Carrera paralela de fuentes para: {doi or title[:50]}")
        b, source, cache_key = _race_stage2_sources(doi, title)
        if b and source and cache_key:
            result = _resolve(b, source, cache_key)
            if result:
                return result
        skip_remote_sources = True

    # ── Nivel 2: Unpaywall ────────────────────────────────────────────────
    if not skip_remote_sources and doi:
        logger.info(f"[L2] Unpaywall DOI: {doi}")
        b = _level2_unpaywall(doi)
        if b:
            result = _resolve(b, 'pdf_unpaywall', f"unpaywall:{doi}")
            if result:
                return result

    # ── Nivel 2.5: Sci-Hub ────────────────────────────────────────────────
    if not skip_remote_sources and doi:
        logger.info(f"[L2.5] Sci-Hub DOI: {doi}")
        b = _level_scihub(doi)
        if b:
            result = _resolve(b, 'pdf_scihub', f"scihub:{doi}")
            if result:
                return result

    # ── Nivel 3: Semantic Scholar ─────────────────────────────────────────
    logger.info(f"[L3] Semantic Scholar: {title[:50]}")
    b = None if skip_remote_sources else _level3_semantic_scholar(doi, title)
    if b:
        result = _resolve(b, 'pdf_s2', f"s2:{doi or title[:40]}")
        if result:
            return result

    # ── Nivel 3.5: Crossref Title Search ──────────────────────────────────
    logger.info(f"[L3.5] Crossref Title: {title[:50]}")
    b = None if skip_remote_sources else _level3_5_crossref_title(title)
    if b:
        result = _resolve(b, 'pdf_crossref', f"crossref:{title[:40]}")
        if result:
            return result

    # ── Nivel 4: Abstract proxy (válido para screening, limitado para síntesis)
    abstract = str(article.get('abstract', '') or '')
    if len(abstract) > 200:
        logger.warning(f"[L4] Abstract proxy ({len(abstract)}c): {title[:50]}")
        article['full_text'] = abstract
        article['full_text_source'] = 'abstract_proxy'
        article['acquisition_status'] = 'abstract_proxy'
        article['is_pdf_downloaded'] = False
        state[key] = {'status': 'abstract_proxy', 'title': title[:80]}
        # También va a cola manual para intentar obtener PDF completo después
        _queue_manual(article, state, key)
        _save_state({key: state[key]})
        return article

    # ── Nivel 5: Cola manual ──────────────────────────────────────────────
    logger.error(f"[L5] Todos los niveles fallaron: {title[:60]}")
    article['full_text'] = ''
    article['full_text_source'] = 'manual_pending'
    article['acquisition_status'] = 'manual_pending'
    article['is_pdf_downloaded'] = False
    state[key] = {'status': 'manual_pending', 'title': title[:80]}
    _queue_manual(article, state, key)
    _save_state({key: state[key]})
    return article


def acquire_unpaywall_text(
    article: Dict,
    timeout_short: int = STAGE2_FAST_TIMEOUT_SHORT,
    timeout_long: int = STAGE2_FAST_TIMEOUT_LONG,
) -> Dict:
    """Obtiene texto usando solo Unpaywall; si falla, degrada a abstract."""
    result = article.copy()
    doi = result.get('doi') or _extract_doi(result.get('pdf_url', ''))
    title = str(result.get('title', '') or '')

    if doi:
        logger.info(f"[GoldEval] Unpaywall only: {doi}")
        pdf_bytes = _level2_unpaywall(
            str(doi),
            timeout_short=timeout_short,
            timeout_long=timeout_long,
        )
        if pdf_bytes and _is_pdf(pdf_bytes):
            text = extract_text_with_timeout(pdf_bytes)
            if len(text) >= 500:
                result['full_text'] = extract_selective_sections_from_text(text)
                result['full_text_source'] = 'unpaywall_pdf_sections'
                result['acquisition_status'] = 'pdf_unpaywall'
                result['is_pdf_downloaded'] = True
                logger.info(f"[GoldEval] Unpaywall texto OK ({len(result['full_text'])}c): {title[:50]}")
                return result

    abstract = str(result.get('abstract', '') or '')
    result['full_text'] = abstract
    result['full_text_source'] = 'abstract_proxy'
    result['acquisition_status'] = 'abstract_proxy'
    result['is_pdf_downloaded'] = False
    logger.info(f"[GoldEval] Unpaywall fallback abstract ({len(abstract)}c): {title[:50]}")
    return result


# ---------------------------------------------------------------------------
# Compatibilidad con código existente
# ---------------------------------------------------------------------------

def enrich_initial_search_result(article: Dict) -> Dict:
    """FASE 1: Solo metadata, sin descarga."""
    abstract = article.get('abstract', '') or ''
    article['full_text'] = abstract
    article['full_text_source'] = 'abstract_proxy' if len(abstract) > 800 else 'abstract_short'
    article['is_pdf_downloaded'] = False
    article['needs_pdf_download'] = bool(
        article.get('pdf_url') and len(str(article.get('pdf_url', ''))) > 10
    )
    return article


def download_full_text_lazy(article: Dict, force: bool = False) -> Dict:
    """Wrapper de compatibilidad. Usar acquire_full_text() directamente."""
    return acquire_full_text(article, force=force)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    if "--report" in sys.argv:
        print_report()

    elif "--process-manual" in sys.argv:
        res = process_manual_inbox()
        print(f"\nProcesados: {res['processed']}")
        if res['errors']:
            print("Errores:")
            for e in res['errors']:
                print(f"  - {e}")

    elif "--test" in sys.argv:
        test = {
            'title': 'Attention Is All You Need',
            'doi': '10.48550/arXiv.1706.03762',
            'pdf_url': 'https://arxiv.org/pdf/1706.03762',
            'abstract': 'The dominant sequence transduction models...',
        }
        result = acquire_full_text(test)
        print(f"\nFuente : {result['full_text_source']}")
        print(f"Chars  : {len(result.get('full_text', ''))}")
        print(f"Preview: {result.get('full_text', '')[:200]}")

    else:
        print("Uso:")
        print("  python pdf_fetcher_v2.py --report           # estado de adquisición")
        print("  python pdf_fetcher_v2.py --process-manual   # procesar PDFs del inbox")
        print("  python pdf_fetcher_v2.py --test             # test con ArXiv")
