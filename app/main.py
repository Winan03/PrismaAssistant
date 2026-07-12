import os
import sys
import re
import io
import pandas as pd
import asyncio
import contextvars
import ssl
from concurrent.futures import ThreadPoolExecutor
import threading
import config

# Fix: Forzar stdout/stderr a UTF-8 para evitar mojibake en logs con emojis
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Fix for PubMed SSL: CERTIFICATE_VERIFY_FAILED
ssl._create_default_https_context = ssl._create_unverified_context

# ============================================================
# CACHE DE MODELOS: Windows → D:/AI_MODELS_CACHE | Linux/VPS → /app/.cache/models
# ============================================================
if os.name == 'nt':  # Windows
    CACHE_ROOT = "D:/AI_MODELS_CACHE"
else:  # Linux (VPS Docker)
    CACHE_ROOT = os.getenv("CACHE_ROOT", "/app/.cache/models")

os.makedirs(f"{CACHE_ROOT}/huggingface", exist_ok=True)
os.makedirs(f"{CACHE_ROOT}/sentence_transformers", exist_ok=True)
os.makedirs(f"{CACHE_ROOT}/llama_index", exist_ok=True)

os.environ['HF_HOME'] = f"{CACHE_ROOT}/huggingface"
os.environ['SENTENCE_TRANSFORMERS_HOME'] = f"{CACHE_ROOT}/sentence_transformers"
os.environ['LLAMA_INDEX_CACHE_DIR'] = f"{CACHE_ROOT}/llama_index"
os.environ['TORCH_HOME'] = f"{CACHE_ROOT}/torch"
# ============================================================

from datetime import datetime
import time
import logging
from pathlib import Path
from dataclasses import asdict
from typing import Any, Optional, List, Dict
from collections import Counter

MAX_RESULTS = 20000  # Meta de artículos para búsqueda masiva

from fastapi import FastAPI, Form, Request, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.screening import filters, deduplication, screening, metrics
from app.core import search_engine, database
from app.core.report_generator import create_pdf_report
from app.llm import synthesis, screening_ai
from app.llm.ai_model import init_model
from app.extraction import pdf_extractor
from app.domain import translator, query_expander
from app.utils.csv_ingest import normalise_csv_articles, strip_private_labels
from app.domain.query_expander import expand_query, generate_api_queries_with_llm, expand_query_with_synonyms
from app.utils.export import export_to_csv
from app.utils.eval_screening import evaluate_results
from app.screening.metadata_filter import concept_presence_filter
from app.core.adaptive_retrieval import recover_articles_from_near_misses
from app.core.two_phase_searcher import two_phase_search
from app.domain.eligibility_contract import (
    contract_to_synonym_payload,
    generate_eligibility_contract,
    rank_articles_by_contract,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client_id_var = contextvars.ContextVar('client_id', default=None)
progress_queues = {}

main_loop = None

class ProgressLogHandler(logging.Handler):
    def emit(self, record):
        # Filtrar logs de acceso de uvicorn y polling de SSE para no ensuciar la terminal del usuario
        if "uvicorn.access" in record.name or "/progress/" in record.getMessage():
            return

        cid = client_id_var.get()
        if cid and cid in progress_queues:
            msg = self.format(record)
            try:
                if main_loop and main_loop.is_running():
                    main_loop.call_soon_threadsafe(progress_queues[cid].put_nowait, msg)
                else:
                    progress_queues[cid].put_nowait(msg)
            except Exception:
                pass

logger = logging.getLogger()
progress_handler = ProgressLogHandler()
progress_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(progress_handler)

for d in [".cache", ".cache/sessions", "logs", "static", "templates"]:
    os.makedirs(d, exist_ok=True)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()
    logging.info("🚀 Iniciando servidor y cargando IA...")
    try:
        init_model()
    except Exception as e:
        logging.warning(f"⚠️ No se pudo cargar el modelo local (¿Falta GPU?): {e}")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/progress/{client_id}")
async def get_progress(client_id: str):
    if client_id not in progress_queues:
        progress_queues[client_id] = asyncio.Queue()

    async def event_generator():
        try:
            while True:
                msg = await progress_queues[client_id].get()
                if msg == "END_STREAM":
                    break
                msg = msg.replace('\n', ' ')
                yield f"data: {msg}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if client_id in progress_queues:
                del progress_queues[client_id]

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ============================================================
# 🚦 CONTROL DE CONCURRENCIA PARA IA
# ============================================================
# Máximo 2 peticiones simultáneas a proveedores externos para evitar 429
AI_SEMAPHORE = asyncio.Semaphore(2)

# Model initialization moved to the unified startup_event above

# ============================================================
# 🔄 SISTEMA DE SESIONES PERSISTENTE (MongoDB)
# ============================================================
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import config

if config.ENABLE_MONGODB:
    try:
        mongo_client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=5000)
        mongo_client.server_info()  # Test de conexión
        sessions_db = mongo_client['prisma_db']['sessions']
        logging.info("✅ MongoDB conectado para persistencia de sesiones")
        USE_MONGODB_SESSIONS = True
    except Exception as e:
        logging.warning(f"⚠️ MongoDB no disponible, usando memoria temporal: {e}")
        sessions_db = None
        USE_MONGODB_SESSIONS = False
else:
    logging.info("ℹ️ Persistencia en MongoDB deshabilitada por configuración")
    sessions_db = None
    USE_MONGODB_SESSIONS = False

# Fallback 1: Diccionario en memoria
TEMP_ARTICLES = {}
BASE_DIR = Path(__file__).resolve().parent
SESSION_FILE_DIR = BASE_DIR / ".cache" / "sessions"
os.makedirs(SESSION_FILE_DIR, exist_ok=True)

def _get_session_path(session_id: str) -> Path:
    return SESSION_FILE_DIR / f"session_{session_id}.json"

import json as pyjson
from datetime import datetime

def _serialize_session(data: dict) -> str:
    """Serialización segura para JSON."""
    def converter(o):
        if isinstance(o, datetime):
            return o.isoformat()
        return str(o)
    return pyjson.dumps(data, default=converter)

def save_session(session_id, data: dict):
    """Guarda sesión en Memoria -> Archivo -> MongoDB."""
    s_id = str(session_id)
    # 1. Memoria (Rápido)
    TEMP_ARTICLES[s_id] = data

    # 2. Archivo Local (Persistente ante reinicios)
    # SEGURIDAD: No sobreescribir con datos vacíos si ya existe información
    articles = data.get('articles', [])
    path = _get_session_path(s_id)

    if path.exists() and not articles:
        logging.warning(f"⚠️ Intento de guardar sesión {s_id} VACÍA bloqueado para proteger integridad.")
        return

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(_serialize_session(data))
        logging.info(f"💾 Sesión {s_id} guardada en disco: {path.name} ({len(articles)} artículos)")
    except Exception as e:
        logging.error(f"❌ Error guardando sesión {s_id}: {e}")

    # 3. MongoDB (Nube)
    if USE_MONGODB_SESSIONS:
        try:
            sessions_db.update_one(
                {"_id": s_id},
                {"$set": {**data, "updated_at": datetime.now()}},
                upsert=True
            )
        except Exception as e:
            if "quota" not in str(e).lower():
                logging.warning(f"⚠️ MongoDB save falló (sesión {s_id}): {e}")

def get_session(session_id) -> dict:
    """Recupera sesión: Memoria -> Archivo -> MongoDB."""
    s_id = str(session_id)
    # 1. Memoria
    if s_id in TEMP_ARTICLES:
        return TEMP_ARTICLES[s_id]

    # 2. Archivo Local
    try:
        path = _get_session_path(s_id)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = pyjson.load(f)
                TEMP_ARTICLES[s_id] = data # Cachear en memoria
                logging.info(f"📂 Sesión {s_id} recuperada desde disco.")
                return data
    except Exception as e:
        logging.debug(f"Lectura de archivo de sesión falló: {e}")

    # 3. MongoDB
    if USE_MONGODB_SESSIONS:
        try:
            result = sessions_db.find_one({"_id": s_id})
            if result:
                TEMP_ARTICLES[s_id] = result
                return result
        except:
            pass
    return None

def session_exists(session_id) -> bool:
    """Verifica existencia en cualquier capa."""
    return get_session(session_id) is not None

# Alias legible para compatibilidad con polling de cascade_status
sessions = TEMP_ARTICLES

CASCADE_MAX_ARTICLES = 0  # 0 = ranking completo; un valor positivo aplica un corte explícito.


def _apply_optional_rank_limit(articles: List[Dict[str, Any]], limit: int = 0) -> List[Dict[str, Any]]:
    effective_limit = max(0, int(limit or 0))
    if effective_limit <= 0:
        return articles
    return articles[:effective_limit]


PDF_INCOMPLETE_SOURCES = {
    "abstract_proxy",
    "abstract_short",
    "manual_pending",
    "failed",
    "none",
    "",
}


def _normalise_article_title(title: object) -> str:
    text = str(title or "").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def _article_has_real_pdf(article: Dict[str, Any]) -> bool:
    source = str(article.get("full_text_source") or article.get("acquisition_status") or "").lower()
    if source in PDF_INCOMPLETE_SOURCES:
        return False
    if not article.get("is_pdf_downloaded"):
        return False

    text = str(article.get("full_text") or "")
    if database.is_pdf_real(text):
        return True

    # Manual uploads and selective local extractions may omit references by design.
    return source in {"manual_upload", "local_csv_pdf", "pdf_cascade_extracted", "chromadb_recovery"} and len(text) >= 2500


def _copy_pdf_fields(target: Dict[str, Any], source: Dict[str, Any]) -> bool:
    changed = False
    for key in ("full_text", "is_pdf_downloaded", "needs_pdf_download", "full_text_source", "acquisition_status"):
        if key in source and source.get(key) is not None and source.get(key) != "" and target.get(key) != source.get(key):
            target[key] = source.get(key)
            changed = True
    return changed


def _iter_article_matches(article: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    original_index = article.get("original_index")
    doi = str(article.get("doi") or "").strip().lower()
    title_key = _normalise_article_title(article.get("title"))

    matches: List[Dict[str, Any]] = []
    for candidate in candidates:
        if original_index is not None and str(candidate.get("original_index")) == str(original_index):
            matches.append(candidate)
            continue
        if doi and str(candidate.get("doi") or "").strip().lower() == doi:
            matches.append(candidate)
            continue
        if title_key and _normalise_article_title(candidate.get("title")) == title_key:
            matches.append(candidate)
    return matches


def _enrich_pdf_status(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    ready = 0
    missing = 0
    uploaded = 0

    for article in articles:
        is_ready = _article_has_real_pdf(article)
        source = str(article.get("full_text_source") or article.get("acquisition_status") or "sin_texto")
        text_len = len(str(article.get("full_text") or ""))
        article["pdf_ready"] = is_ready
        article["pdf_status_label"] = "PDF completo" if is_ready else "Falta PDF completo"
        article["pdf_status_detail"] = f"{source}; {text_len} caracteres"
        if source == "manual_upload":
            uploaded += 1
        if is_ready:
            ready += 1
        else:
            missing += 1

    total = len(articles)
    return {
        "total": total,
        "ready": ready,
        "missing": missing,
        "uploaded": uploaded,
        "ready_pct": round((ready / total) * 100, 1) if total else 0.0,
    }


def _sync_included_pdf_state(session_data: Dict[str, Any]) -> Dict[str, Any]:
    included = session_data.get("included_articles", []) or []
    all_articles = session_data.get("articles", []) or []
    changed = False

    for article in included:
        for match in _iter_article_matches(article, all_articles):
            if _article_has_real_pdf(match) and not _article_has_real_pdf(article):
                changed = _copy_pdf_fields(article, match) or changed
                break

        if article.get("is_pdf_downloaded") and not _article_has_real_pdf(article):
            recovered_text = database.recover_full_text(article)
            if recovered_text and database.is_pdf_real(recovered_text):
                article["full_text"] = recovered_text
                article["full_text_source"] = "chromadb_recovery"
                article["is_pdf_downloaded"] = True
                article["needs_pdf_download"] = False
                changed = True

    session_data["included_articles"] = included
    audit = _enrich_pdf_status(included)
    return {"changed": changed, **audit}


def _sync_uploaded_pdf_to_included(session_data: Dict[str, Any], uploaded_article: Dict[str, Any]) -> bool:
    included = session_data.get("included_articles", []) or []
    changed = False
    for article in _iter_article_matches(uploaded_article, included):
        changed = _copy_pdf_fields(article, uploaded_article) or changed
    if changed:
        _enrich_pdf_status(included)
        session_data["included_articles"] = included
    return changed


def _internal_evaluation_allowed(token: str) -> bool:
    expected = str(getattr(config, "INTERNAL_EVALUATION_TOKEN", "") or "").strip()
    if not bool(getattr(config, "ENABLE_INTERNAL_EVALUATION", False)):
        return False
    return bool(expected) and str(token or "").strip() == expected


def _parse_positive_values(value: str) -> List[str]:
    raw_values = [part.strip() for part in str(value or "").split(",")]
    values = [part for part in raw_values if part]
    return values or ["1", "include", "included", "si", "sí", "yes", "relevant"]


def _attach_gold_labels(articles: List[Dict[str, Any]], labels_by_id: Dict[str, int]) -> List[Dict[str, Any]]:
    for article in articles:
        row_id = str(article.get("_csv_row_id") or "")
        if row_id in labels_by_id:
            article["_gold_label"] = labels_by_id[row_id]
    return articles


def _classification_metrics(
    predicted_articles: List[Dict[str, Any]],
    universe_articles: List[Dict[str, Any]],
    stage: str,
) -> Dict[str, Any]:
    labels = {
        str(article.get("_csv_row_id") or ""): int(article.get("_gold_label", 0) or 0)
        for article in universe_articles
        if article.get("_csv_row_id")
    }
    predicted_ids = {
        str(article.get("_csv_row_id") or "")
        for article in predicted_articles
        if article.get("_csv_row_id")
    }
    total_positive = sum(1 for value in labels.values() if value == 1)
    total_negative = len(labels) - total_positive
    true_positive = sum(1 for row_id in predicted_ids if labels.get(row_id) == 1)
    false_positive = sum(1 for row_id in predicted_ids if labels.get(row_id) == 0)
    false_negative = max(0, total_positive - true_positive)
    true_negative = max(0, total_negative - false_positive)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (true_positive + true_negative) / len(labels) if labels else 0.0
    return {
        "stage": stage,
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "predicted_positive": len(predicted_ids),
        "total": len(labels),
    }


def _optional_int(value: object) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _article_id_set(articles: List[Dict[str, Any]]) -> set[str]:
    return {str(article.get("_csv_row_id") or "") for article in articles if article.get("_csv_row_id")}


def _summarise_gold_article(article: Dict[str, Any], reason: str = "") -> Dict[str, Any]:
    details = article.get("deep_screening_details") or {}
    return {
        "row_id": article.get("_csv_row_id", ""),
        "title": article.get("title", "Sin titulo"),
        "year": article.get("year", ""),
        "doi": article.get("doi", ""),
        "label": article.get("_gold_label", ""),
        "score": article.get("deep_screening_score", article.get("similarity_stage1", "")),
        "reason": reason or article.get("exclusion_reason") or details.get("justification", ""),
    }


def _index_by_row_id(articles: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(article.get("_csv_row_id") or ""): article for article in articles if article.get("_csv_row_id")}


def _prepare_local_or_abstract_text(article: Dict[str, Any], use_local_pdf: bool = False) -> Dict[str, Any]:
    abstract = str(article.get("abstract") or "").strip()
    if not use_local_pdf:
        article["full_text"] = abstract
        article["full_text_source"] = "abstract_proxy"
        article["is_pdf_downloaded"] = False
        return article

    pdf_path = str(article.get("pdf_path") or "").strip()
    if pdf_path:
        path = Path(pdf_path)
        try:
            if path.exists():
                pdf_bytes = path.read_bytes()
                if not pdf_bytes.startswith(b"%PDF-"):
                    raise ValueError("invalid_pdf_magic")
                extracted = pdf_extractor.extract_selective_sections_with_timeout(pdf_bytes)
                if extracted and len(extracted.strip()) >= 100:
                    article["full_text"] = extracted
                    article["full_text_source"] = "local_csv_pdf"
                    article["is_pdf_downloaded"] = True
                    return article
                article["pdf_audit_error"] = "pdf_text_too_short"
        except Exception as exc:
            article["pdf_audit_error"] = str(exc)
    article["full_text"] = abstract
    article["full_text_source"] = "abstract_proxy"
    article["is_pdf_downloaded"] = False
    return article


def _trim_stage4_text(text: str) -> str:
    clean = str(text or "").strip()
    max_chars = int(getattr(config, "GOLD_EVAL_STAGE4_MAX_CHARS", 9000) or 9000)
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean
    head_chars = max(3000, int(max_chars * 0.68))
    tail_chars = max(1000, max_chars - head_chars)
    return (
        clean[:head_chars].rstrip()
        + "\n\n[... texto recortado para acelerar evaluacion interna ...]\n\n"
        + clean[-tail_chars:].lstrip()
    )


def _prepare_stage4_text(
    article: Dict[str, Any],
    text_source_mode: str,
    index: int,
    total: int,
) -> Dict[str, Any]:
    title = str(article.get("title") or "Sin titulo")[:70]
    logging.info("[GoldEval] Texto Stage 4 [%d/%d] %s: %s", index, total, text_source_mode, title)

    if text_source_mode == "system_cascade":
        try:
            prepared = pdf_extractor.acquire_full_text(article.copy(), force=True)
            full_text = str(prepared.get("full_text") or "")
            if len(full_text) >= 4000:
                sections = pdf_extractor.extract_selective_sections_from_text(full_text)
                if sections:
                    prepared["full_text"] = sections
                    prepared["full_text_source"] = f"{prepared.get('full_text_source', 'pdf')}_sections"
        except Exception as exc:
            logging.warning("[GoldEval] Cascada PDF fallo para '%s': %s", title, exc)
            prepared = _prepare_local_or_abstract_text(article.copy(), False)
    elif text_source_mode == "unpaywall":
        try:
            prepared = pdf_extractor.acquire_unpaywall_text(article.copy())
        except Exception as exc:
            logging.warning("[GoldEval] Unpaywall fallo para '%s': %s", title, exc)
            prepared = _prepare_local_or_abstract_text(article.copy(), False)
    elif text_source_mode == "local_pdf":
        prepared = _prepare_local_or_abstract_text(article.copy(), True)
    else:
        prepared = _prepare_local_or_abstract_text(article.copy(), False)

    original_chars = len(str(prepared.get("full_text") or ""))
    prepared["full_text"] = _trim_stage4_text(str(prepared.get("full_text") or ""))
    prepared["stage4_text_chars_original"] = original_chars
    prepared["stage4_text_chars"] = len(str(prepared.get("full_text") or ""))

    logging.info(
        "[GoldEval] Texto listo [%d/%d] fuente=%s chars=%d/%d: %s",
        index,
        total,
        prepared.get("full_text_source", "desconocida"),
        prepared.get("stage4_text_chars", 0),
        original_chars,
        title,
    )
    return prepared


def _normalise_uploaded_filename(upload: Optional[UploadFile]) -> str:
    return str(getattr(upload, "filename", "") or "").strip()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def _stage4_llm_summary(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    times = [
        _safe_float(article.get("stage4_llm_seconds"))
        for article in articles
        if _safe_float(article.get("stage4_llm_seconds")) > 0
    ]
    if not times:
        return {"count": 0, "average": 0.0, "min": 0.0, "max": 0.0, "total": 0.0}
    return {
        "count": len(times),
        "average": round(sum(times) / len(times), 3),
        "min": round(min(times), 3),
        "max": round(max(times), 3),
        "total": round(sum(times), 3),
    }


def _stage4_model_choices() -> List[str]:
    return ["deepseek-v4-flash", "gemma3:12b", "gemma4:31b"]


def _embedding_model_label() -> str:
    if bool(getattr(config, "USE_OLLAMA_EMBEDDING", False)):
        model = str(getattr(config, "OLLAMA_EMBEDDING_MODEL", "") or "nomic-embed-text")
        dim = int(getattr(config, "EMBEDDING_DIM", 0) or 0)
        return f"Ollama: {model} ({dim} dimensiones)"
    return str(getattr(config, "EMBEDDING_MODEL", "No configurado"))


def _cascade_rank_key(article: Dict) -> tuple:
    return (
        _safe_float(article.get("_atom_coverage_score")),
        _safe_float(article.get("relevance_score")),
        int(_safe_float(article.get("_concepts_matched_count"))),
        _safe_float(article.get("_concept_bonus")),
        int(_safe_float(article.get("citations"))),
    )


def _citation_sort_key(article: Dict) -> tuple:
    return (
        int(_safe_float(article.get("citations"))),
        _safe_float(article.get("relevance_score")),
        int(_safe_float(article.get("year"))),
    )


def _sort_by_citations(articles: List[Dict]) -> List[Dict]:
    return sorted(articles, key=_citation_sort_key, reverse=True)


def _run_cascade_background(
    articles: List[Dict],
    question: str,
    translated_q: str,
    inclusion_criteria: str,
    exclusion_criteria: str,
    session_id: str,
    sessions_dict: Dict[str, dict],
) -> None:
    try:
        # Lazy imports (mÃ³dulos pesados)
        from app.screening.fast_filter import apply_stage1_fast_filter
        from app.extraction.pdf_extractor import acquire_full_text, extract_selective_sections_from_text
        from app.screening.deep_screener import screen_candidates_cascade

        articles = _apply_optional_rank_limit(list(articles or []), CASCADE_MAX_ARTICLES)
        eligibility_contract = generate_eligibility_contract(
            translated_q or question,
            inclusion_criteria or "",
            exclusion_criteria or "",
        )
        if getattr(config, "ATOM_COVERAGE_RANKING_ENABLED", True):
            articles = rank_articles_by_contract(articles, eligibility_contract)

        passed_s1, _excluded_s1 = apply_stage1_fast_filter(articles, translated_q, threshold=0.45)
        sessions_dict.setdefault(session_id, {})
        sessions_dict[session_id]["cascade_status"] = f"running_s2:{len(passed_s1)}"
        sessions_dict[session_id]["cascade_total"] = len(articles)
        try:
            data = get_session(session_id) or {}
            data["cascade_status"] = sessions_dict[session_id]["cascade_status"]
            data["cascade_total"] = len(articles)
            save_session(session_id, data)
        except Exception:
            pass

        ABSTRACT_MIN_CHARS = int(getattr(config, "ABSTRACT_MIN_CHARS", "800"))

        abstract_sufficient = []
        pdf_needed = []

        for art in passed_s1:
            abstract = (art.get("abstract") or "").strip()
            if len(abstract) >= ABSTRACT_MIN_CHARS:
                art["full_text"] = abstract
                art["full_text_source"] = "abstract_proxy"
                abstract_sufficient.append(art)
            else:
                pdf_needed.append(art)

        from concurrent.futures import as_completed

        if pdf_needed:
            logging.info(
                "[Cascade] %d articulos con abstract < %d chars: descargando PDFs...",
                len(pdf_needed),
                ABSTRACT_MIN_CHARS,
            )
            with ThreadPoolExecutor(max_workers=4) as pool:
                fut_map = {pool.submit(acquire_full_text, art): art for art in pdf_needed}
                for future in as_completed(fut_map):
                    art = fut_map[future]
                    res = future.result()
                    if isinstance(res, dict):
                        art.update(res)

        stage4_input = abstract_sufficient + pdf_needed

        sessions_dict[session_id]["cascade_status"] = f"running_s4:{len(stage4_input)}"
        try:
            data = get_session(session_id) or {}
            data["cascade_status"] = sessions_dict[session_id]["cascade_status"]
            save_session(session_id, data)
        except Exception:
            pass

        passed_stage4, excluded_stage4 = screen_candidates_cascade(
            stage4_input,
            translated_q,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
            eligibility_contract=eligibility_contract,
        )

        if passed_stage4:
            logging.info(
                "[Cascade] Descargando PDFs completos para %d articulos INCLUIDOS...",
                len(passed_stage4),
            )
            with ThreadPoolExecutor(max_workers=4) as pool:
                fut_map = {pool.submit(acquire_full_text, art): art for art in passed_stage4}
                for future in as_completed(fut_map):
                    art = fut_map[future]
                    res = future.result()
                    if isinstance(res, dict):
                        art.update(res)

            for art in passed_stage4:
                art["screened_sections"] = extract_selective_sections_from_text(
                    art.get("full_text", "") or ""
                )

        sessions_dict[session_id]["cascade_results"] = passed_stage4
        sessions_dict[session_id]["cascade_excluded"] = excluded_stage4
        sessions_dict[session_id]["cascade_background"] = [
            article for article in excluded_stage4
            if (article.get("deep_screening_details") or {}).get("decision") == "BACKGROUND_ONLY"
        ]
        sessions_dict[session_id]["cascade_status"] = "done"

        try:
            data = get_session(session_id) or {}
            data["cascade_results"] = passed_stage4
            data["cascade_excluded"] = excluded_stage4
            data["cascade_background"] = sessions_dict[session_id]["cascade_background"]
            data["cascade_status"] = "done"
            data["eligibility_contract"] = eligibility_contract
            save_session(session_id, data)
        except Exception:
            pass

        logging.info(f"âœ… Cascade background: {len(passed_stage4)}/{len(articles)} aprobados")
    except Exception as e:
        sessions_dict.setdefault(session_id, {})
        sessions_dict[session_id]["cascade_status"] = "error"
        try:
            data = get_session(session_id) or {}
            data["cascade_status"] = "error"
            save_session(session_id, data)
        except Exception:
            pass
        logging.error(f"âŒ Cascade background fallÃ³: {e}", exc_info=True)


@app.get("/cascade_status/{session_id}")
def cascade_status(session_id: str):
    s = sessions.get(session_id, {})
    status = s.get("cascade_status", "not_started")
    results = s.get("cascade_results", [])
    approved_ids = [
        (a.get("doi") or a.get("title", ""))
        for a in results
        if isinstance(a, dict)
    ]
    return {
        "status": status,
        "approved": len(results),
        "total": int(s.get("cascade_total") or len(s.get("relevant_articles", []))),
        "approved_ids": approved_ids,
    }

@app.get("/download_audit/{session_id}")
async def download_audit(session_id: int):
    if not session_exists(session_id):
        raise HTTPException(404, "Sesión no encontrada")

    session_data = get_session(session_id)
    excluded = session_data.get("excluded_articles", [])
    question = session_data.get("question", "investigacion")

    if not excluded:
        df = pd.DataFrame([{"Mensaje": "No se encontraron artículos excluidos por el filtro conceptual en esta sesión."}])
    else:
        rows = []
        for art in excluded:
            rows.append({
                "Título": art.get("title", "Sin título"),
                "Año": art.get("year", "N/D"),
                "Revista/Fuente": art.get("journal") or art.get("venue", "N/D"),
                "DOI": art.get("doi", ""),
                "Razón de Exclusión": art.get("_exclusion_reason", "Filtro Temático"),
                "Conceptos Detectados (✅)": ", ".join(art.get("_concepts_matched_list", [])),
                "Conceptos Ausentes (❌)": ", ".join(art.get("_concepts_missing_list", [])),
                "Resumen (Abstract)": art.get("abstract", "N/D")
            })
        df = pd.DataFrame(rows)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Excluidos_Filtro_Conceptual')

    output.seek(0)

    # Nombre representativo: Auditoria_Fase_Identificacion_Filtro_Conceptual_ResumenPregunta_ID
    safe_q = "".join([c if c.isalnum() else "_" for c in question[:40]]).strip("_")
    filename = f"Auditoria_Fase_Identificacion_Filtro_Conceptual_{safe_q}_{session_id}.xlsx"

    headers = {
        'Content-Disposition': f'attachment; filename="{filename}"'
    }
    return StreamingResponse(output, headers=headers, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.get("/download_audit_pdf/{session_id}")
async def download_audit_pdf(session_id: int):
    if not session_exists(session_id):
        raise HTTPException(404, "Sesión no encontrada")

    session_data = get_session(session_id)
    from app.utils.audit_report_pdf import generate_audit_pdf

    try:
        pdf_bytes = generate_audit_pdf(session_data)

        question = session_data.get("question", "investigacion")
        safe_q = "".join([c if c.isalnum() else "_" for c in question[:40]]).strip("_")
        filename = f"Auditoria_Fase_Identificacion_Filtro_Conceptual_{safe_q}_{session_id}.pdf"

        headers = {
            'Content-Disposition': f'attachment; filename="{filename}"'
        }
        import io
        return StreamingResponse(io.BytesIO(pdf_bytes), headers=headers, media_type='application/pdf')
    except Exception as e:
        logging.error(f"❌ Error generando PDF de auditoría: {e}")
        return JSONResponse({"error": f"Error generando PDF: {str(e)}"}, 500)

@app.get("/download_audit_zip/{session_id}")
async def download_audit_zip(session_id: int):
    """Genera un ZIP con todos los CSVs de auditoría de la sesión."""
    import zipfile
    import io
    from pathlib import Path

    audit_dir = Path("logs") / f"session_{session_id}"
    if not audit_dir.exists():
        raise HTTPException(404, "No se encontraron archivos de auditoría para esta sesión")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for csv_file in audit_dir.glob("*.csv"):
            zip_file.write(csv_file, arcname=csv_file.name)

    zip_buffer.seek(0)
    filename = f"Auditoria_PrismaAssistant_{session_id}.zip"
    headers = {'Content-Disposition': f'attachment; filename="{filename}"'}
    return StreamingResponse(zip_buffer, headers=headers, media_type='application/zip')
# ============================================================

def clean_old_cache_and_logs():
    """Limpia archivos CSV viejos al reiniciar."""
    try:
        for f in Path("logs").glob("*.csv"):
            try: f.unlink()
            except: pass
    except: pass

clean_old_cache_and_logs()

def _sanitize_bibtex_key(s: str) -> str:
    """Sanitiza una cadena para usarla como cite_key BibTeX válido."""
    import unicodedata
    # Normalizar unicode → ASCII
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    # Mantener solo alfanuméricos
    return re.sub(r'[^a-zA-Z0-9]', '', s)

def generate_bibtex(article: Dict) -> str:
    """Genera una entrada BibTeX completa, compatible con Zotero/JabRef."""
    try:
        auth = article.get('authors', [])
        if isinstance(auth, list) and auth:
            first_author = auth[0].split(" ")[-1]
        elif isinstance(auth, str) and auth:
            first_author = auth.split(",")[0].split(" ")[-1]
        else:
            first_author = "Unknown"

        year = str(article.get('year', 'nd'))
        title_words = re.findall(r'[a-zA-Z]+', article.get('title', ''))
        title_slug = "".join(title_words[:2])
        cite_key = _sanitize_bibtex_key(f"{first_author}{year}{title_slug}")
        if not cite_key:
            cite_key = f"ref{abs(hash(article.get('title', ''))) % 100000}"

        journal_raw = str(article.get('journal', ''))
        journal_lower = journal_raw.lower()
        entry_type = "article"
        if any(kw in journal_lower for kw in ["conference", "proceeding", "symposium", "workshop"]):
            entry_type = "inproceedings"
        elif "arxiv" in journal_lower:
            entry_type = "misc"

        fields = []
        fields.append(f"  title     = {{{article.get('title', 'No Title')}}}")

        if isinstance(auth, list):
            auth_str = " and ".join(auth)
        else:
            auth_str = str(auth)
        fields.append(f"  author    = {{{auth_str}}}")
        fields.append(f"  year      = {{{year}}}")

        if journal_raw:
            key = "booktitle" if entry_type == "inproceedings" else "journal"
            fields.append(f"  {key:<8} = {{{journal_raw}}}")
        if article.get('volume'):  fields.append(f"  volume    = {{{article.get('volume')}}}")
        if article.get('issue'):   fields.append(f"  number    = {{{article.get('issue')}}}")
        if article.get('pages'):   fields.append(f"  pages     = {{{article.get('pages')}}}")
        if article.get('doi'):     fields.append(f"  doi       = {{{article.get('doi')}}}")
        if article.get('url'):     fields.append(f"  url       = {{{article.get('url')}}}")

        # BibTeX válido: separar campos con coma+newline, SIN coma en el último campo
        body = ",\n".join(fields)
        bib = f"@{entry_type}{{{cite_key},\n{body}\n}}"
        return bib
    except Exception as e:
        logging.warning(f"⚠️ Error generando BibTeX: {e}")
        return ""


def _format_authors_display(raw_authors: Any) -> str:
    """Devuelve autores completos para UI; no recorta con et al. ni iniciales artificiales."""
    def author_to_text(author: Any) -> str:
        if isinstance(author, dict):
            given = str(author.get("given") or author.get("first") or "").strip()
            family = str(author.get("family") or author.get("last") or "").strip()
            literal = str(author.get("name") or author.get("literal") or "").strip()
            return " ".join(part for part in [given, family] if part).strip() or literal
        return str(author).strip()

    if isinstance(raw_authors, list):
        authors = [author_to_text(author) for author in raw_authors if author_to_text(author)]
    elif isinstance(raw_authors, str):
        separator = ";" if ";" in raw_authors else " and "
        authors = [part.strip() for part in raw_authors.split(separator) if part.strip()]
    else:
        authors = []
    return "; ".join(authors) if authors else "Autores no disponibles"


def normalize_article_for_csv(article: Dict, ai_fields: List[str] = None) -> Dict:
    """Asegura que campos críticos como URL y campos dinámicos de IA estén limpios."""
    url = article.get('url') or article.get('pdf_url') or article.get('link') or ""

    if not url and article.get('doi'):
        url = f"https://doi.org/{article.get('doi')}"

    if not url and (article.get('source') == 'PubMed' or str(article.get('id', '')).isdigit()):
        pmid = article.get('pubmed_id') or article.get('id')
        if pmid: url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    article['url'] = url
    article['source_url'] = url
    article['authors_display'] = _format_authors_display(article.get('authors') or article.get('author'))
    if not article.get('pdf_url') and url:
        article['pdf_url'] = url
        article['needs_pdf_download'] = True

    article['bibtex'] = generate_bibtex(article)

    # Limpieza de HTML para campos de IA (evitar rupturas en CSV)
    if ai_fields:
        for field in ai_fields:
            if field not in article: article[field] = ""
            if article.get(field):
                val = str(article[field])
                clean_val = re.sub('<[^<]+?>', ' ', val)
                clean_val = re.sub(' +', ' ', clean_val).strip()
                article[field] = clean_val

    return article

@app.get("/health")
async def health_check():
    """Health check endpoint para Docker y monitoreo externo."""
    return {"status": "ok", "service": "prisma-assistant"}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "internal_evaluation_enabled": bool(getattr(config, "ENABLE_INTERNAL_EVALUATION", False)),
    })

@app.post("/csv/preview", response_class=JSONResponse)
async def csv_preview(file: UploadFile = File(...)):
    filename = _normalise_uploaded_filename(file)
    if not filename.lower().endswith(".csv"):
        raise HTTPException(400, "Solo se aceptan archivos CSV.")
    try:
        content = await file.read()
        _, preview = normalise_csv_articles(content, include_labels=False)
        return preview
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logging.exception("Error previsualizando CSV")
        raise HTTPException(500, f"No se pudo leer el CSV: {exc}")


@app.get("/evaluation", response_class=HTMLResponse)
async def evaluation_page(request: Request, token: str = ""):
    if not _internal_evaluation_allowed(token):
        raise HTTPException(status_code=404, detail="Not found")
    return templates.TemplateResponse("evaluation.html", {
        "request": request,
        "token": token,
        "stage4_model_choices": _stage4_model_choices(),
        "form": {},
        "preview": None,
        "result": None,
        "error": None,
    })


@app.post("/internal/evaluate_gold", response_class=HTMLResponse)
async def evaluate_gold_standard(
    request: Request,
    token: str = Form(...),
    question: str = Form(...),
    inclusion_criteria: str = Form(""),
    exclusion_criteria: str = Form(""),
    start_year: str = Form(""),
    end_year: str = Form(""),
    academic_quality: str = Form("false"),
    stage1_threshold: str = Form("0.45"),
    cascade_limit: str = Form("0"),
    stage4_model: str = Form(""),
    compare_stage4_models: str = Form("false"),
    text_source_mode: str = Form("abstract"),
    use_local_pdfs: str = Form("false"),
    skip_concept_filter: str = Form("false"),
    label_column: str = Form(""),
    positive_values: str = Form("1,include,included,si,sí,yes,relevant"),
    gold_csv: UploadFile = File(...),
):
    if not _internal_evaluation_allowed(token):
        raise HTTPException(status_code=404, detail="Not found")

    form_values = {
        "question": question,
        "inclusion_criteria": inclusion_criteria,
        "exclusion_criteria": exclusion_criteria,
        "start_year": start_year,
        "end_year": end_year,
        "academic_quality": academic_quality,
        "stage1_threshold": stage1_threshold,
        "cascade_limit": cascade_limit,
        "stage4_model": stage4_model,
        "compare_stage4_models": compare_stage4_models,
        "text_source_mode": text_source_mode,
        "use_local_pdfs": use_local_pdfs,
        "skip_concept_filter": skip_concept_filter,
        "label_column": label_column,
        "positive_values": positive_values,
    }

    filename = _normalise_uploaded_filename(gold_csv)
    if not filename.lower().endswith(".csv"):
        return templates.TemplateResponse("evaluation.html", {
            "request": request,
            "token": token,
            "stage4_model_choices": _stage4_model_choices(),
            "form": form_values,
            "preview": None,
            "result": None,
            "error": "Solo se aceptan archivos CSV.",
        })

    total_start = time.perf_counter()
    timings: Dict[str, float] = {}
    try:
        step_start = time.perf_counter()
        labeled_articles, preview = normalise_csv_articles(
            await gold_csv.read(),
            source_name="Gold Standard CSV",
            include_labels=True,
            label_column=label_column.strip(),
            positive_values=_parse_positive_values(positive_values),
        )
        timings["import_csv"] = round(time.perf_counter() - step_start, 3)

        labels_by_id = {
            str(article.get("_csv_row_id") or ""): int(article.get("_gold_label", 0) or 0)
            for article in labeled_articles
            if article.get("_csv_row_id") and "_gold_label" in article
        }
        if not labels_by_id:
            raise ValueError("No se detecto una columna de gold standard. Usa manual_decision, Inclusión manual o indica la columna exacta.")
        if preview.get("unparsed_label_values"):
            raise ValueError(
                f"Hay etiquetas no reconocidas: {preview['unparsed_label_values']}. "
                "Ajusta la columna o los valores positivos antes de medir metricas."
            )

        model_articles = [
            normalize_article_for_csv(strip_private_labels(article.copy()))
            for article in labeled_articles
        ]
        gold_universe = _attach_gold_labels([article.copy() for article in model_articles], labels_by_id)
        parsed_start_year = _optional_int(start_year)
        parsed_end_year = _optional_int(end_year)
        threshold = min(1.0, max(0.0, _safe_float(stage1_threshold, 0.45)))
        effective_cascade_limit = max(0, int(_safe_float(cascade_limit, 0)))
        available_stage4_models = _stage4_model_choices()
        selected_stage4_model = str(stage4_model or "").strip() or available_stage4_models[0]
        if selected_stage4_model not in available_stage4_models:
            available_stage4_models.insert(0, selected_stage4_model)
        compare_models = compare_stage4_models == "true"
        selected_text_source = str(text_source_mode or "abstract").strip().lower()
        if use_local_pdfs == "true" and selected_text_source == "abstract":
            selected_text_source = "local_pdf"
        if selected_text_source not in {"abstract", "local_pdf", "unpaywall", "system_cascade"}:
            selected_text_source = "abstract"
        compare_pair = ["gemma3:12b", "gemma4:31b"]
        stage4_models_to_run = (
            [selected_stage4_model] + [model for model in compare_pair if model != selected_stage4_model]
            if compare_models
            else [selected_stage4_model]
        )

        step_start = time.perf_counter()
        eligibility_contract = generate_eligibility_contract(
            question,
            inclusion_criteria or "",
            exclusion_criteria or "",
        )
        metadata_candidates = filters.apply_filters(
            model_articles,
            start_year=parsed_start_year,
            end_year=parsed_end_year,
            academic_only=(academic_quality == "true"),
        )
        dedup_candidates, duplicate_candidates = deduplication.remove_exact_duplicates(metadata_candidates)
        timings["metadata_dedup"] = round(time.perf_counter() - step_start, 3)

        step_start = time.perf_counter()
        concept_candidates = dedup_candidates
        concept_excluded: List[Dict[str, Any]] = []
        concept_report: Dict[str, Any] = {
            "total": len(dedup_candidates),
            "passed": len(dedup_candidates),
            "excluded": 0,
            "skipped": True,
        }
        try:
            if skip_concept_filter == "true":
                logging.info("[GoldEval] Bypass de filtro conceptual solicitado — saltando...")
            else:
                synonym_data = contract_to_synonym_payload(eligibility_contract)
                if isinstance(synonym_data, dict) and synonym_data.get("synonyms"):
                    concept_candidates, concept_excluded, concept_report_obj = concept_presence_filter(
                        dedup_candidates,
                        synonym_data=synonym_data,
                    )
                    concept_report = asdict(concept_report_obj)
        except Exception as exc:
            logging.warning("[GoldEval] Prefiltro conceptual omitido: %s", exc)
        timings["concept_filter"] = round(time.perf_counter() - step_start, 3)

        screening_pool = concept_candidates
        candidates_before_cap = len(screening_pool)
        if getattr(config, "ATOM_COVERAGE_RANKING_ENABLED", True):
            screening_pool = rank_articles_by_contract(screening_pool, eligibility_contract)
        cascade_candidates = sorted(screening_pool, key=_cascade_rank_key, reverse=True)
        if effective_cascade_limit > 0:
            cascade_candidates = cascade_candidates[:effective_cascade_limit]

        step_start = time.perf_counter()
        from app.screening.fast_filter import apply_stage1_fast_filter
        passed_stage1, excluded_stage1 = apply_stage1_fast_filter(
            cascade_candidates,
            question,
            threshold=threshold,
        )
        timings["stage1"] = round(time.perf_counter() - step_start, 3)

        stage4_input = passed_stage1

        step_start = time.perf_counter()
        logging.info(
            "[GoldEval] Preparando texto para Stage 4: %d candidatos | modo=%s",
            len(stage4_input),
            selected_text_source,
        )
        indexed_stage4_input = [
            (index, article)
            for index, article in enumerate(stage4_input, start=1)
        ]
        ABSTRACT_EVAL_MIN_CHARS = int(getattr(config, "GOLD_EVAL_ABSTRACT_MIN_CHARS", "800"))

        if selected_text_source in {"local_pdf", "unpaywall", "system_cascade"}:
            with ThreadPoolExecutor(max_workers=2) as executor:
                prepared_stage1 = list(
                    executor.map(
                        lambda item: _prepare_stage4_text(
                            item[1],
                            selected_text_source,
                            item[0],
                            len(stage4_input),
                        ),
                        indexed_stage4_input,
                    )
                )
        else:
            def _prepare_abstract_optimized(article, idx, total):
                abstract = (article.get("abstract") or "").strip()
                if len(abstract) >= ABSTRACT_EVAL_MIN_CHARS:
                    article["full_text"] = abstract
                    article["full_text_source"] = "abstract_proxy"
                    article["is_pdf_downloaded"] = False
                    return article
                logging.info(
                    "[GoldEval] Abstract corto (%d chars) [%d/%d]: %s — intentando PDF fallback",
                    len(abstract), idx, total,
                    str(article.get("title") or "Sin titulo")[:70],
                )
                try:
                    prepared = pdf_extractor.acquire_full_text(article.copy(), force=True)
                    full_text = str(prepared.get("full_text") or "")
                    if len(full_text.strip()) >= ABSTRACT_EVAL_MIN_CHARS:
                        return prepared
                except Exception as exc:
                    logging.warning("[GoldEval] PDF fallback fallo: %s", exc)
                article["full_text"] = abstract
                article["full_text_source"] = "abstract_proxy_fallback"
                article["is_pdf_downloaded"] = False
                return article

            prepared_stage1 = [
                _prepare_abstract_optimized(article, index, len(stage4_input))
                for index, article in indexed_stage4_input
            ]
        timings["text_preparation"] = round(time.perf_counter() - step_start, 3)
        stage4_text_lengths = [
            len(str(article.get("full_text") or ""))
            for article in prepared_stage1
        ]
        avg_stage4_text_chars = round(
            sum(stage4_text_lengths) / len(stage4_text_lengths),
            0,
        ) if stage4_text_lengths else 0
        max_stage4_text_chars = max(stage4_text_lengths) if stage4_text_lengths else 0
        logging.info(
            "[GoldEval] Preparacion de texto completada en %.3fs. Entrando a Stage 4...",
            timings["text_preparation"],
        )

        from app.screening.deep_screener import screen_candidates_cascade

        def labelled_copy(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return _attach_gold_labels([item.copy() for item in items], labels_by_id)

        common_metrics = {
            "metadata": _classification_metrics(labelled_copy(metadata_candidates), gold_universe, "Metadata"),
            "concept": _classification_metrics(labelled_copy(concept_candidates), gold_universe, "Conceptos"),
            "stage1": _classification_metrics(labelled_copy(passed_stage1), gold_universe, "Stage 1"),
        }

        stage4_runs: List[Dict[str, Any]] = []
        for model_name in stage4_models_to_run:
            step_start = time.perf_counter()
            run_articles = [article.copy() for article in prepared_stage1]
            run_passed, run_excluded = screen_candidates_cascade(
                run_articles,
                question,
                inclusion_criteria=inclusion_criteria or "",
                exclusion_criteria=exclusion_criteria or "",
                eligibility_contract=None if skip_concept_filter == "true" else eligibility_contract,
                judge_model=model_name,
                skip_atom_guard=(skip_concept_filter == "true"),
            )
            run_seconds = round(time.perf_counter() - step_start, 3)
            run_summary = _stage4_llm_summary(run_passed + run_excluded)
            run_metrics = _classification_metrics(
                labelled_copy(run_passed),
                gold_universe,
                f"Stage 4 final ({model_name})",
            )
            stage4_runs.append({
                "model": model_name,
                "passed": run_passed,
                "excluded": run_excluded,
                "seconds": run_seconds,
                "llm_summary": run_summary,
                "metrics": run_metrics,
                "included": len(run_passed),
                "excluded_count": len(run_excluded),
            })

        selected_run = stage4_runs[0]
        final_stage4 = selected_run["passed"]
        excluded_stage4 = selected_run["excluded"]
        llm_summary = selected_run["llm_summary"]
        timings["stage4_llm"] = selected_run["seconds"]
        timings["stage4_avg_llm"] = llm_summary["average"]
        metrics_rows = list(common_metrics.values()) + [run["metrics"] for run in stage4_runs]

        final_ids = _article_id_set(final_stage4)
        stage_ids = {
            "metadata": _article_id_set(metadata_candidates),
            "dedup": _article_id_set(dedup_candidates),
            "concept": _article_id_set(concept_candidates),
            "screening_pool": _article_id_set(screening_pool),
            "cascade": _article_id_set(cascade_candidates),
            "stage1": _article_id_set(passed_stage1),
            "stage4_input": _article_id_set(stage4_input),
        }
        concept_excluded_by_id = _index_by_row_id(concept_excluded)
        stage1_excluded_by_id = _index_by_row_id(excluded_stage1)
        stage4_excluded_by_id = _index_by_row_id(excluded_stage4)

        false_positives = [
            _summarise_gold_article(article, "El pipeline lo retuvo, pero el gold standard es 0.")
            for article in labelled_copy(final_stage4)
            if int(article.get("_gold_label", 0) or 0) == 0
        ]
        false_negatives: List[Dict[str, Any]] = []
        for article in gold_universe:
            row_id = str(article.get("_csv_row_id") or "")
            if int(article.get("_gold_label", 0) or 0) != 1 or row_id in final_ids:
                continue
            if row_id not in stage_ids["metadata"]:
                reason = "Excluido por filtros de año/calidad academica."
            elif row_id not in stage_ids["dedup"]:
                reason = "Eliminado como duplicado antes de la cascada."
            elif row_id not in stage_ids["screening_pool"]:
                excluded = concept_excluded_by_id.get(row_id, {})
                reason = excluded.get("_exclusion_reason", "Excluido por prefiltro conceptual.")
            elif row_id not in stage_ids["cascade"]:
                reason = "No fue evaluado por el limite de cascada."
            elif row_id not in stage_ids["stage1"]:
                excluded = stage1_excluded_by_id.get(row_id, {})
                reason = excluded.get("exclusion_reason", "Excluido en Stage 1.")
            elif row_id not in stage_ids["stage4_input"]:
                reason = "No llego a Stage 4."
            else:
                excluded = stage4_excluded_by_id.get(row_id, {})
                reason = excluded.get("exclusion_reason", "Excluido en Stage 4.")
            false_negatives.append(_summarise_gold_article(article, reason))

        timings["total"] = round(time.perf_counter() - total_start, 3)
        timing_details = [
            {
                "stage": "Importacion CSV",
                "seconds": timings.get("import_csv", 0.0),
                "detail": f"{len(gold_universe)} filas validas",
            },
            {
                "stage": "Metadatos y duplicados",
                "seconds": timings.get("metadata_dedup", 0.0),
                "detail": f"{len(metadata_candidates)} candidatos; {len(duplicate_candidates)} duplicados",
            },
            {
                "stage": "Filtro conceptual",
                "seconds": timings.get("concept_filter", 0.0),
                "detail": f"{len(concept_candidates)}/{len(dedup_candidates)} candidatos pasaron",
            },
            {
                "stage": "Stage 1 similitud",
                "seconds": timings.get("stage1", 0.0),
                "detail": f"{len(passed_stage1)}/{len(cascade_candidates)} candidatos pasaron",
            },
            {
                "stage": "Preparacion PDF/texto",
                "seconds": timings.get("text_preparation", 0.0),
                "detail": (
                    f"{len(prepared_stage1)} documentos preparados; "
                    f"modo={selected_text_source}; "
                    f"texto promedio={avg_stage4_text_chars} chars; "
                    f"max={max_stage4_text_chars} chars"
                ),
            },
        ]
        for run in stage4_runs:
            run_summary = run["llm_summary"]
            timing_details.append({
                "stage": f"Stage 4 LLM ({run['model']})",
                "seconds": run["seconds"],
                "detail": f"{run_summary['count']} llamadas; promedio={run_summary['average']}s",
            })
        timing_details.append({
            "stage": "Promedio respuesta LLM seleccionado",
            "seconds": llm_summary["average"],
            "detail": f"modelo={selected_run['model']}; min={llm_summary['min']}s; max={llm_summary['max']}s",
        })
        total_seconds = round(timings.get("total", 0.0), 3)
        llm_seconds = round(timings.get("stage4_llm", 0.0), 3)
        prep_seconds = round(timings.get("text_preparation", 0.0), 3)
        result = {
            "filename": filename,
            "counts": {
                "gold_total": len(gold_universe),
                "gold_positive": sum(1 for article in gold_universe if int(article.get("_gold_label", 0) or 0) == 1),
                "gold_negative": sum(1 for article in gold_universe if int(article.get("_gold_label", 0) or 0) == 0),
                "metadata_candidates": len(metadata_candidates),
                "metadata_gold_positive": sum(
                    1
                    for article in metadata_candidates
                    if labels_by_id.get(str(article.get("_csv_row_id") or "")) == 1
                ),
                "duplicates_removed": len(duplicate_candidates),
                "concept_candidates": len(concept_candidates),
                "cascade_candidates_before_cap": candidates_before_cap,
                "cascade_candidates_evaluated": len(cascade_candidates),
                "stage1_passed": len(passed_stage1),
                "local_pdf_texts": sum(
                    1
                    for article in prepared_stage1
                    if article.get("full_text_source") == "local_csv_pdf"
                ),
                "system_cascade_texts": sum(
                    1
                    for article in prepared_stage1
                    if article.get("full_text_source") not in {"abstract_proxy", "local_csv_pdf"}
                ),
                "stage4_included": selected_run["included"],
                "stage4_excluded": selected_run["excluded_count"],
                "stage4_text_avg_chars": avg_stage4_text_chars,
                "stage4_text_max_chars": max_stage4_text_chars,
                "stage4_text_limit_chars": int(getattr(config, "GOLD_EVAL_STAGE4_MAX_CHARS", 9000) or 9000),
            },
            "selected_stage4_model": selected_run["model"],
            "compare_stage4_models": compare_models,
            "model_config": {
                "stage4_model": selected_run["model"],
                "embedding_model": _embedding_model_label(),
                "text_source": selected_text_source,
            },
            "model_runs": [
                {
                    "model": run["model"],
                    "included": run["included"],
                    "excluded": run["excluded_count"],
                    "seconds": run["seconds"],
                    "avg_llm_seconds": run["llm_summary"]["average"],
                    "precision": run["metrics"]["precision"],
                    "recall": run["metrics"]["recall"],
                    "f1": run["metrics"]["f1"],
                    "accuracy": run["metrics"]["accuracy"],
                    "tp": run["metrics"]["tp"],
                    "fp": run["metrics"]["fp"],
                    "tn": run["metrics"]["tn"],
                    "fn": run["metrics"]["fn"],
                }
                for run in stage4_runs
            ],
            "metrics": metrics_rows,
            "final_metrics": selected_run["metrics"],
            "confusion": selected_run["metrics"],
            "concept_report": concept_report,
            "false_positives": false_positives[:25],
            "false_negatives": false_negatives[:25],
            "timings": timings,
            "timing_details": timing_details,
            "time_summary": {
                "total_seconds": total_seconds,
                "prep_seconds": prep_seconds,
                "llm_seconds": llm_seconds,
                "llm_share": round((llm_seconds / total_seconds) * 100, 1) if total_seconds else 0.0,
                "prep_share": round((prep_seconds / total_seconds) * 100, 1) if total_seconds else 0.0,
            },
            "llm_summary": llm_summary,
        }
        return templates.TemplateResponse("evaluation.html", {
            "request": request,
            "token": token,
            "stage4_model_choices": _stage4_model_choices(),
            "form": form_values,
            "preview": preview,
            "result": result,
            "error": None,
        })
    except ValueError as exc:
        return templates.TemplateResponse("evaluation.html", {
            "request": request,
            "token": token,
            "stage4_model_choices": _stage4_model_choices(),
            "form": form_values,
            "preview": None,
            "result": None,
            "error": str(exc),
        })
    except Exception as exc:
        logging.exception("[GoldEval] Error ejecutando evaluacion interna")
        return templates.TemplateResponse("evaluation.html", {
            "request": request,
            "token": token,
            "stage4_model_choices": _stage4_model_choices(),
            "form": form_values,
            "preview": None,
            "result": None,
            "error": f"No se pudo completar la evaluacion: {exc}",
        })

@app.post("/search", response_class=HTMLResponse)
async def initial_search(
    request: Request,
    background_tasks: BackgroundTasks,
    question: str = Form(...),
    client_id: str = Form(None),
    article_source_mode: str = Form("databases"),
    csv_file: Optional[UploadFile] = File(None),
):
    if client_id:
        client_id_var.set(client_id)
    start = time.perf_counter()
    logging.info(f"📝 Nueva Búsqueda: {question}")

    session_id = str(int(time.time()))
    eligibility_contract = generate_eligibility_contract(question)

    # Búsqueda Paralela de Dos Fases con TF-IDF Bootstrapping
    mode = str(article_source_mode or "databases").strip().lower()
    if mode not in {"databases", "csv_only", "databases_csv"}:
        mode = "databases"

    csv_articles: List[Dict[str, Any]] = []
    csv_import_report: Dict[str, Any] = {}
    if mode in {"csv_only", "databases_csv"} and csv_file and _normalise_uploaded_filename(csv_file):
        if not _normalise_uploaded_filename(csv_file).lower().endswith(".csv"):
            raise HTTPException(400, "Solo se aceptan archivos CSV.")
        try:
            csv_articles, csv_import_report = normalise_csv_articles(
                await csv_file.read(),
                source_name="CSV/Zotero",
                include_labels=False,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        csv_articles = [normalize_article_for_csv(article) for article in csv_articles]
        logging.info("[CSV Upload] %d articulos importados desde CSV", len(csv_articles))

    if mode in {"csv_only", "databases_csv"} and not csv_articles:
        raise HTTPException(400, "Seleccionaste CSV como fuente, pero no subiste un archivo valido.")

    search_articles: List[Dict[str, Any]] = []
    query_audit: List[Dict[str, Any]] = []
    adaptive_lexicon_report: Dict[str, Any] = {}
    seed_count = 0
    enriched_count = 0
    if mode != "csv_only":
        bootstrap_res = await two_phase_search(
            question,
            client_id=client_id_var.get(),
            eligibility_contract=eligibility_contract,
        )
        search_articles = list(bootstrap_res.corpus)
        query_audit = bootstrap_res.query_audit
        adaptive_lexicon_report = bootstrap_res.adaptive_lexicon_report
        seed_count = int(bootstrap_res.seed_count or 0)
        enriched_count = int(bootstrap_res.enriched_count or 0)

    articles = search_articles + csv_articles
    initial_source_total = len(articles)
    exact_duplicates_removed = 0
    if csv_articles and search_articles:
        before_dedup = len(articles)
        articles, _ = deduplication.remove_exact_duplicates(articles)
        exact_duplicates_removed = max(0, before_dedup - len(articles))
    if getattr(config, "ATOM_COVERAGE_RANKING_ENABLED", True):
        articles = rank_articles_by_contract(articles, eligibility_contract)
    source_counts = dict(Counter(a.get("source", "Otros") for a in articles))

    # Metadatos para auditoría
    raw_total = max(initial_source_total, seed_count + enriched_count + len(csv_articles))

    # Para el diagrama PRISMA necesitamos el total antes de deduplicar.
    # La recuperacion adaptativa puede ampliar el pool; no debe contarse como duplicado.
    duplicates_count = max(0, raw_total - len(articles), exact_duplicates_removed)

    unique_total = len(articles)
    concept_discarded = 0
    excluded_articles = []
    cp_report = {"total": unique_total, "passed": unique_total, "excluded": 0, "reduction_pct": 0}
    graph_recovery_report = {}

    # ============================================================
    # FILTRO DE PRESENCIA DE CONCEPTOS (Post-API, Pre-ChromaDB)
    # ============================================================
    try:
        synonym_data = contract_to_synonym_payload(eligibility_contract) or expand_query_with_synonyms(question)
        if isinstance(synonym_data, dict) and synonym_data.get("synonyms"):
            articles, excluded_articles, cp_report_obj = concept_presence_filter(
                articles,
                synonym_data=synonym_data
                # v29: Se omite min_concepts_required para usar la lógica proporcional (70% floor) interna
            )
            cp_report = asdict(cp_report_obj)
            concept_discarded = cp_report['excluded']
            unique_total = cp_report['total']
            min_recovery_candidates = int(getattr(config, "ADAPTIVE_GRAPH_RECOVERY_MIN_CANDIDATES", 30))
            allow_graph_recovery = mode != "csv_only"
            if len(articles) < min_recovery_candidates and excluded_articles and allow_graph_recovery:
                recovered_articles, graph_recovery_report = recover_articles_from_near_misses(
                    excluded_articles,
                    question,
                    eligibility_contract,
                )
                if recovered_articles:
                    recovered_articles = [normalize_article_for_csv(a) for a in recovered_articles if isinstance(a, dict)]
                    combined_articles, _ = deduplication.remove_exact_duplicates(articles + recovered_articles)
                    articles, recovered_excluded, recovered_report_obj = concept_presence_filter(
                        combined_articles,
                        synonym_data=synonym_data,
                    )
                    excluded_articles.extend(recovered_excluded)
                    cp_report = asdict(recovered_report_obj)
                    concept_discarded = cp_report["excluded"]
                    unique_total = cp_report["total"]
                    logging.info(
                        "[AdaptiveGraph] Recuperacion inicial: %d vecinos | %d candidatos tras prefiltro",
                        len(recovered_articles),
                        len(articles),
                    )
            elif len(articles) < min_recovery_candidates and excluded_articles:
                graph_recovery_report = {
                    "attempted": False,
                    "seed_count": 0,
                    "recovered_count": 0,
                    "reason": "csv_only_source_locked",
                }
                logging.info(
                    "[AdaptiveGraph] Omitido: modo Solo CSV mantiene el universo original de %d articulos",
                    cp_report.get("total", len(articles)),
                )
        else:
            excluded_articles = []
            concept_discarded = 0
            unique_total = len(articles)
            logging.info("ℹ️ [Concept Filter] Sin datos de sinónimos, omitiendo")
    except Exception as e:
        excluded_articles = []
        concept_discarded = 0
        unique_total = len(articles)
        logging.warning(f"⚠️ [Concept Filter] Error (no crítico): {e}")
    # ============================================================

    # Asegurar que articles es una lista plana de diccionarios
    articles = [a for a in articles if isinstance(a, dict)]
    articles = [normalize_article_for_csv(a) for a in articles]
    articles = _sort_by_citations(articles)

    # Crear carpeta de auditoría de sesión
    session_id = abs(hash(f"{question}_{time.time()}")) % (10 ** 8)
    audit_dir = Path("logs") / f"session_{session_id}"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # v17.2: Embedding de RAG aplazado hasta tener el ranking final.
    # background_tasks.add_task(database.save_to_milvus, articles)

    # Exportar Auditoría Inicial
    pd.DataFrame(articles).to_csv(audit_dir / "audit_0_inicial.csv", index=False)
    if excluded_articles:
        pd.DataFrame(excluded_articles).to_csv(audit_dir / "audit_1_excluidos_concepto.csv", index=False)
    if query_audit:
        pd.DataFrame(query_audit).to_csv(audit_dir / "audit_search_queries.csv", index=False)
    if csv_import_report:
        with open(audit_dir / "audit_csv_import.json", "w", encoding="utf-8") as f:
            pyjson.dump(csv_import_report, f, ensure_ascii=False, indent=2)

    # 🔥 v6: Evaluación Automática de Calidad (Initial Search)
    evaluate_results(articles)

    # 🔥 v9.0: Matriz de Extracción Adaptativa (Solo 1 vez por sesión)
    column_config = screening_ai.propose_columns_from_rq(question)

    t_search = time.perf_counter() - start

    concept_pool_total = int(cp_report.get("total", len(articles)) or len(articles))
    adaptive_graph_recovered_count = int(graph_recovery_report.get("recovered_count", 0) or 0)

    session_data = {
        "question": question,
        "articles": articles,
        "raw_count": raw_total,            # v16.5: Total antes de deduplicación
        "duplicates_count": duplicates_count,
        "concept_pool_count": concept_pool_total,
        "initial_source_count": initial_source_total,
        "search_time": t_search,
        "column_config": column_config,
        "excluded_articles": excluded_articles,  # v19.0: Para auditoría PRISMA
        "concept_report": cp_report if 'cp_report' in locals() else None,
        "eligibility_contract": eligibility_contract,
        "adaptive_lexicon_report": adaptive_lexicon_report,
        "adaptive_graph_report": graph_recovery_report,
        "article_source_mode": mode,
        "csv_import_report": csv_import_report,
        "csv_import_count": len(csv_articles),
        "source_counts": source_counts,
        "query_audit": query_audit,
        "default_sort": "citations_desc",
        "log_prefix": f"session_{session_id}"
    }
    save_session(session_id, session_data)

    years = [int(a.get('year', 0)) for a in articles if a.get('year')]
    y_min = min(years) if years else 2020
    y_max = max(years) if years else 2025

    lang_counts = Counter()
    lang_meta = {
        'en': {'name': 'English', 'flag': '🇬🇧'},
        'es': {'name': 'Español', 'flag': '🇪🇸'},
        'pt': {'name': 'Português', 'flag': '🇧🇷'},
    }
    for a in articles:
        detected = filters.detect_language(a)
        lang_counts[detected] += 1
    # Etiquetar artículos sin revista para mayor claridad
    journals = Counter([filters.get_journal_name(a) or 'Sin Revista / Otros' for a in articles])

    duplicates_removed = duplicates_count
    languages_list = [
        {"code": code, "name": lang_meta.get(code, {}).get('name', code),
         "flag": lang_meta.get(code, {}).get('flag', '🌐'), "count": count}
        for code, count in lang_counts.most_common()
    ]
    languages_list = filters.summarize_languages(articles)
    if client_id and client_id in progress_queues:
        progress_queues[client_id].put_nowait("END_STREAM")

    return templates.TemplateResponse("filters.html", {
        "request": request,
        "session_id": session_id,
        "question": question,
        "total": len(articles),
        "raw_total": raw_total,
        "duplicates_removed": duplicates_removed,
        "concept_pool_total": concept_pool_total,
        "adaptive_graph_recovered_count": adaptive_graph_recovered_count,
        "concept_discarded": concept_discarded,
        "excluded_articles": excluded_articles,
        "concept_report": cp_report if 'cp_report' in locals() else None,
        "year_min": y_min,
        "year_max": y_max,
        "journals": dict(journals.most_common()),
        "languages": languages_list,
        "all_journals": [{"name": k, "count": v} for k, v in journals.most_common()],
        "session_inclusion_criteria": session_data.get("last_inclusion_criteria", ""),
        "session_exclusion_criteria": session_data.get("last_exclusion_criteria", ""),
    })

@app.post("/update_filter_count", response_class=JSONResponse)
async def update_filter_count(request: Request, session_id: int = Form(...),
                            start_year: Optional[int] = Form(None),
                            end_year: Optional[int] = Form(None),
                            languages: Optional[str] = Form(None),
                            journals: Optional[str] = Form(None),
                            open_access: Optional[str] = Form(None),
                            academic_quality: Optional[str] = Form(None)):

    if not session_exists(session_id):
        return JSONResponse({"error": "Sesión expirada"}, 400)

    session_data = get_session(session_id)
    raw = session_data.get("articles", [])
    raw_total = session_data.get("raw_count", len(raw))
    dupes_removed = max(0, int(session_data.get("duplicates_count", raw_total - len(raw)) or 0))

    # Obtener reporte conceptual previo
    concept_report = session_data.get("concept_report", {})
    conceptual_excluded = concept_report.get("excluded", 0)

    # Parsear listas
    selected_langs = languages.split(',') if languages else []
    selected_journals = filters.parse_journal_filters(journals)
    selected_journal_keys = {filters.normalize_journal_name(j) for j in selected_journals}
    is_oa_only = open_access == 'true'
    is_academic_only = academic_quality == 'true'

    # Motor de Filtrado Multidimensional
    filtered = []
    excluidos_revista = []
    excluidos_anios = []
    excluidos_oa = []
    excluidos_idioma = []
    excluidos_revista_seleccionada = []
    journal_options_articles = []

    for a in raw:
        # --- 1. FILTRO DE CALIDAD ACADÉMICA (Prioridad Alta) ---
        # Si el usuario quiere solo revistas, descartamos pre-prints (Sin Revista / Año 0) AQUÍ.
        try:
            year = int(a.get('year', 0) or 0)
        except (TypeError, ValueError):
            year = 0
        journal_val = filters.get_journal_name(a)

        if is_academic_only:
            if not filters.has_academic_venue(a):
                excluidos_revista.append(a)
                continue

        if is_oa_only and not filters.is_truly_open_access(a):
            excluidos_oa.append(a)
            continue

        # --- 2. FILTRO DE IDIOMAS ---
        if selected_langs:
            detected_lang = filters.detect_language(a)
            if detected_lang not in selected_langs:
                excluidos_idioma.append(a)
                continue

        # --- 3. FILTRO DE AÑOS (Solo para los que pasaron Calidad) ---
        if start_year and year < start_year:
            excluidos_anios.append(a)
            continue
        if end_year and year > end_year:
            excluidos_anios.append(a)
            continue

        # --- 4. FILTRO DE REVISTAS ESPECÍFICAS ---
        journal_options_articles.append(a)
        if selected_journal_keys and filters.normalize_journal_name(journal_val) not in selected_journal_keys:
            excluidos_revista_seleccionada.append(a)
            continue

        filtered.append(a)

    unique_ids = set()
    unique_records = 0
    excluidos_duplicados_semanticos = []
    for a in filtered:
        key = a.get('doi') or a.get('title', '').lower()
        if key and key not in unique_ids:
            unique_ids.add(key)
            unique_records += 1
        else:
            excluidos_duplicados_semanticos.append(a)

    # IMPORTANTE: unique_records es lo que debe ir en la caja de FINALES
    logging.info(f"📊 [UpdateFilter] Raw: {raw_total} | Unique (Finales): {unique_records} | Metadatos Excluidos: {len(raw) - len(filtered)}")

    # Recalcular idiomas y revistas para la UI
    from collections import Counter
    lang_counts = Counter()
    for a in filtered:
        detected = filters.detect_language(a)
        lang_counts[detected] += 1

    lang_meta = {
        'en': {'name': 'English', 'flag': '🇬🇧'},
        'es': {'name': 'Español', 'flag': '🇪🇸'},
        'pt': {'name': 'Português', 'flag': '🇧🇷'},
    }
    languages_list = [
        {"code": code, "name": lang_meta.get(code, {}).get('name', code),
         "flag": lang_meta.get(code, {}).get('flag', '🌐'), "count": count}
        for code, count in lang_counts.most_common()
    ]

    languages_list = filters.summarize_languages(filtered)

    journals_counter = Counter([
        filters.get_journal_name(a) or 'Sin Revista / Otros'
        for a in journal_options_articles
    ])
    top_journals = [{"name": k, "count": v} for k, v in journals_counter.most_common()]
    total_journals_count = len(journals_counter)

    # --- PERSISTENCIA DE AUDITORÍA ---
    # Asegurar que la sesión existe en memoria
    if session_id not in session_data:
        session_data[session_id] = {
            'query': 'Restaurada tras reinicio',
            'timestamp': time.time(),
            'articles': []
        }

    # Guardar registros excluidos en la sesión para el ZIP
    session_data[session_id]['excluidos_anios'] = [a.get('title', 'Sin título') for a in excluidos_anios]
    session_data[session_id]['excluidos_revista'] = [a.get('title', 'Sin título') for a in excluidos_revista]
    session_data[session_id]['excluidos_oa'] = [a.get('title', 'Sin título') for a in excluidos_oa]
    session_data[session_id]['duplicates_list'] = excluidos_duplicados_semanticos

    # Guardar en archivo local
    import csv
    from pathlib import Path
    session_dir = Path("logs") / f"session_{session_id}"
    os.makedirs(session_dir, exist_ok=True)

    # CSV 2: Duplicados
    dupes_path = os.path.join(session_dir, "audit_excluidos_duplicados.csv")
    with open(dupes_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Título', 'Fuente', 'Año', 'Razón'])
        for art in excluidos_duplicados_semanticos:
            writer.writerow([art.get('title',''), art.get('source',''), art.get('year',0), 'Detección por Título/DOI'])

    # CSV 3: Excluidos por Revista
    journal_path = os.path.join(session_dir, "audit_excluidos_sin_revista.csv")
    with open(journal_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Título', 'Razón'])
        for art in excluidos_revista:
            writer.writerow([art.get('title',''), 'Calidad Académica (Sin Revista)'])

    # CSV 4: Excluidos por Años
    years_path = os.path.join(session_dir, "audit_excluidos_por_anio.csv")
    with open(years_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Título', 'Razón'])
        for art in excluidos_anios:
            writer.writerow([art.get('title',''), 'Fuera de Rango Cronológico'])

    # CSV 5: Excluidos por Idioma
    lang_path = os.path.join(session_dir, "audit_excluidos_por_idioma.csv")
    with open(lang_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Título', 'Idioma Detectado', 'Idiomas Permitidos'])
        for art in excluidos_idioma:
            writer.writerow([art.get('title',''), filters.detect_language(art), str(selected_langs)])

    # CSV 6: Excluidos por Selección Manual de Revistas
    manual_journal_path = os.path.join(session_dir, "audit_excluidos_revista_manual.csv")
    with open(manual_journal_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Título', 'Revista', 'Razón'])
        for art in excluidos_revista_seleccionada:
            writer.writerow([art.get('title',''), filters.get_journal_name(art) or 'Unknown', 'No seleccionada por el usuario'])

    # Guardar registros excluidos en la sesión para el ZIP
    session_data['excluidos_anios'] = [a.get('title', 'Sin título') for a in excluidos_anios]
    session_data['excluidos_revista'] = [a.get('title', 'Sin título') for a in excluidos_revista]
    session_data['excluidos_idioma'] = [a.get('title', 'Sin título') for a in excluidos_idioma]
    session_data['excluidos_revista_manual'] = [a.get('title', 'Sin título') for a in excluidos_revista_seleccionada]
    session_data['duplicates_list'] = excluidos_duplicados_semanticos
    session_data['excluidos_oa'] = [a.get('title', 'Sin titulo') for a in excluidos_oa]

    # Actualizar la sesión en memoria
    TEMP_ARTICLES[str(session_id)] = session_data

    # --- CÁLCULO ESTRICTO PARA DIAGRAMA PRISMA ---
    raw_total = session_data.get("raw_count", 0)
    concept_report = session_data.get("concept_report", {})

    # 1. Deduplicación (Sobre el bruto)
    concept_total = int(concept_report.get("total", len(raw)) or len(raw))
    dupes_removed = max(
        0,
        int(session_data.get("duplicates_count", raw_total - min(raw_total, concept_total)) or 0)
    )

    # 2. Exclusión Conceptual (Sobre los únicos)
    conceptual_excluded = concept_report.get("excluded", 0)

    # 3. Candidatos (Los que pasaron la IA)
    passed_concept_count = concept_report.get("passed", 0)

    # 4. Exclusión por Filtros de Metadatos (Sobre los que pasaron concepto)
    metadata_excluded = max(0, passed_concept_count - unique_records)

    # 5. Finales (Lo que realmente queda tras filtrar todo)
    final_count = unique_records
    session_data['last_filter_counts'] = {
        "raw_total": raw_total,
        "duplicates_count": dupes_removed,
        "concept_total": concept_total,
        "concept_discarded": conceptual_excluded,
        "candidates": passed_concept_count,
        "metadata_discarded": metadata_excluded,
        "final_count": final_count,
        "excluded_years": len(excluidos_anios),
        "excluded_open_access": len(excluidos_oa),
        "excluded_academic_quality": len(excluidos_revista),
        "excluded_languages": len(excluidos_idioma),
        "excluded_manual_journals": len(excluidos_revista_seleccionada),
    }

    # Definir directorio de auditoría
    audit_dir = Path("logs") / f"session_{session_id}"
    audit_dir.mkdir(parents=True, exist_ok=True)

    return JSONResponse({
        "raw_total": raw_total,
        "duplicates_count": dupes_removed,
        "unique_records": passed_concept_count,
        "concept_total": concept_total,
        "adaptive_graph_recovered_count": int(
            (session_data.get("adaptive_graph_report") or {}).get("recovered_count", 0) or 0
        ),
        "concept_discarded": conceptual_excluded,
        "metadata_discarded": metadata_excluded,
        "filtered_count": len(filtered),
        "final_count": final_count,
        "languages": languages_list,
        "all_journals": top_journals,
        "total_journals": total_journals_count,
        "audit_path": str(audit_dir)
    })

@app.get("/apply_filters")
async def apply_filters_get():
    return RedirectResponse(url="/")

# ==============================================================================
@app.post("/apply_filters")
def apply_filters(request: Request, background_tasks: BackgroundTasks, session_id: str = Form(...), question: str = Form(...),
                        start_year: int = Form(2000), end_year: int = Form(2025),
                        open_access: Optional[str] = Form("false"),
                        journals: Optional[str] = Form(None),
                        inclusion_criteria: Optional[str] = Form(""),
                        exclusion_criteria: Optional[str] = Form(""),
                        academic_quality: Optional[str] = Form("false"),
                        client_id: str = Form(None)):

    if client_id:
        client_id_var.set(client_id)

    if not session_exists(session_id):
        if client_id and client_id in progress_queues:
            progress_queues[client_id].put_nowait("END_STREAM")
        raise HTTPException(400, "Sesión expirada")

    data = get_session(session_id)
    articles = data["articles"]
    # ✅ Inyectar índice original para permitir actualizaciones en carga manual
    for i, art in enumerate(articles):
        art['original_index'] = i

    start_time = time.time()
    log_prefix = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(start_time)}"

    filtered = filters.apply_filters(
        articles,
        start_year=start_year,
        end_year=end_year,
        open_access=(open_access == "true"),
        journals=filters.parse_journal_filters(journals),
        academic_only=(academic_quality == "true")
    )
    # ✅ OPTIMIZACIÓN EXTREMA 3: Evitamos la deduplicación semántica (O(N²)) en la bolsa de 9,000 artículos.
    # Solo aplicamos deduplicación exacta aquí. La IA se encargará de descartar cualquier ruido después.
    unique, _ = deduplication.remove_exact_duplicates(filtered)
    unique = [normalize_article_for_csv(a) for a in unique]
    unique = sorted(
        unique,
        key=_citation_sort_key,
        reverse=True,
    )
    unique = _apply_optional_rank_limit(unique, CASCADE_MAX_ARTICLES)

    # ✅ OPTIMIZACIÓN: Evitar screening redundante de 3 minutos si la pregunta no ha cambiado
    existing_relevant = data.get("relevant_articles", [])
    last_question = data.get("last_screening_question", "")

    if existing_relevant and last_question == question:
        logging.info("♻️ Reutilizando screening previo para la misma pregunta...")
        candidates = existing_relevant
        query_for_screening = data.get("last_translated_question", question)
    else:
        logging.info("🌍 Preparando Query Semántica...")
        # v9.0: Traducción profesional con DeepL + Caché
        query_for_screening = translator.translate_to_english(question)
        data["last_translated_question"] = query_for_screening

        candidates, excluded = screening.screen_articles(
            unique, query_for_screening, max_results=0,
            original_question=question,
            inclusion_criteria=inclusion_criteria or "",
            exclusion_criteria=exclusion_criteria or ""
        )
        data["excluded_articles"] = excluded
        data["last_screening_question"] = question
        # Persistir criterios para poder detectar cambios futuros
        data["last_inclusion_criteria"] = inclusion_criteria or ""
        data["last_exclusion_criteria"] = exclusion_criteria or ""

    ranked_candidates = list(candidates)

    final_ranking = ranked_candidates

    # El ranking completo queda disponible para WSS@95 y auditoria de priorizacion.

    # Persistir todos los articulos del ranking final, sin recorte Top-N.
    background_tasks.add_task(database.save_to_milvus, final_ranking)

    data["relevant_articles"] = final_ranking

    # ✅ v11.1: Activar intención de descarga proactiva para artículos con URL
    for art in final_ranking:
        if art.get('url') and len(str(art.get('url'))) > 10:
            if not art.get('is_pdf_downloaded'):
                art['needs_pdf_download'] = True

    relevant = final_ranking

    # ✅ FUSIÓN v8.3: Sincronización proactiva de PDF con ChromaDB para evitar "SIN PDF" falso
    logging.info(f"🔍 Sincronizando estado PDF para {len(relevant)} artículos...")
    try:
        collection = database.ensure_collection()
        for art in relevant:
            if not art.get('is_pdf_downloaded'):
                # Check ChromaDB usando el hash estable
                import hashlib
                author = art.get("authors")[0] if art.get("authors") else art.get("author", "")
                unique_str = f"{art.get('title', '')}_{str(art.get('year',''))}_{author}"
                title_hash = hashlib.md5(unique_str.encode('utf-8', errors='ignore')).hexdigest()
                first_chunk_id = f"chunk_{title_hash}_0"

                try:
                    existing = collection.get(ids=[first_chunk_id])
                    if existing and existing['ids']:
                        # v11.3: Validación ROBUSTA (Longitud + Estructura) via database.is_pdf_real
                        document_text = existing['documents'][0] if existing['documents'] else ""
                        is_full = (existing['metadatas'][0].get('is_full_text') == "True" and
                                  database.is_pdf_real(document_text))

                        if is_full:
                            art['is_pdf_downloaded'] = True
                            art['needs_pdf_download'] = False
                            art['full_text_source'] = 'chromadb_recovery'
                        else:
                            # Es solo el abstract o un PDF insuficiente, necesitamos el real
                            art['is_pdf_downloaded'] = False
                            art['needs_pdf_download'] = True
                except:
                    pass
    except Exception as e:
        logging.warning(f"⚠️ Error sincronizando PDF: {e}")

    logging.info(f"🚀 Renderizando {len(relevant)} artículos (Prioridad URL aplicada).")

    # 🔥 v7: Evaluación Automática de Calidad (Ranking & Top-10)
    # Se evalúa sobre los 50 artículos finales que verá el usuario.
    evaluate_results(relevant)

    export_to_csv(relevant, f"{log_prefix}_log_3_FINAL_70percent.csv")

    # v12: Cascade Screening automático en background (no bloqueante)
    # La descarga de PDFs se realiza dentro de _run_cascade_background
    # solo para los artículos que pasen el Stage 4 (LLM Juez)
    sessions.setdefault(str(session_id), data)
    sessions[str(session_id)]["cascade_status"] = "running_s1"
    threading.Thread(
        target=_run_cascade_background,
        args=(
            [a.copy() for a in relevant],
            question,
            data.get("last_translated_question", question),
            inclusion_criteria or "",
            exclusion_criteria or "",
            str(session_id),
            sessions,
        ),
        daemon=True,
    ).start()

    data.update({
        "relevant_articles": relevant,
        "dedup_articles": unique,
        "log_prefix": log_prefix
    })

    # Normalizar con campos dinámicos para el export inicial
    ai_keys = [col['key'] for col in data.get("column_config", {}).get("columnas", [])]
    relevant = [normalize_article_for_csv(a, ai_keys) for a in relevant]

    return templates.TemplateResponse("screening.html", {
        "request": request,
        "session_id": session_id,
        "question": question,
        "articles": relevant,
        "column_config": data.get("column_config") # Enviar config al frontend
    })


@app.post("/run_cascade_screening")
async def run_cascade_screening(
    request: Request,
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    question: str = Form(...),
    start_year: int = Form(2000),
    end_year: int = Form(2025),
    open_access: Optional[str] = Form("false"),
    journals: Optional[str] = Form(None),
    inclusion_criteria: Optional[str] = Form(""),
    exclusion_criteria: Optional[str] = Form(""),
    academic_quality: Optional[str] = Form("false"),
    cascade_limit: int = Form(0),
    stage1_threshold: str = Form("0.70"),
    client_id: str = Form(None)
):
    if client_id:
        client_id_var.set(client_id)

    if not session_exists(session_id):
        if client_id and client_id in progress_queues:
            progress_queues[client_id].put_nowait("END_STREAM")
        raise HTTPException(400, "Sesión expirada")

    data = get_session(session_id)
    articles = data["articles"]

    # Inyectar índice original para carga manual
    for i, art in enumerate(articles):
        art['original_index'] = i

    start_time = time.time()
    log_prefix = f"cascade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(start_time)}"

    if client_id and client_id in progress_queues:
        progress_queues[client_id].put_nowait("⚡ Iniciando Pipeline de Cribado en Cascada con prefiltro PICO...")

    # --- 1. APLICAR FILTROS DE METADATOS Y DEDUPLICACIÓN EXACTA ---
    if client_id and client_id in progress_queues:
        progress_queues[client_id].put_nowait("🔍 [Etapa 0] Aplicando filtros de metadatos y deduplicación exacta...")

    filtered = filters.apply_filters(
        articles,
        start_year=start_year,
        end_year=end_year,
        open_access=(open_access == "true"),
        journals=filters.parse_journal_filters(journals),
        academic_only=(academic_quality == "true")
    )

    unique, _ = deduplication.remove_exact_duplicates(filtered)
    unique = [normalize_article_for_csv(a) for a in unique]
    metadata_candidates = len(unique)
    excluded_stage0_concept: List[Dict] = []
    cascade_concept_report = None
    eligibility_contract = data.get("eligibility_contract") or generate_eligibility_contract(
        question,
        inclusion_criteria or "",
        exclusion_criteria or "",
    )
    try:
        synonym_data = contract_to_synonym_payload(eligibility_contract) or expand_query_with_synonyms(question)
        unique, excluded_stage0_concept, cascade_concept_report_obj = concept_presence_filter(
            unique,
            synonym_data=synonym_data,
        )
        cascade_concept_report = asdict(cascade_concept_report_obj)
        logging.info(
            "[Cascade] Prefiltro PICO estricto: %d candidatos | %d excluidos",
            len(unique),
            len(excluded_stage0_concept),
        )
        if client_id and client_id in progress_queues:
            progress_queues[client_id].put_nowait(
                f"[Etapa 0] Prefiltro PICO: {len(unique)} candidatos; {len(excluded_stage0_concept)} excluidos."
            )
        min_recovery_candidates = int(getattr(config, "ADAPTIVE_GRAPH_RECOVERY_MIN_CANDIDATES", 30))
        if len(unique) < min_recovery_candidates and excluded_stage0_concept:
            recovered_articles, graph_report = recover_articles_from_near_misses(
                excluded_stage0_concept,
                question,
                eligibility_contract,
            )
            data["adaptive_graph_report"] = graph_report
            if recovered_articles:
                recovered_articles = [normalize_article_for_csv(a) for a in recovered_articles if isinstance(a, dict)]
                recovered_articles = filters.apply_filters(
                    recovered_articles,
                    start_year=start_year,
                    end_year=end_year,
                    open_access=(open_access == "true"),
                    journals=filters.parse_journal_filters(journals),
                    academic_only=(academic_quality == "true"),
                )
                combined_unique, _ = deduplication.remove_exact_duplicates(unique + recovered_articles)
                unique, recovered_excluded, recovered_report_obj = concept_presence_filter(
                    combined_unique,
                    synonym_data=synonym_data,
                )
                excluded_stage0_concept.extend(recovered_excluded)
                cascade_concept_report = asdict(recovered_report_obj)
                logging.info(
                    "[AdaptiveGraph] Recuperacion por grafo: %d vecinos | %d candidatos tras prefiltro",
                    len(recovered_articles),
                    len(unique),
                )
                if client_id and client_id in progress_queues:
                    progress_queues[client_id].put_nowait(
                        f"[Etapa 0] Snowballing: {len(recovered_articles)} vecinos; {len(unique)} candidatos."
                    )
    except Exception as e:
        logging.warning("[Cascade] Prefiltro PICO omitido por error: %s", e)

    candidates_before_cap = len(unique)
    effective_cascade_limit = int(cascade_limit or 0)
    if getattr(config, "ATOM_COVERAGE_RANKING_ENABLED", True):
        unique = rank_articles_by_contract(unique, eligibility_contract)
    unique = sorted(
        unique,
        key=_cascade_rank_key,
        reverse=True,
    )
    if effective_cascade_limit > 0:
        unique = unique[:effective_cascade_limit]
    if candidates_before_cap > len(unique):
        logging.warning(
            "[Cascade] Se evaluaran %d/%d candidatos por limite de cascada=%d",
            len(unique),
            candidates_before_cap,
            effective_cascade_limit,
        )
        if client_id and client_id in progress_queues:
            progress_queues[client_id].put_nowait(
                f"[Aviso] La cascada evaluara {len(unique)} de {candidates_before_cap} candidatos mejor rankeados."
            )

    # --- 2. STAGE 1: Filtro rápido por similitud coseno del Abstract ---
    threshold = min(1.0, max(0.0, _safe_float(stage1_threshold, 0.70)))
    if client_id and client_id in progress_queues:
        progress_queues[client_id].put_nowait(f"⚡ [Stage 1] Ejecutando similitud coseno de abstracts contra la RQ (Umbral: {threshold})...")

    from app.screening.fast_filter import apply_stage1_fast_filter
    passed_stage1, excluded_stage1 = apply_stage1_fast_filter(
        unique,
        question,
        threshold=threshold
    )

    if client_id and client_id in progress_queues:
        progress_queues[client_id].put_nowait(f"📊 [Stage 1] Pasaron: {len(passed_stage1)} | Excluidos: {len(excluded_stage1)}")

    ABSTRACT_MIN_CHARS = int(getattr(config, "ABSTRACT_MIN_CHARS", "800"))
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from app.extraction.pdf_extractor import acquire_full_text

    # --- 2.5: Preparar textos para cribado por abstract (Evitar descargas previas) ---
    for art in passed_stage1:
        art["full_text"] = (art.get("abstract") or "").strip()
        art["full_text_source"] = "abstract_proxy"
    stage4_input = passed_stage1

    # --- 3. STAGE 4: Cribado LLM Profundo ---
    if client_id and client_id in progress_queues:
        progress_queues[client_id].put_nowait(
            f"🤖 [Stage 4] Cribado LLM sobre {len(stage4_input)} candidatos por Abstract..."
        )

    from app.screening.deep_screener import screen_candidates_cascade
    from app.extraction.pdf_extractor import extract_selective_sections_from_text
    passed_stage4, excluded_stage4 = screen_candidates_cascade(
        stage4_input,
        question,
        inclusion_criteria=inclusion_criteria or "",
        exclusion_criteria=exclusion_criteria or "",
        eligibility_contract=eligibility_contract,
    )

    # --- 3.5: Descargar PDFs completos + extraer secciones SOLO de incluidos ---
    if passed_stage4:
        if client_id and client_id in progress_queues:
            progress_queues[client_id].put_nowait(
                f"📥 Descargando PDFs completos para {len(passed_stage4)} artículos INCLUIDOS..."
            )
        with ThreadPoolExecutor(max_workers=4) as pool:
            fut_map = {pool.submit(acquire_full_text, art, force=True): art for art in passed_stage4}
            for future in as_completed(fut_map):
                art = fut_map[future]
                res = future.result()
                if isinstance(res, dict):
                    art.update(res)

        for art in passed_stage4:
            if art.get("is_pdf_downloaded"):
                art["full_text"] = extract_selective_sections_from_text(
                    art.get("full_text", "") or ""
                )
                art["full_text_source"] = "pdf_cascade_extracted"
            else:
                art["full_text"] = (art.get("abstract") or "").strip()
                art["full_text_source"] = "abstract_proxy"

    if client_id and client_id in progress_queues:
        progress_queues[client_id].put_nowait(f"🏁 [Stage 4] Cribado profundo finalizado. Incluidos: {len(passed_stage4)} | Excluidos: {len(excluded_stage4)}")

    # Guardar en base de datos de manera asíncrona para no bloquear
    background_tasks.add_task(database.save_to_milvus, passed_stage4)
    background_stage4 = [
        article for article in excluded_stage4
        if (article.get("deep_screening_details") or {}).get("decision") == "BACKGROUND_ONLY"
    ]

    # Actualizar sesión con los resultados
    data["relevant_articles"] = passed_stage4
    data["background_articles"] = background_stage4
    data["excluded_articles"] = excluded_stage4 + excluded_stage1 + excluded_stage0_concept
    data["cascade_stage1_excluded"] = len(excluded_stage1)
    data["cascade_stage4_excluded"] = len(excluded_stage4)
    data["cascade_stage4_included"] = len(passed_stage4)
    data["cascade_concept_excluded"] = len(excluded_stage0_concept)
    data["last_screening_question"] = question
    data["last_inclusion_criteria"] = inclusion_criteria or ""
    data["last_exclusion_criteria"] = exclusion_criteria or ""
    data["dedup_articles"] = unique
    data["cascade_metadata_candidates"] = metadata_candidates
    data["cascade_candidates_before_cap"] = candidates_before_cap
    data["cascade_candidates_evaluated"] = len(unique)
    data["cascade_concept_report"] = cascade_concept_report
    data["cascade_concept_excluded"] = excluded_stage0_concept
    data["eligibility_contract"] = eligibility_contract
    data["log_prefix"] = log_prefix

    # Crear carpeta de auditoría de sesión
    session_dir = Path("logs") / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    # Exportar CSV de auditoría para Stage 1 y Stage 4
    if excluded_stage0_concept:
        pd.DataFrame(excluded_stage0_concept).to_csv(session_dir / "audit_stage0_excluidos_pico.csv", index=False)
    if excluded_stage1:
        pd.DataFrame(excluded_stage1).to_csv(session_dir / "audit_stage1_excluidos_similitud.csv", index=False)
    if excluded_stage4:
        pd.DataFrame(excluded_stage4).to_csv(session_dir / "audit_stage4_excluidos_llm.csv", index=False)
    if background_stage4:
        pd.DataFrame(background_stage4).to_csv(session_dir / "audit_stage4_background_only.csv", index=False)
    if passed_stage4:
        pd.DataFrame(passed_stage4).to_csv(session_dir / "audit_stage4_incluidos.csv", index=False)

    save_session(session_id, data)

    if client_id and client_id in progress_queues:
        progress_queues[client_id].put_nowait("END_STREAM")

    return templates.TemplateResponse("screening.html", {
        "request": request,
        "session_id": session_id,
        "question": question,
        "articles": passed_stage4,
        "column_config": data.get("column_config")
    })


@app.post("/translate_abstract")
async def translate_abstract_endpoint(request: Request):
    data = await request.json()
    try:
        translation = screening_ai.translate_abstract_to_spanish(data.get("abstract", ""))
        return JSONResponse({"translation": translation})
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)

@app.post("/generate_column")
async def generate_column_endpoint(request: Request):
    req_data = await request.json()
    try:
        article = req_data.get('article')
        column_key = req_data.get('column_name')
        session_id = str(req_data.get('session_id', ''))
        article_index = req_data.get('article_index', -1)  # v10.8 fix
        question = req_data.get('question', '')

        if not session_id or not session_exists(session_id):
            logging.warning(f"⚠️ Sesión no encontrada: '{session_id}'")
            return JSONResponse({"error": "Sesión expirada o no encontrada"}, 400)

        session_data = get_session(session_id)
        # Si la pregunta no viene en la request, usar la de la sesión
        if not question:
            question = session_data.get("question", "")

        column_config = session_data.get("column_config", {})
        config_hash = column_config.get("config_hash", "default")

        # v9.8: Fortalecer la recuperación del artículo (Session-Sync)
        # Si el frontend envía un objeto parcial, buscamos el completo en la sesión
        title = article.get('title', '')
        session_articles = session_data.get("relevant_articles", [])
        full_article = next((a for a in session_articles if a.get('title') == title), article)

        # Sincronizar campos críticos de vuelta al objeto de trabajo
        for key in ['full_text', 'is_pdf_downloaded', 'needs_pdf_download', 'full_text_source', 'requires_pdf_verification', 'criteria_breakdown']:
            if full_article.get(key) and not article.get(key):
                article[key] = full_article[key]

        # Buscar la instrucción específica para esta columna
        col_item = next((c for c in column_config.get("columnas", []) if c['key'] == column_key), None)
        if not col_item:
            logging.error(f"❌ Columna '{column_key}' no existe en la config de la sesión {session_id}")
            return JSONResponse({"error": f"Columna {column_key} no configurada"}, 400)

        # v16.4: Bloquear JIT si el artículo requiere PDF manual (no descarga automática)
        if not article.get('is_pdf_downloaded') and article.get('needs_pdf_download'):
            logging.info(f"⏸️ Artículo pendiente de PDF manual, omitiendo análisis: {article.get('title', '')[:40]}")
            return JSONResponse({
                "value": "<div class='text-amber-600 text-xs'>⬆️ Sube el PDF manualmente para analizar</div>",
                "column_name": column_key,
                "is_pdf_downloaded": False,
                "pdf_pending": True
            })

        # v11.3: Recuperación Robusta + JIT Trigger (Heurística Estructural)
        recovered = False
        if article.get('is_pdf_downloaded'):
            current_text = str(article.get('full_text', ''))
            if not database.is_pdf_real(current_text):
                logging.info(f"🧩 Texto insuficiente o sin estructura ({len(current_text)} chars), intentando recuperación desde ChromaDB...")
                recovered_text = database.recover_full_text(article)
                if recovered_text and database.is_pdf_real(recovered_text):
                    article['full_text'] = recovered_text
                    article['full_text_source'] = 'chromadb_recovery'
                    recovered = True
                else:
                    recovered = False

        # v11.3: Descarga JIT si no hay texto suficiente (Heurística Estructural)
        # v16.4: Solo ejecutar si el artículo NO requiere PDF manual (needs_pdf_download=False/None)
        if not recovered and not article.get('needs_pdf_download'):
            current_text = str(article.get('full_text', ''))
            if not database.is_pdf_real(current_text):
                logging.info(f"📥 Descarga JIT de emergencia (Falta estructura o longitud) para: {article.get('title')[:30]}...")
                article = pdf_extractor.download_full_text_lazy(article, force=True)

                # v11.3: Persistencia inmediata en ChromaDB si cumple la heurística
                new_text = str(article.get('full_text', ''))
                if article.get('is_pdf_downloaded') and database.is_pdf_real(new_text):
                    try:
                        database.save_to_milvus([article])
                        logging.info(f"💾 PDF indexado permanentemente en ChromaDB: {article.get('title')[:30]}")
                    except Exception as e:
                        logging.error(f"⚠️ Error indexando descarga JIT: {e}")

                # Actualizar en la sesión para persistir la descarga
                if article_index >= 0:
                    session_data['articles'][article_index] = article

        # v16.5: Guardia final — si después de JIT todavía no hay PDF real, bloquear análisis
        has_real_pdf = article.get('is_pdf_downloaded') and database.is_pdf_real(str(article.get('full_text', '')))
        if not has_real_pdf:
            logging.info(f"⛔ Sin PDF real tras todos los intentos, bloqueando análisis: {article.get('title', '')[:40]}")
            # Marcar en sesión para que el frontend no vuelva a intentarlo
            if article_index >= 0:
                session_data['articles'][article_index]['needs_pdf_download'] = True
            return JSONResponse({
                "value": "<div class='text-amber-600 text-xs p-2 bg-amber-50 border border-amber-200 rounded'><i class='fas fa-upload mr-1'></i> PDF no disponible automáticamente — súbelo manualmente para analizar</div>",
                "column_name": column_key,
                "is_pdf_downloaded": False,
                "pdf_pending": True
            })

        async with AI_SEMAPHORE:
            await asyncio.sleep(0.5)
            result = screening_ai._generate_columns_for_article(
                article,
                [col_item],
                research_question=question,
                config_hash=config_hash
            )

        val = result.get(column_key, "⚠️ No extraído")
        return JSONResponse({
            "value": val,
            "column_name": column_key,
            "is_pdf_downloaded": article.get('is_pdf_downloaded', False)
        })

    except Exception as e:
        logging.error(f"❌ Error crítico en generate_column: {e}")
        return JSONResponse({"error": str(e), "value": "Error interno"}, 500)

@app.post("/upload_pdf")
async def upload_pdf_endpoint(
    sessionId: str = Form(...),
    articleIndex: int = Form(...),
    file: UploadFile = File(...)
):
    """
    Endpoint para carga manual de PDFs de contingencia.
    ✅ Extrae texto, actualiza sesión y sincroniza ChromaDB.
    """
    try:
        if not session_exists(sessionId):
            return JSONResponse({"error": "Sesión no encontrada"}, 404)

        session_data = get_session(sessionId)
        articles = session_data.get("articles", [])

        if articleIndex < 0 or articleIndex >= len(articles):
            return JSONResponse({"error": "Índice de artículo inválido"}, 400)

        # 1. Leer y extraer texto del PDF
        # validated with PDF magic number before extraction
        content = await file.read()
        if not content or not content.startswith(b"%PDF-"):
            return JSONResponse({"error": "El archivo subido no parece un PDF valido."}, 400)
        text = pdf_extractor.extract_text_with_timeout(content, timeout_seconds=20)
        if len(text) < 200:
            text = pdf_extractor.extract_selective_sections_with_timeout(content, timeout_seconds=20)

        if len(text) < 200:
            return JSONResponse({"error": "El PDF parece estar vacío o ser una imagen. No se pudo extraer texto."}, 400)

        # 2. Actualizar metadata del artículo
        article = articles[articleIndex]
        article['full_text'] = text[:80000] # Límite de seguridad
        article['is_pdf_downloaded'] = True
        article['needs_pdf_download'] = False
        article['full_text_source'] = 'manual_upload'
        _enrich_pdf_status([article])

        # 3. Sincronizar con ChromaDB (Chunking automático)
        logging.info(f" ⬆️ Carga manual: Sincronizando '{article.get('title')[:30]}...' con ChromaDB")
        database.save_to_milvus([article])
        _sync_uploaded_pdf_to_included(session_data, article)

        # 4. Guardar sesión
        save_session(sessionId, session_data)

        return JSONResponse({
            "success": True,
            "message": "PDF procesado y sincronizado correctamente",
            "pdf_audit": _enrich_pdf_status(session_data.get("included_articles", [])),
            "article": article
        })

    except Exception as e:
        logging.error(f"❌ Error en upload_pdf: {e}")
        return JSONResponse({"error": str(e)}, 500)

@app.post("/submit_screening")
async def submit_screening(request: Request):
    data = await request.json()
    session_id = int(data.get("sessionId"))
    client_id = data.get("client_id")

    if client_id:
        client_id_var.set(client_id)

    if not session_exists(session_id):
        logging.error(f"❌ Sesión {session_id} no encontrada")
        if client_id and client_id in progress_queues:
            progress_queues[client_id].put_nowait("END_STREAM")
        return JSONResponse({"error": "Sesión expirada o no válida"}, 400)

    session_data = get_session(session_id)
    log_prefix = session_data.get("log_prefix", "session")

    # ✅ FUSIÓN CRÍTICA: Recuperar evidencia descargada en segundo plano
    original_articles = session_data.get("articles", [])
    original_map = {a.get('original_index'): a for a in original_articles if 'original_index' in a}

    included, excluded = [], []
    for _, item in data.get("articles", {}).items():
        art = item['data']

        # Recuperar evidencia técnica si el fondo la descargó mientras el usuario cribaba
        orig_idx = art.get('original_index')
        if orig_idx is not None and orig_idx in original_map:
            original = original_map[orig_idx]
            if original.get('is_pdf_downloaded') and not art.get('is_pdf_downloaded'):
                art['full_text'] = original.get('full_text')
                art['is_pdf_downloaded'] = True
                art['needs_pdf_download'] = False
                art['full_text_source'] = original.get('full_text_source')

        ai_fields = item.get('aiGeneratedFields', {})
        art.update(ai_fields)

        art['researcher_notes'] = item.get('notes', '')
        art['translation'] = item.get('translation', '')

        art = normalize_article_for_csv(art)
        art.setdefault('similarity', 0.0)  # v16.6: garantizar campo para Jinja2


        if item['status'] == 'included':
            # Asegurar que preservamos el índice original si existe
            if 'original_index' in item['data']:
                art['original_index'] = item['data']['original_index']
            included.append(art)
        else:
            art['exclusion_reason'] = item.get('exclusionReason', '')
            excluded.append(art)

    export_to_csv(excluded, f"{log_prefix}_log_5_excluded.csv")
    export_to_csv(included, f"{log_prefix}_log_6_included.csv")

    session_data["included_articles"] = included
    session_data["screening_excluded_articles"] = excluded
    session_data["screening_included_count"] = len(included)
    session_data["screening_excluded_count"] = len(excluded)
    pdf_audit = _sync_included_pdf_state(session_data)
    included = session_data.get("included_articles", included)
    save_session(session_id, session_data)

    avg_sim = sum(a.get('similarity', 0) for a in included) / len(included) if included else 0
    journals = set(a.get('journal') for a in included)

    if client_id and client_id in progress_queues:
        progress_queues[client_id].put_nowait("END_STREAM")

    included_for_view = [normalize_article_for_csv(a) for a in included]

    return templates.TemplateResponse("review.html", {
        "request": request,
        "session_id": session_id,
        "included_articles": included_for_view,
        "total_included": len(included),
        "total_excluded": len(excluded),
        "avg_similarity": f"{avg_sim*100:.1f}",
        "unique_journals": len(journals),
        "pdf_audit": pdf_audit,
        "question": data.get("question")
    })

# ==============================================================================
# 🚀 ENDPOINT CORREGIDO (SIN BLOQUEO DEL SERVIDOR)
# ==============================================================================
@app.post("/generate_synthesis")
async def generate_synthesis_endpoint(request: Request):
    """
    Genera síntesis académica completa (PDF + Resumen Web).
    CORREGIDO: Se ejecuta en un hilo secundario para no congelar FastAPI.
    """
    data = await request.json()
    session_id = int(data.get("sessionId"))
    question = data.get("question")
    client_id = data.get("client_id")
    allow_missing_pdfs = bool(data.get("allow_missing_pdfs"))

    if client_id:
        client_id_var.set(client_id)

    # 1. Validaciones
    if not session_exists(session_id):
        if client_id and client_id in progress_queues:
            progress_queues[client_id].put_nowait("END_STREAM")
        return JSONResponse({"error": "Sesión no válida"}, 400)

    session_data = get_session(session_id)
    pdf_audit = _sync_included_pdf_state(session_data)
    included = session_data.get("included_articles", [])
    if pdf_audit.get("changed"):
        save_session(session_id, session_data)

    if not included:
        return JSONResponse({"error": "No hay artículos seleccionados."}, 400)

    # 2. Calcular métricas (incluyendo search_queries reales y conteo por fuente)
    if pdf_audit.get("missing", 0) > 0 and not allow_missing_pdfs:
        if client_id and client_id in progress_queues:
            progress_queues[client_id].put_nowait("END_STREAM")
        return JSONResponse({
            "error": (
                f"Faltan {pdf_audit.get('missing', 0)} PDFs completos. "
                "Sube los archivos o confirma que deseas continuar con abstracts."
            ),
            "pdf_audit": pdf_audit
        }, 409)
    if allow_missing_pdfs:
        for article in included:
            if not _article_has_real_pdf(article):
                article["needs_pdf_download"] = False

    def _count_by_source(articles: list) -> dict:
        """Cuenta artículos por fuente (BD)."""
        counts = {}
        for art in articles:
            src = art.get('source', 'Otra')
            counts[src] = counts.get(src, 0) + 1
        return counts

    filter_counts = session_data.get("last_filter_counts", {}) or {}
    if not filter_counts:
        concept_report = session_data.get("concept_report") or {}
        cascade_report = session_data.get("cascade_concept_report") or {}
        concept_total = int(cascade_report.get("total", concept_report.get("total", len(session_data.get("articles", [])))) or 0)
        concept_discarded = int(cascade_report.get("excluded", concept_report.get("excluded", 0)) or 0)
        filter_counts = {
            "raw_total": int(session_data.get("raw_count", len(session_data.get("articles", []))) or 0),
            "duplicates_count": int(session_data.get("duplicates_count", 0) or 0),
            "concept_total": concept_total,
            "concept_discarded": concept_discarded,
            "candidates": int(cascade_report.get("passed", concept_report.get("passed", len(session_data.get("relevant_articles", [])))) or 0),
            "metadata_discarded": 0,
            "final_count": len(included),
        }

    metrics = {
        "total": len(session_data.get("articles", [])),
        "after_filter": len(session_data.get("dedup_articles", [])),
        "after_dedup": len(session_data.get("dedup_articles", [])),
        "relevant": len(session_data.get("relevant_articles", [])),
        "final_included": len(included),
        "screening_excluded": int(session_data.get("screening_excluded_count", 0) or 0),
        "search_queries": session_data.get("search_queries", {}),
        "query_audit": session_data.get("query_audit", []),
        "pdf_audit": pdf_audit,
        "filter_counts": filter_counts,
        "inclusion_criteria_text": session_data.get("last_inclusion_criteria", ""),
        "exclusion_criteria_text": session_data.get("last_exclusion_criteria", ""),
        "article_source_mode": session_data.get("article_source_mode", ""),
        "csv_import_count": int(session_data.get("csv_import_count", 0) or 0),
        "embedding_model": "Ollama: nomic-embed-text",
        "source_counts": _count_by_source(session_data.get("articles", [])),  # 🔥 NUEVO: Conteo por BD
    }

    logging.info(f"🧪 Iniciando síntesis BACKGROUND para Sesión {session_id}...")

    logging.info(
        "PDF audit antes de sintesis: %s/%s completos; %s pendientes; %s subidos manualmente.",
        pdf_audit.get("ready", 0),
        pdf_audit.get("total", 0),
        pdf_audit.get("missing", 0),
        pdf_audit.get("uploaded", 0),
    )

    # ==========================================================================
    # ⚡ MAGIA ASÍNCRONA: Wrapper para tarea pesada
    # ==========================================================================
    loop = asyncio.get_event_loop()

    def heavy_task_wrapper():
        """Función síncrona que contiene todo el trabajo pesado de CPU/IA"""
        # A. Generar síntesis (IA)
        logging.info("   --> Ejecutando IA (esto puede tardar)...")
        s_data = synthesis.generate_synthesis_full(
            articles=included,
            question=question,
            metrics=metrics
        )

        # B. Generar PDF
        logging.info("   --> Renderizando PDF...")
        log_prefix = session_data.get("log_prefix", "session")
        pdf_fname = f"{log_prefix}_REPORTE_PRISMA.pdf"
        pdf_fpath = f"static/{pdf_fname}"

        p_path = create_pdf_report(
            synthesis_data=s_data,
            metrics=metrics,
            articles=included,
            question=question,
            pdf_path=pdf_fpath
        )

        return s_data, p_path, pdf_fname

    # ==========================================================================
    # ⚡ EJECUCIÓN: Mandar al ThreadPool y esperar sin bloquear
    # ==========================================================================
    try:
        # Aquí FastAPI libera el servidor mientras 'heavy_task_wrapper' corre en otro hilo
        synthesis_data, pdf_path, pdf_filename = await loop.run_in_executor(None, heavy_task_wrapper)

        # Verificar éxito de la síntesis
        if not synthesis_data.get('metadata'):
            raise Exception("La IA no generó metadata válida.")

        # Guardar ruta del PDF en la sesión
        if pdf_path:
            session_data['pdf_path'] = pdf_path
            save_session(session_id, session_data)
            logging.info(f"✅ Tarea completada. PDF: {pdf_path}")
        else:
            logging.warning("⚠️ La tarea terminó pero no se generó PDF.")

    except Exception as e:
        logging.error(f"❌ Error en proceso background: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return JSONResponse({"error": f"Error generando análisis: {str(e)}"}, 500)

    # ===== 5. EXPORTAR CSV (Operación rápida) =====
    included_normalized = included
    try:
        included_normalized = [normalize_article_for_csv(a) for a in included]
        export_to_csv(included_normalized, f"{session_data.get('log_prefix')}_TABLA_FINAL.csv")
    except Exception as e:
        logging.warning(f"⚠️ No se pudo exportar CSV: {e}")

    # ===== 6. PREPARAR RESPUESTA =====
    metrics_data = {
        **metrics,
        "t_search": round(session_data.get("search_time", 0), 2),
        "total_time": 15.0 # Estimado
    }

    synth_brief = synthesis_data['metadata'].get('resumen', "Resumen no disponible.")
    pdf_exists = pdf_path and Path(pdf_path).exists()

    # 🔥 v6: Evaluación Automática de Calidad (Post-Screening)
    try:
        evaluate_results(included)
    except Exception as e:
        logging.warning(f"⚠️ Evaluación post-screening falló (no fatal): {e}")

    if client_id and client_id in progress_queues:
        progress_queues[client_id].put_nowait("END_STREAM")

    return templates.TemplateResponse("results.html", {
        "request": request,
        "synthesis": synth_brief,
        "metrics": metrics_data,
        "final_articles": included_normalized,
        "plots": {"prisma": ""},
        "session_id": session_id,
        "question": question,
        "pdf_available": pdf_exists,
        "pdf_filename": pdf_filename if pdf_exists else ""
    })

def _compute_prisma_filter_counts(
    articles: List[Dict],
    raw_total: int,
    concept_report: Dict,
    start_year: Optional[int],
    end_year: Optional[int],
    languages: Optional[str],
    journals: Optional[str],
    open_access: Optional[str],
    academic_quality: Optional[str],
) -> Dict[str, int]:
    selected_langs = languages.split(',') if languages else []
    selected_journals = filters.parse_journal_filters(journals)
    selected_journal_keys = {filters.normalize_journal_name(j) for j in selected_journals}
    is_oa_only = open_access == 'true'
    is_academic_only = academic_quality == 'true'

    filtered: List[Dict] = []
    excluded_years = 0
    excluded_open_access = 0
    excluded_academic_quality = 0
    excluded_languages = 0
    excluded_manual_journals = 0

    for article in articles:
        try:
            year = int(article.get('year', 0) or 0)
        except (TypeError, ValueError):
            year = 0

        if is_academic_only and not filters.has_academic_venue(article):
            excluded_academic_quality += 1
            continue

        if is_oa_only and not filters.is_truly_open_access(article):
            excluded_open_access += 1
            continue

        if selected_langs and filters.detect_language(article) not in selected_langs:
            excluded_languages += 1
            continue

        if start_year and year < start_year:
            excluded_years += 1
            continue
        if end_year and year > end_year:
            excluded_years += 1
            continue

        journal_name = filters.get_journal_name(article)
        if selected_journal_keys and filters.normalize_journal_name(journal_name) not in selected_journal_keys:
            excluded_manual_journals += 1
            continue

        filtered.append(article)

    unique_ids = set()
    unique_records = 0
    for article in filtered:
        key = article.get('doi') or article.get('title', '').lower()
        if key and key not in unique_ids:
            unique_ids.add(key)
            unique_records += 1

    concept_total = int(concept_report.get("total", len(articles)) or len(articles))
    conceptual_excluded = int(concept_report.get("excluded", 0) or 0)
    passed_concept_count = int(concept_report.get("passed", len(articles)) or len(articles))

    return {
        "raw_total": int(raw_total or len(articles)),
        "duplicates_count": max(0, int(raw_total or len(articles)) - concept_total),
        "concept_total": concept_total,
        "concept_discarded": conceptual_excluded,
        "candidates": passed_concept_count,
        "metadata_discarded": max(0, passed_concept_count - unique_records),
        "final_count": unique_records,
        "excluded_years": excluded_years,
        "excluded_open_access": excluded_open_access,
        "excluded_academic_quality": excluded_academic_quality,
        "excluded_languages": excluded_languages,
        "excluded_manual_journals": excluded_manual_journals,
    }

@app.get("/get_prisma_diagram/{session_id}")
async def get_prisma_diagram(
    session_id: int,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    languages: Optional[str] = None,
    journals: Optional[str] = None,
    open_access: Optional[str] = None,
    academic_quality: Optional[str] = None,
):
    if not session_exists(session_id):
        return JSONResponse({"mermaid": "graph TD\n  Error[Sesión no encontrada]"}, status_code=404)

    session_data = get_session(session_id)
    articles = session_data.get("articles", [])
    raw_total = session_data.get("raw_count", len(articles))
    concept_report = session_data.get("concept_report", {})
    has_live_filters = any([
        start_year is not None,
        end_year is not None,
        bool(languages),
        bool(journals),
        open_access is not None,
        academic_quality is not None,
    ])
    last_counts = (
        _compute_prisma_filter_counts(
            articles,
            raw_total,
            concept_report,
            start_year,
            end_year,
            languages,
            journals,
            open_access,
            academic_quality,
        )
        if has_live_filters
        else session_data.get("last_filter_counts", {})
    )
    query_text = session_data.get("question") or session_data.get("query") or "Investigacion sistematica"

    # Datos de Auditoría
    n_encontrados = int(last_counts.get("raw_total", raw_total) or 0)
    n_unicos = int(last_counts.get("concept_total", concept_report.get("total", len(articles))) or 0)
    n_duplicados = max(0, int(last_counts.get("duplicates_count", max(0, raw_total - n_unicos)) or 0))
    n_excl_concept = int(last_counts.get("concept_discarded", concept_report.get("excluded", 0)) or 0)
    n_candidatos = int(last_counts.get("candidates", concept_report.get("passed", n_unicos)) or 0)
    graph_recovered_count = int(
        (session_data.get("adaptive_graph_report") or {}).get("recovered_count", 0) or 0
    )
    pool_extra_count = max(0, n_unicos - n_encontrados)
    pool_label = (
        "Pool conceptual tras deduplicacion y recuperacion"
        if pool_extra_count or graph_recovered_count
        else "Registros despues de deduplicacion"
    )
    pool_extra_line = (
        f"<br/>Recuperacion adaptativa: +{graph_recovered_count or pool_extra_count}"
        if pool_extra_count or graph_recovered_count
        else ""
    )

    excl_anios = int(last_counts.get("excluded_years", len(session_data.get('excluidos_anios', []))) or 0)
    excl_oa = int(last_counts.get("excluded_open_access", len(session_data.get('excluidos_oa', []))) or 0)
    excl_revista = int(last_counts.get("excluded_academic_quality", len(session_data.get('excluidos_revista', []))) or 0)
    excl_idioma = int(last_counts.get("excluded_languages", len(session_data.get('excluidos_idioma', []))) or 0)
    excl_manual = int(last_counts.get("excluded_manual_journals", len(session_data.get('excluidos_revista_manual', []))) or 0)
    n_finales = int(last_counts.get("final_count", max(0, n_candidatos - excl_anios - excl_oa - excl_revista - excl_idioma - excl_manual)) or 0)

    n_stage1_excluded = int(session_data.get("cascade_stage1_excluded", 0) or 0)
    n_stage4_excluded = int(session_data.get("cascade_stage4_excluded", 0) or 0)
    n_stage4_included = int(session_data.get("cascade_stage4_included", 0) or 0)
    cascade_available = n_stage4_included > 0 or n_stage4_excluded > 0

    # Estrategias de Búsqueda (Simuladas de los logs para realismo académico)
    sources_info = {
        "PubMed": {"count": 0, "query": "Consulta registrada en auditoria"},
        "Semantic Scholar": {"count": 0, "query": "Consulta registrada en auditoria"},
        "OpenAlex": {"count": 0, "query": "Consulta registrada en auditoria"},
        "Europe PMC": {"count": 0, "query": "Consulta registrada en auditoria"}
    }

    search_queries_data = session_data.get("search_queries", {})
    if search_queries_data:
        for source in sources_info:
            q = search_queries_data.get(source) or search_queries_data.get(source.lower())
            if q:
                sources_info[source]["query"] = q[:80]

    # Reemplazar con datos reales de la sesión si existen
    actual_sources = Counter([a.get('source', 'Otros') for a in articles])
    for s, c in actual_sources.items():
        if s in sources_info: sources_info[s]["count"] = c

    source_counts_for_diagram = session_data.get("source_counts") or dict(actual_sources)
    source_lines = "<br/>".join(
        f"{source}: {count}" for source, count in Counter(source_counts_for_diagram).most_common(4)
    ) or "Fuentes no registradas"
    total_metadata_excluded = excl_anios + excl_oa + excl_revista + excl_idioma + excl_manual

    mermaid_code = f"""
    graph TD
        %% CABECERA Y OBJETIVO
        subgraph "PREGUNTA DE INVESTIGACION / OBJETIVO"
            OBJ["<b>{query_text[:100]}...</b>"]
        end

        %% FASE 1: IDENTIFICACIÓN
        subgraph "1º FILTRO: BÚSQUEDA (IDENTIFICACIÓN)"
            direction TB
            subgraph "RESULTADOS POR FUENTE Y ESTRATEGIA"
                DB1["<b>PubMed: {sources_info['PubMed']['count']}</b><br/><i>{sources_info['PubMed']['query']}</i>"]
                DB2["<b>Semantic Scholar: {sources_info['Semantic Scholar']['count']}</b><br/><i>{sources_info['Semantic Scholar']['query']}</i>"]
                DB3["<b>OpenAlex: {sources_info['OpenAlex']['count']}</b><br/><i>{sources_info['OpenAlex']['query']}</i>"]
                DB4["<b>Europe PMC: {sources_info['Europe PMC']['count']}</b><br/><i>{sources_info['Europe PMC']['query']}</i>"]
            end

            CLEAN["<b>LIMPIEZA INICIAL</b><br/>{max(0, 10000 - n_encontrados)} descartados<br/>Metadatos incompletos/Timeout"]
            RAW["<b>Total Inicial Bruto</b><br/>{n_encontrados} documentos"]

            DB1 & DB2 & DB3 & DB4 --> RAW
            RAW --> CLEAN
        end

        %% FASE 2: CRIBADO
        subgraph "2º FILTRO: DEDUPLICACIÓN"
            D1["Búsqueda aplicando eliminación de duplicados:<br/><b>Total Únicos: {n_unicos}</b>"]
            D2["<b>SE DESCARTARON</b><br/>{n_duplicados} duplicados<br/>Solapamiento entre BBDD"]
            D1 -.-> D2
        end
        CLEAN --> D1

        subgraph "3º FILTRO: CRIBADO IA"
            IA1["Cribado automático mediante modelos de lenguaje<br/>por relevancia semántica:<br/><b>Resultados: {n_candidatos}</b>"]
            IA2["<b>SE DESCARTARON</b><br/>{n_excl_concept} referencias<br/>Irrelevantes según resumen/abstract"]
            IA1 -.-> IA2
        end
        D1 --> IA1

        %% FASE 3: ELEGIBILIDAD (Dinámica)
        subgraph "4 FILTRO: ELEGIBILIDAD"
                E1["Aplicacion de criterios secundarios<br/>(Anios, Calidad, Idioma)<br/><b>Resultados: {n_finales}</b>"]
                E2["<b>SE DESCARTARON</b><br/>{excl_anios + excl_revista + excl_idioma} documentos<br/>No cumplen criterios de elegibilidad"]
                E1 -.-> E2
        end

        %% FASE 4: INCLUSIÓN
        subgraph "ESTUDIOS INCLUIDOS"
            FIN["<b>{n_finales}</b><br/>ESTUDIOS INCLUIDOS<br/>REVISION VERIFICADA"]
        end
        IA1 --> E1
        E1 --> FIN

        %% ESTILOS PREMIUM
        style OBJ fill:#f8f9fa,stroke:#dee2e6,stroke-width:1px
        style DB1 fill:#fff,stroke:#dee2e6
        style DB2 fill:#fff,stroke:#dee2e6
        style DB3 fill:#fff,stroke:#dee2e6
        style DB4 fill:#fff,stroke:#dee2e6
        style RAW fill:#fff,stroke:#004a99,stroke-width:2px
        style CLEAN fill:#fff,stroke:#dc3545,stroke-dasharray: 5 5
        style D1 fill:#fff,stroke:#333
        style D2 fill:#f8f9fa,stroke:#dee2e6
        style IA1 fill:#fff,stroke:#333
        style IA2 fill:#fff5f5,stroke:#ffc9c9
        style FIN fill:#e6fffa,stroke:#28a745,stroke-width:3px
    """

    if cascade_available:
        mermaid_code = f"""
    flowchart TB
        A["<b>Identificacion</b><br/>Registros identificados: {n_encontrados}<br/>{source_lines}"]
        R1["Registros eliminados antes del cribado<br/>Duplicados: {n_duplicados}"]
        B["{pool_label}<br/><b>{n_unicos}</b>{pool_extra_line}"]
        C["Registros cribados por titulo/resumen<br/><b>{n_unicos}</b>"]
        R2["Registros excluidos por relevancia<br/>{n_excl_concept}"]
        D["Registros candidatos para elegibilidad<br/><b>{n_candidatos}</b>"]
        E["Elegibilidad por metadatos<br/><b>{n_finales}</b> elegibles"]
        R3["Excluidos por filtros<br/>Anio: {excl_anios} | OA: {excl_oa}<br/>Calidad: {excl_revista}<br/>Idioma: {excl_idioma}<br/><b>Total: {total_metadata_excluded}</b>"]
        S1["Stage 1 - Filtro rapido<br/><b>{n_candidatos - n_stage1_excluded}</b> pasaron"]
        RS1["Excluidos en Stage 1<br/><b>{n_stage1_excluded}</b>"]
        S4["Stage 4 - LLM Juez<br/><b>{n_stage4_included}</b> incluidos"]
        RS4["Excluidos en Stage 4<br/><b>{n_stage4_excluded}</b>"]
        FIN["<b>{n_stage4_included}</b><br/>estudios incluidos en revision"]

        A --> B --> C --> D --> E --> S1 --> S4 --> FIN
        B -.-> R1
        C -.-> R2
        E -.-> R3
        S1 -.-> RS1
        S4 -.-> RS4

        style A fill:#f8fafc,stroke:#0f766e,stroke-width:2px
        style B fill:#fff,stroke:#64748b
        style C fill:#fff,stroke:#64748b
        style D fill:#eff6ff,stroke:#2563eb,stroke-width:2px
        style E fill:#fff,stroke:#64748b
        style R1 fill:#fff7ed,stroke:#f97316,stroke-dasharray: 4 4
        style R2 fill:#fff1f2,stroke:#e11d48,stroke-dasharray: 4 4
        style R3 fill:#fff1f2,stroke:#e11d48,stroke-dasharray: 4 4
        style S1 fill:#fee2e2,stroke:#dc2626
        style S4 fill:#dcfce7,stroke:#16a34a
        style FIN fill:#ecfdf5,stroke:#0f766e,stroke-width:3px
        style RS1 fill:#fff1f2,stroke:#e11d48,stroke-dasharray: 4 4
        style RS4 fill:#fff1f2,stroke:#e11d48,stroke-dasharray: 4 4
    """
    else:
        mermaid_code = f"""
    flowchart TB
        A["<b>Identificacion</b><br/>Registros identificados: {n_encontrados}<br/>{source_lines}"]
        R1["Registros eliminados antes del cribado<br/>Duplicados: {n_duplicados}"]
        B["{pool_label}<br/><b>{n_unicos}</b>{pool_extra_line}"]
        C["Registros cribados por titulo/resumen<br/><b>{n_unicos}</b>"]
        R2["Registros excluidos por relevancia<br/>{n_excl_concept}"]
        D["Registros candidatos para elegibilidad<br/><b>{n_candidatos}</b>"]
        E["Elegibilidad por metadatos<br/><b>{n_finales}</b> elegibles"]
        R3["Excluidos por filtros<br/>Anio: {excl_anios} | OA: {excl_oa}<br/>Calidad/revista: {excl_revista}<br/>Idioma: {excl_idioma} | Revista manual: {excl_manual}<br/><b>Total: {total_metadata_excluded}</b>"]
        FIN["<b>{n_finales}</b><br/>estudios para screening profundo"]

        A --> B --> C --> D --> E --> FIN
        B -.-> R1
        C -.-> R2
        E -.-> R3

        style A fill:#f8fafc,stroke:#0f766e,stroke-width:2px
        style B fill:#fff,stroke:#64748b
        style C fill:#fff,stroke:#64748b
        style D fill:#eff6ff,stroke:#2563eb,stroke-width:2px
        style E fill:#fff,stroke:#64748b
        style FIN fill:#ecfdf5,stroke:#0f766e,stroke-width:3px
        style R1 fill:#fff7ed,stroke:#f97316,stroke-dasharray: 4 4
        style R2 fill:#fff1f2,stroke:#e11d48,stroke-dasharray: 4 4
        style R3 fill:#fff1f2,stroke:#e11d48,stroke-dasharray: 4 4
    """

    import html as html_lib
    source_counter = Counter(source_counts_for_diagram)
    if source_counter and sum(source_counter.values()) == n_encontrados:
        source_rows = "".join(
            f"<div><span>{html_lib.escape(str(source))}</span><strong>{int(count)}</strong></div>"
            for source, count in source_counter.most_common(4)
        )
    else:
        source_names = ", ".join(str(source) for source in source_counter.keys()) or "Fuentes no registradas"
        source_rows = f"<div><span>{html_lib.escape(source_names)}</span></div>"
    exclusion_rows = "".join([
        f"<li><span>Fuera de rango temporal</span><strong>{excl_anios}</strong></li>",
        f"<li><span>Sin acceso abierto</span><strong>{excl_oa}</strong></li>",
        f"<li><span>Calidad académica / sin revista</span><strong>{excl_revista}</strong></li>",
        f"<li><span>Idioma no permitido</span><strong>{excl_idioma}</strong></li>",
        f"<li><span>Revista no seleccionada</span><strong>{excl_manual}</strong></li>",
    ])
    diagram_html = f"""
    <section class="prisma2020" aria-label="Diagrama de flujo PRISMA 2020">
      <header>
        <h3>Diagrama de flujo PRISMA 2020</h3>
        <p>Trazabilidad de identificación, cribado, elegibilidad e inclusión antes del cribado profundo con IA</p>
      </header>
      <div class="prisma-grid">
        <div class="stage-label identification">Identificación</div>
        <div class="box primary">
          <div class="box-title">Registros identificados en bases de datos</div>
          <div class="box-count">n = {n_encontrados}</div>
          <div class="source-list">{source_rows}</div>
        </div>
        <div class="box side">
          <div class="box-title">Registros retirados antes del cribado</div>
          <ul>
            <li><span>Duplicados eliminados</span><strong>{n_duplicados}</strong></li>
          </ul>
        </div>

        <div class="arrow down"></div>
        <div class="stage-label screening">Cribado</div>
        <div class="box">
          <div class="box-title">Registros cribados por título y resumen</div>
          <div class="box-count">n = {n_unicos}</div>
        </div>
        <div class="box excluded">
          <div class="box-title">Registros excluidos por relevancia conceptual</div>
          <div class="box-count">n = {n_excl_concept}</div>
        </div>

        <div class="arrow down"></div>
        <div class="stage-label eligibility">Elegibilidad</div>
        <div class="box">
          <div class="box-title">Registros evaluados por criterios de elegibilidad</div>
          <div class="box-count">n = {n_candidatos}</div>
        </div>
        <div class="box excluded">
          <div class="box-title">Registros excluidos con razones</div>
          <ul>{exclusion_rows}</ul>
          <div class="box-total">Total n = {total_metadata_excluded}</div>
        </div>

        <div class="arrow down"></div>
        <div class="stage-label included">Incluidos</div>
        <div class="box included-box">
          <div class="box-title">Estudios retenidos para cribado con IA</div>
          <div class="box-count">n = {n_finales}</div>
        </div>
      </div>
    </section>
    """

    return {"mermaid": mermaid_code, "html": diagram_html}

@app.get("/download_pdf/{filename}")
async def download_pdf_endpoint(filename: str):
    """Endpoint para descargar el PDF generado."""
    file_path = f"static/{filename}"
    if Path(file_path).exists():
        return FileResponse(
            file_path,
            media_type="application/pdf",
            filename=filename
        )
    return HTMLResponse("PDF no encontrado", 404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
