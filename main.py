import os
import sys
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

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
from typing import Optional, List, Dict
from collections import Counter

from fastapi import FastAPI, Form, Request, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from modules.logic import filters, deduplication, screening, metrics
from modules.core import search_engine, database
from modules.core.report_generator import create_pdf_report
from modules.ai import rag_pipeline, synthesis, screening_ai
from modules.ai.ai_model import init_model
from utils import pdf_extractor
from utils.query_expander import expand_query, generate_api_queries_with_llm
from utils.export import export_to_csv
from utils.eval_screening import evaluate_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

for d in [".cache", ".cache/sessions", "logs", "static", "templates"]:
    os.makedirs(d, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ============================================================
# 🚦 CONTROL DE CONCURRENCIA PARA IA
# ============================================================
# Máximo 2 peticiones simultáneas a proveedores externos para evitar 429
AI_SEMAPHORE = asyncio.Semaphore(2)

@app.on_event("startup")
async def startup_event():
    logging.info("🚀 Iniciando servidor y cargando IA...")
    # Esto cargará el modelo en la VRAM al arrancar
    try:
        init_model() 
    except Exception as e:
        logging.warning(f"⚠️ No se pudo cargar el modelo local (¿Falta GPU?): {e}")

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
    try:
        path = _get_session_path(s_id)
        with open(path, "w", encoding="utf-8") as f:
            f.write(_serialize_session(data))
        logging.info(f"💾 Sesión {s_id} guardada en disco: {path.name}")
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

def normalize_article_for_csv(article: Dict, ai_fields: List[str] = None) -> Dict:
    """Asegura que campos críticos como URL y campos dinámicos de IA estén limpios."""
    url = article.get('url') or article.get('pdf_url') or article.get('link') or ""
    
    if not url and article.get('doi'):
        url = f"https://doi.org/{article.get('doi')}"
    
    if not url and (article.get('source') == 'PubMed' or str(article.get('id', '')).isdigit()):
        pmid = article.get('pubmed_id') or article.get('id')
        if pmid: url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    article['url'] = url
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
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
async def initial_search(request: Request, background_tasks: BackgroundTasks, question: str = Form(...)):
    start = time.perf_counter()
    logging.info(f"📝 Nueva Búsqueda: {question}")

    terms = expand_query(question, max_terms=10)
    articles, t_search, search_queries_used, raw_count = search_engine.search_articles(terms, max_results=1000, original_question=question)
    
    if not articles:
        return HTMLResponse("<h1>No se encontraron artículos. Intenta ampliar tus términos.</h1>")
    
    articles = [normalize_article_for_csv(a) for a in articles]

    # v17.2: Embedding en BACKGROUND — no bloquear al usuario durante 12+ min
    # Los abstractos se sobreescriben de todas formas cuando se descargan los PDFs reales.
    background_tasks.add_task(database.save_to_milvus, articles)
    export_to_csv(articles, "log_0_initial_search.csv")
    
    # 🔥 v6: Evaluación Automática de Calidad (Initial Search)
    evaluate_results(articles)
    
    # 🔥 v9.0: Matriz de Extracción Adaptativa (Solo 1 vez por sesión)
    column_config = screening_ai.propose_columns_from_rq(question)
    
    session_id = abs(hash(f"{question}_{time.time()}")) % (10 ** 8)
    
    session_data = {
        "question": question,
        "articles": articles,
        "raw_count": raw_count,            # v16.5: Total antes de deduplicación
        "search_terms": terms,
        "search_time": t_search,
        "search_queries": search_queries_used,
        "column_config": column_config,
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
    journals = Counter([str(a.get('journal', 'Unknown')) for a in articles])
    
    duplicates_removed = raw_count - len(articles)
    languages_list = [
        {"code": code, "name": lang_meta.get(code, {}).get('name', code), 
         "flag": lang_meta.get(code, {}).get('flag', '🌐'), "count": count}
        for code, count in lang_counts.most_common()
    ]
    return templates.TemplateResponse("filters.html", {
        "request": request,
        "session_id": session_id,
        "question": question,
        "total": len(articles),
        "raw_total": raw_count,
        "duplicates_removed": duplicates_removed,
        "year_min": y_min,
        "year_max": y_max,
        "languages": languages_list,
        "all_journals": [{"name": k, "count": v} for k, v in journals.most_common(20)],
        # Pre-cargar criterios I/E si el usuario vuelve a la vista de filtros
        "session_inclusion_criteria": session_data.get("last_inclusion_criteria", ""),
        "session_exclusion_criteria": session_data.get("last_exclusion_criteria", ""),
    })

@app.post("/update_filter_count", response_class=JSONResponse)
async def update_filter_count(request: Request, session_id: int = Form(...), 
                            start_year: Optional[int] = Form(None),
                            end_year: Optional[int] = Form(None),
                            open_access: Optional[str] = Form(None)):
    
    if not session_exists(session_id):
        return JSONResponse({"error": "Sesión expirada"}, 400)
    
    session_data = get_session(session_id)
    raw = session_data["articles"]
    base_duplicates_removed = session_data.get("raw_count", len(raw)) - len(raw)  # v16.5: dupes eliminados en búsqueda
    
    filtered = [
        a for a in raw 
        if (not start_year or a.get('year', 0) >= start_year) and
           (not end_year or a.get('year', 0) <= end_year)
    ]
    
    unique_ids = set()
    unique_count = 0
    for a in filtered:
        key = a.get('doi') or a.get('title', '').lower()
        if key and key not in unique_ids:
            unique_ids.add(key)
            unique_count += 1
            
    return JSONResponse({
        "filtered_count": len(filtered),
        "duplicates_count": base_duplicates_removed + (len(filtered) - unique_count),
        "final_count": unique_count
    })

@app.get("/apply_filters")
async def apply_filters_get():
    return RedirectResponse(url="/")

# ==============================================================================
# 🧵 TAREA EN SEGUNDO PLANO: DESCARGA PROACTIVA DE PDFS
# ==============================================================================
def batch_pdf_downloader(articles: list, session_id: str):
    """
    Descarga PDFs de forma síncrona para que el frontend refleje el estado real de inmediato.
    Utiliza paralelismo interno para no demorar demasiado.
    """
    to_download = [a for a in articles if a.get('needs_pdf_download') and not a.get('is_pdf_downloaded')]
    
    if not to_download:
        return articles

    logging.info(f" 📥 Sincronizando {len(to_download)} artículos con PDF antes de mostrar resultados...")
    from concurrent.futures import ThreadPoolExecutor
    # v17.0: 4 workers paralelos. El domain lock en pdf_extractor serializa por dominio, evitando bans.
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(pdf_extractor.download_full_text_lazy, to_download))
    
    # Sincronizar con ChromaDB inmediatamente
    downloaded = [a for a in to_download if a.get('is_pdf_downloaded')]
    if downloaded:
        logging.info(f" 💾 Indexando {len(downloaded)} nuevos PDFs en ChromaDB...")
        try:
            database.save_to_milvus(downloaded)
        except Exception as e:
            logging.error(f"❌ Error indexando batch: {e}")
            
    return articles

@app.post("/apply_filters")
async def apply_filters(request: Request, background_tasks: BackgroundTasks, session_id: str = Form(...), question: str = Form(...),
                        start_year: int = Form(2000), end_year: int = Form(2025),
                        open_access: Optional[str] = Form("false"),
                        inclusion_criteria: Optional[str] = Form(""),
                        exclusion_criteria: Optional[str] = Form("")):
    
    if not session_exists(session_id):
        raise HTTPException(400, "Sesión expirada")
    
    data = get_session(session_id)
    articles = data["articles"]
    # ✅ Inyectar índice original para permitir actualizaciones en carga manual
    for i, art in enumerate(articles):
        art['original_index'] = i
        
    start_time = time.time()
    log_prefix = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(start_time)}"
    
    filtered = filters.apply_filters(articles, start_year=start_year, end_year=end_year, open_access=(open_access == "true"))
    unique, _ = deduplication.remove_semantic_duplicates(filtered, 0.92)
    
    unique = [normalize_article_for_csv(a) for a in unique]
    
    # ✅ OPTIMIZACIÓN: Evitar screening redundante de 3 minutos si la pregunta no ha cambiado
    existing_relevant = data.get("relevant_articles", [])
    last_question = data.get("last_screening_question", "")
    
    if existing_relevant and last_question == question:
        logging.info("♻️ Reutilizando screening previo para la misma pregunta...")
        candidates = existing_relevant
    else:
        logging.info("🌍 Preparando Query Semántica...")
        try:
            query_en = screening_ai.translate_question_to_english(question)
            query_for_screening = query_en if query_en and "Error" not in query_en else question
        except:
            query_for_screening = question

        candidates = screening.screen_articles(
            unique, query_for_screening, max_results=200,
            original_question=question,
            inclusion_criteria=inclusion_criteria or "",
            exclusion_criteria=exclusion_criteria or ""
        )
        data["last_screening_question"] = question
        # Persistir criterios para poder detectar cambios futuros
        data["last_inclusion_criteria"] = inclusion_criteria or ""
        data["last_exclusion_criteria"] = exclusion_criteria or ""
    
    with_url = []
    without_url = []
    
    for art in candidates:
        has_valid_url = art.get('url') and len(str(art.get('url'))) > 10
        if has_valid_url:
            with_url.append(art)
        else:
            without_url.append(art)
            
    final_top_100 = with_url[:100]
    
    if len(final_top_100) < 100:
        needed = 100 - len(final_top_100)
        final_top_100.extend(without_url[:needed])
        
    # ✅ v11.1: Activar intención de descarga proactiva para artículos con URL
    for art in final_top_100:
        if art.get('url') and len(str(art.get('url'))) > 10:
            if not art.get('is_pdf_downloaded'):
                art['needs_pdf_download'] = True
    
    relevant = final_top_100

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
    
    # ✅ v11.4: Descarga SÍNCRONA antes de mostrar resultados (Solicitud de Usuario)
    # Esto asegura que las insignias PDF sean veraces desde el primer renderizado.
    relevant = batch_pdf_downloader(relevant, session_id)

    data.update({
        "relevant_articles": relevant, 
        "dedup_articles": unique, 
        "log_prefix": log_prefix
    })
    
    # Normalizar con campos dinámicos para el export inicial
    ai_keys = [col['key'] for col in data.get("column_config", {}).get("columnas", [])]
    relevant = [normalize_article_for_csv(a, ai_keys) for a in relevant]

    save_session(session_id, data)

    return templates.TemplateResponse("screening.html", {
        "request": request,
        "session_id": session_id,
        "question": question,
        "articles": relevant,
        "column_config": data.get("column_config") # Enviar config al frontend
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
        for key in ['full_text', 'is_pdf_downloaded', 'needs_pdf_download', 'full_text_source']:
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
        import pdfplumber
        from io import BytesIO
        
        content = await file.read()
        with pdfplumber.open(BytesIO(content)) as pdf:
            # Extraer hasta 30 páginas
            text = "\n".join([p.extract_text() or "" for p in pdf.pages[:30]])
        
        if len(text) < 200:
            return JSONResponse({"error": "El PDF parece estar vacío o ser una imagen. No se pudo extraer texto."}, 400)
            
        # 2. Actualizar metadata del artículo
        article = articles[articleIndex]
        article['full_text'] = text[:80000] # Límite de seguridad
        article['is_pdf_downloaded'] = True
        article['needs_pdf_download'] = False
        article['full_text_source'] = 'manual_upload'
        
        # 3. Sincronizar con ChromaDB (Chunking automático)
        logging.info(f" ⬆️ Carga manual: Sincronizando '{article.get('title')[:30]}...' con ChromaDB")
        database.save_to_milvus([article])
        
        # 4. Guardar sesión
        save_session(sessionId, session_data)
        
        return JSONResponse({
            "success": True, 
            "message": "PDF procesado y sincronizado correctamente",
            "article": article
        })
        
    except Exception as e:
        logging.error(f"❌ Error en upload_pdf: {e}")
        return JSONResponse({"error": str(e)}, 500)

@app.post("/submit_screening")
async def submit_screening(request: Request):
    data = await request.json()
    session_id = int(data.get("sessionId"))
    
    if not session_exists(session_id):
        logging.error(f"❌ Sesión {session_id} no encontrada")
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
    save_session(session_id, session_data)
    
    avg_sim = sum(a.get('similarity', 0) for a in included) / len(included) if included else 0
    journals = set(a.get('journal') for a in included)
    
    return templates.TemplateResponse("review.html", {
        "request": request, 
        "session_id": session_id,
        "included_articles": included,
        "total_included": len(included),
        "total_excluded": len(excluded),
        "avg_similarity": f"{avg_sim*100:.1f}",
        "unique_journals": len(journals),
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
    
    # 1. Validaciones
    if not session_exists(session_id):
        return JSONResponse({"error": "Sesión no válida"}, 400)
    
    session_data = get_session(session_id)
    included = session_data.get("included_articles", [])
    
    if not included:
        return JSONResponse({"error": "No hay artículos seleccionados."}, 400)
    
    # 2. Calcular métricas (incluyendo search_queries reales y conteo por fuente)
    def _count_by_source(articles):
        """Cuenta artículos por fuente (BD)."""
        counts = {}
        for art in articles:
            src = art.get('source', 'Otra')
            counts[src] = counts.get(src, 0) + 1
        return counts
    
    metrics = {
        "total": len(session_data.get("articles", [])),
        "after_filter": len(session_data.get("dedup_articles", [])),
        "after_dedup": len(session_data.get("dedup_articles", [])),
        "relevant": len(session_data.get("relevant_articles", [])),
        "final_included": len(included),
        "search_queries": session_data.get("search_queries", {}),
        "source_counts": _count_by_source(session_data.get("articles", [])),  # 🔥 NUEVO: Conteo por BD
    }

    logging.info(f"🧪 Iniciando síntesis BACKGROUND para Sesión {session_id}...")

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

    return templates.TemplateResponse("results.html", {
        "request": request,
        "synthesis": synth_brief,
        "metrics": metrics_data,
        "final_articles": included,
        "plots": {"prisma": ""},
        "session_id": session_id,
        "question": question,
        "pdf_available": pdf_exists,
        "pdf_filename": pdf_filename if pdf_exists else ""
    })

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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)