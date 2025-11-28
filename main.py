import os
import sys
import re # Necesario para limpiar HTML

# ============================================================
# 🚨 FIX CRÍTICO DE ESPACIO EN DISCO (C: LLENO)
# ============================================================
CACHE_ROOT = "D:/AI_MODELS_CACHE"
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

from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from modules import (
    search_engine, filters, deduplication, screening,
    rag_pipeline, synthesis, metrics, database, screening_ai, 
    pdf_extractor 
)
from utils.query_expander import expand_query
from utils.export import export_to_csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

for d in [".cache", "logs", "static", "templates"]:
    os.makedirs(d, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Almacenamiento temporal en memoria (Diccionario global)
TEMP_ARTICLES = {}

def clean_old_cache_and_logs():
    """Limpia archivos CSV viejos al reiniciar."""
    try:
        for f in Path("logs").glob("*.csv"): 
            try: f.unlink()
            except: pass
    except: pass

clean_old_cache_and_logs()

# --- FUNCIÓN AUXILIAR: Generador de BibTeX Robusto ---
def generate_bibtex(article: Dict) -> str:
    """Genera una entrada BibTeX completa y profesional."""
    try:
        # 1. Determinar ID (Clave de cita)
        auth = article.get('authors', [])
        first_author = auth[0].split(" ")[-1] if auth and isinstance(auth, list) else "Unknown"
        if isinstance(auth, str): first_author = auth.split(",")[0].split(" ")[-1]
        year = article.get('year', 'n.d.')
        # Limpiar título para la clave (primeras 2 palabras)
        title_slug = "".join(re.findall(r'[a-zA-Z]+', article.get('title', ''))[:2])
        cite_key = f"{first_author}{year}{title_slug}"
        
        # 2. Determinar Tipo de Entrada
        journal = str(article.get('journal', '')).lower()
        entry_type = "article"
        if "conference" in journal or "proceeding" in journal or "symposium" in journal:
            entry_type = "inproceedings"
        elif "arxiv" in journal:
            entry_type = "misc" # arXiv suele citarse como misc o techreport
            
        # 3. Construir campos
        bib = f"@{entry_type}{{{cite_key},\n"
        bib += f"  title = {{{article.get('title', 'No Title')}}},\n"
        
        # Autores (formato: Apellido, Nombre and ...)
        if isinstance(auth, list):
            auth_str = " and ".join(auth)
        else:
            auth_str = str(auth)
        bib += f"  author = {{{auth_str}}},\n"
        
        bib += f"  year = {{{year}}},\n"
        
        if article.get('journal'):
            bib += f"  journal = {{{article.get('journal')}}},\n"
            
        # Campos extra que faltaban
        if article.get('volume'): bib += f"  volume = {{{article.get('volume')}}},\n"
        if article.get('issue'): bib += f"  number = {{{article.get('issue')}}},\n"
        if article.get('pages'): bib += f"  pages = {{{article.get('pages')}}},\n"
        if article.get('publisher'): bib += f"  publisher = {{{article.get('publisher')}}},\n"
        if article.get('doi'): bib += f"  doi = {{{article.get('doi')}}},\n"
        if article.get('url'): bib += f"  url = {{{article.get('url')}}},\n"
        
        # Cerrar
        bib += "}"
        return bib
    except Exception as e:
        return ""

# --- FUNCIÓN AUXILIAR PARA NORMALIZAR DATOS ANTES DE CSV ---
def normalize_article_for_csv(article: Dict) -> Dict:
    """Asegura que campos críticos como URL y metodología existan y estén limpios."""
    # 1. Normalización de URL (Prioridad: url > pdf_url > link > doi > pubmed_id)
    url = article.get('url') or article.get('pdf_url') or article.get('link') or ""
    
    # 1a. Si no hay URL pero hay DOI, construir URL DOI
    if not url and article.get('doi'):
        url = f"https://doi.org/{article.get('doi')}"
    
    # 1b. Si no hay URL pero es PubMed (tiene ID numérico), construir URL PubMed
    if not url and (article.get('source') == 'PubMed' or str(article.get('id', '')).isdigit()):
        pmid = article.get('pubmed_id') or article.get('id')
        if pmid:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    article['url'] = url
    
    # Aseguramos que pdf_url también tenga valor si url lo tiene (para el botón de descarga)
    if not article.get('pdf_url') and url:
        article['pdf_url'] = url
        # Marcamos que vale la pena intentar descargar si tenemos URL
        article['needs_pdf_download'] = True

    # 2. Generación de BibTeX (Nuevo Feature)
    # Generamos el bibtex y lo guardamos en el objeto para que el frontend lo consuma fácil
    article['bibtex'] = generate_bibtex(article)

    # 3. Limpieza de HTML en campos de IA (para que el CSV sea legible)
    ai_fields = ['methodology', 'population', 'key_findings', 'limitations', 'conclusions', 
                 'study_design', 'objectives', 'independent_variables', 'dependent_variables', 'summary']
    
    for field in ai_fields:
        val = article.get(field)
        if val and isinstance(val, str) and ('<' in val): # Si parece HTML
            clean_key = f"{field}_clean"
            if article.get(clean_key):
                article[field] = article[clean_key]
            else:
                clean_val = re.sub('<[^<]+?>', ' ', val)
                clean_val = re.sub(' +', ' ', clean_val).strip()
                article[field] = clean_val
                
    return article

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
async def initial_search(request: Request, question: str = Form(...)):
    start = time.perf_counter()
    logging.info(f"📝 Nueva Búsqueda: {question}")

    terms = expand_query(question, max_terms=10)
    articles, t_search = search_engine.search_articles(terms, max_results=1000)
    
    if not articles:
        return HTMLResponse("<h1>No se encontraron artículos. Intenta ampliar tus términos.</h1>")
    
    # Normalizamos INMEDIATAMENTE para que el primer CSV ya tenga URLs y BibTeX
    articles = [normalize_article_for_csv(a) for a in articles]

    database.save_to_milvus(articles)
    export_to_csv(articles, "log_0_initial_search.csv")
    
    session_id = abs(hash(f"{question}_{time.time()}")) % (10 ** 8)
    TEMP_ARTICLES[session_id] = {
        "question": question,
        "articles": articles,
        "search_terms": terms,
        "search_time": t_search,
        "log_prefix": f"session_{session_id}"
    }
    
    years = [int(a.get('year', 0)) for a in articles if a.get('year')]
    y_min = min(years) if years else 2020
    y_max = max(years) if years else 2025
    
    lang_counts = Counter()
    for a in articles: lang_counts['en'] += 1 
    journals = Counter([str(a.get('journal', 'Unknown')) for a in articles])
    
    return templates.TemplateResponse("filters.html", {
        "request": request,
        "session_id": session_id,
        "question": question,
        "total": len(articles),
        "year_min": y_min,
        "year_max": y_max,
        "languages": [{"code": "en", "name": "Inglés", "count": len(articles)}],
        "all_journals": [{"name": k, "count": v} for k, v in journals.most_common(20)]
    })

@app.post("/update_filter_count", response_class=JSONResponse)
async def update_filter_count(request: Request, session_id: int = Form(...), 
                            start_year: Optional[int] = Form(None),
                            end_year: Optional[int] = Form(None),
                            open_access: Optional[str] = Form(None)):
    
    if session_id not in TEMP_ARTICLES: return JSONResponse({"error": "Expirado"}, 400)
    raw = TEMP_ARTICLES[session_id]["articles"]
    
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
        "duplicates_count": len(filtered) - unique_count,
        "final_count": unique_count
    })

@app.get("/apply_filters")
async def apply_filters_get():
    return RedirectResponse(url="/")

@app.post("/apply_filters")
async def apply_filters(request: Request, session_id: int = Form(...), question: str = Form(...),
                        start_year: int = Form(2000), end_year: int = Form(2025),
                        open_access: Optional[str] = Form("false")):
    
    if session_id not in TEMP_ARTICLES: raise HTTPException(400, "Sesión expirada")
    
    data = TEMP_ARTICLES[session_id]
    articles = data["articles"]
    start_time = time.time()
    log_prefix = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(start_time)}"
    
    filtered = filters.apply_filters(articles, start_year=start_year, end_year=end_year, open_access=(open_access == "true"))
    unique, _ = deduplication.remove_semantic_duplicates(filtered, 0.92)
    
    # IMPORTANTE: Volvemos a normalizar aquí por si acaso
    unique = [normalize_article_for_csv(a) for a in unique]
    
    logging.info("🌍 Preparando Query Semántica...")
    try:
        query_en = screening_ai.translate_question_to_english(question)
        query_for_screening = query_en if query_en and "Error" not in query_en else question
    except:
        query_for_screening = question

    # --- LÓGICA DE PRIORIZACIÓN DE URLs ("PDF FIRST") ---
    candidates = screening.screen_articles(unique, query_for_screening, max_results=200)
    
    with_url = []
    without_url = []
    
    for art in candidates:
        has_valid_url = art.get('url') and len(str(art.get('url'))) > 10
        if has_valid_url:
            with_url.append(art)
        else:
            without_url.append(art)
            
    final_top_50 = with_url[:50]
    
    if len(final_top_50) < 50:
        needed = 50 - len(final_top_50)
        final_top_50.extend(without_url[:needed])
        
    relevant = final_top_50
    # ----------------------------------------------------

    logging.info(f"🚀 Renderizando {len(relevant)} artículos (Prioridad URL aplicada).")

    export_to_csv(relevant, f"{log_prefix}_log_3_FINAL_70percent.csv")
    
    data.update({"relevant_articles": relevant, "dedup_articles": unique, "log_prefix": log_prefix})
    
    for art in relevant:
        if 'abstract' in art and art['abstract']:
            art['abstract'] = art['abstract'][:5000]  # límite razonable
        if 'full_text' in art:
            del art['full_text']  # NUNCA envíes full_text al frontend
        # Elimina campos pesados que no usas en screening
        art.pop('embedding', None)
        art.pop('pdf_content', None)

    return templates.TemplateResponse("screening.html", {
        "request": request,
        "session_id": session_id,
        "question": question,
        "articles": relevant
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
        column_name = req_data.get('column_name')
        question = req_data.get('question', '')  # ✅ AGREGAR ESTO
        
        # JIT Download
        if not article.get('is_pdf_downloaded') and article.get('needs_pdf_download'):
            logging.info(f"📥 Descarga JIT iniciada para: {article.get('title')[:30]}...")
            article = pdf_extractor.download_full_text_lazy(article)
            
        result = screening_ai._generate_columns_for_article(
            article, 
            [column_name],
            research_question=question  # ✅ PASAR LA PREGUNTA
        )
        val = result.get(column_name, "⚠️ No extraído")
        
        return JSONResponse({"value": val, "column_name": column_name})

    except Exception as e:
        logging.error(f"❌ Error en generate_column: {e}")
        return JSONResponse({"error": str(e), "value": "Error en servidor"}, 500)

@app.post("/submit_screening")
async def submit_screening(request: Request):
    data = await request.json()
    session_id = int(data.get("sessionId"))
    if session_id not in TEMP_ARTICLES: return JSONResponse({"error": "Expirado"}, 400)
    
    session_data = TEMP_ARTICLES[session_id]
    log_prefix = session_data.get("log_prefix", "session")
    
    included, excluded = [], []
    for _, item in data.get("articles", {}).items():
        art = item['data']
        
        ai_fields = item.get('aiGeneratedFields', {})
        art.update(ai_fields) 
        
        art['researcher_notes'] = item.get('notes', '')
        art['translation'] = item.get('translation', '')
        
        art = normalize_article_for_csv(art)
        
        if item['status'] == 'included': included.append(art)
        else: 
            art['exclusion_reason'] = item.get('exclusionReason', '')
            excluded.append(art)
    
    export_to_csv(excluded, f"{log_prefix}_log_5_excluded.csv")
    export_to_csv(included, f"{log_prefix}_log_6_included.csv")
    
    # GUARDAMOS LOS INCLUIDOS EN LA SESIÓN PARA USARLOS LUEGO
    TEMP_ARTICLES[session_id]["included_articles"] = included
    
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
# 🌟 ENDPOINT DE SÍNTESIS CORREGIDO (Usa SOLAMENTE artículos incluidos)
# ==============================================================================
@app.post("/generate_synthesis")
async def generate_synthesis_endpoint(request: Request):
    """
    Genera DOBLE síntesis:
    1. Breve (estilo Elicit) para mostrar en HTML
    2. Completa para PDF descargable
    """
    data = await request.json()
    session_id = int(data.get("sessionId"))
    question = data.get("question")
    session_data = TEMP_ARTICLES.get(session_id, {})
    
    included = session_data.get("included_articles", [])
    
    if not included:
        return JSONResponse({"error": "No hay artículos seleccionados."}, 400)
    
    logging.info(f"🧪 Generando síntesis dual con {len(included)} artículos.")
    
    # ===== 1. SÍNTESIS BREVE (HTML) =====
    try:
        from modules.synthesis import generate_synthesis_brief
        synth_brief = generate_synthesis_brief(included, question)
    except Exception as e:
        logging.error(f"Error en síntesis breve: {e}")
        synth_brief = "Error al generar resumen ejecutivo."
    
    # ===== 2. SÍNTESIS COMPLETA (PDF) =====
    try:
        from modules.synthesis import generate_synthesis_full
        synth_full = generate_synthesis_full(included, question)
    except Exception as e:
        logging.error(f"Error en síntesis completa: {e}")
        synth_full = synth_brief  # Fallback
    
    # ===== 3. GENERAR PDF =====
    log_prefix = session_data.get("log_prefix", "session")
    
    try:
        from modules.report_generator import create_pdf_report
        pdf_path = create_pdf_report(
            synthesis_text=synth_full,
            metrics={
                "total": len(session_data.get("articles", [])),
                "after_filter": len(session_data.get("dedup_articles", [])),
                "after_dedup": len(session_data.get("dedup_articles", [])),
                "relevant": len(session_data.get("relevant_articles", [])),
                "final_included": len(included),
            },
            articles=included,
            question=question,
            pdf_path=f"static/{log_prefix}_REPORTE_PRISMA.pdf"
        )
        session_data['pdf_path'] = pdf_path
    except Exception as e:
        logging.error(f"❌ Error generando PDF: {e}")
        pdf_path = None
    
    # ===== 4. GUARDAR CSV FINAL =====
    included_normalized = [normalize_article_for_csv(a) for a in included]
    export_to_csv(included_normalized, f"{log_prefix}_TABLA_FINAL.csv")
    
    # ===== 5. MÉTRICAS =====
    metrics_data = {
        "total": len(session_data.get("articles", [])),
        "after_filter": len(session_data.get("dedup_articles", [])),
        "after_dedup": len(session_data.get("dedup_articles", [])),
        "relevant": len(session_data.get("relevant_articles", [])),
        "final_included": len(included),
        "t_search": round(session_data.get("search_time", 0), 2),
        "t_filter": 0.5,
        "t_dedup": 1.2,
        "t_screen": 2.5,
        "t_synth": 8.0,
        "total_time": 15.0
    }
    
    return templates.TemplateResponse("results.html", {
        "request": request,
        "synthesis": synth_brief,  # Síntesis BREVE para HTML
        "metrics": metrics_data,
        "final_articles": included,
        "plots": {"prisma": ""},
        "session_id": session_id,
        "question": question,
        "pdf_available": pdf_path is not None,
        "pdf_filename": os.path.basename(pdf_path) if pdf_path else ""
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