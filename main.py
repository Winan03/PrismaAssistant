import os
from datetime import datetime
import time
import logging
from pathlib import Path
from typing import Optional
from collections import Counter

from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from modules import (
    search_engine, filters, deduplication, screening,
    rag_pipeline, synthesis, metrics, database, screening_ai
)
from modules import screening_ai 
from modules.grok_filter import batch_filter_articles
from utils.query_expander import expand_query
from utils.export import export_to_csv
from modules.prisma_criteria import score_articles_with_prisma


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.makedirs(".cache", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def clean_old_cache_and_logs():
    for dir_path in [".cache", "logs"]:
        cache_dir = Path(dir_path)
        if cache_dir.exists():
            for f in cache_dir.glob("*"):
                try:
                    if f.is_file():
                        f.unlink()
                except Exception as e:
                    logging.warning(f"No se pudo eliminar {f}: {e}")
        os.makedirs(dir_path, exist_ok=True)
    logging.info("🧹 Cache y Logs preparados")

clean_old_cache_and_logs()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

TEMP_ARTICLES = {}
META_PROMPTS_CACHE = {}

def extract_filter_metadata(articles):
    years = [a.get("year", 0) for a in articles if a.get("year", 0) > 1900]
    lang_counts = Counter()
    
    english_words = {" the ", " and ", " for ", " with ", " this ", " that ", " was ", " are "}
    spanish_words = {" de ", " la ", " el ", " en ", " los ", " las ", " con ", " por ", " para ", " que "}
    portuguese_words = {" da ", " do ", " em ", " para ", " com ", " uma ", " dos ", " das ", " pela ", " pelo "}

    for a in articles:
        text_to_check = f" {a.get('title', '').lower()} {a.get('abstract', '').lower()} "
        en_count = sum(1 for word in english_words if word in text_to_check)
        es_count = sum(1 for word in spanish_words if word in text_to_check)
        pt_count = sum(1 for word in portuguese_words if word in text_to_check)
        counts = {'en': en_count, 'es': es_count, 'pt': pt_count}
        max_lang = max(counts, key=counts.get)
        lang_counts[max_lang] += 1 if counts[max_lang] >= 2 else 0

    journal_counts = Counter(a.get("journal", "").strip() for a in articles if a.get("journal", "") and len(a.get("journal", "")) > 3)
    sorted_journals = sorted(journal_counts.items(), key=lambda item: item[0])
    
    return {
        "year_min": min(years) if years else 2000,
        "year_max": max(years) if years else datetime.now().year,
        "languages": [
            {"code": "en", "name": "Inglés", "flag": "🇺🇸", "count": lang_counts.get("en", 0)},
            {"code": "es", "name": "Español", "flag": "🇪🇸", "count": lang_counts.get("es", 0)},
            {"code": "pt", "name": "Portugués", "flag": "🇧🇷", "count": lang_counts.get("pt", 0)},
        ],
        "all_journals": [{"name": j, "count": c} for j, c in sorted_journals],
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
async def initial_search(request: Request, question: str = Form(...)):
    start = time.perf_counter()
    logging.info(f"📝 Pregunta Original: {question}")

    terms = expand_query(question, max_terms=15)
    search_terms = terms[:10]
    logging.info(f"🔍 Términos (Grok-3): {search_terms}")

    articles, t_search = search_engine.search_articles(search_terms, max_results=600)
    
    if not articles:
        return HTMLResponse("No results", status_code=200)
    
    database.save_to_milvus(articles)
    export_to_csv(articles, "log_0_initial_search.csv")
    
    session_id = abs(hash(f"{question}_{time.time()}")) % (10 ** 8)
    english_context_query = ". ".join(search_terms[:5]) 

    TEMP_ARTICLES[session_id] = {
        "question": question,
        "english_query": english_context_query,
        "articles": articles,
        "search_terms": search_terms,
        "search_time": t_search,
        "log_prefix": f"session_{session_id}"
    }
    
    metadata = extract_filter_metadata(articles)
    return templates.TemplateResponse("filters.html", {
        "request": request,
        "session_id": session_id,
        "question": question,
        "total": len(articles),
        "year_min": metadata["year_min"],
        "year_max": metadata["year_max"],
        "languages": metadata["languages"],
        "all_journals": metadata["all_journals"],
    })

@app.post("/update_filter_count", response_class=JSONResponse)
async def update_filter_count(
    request: Request,
    session_id: int = Form(...),
    start_year: Optional[int] = Form(None),
    end_year: Optional[int] = Form(None),
    quartiles: Optional[str] = Form(None),
    open_access: Optional[str] = Form(None),
    languages: Optional[str] = Form(None),
    journals: Optional[str] = Form(None)
):
    if session_id not in TEMP_ARTICLES:
        return JSONResponse({"error": "Sesión expirada"}, status_code=400)
    
    articles = TEMP_ARTICLES[session_id]["articles"]
    journal_list = [j.strip() for j in journals.split(',') if j.strip()] if journals else []
    lang_code = languages.split(',')[0] if languages else None
    
    filtered = filters.apply_filters(
        articles=articles,
        start_year=start_year,
        end_year=end_year,
        open_access=(open_access == "true"),
        language=lang_code,
        journals=journal_list if journal_list else None,
        quartiles=None
    )
    
    seen_doi = set()
    unique_count = 0
    for a in filtered:
        doi = a.get("doi")
        if doi and doi in seen_doi:
            continue
        if doi:
            seen_doi.add(doi)
        unique_count += 1
    
    duplicates = len(filtered) - unique_count
    final_count = unique_count
    
    logging.info(f"📊 AJAX: {len(articles)} → {len(filtered)} filtrados → {final_count} únicos")
    
    return JSONResponse({
        "filtered_count": len(filtered),
        "duplicates_count": duplicates,
        "final_count": final_count
    })


@app.post("/apply_filters")
async def apply_filters(
    request: Request,
    session_id: int = Form(...),
    question: str = Form(...),
    start_year: Optional[int] = Form(None),
    end_year: Optional[int] = Form(None),
    open_access: Optional[str] = Form("false"),
    languages: Optional[str] = Form(None),
    journals: Optional[str] = Form(None),
):
    """
    ✅ FLUJO CORRECTO:
    1. Aplicar filtros físicos
    2. Deduplicación
    3. Screening semántico (70%)
    4. ✅ REDIRIGIR A screening.html (NO generar síntesis aún)
    """
    try:
        start_time = time.time()
        t_filter_start = time.time()
        
        if session_id not in TEMP_ARTICLES:
            raise HTTPException(status_code=400, detail="Sesión expirada")
        
        session_data = TEMP_ARTICLES[session_id]
        raw_articles = session_data["articles"]
        search_terms = session_data["search_terms"]
        
        log_prefix = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(start_time)}"
        os.makedirs(Path("logs"), exist_ok=True)

        logging.info(f"Iniciando revisión sistemática: {question}")
        logging.info(f"Artículos brutos: {len(raw_articles)}")

        # ==================================================================
        # 1. APLICAR FILTROS FÍSICOS
        # ==================================================================
        
        journal_list = [j.strip() for j in journals.split(',') if j.strip()] if journals else None
        lang_code = languages.split(',')[0] if languages else None
        
        articles_filtered = filters.apply_filters(
            articles=raw_articles,
            start_year=start_year,
            end_year=end_year,
            open_access=(open_access == "true"),
            language=lang_code,
            journals=journal_list,
            quartiles=None
        )
        
        t_filter = time.time() - t_filter_start
        
        if not articles_filtered:
            raise HTTPException(status_code=500, detail="No quedaron artículos tras filtros")
            
        export_to_csv(articles_filtered, f"{log_prefix}_log_1_after_filters.csv")

        # ==================================================================
        # 2. DEDUPLICACIÓN
        # ==================================================================
        t_dedup_start = time.time()
        
        articles_after_exact, _ = deduplication.remove_exact_duplicates(articles_filtered)
        unique_articles, _ = deduplication.remove_semantic_duplicates(
            articles_after_exact, 
            similarity_threshold=0.95
        )
        
        t_dedup = time.time() - t_dedup_start
        
        export_to_csv(unique_articles, f"{log_prefix}_log_2_after_dedup.csv")
        logging.info(f"Tras deduplicación: {len(unique_articles)} únicos")

        # ==================================================================
        # 3. SCREENING SEMÁNTICO CON 70%
        # ==================================================================
        t_screen_start = time.time()
        
        logging.info("🌍 Preparando pregunta para screening semántico...")

        try:
            query_en = screening_ai.translate_question_to_english(question)
            
            if query_en and query_en != question:
                logging.info(f"   ✅ Pregunta traducida exitosamente")
                logging.info(f"      ES: {question[:100]}...")
                logging.info(f"      EN: {query_en[:100]}...")
            else:
                logging.warning("   ⚠️ Traducción no disponible, usando términos de búsqueda")
                query_en = ". ".join(search_terms[:5])
                logging.info(f"      Contexto: {query_en}")

        except Exception as e:
            logging.error(f"   ❌ Error en traducción: {e}")
            query_en = ". ".join(search_terms[:5])
            logging.info(f"      Fallback a términos: {query_en}")

        logging.info(f"🧬 Ejecutando screening semántico con query en inglés...")
        relevant_articles = screening.screen_articles(unique_articles, query_en)

        t_screen = time.time() - t_screen_start
        
        export_to_csv(relevant_articles, f"{log_prefix}_log_3_FINAL_70percent.csv")
        logging.info(f"✅ SCREENING COMPLETADO (≥70%): {len(relevant_articles)} artículos")

        # ==================================================================
        # 4. ✅ GUARDAR EN SESIÓN Y REDIRIGIR A SCREENING MANUAL
        # ==================================================================
        
        TEMP_ARTICLES[session_id].update({
            "filtered_articles": articles_filtered,
            "dedup_articles": unique_articles,
            "relevant_articles": relevant_articles,
            "log_prefix": log_prefix,
            "year_range": (start_year, end_year),
            "t_filter": t_filter,
            "t_dedup": t_dedup,
            "t_screen": t_screen
        })
        
        elapsed = time.time() - start_time
        logging.info(f"Fase 1 completada en {elapsed:.1f}s - Redirigiendo a cribado manual")

        # ✅ REDIRIGIR A SCREENING.HTML (NO A RESULTS)
        return templates.TemplateResponse("screening.html", {
            "request": request,
            "session_id": session_id,
            "question": question,
            "articles": relevant_articles
        })

    except Exception as e:
        logging.error(f"Error crítico en apply_filters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

               
@app.post("/translate_abstract")
async def translate_abstract_endpoint(request: Request):
    data = await request.json()
    abstract = data.get("abstract", "")
    if not abstract:
        return JSONResponse({"error": "Abstract vacío"}, status_code=400)
    
    try:
        translation = screening_ai.translate_abstract_to_spanish(abstract)
        return JSONResponse({"translation": translation})
    except Exception as e:
        logging.error(f"❌ Error traduciendo abstract: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/generate_column")
async def generate_column_endpoint(request: Request):
    data = await request.json()
    article_data = data.get("article")
    column_name = data.get("column_name")
    session_id = data.get("session_id")
    
    if not article_data or not column_name:
        return JSONResponse({"error": "Datos incompletos"}, status_code=400)
    
    session_data = TEMP_ARTICLES.get(session_id, {})
    question = session_data.get("question", "")
    
    global META_PROMPTS_CACHE
    meta_prompts = META_PROMPTS_CACHE.get(question, {})
    
    if not meta_prompts:
        meta_prompts = {
            column_name: {
                "keywords": ["method", "results", "data"],
                "extraction_strategy": f"Extract {column_name} information",
                "output_format": "Brief summary"
            }
        }
    
    try:
        enriched = screening_ai._generate_columns_for_article(
            article_data,
            [column_name]
        )
        
        return JSONResponse({
            "value": enriched.get(column_name, "No extraído"),
            "column_name": column_name
        })
    
    except Exception as e:
        logging.error(f"❌ Error generando columna '{column_name}': {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/submit_screening")
async def submit_screening(request: Request):
    """
    ✅ PROCESA DECISIONES DEL INVESTIGADOR
    ✅ REDIRIGE A review.html (NO A results.html)
    """
    data = await request.json()
    session_id = int(data.get("sessionId"))
    
    if session_id not in TEMP_ARTICLES:
        return JSONResponse({"error": "Sesión expirada"}, status_code=400)
    
    session_data = TEMP_ARTICLES[session_id]
    log_prefix = session_data["log_prefix"]
    question = data.get("question")
    
    # Procesar decisiones
    included_articles = []
    excluded_articles = []
    
    for article_id, article_data in data.get("articles", {}).items():
        article_info = article_data["data"].copy()
        article_info["researcher_notes"] = article_data.get("notes", "")
        article_info["translation"] = article_data.get("translation", "")
        
        # Columnas IA
        ai_fields = article_data.get("aiGeneratedFields", {})
        for key, value in ai_fields.items():
            article_info[key] = value
        
        if article_data["status"] == "included":
            included_articles.append(article_info)
        elif article_data["status"] == "excluded":
            article_info["exclusion_reason"] = article_data.get("exclusionReason", "")
            excluded_articles.append(article_info)
    
    # Guardar logs
    export_to_csv(excluded_articles, f"{log_prefix}_log_5_excluded_by_researcher.csv")
    export_to_csv(included_articles, f"{log_prefix}_log_6_final_included_by_researcher.csv")
    
    logging.info(f"✅ Cribado manual: {len(included_articles)} incluidos, {len(excluded_articles)} excluidos")
    database.save_to_milvus(included_articles)
    
    # Guardar en sesión
    TEMP_ARTICLES[session_id]["included_articles"] = included_articles
    TEMP_ARTICLES[session_id]["excluded_articles"] = excluded_articles
    
    # Calcular métricas básicas para review.html
    avg_similarity = sum(a.get("similarity", 0) for a in included_articles) / len(included_articles) if included_articles else 0
    unique_journals = len(set(a.get("journal", "Unknown") for a in included_articles))
    
    # ✅ REDIRIGIR A REVIEW.HTML (NO A RESULTS)
    return templates.TemplateResponse("review.html", {
        "request": request,
        "session_id": session_id,
        "question": question,
        "included_articles": included_articles,
        "excluded_articles": excluded_articles,
        "total_included": len(included_articles),
        "total_excluded": len(excluded_articles),
        "avg_similarity": round(avg_similarity * 100, 1),
        "unique_journals": unique_journals
    })

@app.post("/generate_synthesis")
async def generate_synthesis_endpoint(request: Request):
    """
    ✅ GENERA LA SÍNTESIS FINAL
    ✅ AHORA SÍ VA A results.html
    """
    data = await request.json()
    session_id = int(data.get("sessionId"))
    question = data.get("question")
    
    if session_id not in TEMP_ARTICLES:
        return JSONResponse({"error": "Sesión expirada"}, status_code=400)
    
    session_data = TEMP_ARTICLES[session_id]
    included_articles = session_data.get("included_articles", [])
    
    if not included_articles:
        return JSONResponse({"error": "No hay artículos para sintetizar"}, status_code=400)
    
    logging.info(f"🔬 Generando síntesis para {len(included_articles)} artículos...")
    
    log_prefix = session_data.get("log_prefix", f"session_{session_id}")
    
    # Generar síntesis con RAG
    t_synth_start = time.time()
    rag_results = rag_pipeline.retrieve_relevant(question, top_k=min(10, len(included_articles)))
    
    try:
        synth = synthesis.generate_synthesis(rag_results, question)
    except Exception as e:
        logging.error(f"❌ Síntesis: {e}")
        synth = f"⚠️ Error generando síntesis. {len(rag_results)} artículos recuperados."
    
    t_synth = time.time() - t_synth_start
    
    with open(f"logs/{log_prefix}_RESUMEN_EJECUTIVO.txt", "w", encoding="utf-8") as f:
        f.write(synth)
    
    export_to_csv(included_articles, f"{log_prefix}_TABLA_FINAL.csv")
    
    # Calcular métricas finales
    m = {
        "total": len(session_data.get("articles", [])),
        "after_filter": len(session_data.get("filtered_articles", [])),
        "after_dedup": len(session_data.get("dedup_articles", [])),
        "relevant": len(session_data.get("relevant_articles", [])),
        "final_included": len(included_articles),
        "t_search": session_data.get("search_time", 0),
        "t_filter": session_data.get("t_filter", 0),
        "t_dedup": session_data.get("t_dedup", 0),
        "t_screen": session_data.get("t_screen", 0),
        "t_synth": round(t_synth, 2),
        "total_time": round(
            session_data.get("search_time", 0) + 
            session_data.get("t_filter", 0) + 
            session_data.get("t_dedup", 0) + 
            session_data.get("t_screen", 0) + 
            t_synth, 2
        ),
        "exclusion_filter_percent": round(
            (len(session_data.get("articles", [])) - len(session_data.get("filtered_articles", []))) / 
            len(session_data.get("articles", [])) * 100, 1
        ) if session_data.get("articles") else 0
    }
    
    # Generar plots
    try:
        plots = metrics.generate_plots(m)
    except Exception as e:
        logging.warning(f"⚠️ Error generando plots: {e}")
        plots = {"prisma": "<p>Gráfico no disponible</p>", "time": "<p>Gráfico no disponible</p>"}
    
    # ✅ AHORA SÍ VA A RESULTS.HTML
    return templates.TemplateResponse("results.html", {
        "request": request,
        "metrics": m,
        "synthesis": synth,
        "plots": plots,
        "pdf": "synthesis.pdf",
        "log_files": [
            f"{log_prefix}_log_1_after_filters.csv",
            f"{log_prefix}_log_2_after_dedup.csv",
            f"{log_prefix}_log_3_FINAL_70percent.csv",
            f"{log_prefix}_log_5_excluded_by_researcher.csv",
            f"{log_prefix}_log_6_final_included_by_researcher.csv",
            f"{log_prefix}_TABLA_FINAL.csv"
        ],
        "session_id": session_id,
        "question": question,
        "final_articles": included_articles
    })

@app.get("/export_filtered_list")
async def download_filtered_list(session_id: Optional[int] = None, log_file: Optional[str] = None):
    if not session_id or not log_file:
        return HTMLResponse(content="Falta ID", status_code=400)
    
    safe_filename = Path(log_file).name
    if not safe_filename.startswith(f"session_{session_id}"):
         return HTMLResponse(content="Inválido", status_code=400)

    log_path = Path("logs") / safe_filename
    if not log_path.exists():
        return HTMLResponse(content="No encontrado", status_code=404)

    return FileResponse(log_path, filename=safe_filename, media_type="text/csv")

@app.get("/download")
async def download():
    return FileResponse("sintesis_prisma.pdf", filename="sintesis_prisma.pdf", media_type="application/pdf")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)