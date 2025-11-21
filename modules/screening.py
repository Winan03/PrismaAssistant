from sentence_transformers import SentenceTransformer, util
import config
import logging
import numpy as np

from modules.screening_ai import translate_question_to_english

# Cargar modelo una sola vez
_model = SentenceTransformer(config.EMBEDDING_MODEL)

def get_embedding(text):
    """Genera embedding normalizado"""
    if not text or not text.strip():
        return None
    try:
        return _model.encode(text, normalize_embeddings=True)
    except Exception as e:
        logging.error(f"❌ Error generando embedding: {e}")
        return None


def screen_articles(articles, query_input):
    if not articles:
        logging.warning("⚠️ Sin artículos para cribar")
        return []

    # Construir query embedding
    if isinstance(query_input, list) and query_input:
        # Ya son términos en inglés (de query_expander)
        query_text = ". ".join(query_input)
        logging.info(f"🧬 Usando {len(query_input)} términos expandidos en inglés.")
    else:
        # ✅ NUEVO: Es una pregunta directa → traducir
        original_query = str(query_input)
        query_text = translate_question_to_english(original_query)
        
        if query_text != original_query:
            logging.info(f"🌍 Pregunta traducida para embeddings:")
            logging.info(f"   Original: {original_query[:100]}")
            logging.info(f"   Inglés: {query_text[:100]}")
        else:
            logging.info(f"🧬 Pregunta ya en inglés: {query_text[:100]}")
    
    q_emb = get_embedding(query_text)
    if q_emb is None:
        logging.error("❌ No se pudo generar embedding de la query")
        return []
    
    # ✅ PREPARAR TEXTOS PRIORIZANDO FULL_TEXT
    corpus_texts = []
    valid_articles = []
    text_sources = []  # Para estadísticas
    
    for a in articles:
        # ✅ PRIORIDAD: full_text > abstract > title
        full_text = a.get("full_text", "")
        abstract = a.get("abstract", "")
        title = a.get("title", "")
        
        if full_text:
            # Usar texto completo (límite 15k chars para embedding)
            text = f"{title} {full_text[:15000]}"
            source = "PDF"
        elif abstract:
            text = f"{title} {abstract}"
            source = "Abstract"
        else:
            text = title
            source = "Title only"
        
        if text.strip():
            corpus_texts.append(text)
            valid_articles.append(a)
            text_sources.append(source)
        else:
            a["similarity"] = 0.0
            a["relevance"] = "Sin texto"
            a["excluded_reason"] = "Sin contenido válido"

    if not valid_articles:
        logging.warning("⚠️ No hay artículos con texto válido")
        return []

    # ✅ ESTADÍSTICAS DE FUENTES
    from collections import Counter
    source_counts = Counter(text_sources)
    logging.info(f"📊 Fuentes de texto:")
    for source, count in source_counts.items():
        logging.info(f"   - {source}: {count} artículos")

    # Batch encoding
    logging.info(f"🚀 Generando {len(corpus_texts)} embeddings en batch...")
    try:
        corpus_embeddings = _model.encode(
            corpus_texts, 
            normalize_embeddings=True, 
            show_progress_bar=True,
            batch_size=32  # ✅ Optimizar para PDFs grandes
        )
    except Exception as e:
        logging.error(f"❌ Error en batch encoding: {e}")
        return []

    # Calcular similitudes
    all_sims = util.cos_sim(q_emb, corpus_embeddings)[0].cpu().numpy()

    relevant_articles = []
    excluded_articles = []
    
    # ✅ THRESHOLD DEL 70% (lo que pidió tu profesor)
    THRESHOLD = 0.70
    
    for i, a in enumerate(valid_articles):
        sim = float(all_sims[i])
        a["similarity"] = round(sim, 3)
        a["text_source"] = text_sources[i]  # ✅ Añadir para debugging

        if sim >= THRESHOLD:
            if sim >= 0.85:
                a["relevance"] = "⭐ Altamente relevante"
            elif sim >= 0.75:
                a["relevance"] = "✅ Muy relevante"
            else:
                a["relevance"] = "📌 Relevante"
            relevant_articles.append(a)
        else:
            a["relevance"] = "❌ Poco relevante"
            a["excluded_reason"] = f"Relevancia insuficiente ({int(sim*100)}% < {int(THRESHOLD*100)}%)"
            excluded_articles.append(a)

    relevant_articles.sort(key=lambda x: x["similarity"], reverse=True)
    
    # ✅ LOGGING MEJORADO
    if relevant_articles:
        scores = [a["similarity"] for a in relevant_articles]
        logging.info(f"   📊 Scores relevantes: min={min(scores):.2f}, max={max(scores):.2f}, promedio={np.mean(scores):.2f}")
        
        # ✅ ESTADÍSTICAS POR FUENTE EN RELEVANTES
        relevant_sources = Counter([a["text_source"] for a in relevant_articles])
        logging.info(f"   📚 Artículos relevantes por fuente:")
        for source, count in relevant_sources.items():
            logging.info(f"      - {source}: {count}")
    
    logging.info(f"✅ Screening completado:")
    logging.info(f"   - Total analizados: {len(valid_articles)}")
    logging.info(f"   - PASAN (>={int(THRESHOLD*100)}%): {len(relevant_articles)}")
    logging.info(f"   - EXCLUIDOS (<{int(THRESHOLD*100)}%): {len(excluded_articles)}")
    
    # Guardar excluidos para log
    if excluded_articles:
        from utils.export import export_to_csv
        export_to_csv(excluded_articles, "log_excluded_by_similarity.csv")
    
    return relevant_articles