import logging
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util

# ============================================================
# 🧠 CONFIGURACIÓN DEL MODELO (SPECTER2)
# ============================================================
MODEL_NAME = 'allenai/specter2_base' 
_model = None

def get_model():
    global _model
    if _model is None:
        logging.info(f"🧠 Cargando modelo especializado: {MODEL_NAME}...")
        try:
            _model = SentenceTransformer(MODEL_NAME)
        except Exception as e:
            logging.error(f"❌ Error cargando SPECTER2: {e}")
            _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def get_embedding(text: str) -> np.ndarray:
    try:
        model = get_model()
        return model.encode(text, convert_to_numpy=True)
    except Exception as e:
        logging.error(f"❌ Error embedding único: {e}")
        return None

def screen_articles(articles: List[Dict], query: str, threshold: float = 0.70, max_results: int = 50) -> List[Dict]:
    """
    Filtra artículos usando SPECTER2 con garantía de Top 50.
    
    ESTRATEGIA:
    1. Calcula similitud semántica
    2. Prioriza artículos CON URL/PDF
    3. Asegura retornar exactamente 50 artículos (o menos si no hay suficientes)
    """
    if not articles: 
        return []

    model = get_model()
    
    # 1. Preparar textos (Título + [SEP] + Abstract)
    texts = [f"{art.get('title', '')} [SEP] {art.get('abstract', '')}" for art in articles]
    
    logging.info(f"🚀 Screening semántico de {len(texts)} artículos...")
    
    try:
        query_embedding = model.encode(query, convert_to_tensor=True)
        corpus_embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        scores = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()
        
        for i, art in enumerate(articles):
            art['similarity'] = float(scores[i])

    except Exception as e:
        logging.error(f"❌ Error crítico en screening: {e}")
        # Si falla, asignar similitud por orden original
        for i, art in enumerate(articles):
            art['similarity'] = 1.0 - (i / len(articles))
        
    # ============================================================
    # 🎯 ESTRATEGIA DE SELECCIÓN
    # ============================================================
    
    # 2. Separar artículos con URL vs sin URL
    with_url = [art for art in articles if art.get('url') and len(str(art.get('url'))) > 10]
    without_url = [art for art in articles if not (art.get('url') and len(str(art.get('url'))) > 10)]
    
    # 3. Ordenar ambos grupos por similitud
    with_url.sort(key=lambda x: x['similarity'], reverse=True)
    without_url.sort(key=lambda x: x['similarity'], reverse=True)
    
    # 4. ESTRATEGIA ADAPTATIVA:
    #    - Primero: Todos los artículos con URL (hasta 50)
    #    - Si faltan: Completar con los mejores sin URL
    
    final_selection = []
    
    # Paso 1: Agregar artículos con URL (hasta 50)
    for art in with_url[:max_results]:
        if art['similarity'] >= 0.60:  # Umbral mínimo relajado
            final_selection.append(art)
    
    # Paso 2: Si tenemos menos de 50, completar con sin URL
    if len(final_selection) < max_results:
        needed = max_results - len(final_selection)
        for art in without_url[:needed]:
            if art['similarity'] >= 0.55:  # Umbral aún más bajo para completar
                final_selection.append(art)
    
    # Paso 3: Si AÚN no llegamos a 50, tomar los mejores por ranking puro
    if len(final_selection) < max_results:
        needed = max_results - len(final_selection)
        all_remaining = [art for art in articles if art not in final_selection]
        all_remaining.sort(key=lambda x: x['similarity'], reverse=True)
        final_selection.extend(all_remaining[:needed])
    
    # ============================================================
    # ✂️ LÍMITE DURO: MÁXIMO 50
    # ============================================================
    final_selection.sort(key=lambda x: x['similarity'], reverse=True)
    
    if len(final_selection) > max_results:
        logging.info(f"✂️ Limitando resultados: {len(final_selection)} -> {max_results}")
        final_selection = final_selection[:max_results]
    
    # ============================================================
    # 📊 REPORTE DE SELECCIÓN
    # ============================================================
    count_with_url = sum(1 for art in final_selection if art.get('url'))
    count_without_url = len(final_selection) - count_with_url
    avg_similarity = np.mean([art['similarity'] for art in final_selection]) if final_selection else 0
    
    logging.info(f"""
    ✅ SCREENING COMPLETADO:
       📄 Total seleccionados: {len(final_selection)}
       🔗 Con URL/PDF: {count_with_url} ({count_with_url/len(final_selection)*100:.1f}%)
       ❌ Sin URL: {count_without_url}
       📊 Similitud promedio: {avg_similarity*100:.1f}%
       🎯 Rango de similitud: {min(art['similarity'] for art in final_selection)*100:.1f}% - {max(art['similarity'] for art in final_selection)*100:.1f}%
    """)
    
    return final_selection