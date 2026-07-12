import logging
import numpy as np
from typing import List, Dict, Tuple
from app.llm.embedding_service import get_single_embedding, get_embeddings

logger = logging.getLogger(__name__)

def apply_stage1_fast_filter(
    articles: List[Dict],
    question: str,
    threshold: float = 0.45,
    min_abstract_len: int = 150
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stage 1: Filtro rápido por similitud de coseno entre abstract y pregunta de investigación.
    
    Reglas:
    1. Si len(abstract) < min_abstract_len (150 chars), el artículo se salta el filtro y pasa
       automáticamente a la siguiente fase para evitar sesgo por abstract truncado o ausente.
    2. En caso contrario, se calcula el embedding del abstract y la similitud de coseno contra
       el embedding de la pregunta de investigación.
    3. Si la similitud >= threshold (0.45), el artículo pasa.
    4. Si es menor, se marca como excluido por baja similitud.
    
    Returns:
        Tuple[List[Dict], List[Dict]]: (passed_articles, excluded_articles)
    """
    if not articles:
        return [], []
    
    logger.info("⚡ [Stage 1] Iniciando filtro rápido de similitud sobre %d artículos...", len(articles))
    
    # 1. Obtener embedding de la pregunta de investigación
    try:
        rq_embedding = get_single_embedding(question)
        rq_norm = np.linalg.norm(rq_embedding)
    except Exception as e:
        logger.error("❌ [Stage 1] Error al generar embedding de la RQ: %s. Pasando todos los artículos.", e)
        # Fallback de seguridad: si falla el embedding, pasan todos
        for a in articles:
            a['passed_stage1'] = True
            a['similarity_stage1'] = 1.0
            a['stage1_bypass'] = True
            a['exclusion_reason'] = None
        return articles, []

    # 2. Separar los que se saltan el filtro por longitud de los que se evalúan
    to_evaluate: List[Dict] = []
    passed: List[Dict] = []
    excluded: List[Dict] = []
    
    for a in articles:
        abstract = (a.get('abstract') or '').strip()
        if len(abstract) < min_abstract_len:
            # Bypass por abstract demasiado corto / ausente
            a['passed_stage1'] = True
            a['similarity_stage1'] = 1.0
            a['stage1_bypass'] = True
            a['exclusion_reason'] = None
            passed.append(a)
        else:
            a['stage1_bypass'] = False
            to_evaluate.append(a)
            
    logger.info("🔍 [Stage 1] Bypass automático (abstract < %d chars): %d artículos", min_abstract_len, len(passed))
    logger.info("🔍 [Stage 1] Evaluando similitud para %d artículos...", len(to_evaluate))

    if to_evaluate:
        # Extraer abstracts y generar embeddings en lote para velocidad óptima
        abstracts = [(a.get('abstract') or '') for a in to_evaluate]
        try:
            abs_embeddings = get_embeddings(abstracts)
            
            for idx, a in enumerate(to_evaluate):
                abs_emb = abs_embeddings[idx]
                abs_norm = np.linalg.norm(abs_emb)
                
                if rq_norm == 0 or abs_norm == 0:
                    similarity = 0.0
                else:
                    similarity = float(np.dot(rq_embedding, abs_emb) / (rq_norm * abs_norm))
                
                a['similarity_stage1'] = round(similarity, 4)
                
                if similarity >= threshold:
                    a['passed_stage1'] = True
                    a['exclusion_reason'] = None
                    passed.append(a)
                else:
                    a['passed_stage1'] = False
                    a['exclusion_reason'] = f"Baja similitud semántica en abstract (Similitud: {similarity:.2f} < {threshold:.2f})"
                    excluded.append(a)
        except Exception as e:
            logger.error("❌ [Stage 1] Error al generar embeddings por lote: %s. Pasando restantes.", e)
            for a in to_evaluate:
                a['passed_stage1'] = True
                a['similarity_stage1'] = 1.0
                a['stage1_bypass'] = True
                passed.append(a)

    logger.info("✅ [Stage 1] Completado. Pasaron: %d | Excluidos: %d", len(passed), len(excluded))
    return passed, excluded
