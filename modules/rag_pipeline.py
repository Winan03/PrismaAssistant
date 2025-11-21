"""
RAG con ChromaDB - Recuperación semántica de artículos
"""
from modules.database import ensure_collection
from modules.screening import get_embedding
import config
import logging

def retrieve_relevant(query, top_k=8):
    """
    Recupera artículos más relevantes usando búsqueda vectorial en ChromaDB.
    
    Args:
        query: Pregunta de investigación
        top_k: Número de artículos a recuperar
        
    Returns:
        Lista de artículos con title, abstract, doi, score
    """
    try:
        collection = ensure_collection()

        # Generar embedding de la consulta
        query_emb = get_embedding(query)
        if query_emb is None:
            logging.error("❌ No se pudo generar embedding de la consulta")
            return []

        logging.info(f"🔍 Buscando en ChromaDB: top {top_k} resultados")

        # Búsqueda semántica
        results = collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=min(top_k, collection.count()),
            include=["metadatas", "documents", "distances"]
        )

        # Procesar resultados
        articles = []
        if results and results.get("metadatas") and len(results["metadatas"]) > 0:
            metadatas = results["metadatas"][0]
            distances = results.get("distances", [[]])[0]
            
            for i, metadata in enumerate(metadatas):
                # ChromaDB devuelve distancia (menor = más similar)
                # Convertir a score (mayor = más similar)
                distance = distances[i] if i < len(distances) else 1.0
                score = 1.0 - distance  # Normalizar a 0-1
                
                articles.append({
                    "title": metadata.get("title", ""),
                    "abstract": metadata.get("abstract", ""),
                    "doi": metadata.get("doi", ""),
                    "score": round(score, 3),
                    "metadata": {
                        "year": int(metadata.get("year", 0))
                    }
                })

        logging.info(f"✅ Recuperados {len(articles)} artículos de ChromaDB")
        return articles

    except Exception as e:
        logging.error(f"❌ Error en RAG: {e}")
        return []