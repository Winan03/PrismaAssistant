"""
1. Guarda texto completo del PDF en ChromaDB (no solo 1000 chars)
2. Añade campo 'full_text_length' para debugging
3. Mejor manejo de errores
"""
from pymongo import MongoClient
import chromadb
from chromadb.config import Settings
import config
import logging
import os

logging.basicConfig(level=logging.INFO)

# ==========================
# MongoDB Atlas (OPCIONAL)
# ==========================
_mongo_client = None
_mongo_available = False

def init_mongo():
    """Inicializa conexión a MongoDB"""
    global _mongo_client, _mongo_available
    try:
        _mongo_client = MongoClient(
            config.MONGODB_URI, 
            serverSelectionTimeoutMS=5000
        )
        _mongo_client.server_info()
        _mongo_available = True
        logging.info("✅ MongoDB Atlas conectado")
    except Exception as e:
        _mongo_available = False
        logging.warning(f"⚠️ MongoDB no disponible: {e}")

def get_mongo_collection():
    if not _mongo_available:
        return None
    if _mongo_client is None:
        init_mongo()
    if _mongo_available:
        db = _mongo_client.get_database()
        return db.articles
    return None

def save_to_mongo(articles):
    if not articles:
        return
    
    collection = get_mongo_collection()
    if collection is not None:
        try:
            collection.insert_many(articles)
            logging.info(f"✅ Guardados {len(articles)} artículos en MongoDB")
        except Exception as e:
            logging.warning(f"⚠️ No se pudo guardar en MongoDB: {e}")
    else:
        logging.info(f"ℹ️ MongoDB deshabilitado - {len(articles)} artículos solo en memoria")

# ==========================
# ChromaDB (Reemplazo de Milvus)
# ==========================
_chroma_client = None
_chroma_collection = None

def get_chroma_client():
    """Obtiene cliente de ChromaDB"""
    global _chroma_client
    if _chroma_client is None:
        chroma_dir = os.path.join(os.getcwd(), "chroma_db")
        os.makedirs(chroma_dir, exist_ok=True)
        
        _chroma_client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        logging.info(f"✅ ChromaDB conectado en: {chroma_dir}")
    return _chroma_client

def ensure_collection():
    """Asegura que la colección existe"""
    global _chroma_collection
    
    if _chroma_collection is not None:
        return _chroma_collection
    
    client = get_chroma_client()
    coll_name = config.MILVUS_COLLECTION
    
    try:
        _chroma_collection = client.get_collection(name=coll_name)
        logging.info(f"✅ Colección '{coll_name}' cargada")
    except:
        _chroma_collection = client.create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine"}
        )
        logging.info(f"✅ Colección '{coll_name}' creada")
    
    return _chroma_collection

def save_to_milvus(articles):
    """
    Guarda artículos en ChromaDB con texto completo del PDF.
    ✅ CAMBIOS CRÍTICOS:
    1. Guarda hasta 30,000 caracteres (no 1000)
    2. Añade metadata 'full_text_length' para verificar
    3. Prioriza full_text > abstract
    """
    if not articles:
        logging.info("⚠️ Sin artículos para guardar en ChromaDB")
        return
    
    try:
        from modules.screening import get_embedding 
        collection = ensure_collection()

        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        successful_saves = 0
        failed_saves = 0

        for i, a in enumerate(articles):
            # ✅ PRIORIZAR FULL_TEXT (del PDF)
            full_text = a.get("full_text", "")
            abstract = a.get("abstract", "")
            title = a.get("title", "")
            
            # Construir texto para embedding
            if full_text:
                # Si hay PDF, usar título + full_text
                text_for_embedding = f"{title} {full_text}"
                source = "PDF"
            elif abstract:
                # Fallback a abstract
                text_for_embedding = f"{title} {abstract}"
                source = "Abstract"
            else:
                # Solo título (caso extremo)
                text_for_embedding = title
                source = "Title only"
            
            if not text_for_embedding.strip():
                failed_saves += 1
                continue
            
            # ✅ LÍMITE MÁS GENEROSO: 30,000 caracteres
            text_limited = text_for_embedding[:30000]
            text_length = len(text_for_embedding)

            try:
                embedding = get_embedding(text_limited)
                if embedding is None:
                    failed_saves += 1
                    continue

                article_id = f"article_{hash(title)}_{i}"
                
                ids.append(article_id)
                embeddings.append(embedding.tolist())
                
                # ✅ GUARDAR TEXTO COMPLETO (no solo 1000 chars)
                # ChromaDB acepta hasta ~50k caracteres por documento
                documents.append(text_limited)
                
                # ✅ METADATA MEJORADA
                metadatas.append({
                    "title": title[:500],
                    "doi": a.get("doi", "")[:100],
                    "year": str(a.get("year", 0)),
                    "abstract": abstract[:500],
                    "has_full_text": "True" if full_text else "False",
                    "text_source": source,  # ✅ NUEVO: Para debugging
                    "full_text_length": str(text_length)  # ✅ NUEVO: Verificar extracción
                })
                
                successful_saves += 1
                
            except Exception as e:
                logging.warning(f"⚠️ Error procesando artículo {title[:50]}: {e}")
                failed_saves += 1
                continue

        if ids:
            try:
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
                logging.info(f"✅ Guardados {successful_saves} vectores en ChromaDB")
                
                # ✅ ESTADÍSTICAS DE TEXTO COMPLETO
                with_pdf = sum(1 for m in metadatas if m["has_full_text"] == "True")
                avg_length = sum(int(m["full_text_length"]) for m in metadatas) / len(metadatas)
                
                logging.info(f"   📊 {with_pdf}/{len(metadatas)} con texto completo de PDF")
                logging.info(f"   📏 Longitud promedio: {int(avg_length)} caracteres")
                
                if failed_saves > 0:
                    logging.warning(f"   ⚠️ {failed_saves} artículos fallaron al guardar")
                    
            except Exception as e:
                logging.error(f"❌ Error insertando en ChromaDB: {e}")
        else:
            logging.warning("⚠️ No hay datos válidos para insertar")
            
    except Exception as e:
        logging.error(f"❌ Error en save_to_milvus (ChromaDB): {e}")

# Inicializar MongoDB al importar
init_mongo()