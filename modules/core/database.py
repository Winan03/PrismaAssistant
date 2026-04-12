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
import re
import hashlib
from modules.logic.screening import get_embedding

logging.basicConfig(level=logging.INFO)

# ==========================
# MongoDB Atlas (OPCIONAL)
# ==========================
_mongo_client = None
_mongo_available = False

def init_mongo():
    """Inicializa conexión a MongoDB"""
    global _mongo_client, _mongo_available
    if not config.ENABLE_MONGODB:
        _mongo_available = False
        return

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
            if "quota" not in str(e).lower():
                logging.warning(f"⚠️ No se pudo guardar en MongoDB: {e}")
    else:
        if not config.ENABLE_MONGODB:
            pass # Silencio si está deshabilitado
        else:
            logging.info(f"ℹ️ MongoDB no disponible - {len(articles)} artículos solo en memoria")

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

# ============================================================
# 🔍 HEURÍSTICA DE DETECCIÓN DE PDF REAL (v11.3)
# ============================================================
def is_pdf_real(text: str) -> bool:
    """
    Determina si un texto es un PDF completo o solo metadatos/abstracts extensos.
    Requisitos:
    1. Longitud > 6500 caracteres.
    2. Presencia de al menos 2 secciones estructurales académicas.
    """
    if not text: return False
    text_len = len(text)
    if text_len < 6500: return False
    
    # Marcadores de estructura (case insensitive)
    markers = [
        r"methodology|metodología",
        r"results?|resultados",
        r"discussion|discusión",
        r"references|referencias|bibliography",
        r"introduction|introducción",
        r"conclusion|conclusión",
        r"section|sección",
        r"table\s+\d+|figure\s+\d+|tabla\s+\d+|figura\s+\d+"
    ]
    
    found_count = 0
    text_lower = text.lower()
    for m in markers:
        if re.search(m, text_lower):
            found_count += 1
            if found_count >= 2: break
            
    return found_count >= 2

# ============================================================

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
    Guarda artículos en ChromaDB con fragmentación (chunking) para RAG de alta fidelidad.
    ✅ MEJORAS Q1:
    1. Fragmentación semántica (~1000 chars por chunk)
    2. Metadata heredada en cada fragmento (año, autor, título)
    3. Permite búsqueda precisa de evidencia técnica
    """
    if not articles:
        return
    
    try:
        collection = ensure_collection()

        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        import hashlib
        successful_chunks = 0
        skipped_articles = 0
        CHUNK_SIZE = 1200
        OVERLAP = 200
        seen_ids_in_batch = set()

        for i, a in enumerate(articles):
            title = a.get("title", "")
            if not title: continue
            
            author = ""
            if a.get("authors"):
                author = a.get("authors")[0]
            elif a.get("author"):
                author = a.get("author")
            
            # v11.3: Determinación ROBUSTA (Longitud + Estructura)
            current_full_text = a.get("full_text", "")
            is_new_text_pdf = is_pdf_real(current_full_text)
            
            # ✅ ID Robusto basado en Hash de Título + Autor + Año (Entidad Única)
            unique_str = f"{title}_{str(a.get('year',''))}_{author}"
            title_hash = hashlib.md5(unique_str.encode('utf-8', errors='ignore')).hexdigest()
            first_chunk_id = f"chunk_{title_hash}_0"

            # ✅ IDEMPOTENCIA INTELIGENTE: Verificar si el artículo ya tiene fragmentos
            try:
                existing = collection.get(ids=[first_chunk_id])
                if existing and existing['ids']:
                    # v11.0: Solo saltar si el nuevo texto NO es mejor que el que ya tenemos
                    # Si el nuevo es PDF y el de la DB es Abstract, PROCEDEMOS (sobrescribimos)
                    db_meta = existing['metadatas'][0]
                    # Ojo: corregimos falsos positivos históricos verificando que el metadato era True
                    was_full_in_db = db_meta.get('is_full_text') == "True"
                    
                    if was_full_in_db and not is_new_text_pdf:
                        # Ya tenemos PDF (o eso cree la DB) y el nuevo es abstract, saltar
                        skipped_articles += 1
                        continue
                    elif was_full_in_db == is_new_text_pdf:
                        # Calidad idéntica, saltar para evitar embeddings innecesarios
                        skipped_articles += 1
                        continue
                    else:
                        logging.info(f"🔄 Re-indexando: Actualizando abstract -> PDF real para: {title[:30]}...")
                        # Procedemos a indexar, el .upsert() se encargará de los IDs
            except:
                pass

            full_text = current_full_text
            abstract = a.get("abstract", "")
            year = str(a.get("year", ""))
            
            # Texto base
            base_text = full_text if is_new_text_pdf else abstract
            if not base_text: base_text = title
            
            # v14.0: Fragmentación Semántica Priorizada (Párrafos)
            text_len = len(base_text)
            chunks = []
            
            # 1. Intentar dividir por párrafos dobles o simples
            paragraphs = re.split(r'\n\n+', base_text)
            current_chunk = ""
            
            for p in paragraphs:
                if len(current_chunk) + len(p) < CHUNK_SIZE:
                    current_chunk += p + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                    # Si el párrafo solo es gigante, cortarlo a machete (fallback)
                    if len(p) > CHUNK_SIZE:
                        for start in range(0, len(p), CHUNK_SIZE - OVERLAP):
                            chunks.append(p[start:start + CHUNK_SIZE])
                        current_chunk = ""
                    else:
                        current_chunk = p + "\n\n"
            
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Generar vectores e IDs para cada chunk
            for j, chunk_text in enumerate(chunks):
                try:
                    text_for_embedding = f"Articulo: {title}. Pasaje: {chunk_text}"
                    embedding = get_embedding(text_for_embedding)
                    
                    if embedding is not None:
                        chunk_id = f"chunk_{title_hash}_{j}"
                        
                        # Evitar duplicados en el mismo batch
                        if chunk_id in seen_ids_in_batch:
                            continue
                        seen_ids_in_batch.add(chunk_id)

                        ids.append(chunk_id)
                        embeddings.append(embedding.tolist())
                        documents.append(chunk_text)
                        metadatas.append({
                            "title": title[:200],
                            "author": author[:100],
                            "year": year,
                            "original_article_idx": i,
                            "chunk_index": j,
                            "is_full_text": "True" if is_new_text_pdf else "False" # Etiqueta estricta
                        })
                        successful_chunks += 1
                except Exception as e:
                    continue
        
        if skipped_articles > 0:
            logging.info(f"♻️ Idempotencia ChromaDB: {skipped_articles} artículos ya actualizados saltados.")

        if ids:
            # v10.2: Usar upsert para permitir actualizaciones de abstract -> full-text
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logging.info(f"✅ ChromaDB: Guardados/Actualizados {successful_chunks} fragmentos para {len(articles) - skipped_articles} artículos.")
            
    except Exception as e:
        logging.error(f"❌ Error en save_to_milvus (Chunking): {e}")

def recover_full_text(article: dict) -> str:
    """
    Reconstruye el texto completo a partir de los chunks en ChromaDB.
    """
    import hashlib
    try:
        title = article.get("title", "")
        author = ""
        authors = article.get("authors")
        if authors and isinstance(authors, list):
            for a in authors:
                if a: 
                    author = a
                    break
        elif article.get("author"):
            author = article.get("author")
            
        # v10.6: Identidad Robusta (Idempotencia)
        doi = article.get("doi", "")
        if doi and isinstance(doi, str) and len(doi.strip()) > 5:
            # Prioridad 1: DOI (limpio y en minúsculas)
            unique_str = doi.strip().lower()
        else:
            # Prioridad 2: Título Normalizado + Año + Autor
            import re
            title_clean = re.sub(r'[^\w\s]', '', title.lower()).strip()
            unique_str = f"{title_clean}_{str(article.get('year',''))}_{author}"
            
        title_hash = hashlib.md5(unique_str.encode('utf-8', errors='ignore')).hexdigest()
        
        collection = ensure_collection()
        all_text = []
        
        # Recuperar hasta 100 chunks (~120k caracteres)
        for j in range(100):
            chunk_id = f"chunk_{title_hash}_{j}"
            try:
                res = collection.get(ids=[chunk_id])
                if res and res.get('documents') and len(res['documents']) > 0:
                    all_text.append(res['documents'][0])
                else: 
                    break
            except: 
                break
        
        full_text = "\n".join(all_text)
        if len(full_text) > 200:
            logging.info(f" 🧩 Texto recuperado de ChromaDB ({len(full_text)} chars) para: {title[:40]}...")
            return full_text
        return ""
    except Exception as e:
        logging.error(f"❌ Error recuperando texto de ChromaDB: {e}")
        return ""

# Inicializar MongoDB al importar
init_mongo()
