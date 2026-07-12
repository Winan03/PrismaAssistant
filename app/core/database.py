import logging
from pymongo import MongoClient
import chromadb
from chromadb.config import Settings
import config
import os
import re
import hashlib
from app.screening.screening import get_embedding

logging.basicConfig(level=logging.INFO)

# Silenciar telemetría interna de ChromaDB
# Método 1: Nivel CRITICAL en el logger específico de chromadb telemetry
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product_telemetry").setLevel(logging.CRITICAL)

# Método 2: Filtro en los handlers del root (ChromaDB propaga a handlers, no al logger)
class _ChromaTelemetryFilter(logging.Filter):
    def filter(self, record):
        return "Failed to send telemetry event" not in record.getMessage()

_telem_filter = _ChromaTelemetryFilter()
for _h in logging.root.handlers:
    _h.addFilter(_telem_filter)
# También al root logger por si se agregan handlers luego
logging.getLogger().addFilter(_telem_filter)

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

class UnifiedChromaEmbeddingFunction:
    """
    Custom embedding function class for ChromaDB that wraps our unified get_embeddings.
    This prevents ChromaDB from downloading or loading the default all-MiniLM-L6-v2.
    """
    def __call__(self, input: list) -> list:
        from app.llm.embedding_service import get_embeddings
        embs = get_embeddings(input)
        return embs.tolist()

_embedding_fn = UnifiedChromaEmbeddingFunction()

def ensure_collection():
    """
    Asegura que la colección existe y tiene las dimensiones correctas.
    v18.0: Fix robusto para VPS — verifica dimensiones con test-upsert cuando
    peek() no retorna embeddings (colección vacía o con embeddings no incluidos).
    """
    global _chroma_collection
    
    if _chroma_collection is not None:
        return _chroma_collection
    
    client = get_chroma_client()
    coll_name = config.MILVUS_COLLECTION
    expected_dim = config.EMBEDDING_DIM
    
    def _delete_and_recreate(reason: str):
        """Helper para borrar y recrear la colección limpiamente."""
        logging.warning(f"⚠️ {reason}. Recreando colección '{coll_name}' con dim={expected_dim}...")
        try:
            client.delete_collection(name=coll_name)
        except Exception:
            pass
        coll = client.create_collection(
            name=coll_name,
            embedding_function=_embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        logging.info(f"✅ Colección '{coll_name}' recreada (dim={expected_dim})")
        return coll

    try:
        existing = client.get_collection(name=coll_name, embedding_function=_embedding_fn)

        # --- Intento 1: verificar vía peek() ---
        sample = existing.peek(limit=1)
        if sample and sample.get("embeddings") and len(sample["embeddings"]) > 0:
            actual_dim = len(sample["embeddings"][0])
            if actual_dim != expected_dim:
                _chroma_collection = _delete_and_recreate(
                    f"Colección tiene dim={actual_dim}, modelo produce dim={expected_dim}"
                )
                return _chroma_collection
            # Dimensiones OK
            _chroma_collection = existing
            logging.info(f"✅ Colección '{coll_name}' cargada (dim verificada={expected_dim})")
            return _chroma_collection

        # --- Intento 2: peek() vino vacío — hacer test-upsert para verificar dimensión ---
        # Esto ocurre cuando la colección existe pero fue creada con include=[] o está vacía.
        logging.info(f"🔍 peek() sin embeddings — verificando dimensión con test-upsert...")
        _TEST_ID = "__dim_check_probe__"
        test_vector = [0.0] * expected_dim
        try:
            existing.upsert(
                ids=[_TEST_ID],
                embeddings=[test_vector],
                documents=["probe"],
                metadatas=[{"probe": "true"}]
            )
            # Si llegó aquí sin error, la colección acepta la dimensión correcta — limpiar probe
            try:
                existing.delete(ids=[_TEST_ID])
            except Exception:
                pass
            _chroma_collection = existing
            logging.info(f"✅ Colección '{coll_name}' compatible con dim={expected_dim} (test-upsert OK)")
        except Exception as probe_err:
            probe_msg = str(probe_err)
            if "dimension" in probe_msg.lower() or "dimensionality" in probe_msg.lower():
                _chroma_collection = _delete_and_recreate(
                    f"Test-upsert reveló incompatibilidad de dimensión: {probe_msg[:120]}"
                )
            else:
                # Error desconocido en el probe — aceptar la colección de todas formas
                logging.warning(f"⚠️ Test-upsert falló con error no-dimensional: {probe_msg[:120]}. Aceptando colección.")
                _chroma_collection = existing

    except Exception:
        # La colección no existe — crearla
        _chroma_collection = client.create_collection(
            name=coll_name,
            embedding_function=_embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        logging.info(f"✅ Colección '{coll_name}' creada (dim={expected_dim})")
    
    return _chroma_collection

def save_to_milvus(articles):
    """
    Guarda artículos en ChromaDB con fragmentación (chunking) para RAG de alta fidelidad.
    v17.0: Batch embeddings — una sola llamada al modelo para todos los chunks (5-10x más rápido).
    """
    if not articles:
        return

    try:
        collection = ensure_collection()
        from app.llm.embedding_service import get_embeddings as _batch_embed

        texts_to_embed = []
        chunk_info = []  # lista de (chunk_id, metadata_dict, doc_text)

        CHUNK_SIZE = 1200
        OVERLAP = 200
        MAX_CHUNKS_PER_ARTICLE = 40  # v17.1: Evitar que survey papers de 400k chars generen 300+ chunks
        seen_ids_in_batch = set()
        skipped_articles = 0

        for i, a in enumerate(articles):
            title = a.get("title", "")
            if not title:
                continue

            author = ""
            if a.get("authors"):
                author = a.get("authors")[0]
            elif a.get("author"):
                author = a.get("author")

            current_full_text = a.get("full_text", "")
            is_new_text_pdf = is_pdf_real(current_full_text)

            unique_str = f"{title}_{str(a.get('year', ''))}_{author}"
            title_hash = hashlib.md5(unique_str.encode('utf-8', errors='ignore')).hexdigest()
            first_chunk_id = f"chunk_{title_hash}_0"

            # ✅ Idempotencia inteligente: sólo saltamos si la calidad en DB es igual o mejor
            try:
                existing = collection.get(ids=[first_chunk_id])
                if existing and existing['ids']:
                    db_meta = existing['metadatas'][0]
                    was_full_in_db = db_meta.get('is_full_text') == "True"

                    if was_full_in_db and not is_new_text_pdf:
                        # DB ya tiene PDF completo, el nuevo es abstract → saltar
                        skipped_articles += 1
                        continue
                    elif was_full_in_db == is_new_text_pdf:
                        # Misma calidad → saltar
                        skipped_articles += 1
                        continue
                    else:
                        logging.info(f"🔄 Re-indexando: Actualizando abstract -> PDF real para: {title[:30]}...")
                        # Continuamos para sobrescribir vía upsert
            except Exception:
                pass

            base_text = current_full_text if is_new_text_pdf else a.get("abstract", "")
            if not base_text:
                base_text = title

            # Fragmentación semántica por párrafos
            paragraphs = re.split(r'\n\n+', base_text)
            chunks = []
            current_chunk = ""
            for p in paragraphs:
                if len(current_chunk) + len(p) < CHUNK_SIZE:
                    current_chunk += p + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    if len(p) > CHUNK_SIZE:
                        for start in range(0, len(p), CHUNK_SIZE - OVERLAP):
                            chunks.append(p[start:start + CHUNK_SIZE])
                        current_chunk = ""
                    else:
                        current_chunk = p + "\n\n"
            if current_chunk:
                chunks.append(current_chunk.strip())

            for j, chunk_text in enumerate(chunks):
                if j >= MAX_CHUNKS_PER_ARTICLE:  # v17.1: cap para papers gigantes
                    break
                chunk_id = f"chunk_{title_hash}_{j}"
                if chunk_id in seen_ids_in_batch:
                    continue
                seen_ids_in_batch.add(chunk_id)

                texts_to_embed.append(f"Articulo: {title}. Pasaje: {chunk_text}")
                chunk_info.append((chunk_id, {
                    "title": title[:200],
                    "author": author[:100],
                    "year": str(a.get("year", "")),
                    "original_article_idx": i,
                    "chunk_index": j,
                    "is_full_text": "True" if is_new_text_pdf else "False"
                }, chunk_text))

        if skipped_articles > 0:
            logging.info(f"♻️ Idempotencia ChromaDB: {skipped_articles} artículos ya actualizados saltados.")

        if not texts_to_embed:
            return

        # ✅ v17.0: BATCH EMBEDDINGS — una sola llamada al modelo para TODOS los chunks
        # Antes: get_embedding() × N (cientos de llamadas: ~1 seg c/u → minutos)
        # Ahora: get_embeddings(all_texts) → SentenceTransformer vectoriza en batch → 5-10x más rápido
        logging.info(f"⚡ Generando embeddings en batch para {len(texts_to_embed)} chunks...")
        batch_result = _batch_embed(texts_to_embed)  # shape: (N, 768)

        ids_final, embs_final, metas_final, docs_final = [], [], [], []
        for idx, (chunk_id, meta, doc) in enumerate(chunk_info):
            emb = batch_result[idx]
            if emb is not None:
                ids_final.append(chunk_id)
                embs_final.append(emb.tolist())
                metas_final.append(meta)
                docs_final.append(doc)

        # ✅ v31: BATCH UPSERT — Dividimos los chunks en grupos de 2000
        # Evita el error 'Batch size exceeds maximum batch size' (límite de 5461)
        # permitiendo procesar 20k+ artículos sin que ChromaDB explote.
        BATCH_UPSERT_SIZE = 2000
        if ids_final:
            for start_idx in range(0, len(ids_final), BATCH_UPSERT_SIZE):
                end_idx = start_idx + BATCH_UPSERT_SIZE
                collection.upsert(
                    ids=ids_final[start_idx:end_idx],
                    embeddings=embs_final[start_idx:end_idx],
                    metadatas=metas_final[start_idx:end_idx],
                    documents=docs_final[start_idx:end_idx]
                )
            logging.info(f"✅ ChromaDB: Guardados {len(ids_final)} fragmentos en lotes para {len(articles) - skipped_articles} artículos.")

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
