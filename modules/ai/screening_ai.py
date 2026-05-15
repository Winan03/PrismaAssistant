import logging
import os
import time
import json
import hashlib
import re
from typing import Dict, List, Optional
from modules.ai import ai_model

# ============================================================
# 📦 IMPORTACIÓN DE GOOGLE GENERATIVE AI SDK
# ============================================================
try:
    from google import genai
    from google.genai import types
except ImportError:
    logging.error("❌ Falta la librería google-genai. Instálala con: pip install google-genai")
    raise

# Configuración del logging principal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================
# 📇 SILENCIADOR DE RUIDO (PDF WARNINGS)
# ============================================================
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ============================================================
# 📊 PRESUPUESTO DE TOKENS POR COLUMNA (v9.5)
# ============================================================
COLUMN_TOKEN_BUDGET = {
    "default_budget": 600,  # v11.17: Presupuesto base por columna
}

# Límite de palabras por column key (v11.17). Esto se incrusta en el prompt.
COLUMN_WORD_LIMITS = {
    "summary":     120,
    "objectives":   80,
    "methodology": 130,
    "subject":      80,
    "variables":    100,
    "findings":     150,
    "limitations":  80,
    "__default__": 100,
}

# ============================================================
# ⚡ CONFIGURACIÓN DEL MOTOR GEMINI
# ============================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    logging.warning("⚠️ No se encontró GEMINI_API_KEY. Configúralo en tu .env")
    raise ValueError("GEMINI_API_KEY es requerido")

# Inicializar el cliente Gemini
client = genai.Client(api_key=GEMINI_API_KEY)

# Modelos disponibles (Noviembre 2025)
# NIVEL GRATUITO:
# - gemini-2.0-flash: 15 RPM, 1M TPM, 200 RPD ✅ RECOMENDADO
# - gemini-2.5-flash: 10 RPM, 250K TPM, 250 RPD
# - gemini-2.5-flash-lite: 15 RPM, 250K TPM, 1000 RPD (más rápido pero menos potente)
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

logging.info(f"🚀 Usando modelo Gemini: {MODEL_NAME}")
logging.info(f"📊 Límites estimados: ~15 RPM, ~1M TPM (verificar en AI Studio)")

# ============================================================
# 🛠️ UTILIDADES DE CACHE (VOLÁTIL v9.8.1)
# ============================================================
def load_from_cache(article: Dict, column: str, question: str = "", config_hash: str = "") -> str:
    """Caché volátil: busca directamente en el objeto del artículo (Memoria de Sesión)"""
    ai_data = article.get('ai_data', {})
    val = ai_data.get(column)
    if val and "Error" not in str(val):
        return val
    return None

def save_to_cache(article: Dict, column: str, value: str, question: str = "", config_hash: str = ""):
    """Caché volátil: guarda en el objeto del artículo (Memoria de Sesión)"""
    if not value or "Error" in str(value) or "No especificado" in str(value): 
        return
    if 'ai_data' not in article:
        article['ai_data'] = {}
    article['ai_data'][column] = value

# ============================================================
# 🧠 GENERACIÓN DE CONTEXTO
# ============================================================
def prepare_context(abstract: str, full_text: str) -> str:
    """Prioriza texto completo, fallback a abstract. v15.5: Limite aumentado a 150k chars."""
    if full_text and len(full_text) > 1000:
        return full_text[:150000]
    return abstract

# ============================================================
# 🎯 PROMPTS ESPECIALIZADOS (ESTILO ELICIT)
# ============================================================

def get_system_prompt_for_column(column_label: str, research_question: str, specific_instruction: str = "") -> str:
    """
    Genera prompt dinámico basado en las instrucciones de la matriz adaptativa.
    """
    base_instr = f"""
    PROTOCOLO LÓGICA DE HIERRO v15.0 (Iron Logic Protocol):
    1. BLOQUE DE ANÁLISIS VISIBLE (OBLIGATORIO): Debes iniciar tu respuesta con un bloque `<analisis>`. En este bloque, realiza lo siguiente:
       - LISTA DE HERRAMIENTAS HALLADAS: Identifica todas las herramientas mencionadas.
       - FILTRO DE DESCARTE: Clasifica cuáles son del estudio actual y cuáles son "Trabajos Relacionados" o "Herramientas Comparadas/Descartadas".
       - MAPEO DE MÉTRICAS: Vincula Números -> Entidades -> Sección del PDF.
       - FIREWALL DE TÉRMINOS: Si el PDF no menciona "LLM", declara EXPLÍCITAMENTE "Sin presencia de LLM en el texto".
    2. ETIQUETADO DE EVIDENCIA:
       - `[EVIDENCIA: Sección]` para datos del experimento propio (Tablas, Resultados).
       - `[CONTEXTO: Sección]` para citas de trabajos previos o motivación (Introducción).
    3. REGLA DE LA PROPIA DEBILIDAD: En limitaciones, solo reporta lo que los autores admiten fallar. Si es una crítica a otros, ignórala.
    4. CIERRE DE RIGOR: Termina con: ✅ Verificado con Lógica de Hierro.
    """
    
    if specific_instruction:
        return f"{base_instr}\nTAREA ESPECÍFICA PARA '{column_label}': {specific_instruction}"
    
    # Fallback genérico por si falla la config dinámica
    return f"{base_instr}\nExtrae la información más relevante sobre '{column_label}' que responda a la RQ."


def propose_columns_from_rq(research_question: str) -> Dict:
    """
    Detecta el dominio y propone la matriz de extracción (v9.0).
    Estrategia de reintento con backoff exponencial y multi-proveedor.
    """
    system_prompt = """
    Eres un experto en diseño de matrices de extracción para Revisiones Sistemáticas de Literatura (RSL).
    
    CONTEXTO:
    Estas columnas aparecerán en una tabla donde el investigador revisará cada artículo y decidirá 
    si incluirlo o excluirlo del RSL. Es como una ficha de evaluación rápida por artículo.

    Tu tarea: Analiza la RQ y propón ENTRE 5 Y 7 columnas que ayuden al investigador a decidir 
    si el artículo responde a la Pregunta de Investigación.

    REGLAS PARA NOMBRES DE COLUMNAS (campo 'label'):
    - Deben ser CLAROS y AUTOEXPLICATIVOS para cualquier investigador, no tecnicismos.
    - Usa frases cortas orientadas al lector como: "Tipo de estudio", "Herramienta evaluada", 
      "Efecto en falsos positivos", "Metodología aplicada", "Conclusión principal", 
      "Población / Muestra", "Hallazgos clave", etc.
    - EVITA nombres abstractos como "Código analizado", "Variables", "Output" o términos internos.
    - El nombre debe decirle al investigador exactamente QUÉ tipo de información verá en esa celda.

    REGLAS GENERALES:
    - MULTIDOMINIO: las columnas deben ser específicas del dominio detectado (Medicina, Derecho, 
      Ingeniería, Psicología, etc.). No uses siempre los mismos.
    - ATOMICIDAD: cada columna captura UN solo aspecto.
    - CONCISIÓN: las instrucciones ('pregunta') piden 3-5 puntos breves, NUNCA tablas.
    - DETALLE DE CONFIGURACION: cuando la columna involucre elementos comparados, herramientas,
      intervenciones, o componentes del estudio, la 'pregunta' DEBE pedir también: versión usada,
      parámetros clave o configuración específica reportada por los autores (si aplica).
    - DISTINCION MENCIONADO/USADO: la 'pregunta' DEBE pedir distinguir entre elementos solo mencionados
      como contexto o candidatos descartados, vs. los que el artículo realmente utilizó en el estudio.
    - IDIOMA: todos los campos 'label' y 'pregunta' en ESPAÑOL.
    - KEYS: snake_case sin tildes (ej: 'tipo_estudio', 'hallazgos_clave').
    - ICONS: file-text, target, flask, users, bar-chart, check-circle, alert-triangle, cpu, shield, book, layers, activity.
    
    RESPONDE ÚNICAMENTE con el objeto JSON (sin texto antes ni después):
    {
      "tipo_investigacion": "<cuantitativa|cualitativa|mixta|teorica>",
      "dominio": "<dominio detectado en español>",
      "columnas": [
        {"key": "<snake_case>", "label": "<Nombre corto y claro para el investigador>", "icon": "<icon>", "pregunta": "<qué extraer exactamente, en puntos breves, en español>"},
        ...
      ]
    }
    """
    user_prompt = f"RQ: {research_question}\nPropón la matriz de extracción óptima."
    
    logging.info(f"\U0001f9e9 Generando matriz de columnas dinámica | RQ: '{research_question[:80]}...'")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            data = call_ai_api(instruction=user_prompt, user_prompt="", system_prompt=system_prompt, max_tokens=8192)  # v16.2: tokens altos para JSON completo sin truncar
            if data and isinstance(data, dict) and "columnas" in data:
                # Añadir un ID único para la caché de esta configuración
                config_str = json.dumps(data.get("columnas"), sort_keys=True)
                data["config_hash"] = hashlib.md5(config_str.encode()).hexdigest()[:8]
                col_names = [c.get('label', c.get('key')) for c in data.get('columnas', [])]
                logging.info(f"✅ Matriz Dinámica Generada - Dominio: {data.get('dominio')} | Columnas: {col_names}")
                return data
        except Exception as e:
            logging.warning(f"⚠️ Error proponiendo columnas (intento {attempt+1}): {e}")
        
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            logging.info(f"🔄 Reintentando en {wait_time}s...")
            time.sleep(wait_time)

    # Fallback crítico si todo falla (Matriz Estándar)
    logging.warning("\u26a0\ufe0f USANDO MATRIZ FALLBACK GENÉRICA (el LLM no pudo generar columnas específicas para la RQ). Las columnas son genéricas.")
    return {
        "dominio": "No detectado",
        "tipo_investigacion": "No detectada",
        "config_hash": "fallback_v93",
        "columnas": [
            {"key": "summary", "label": "Resumen", "icon": "file-text", "pregunta": "¿Cuál es el aporte principal del estudio?"},
            {"key": "objectives", "label": "Objetivo", "icon": "target", "pregunta": "¿Qué intención tenía el autor?"},
            {"key": "methodology", "label": "Metodología", "icon": "flask", "pregunta": "Describe el procedimiento del estudio sin incluir resultados."},
            {"key": "subject", "label": "Sujeto / Objeto", "icon": "users", "pregunta": "Representa la población o base de datos analizada."},
            {"key": "variables", "label": "Variables / Métricas", "icon": "bar-chart", "pregunta": "¿Qué indicadores se midieron exactamente?"},
            {"key": "findings", "label": "Hallazgos", "icon": "check-circle", "pregunta": "Extrae los resultados y datos numéricos finales."},
            {"key": "limitations", "label": "Limitaciones", "icon": "alert-triangle", "pregunta": "Puntos débiles declarados por el autor."}
        ]
    }


def get_user_prompt_for_column(column: str, context: str, research_question: str) -> str:
    """User prompt v9.7 con pre-verificacion anti-alucinacion de RQ."""
    return f"""CONTENIDO DEL ARTICULO:
{context}

---
INSTRUCCION PARA EVITAR ALUCINACIONES:
Antes de analizar la columna, realiza internamente estos 3 pasos:
[PASO 1 - VOCABULARIO DEL ARTICULO]: Lee el texto y lista los términos EXACTOS con los que el
  ARTICULO denomina los elementos que estudia (copia sus palabras literales, no las sustituyas
  con vocabulario de la RQ). Si el artículo usa un nombre distinto al de la RQ, usa el nombre
  del artículo.
[PASO 2 - VERIFICACION]: Comprueba si lo que encontraste en el Paso 1 coincide exactamente
  con los elementos que la RQ plantea. Si no coinciden, es una discrepancia — anotala.
[PASO 3 - EXTRACCION]: Responde la columna basandote SOLO en lo del Paso 1.
  Si detectaste discrepancia en el Paso 2, indicalo con: "El articulo no aborda [X de la RQ]
  — aborda [lo que realmente estudia segun Paso 1]."

IMPORTANTE: No incluyas los pasos 1 y 2 en tu respuesta. Solo muestra el resultado del Paso 3.

---
AHORA ANALIZA LA COLUMNA: {column}"""

# ============================================================
# ⚡ LLAMADA MULTI-PROVEEDOR (GROQ -> CEREBRAS -> GH MODELS)
# ============================================================

def extract_json_from_text(text: str) -> Dict:
    """Extrae el primer objeto JSON válido encontrado de forma ultra-robusta."""
    if not text or not isinstance(text, str): 
        return None
    
    # 1. Intento directo (limpio)
    try:
        content = text.strip()
        # Eliminar posibles backticks de markdown al inicio/fin
        if content.startswith('```json'): content = content[7:]
        if content.startswith('```'): content = content[3:]
        if content.endswith('```'): content = content[:-3]
        return json.loads(content.strip())
    except:
        pass

    # 2. Regex para bloques JSON { ... } o [ ... ]
    try:
        # Intentar buscar bloques delimitados por llaves o corchetes
        match = re.search(r'(\{.*\}|\[.*\])', text.replace('\n', ' '), re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            # Heurística mínima de balance de llaves para JSON truncados
            if candidate.startswith('{'):
                if candidate.count('{') > candidate.count('}'):
                    candidate += '}'
            elif candidate.startswith('['):
                if candidate.count('[') > candidate.count(']'):
                    candidate += ']'
            return json.loads(candidate)
    except:
        pass
            
    return None

def call_ai_api(instruction: str, user_prompt: str = "", max_tokens: int = 250, system_prompt: str = None) -> Dict:
    """
    Estrategia v9.8: Unificada vía ai_model.py.
    """
    content = ai_model.generate_text(instruction, user_prompt, max_tokens=max_tokens, system_prompt=system_prompt)
    
    if not content or "Error" in content:
        return None
    
    # DEBUG TEMPORAL: ver el contenido exacto que retorna el modelo
    logging.info(f"🔍 [DEBUG call_ai_api] Raw content ({len(content)} chars): {repr(content[:300])}")
        
    data = extract_json_from_text(content)
    return data if data else {"respuesta": content}

# ============================================================
# 🎨 FORMATEO DE RESPUESTAS (ESTILO ELICIT)
# ============================================================

def _recursive_data_to_html(data) -> str:
    """Convierte cualquier estructura de datos (dict, list, str) en HTML limpio sin sintaxis técnica."""
    if not data:
        return ""
    
    if isinstance(data, list):
        items_html = "".join([f"<li class='mb-1 last:mb-0'>{_recursive_data_to_html(item)}</li>" for item in data if item])
        return f"<ul class='list-disc pl-4 space-y-1 my-1'>{items_html}</ul>" if items_html else ""
    
    if isinstance(data, dict):
        # Si es un dict, intentamos extraer el valor principal si existe
        main_keys = ["respuesta", "text", "summary", "content", "objetivo", "value", "resultado"]
        for k in main_keys:
            if k in data and data[k]:
                # Si el valor tiene otros campos, los mostramos abajo
                other_fields = {key: val for key, val in data.items() if key != k and val and val != "No especificado"}
                main_val = _recursive_data_to_html(data[k])
                if other_fields:
                    return f"{main_val}<div class='mt-1 border-t border-slate-100 pt-1'>{_recursive_data_to_html(other_fields)}</div>"
                return main_val
        
        # Si no hay key principal, mostramos pares key-value de forma elegante
        rows = []
        for k, v in data.items():
            if v and v != "No especificado":
                label = k.replace('_', ' ').capitalize()
                rows.append(f"<div class='mb-1'><span class='font-semibold text-slate-800'>{label}:</span> {_recursive_data_to_html(v)}</div>")
        return "".join(rows)

    # Limpieza final de strings
    texto = str(data).strip()
    # Eliminar asteriscos de negrita/itálica de markdown
    texto = re.sub(r'\*\*(.*?)\*\*', r'<strong class="text-slate-900">\1</strong>', texto)
    texto = re.sub(r'\*(.*?)\*', r'\1', texto)
    # Eliminar backticks
    texto = texto.replace('`', '')
    # Eliminar llaves o corchetes residuales si se colaron como texto
    if len(texto) > 2 and ((texto.startswith('{') and texto.endswith('}')) or (texto.startswith('[') and texto.endswith(']'))):
        try:
            # Re-procesar por si es un string que contiene JSON
            parsed = json.loads(texto)
            return _recursive_data_to_html(parsed)
        except:
            # v10.7: Rescate codicioso para JSON truncado (sin comilla final)
            # Como último recurso, intentar extraer lo que haya dentro de "respuesta": "..."
            match = re.search(r'"respuesta":\s*"(.*)', texto, re.DOTALL)
            if match:
                content = match.group(1).strip()
                # Limpiar si hay un cierre de JSON al final
                if content.endswith('"}'): content = content[:-2]
                elif content.endswith('}'): content = content[:-1]
                return _recursive_data_to_html(content)
            pass
            
    return texto

def format_response_for_html(column: str, data: any) -> str:
    """
    Convierte la respuesta de la IA en HTML limpio para la tabla.
    Maneja tanto JSON estructurado como strings directos. v10.4 Premium.
    """
    if not data:
        return "<span class='text-gray-400'>No disponible</span>"

    # Procesamiento base
    texto_html = _recursive_data_to_html(data)

    if not texto_html or ("Sin información" in texto_html and len(texto_html) < 25):
        return "<span class='text-gray-400 italic'>Sin información relevante</span>"

    # Formatear etiquetas de relevancia con colores (se mantiene del anterior)
    def color_status(match):
        tag = match.group(0)
        if "✅" in tag: return f"<span class='inline-flex items-center gap-1 font-bold text-emerald-600 bg-emerald-50 px-1.5 py-0.5 rounded'>{tag}</span>"
        if "⚠️" in tag: return f"<span class='inline-flex items-center gap-1 font-bold text-amber-600 bg-amber-50 px-1.5 py-0.5 rounded'>{tag}</span>"
        if "❌" in tag: return f"<span class='inline-flex items-center gap-1 font-bold text-rose-600 bg-rose-50 px-1.5 py-0.5 rounded'>{tag}</span>"
        return tag

    texto_html = re.sub(r'(✅|⚠️|❌)\s*(Relevante|Parcialmente relevante|No relevante)', color_status, texto_html)
    
    # Formateo final de saltos de línea (si quedan)
    texto_html = texto_html.replace('\\n', '<br>').replace('\n', '<br>')
    
    return f"<div class='text-[13px] leading-snug text-slate-700 font-normal'>{texto_html}</div>"


# ============================================================
# 📑 GESTIÓN DE CONTEXTO DE ALTA DENSIDAD (v9.1)
# ============================================================

def get_smart_context(full_text: str, abstract: str, keywords: List[str] = None) -> str:
    """
    v10.6: Large Context Stratified Sampling.
    Amplía la ventana a 45,000 caracteres para asegurar cobertura de tablas y anexos.
    """
    if not full_text or len(full_text) < 2000:
        return abstract[:6000]

    total = len(full_text)
    target_limit = 24000 # v15.7: Ajuste estricto para límite de 8k tokens
    
    segments = [
        ("INTRODUCCIÓN_Y_OBJETIVOS", full_text[0:8000]),
        ("METODOLOGÍA_Y_DISEÑO", full_text[total//4 : total//4 + 8000]),
        ("RESULTADOS_Y_TABLAS_1", full_text[total//2 : total//2 + 8000]),
        ("RESULTADOS_Y_TABLAS_2", full_text[3*total//4 : 3*total//4 + 8000]),
        ("DISCUSIÓN_Y_CONCLUSIONES", full_text[-8000:])
    ]
    
    combined = ""
    for label, content in segments:
        combined += f"\n\n--- SECCIÓN: {label} ---\n{content}"
    
    logging.info(f"🧬 Smart Context v10.6 (24k Chars): {len(combined)} chars generados.")
    return combined[:target_limit]



# ============================================================
# 🌐 TRADUCCIÓN DE KEYWORDS DINÁMICA VÍA LLM (v11.22)
# Multidominio: funciona para cualquier dominio académico.
# Caché en memoria: 1 llamada al LLM por columna única por sesión.
# ============================================================
_keyword_translation_cache: Dict[str, List[str]] = {}


def _get_english_search_terms(col_label: str, col_instruction: str, research_question: str = "") -> List[str]:
    """
    v11.22: Genera dinámicamente los términos de búsqueda en inglés para una columna.
    Usa el LLM para traducir el concepto de la columna (en cualquier idioma)
    a los términos clave en inglés que aparecerían en un paper académico.
    El resultado se cachea por columna para la sesión actual.
    """
    # Clave de caché: combinación de label + primeros 80 chars de la instrucción
    cache_key = f"{col_label}||{col_instruction[:80]}"
    if cache_key in _keyword_translation_cache:
        logging.debug(f"🔑 Keywords (caché) para '{col_label}': {_keyword_translation_cache[cache_key][:8]}")
        return _keyword_translation_cache[cache_key]

    # Prompt ultra-corto y rápido para el LLM
    system_prompt = """You are a search term expert for academic papers. Given a column name and description (in any language), output ONLY a JSON list of 15-20 English search terms that would appear in academic papers related to that concept. Terms must be lowercase. No explanations. Only JSON array."""
    
    user_prompt = f"""Column: "{col_label}"
Description: "{col_instruction[:200]}"
Context: "{research_question[:100]}"

Output ONLY a JSON array like: ["term1", "term2", "term3", ...]"""

    try:
        result = call_ai_api(system_prompt, user_prompt, max_tokens=200)
        if result and isinstance(result, list):
            terms = [str(t).lower().strip() for t in result if t and len(str(t)) > 2]
        elif result and isinstance(result, str):
            # Intentar parsear el JSON de la respuesta
            match = re.search(r'\[.*?\]', str(result), re.DOTALL)
            if match:
                import json as _json
                terms = _json.loads(match.group(0))
                terms = [str(t).lower().strip() for t in terms if t and len(str(t)) > 2]
            else:
                # Fallback: tokenizar la respuesta como palabras
                terms = re.findall(r'\b\w{3,}\b', str(result).lower())[:20]
        else:
            terms = []
        
        if terms:
            _keyword_translation_cache[cache_key] = terms
            logging.info(f"🌐 Keywords LLM para '{col_label}': {terms[:10]}")
            return terms
    except Exception as e:
        logging.warning(f"⚠️ Error generando keywords LLM para '{col_label}': {e}")
    
    # Fallback mínimo si el LLM falla: tokenizar el label/instrucción directamente
    fallback = re.findall(r'\b\w{4,}\b', (col_label + " " + col_instruction).lower())[:15]
    _keyword_translation_cache[cache_key] = fallback
    return fallback

def get_column_aware_context(full_text: str, abstract: str, col_label: str, col_instruction: str, research_question: str = "", chunk_offset: int = 0) -> str:
    """
    v12.5: RAG columna-específico priorizando Chunks sobre Intro.
    - Chunks aparecen AL INICIO del context para que el modelo local los vea.
    - chunk_offset permite reintentar con el siguiente set de fragmentos relevantes.
    """
    if not full_text or len(full_text) < 2000:
        return abstract[:6000]

    # 1. Keywords y Query Semántica
    keywords = _get_english_search_terms(col_label, col_instruction, research_question)
    semantic_query = f"{col_label}: {col_instruction}. {research_question}"

    # 2. Chunking Semántico (v14.0)
    CHUNK_SIZE = 1200
    paragraphs = re.split(r'\n\n+', full_text)
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) < CHUNK_SIZE:
            current_chunk += p + "\n\n"
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"
    if current_chunk: chunks.append(current_chunk.strip())

    # 3. Score Vectorial (v14.0)
    # Importación perezosa para evitar dependencias circulares si las hubiera
    from modules.logic.screening import get_embedding, cosine_similarity
    
    scored = []
    try:
        # v14.0: Usar similitud de embeddings (MPNet) para mayor precisión
        query_emb = get_embedding(semantic_query)
        chunk_texts = [c for c in chunks if len(c) > 50]
        if chunk_texts:
            chunk_embs = ai_model.LocalModel.get_instance()._call_embeddings(chunk_texts) if hasattr(ai_model.LocalModel.get_instance(), '_call_embeddings') else None
            # Fallback a embedding_service si el local no tiene ese método
            if chunk_embs is None:
                from modules.ai.embedding_service import get_embeddings
                chunk_embs = get_embeddings(chunk_texts)
            
            similarities = cosine_similarity([query_emb], chunk_embs)[0]
            
            for i, sim in enumerate(similarities):
                # Mezclamos Vector Sim + Keyword Boost (v14.1)
                kw_boost = sum(1 for kw in keywords if kw in chunk_texts[i].lower()) * 0.05
                final_score = sim + kw_boost
                scored.append((final_score, i * CHUNK_SIZE, chunk_texts[i]))
    except Exception as e:
        logging.warning(f"⚠️ Fallo RAG vectorial interno: {e}. Usando fallback por keywords.")
        for i, chunk in enumerate(chunks):
            score = sum(1 for kw in keywords if kw in chunk.lower())
            scored.append((score, i * CHUNK_SIZE, chunk))

    scored.sort(reverse=True)

    # 4. Selección de Chunks (Set 1 o Set 2 según offset)
    TOP_N = 10
    start_idx = chunk_offset * TOP_N
    end_idx = start_idx + TOP_N
    selected = sorted(scored[start_idx:end_idx], key=lambda x: x[1])

    context_parts = []
    for score, pos, chunk in selected:
        pct = int(pos / len(full_text) * 100)
        context_parts.append(f"\n--- [FRAGMENTO RELEVANTE (~{pct}% del PDF, relevancia={score})] ---\n{chunk}")

    column_context = "".join(context_parts)

    # 5. ESTRUCTURA v12.10: Chunks de RAG como EVIDENCIA y Backup como CONTEXTO
    total = len(full_text)
    intro_context = f"\n\n--- [SECCIÓN: SOLO CONTEXTO/INTRODUCCIÓN/MOTIVACIÓN] ---\n{full_text[:4000]}"
    results_context = f"\n--- [SECCIÓN: ZONA DE EVIDENCIA/CONCLUSIONES] ---\n{full_text[max(0, total-3000):]}"
    
    # Marcar los chunks del RAG como evidencia prioritaria
    evidence_rag = "\n--- [ZONA DE EVIDENCIA: FRAGMENTOS RELEVANTES (RAG)] ---\n" + column_context
    
    final_context = evidence_rag + intro_context + results_context
    logging.info(f"🎯 Multi-Domain Rigor Context v12.10 (columna='{col_label}'): {len(final_context):,} chars")
    return final_context[:24000] # Ventana ultra-segura para Llama 3 (8000 tokens)



    return final_context[:45000]

def _iron_logic_extraction_guard(raw_text: str, context_text: str, research_question: str) -> str:
    """
    v15.0: El "Escudo de Python" que limpia y verifica la salida del modelo.
    1. Extrae rastro de análisis (internamente).
    2. Limpia etiquetas de rastro para el usuario final.
    """
    if not raw_text or not isinstance(raw_text, str): return raw_text
    
    # 1. Separar análisis de resultado
    clean_output = raw_text
    # Intentar capturar análisis tanto si tiene tags como si no (basado en v15.0 instructions)
    if "<analisis>" in raw_text:
        parts = raw_text.split("</analisis>") if "</analisis>" in raw_text else raw_text.split("<analisis>")
        
        if "</analisis>" in raw_text:
            # Flujo ideal: el modelo cerró el tag
            clean_output = parts[1].strip()
        else:
            # Fallback 1: El modelo abrió <analisis> pero nunca lo cerró. 
            # A menudo, Llama 3 pone los números 1, 2, 3, 4 y luego el resultado.
            # Intentamos limpiar el texto asumiendo que el resultado final está al final.
            analysis_text = parts[1]
            # Heurística: Si vemos un separador claro o viñetas (•), cortamos ahí
            if "•" in analysis_text:
                clean_output = analysis_text[analysis_text.find("•"):].strip()
            elif "\n\n" in analysis_text:
                # Tomamos los últimos párrafos como resultado
                clean_output = "\n\n".join(analysis_text.split("\n\n")[-2:]).strip()
            else:
                 # Si no hay forma de separar, mostramos todo sin advertencia de error para no asustar al usuario
                 clean_output = analysis_text.strip()

    # 2. Verificación de Trazabilidad
    if "[EVIDENCIA" not in clean_output and "[CONTEXTO" not in clean_output and len(clean_output) > 40:
        clean_output = "⚠️ [Aviso: Trazabilidad no etiquetada]\n" + clean_output

    return clean_output

# v11.21: Anti-Placeholder — detectar respuestas vacías y reintentar
def _is_placeholder_response(resp):
    if not resp:
        return True
    raw = str(resp).strip().lower()
    
    # 1. Si la respuesta es extremadamente corta (ej: error o vacío), es placeholder
    if len(raw) < 15:
        return True
    
    # 2. Si es corta (< 180 chars) y contiene frases evasivas o de "no encontrado"
    placeholder_phrases = [
        "no reportado en", "no encontrado en", "no disponible", "no se reporta en",
        "no hay información", "sin información", "no menciona", "no especifica",
        "la columna no", "[sin informacion]", "no detallada"
    ]
    
    # Solo marcamos como placeholder si contiene la frase Y es relativamente corta
    # (Evitamos falsos positivos en descripciones largas que mencionan una limitación)
    if len(raw) < 180 and any(p in raw for p in placeholder_phrases):
        return True
        
    return False

# ============================================================
# 🔄 PROCESAMIENTO PRINCIPAL (v9.1)
# ============================================================
def _generate_columns_for_article(article: Dict, columns_config: List[Dict], research_question: str = "", config_hash: str = "") -> Dict:
    """Genera columnas dinámicas con contexto de alta densidad y blindaje anti-alucinación."""
    
    title = article.get('title', '')
    full_text = article.get('full_text', '')   # v10.8 fix: assignment was missing
    abstract = article.get('abstract', '')     # v10.8 fix: assignment was missing
    real_text_len = len(str(full_text))        # v10.9 fix
    
    # v10.7: Detección Estricta de Calidad de Contexto para evitar alucinaciones
    is_real_pdf = article.get('is_pdf_downloaded', False) and real_text_len > 5000
    context_type = "PDF COMPLETO (RAG-Bilingüe v11.21)" if is_real_pdf else "SOLO ABSTRACT (Limitado)"


    results = {}
    for col in columns_config:
        col_key = col['key']
        col_label = col['label']
        col_instruction = col['pregunta']
        
        cached_val = load_from_cache(article, col_key, research_question, config_hash)
        if cached_val:
            results[col_key] = cached_val
            continue

        # v11.22: RAG columna-específico MULTIDOMINIO
        if is_real_pdf:
            context = get_column_aware_context(full_text, abstract, col_label, col_instruction, research_question)
            anti_hallucination = (
                f"ESTADO DEL CONTEXTO: Recibes fragmentos del PDF seleccionados específicamente para la columna '{col_label}' "
                f"mediante RAG columna-específico ({real_text_len} chars totales en el PDF). "
                "Si un dato no aparece en los fragmentos, escribe SOLO: 'No reportado en los fragmentos analizados.'"
            )
        else:
            context = abstract[:6000]
            anti_hallucination = (
                f"ESTADO DEL CONTEXTO: SOLO tienes el RESUMEN/ABSTRACT ({real_text_len} chars). "
                "INSTRUCCIÓN CRÍTICA: NO inventes herramientas ni porcentajes. "
                "Si no está en el resumen, di estrictamente 'Información no detallada en el resumen'."
            )

        # v11.19: Límite de palabras adaptativo por columna
        word_limit = COLUMN_WORD_LIMITS.get(col_key, COLUMN_WORD_LIMITS["__default__"])
        logging.info(f"⚡ Analizando '{col_label}' ({context_type}, max {word_limit} palabras): {title[:40]}...")
        
        # v11.23: Lista de otras columnas para el anti-espagueti (PASO 4)
        other_cols = ", ".join(
            f'"{c["label"]}"' for c in columns_config if c["key"] != col_key
        )

        system_prompt = f"""Eres un extractor experto de datos para Revisiones Sistemáticas de Literatura (RSL). Debes llenar UNA SOLA celda de una tabla de síntesis para la columna: "{col_label}".

INSTRUCCIONES DE EXTRACCIÓN:
1. Analiza el fragmento de texto proporcionado buscando ÚNICAMENTE la información solicitada en la instrucción de la columna.
2. Escribe tu proceso de razonamiento internamente dentro de un bloque `<analisis>...</analisis>` para asegurar precisión.
3. Luego del bloque de análisis, escribe la extracción final utilizando viñetas (•).
4. Usa referenciación para mostrar de dónde sacaste el dato:
   - Usa `[EVIDENCIA: Nombre de la Sección]` para resultados, métodos o datos extraídos.
   - Usa `[CONTEXTO: Nombre de la Sección]` para información de introducción o antecedentes.
5. Tu extracción final debe tener un MÁXIMO de {word_limit} palabras.

REGLAS DE ORO ESTRICTAS (Anti-Alucinación y Anti-Fuga):
- NO OCULTES EL ANÁLISIS, es obligatorio razonar dentro de `<analisis>...</analisis>` antes de dar el resultado.
- 🚨 ANTI-FUGA RAG: ESTÁ TERMINANTEMENTE PROHIBIDO incluir metadatos del buscador en tu respuesta final (ej. "relevancia=0.37" o "ZONA DE EVIDENCIA"). Usa exclusivamente referencias humanas como `[EVIDENCIA: Sección de Metodología]` o `[EVIDENCIA: Resultados de la Tabla 2]`.
- 🚨 ANTI-AMNESIA DE LISTADOS: Si estás extrayendo herramientas o resultados, TU DEBER es leer hasta la última línea del texto. No reportes solo las primeras herramientas mencionadas; busca si hay más escondidas al final.
- 🚨 ANTI-AMNESIA MÉTRICA: JAMÁS declares que "no hay métricas" o "no hay resultados" hasta no haber escaneado el documento entero, incluyendo párrafos finales y anexos.
- 🚨 ANTI-FUEGO AMIGO: Si lees que los autores critican investigaciones o metodologías pasadas, NO asumas que están criticando su propio trabajo. Esas quejas NO son "limitaciones del estudio actual".
- 🚨 ANTI-SESGO RQ: La Pregunta de Investigación Global es solo contexto. Si el texto reporta algo diferente a lo que busca la RQ, extrae la VERDAD del texto (Ej: si dice "Machine Learning" extrae "Machine Learning", no asumas "LLMs").
- Si luego de buscar exhautsivamente el dato realmente no existe en el texto, tu única respuesta debe ser: "No se reporta información específica en los fragmentos analizados."
- Responde siempre con viñetas concisas y en español científico y profesional."""

        user_prompt = f"""Artículo: {title}
Pregunta de Investigación Global: {research_question}
Columna a extraer: "{col_label}"
Instrucción específica para esta columna: {col_instruction}

Texto del artículo ({context_type}):
{context}

Escribe tu bloque <analisis> detallando qué encontraste y por qué, y luego tu extracción final en viñetas (•)."""
        
        data = call_ai_api(system_prompt, user_prompt, max_tokens=900)  # v11.19: tokens para respuesta completa
        
        
        if _is_placeholder_response(data):
            logging.warning(f"⚠️ Respuesta placeholder detectada para '{col_label}' en '{title[:30]}'. Reintentando con RAG-Shift (Chunks 11-20)...")
            
            # v12.5: Reintento con Ventana Deslizante (RAG-Shift)
            # Buscamos en el siguiente set de fragmentos relevantes
            retry_context = get_column_aware_context(full_text, abstract, col_label, col_instruction, research_question, chunk_offset=1)
            
            retry_system = f"Eres un extractor de datos de papers académicos. Extrae información sobre: {col_label}. Responde en español con bullets (•). Si hay datos numéricos, inclúyelos. Máximo {word_limit} palabras."
            retry_user = f"Artículo: {title}\nColumna: {col_label}\nBusca específicamente: {col_instruction}\n\nTexto (Ventana Técnica 2):\n{retry_context[:24000]}"
            
            data_retry = call_ai_api(retry_system, retry_user, max_tokens=600)
            if data_retry and not _is_placeholder_response(data_retry):
                data = data_retry
                logging.info(f"✅ Reintento RAG-Shift exitoso para '{col_label}': {len(str(data_retry))} chars")
            else:
                logging.warning(f"❌ Reintento RAG-Shift fallido para '{col_label}'")
        
        if not data:
            val = "<span class='text-red-400 text-xs'>Error de conexión</span>"
        else:
            # v15.0: Guardián de Lógica de Hierro
            if isinstance(data, str):
                data = _iron_logic_extraction_guard(data, context, research_question)
            elif isinstance(data, dict) and "respuesta" in data:
                data["respuesta"] = _iron_logic_extraction_guard(data["respuesta"], context, research_question)

            val = format_response_for_html(col_key, data)
            
        save_to_cache(article, col_key, val, research_question, config_hash)
        results[col_key] = val
        
    return results


# ============================================================
# 🌐 TRADUCCIÓN (USANDO GEMINI)
# ============================================================

def translate_abstract_to_spanish(text: str) -> str:
    """Traduce abstract usando Gemini"""
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"Traduce este abstract científico al español manteniendo términos técnicos:\n\n{text[:2000]}",
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=500
            )
        )
        return response.text
    except:
        return text


def translate_question_to_english(text: str) -> str:
    """Traduce pregunta de investigación a inglés"""
    try:
        prompt = f"Translate this research question to English (preserve technical terms).\nOUTPUT ONLY THE TRANSLATED TEXT. Do not include introductory phrases like 'Here is the translation'. No yapping.\n\n{text}"
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=200
            )
        )
        return response.text.strip()
    except:
        return text