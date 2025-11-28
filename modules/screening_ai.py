import logging
import os
import time
import json
import hashlib
from typing import Dict, List, Optional

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

# Cache para ahorrar llamadas
CACHE_DIR = ".cache/ai_columns"
os.makedirs(CACHE_DIR, exist_ok=True)

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
# 🛠️ UTILIDADES DE CACHE
# ============================================================
def get_cache_key(title: str, column: str, question: str = "") -> str:
    """Cache key incluye la pregunta para contexto específico"""
    key = f"{title}_{column}_{question[:50]}_v50_gemini".encode('utf-8')
    return hashlib.md5(key).hexdigest()

def load_from_cache(title: str, column: str, question: str = "") -> str:
    try:
        path = os.path.join(CACHE_DIR, f"{get_cache_key(title, column, question)}.txt")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f: 
                return f.read()
    except: 
        pass
    return None

def save_to_cache(title: str, column: str, value: str, question: str = ""):
    if "Error" in value or "No especificado" in value: 
        return
    try:
        with open(os.path.join(CACHE_DIR, f"{get_cache_key(title, column, question)}.txt"), 'w', encoding='utf-8') as f: 
            f.write(value)
    except: 
        pass

# ============================================================
# 🧠 GENERACIÓN DE CONTEXTO
# ============================================================
def prepare_context(abstract: str, full_text: str) -> str:
    """Prioriza texto completo, fallback a abstract"""
    if full_text and len(full_text) > 1000:
        return full_text[:20000]
    return abstract

# ============================================================
# 🎯 PROMPTS ESPECIALIZADOS (ESTILO ELICIT)
# ============================================================

def get_system_prompt_for_column(column: str, research_question: str) -> str:
    """System prompt contextualizado por columna y pregunta de investigación"""
    
    base_context = f"""Eres un asistente experto en revisiones sistemáticas de literatura científica.

CONTEXTO DE LA INVESTIGACIÓN:
La revisión sistemática busca responder: "{research_question}"

TU TAREA:
Extraer información ESPECÍFICA y RELEVANTE para esta pregunta de investigación.

REGLAS ESTRICTAS:
1. SOLO reporta lo que está EXPLÍCITAMENTE en el texto
2. Si no encuentras la información, responde "No especificado"
3. NO inventes, NO asumas, NO generalices
4. Usa terminología técnica precisa (nombres de algoritmos, métricas exactas, valores numéricos)
5. Escribe en estilo académico denso (como Elicit)
6. Responde SOLO con JSON válido, sin markdown ni explicaciones adicionales
"""

    column_specific = {
        "summary": """
COLUMNA: Resumen Ejecutivo
OBJETIVO: Sintetizar en 2-3 oraciones:
- El objetivo principal del estudio
- El enfoque metodológico general
- La contribución clave

FORMATO JSON ESPERADO:
{
    "objetivo": "Una oración clara del objetivo principal",
    "enfoque": "Una oración del método/diseño usado",
    "contribucion": "Una oración de qué aporta el estudio"
}
""",
        
        "methodology": f"""
COLUMNA: Metodología
OBJETIVO: Extraer detalles técnicos del diseño experimental/computacional:

BUSCA ESPECÍFICAMENTE (en relación a: {research_question}):
- Tipo de estudio (experimental, simulación, comparativo, etc.)
- Algoritmos/Modelos específicos usados (nombres propios como "LSTM", "CNN-BiLSTM", "Random Forest")
- Frameworks/Herramientas (TensorFlow, Keras, Scikit-learn, etc.)
- Arquitectura del sistema (capas, parámetros, configuraciones)
- Proceso de entrenamiento/validación (k-fold, train-test split, etc.)

FORMATO JSON ESPERADO:
{{
    "tipo_estudio": "Experimental / Simulación / Comparativo / etc.",
    "algoritmos": ["Nombre1", "Nombre2", "..."],
    "frameworks": ["Tool1", "Tool2"],
    "arquitectura": "Descripción técnica breve de la estructura",
    "validacion": "Método de validación usado"
}}

IMPORTANTE: Si encuentras varios modelos, LISTA TODOS.
""",

        "population": f"""
COLUMNA: Población/Datasets
OBJETIVO: Identificar las fuentes de datos específicas.

BUSCA (relevante para: {research_question}):
- Nombres propios de datasets (NSL-KDD, CICIDS2017, BoT-IoT, UNSW-NB15, etc.)
- Tamaño del dataset (número de muestras, registros, paquetes)
- Tipo de datos (tráfico de red, logs, sensores IoT, etc.)
- Origen (simulado, real-world, público, privado)
- Proporción de clases (benign vs malicious)

FORMATO JSON ESPERADO:
{{
    "datasets": ["Nombre1", "Nombre2"],
    "tamano": "N muestras / registros",
    "tipo_datos": "Descripción breve",
    "origen": "Real-world / Simulado / Benchmark",
    "distribucion_clases": "% benign vs % attacks"
}}
""",

        "independent_variables": f"""
COLUMNA: Variables Independientes (Inputs/Factores)
OBJETIVO: Identificar QUÉ se manipuló o varió en el estudio.

CONTEXTO: En el estudio "{research_question}", las variables independientes son los INPUTS o factores que el investigador controla/modifica.

EJEMPLOS DE VARIABLES INDEPENDIENTES:
- Configuraciones del modelo (learning rate, epochs, batch size)
- Tipos de algoritmos comparados (LSTM vs GRU vs CNN)
- Características de entrada (features seleccionadas)
- Parámetros del sistema (umbrales de detección, ventanas de tiempo)
- Condiciones experimentales (niveles de carga, tipos de ataques)

FORMATO JSON ESPERADO:
{{
    "variables": [
        {{
            "nombre": "Nombre descriptivo",
            "valores": "Valores probados (ej: 'learning rates: 0.001, 0.01, 0.1')",
            "rol": "Qué representa esta variable en el experimento"
        }}
    ]
}}

IMPORTANTE: NO confundas con métricas de resultado (esas son variables dependientes).
""",

        "dependent_variables": f"""
COLUMNA: Variables Dependientes (Outcomes/Métricas)
OBJETIVO: Identificar QUÉ se midió como resultado.

CONTEXTO: Para "{research_question}", las variables dependientes son las MÉTRICAS/RESULTADOS que se observaron.

EJEMPLOS DE VARIABLES DEPENDIENTES:
- Métricas de rendimiento (Accuracy, Precision, Recall, F1-Score, AUC-ROC)
- Tiempo de ejecución (latencia, throughput)
- Consumo de recursos (CPU, RAM, energía)
- Tasa de detección (True Positive Rate, False Positive Rate)
- Robustez (performance bajo adversarial attacks)

FORMATO JSON ESPERADO:
{{
    "metricas": [
        {{
            "nombre": "Nombre de la métrica",
            "valor": "Valor reportado (ej: '95.3%', '0.5 ms')",
            "interpretacion": "Qué significa (mejor rendimiento, peor latencia, etc.)"
        }}
    ]
}}
""",

        "study_design": """
COLUMNA: Diseño del Estudio
OBJETIVO: Clasificar el tipo de investigación.

TIPOS COMUNES:
- Experimental (prueba una hipótesis con control de variables)
- Comparativo (compara múltiples enfoques)
- Simulación (evalúa en entorno controlado)
- Caso de estudio (análisis de un sistema específico)
- Revisión sistemática / Meta-análisis

FORMATO JSON ESPERADO:
{
    "tipo": "Experimental / Comparativo / Simulación / etc.",
    "justificacion": "Por qué se clasifica así (1 oración)"
}
""",

        "objectives": """
COLUMNA: Objetivos del Estudio
OBJETIVO: Listar los objetivos específicos con verbos de acción.

FORMATO JSON ESPERADO:
{
    "objetivos": [
        "Evaluar el rendimiento de...",
        "Comparar la eficacia de...",
        "Proponer un nuevo método para...",
        "Analizar el impacto de..."
    ]
}

USA VERBOS: Evaluar, Comparar, Proponer, Desarrollar, Analizar, Demostrar, Validar, etc.
""",

        "key_findings": f"""
COLUMNA: Hallazgos Clave
OBJETIVO: Resumir los resultados principales CON DATOS NUMÉRICOS.

PARA "{research_question}", reporta:
- Resultados cuantitativos con valores exactos
- Comparaciones entre métodos (ej: "X superó a Y en 5%")
- Descubrimientos inesperados
- Confirmaciones/refutaciones de hipótesis

FORMATO JSON ESPERADO:
{{
    "hallazgos": [
        {{
            "resultado": "Descripción del hallazgo con NÚMEROS",
            "metrica": "Métrica asociada (Accuracy, F1, etc.)",
            "valor": "Valor numérico exacto"
        }}
    ]
}}

EJEMPLO:
"El modelo LSTM alcanzó 97.2% de accuracy, superando a CNN (94.1%) en detección de DDoS."
""",

        "limitations": """
COLUMNA: Limitaciones
OBJETIVO: Identificar restricciones metodológicas o técnicas EXPLÍCITAS.

BUSCA:
- Limitaciones de los datasets (sesgo, tamaño limitado, falta de diversidad)
- Restricciones del modelo (alto costo computacional, no escalable)
- Amenazas a la validez (overfitting, falta de validación externa)
- Trabajo futuro mencionado (qué faltó hacer)

FORMATO JSON ESPERADO:
{
    "limitaciones": [
        "Limitación 1 (con explicación breve)",
        "Limitación 2",
        "..."
    ]
}
"""
    }
    
    return base_context + column_specific.get(column, "")


def get_user_prompt_for_column(column: str, context: str, research_question: str) -> str:
    """User prompt con el texto del artículo"""
    return f"""
PREGUNTA DE INVESTIGACIÓN: {research_question}

TEXTO DEL ARTÍCULO:
{context}

---

TAREA: Extrae la información para la columna "{column}" siguiendo las instrucciones del sistema.

RECUERDA:
- Responde SOLO con JSON válido (sin markdown)
- Si no encuentras la info, usa "No especificado"
- Incluye VALORES NUMÉRICOS cuando estén disponibles
"""

# ============================================================
# ⚡ LLAMADA A LA API GEMINI (CON CONTROL DE RATE LIMITS)
# ============================================================
LAST_CALL_TIMESTAMP = 0
REQUEST_INTERVAL = 4.5  # Gemini free tier: ~15 RPM = 1 req cada 4 segundos

def call_gemini_api(system_prompt: str, user_prompt: str, max_tokens: int = 700) -> Dict:
    """
    Llama a Gemini API con manejo robusto de rate limits.
    
    Límites Gemini 2.0 Flash (Free Tier):
    - 15 RPM (requests per minute)
    - 1M TPM (tokens per minute)
    - 200 RPD (requests per day)
    """
    global LAST_CALL_TIMESTAMP
    
    # Control de rate limiting (client-side)
    elapsed = time.time() - LAST_CALL_TIMESTAMP
    if elapsed < REQUEST_INTERVAL:
        time.sleep(REQUEST_INTERVAL - elapsed)
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            LAST_CALL_TIMESTAMP = time.time()
            
            # Prompt combinado (Gemini funciona mejor con prompt único)
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Llamada a Gemini con configuración optimizada
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=combined_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.05,  # Baja temperatura para extracción precisa
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json"  # Forzar respuesta JSON
                )
            )
            
            # Extraer texto de la respuesta
            content = response.text
            
            if not content:
                logging.warning("⚠️ Respuesta vacía de Gemini")
                time.sleep(2)
                continue
            
            # Limpiar markdown si viene con ```json
            content = content.replace('```json', '').replace('```', '').strip()
            
            # Parsear JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logging.error(f"❌ Respuesta no JSON: {content[:200]}")
                # Intentar extraer JSON embebido
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return None
                
        except Exception as e:
            error_msg = str(e).lower()
            
            # Manejo de errores específicos de Gemini
            if "429" in error_msg or "quota" in error_msg or "rate" in error_msg:
                wait_time = 30 * (attempt + 1)  # Backoff exponencial
                logging.warning(f"⏳ Rate limit excedido. Esperando {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            elif "401" in error_msg or "unauthorized" in error_msg:
                logging.error("❌ API Key inválida. Verifica GEMINI_API_KEY")
                return None
            
            elif "403" in error_msg or "forbidden" in error_msg:
                logging.error("❌ Acceso denegado. Verifica permisos del proyecto")
                return None
            
            else:
                logging.error(f"❌ Error Gemini API (intento {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(5)
                
    return None

# ============================================================
# 🎨 FORMATEO DE RESPUESTAS (ESTILO ELICIT)
# ============================================================

def format_response_for_html(column: str, data: Dict) -> str:
    """Convierte JSON en HTML formateado estilo Elicit"""
    
    if not data or data.get("error"):
        return "<div class='text-gray-400 text-xs italic'>No especificado en el texto.</div>"
    
    # SUMMARY
    if column == "summary":
        obj = data.get("objetivo", "?")
        enf = data.get("enfoque", "?")
        con = data.get("contribucion", "?")
        
        if obj == "?" or "No especificado" in obj:
            return "<div class='text-gray-400 text-xs italic'>Información insuficiente.</div>"
        
        return f"""
        <div class='space-y-2 text-sm'>
            <div><span class='font-semibold text-slate-600'>Objetivo:</span> <span class='text-slate-700'>{obj}</span></div>
            <div><span class='font-semibold text-slate-600'>Enfoque:</span> <span class='text-slate-700'>{enf}</span></div>
            <div><span class='font-semibold text-slate-600'>Contribución:</span> <span class='text-slate-700'>{con}</span></div>
        </div>
        """
    
    # METHODOLOGY
    if column == "methodology":
        tipo = data.get("tipo_estudio", "No especificado")
        algos = data.get("algoritmos", [])
        frameworks = data.get("frameworks", [])
        arq = data.get("arquitectura", "")
        val = data.get("validacion", "")
        
        html = f"<div class='text-sm text-slate-700 space-y-1.5'>"
        html += f"<div><span class='font-semibold'>Tipo:</span> {tipo}</div>"
        
        if algos and algos != ["No especificado"]:
            html += f"<div><span class='font-semibold'>Algoritmos:</span> {', '.join(algos)}</div>"
        
        if frameworks and frameworks != ["No especificado"]:
            html += f"<div><span class='font-semibold'>Frameworks:</span> {', '.join(frameworks)}</div>"
        
        if arq and "No especificado" not in arq:
            html += f"<div><span class='font-semibold'>Arquitectura:</span> {arq}</div>"
        
        if val and "No especificado" not in val:
            html += f"<div><span class='font-semibold'>Validación:</span> {val}</div>"
        
        html += "</div>"
        return html
    
    # POPULATION
    if column == "population":
        datasets = data.get("datasets", [])
        tam = data.get("tamano", "")
        tipo = data.get("tipo_datos", "")
        
        if not datasets or datasets == ["No especificado"]:
            return "<div class='text-gray-400 text-xs italic'>Datasets no especificados.</div>"
        
        html = f"<div class='text-sm text-slate-700 space-y-1.5'>"
        html += f"<div><span class='font-semibold'>Datasets:</span> {', '.join(datasets)}</div>"
        
        if tam and "No especificado" not in tam:
            html += f"<div><span class='font-semibold'>Tamaño:</span> {tam}</div>"
        
        if tipo and "No especificado" not in tipo:
            html += f"<div><span class='font-semibold'>Tipo:</span> {tipo}</div>"
        
        html += "</div>"
        return html
    
    # STUDY DESIGN
    if column == "study_design":
        tipo = data.get("tipo", "No especificado")
        justif = data.get("justificacion", "")
        
        html = "<div class='text-sm text-slate-700 space-y-2'>"
        html += f"<div><span class='font-semibold text-indigo-600'>Tipo:</span> <span class='font-medium'>{tipo}</span></div>"
        
        if justif and "No especificado" not in justif:
            html += f"<div class='text-slate-600 italic text-xs leading-relaxed' style='word-wrap: break-word; overflow-wrap: break-word;'>\"{justif}\"</div>"
        
        html += "</div>"
        return html
    
    # OBJECTIVES
    if column == "objectives":
        objetivos = data.get("objetivos", [])
        
        if not objetivos or objetivos == ["No especificado"]:
            return "<div class='text-gray-400 text-xs italic'>No identificados.</div>"
        
        html = "<ul class='text-sm text-slate-700 space-y-1.5 list-disc list-inside'>"
        for obj in objetivos:
            if "No especificado" not in obj:
                html += f"<li class='leading-relaxed' style='word-wrap: break-word;'>{obj}</li>"
        html += "</ul>"
        return html
    
    # INDEPENDENT VARIABLES
    if column == "independent_variables":
        variables = data.get("variables", [])
        
        if not variables or len(variables) == 0:
            return "<div class='text-gray-400 text-xs italic'>Variables no identificadas.</div>"
        
        html = "<ul class='text-sm text-slate-700 space-y-1 list-disc list-inside'>"
        for v in variables:
            nombre = v.get("nombre", "")
            valores = v.get("valores", "")
            if nombre and "No especificado" not in nombre:
                html += f"<li style='word-wrap: break-word;'><span class='font-semibold'>{nombre}</span>: {valores}</li>"
        html += "</ul>"
        return html
    
    # DEPENDENT VARIABLES
    if column == "dependent_variables":
        metricas = data.get("metricas", [])
        
        if not metricas or len(metricas) == 0:
            return "<div class='text-gray-400 text-xs italic'>Métricas no reportadas.</div>"
        
        html = "<ul class='text-sm text-slate-700 space-y-1 list-disc list-inside'>"
        for m in metricas:
            nombre = m.get("nombre", "")
            valor = m.get("valor", "")
            if nombre and "No especificado" not in nombre:
                html += f"<li style='word-wrap: break-word;'><span class='font-semibold'>{nombre}</span>: {valor}</li>"
        html += "</ul>"
        return html
    
    # KEY FINDINGS
    if column == "key_findings":
        hallazgos = data.get("hallazgos", [])
        
        if not hallazgos or len(hallazgos) == 0:
            return "<div class='text-gray-400 text-xs italic'>Resultados no especificados.</div>"
        
        html = "<div class='text-sm text-slate-700 space-y-2'>"
        for h in hallazgos:
            resultado = h.get("resultado", "")
            if resultado and "No especificado" not in resultado:
                html += f"<div style='word-wrap: break-word;'>• {resultado}</div>"
        html += "</div>"
        return html
    
    # LIMITATIONS
    if column == "limitations":
        limitaciones = data.get("limitaciones", [])
        
        if not limitaciones or limitaciones == ["No especificado"]:
            return "<div class='text-gray-400 text-xs italic'>No mencionadas explícitamente.</div>"
        
        html = "<ul class='text-sm text-slate-700 space-y-1 list-disc list-inside'>"
        for lim in limitaciones:
            if "No especificado" not in lim:
                html += f"<li style='word-wrap: break-word;'>{lim}</li>"
        html += "</ul>"
        return html
    
    # FALLBACK GENÉRICO
    return f"<pre class='text-xs text-slate-600' style='white-space: pre-wrap; word-wrap: break-word;'>{json.dumps(data, indent=2, ensure_ascii=False)}</pre>"


# ============================================================
# 🔄 PROCESAMIENTO PRINCIPAL
# ============================================================

def _generate_columns_for_article(article: Dict, columns: List[str], research_question: str = "") -> Dict:
    """Genera columnas con contexto de la pregunta de investigación usando Gemini"""
    
    title = article.get('title', '')
    context = prepare_context(article.get('abstract', ''), article.get('full_text', ''))
    
    if len(context) < 50:
        for col in columns: 
            article[col] = "<div class='text-red-400 text-xs'>⚠️ Texto no disponible</div>"
        return article

    for col in columns:
        cached_val = load_from_cache(title, col, research_question)
        if cached_val and "Error" not in cached_val:
            article[col] = cached_val
            continue
            
        logging.info(f"⚡ Extrayendo '{col}': {title[:40]}...")
        
        system_prompt = get_system_prompt_for_column(col, research_question)
        user_prompt = get_user_prompt_for_column(col, context, research_question)
        
        data = call_gemini_api(system_prompt, user_prompt, max_tokens=700)
        
        if not data:
            val = "<span class='text-red-400 text-xs'>Error de conexión</span>"
        else:
            val = format_response_for_html(col, data)
        
        article[col] = val
        save_to_cache(title, col, val, research_question)
        
    return article


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
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"Translate this research question to English (preserve technical terms):\n{text}",
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=200
            )
        )
        return response.text
    except:
        return text