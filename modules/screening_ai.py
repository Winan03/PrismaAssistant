import logging
import os
import time
import json
import hashlib
import requests
from typing import Dict, List, Optional
from openai import OpenAI, RateLimitError, APIError 

# Configuración del logging principal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================
# 📇 SILENCIADOR DE RUIDO (PDF WARNINGS)
# ============================================================
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
# ============================================================

# Cache para ahorrar llamadas
CACHE_DIR = ".cache/ai_columns"
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================================
# ⚡ CONFIGURACIÓN DEL MOTOR (GITHUB MODELS / OPENROUTER)
# ============================================================
API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("GITHUB_TOKEN") 
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://models.github.ai/inference")
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini") 

if not API_KEY:
    logging.warning("⚠️ No se encontró API KEY. Asegúrate de configurar OPENROUTER_API_KEY o GITHUB_TOKEN en tu .env")

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY or "dummy-key",
)

# ============================================================
# 🛠️ UTILIDADES DE CACHE
# ============================================================
def get_cache_key(title: str, column: str, question: str = "") -> str:
    """Cache key incluye la pregunta para contexto específico"""
    key = f"{title}_{column}_{question[:50]}_v50_elicit_style".encode('utf-8')
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
        return full_text[:20000]  # Aumentado para mejor extracción
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
- Responde SOLO con JSON válido
- Si no encuentras la info, usa "No especificado"
- Incluye VALORES NUMÉRICOS cuando estén disponibles
"""

# ============================================================
# ⚡ LLAMADA A LA API (ROBUSTA CON REINTENTOS)
# ============================================================
LAST_CALL_TIMESTAMP = 0
REQUEST_INTERVAL = 2.5  # Reducido para mayor velocidad

def call_ai_api(messages: List[Dict], max_tokens: int = 600) -> Dict:
    """Llama a Puter.js API (gratuita e ilimitada)"""
    global LAST_CALL_TIMESTAMP
    
    elapsed = time.time() - LAST_CALL_TIMESTAMP
    if elapsed < 1.0:  # Reducido a 1 segundo
        time.sleep(1.0 - elapsed)
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            LAST_CALL_TIMESTAMP = time.time()
            
            # Construir prompt combinado (Puter espera texto plano)
            system_msg = next((m['content'] for m in messages if m['role'] == 'system'), '')
            user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
            combined_prompt = f"{system_msg}\n\n{user_msg}\n\nResponde SOLO con JSON válido."
            
            # Llamada a Puter.js API
            response = requests.post(
                "https://api.puter.com/ai/chat",  # Endpoint correcto (verifica docs)
                json={
                    "prompt": combined_prompt,
                    "model": "gpt-4o-mini",  # o "gpt-5-nano" según tus necesidades
                    "temperature": 0.05,
                    "max_tokens": max_tokens
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                # Puter devuelve la respuesta en data['response'] o data['text']
                content = data.get('response') or data.get('text') or data.get('content', '')
                
                # Limpiar markdown si viene con ```json
                content = content.replace('```json', '').replace('```', '').strip()
                
                return json.loads(content)
            else:
                logging.warning(f"⚠️ Puter API error {response.status_code}: {response.text}")
                time.sleep(2)
                continue
                
        except requests.Timeout:
            logging.warning(f"⏳ Timeout en intento {attempt + 1}")
            time.sleep(3)
            
        except json.JSONDecodeError as e:
            logging.error(f"❌ Respuesta no JSON: {content[:200]}")
            return None
            
        except Exception as e:
            logging.error(f"❌ Error Puter API: {e}")
            if attempt == max_retries - 1:
                return None
                
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
    
    # ✅ STUDY DESIGN (NUEVO - ARREGLADO)
    if column == "study_design":
        tipo = data.get("tipo", "No especificado")
        justif = data.get("justificacion", "")
        
        html = "<div class='text-sm text-slate-700 space-y-2'>"
        html += f"<div><span class='font-semibold text-indigo-600'>Tipo:</span> <span class='font-medium'>{tipo}</span></div>"
        
        if justif and "No especificado" not in justif:
            # ✅ CLAVE: Agregamos word-wrap inline para justificación larga
            html += f"<div class='text-slate-600 italic text-xs leading-relaxed' style='word-wrap: break-word; overflow-wrap: break-word;'>\"{justif}\"</div>"
        
        html += "</div>"
        return html
    
    # ✅ OBJECTIVES (NUEVO)
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
    
    # FALLBACK GENÉRICO (con word-wrap)
    return f"<pre class='text-xs text-slate-600' style='white-space: pre-wrap; word-wrap: break-word;'>{json.dumps(data, indent=2, ensure_ascii=False)}</pre>"


# ============================================================
# 🔄 PROCESAMIENTO PRINCIPAL
# ============================================================

def _generate_columns_for_article(article: Dict, columns: List[str], research_question: str = "") -> Dict:
    """Genera columnas con contexto de la pregunta de investigación"""
    
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
        
        messages = [
            {"role": "system", "content": get_system_prompt_for_column(col, research_question)},
            {"role": "user", "content": get_user_prompt_for_column(col, context, research_question)}
        ]
        
        data = call_ai_api(messages, max_tokens=700)
        
        if not data:
            val = "<span class='text-red-400 text-xs'>Error de conexión</span>"
        else:
            val = format_response_for_html(col, data)
        
        article[col] = val
        save_to_cache(title, col, val, research_question)
        
    return article


# ============================================================
# 🌐 TRADUCCIÓN (SIN CAMBIOS)
# ============================================================

def translate_abstract_to_spanish(text: str) -> str:
    messages = [
        {"role": "system", "content": "Eres un traductor académico especializado."},
        {"role": "user", "content": f"Traduce este abstract al español manteniendo términos técnicos:\n\n{text[:2000]}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=500
        )
        return response.choices[0].message.content
    except:
        return text


def translate_question_to_english(text: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": f"Translate to English (preserve technical terms):\n{text}"}],
            temperature=0.1
        )
        return response.choices[0].message.content
    except:
        return text