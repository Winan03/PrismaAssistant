# rag_analyzer.py - Sistema RAG basado en Embeddings (Sin LLM por artículo)
"""
Módulo que implementa análisis RAG usando Embeddings + Clustering:
1. Extrae aportes usando similitud semántica (NO LLM por artículo)
2. Agrupa aportes similares usando clustering K-Means
3. Solo usa LLM para generar explicaciones finales (máx 5-10 llamadas)

Ventajas:
- Mucho más rápido (embeddings locales)
- Sin límites de rate limit
- Resultados consistentes
"""

import logging
import re
import numpy as np
from collections import defaultdict
from typing import Any, List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from app.llm.embedding_service import get_embeddings, get_single_embedding, check_service
from app.llm.ai_model import LocalModel
from app.core.database import ensure_collection

# ==============================================================================
# 📚 FORMATO APA V7 PARA CITAS
# ==============================================================================

def format_apa_citation(article: dict) -> str:
    """
    Genera cita APA V7 narrativa para un artículo.
    
    Reglas APA V7:
    - 1 autor: "García (2023)"
    - 2 autores: "García y López (2023)"
    - 3+ autores: "García et al. (2023)"
    """
    authors = article.get('authors', [])
    year = article.get('year', 'n.d.')
    
    # Manejar año como número o string
    if isinstance(year, int):
        year = str(year)
    elif not year or year == 'Unknown':
        year = 'n.d.'
    
    # Si no hay autores, usar título abreviado
    if not authors or authors == ['Unknown'] or (isinstance(authors, list) and len(authors) == 0):
        title = article.get('title', 'Sin título')
        title_words = title.split()[:4]
        short_title = ' '.join(title_words)
        if len(title_words) < len(title.split()):
            short_title += '...'
        return f'"{short_title}" ({year})'
    
    # Convertir a lista si es string
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(',') if a.strip()]
    
    def get_surname(name: str) -> str:
        """
        Extrae el apellido de un nombre completo.
        Maneja casos especiales: nombres de una sola palabra, letras sueltas, etc.
        """
        if not name or len(name.strip()) < 2:
            return None

        # Lista de nombres comunes que suelen confundirse con apellidos (Filtro Heurístico)
        first_names = {
            'natalia', 'maria', 'juan', 'jose', 'luis', 'ana', 'carlos', 'elena', 
            'antun', 'abraham', 'david', 'john', 'peter', 'mary', 'ching', 'shixia', 
            'linghui', 'xing', 'yi', 'yan', 'wei'
        }
        
        name = name.strip()
        # Si el nombre tiene coma, el primer componente es definitivamente el apellido (Cancio, R.)
        if ',' in name:
            surname_part = name.split(',')[0].strip()
            if len(surname_part) >= 2:
                return surname_part

        parts = name.split()
        if len(parts) == 0:
            return None
        
        # Si solo hay una palabra, usarla si tiene longitud razonable (min 3 chars)
        if len(parts) == 1:
            word = parts[0].strip('.,')
            if len(word) >= 3:
                return word
            return None
        
        # Múltiples partes sin coma: Prioridad al ÚLTIMO término
        last_word = parts[-1].strip('.,')
        first_word = parts[0].strip('.,')
        
        # Si la primera palabra está en la lista de nombres y hay más palabras, usar la última
        if first_word.lower() in first_names and len(parts) > 1:
            return last_word
        
        # Si la última palabra está en la lista de nombres y la primera no, usar la primera
        if last_word.lower() in first_names and first_word.lower() not in first_names:
            return first_word
            
        # Por defecto en nombres sin coma, buscar la última palabra larga.
        for i in range(len(parts) - 1, -1, -1):
            word = parts[i].strip('.,')
            if len(word) >= 3 and word.lower() not in first_names and not word.isupper():
                return word
        
        # Fallback al primero si nada funcionó
        return first_word if len(first_word) >= 3 else last_word

    
    # Obtener apellido del primer autor
    first_surname = get_surname(authors[0]) if authors else None
    
    # Si no se pudo extraer apellido válido, usar título abreviado
    if not first_surname:
        title = article.get('title', 'Sin título')
        title_words = title.split()[:3]
        short_title = ' '.join(title_words)
        return f'"{short_title}..." ({year})'
    
    if len(authors) == 1:
        return f"{first_surname} ({year})"
    elif len(authors) == 2:
        second_surname = get_surname(authors[1])
        if second_surname:
            return f"{first_surname} y {second_surname} ({year})"
        else:
            return f"{first_surname} et al. ({year})"
    else:
        return f"{first_surname} et al. ({year})"


def format_apa_references_list(articles: list) -> str:
    """Genera string con múltiples citas APA separadas por coma."""
    if not articles:
        return ""
    
    citations = [format_apa_citation(art) for art in articles]
    
    # Limitar a 5 citas para evitar tablas muy anchas
    if len(citations) > 5:
        citations = citations[:5]
    
    return ", ".join(citations)


# ==============================================================================
# 📊 CATEGORÍAS DINÁMICAS POR DOMINIO Y TIPO DE RQ
# ==============================================================================

# Categorías BASE por tipo de pregunta (genéricas, se adaptan al dominio)
RQ_CATEGORY_BASE = {
    "aplicación": [
        "Sistemas de asistencia virtual",
        "Generación automática de contenido",
        "Evaluación y retroalimentación automatizada",
        "Chatbots y asistentes conversacionales",
        "Personalización de intervenciones",
        "Análisis predictivo",
        "Monitoreo y seguimiento",
        "Gamificación adaptativa"
    ],
    "métricas": [
        "Satisfacción del usuario",
        "Eficacia de la intervención",
        "Tasa de adherencia",
        "Tiempo de respuesta",
        "Engagement/participación",
        "Precisión de recomendaciones",
        "Calidad del contenido generado",
        "Resultados a largo plazo"
    ],
    "algoritmos": [
        "Modelos de lenguaje (GPT, LLaMA)",
        "Redes neuronales recurrentes (RNN/LSTM)",
        "Transformers y atención",
        "Modelos de difusión",
        "Aprendizaje por refuerzo",
        "Redes generativas adversarias (GAN)",
        "Modelos multimodales",
        "Fine-tuning y adaptación de dominio"
    ],
    "arquitecturas": [
        "Arquitectura cliente-servidor",
        "Microservicios",
        "APIs RESTful",
        "Integración con sistemas existentes",
        "Procesamiento en la nube",
        "Edge computing",
        "Arquitectura híbrida",
        "Sistemas distribuidos"
    ],
    "beneficios": [
        "Mayor personalización",
        "Accesibilidad mejorada",
        "Escalabilidad",
        "Retroalimentación inmediata",
        "Reducción de costos",
        "Disponibilidad 24/7",
        "Análisis predictivo",
        "Reducción de barreras de acceso"
    ],
    "desafíos": [
        "Privacidad de datos",
        "Sesgos algorítmicos",
        "Calidad del contenido generado",
        "Brecha digital",
        "Resistencia al cambio",
        "Costos de implementación",
        "Dependencia tecnológica",
        "Validación científica"
    ]
}

# Categorías ESPECÍFICAS por dominio (reemplazan/complementan las genéricas)
DOMAIN_SPECIFIC_CATEGORIES = {
    "salud_mental": {
        "aplicación": [
            "Chatbots para terapia cognitivo-conductual",
            "Sistemas de detección de crisis",
            "Acompañamiento emocional automatizado",
            "Screening y evaluación psicológica",
            "Intervenciones de mindfulness guiadas",
            "Seguimiento de síntomas y estados de ánimo",
            "Psicoeducación automatizada",
            "Grupos de apoyo virtuales"
        ],
        "métricas": [
            "Reducción de ansiedad/depresión",
            "Adherencia al tratamiento",
            "Bienestar emocional percibido",
            "Frecuencia de uso",
            "Satisfacción con la intervención",
            "Reducción de hospitalizaciones",
            "Mejora en calidad de vida",
            "Acceso a servicios de salud mental"
        ],
        "beneficios": [
            "Acceso 24/7 a soporte emocional",
            "Reducción de estigma",
            "Escalabilidad de intervenciones",
            "Anonimato y privacidad",
            "Complemento a terapia tradicional",
            "Detección temprana de crisis",
            "Reducción de listas de espera",
            "Personalización del tratamiento"
        ],
        "desafíos": [
            "Riesgo en situaciones de crisis",
            "Limitaciones en empatía artificial",
            "Validación clínica necesaria",
            "Privacidad de datos sensibles",
            "Supervisión profesional requerida",
            "Sesgo en detección de riesgo",
            "Dependencia del usuario",
            "Regulación ética y legal"
        ]
    },
    "educación": {
        "aplicación": [
            "Sistemas de tutoría inteligente",
            "Generación automática de contenido educativo",
            "Evaluación y retroalimentación automatizada",
            "Chatbots educativos",
            "Personalización de rutas de aprendizaje",
            "Asistentes virtuales de escritura",
            "Análisis de datos de aprendizaje",
            "Gamificación adaptativa"
        ],
        "métricas": [
            "Rendimiento académico",
            "Satisfacción estudiantil",
            "Tasa de retención/abandono",
            "Tiempo de aprendizaje",
            "Engagement/participación",
            "Precisión de recomendaciones",
            "Calidad del contenido generado",
            "Efectividad pedagógica"
        ],
        "beneficios": [
            "Mayor personalización del aprendizaje",
            "Reducción de carga docente",
            "Accesibilidad mejorada",
            "Escalabilidad",
            "Retroalimentación inmediata",
            "Adaptación al ritmo individual",
            "Análisis predictivo de rendimiento",
            "Inclusión educativa"
        ],
        "desafíos": [
            "Privacidad de datos estudiantiles",
            "Sesgos algorítmicos",
            "Plagio y honestidad académica",
            "Brecha digital",
            "Resistencia docente al cambio",
            "Costos de implementación",
            "Dependencia tecnológica",
            "Validación pedagógica"
        ]
    },
    "medicina": {
        "aplicación": [
            "Diagnóstico asistido por IA",
            "Chatbots de triaje médico",
            "Generación de informes clínicos",
            "Monitoreo remoto de pacientes",
            "Asistentes para adherencia a medicación",
            "Análisis de imágenes médicas",
            "Predicción de riesgos de salud",
            "Planificación de tratamientos"
        ],
        "métricas": [
            "Precisión diagnóstica",
            "Reducción de errores médicos",
            "Satisfacción del paciente",
            "Tiempo de espera reducido",
            "Adherencia al tratamiento",
            "Outcomes clínicos",
            "Costo-efectividad",
            "Seguridad del paciente"
        ],
        "beneficios": [
            "Diagnóstico más rápido",
            "Acceso a atención 24/7",
            "Reducción de carga para profesionales",
            "Detección temprana de enfermedades",
            "Medicina personalizada",
            "Escalabilidad de servicios",
            "Reducción de costos",
            "Mejor seguimiento de pacientes"
        ],
        "desafíos": [
            "Responsabilidad legal por errores",
            "Validación clínica rigurosa",
            "Privacidad de datos médicos",
            "Regulación y certificación",
            "Integración con sistemas existentes",
            "Confianza de profesionales",
            "Equidad en el acceso",
            "Explicabilidad de decisiones"
        ]
    },
    "engineering": {
        "aplicación": [
            "Detección de vulnerabilidades en código fuente",
            "Análisis estático de seguridad (SAST)",
            "Reducción de falsos positivos",
            "Generación de código seguro",
            "Auditoría automatizada de código",
            "Análisis de contratos inteligentes",
            "Fuzzing y pruebas de seguridad",
            "Revisión automatizada de código"
        ],
        "métricas": [
            "Tasa de falsos positivos (FPR)",
            "Precisión en detección de vulnerabilidades",
            "Recall y exhaustividad",
            "F1-Score",
            "Tiempo de análisis",
            "Cobertura de código analizado",
            "Tasa de verdaderos positivos (TPR)",
            "AUC-ROC"
        ],
        "algoritmos": [
            "Modelos de lenguaje grande (GPT, LLaMA, CodeLlama)",
            "Transformers para análisis de código",
            "Fine-tuning en datasets de vulnerabilidades",
            "Prompt engineering para seguridad",
            "Análisis basado en grafos (CPG/AST)",
            "Modelos pre-entrenados en código (CodeBERT)",
            "Técnicas de refactorización de código",
            "Aprendizaje por transferencia"
        ],
        "beneficios": [
            "Reducción de falsos positivos vs SAST",
            "Detección de vulnerabilidades complejas",
            "Comprensión semántica del código",
            "Escalabilidad del análisis",
            "Automatización del proceso de revisión",
            "Complemento a herramientas tradicionales",
            "Adaptabilidad a múltiples lenguajes",
            "Explicabilidad de hallazgos"
        ],
        "desafíos": [
            "Alucinaciones y falsos hallazgos",
            "Dependencia de datos de entrenamiento",
            "Limitaciones en contexto de ventana",
            "Memorización vs generalización",
            "Reproducibilidad de resultados",
            "Costo computacional",
            "Ataques adversarios al modelo",
            "Validación en proyectos reales"
        ]
    }
}

# ======================================================================
# 🧠 DETECCIÓN DINÁMICA DE DOMINIO Y CATEGORÍAS (v17 - LLM-based)
# ======================================================================

# Caché global para no repetir llamadas LLM por el mismo tema
_domain_cache = {}         # topic -> domain_name


def _topic_cache_label(topic: str) -> str:
    label = re.sub(r"\s+", " ", str(topic or "").strip().lower())
    return label[:120] or "general"


def build_domain_agnostic_categories(rq_type: str, topic: str = "") -> List[str]:
    """Categorias semilla por eje de PI; no fijan ningun dominio concreto."""
    rq_norm = re.sub(r"[^a-z]", "", str(rq_type or "").lower())
    if "trica" in rq_norm:
        rq_key = "metricas"
    elif "desaf" in rq_norm or "reto" in rq_norm:
        rq_key = "desafios"
    elif "aplic" in rq_norm:
        rq_key = "aplicacion"
    else:
        rq_key = rq_norm or "aplicacion"

    category_map: Dict[str, List[str]] = {
        "datos": [
            "Tipos de datos analizados",
            "Fuentes de informacion utilizadas",
            "Variables relevantes del estudio",
            "Preparacion y calidad de datos",
            "Representaciones del fenomeno estudiado",
        ],
        "algoritmos": [
            "Modelos y metodos principales",
            "Enfoques comparativos o baselines",
            "Tecnicas avanzadas reportadas",
            "Enfoques hibridos o integrados",
            "Adaptacion al contexto de estudio",
        ],
        "metricas": [
            "Metricas de rendimiento",
            "Indicadores de calidad",
            "Criterios de comparacion",
            "Validacion empirica",
            "Analisis de errores o limitaciones",
        ],
        "optimizacion": [
            "Seleccion de variables o caracteristicas",
            "Ajuste de parametros",
            "Validacion y comparacion experimental",
            "Preprocesamiento y normalizacion",
            "Adaptacion o transferencia de modelos",
        ],
        "beneficios": [
            "Aportes metodologicos",
            "Beneficios practicos",
            "Limitaciones reportadas",
            "Brechas de investigacion",
            "Implicaciones para el dominio",
        ],
        "desafios": [
            "Limitaciones metodologicas",
            "Riesgos o sesgos reportados",
            "Restricciones de datos",
            "Barreras de implementacion",
            "Necesidades de validacion",
        ],
        "aplicacion": [
            "Aplicaciones principales",
            "Enfoques metodologicos",
            "Evidencia empirica",
            "Resultados reportados",
            "Limitaciones y brechas",
        ],
    }
    return list(category_map.get(rq_key, category_map["aplicacion"]))
_categories_cache = {}     # (domain, rq_type) -> [categorías]


def detect_domain(topic: str) -> str:
    """
    Detecta el dominio del tema usando LLM (v17: dinámico, no hardcodeado).
    
    Flujo:
    1. Busca en caché si ya detectamos este dominio
    2. Si no, pregunta al LLM cuál es el dominio exacto
    3. Guarda en caché para no repetir
    4. Fallback a keywords si LLM falla
    
    Returns:
        Nombre del dominio en español (ej: 'ingeniería de seguridad de software')
    """
    global _domain_cache
    
    # Caché: si ya lo detectamos, retornar
    topic_key = topic.strip().lower()[:200]
    if topic_key in _domain_cache:
        return _domain_cache[topic_key]

    keyword_domain = _detect_domain_keywords(topic)
    if keyword_domain != 'general':
        _domain_cache[topic_key] = keyword_domain
        logging.info(f"Dominio detectado por keywords: '{keyword_domain}'")
        return keyword_domain
    
    # Intentar detección con LLM
    try:
        from app.llm.ai_model import LocalModel
        model = LocalModel.get_instance()
        
        prompt = f"""Dado este tema de investigación académica:
"{topic}"

¿Cuál es el dominio académico ESPECÍFICO de este tema?

Responde con UNA SOLA FRASE de 3-6 palabras en español. Sé específico con el subdominio.

Ejemplos de respuestas válidas:
- "ingeniería de seguridad de software"
- "educación superior en medicina"
- "psicología clínica infantil"
- "agricultura de precisión"
- "derecho tributario internacional"
- "inteligencia artificial en salud"

Responde SOLO con la frase del dominio, nada más:"""
        
        response = model.generate(prompt, "Detección de dominio", max_tokens=50)
        if response:
            domain = response.strip().strip('"').strip("'").strip('.').lower()
            # Validar que sea razonable (3-50 chars, no sea basura)
            if 3 <= len(domain) <= 60 and '\n' not in domain:
                logging.info(f"🎯 Dominio detectado por LLM: '{domain}'")
                _domain_cache[topic_key] = domain
                return domain
    except Exception as e:
        logging.warning(f"⚠️ Falló detección de dominio por LLM: {e}")
    
    # Fallback: detección por keywords
    domain = _detect_domain_keywords(topic)
    _domain_cache[topic_key] = domain
    logging.info(f"🎯 Dominio detectado por keywords (fallback): '{domain}'")
    return domain


def _detect_domain_keywords(topic: str) -> str:
    """Fallback: detecta dominio con keywords si LLM falla."""
    topic_lower = topic.lower()

    keyword_map = {
        'salud_mental': ['salud mental', 'mental health', 'ansiedad', 'depresión', 'terapia', 'psicológico'],
        'educación': ['educación', 'education', 'aprendizaje', 'learning', 'estudiante', 'docente', 'pedagógico'],
        'ingeniería de software': ['vulnerabilidad', 'código', 'software', 'sast', 'seguridad', 'security', 'llm', 'ciberseguridad'],
        'medicina': ['medicina', 'medicine', 'médico', 'clínico', 'diagnóstico', 'hospital', 'paciente'],
    }
    
    for domain, keywords in keyword_map.items():
        if any(kw in topic_lower for kw in keywords):
            return domain
    
    return 'general'


def get_categories_for_rq(rq_text: str, topic: str = "") -> List[str]:
    """
    Genera categorías dinámicas para una RQ usando LLM (v17).
    
    Flujo:
    1. Detecta el dominio (desde caché o LLM)
    2. Si hay categorías hardcodeadas para este dominio, las usa como seed
    3. Si NO hay, pide al LLM que genere categorías específicas
    4. Guarda en caché para reutilizar
    """
    global _categories_cache
    
    # Etiqueta solo para cache: no activa ramas por dominio ni temas probados.
    domain = _topic_cache_label(topic or rq_text)
    
    # Detectar tipo de RQ
    rq_lower = rq_text.lower()
    rq_type = "aplicación"  # default
    type_keywords = {
        "datos": ["dato", "dataset", "entrenamiento", "corpus", "caracteristica", "caracteristicas", "metrica de software"],
        "optimizacion": ["optimiz", "hiperparam", "seleccion", "balanceo", "validacion cruzada", "representacion"],
        "métricas": ["métrica", "evalua", "mide", "impacto", "resultado", "efectividad", "rendimiento", "compara"],
        "algoritmos": ["algoritmo", "modelo", "técnica", "mecanismo", "método"],
        "beneficios": ["beneficio", "ventaja", "oportunidad", "potencial"],
        "desafíos": ["desafío", "reto", "limitación", "problema", "barrera"],
        "aplicación": ["aplica", "implementa", "usa", "utiliza", "evolución", "evoluciona"],
    }
    for rtype, keywords in type_keywords.items():
        if any(word in rq_lower for word in keywords):
            rq_type = rtype
            break
    
    cache_key = (domain, rq_type)
    if cache_key in _categories_cache:
        return _categories_cache[cache_key]

    categories = build_domain_agnostic_categories(rq_type, topic or rq_text)
    _categories_cache[cache_key] = categories
    return categories
    
    # Buscar en hardcodeados primero
    for hardcoded_key, domain_cats in DOMAIN_SPECIFIC_CATEGORIES.items():
        if (hardcoded_key in domain or domain in hardcoded_key) and rq_type in domain_cats:
            _categories_cache[cache_key] = domain_cats[rq_type]
            return domain_cats[rq_type]
    
    # Si no hay hardcodeados → pedir al LLM que genere categorías
    try:
        from app.llm.ai_model import LocalModel
        model = LocalModel.get_instance()
        
        prompt = f"""Actúa como un experto en revisiones sistemáticas del dominio: "{domain}".

Para la siguiente pregunta de investigación:
"{rq_text}"

Genera exactamente 6 categorías técnicas ESPECÍFICAS del dominio "{domain}" que representen los aportes principales que la literatura académica típicamente reporta para este tipo de pregunta.

REGLAS:
- Cada categoría debe ser una frase de 3-8 palabras
- Deben ser ESPECÍFICAS del dominio "{domain}", NO genéricas
- NO uses categorías de educación a menos que el dominio sea educación
- NO uses categorías de medicina a menos que el dominio sea medicina

Responde SOLO con las 6 categorías separadas por comas, nada más:"""
        
        response = model.generate(prompt, f"Categorías para {domain}/{rq_type}", max_tokens=300)
        if response and ',' in response:
            categories = [c.strip().strip('"').strip("'") for c in response.split(',') if len(c.strip()) > 5]
            if len(categories) >= 3:
                logging.info(f"🧠 Categorías generadas por LLM para [{domain}/{rq_type}]: {len(categories)} categorías")
                _categories_cache[cache_key] = categories
                return categories
    except Exception as e:
        logging.warning(f"⚠️ Falló generación de categorías por LLM: {e}")
    
    # Fallback final: usar categorías base genéricas
    fallback = RQ_CATEGORY_BASE.get(rq_type, RQ_CATEGORY_BASE["aplicación"])
    _categories_cache[cache_key] = fallback
    return fallback




# ==============================================================================
# 🧠 ANALIZADOR RAG BASADO EN EMBEDDINGS
# ==============================================================================

class RAGAnalyzer:
    """
    Analizador RAG que usa Embeddings + Clustering (NO LLM por artículo).
    
    Flujo:
    1. Genera embeddings de categorías predefinidas
    2. Genera embeddings de abstracts de artículos
    3. Clasifica cada artículo en la categoría más similar
    4. Solo usa LLM para el párrafo explicativo final
    """
    
    _embedding_model = None
    
    def __init__(self, use_full_text: bool = True) -> None:
        self.use_full_text = use_full_text
        self.min_categories = 5
        self.max_categories = 10
        self._init_embedding_model()
        
        # Conexión a ChromaDB
        try:
            self.collection = ensure_collection()
        except Exception as e:
            logging.error(f"❌ Error conectando Analizador a ChromaDB: {e}")
            self.collection = None
    
    @classmethod
    def _init_embedding_model(cls) -> None:
        """Inicializa el servicio unificado de embeddings usado por todo el sistema."""
        if cls._embedding_model is None:
            logging.info("🔄 Inicializando servicio de embeddings para RAG...")
            get_single_embedding("healthcheck rag analyzer")
            status = check_service()
            logging.info(
                "✅ Embedding service activo: backend=%s, modelo=%s, dim=%s",
                status.get("backend"),
                status.get("model"),
                status.get("effective_dim"),
            )
            cls._embedding_model = get_embeddings
    
    def get_article_content(self, article: dict) -> str:
        """Obtiene el contenido del artículo para análisis."""
        if self.use_full_text:
            content = article.get('full_text', '') or article.get('pdf_text', '')
            if content and len(content) > 500:
                return content[:4000]  # Limitar para embeddings
        
        # Fallback a abstract + título
        abstract = article.get('abstract', '') or article.get('summary', '')
        title = article.get('title', '')
        return f"{title}. {abstract[:1500]}" if abstract else title

    def classify_article_by_embedding(self, article, categories, category_embeddings):
        """Clasifica un artículo en la categoría más similar usando embeddings."""
        article_content = self.get_article_content(article)
        if not article_content:
            return "Otros", 0.0
            
        article_embedding = get_single_embedding(article_content)
        similarities = cosine_similarity([article_embedding], category_embeddings)[0]
        
        best_idx = np.argmax(similarities)
        best_score = float(similarities[best_idx])
        
        return categories[best_idx], best_score

    def score_article_categories(
        self,
        article: dict,
        categories: List[str],
        category_embeddings: Any,
    ) -> List[Tuple[str, float]]:
        """Devuelve categorias ordenadas por similitud para permitir evidencia multi-etiqueta."""
        article_content = self.get_article_content(article)
        if not article_content or not categories:
            return []

        article_embedding = get_single_embedding(article_content)
        similarities = cosine_similarity([article_embedding], category_embeddings)[0]
        ranked_indexes = np.argsort(similarities)[::-1]
        return [(categories[idx], float(similarities[idx])) for idx in ranked_indexes]

    def _parse_refined_categories(self, raw_response: str, base_categories: List[str]) -> List[str]:
        """Normaliza aportes propuestos por el LLM sin depender del dominio."""
        if not raw_response:
            return base_categories[:5]

        raw = raw_response.replace("```json", "").replace("```", "").strip()
        if "no se identificaron aportes" in raw.lower():
            return []

        candidates = re.split(r"[\n;,]+", raw)
        cleaned: List[str] = []
        seen = set()
        for candidate in candidates:
            item = re.sub(r"^\s*(?:[-*]|\d+[\).\:-])\s*", "", candidate).strip(" \t\r\n\"'")
            item = re.sub(r"\s+", " ", item)
            if re.search(r"\)\s+(?:es|son|resulta|representa)\b", item, flags=re.IGNORECASE):
                continue
            item = re.sub(r"\s*\([^)]*$", "", item).strip(" ()")
            item = re.sub(r"\)\s+(?:es|son|resulta|representa)\b.*$", "", item, flags=re.IGNORECASE).strip(" ()")
            if re.search(r"\b(?:es|son)\s+relevante\b", item, flags=re.IGNORECASE):
                continue
            if re.search(r"\b(?:para|porque|debido a que)\b.*\b(?:es|son)\b", item, flags=re.IGNORECASE):
                continue
            if not item or len(item.split()) < 3:
                continue
            if len(item.split()) > 18:
                item = " ".join(item.split()[:18])
            norm = re.sub(r"\W+", " ", item.lower()).strip()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            cleaned.append(item)
            if len(cleaned) >= 5:
                break

        if len(cleaned) < 3:
            for base in base_categories:
                norm = re.sub(r"\W+", " ", str(base).lower()).strip()
                if norm and norm not in seen:
                    cleaned.append(str(base))
                    seen.add(norm)
                if len(cleaned) >= 5:
                    break

        return cleaned[:5]

    @staticmethod
    def _article_key(article: dict) -> str:
        doi = str(article.get("doi") or article.get("DOI") or "").strip().lower()
        if doi:
            return f"doi:{doi}"
        title = re.sub(r"\W+", " ", str(article.get("title") or "").lower()).strip()
        year = str(article.get("year") or "").strip()
        return f"title:{title}|{year}"

    def _query_evidence(self, query: str, n_results: int = 10) -> str:
        """Recupera los fragmentos de evidencia más relevantes de ChromaDB."""
        if self.collection is None:
            return ""
        
        try:
            query_embedding = get_single_embedding(query).tolist()
            
            # Buscar en la colección
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            snippets = []
            if results and 'documents' in results and len(results['documents']) > 0:
                for docs, metas in zip(results['documents'][0], results['metadatas'][0]):
                    author = metas.get('author')
                    year = metas.get('year')
                    if author and year:
                        snippets.append(f"[{author}, {year}]: {docs}")
                    else:
                        snippets.append(f"[Fragmento de artículo]: {docs}")
            
            return "\n---\n".join(snippets)
        except Exception as e:
            logging.warning(f"⚠️ Error consultando evidencia en ChromaDB: {e}")
            return ""
    
    def analyze_rq(self, rq: str, rq_theme: str, articles: list, topic: str) -> Dict:
        """
        Análisis de una pregunta de investigación usando RAG y LLM para clasificación de alta fidelidad.
        """
        logging.info(f"📊 Analizando RQ con RAG Semántico: {rq[:50]}...")
        
        # 1. Obtener categorías relevantes (Base de conocimiento)
        base_categories = get_categories_for_rq(rq, topic)
        
        # 2. Obtener EVIDENCIA REAL profunda de ChromaDB (RAG de alta fidelidad)
        # Buscamos fragmentos que hablen del tema específico en los papers incluidos
        logging.info(f" 🔍 Recuperando evidencia profunda para '{rq_theme}'...")
        context_text = self._query_evidence(f"{rq_theme}. {rq}", n_results=12)
        
        if not context_text:
            # Fallback a abstracts si ChromaDB falla o está vacía
            context_samples = []
            for art in articles[:8]:
                content = self.get_article_content(art)
                if content:
                    context_samples.append(content[:500])
            context_text = "\n---\n".join(context_samples)
        
        # Usar LLM para proponer categorías basadas en el contenido REAL
        refined_categories = base_categories
        try:
            model = LocalModel.get_instance()
            
            refinement_prompt = f"""Actúa como un Especialista en Síntesis de Evidencia para el dominio: "{topic}".
Observa estos fragmentos de artículos sobre la pregunta: "{rq}".
Extrae exactamente 4-5 "Aportes" técnicos y específicos (no genéricos) que se encuentran en estos textos.

Fragmentos de los artículos seleccionados:
{context_text[:3000]}

Categorías sugeridas (ajústalas si los textos dicen algo más específico):
{", ".join(base_categories)}

REGLAS CRÍTICAS DE EVIDENCIA (PhD Standard):
1. SI NO ENCUENTRAS EVIDENCIA real en los fragmentos para esta pregunta, responde exactamente: "No se identificaron aportes específicos en la literatura seleccionada para esta pregunta de investigación."
2. PROHIBIDO INVENTAR AUTORES: Solo usa nombres que veas en los fragmentos (ej: [Autor, Año]).
3. PROHIBIDO ALUCINAR TEMAS: Los aportes DEBEN estar ESTRICTAMENTE dentro del dominio "{topic}". Si los textos hablan de vulnerabilidades de código, TODOS los aportes deben ser sobre seguridad de software, NO sobre educación, pedagogía, medicina u otros dominios ajenos.
4. PROHIBIDO mezclar dominios: NO menciones docentes, estudiantes, aulas, gramática, textos argumentativos, ni ningún concepto de educación — a menos que el tema de investigación sea explícitamente educativo.
5. Si solo hay información parcial, menciónalo honestamente.
6. Cada aporte debe responder a ESTA pregunta, no a otra PI de la revision.
7. Cada aporte debe ser una frase técnica concisa (5-15 palabras) directamente extraída de la evidencia.

Responde ÚNICAMENTE con una lista de 5 aportes en español, separados por comas. NO incluyas explicaciones, solo los aportes.
"""
            llm_response = model.generate(refinement_prompt, f"Cat refinement: {rq_theme}", max_tokens=800)
            refined_categories = self._parse_refined_categories(llm_response, base_categories)
        except Exception as e:
            logging.warning(f"Error refinando categorías con LLM: {e}")

        # 3. Clasificar artículos en estas categorías REFINADAS
        if not refined_categories:
            return {
                'rq': rq,
                'theme': rq_theme,
                'categories': [],
                'total_articles': 0,
                'no_contribution': len(articles)
            }

        category_embeddings = get_embeddings(refined_categories)
        category_articles = defaultdict(list)
        article_assignments = []
        no_contribution = 0

        for article in articles:
            ranked_categories = self.score_article_categories(article, refined_categories, category_embeddings)
            if not ranked_categories or ranked_categories[0][1] < 0.19:
                no_contribution += 1
                continue

            primary_category = ranked_categories[0][0]
            category_articles[primary_category].append(article)
            article_assignments.append({
                "key": self._article_key(article),
                "article": article,
                "primary": primary_category,
                "ranked": dict(ranked_categories),
            })

        # Cada articulo cuenta una sola vez. Solo redistribuimos desde categorias
        # dominantes cuando hay una segunda categoria cercana y vacia.
        target_min_categories = min(4, len(refined_categories), len(article_assignments))
        moved_keys = set()
        while len([c for c, arts in category_articles.items() if arts]) < target_min_categories:
            empty_categories = [c for c in refined_categories if not category_articles.get(c)]
            if not empty_categories:
                break

            moved = False
            for target_category in empty_categories:
                best_move = None
                best_score = 0.0
                for assignment in article_assignments:
                    if assignment["key"] in moved_keys:
                        continue
                    source_category = assignment["primary"]
                    if source_category == target_category or len(category_articles[source_category]) <= 1:
                        continue

                    target_score = float(assignment["ranked"].get(target_category, 0.0))
                    source_score = float(assignment["ranked"].get(source_category, 0.0))
                    if target_score >= 0.18 and (source_score - target_score) <= 0.18 and target_score > best_score:
                        best_move = assignment
                        best_score = target_score

                if not best_move:
                    continue

                source_category = best_move["primary"]
                article = best_move["article"]
                category_articles[source_category] = [
                    a for a in category_articles[source_category]
                    if self._article_key(a) != best_move["key"]
                ]
                category_articles[target_category].append(article)
                best_move["primary"] = target_category
                moved_keys.add(best_move["key"])
                moved = True
                break

            if not moved:
                break
        # 4. Formatear resultados
        result_categories = []
        total_in_results = sum(len(arts) for arts in category_articles.values())
        
        for cat_name, arts in category_articles.items():
            if not arts:
                continue
            # Recolectar referencias para esta categoría
            ref_list = []
            for art in arts:
                # Assuming format_apa_citation is defined elsewhere and available
                # If not, this will cause an error. I'll assume it's available.
                ref = format_apa_citation(art) 
                if ref and ref not in ref_list:
                    ref_list.append(ref)
            
            # La cantidad cuenta articulos asignados, aunque una referencia se repita por metadata.
            count = len(arts)
            
            result_categories.append({
                'name': cat_name,
                'count': count,
                'percentage': (count / total_in_results * 100) if total_in_results > 0 else 0,
                'references': ", ".join(ref_list)
            })
            
        # Ordenar por frecuencia
        result_categories.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            'rq': rq,
            'theme': rq_theme,
            'categories': result_categories,
            'total_articles': total_in_results,
            'no_contribution': no_contribution
        }
    
    def analyze_all_rqs(self, rqs: List[List[str]], articles: list, topic: str) -> List[Dict]:
        """Analiza todas las preguntas de investigación."""
        results = []
        
        for rq_data in rqs:
            if len(rq_data) >= 3:
                rq_num = rq_data[0]
                rq_theme = rq_data[1]
                rq_question = rq_data[2]
                
                analysis = self.analyze_rq(rq_question, rq_theme, articles, topic)
                analysis['number'] = rq_num
                results.append(analysis)
        
        return results


# ==============================================================================
# 📝 GENERADOR DE TEXTO EXPLICATIVO (USA LLM SOLO AQUÍ)
# ==============================================================================

def generate_rq_explanation(rq_analysis: Dict, topic: str) -> str:
    """
    Genera un párrafo explicativo para los resultados de una RQ.
    ESTA ES LA ÚNICA FUNCIÓN QUE USA LLM (1 llamada por RQ = máx 5 llamadas).
    """

    categories = rq_analysis.get('categories', [])
    if not categories:
        return "No se identificaron aportes significativos para esta pregunta de investigación."
    
    # Preparar datos para el prompt
    top_categories = categories[:3]
    cat_summary = "\n".join([
        f"- {cat['name']}: {cat['count']} artículos ({cat['percentage']:.1f}%)" 
        for cat in top_categories
    ])
    
    total = rq_analysis.get('total_articles', 0)
    rq = rq_analysis.get('rq', '')
    theme = rq_analysis.get('theme', '')
    
    # 6. Preparar EVIDENCIA para la explicación final (PhD level) (Usando analyzer local)
    analyzer = RAGAnalyzer()
    
    top_cat_names = [c['name'] for c in categories[:2]]
    evidence_query = f"{rq} {', '.join(top_cat_names)}"
    deep_evidence = analyzer._query_evidence(evidence_query, n_results=8)
    
    prompt = f"""Actúa como un Especialista en Síntesis de Evidencia (PhD).
Escribe un análisis académico profundo (aprox 120-150 palabras) sobre los hallazgos de la tabla de aportes detallada abajo.

PREGUNTA DE INVESTIGACIÓN (PI): {rq}
TEMA DE ANÁLISIS: {theme}
DATOS DE LA TABLA (Categorías y Frecuencias):
{cat_summary}
TOTAL DE ARTÍCULOS ANALIZADOS PARA ESTA PI: {total}

EVIDENCIA TÉCNICA EXTRAÍDA DE LOS PAPERS (USA ESTO PARA SER ASERTIVO):
{deep_evidence if deep_evidence else 'No hay fragmentos adicionales, usa el contexto de la tabla.'}

REGLAS DE ORO (SISTÉMICAS):
1. ANÁLISIS DE MECANISMOS (ASERTIVIDAD Q1): No uses frases dubitativas como "probablemente". Usa la evidencia técnica para explicar el "cómo" o "por qué". 
   - Ejemplo: "La superioridad de X se fundamenta en su capacidad de [Mecanismo], logrando una reducción de [Efecto] según se observa en la evidencia..."
2. FORMATO APA ESTRICTO: NUNCA menciones títulos de papers. Usa exclusivamente "Autor (Año)". Valida el año de la metadata de evidencia si está disponible.
3. REFERENCIA A TABLA: Inicia el párrafo refiriéndote a los resultados presentados.
4. FLUIDEZ Y CONECTORES: Usa conectores de alto nivel.
5. DINAMISMO: El análisis debe ser profundo para cualquier dominio.

ESCRIBE EL ANÁLISIS EN ESPAÑOL:"""
    
    try:
        model = LocalModel.get_instance()
        response = model.generate(prompt, "", max_tokens=400)
        
        # Limpiar respuesta
        response = response.strip()
        response = re.sub(r'^(Párrafo:|Respuesta:|Explicación:)\s*', '', response, flags=re.IGNORECASE)
        return response.strip()
        
    except Exception as e:
        logging.warning(f"Error generando explicación: {e}")
        # Fallback con texto generado localmente (sin LLM)
        return (
            f"El análisis de los {total} artículos que aportan a esta pregunta revela "
            f"tendencias significativas en las siguientes áreas: {cat_summary}. "
            f"Estos hallazgos reflejan las principales líneas de investigación "
            f"identificadas en la literatura sobre {theme}. "
            f"La distribución muestra una concentración notable en las categorías principales, "
            f"evidenciando áreas prioritarias para la investigación actual."
        )
