"""
Configuración de categorías de análisis para diferentes dominios de investigación.
Personaliza esto según tu área de estudio para mejorar las tablas de resultados.
"""

# ==============================================================================
# 📊 PATRONES DE CATEGORIZACIÓN POR DOMINIO
# ==============================================================================

CATEGORY_PATTERNS = {
    
    # === MACHINE LEARNING / IA ===
    "machine_learning": {
        "Modelos de Aprendizaje Supervisado": [
            "random forest", "decision tree", "svm", "support vector machine",
            "logistic regression", "naive bayes", "k-nearest", "knn",
            "gradient boosting", "xgboost", "lightgbm"
        ],
        "Redes Neuronales Profundas": [
            "neural network", "deep learning", "cnn", "convolutional",
            "rnn", "recurrent", "lstm", "gru", "transformer",
            "attention mechanism", "bert", "gpt"
        ],
        "Modelos Generativos": [
            "gan", "generative adversarial", "vae", "variational autoencoder",
            "diffusion model", "generative model", "autoencoder"
        ],
        "Aprendizaje No Supervisado": [
            "clustering", "k-means", "hierarchical clustering",
            "dbscan", "pca", "dimensionality reduction", "t-sne"
        ],
        "Métricas de Evaluación": [
            "accuracy", "precision", "recall", "f1-score", "f1 score",
            "auc", "roc curve", "confusion matrix", "sensitivity",
            "specificity", "mse", "mae", "rmse"
        ],
        "Técnicas de Optimización": [
            "adam", "sgd", "optimization", "hyperparameter tuning",
            "grid search", "random search", "cross-validation"
        ]
    },
    
    # === SALUD / MEDICINA ===
    "healthcare": {
        "Tipos de Estudio": [
            "randomized controlled trial", "rct", "cohort study",
            "case-control", "cross-sectional", "systematic review",
            "meta-analysis", "clinical trial"
        ],
        "Métodos Diagnósticos": [
            "diagnostic", "screening", "detection", "prediction model",
            "risk assessment", "biomarker", "imaging"
        ],
        "Poblaciones de Estudio": [
            "adult", "pediatric", "elderly", "geriatric",
            "pregnant", "diabetes", "hypertension", "cardiovascular"
        ],
        "Intervenciones": [
            "treatment", "therapy", "intervention", "medication",
            "surgery", "rehabilitation", "prevention"
        ],
        "Resultados Clínicos": [
            "mortality", "morbidity", "quality of life", "adverse events",
            "efficacy", "effectiveness", "safety"
        ]
    },
    
    # === EDUCACIÓN ===
    "education": {
        "Metodologías Pedagógicas": [
            "active learning", "flipped classroom", "blended learning",
            "online learning", "e-learning", "gamification",
            "project-based learning", "collaborative learning"
        ],
        "Tecnologías Educativas": [
            "learning management system", "lms", "mooc",
            "virtual reality", "augmented reality", "educational software",
            "adaptive learning", "intelligent tutoring"
        ],
        "Niveles Educativos": [
            "primary education", "secondary", "higher education",
            "university", "k-12", "undergraduate", "graduate"
        ],
        "Evaluación del Aprendizaje": [
            "assessment", "formative assessment", "summative assessment",
            "learning outcomes", "competencies", "skills development"
        ]
    },
    
    # === CIENCIAS SOCIALES ===
    "social_sciences": {
        "Métodos de Investigación": [
            "qualitative", "quantitative", "mixed methods",
            "ethnography", "case study", "survey", "interview",
            "focus group", "participatory research"
        ],
        "Teorías Aplicadas": [
            "social cognitive theory", "theory of planned behavior",
            "diffusion of innovations", "social network theory",
            "institutional theory"
        ],
        "Análisis de Datos": [
            "thematic analysis", "content analysis", "discourse analysis",
            "statistical analysis", "regression", "structural equation"
        ]
    },
    
    # === INGENIERÍA DE SOFTWARE ===
    "software_engineering": {
        "Metodologías de Desarrollo": [
            "agile", "scrum", "kanban", "waterfall", "devops",
            "continuous integration", "continuous deployment", "ci/cd"
        ],
        "Arquitecturas de Software": [
            "microservices", "monolithic", "serverless",
            "event-driven", "service-oriented", "layered architecture"
        ],
        "Pruebas de Software": [
            "unit testing", "integration testing", "system testing",
            "test-driven development", "tdd", "automated testing"
        ],
        "Calidad de Software": [
            "code quality", "technical debt", "code smell",
            "refactoring", "maintainability", "reliability"
        ]
    },
    
    # === REALIDAD VIRTUAL / AUMENTADA ===
    "vr_ar": {
        "Tipos de Tecnología": [
            "virtual reality", "augmented reality", "mixed reality",
            "extended reality", "xr", "immersive technology"
        ],
        "Aplicaciones": [
            "training", "simulation", "education", "entertainment",
            "therapy", "rehabilitation", "design", "visualization"
        ],
        "Dispositivos": [
            "head-mounted display", "hmd", "oculus", "htc vive",
            "hololens", "ar glasses", "haptic device"
        ],
        "Métricas de Usabilidad": [
            "presence", "immersion", "simulator sickness",
            "user experience", "usability", "engagement"
        ]
    },
    
    # === CATEGORÍAS GENERALES (Siempre activas) ===
    "general": {
        "Tipos de Datos": [
            "demographic data", "clinical data", "sensor data",
            "behavioral data", "survey data", "experimental data",
            "observational data", "secondary data"
        ],
        "Limitaciones Comunes": [
            "small sample size", "bias", "confounding",
            "generalizability", "validity", "reliability",
            "missing data", "selection bias"
        ],
        "Recomendaciones Futuras": [
            "further research", "larger sample", "longitudinal study",
            "replication", "validation", "multicenter study"
        ]
    }
}

# ==============================================================================
# 🎯 FUNCIÓN DE AUTO-DETECCIÓN DE DOMINIO
# ==============================================================================

def detect_domain_from_question(question: str) -> str:
    """
    Detecta automáticamente el dominio de investigación según la pregunta.
    Retorna la clave del dominio para usar los patrones correctos.
    """
    question_lower = question.lower()
    
    # Patrones de detección
    domain_indicators = {
        "machine_learning": [
            "machine learning", "deep learning", "neural network",
            "algoritmo", "modelo de aprendizaje", "clasificación",
            "predicción", "inteligencia artificial", "ia"
        ],
        "healthcare": [
            "salud", "medicina", "paciente", "clínico", "diagnóstico",
            "tratamiento", "enfermedad", "diabetes", "cáncer", "health"
        ],
        "education": [
            "educación", "aprendizaje", "estudiante", "enseñanza",
            "pedagogía", "e-learning", "educational", "learning"
        ],
        "vr_ar": [
            "realidad virtual", "realidad aumentada", "vr", "ar",
            "inmersivo", "simulación", "virtual reality"
        ],
        "software_engineering": [
            "software", "desarrollo", "programación", "código",
            "testing", "arquitectura", "metodología ágil"
        ],
        "social_sciences": [
            "social", "comportamiento", "actitud", "percepción",
            "cualitativo", "etnografía", "entrevista"
        ]
    }
    
    # Contar coincidencias
    scores = {}
    for domain, keywords in domain_indicators.items():
        score = sum(1 for kw in keywords if kw in question_lower)
        if score > 0:
            scores[domain] = score
    
    # Retornar el dominio con mayor puntuación
    if scores:
        detected_domain = max(scores, key=scores.get)
        return detected_domain
    
    # Por defecto, usar general + machine_learning (más común)
    return "machine_learning"

def get_patterns_for_question(question: str) -> dict:
    """
    Retorna los patrones de categorización apropiados para la pregunta.
    Combina el dominio específico con categorías generales.
    """
    domain = detect_domain_from_question(question)
    
    # Combinar patrones del dominio + generales
    patterns = {}
    
    if domain in CATEGORY_PATTERNS:
        patterns.update(CATEGORY_PATTERNS[domain])
    
    # Siempre agregar categorías generales
    patterns.update(CATEGORY_PATTERNS["general"])
    
    return patterns

# ==============================================================================
# 📝 EJEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    # Prueba de detección
    test_questions = [
        "¿Cuál es el impacto de la IA Generativa en la personalización del aprendizaje universitario?",
        "¿Qué modelos de Machine Learning se usan para detectar diabetes tipo 2?",
        "¿Cómo mejora la realidad virtual el entrenamiento en arquitectura?",
    ]
    
    for q in test_questions:
        domain = detect_domain_from_question(q)
        patterns = get_patterns_for_question(q)
        
        print(f"\nPregunta: {q}")
        print(f"Dominio detectado: {domain}")
        print(f"Categorías disponibles: {list(patterns.keys())}")