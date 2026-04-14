import os
from dotenv import load_dotenv

load_dotenv()

# ==========================
# 🚀 GOOGLE GEMINI API (RECOMENDADO - NIVEL GRATUITO)
# ==========================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5"),
]
GEMINI_API_KEYS = [k for k in GEMINI_API_KEYS if k]  # Filtrar vacíos/None

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

# ==========================
# 1. CEREBRO: GitHub Models (GPT-4o)
# ==========================

GITHUB_GPT4O_TOKEN = os.getenv("GITHUB_GPT4O_TOKEN")
GITHUB_MODELS_ENDPOINT = os.getenv("GITHUB_MODELS_ENDPOINT", "https://models.github.ai/inference")
GITHUB_GPT4O_MODEL = os.getenv("GITHUB_GPT4O_MODEL", "gpt-4o")
PROMPT_GENERATION_MODEL = os.getenv("PROMPT_GENERATION_MODEL", "gpt-4o")

# LOCAL EXTRACTOR (Qwen 2.5 3B GGUF)
# ==========================
# v12.2: Reactivado — Soporte CUDA habilitado.
# Auto-detecta GPU o CPU automáticamente.
ENABLE_LOCAL_EXTRACTOR = os.getenv("ENABLE_LOCAL_EXTRACTOR", "True").lower() == "true"
LOCAL_EXTRACTOR_PATH = os.getenv("LOCAL_EXTRACTOR_PATH", "models/qwen2.5-3b-instruct-q4_k_m.gguf")


# ==============================================================================
# 2. CEREBRAS (Llama 3.3 70B) - PROVIDER PRINCIPAL (1M tokens/día gratis)
# ==============================================================================
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
CEREBRAS_API_KEYS = [
    os.getenv("CEREBRAS_API_KEY"),
    os.getenv("CEREBRAS_API_KEY_2"),
    os.getenv("CEREBRAS_API_KEY_3"),
]
CEREBRAS_API_KEYS = [k for k in CEREBRAS_API_KEYS if k]  # Filtrar vacíos/None

CEREBRAS_MODEL = "llama3.1-8b"  # v12.6: Verificado vía API (/v1/models)
CEREBRAS_ENDPOINT = "https://api.cerebras.ai/v1/chat/completions"

# ==============================================================================
# 3. GROQ (Llama 3.3 70B) - PARA SYNTHESIS
# ==============================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Usamos la variable estándar del .env
GROQ_MODEL = "llama-3.3-70b-versatile"  # Modelo más potente
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# ==========================
# 3. MÚSCULO AUXILIAR: OpenRouter (Modelos 100% Gratuitos)
# ==========================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Modelo PRINCIPAL gratuito: OpenAI gpt-oss-120b (120B MoE, 131K ctx, $0, bajo tráfico)
# Open-weight de OpenAI — muy potente y menos saturado que Llama/Mistral
OPENROUTER_MODEL = "openai/gpt-oss-120b"
# Modelo ALTERNATIVO 1: Qwen3 Next 80B A3B (80B, 262K ctx, $0, tráfico muy bajo)
OPENROUTER_MODEL_ALT = "qwen/qwen3-next-80b:free"
# Modelo ALTERNATIVO 2: Arcee Trinity Large (400B MoE, 131K ctx, $0)
OPENROUTER_MODEL_ALT2 = "arcee-ai/trinity-large-preview:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_RATE_LIMIT_RPM = 20
OPENROUTER_RATE_LIMIT_TPM = 200000

# ==========================
# 4. APIs Externas
# ==========================

DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
DEEPL_API_URL = os.getenv("DEEPL_API_URL", "https://api-free.deepl.com/v2/translate")

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
SEMANTIC_SCHOLAR_MAX_RESULTS = 100
SEMANTIC_SCHOLAR_RATE_LIMIT = 1.0

# Email institucional para APIs académicas (OpenAlex polite pool, Europe PMC)
ACADEMIC_EMAIL = "jnacarinoa1@upao.edu.pe"

REDALYC_API_KEY = os.getenv("REDALYC_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

# ==========================
# 5. Base de Datos y Vectores
# ==========================

MONGODB_URI = os.getenv("MONGODB_URI")
ENABLE_MONGODB = os.getenv("ENABLE_MONGODB", "False").lower() == "true"
MILVUS_URI = "chroma_db"
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "articles_collection_v5")  # v5: nueva dimensión 384

# v17.3: Cambiado de all-mpnet-base-v2 (110M, 768-dim) a all-MiniLM-L6-v2 (22M, 384-dim)
# Resultado: 5x más rápido en CPU. De ~12 min a ~2.5 min en el indexado inicial.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ========================================
# ✅ THRESHOLDS
# ========================================

SIMILARITY_RELEVANT = 0.70   
SIMILARITY_MAYBE = 0.60      
DUPLICATE_THRESHOLD = 0.85  
GROK_PREFILTER_THRESHOLD = 0.50  
PRISMA_AUTO_THRESHOLD = 65  

# ==========================
# 6. Columnas Dinámicas
#==========================

DYNAMIC_COLUMNS = {
    "summary": "Resumen",
    "objectives": "Objetivo",
    "methodology": "Metodología",
    "subject": "Objeto de estudio",
    "variables": "Variables",
    "key_findings": "Hallazgos",
    "limitations": "Limitaciones"
}

# ==========================
# Cache y Estrategia
# ==========================

COLUMN_GENERATION_STRATEGY = "hybrid"
ENABLE_CACHE = True

# ==========================
# Validación
# ==========================

def validate_config():
    """Valida configuración"""
    warnings = [] 

    if not GITHUB_MODELS_TOKEN:
        warnings.append("⚠️ GITHUB_MODELS_TOKEN (Grok-3) no configurado.")  

    if not GROQ_API_KEY:
        warnings.append("⚠️ GROQ_API_KEY (Llama 3.3 70B) no configurada.")

    if not OPENROUTER_API_KEY:
        warnings.append("⚠️ OPENROUTER_API_KEY (GPT-4o-mini) no configurada.")

    if not DEEPL_API_KEY:
        warnings.append("⚠️ DEEPL_API_KEY no configurada.")

    if not SEMANTIC_SCHOLAR_API_KEY:
        warnings.append("⚠️ SEMANTIC_SCHOLAR_API_KEY no configurada.")

    if not HUGGINGFACE_API_KEY:
        warnings.append("⚠️ HUGGINGFACE_API_KEY no configurada (Necesaria para síntesis remota).")

    if warnings:
        for w in warnings:
            print(w)

    print("✅ Configuración validada")
    print(f"   - Cerebro (Reglas): {PROMPT_GENERATION_MODEL}")
    print(f"   - Synthesis (Narrativa): Groq - {GROQ_MODEL}")
    print(f"   - Músculo (Términos): {OPENROUTER_MODEL}")
    print(f"   - Estrategia Columnas: {COLUMN_GENERATION_STRATEGY.upper()}")
    print(f"\n📊 Thresholds configurados:")
    print(f"   - Pre-filtro Grok-3: {int(GROK_PREFILTER_THRESHOLD*100)}%")
    print(f"   - ✅ Screening semántico: {int(SIMILARITY_RELEVANT*100)}% (con texto completo)")
    print(f"   - PRISMA automático: {PRISMA_AUTO_THRESHOLD}%")
    print(f"   - Duplicados: {int(DUPLICATE_THRESHOLD*100)}%")