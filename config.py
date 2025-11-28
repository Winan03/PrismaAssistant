import os
from dotenv import load_dotenv

load_dotenv()

# ==========================
# 1. CEREBRO: GitHub Models (Grok-3)
# ==========================

GITHUB_MODELS_TOKEN = os.getenv("GITHUB_MODELS_TOKEN")
GITHUB_MODELS_ENDPOINT = os.getenv("GITHUB_MODELS_ENDPOINT", "https://models.github.ai/inference")
PROMPT_GENERATION_MODEL = os.getenv("PROMPT_GENERATION_MODEL", "xai/grok-3")

# ==============================================================================
# 2. GROQ (Llama 3.3 70B) - PARA SYNTHESIS
# ==============================================================================
GROQ_API_KEY = os.getenv("GROQ_syntesis_TOKEN")  # ← Cambiado
GROQ_MODEL = "llama-3.3-70b-versatile"  # Modelo más potente
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# ==========================
# 3. MÚSCULO AUXILIAR: OpenRouter (Backup)
# ==========================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://models.github.ai/inference")
OPENROUTER_RATE_LIMIT_RPM = 15
OPENROUTER_RATE_LIMIT_TPM = 150000

# ==========================
# 4. APIs Externas
# ==========================

DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
DEEPL_API_URL = os.getenv("DEEPL_API_URL", "https://api-free.deepl.com/v2/translate")

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
SEMANTIC_SCHOLAR_MAX_RESULTS = 100
SEMANTIC_SCHOLAR_RATE_LIMIT = 1.0

REDALYC_API_KEY = os.getenv("REDALYC_API_KEY")

# ==========================
# 5. Base de Datos y Vectores
# ==========================

MONGODB_URI = os.getenv("MONGODB_URI")
MILVUS_URI = "chroma_db"
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "articles_collection")

EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDDING_DIM = 768

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
    "summary": "Resumen (Español)",
    "methodology": "Metodología",
    "population": "Población",
    "key_findings": "Hallazgos Clave",
    "limitations": "Limitaciones",
    "conclusions": "Conclusiones"
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