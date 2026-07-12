import os
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

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
GEMINI_API_KEYS = [k for k in GEMINI_API_KEYS if k]

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
CEREBRAS_API_KEYS = [k for k in CEREBRAS_API_KEYS if k]

CEREBRAS_MODEL = "llama3.1-8b"
CEREBRAS_ENDPOINT = "https://api.cerebras.ai/v1/chat/completions"

# ==============================================================================
# 3. GROQ (Llama 3.3 70B) - PARA SYNTHESIS
# ==============================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# ==========================
# 3. MÚSCULO AUXILIAR: OpenRouter (Modelos 100% Gratuitos)
# ==========================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "openai/gpt-oss-120b"
OPENROUTER_MODEL_ALT = "qwen/qwen3-next-80b:free"
OPENROUTER_MODEL_ALT2 = "arcee-ai/trinity-large-preview:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_RATE_LIMIT_RPM = 20
OPENROUTER_RATE_LIMIT_TPM = 200000

# ==========================================
# 🤖 DEEPSEEK (V4 Flash) — PROVIDER PRIORITARIO
# ==========================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"

# ==========================================
# 🌐 OLLAMA CLOUD / LOCAL CONFIGURATION
# ==========================================
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_ENDPOINT_DEFAULT = (
    "https://ollama.com/v1/chat/completions"
    if OLLAMA_API_KEY
    else "http://localhost:11434/v1/chat/completions"
)
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", OLLAMA_ENDPOINT_DEFAULT)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")
OLLAMA_MODEL_PLANNER = os.getenv("OLLAMA_MODEL_PLANNER", OLLAMA_MODEL)
OLLAMA_MODEL_JUDGE = os.getenv("OLLAMA_MODEL_JUDGE", OLLAMA_MODEL)
OLLAMA_MODEL_FAST = os.getenv("OLLAMA_MODEL_FAST", OLLAMA_MODEL)
OLLAMA_ROLE_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv("OLLAMA_ROLE_FALLBACK_MODELS", f"{OLLAMA_MODEL_JUDGE},{OLLAMA_MODEL_FAST}").split(",")
    if model.strip()
]
OLLAMA_IS_CLOUD = "ollama.com" in OLLAMA_ENDPOINT.lower()
OLLAMA_AUTO_PULL = os.getenv("OLLAMA_AUTO_PULL", "False").lower() == "true"
OLLAMA_STAGE4_CONCURRENCY = int(os.getenv("OLLAMA_STAGE4_CONCURRENCY", "100"))
STAGE4_LLM_MAX_TOKENS = int(os.getenv("STAGE4_LLM_MAX_TOKENS", "4096"))
GOLD_EVAL_STAGE4_MAX_CHARS = int(os.getenv("GOLD_EVAL_STAGE4_MAX_CHARS", "9000"))

# Umbral mínimo de caracteres para usar abstract en Stage 4 (~150 palabras)
ABSTRACT_MIN_CHARS = int(os.getenv("ABSTRACT_MIN_CHARS", "800"))
GOLD_EVAL_ABSTRACT_MIN_CHARS = int(os.getenv("GOLD_EVAL_ABSTRACT_MIN_CHARS", "800"))

# ==========================
# 4. APIs Externas
# ==========================

DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
DEEPL_API_URL = os.getenv("DEEPL_API_URL", "https://api-free.deepl.com/v2/translate")

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
SEMANTIC_SCHOLAR_MAX_RESULTS = 100
SEMANTIC_SCHOLAR_RATE_LIMIT = 1.0

ACADEMIC_EMAIL = os.getenv("ACADEMIC_EMAIL", "info@prisma-assistant.edu")

REDALYC_API_KEY = os.getenv("REDALYC_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

# ==========================
# 5. Base de Datos y Vectores
# ==========================

MONGODB_URI = os.getenv("MONGODB_URI")
ENABLE_MONGODB = os.getenv("ENABLE_MONGODB", "False").lower() == "true"
MILVUS_URI = "chroma_db"
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "articles_collection_v8_specter")

# ==========================
# Internal thesis evaluation mode
# ==========================
ENABLE_INTERNAL_EVALUATION = os.getenv("ENABLE_INTERNAL_EVALUATION", "False").lower() == "true"
INTERNAL_EVALUATION_TOKEN = os.getenv("INTERNAL_EVALUATION_TOKEN", "")

# ==========================================
# 🧠 CONFIGURACIÓN DE EMBEDDINGS OLLAMA LOCAL
# ==========================================
USE_OLLAMA_EMBEDDING = os.getenv("USE_OLLAMA_EMBEDDING", "True").lower() == "true"
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_EMBEDDING_BASE_URL = os.getenv("OLLAMA_EMBEDDING_BASE_URL", "http://localhost:11434")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/allenai-specter")

CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

# ========================================
# ✅ THRESHOLDS
# ========================================

SIMILARITY_RELEVANT = 0.70
SIMILARITY_MAYBE = 0.60
DUPLICATE_THRESHOLD = 0.85
GROK_PREFILTER_THRESHOLD = 0.50
PRISMA_AUTO_THRESHOLD = 65

MIN_CITATION_THRESHOLD_OLD = int(os.getenv("MIN_CITATION_THRESHOLD_OLD", "25"))
MIN_CITATION_THRESHOLD_MID = int(os.getenv("MIN_CITATION_THRESHOLD_MID",  "5"))
MIN_CITATION_THRESHOLD_NEW = int(os.getenv("MIN_CITATION_THRESHOLD_NEW",  "0"))
MIN_CITATION_THRESHOLD = int(os.getenv("MIN_CITATION_THRESHOLD", "3"))

# ==========================
# 5.1 Cache and query quality guards
# ==========================

CACHE_LLM_VALIDATION_ENABLED = os.getenv("CACHE_LLM_VALIDATION_ENABLED", "True").lower() == "true"
ELIGIBILITY_CONTRACT_ENABLED = os.getenv("ELIGIBILITY_CONTRACT_ENABLED", "True").lower() == "true"
ATOM_COVERAGE_RANKING_ENABLED = os.getenv("ATOM_COVERAGE_RANKING_ENABLED", "True").lower() == "true"
ATOM_COVERAGE_MIN_DIRECT = float(os.getenv("ATOM_COVERAGE_MIN_DIRECT", "0.55"))

INCLUDE_OPTIONAL_TERMS_IN_STRUCTURED_SEARCH = (
    os.getenv("INCLUDE_OPTIONAL_TERMS_IN_STRUCTURED_SEARCH", "False").lower() == "true"
)
CITATION_FILTER_MODE = os.getenv("CITATION_FILTER_MODE", "rank_only").strip().lower()
SEARCH_QUERY_TIER_LIMITS = {
    "core": int(os.getenv("SEARCH_QUERY_TIER_CORE_LIMIT", "6")),
    "focused": int(os.getenv("SEARCH_QUERY_TIER_FOCUSED_LIMIT", "10")),
    "exact": int(os.getenv("SEARCH_QUERY_TIER_EXACT_LIMIT", "8")),
    "scout": int(os.getenv("SEARCH_QUERY_TIER_SCOUT_LIMIT", "4")),
}
SEARCH_PRESERVE_SHORT_TOKENS = {
    token.strip().lower()
    for token in os.getenv("SEARCH_PRESERVE_SHORT_TOKENS", "ai,ml,nlp,ar,vr,xr").split(",")
    if token.strip()
}
SEMANTIC_SCHOLAR_QUERY_MAX_TERMS = int(os.getenv("SEMANTIC_SCHOLAR_QUERY_MAX_TERMS", "12"))
SEMANTIC_SCHOLAR_RELEVANCE_PER_QUERY = int(os.getenv("SEMANTIC_SCHOLAR_RELEVANCE_PER_QUERY", "30"))
SEMANTIC_SCHOLAR_CITATION_PER_QUERY = int(os.getenv("SEMANTIC_SCHOLAR_CITATION_PER_QUERY", "15"))
SEMANTIC_SCHOLAR_CITATION_QUERY_LIMIT = int(os.getenv("SEMANTIC_SCHOLAR_CITATION_QUERY_LIMIT", "8"))
SEMANTIC_SCHOLAR_ENABLE_CITATION_PASS = (
    os.getenv("SEMANTIC_SCHOLAR_ENABLE_CITATION_PASS", "True").lower() == "true"
)
SEMANTIC_SCHOLAR_CITATION_SORT = os.getenv(
    "SEMANTIC_SCHOLAR_CITATION_SORT",
    "citationCount:desc",
)
SEMANTIC_QUERY_STOPWORDS = {
    token.strip().lower()
    for token in os.getenv(
        "SEMANTIC_QUERY_STOPWORDS",
        "and,or,not,the,a,an,of,in,on,for,with,to,by,from,using,based,"
        "study,research,analysis,effect,impact,approach,method,methods",
    ).split(",")
    if token.strip()
}

ADAPTIVE_LEXICAL_EXPANSION_ENABLED = (
    os.getenv("ADAPTIVE_LEXICAL_EXPANSION_ENABLED", "True").lower() == "true"
)
ADAPTIVE_LEXICON_LLM_ENABLED = (
    os.getenv("ADAPTIVE_LEXICON_LLM_ENABLED", "True").lower() == "true"
)
ADAPTIVE_LEXICON_MAX_ABSTRACTS = int(os.getenv("ADAPTIVE_LEXICON_MAX_ABSTRACTS", "80"))
ADAPTIVE_LEXICON_MAX_TERMS = int(os.getenv("ADAPTIVE_LEXICON_MAX_TERMS", "160"))
ADAPTIVE_LEXICON_TERMS_PER_ATOM = int(os.getenv("ADAPTIVE_LEXICON_TERMS_PER_ATOM", "5"))
ADAPTIVE_LEXICON_MAX_QUERIES = int(os.getenv("ADAPTIVE_LEXICON_MAX_QUERIES", "12"))
ADAPTIVE_GRAPH_RECOVERY_ENABLED = (
    os.getenv("ADAPTIVE_GRAPH_RECOVERY_ENABLED", "True").lower() == "true"
)
ADAPTIVE_GRAPH_RECOVERY_MIN_CANDIDATES = int(
    os.getenv("ADAPTIVE_GRAPH_RECOVERY_MIN_CANDIDATES", "30")
)
ADAPTIVE_GRAPH_RECOVERY_MAX_SEEDS = int(os.getenv("ADAPTIVE_GRAPH_RECOVERY_MAX_SEEDS", "6"))
ADAPTIVE_GRAPH_RECOVERY_PER_SEED = int(os.getenv("ADAPTIVE_GRAPH_RECOVERY_PER_SEED", "12"))
ADAPTIVE_GRAPH_RECOVERY_MAX_ARTICLES = int(os.getenv("ADAPTIVE_GRAPH_RECOVERY_MAX_ARTICLES", "120"))
ADAPTIVE_GRAPH_RECOVERY_DIRECTIONS = [
    value.strip().lower()
    for value in os.getenv("ADAPTIVE_GRAPH_RECOVERY_DIRECTIONS", "references,citations").split(",")
    if value.strip()
]

BROAD_ENRICHMENT_STOPWORDS = {
    "analysis", "approach", "attention", "behavior", "difference", "differences",
    "development", "developmental", "effect", "effects", "evaluation", "framework",
    "impact", "memory", "method", "methods", "model", "models", "non", "outcome",
    "outcomes", "performance", "processing", "result", "results", "student",
    "students", "system", "systems", "technique", "techniques", "technologies",
    "technology", "tool", "tools", "environment", "environments", "setting",
    "settings", "context", "contexts",
}

BROAD_SYNONYM_TERMS = {
    term.strip().lower()
    for term in os.getenv("BROAD_SYNONYM_TERMS", "").split(",")
    if term.strip()
}

GENERIC_EQUIVALENCE_HEAD_TERMS = {
    "system", "systems", "technology", "technologies", "tool", "tools",
    "device", "devices", "method", "methods", "model", "models",
    "approach", "approaches", "program", "programs", "intervention",
    "interventions", "environment", "environments", "setting", "settings",
    "context", "contexts", "population", "populations", "participant",
    "participants", "subject", "subjects", "education", "learning",
}

TERM_VARIANT_STOPWORDS = {
    "and", "or", "of", "in", "on", "for", "with", "to", "the", "a", "an",
}

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

    if not GITHUB_GPT4O_TOKEN:
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
