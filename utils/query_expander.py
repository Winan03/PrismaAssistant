"""
Query Expander - VERSIÓN ESTRATÉGICA (MULTI-QUERY)
PROBLEMA ANTERIOR: Generaba palabras sueltas que creaban búsquedas demasiado amplias/genéricas.
SOLUCIÓN: Genera 3-4 Ecuaciones Booleanas completas (Queries) listas para usar.
"""
import requests
import config
import logging
import json
import re
import os
import hashlib
from typing import List, Optional

# Cache
CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(question: str) -> str:
    # v3: Invalida caches anteriores para forzar la nueva lógica booleana
    key = f"grok_v3_boolean_{question.strip().lower()}"
    return hashlib.md5(key.encode()).hexdigest()

def load_from_cache(question: str) -> Optional[List[str]]:
    cache_key = get_cache_key(question)
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)["terms"]
        except: pass
    return None

def save_to_cache(question: str, terms: List[str]):
    try:
        with open(os.path.join(CACHE_DIR, f"{get_cache_key(question)}.json"), 'w', encoding='utf-8') as f:
            json.dump({"question": question, "terms": terms}, f)
    except: pass

def expand_query_with_grok(question: str, max_terms: int = 5) -> List[str]:
    """
    Usa Grok-3 para generar ECUACIONES DE BÚSQUEDA (Boolean Queries) en lugar de palabras sueltas.
    """
    # 1. Cache
    cached = load_from_cache(question)
    if cached: 
        logging.info(f"🤖 Queries booleanos desde cache (v3)")
        return cached

    # 2. Prompt ESTRATÉGICO (Multi-Query Strategy)
    prompt = f"""Act as an expert systematic review librarian. Create 3 distinct and highly specific BOOLEAN SEARCH QUERIES for PubMed/Semantic Scholar based on this research question.

    Question: "{question}"

    STRATEGY:
    Query 1 (Specific Algorithms): Focus on specific AI methods (Deep Learning, CNN, XGBoost) AND the specific disease AND diagnosis.
    Query 2 (Biomarkers/Tools): Focus on AI AND diagnostic tools (ECG, MRI, Troponin) AND the condition.
    Query 3 (Effectiveness): Focus on performance metrics (AUC, Sensitivity) AND AI AND the condition.

    RULES:
    1. Output ONLY a valid JSON object with a key "queries" containing a list of strings.
    2. Use proper boolean operators (AND, OR). Use parentheses for grouping synonyms.
    3. DO NOT use terms that are too broad like just "AI" or just "Disease" without specific qualifiers.
    4. Example format: ["(Deep Learning OR CNN) AND (Atrial Fibrillation) AND (Early Detection)", "..."]

    JSON Output:"""

    try:
        headers = {
            "Authorization": f"Bearer {config.GITHUB_MODELS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": config.PROMPT_GENERATION_MODEL,
            "messages": [
                {"role": "system", "content": "You are a search query generator. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.4, # Balanceado para precisión sintáctica pero variedad semántica
            "max_tokens": 500,
            "response_format": { "type": "json_object" }
        }

        logging.info(f"🤖 Grok-3 generando estrategias de búsqueda...")
        
        response = requests.post(
            f"{config.GITHUB_MODELS_ENDPOINT}/chat/completions",
            headers=headers,
            json=data,
            timeout=25
        )

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            try:
                data_json = json.loads(content)
                queries = data_json.get("queries", [])
                
                # Limpieza básica de las queries
                clean_queries = []
                for q in queries:
                    # Asegurar que no sean demasiado largas o rotas
                    if q and len(q) > 10:
                        clean_queries.append(q)
                
                if clean_queries:
                    save_to_cache(question, clean_queries)
                    logging.info(f"   ✅ {len(clean_queries)} Estrategias generadas")
                    return clean_queries
                    
            except json.JSONDecodeError:
                logging.warning("⚠️ Error parseando JSON de Grok")

    except Exception as e:
        logging.error(f"❌ Error conexión Grok: {e}")

    # Fallback si falla la IA
    return generate_fallback_queries(question)

def generate_fallback_queries(question: str) -> List[str]:
    """
    Genera queries booleanos manualmente si la IA falla.
    """
    logging.info("🔧 Generando queries booleanos manuales (Fallback)...")
    q = question.lower()
    
    # Conceptos base
    ai_terms = '("artificial intelligence" OR "machine learning" OR "deep learning" OR "neural network")'
    
    # Detección básica de dominio
    cardio = '("cardiovascular" OR "heart" OR "cardiac")'
    if 'atrial' in q: cardio = '("atrial fibrillation" OR "arrhythmia")'
    elif 'failure' in q: cardio = '("heart failure")'
    
    goal = '("diagnosis" OR "screening" OR "prediction")'
    
    # Query 1: Estándar
    q1 = f"{ai_terms} AND {cardio} AND {goal}"
    
    # Query 2: Específico (si detecta términos)
    metrics = '("accuracy" OR "sensitivity" OR "AUC")'
    q2 = f"{ai_terms} AND {cardio} AND {metrics}"
    
    return [q1, q2]

def expand_query(question: str, max_terms: int = 12) -> List[str]:
    """Función principal que ahora devuelve Queries completos, no solo palabras."""
    return expand_query_with_grok(question)