"""
Meta-Prompting con Grok-3
Genera prompts optimizados UNA VEZ por sesión
"""
import requests
import config
import logging
import json
import os
import hashlib
from typing import Dict, Optional

# Cache
PROMPT_CACHE_DIR = ".cache/meta_prompts"
os.makedirs(PROMPT_CACHE_DIR, exist_ok=True)

def get_prompt_cache_key(question: str, column: str) -> str:
    key = f"{question}_{column}".encode('utf-8')
    return hashlib.md5(key).hexdigest()

def load_prompt_from_cache(question: str, column: str) -> Optional[Dict]:
    cache_key = get_prompt_cache_key(question, column)
    cache_file = os.path.join(PROMPT_CACHE_DIR, f"{cache_key}.json")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return None

def save_prompt_to_cache(question: str, column: str, prompt_data: Dict):
    cache_key = get_prompt_cache_key(question, column)
    cache_file = os.path.join(PROMPT_CACHE_DIR, f"{cache_key}.json")
    
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(prompt_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"⚠️ Error guardando cache: {e}")

def generate_meta_prompt_with_grok(question: str, column: str) -> Dict:
    """
    Usa Grok-3 para generar estrategia de extracción
    
    Returns:
        {
            "keywords": ["keyword1", "keyword2"],
            "extraction_strategy": "Descripción de qué buscar",
            "output_format": "Template del resultado"
        }
    """
    # Cache
    cached = load_prompt_from_cache(question, column)
    if cached:
        logging.info(f"   ✅ Cache hit: '{column}'")
        return cached
    
    column_name = config.DYNAMIC_COLUMNS.get(column, column)
    
    meta_prompt = f"""You are an expert prompt engineer for academic paper analysis.

RESEARCH QUESTION:
"{question}"

TARGET COLUMN: {column_name}

YOUR TASK: Generate an extraction strategy for this column that will be applied to ALL research articles.

OUTPUT (JSON ONLY):
{{
  "keywords": ["List 5-8 specific keywords to search in abstracts"],
  "extraction_strategy": "Clear instructions (50-80 words) on what information to extract and how",
  "output_format": "Template showing how to structure the extracted information"
}}

CONTEXT:
- Processing medical/scientific articles
- Information extracted from titles + abstracts only
- Strategy will be applied to 50-500 articles
- Prioritize accuracy and consistency

Generate JSON now (no markdown):"""

    try:
        headers = {
            "Authorization": f"Bearer {config.GITHUB_MODELS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": config.GROK_MODEL,
            "messages": [
                {"role": "system", "content": "You output only valid JSON."},
                {"role": "user", "content": meta_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 600
        }
        
        response = requests.post(
            f"{config.GITHUB_MODELS_ENDPOINT}/chat/completions",
            headers=headers,
            json=payload,
            timeout=20
        )
        
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"].strip()
            
            # Limpiar JSON
            content = content.replace("```json", "").replace("```", "").strip()
            prompt_data = json.loads(content)
            
            # Validar
            required_keys = ["keywords", "extraction_strategy", "output_format"]
            if all(key in prompt_data for key in required_keys):
                save_prompt_to_cache(question, column, prompt_data)
                return prompt_data
        
        elif response.status_code == 429:
            logging.warning("⚠️ Rate limit Grok-3")
    
    except Exception as e:
        logging.error(f"❌ Error Grok-3: {e}")
    
    # Fallback
    return {
        "keywords": ["method", "study", "results"],
        "extraction_strategy": f"Extract {column_name} information from abstract",
        "output_format": "Brief summary"
    }

def generate_all_meta_prompts(question: str) -> Dict[str, Dict]:
    """
    Genera prompts para TODAS las columnas
    """
    logging.info("🤖 Generando meta-prompts con Grok-3...")
    
    meta_prompts = {}
    
    for column_key, column_name in config.DYNAMIC_COLUMNS.items():
        logging.info(f"   📝 {column_name}...")
        
        prompt_data = generate_meta_prompt_with_grok(question, column_key)
        meta_prompts[column_key] = prompt_data
        
        logging.info(f"      ✅ Keywords: {', '.join(prompt_data['keywords'][:3])}...")
    
    logging.info(f"✅ {len(meta_prompts)} meta-prompts generados")
    
    return meta_prompts