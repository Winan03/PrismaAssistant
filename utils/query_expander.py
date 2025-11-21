"""
Query Expander - VERSIÓN ARREGLADA
PROBLEMA: Grok-3 devolvía la pregunta completa en vez de términos
SOLUCIÓN: Prompt más específico + fallback robusto
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
    key = f"grok_{question.strip().lower()}"
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


def expand_query_with_grok(question: str, max_terms: int = 10) -> List[str]:
    """
    Usa Grok-3 para generar términos académicos en Inglés
    """
    # 1. Cache
    cached = load_from_cache(question)
    if cached: 
        logging.info(f"🤖 Términos desde cache")
        return cached

    # 2. Prompt MEJORADO y más específico
    prompt = f"""Extract {max_terms} search keywords from this research question. 

Question: "{question}"

Rules:
1. Output ONLY keywords/phrases in English
2. One term per line
3. No numbers, no explanations
4. Focus on: main concepts, methods, domains
5. Each term should be 1-4 words

Example output format:
large language models
clinical diagnosis
diagnostic accuracy
artificial intelligence
medical decision support

Now extract keywords:"""

    try:
        headers = {
            "Authorization": f"Bearer {config.GITHUB_MODELS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": config.PROMPT_GENERATION_MODEL,
            "messages": [
                {"role": "system", "content": "You are a keyword extractor. Output only keywords, one per line."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,  # Más determinista
            "max_tokens": 200
        }

        logging.info(f"🤖 Grok-3 generando términos de búsqueda...")
        
        response = requests.post(
            f"{config.GITHUB_MODELS_ENDPOINT}/chat/completions",
            headers=headers,
            json=data,
            timeout=20
        )

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            
            # Limpieza: separar por líneas o comas
            terms = []
            for line in content.split('\n'):
                line = line.strip()
                # Saltar líneas vacías, números al inicio, etc
                if not line or line[0].isdigit() or len(line) < 4:
                    continue
                # Remover números al inicio (1. término)
                clean_line = re.sub(r'^\d+[\.\)]\s*', '', line)
                clean_line = clean_line.strip('"').strip("'").strip('-').strip()
                if clean_line and len(clean_line) > 3:
                    terms.append(clean_line.lower())
            
            # Filtrar duplicados y limitar
            final_terms = list(dict.fromkeys(terms))[:max_terms]
            
            if len(final_terms) >= 3:
                save_to_cache(question, final_terms)
                logging.info(f"   ✅ {len(final_terms)} términos generados")
                return final_terms
            else:
                logging.warning("⚠️ Grok devolvió pocos términos, usando fallback")

        elif response.status_code == 429:
            logging.warning("⚠️ Rate limit Grok-3, usando fallback")
        
        else:
            logging.error(f"❌ Grok-3 error {response.status_code}")

    except Exception as e:
        logging.error(f"❌ Error Grok Search: {e}")

    # ========================================
    # FALLBACK ROBUSTO: Extracción manual
    # ========================================
    return extract_terms_manually(question, max_terms)


def extract_terms_manually(question: str, max_terms: int = 10) -> List[str]:
    """
    Fallback MEJORADO: extracción con términos ancla obligatorios
    """
    logging.info("🔧 Usando extracción manual de términos...")
    
    question_lower = question.lower()
    
    # ========================================
    # CAMBIO CRÍTICO: Detectar términos ANCLA obligatorios
    # ========================================
    core_terms = []
    context_terms = []
    
    # 1. Anclas de LLM/IA (al menos UNO debe estar)
    llm_terms = []
    if any(kw in question_lower for kw in ['llm', 'language model', 'gpt', 'chatgpt', 'transformer']):
        llm_terms = ['large language models', 'LLM', 'ChatGPT', 'GPT-4']
    
    # 2. Anclas de dominio médico
    medical_terms = []
    if any(kw in question_lower for kw in ['clinical', 'diagnos', 'patient', 'medical', 'health']):
        medical_terms = ['clinical diagnosis', 'diagnostic accuracy', 'medical decision support']
    
    # 3. Términos de comparación (si aplica)
    comparison_terms = []
    if any(kw in question_lower for kw in ['traditional', 'comparison', 'vs', 'versus', 'compared']):
        comparison_terms = ['traditional methods', 'clinical decision support systems']
    
    # Combinar
    core_terms = llm_terms + medical_terms + comparison_terms
    
    # Si no detectamos nada, usar defaults genéricos
    if not core_terms:
        core_terms = [
            'artificial intelligence',
            'machine learning', 
            'clinical applications'
        ]
        logging.warning("⚠️ Sin términos detectados, usando defaults genéricos")
    
    # Limitar
    final_terms = list(dict.fromkeys(core_terms))[:max_terms]
    
    logging.info(f"   ✅ {len(final_terms)} términos extraídos")
    logging.info(f"   🎯 Anclas: {', '.join(final_terms[:3])}...")
    
    return final_terms


def expand_query(question: str, max_terms: int = 10) -> List[str]:
    """
    Función principal: intenta Grok-3, fallback a manual
    """
    terms = expand_query_with_grok(question, max_terms)
    
    # Validación: si devuelve la pregunta completa, es error
    if len(terms) == 1 and len(terms[0]) > 50:
        logging.warning("⚠️ Grok devolvió pregunta completa, usando fallback")
        terms = extract_terms_manually(question, max_terms)
    
    return terms