"""
Screening AI v7 - RESULTADOS PROFESIONALES
Genera resúmenes estructurados con datos específicos
"""
import requests
import config
import logging
from typing import Dict, List, Optional
import time
import json
import re
import os
import hashlib

# ==========================
# CACHE
# ==========================
CACHE_DIR = ".cache/ai_columns"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(title: str, column: str) -> str:
    key = f"{title}_{column}".encode('utf-8')
    return hashlib.md5(key).hexdigest()

def load_from_cache(title: str, column: str) -> str:
    cache_key = get_cache_key(title, column)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.txt")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            pass
    return None

def save_to_cache(title: str, column: str, value: str):
    cache_key = get_cache_key(title, column)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.txt")
    
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(value)
    except Exception as e:
        logging.warning(f"⚠️ Cache save error: {e}")


# ==========================
# EXTRACCIÓN INTELIGENTE (Tu código original - Rápido)
# ==========================

def extract_numbers_and_stats(text: str) -> List[str]:
    """
    Extrae estadísticas importantes del texto (Ej: "n=500", "87.3%", "p<0.05")
    """
    patterns = [
        r'n\s*=\s*\d+',  # n=500
        r'\d+\.?\d*\s*%',  # 87.3%
        r'p\s*[<>=]\s*0\.\d+',  # p<0.05
        r'\d+\s*(?:patients?|participants?|subjects?|cases)',  # 500 patients
        r'accuracy\s*[:=]?\s*\d+\.?\d*\s*%',  # accuracy: 95%
        r'sensitivity\s*[:=]?\s*\d+\.?\d*\s*%',
        r'specificity\s*[:=]?\s*\d+\.?\d*\s*%',
        r'AUC\s*[:=]?\s*0\.\d+',  # AUC=0.95
    ]
    
    stats = []
    text_lower = text.lower()
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        stats.extend(matches)
    
    seen = set()
    unique_stats = []
    for stat in stats:
        stat_clean = stat.strip()
        if stat_clean not in seen:
            seen.add(stat_clean)
            unique_stats.append(stat_clean)
    
    return unique_stats[:5]


def extract_key_sentences(text: str, keywords: List[str], max_sentences: int = 3) -> List[str]:
    """
    Extrae oraciones que contengan palabras clave específicas
    """
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30]
    
    ranked_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = sum(1 for kw in keywords if kw.lower() in sentence_lower)
        
        if score > 0:
            ranked_sentences.append((score, sentence))
    
    ranked_sentences.sort(key=lambda x: x[0], reverse=True)
    
    return [sent for score, sent in ranked_sentences[:max_sentences]]


def create_bullet_summary(sentences: List[str], stats: List[str] = None) -> str:
    """
    Crea un resumen estructurado con bullets
    """
    result = []
    
    if stats:
        result.append("📊 **Datos clave:**")
        for stat in stats[:3]:
            result.append(f"  • {stat}")
        result.append("")
    
    if sentences:
        for i, sent in enumerate(sentences[:3], 1):
            sent_clean = sent[:200].strip()
            if not sent_clean.endswith('.'):
                sent_clean += '...'
            result.append(f"{i}. {sent_clean}")
    
    return "\n".join(result) if result else "Información no disponible en abstract"


# ==========================
# GENERACIÓN PROFESIONAL POR COLUMNA (Tu código original)
# ==========================

def _generate_columns_for_article(article: Dict, columns: List[str]) -> Dict:
    """
    Genera columnas con resultados PROFESIONALES (usando Regex)
    """
    title = article.get('title', '')
    abstract = article.get('abstract', '') or ''
    
    if len(abstract) < 50:
        for col in columns:
            article[col] = "⚠️ Abstract insuficiente para análisis"
        return article
    
    for col in columns:
        cached = load_from_cache(title, col)
        if cached:
            article[col] = cached
            continue
        
        result = generate_professional_column(title, abstract, col)
        
        article[col] = result
        save_to_cache(title, col, result)
    
    return article


def generate_professional_column(title: str, abstract: str, column: str) -> str:
    """
    Genera contenido PROFESIONAL para cada columna (usando Regex)
    """
    text = f"{title}. {abstract}"
    
    if column == "summary":
        sentences = [s.strip() for s in abstract.split('.') if len(s.strip()) > 20]
        objective = ""
        for sent in sentences[:3]:
            if any(kw in sent.lower() for kw in ['aim', 'objective', 'purpose']):
                objective = sent
                break
        if not objective and sentences:
            objective = sentences[0]
        
        stats = extract_numbers_and_stats(abstract)
        return f"**Objetivo:** {objective[:150]}.\n\n**Datos:** {', '.join(stats[:3])}"

    elif column == "methodology":
        study_types = {
            'systematic review': '📚 Revisión Sistemática',
            'meta-analysis': '📊 Meta-análisis',
            'randomized controlled trial': '🎲 Ensayo Clínico Aleatorizado (RCT)',
            'cohort study': '👥 Estudio de Cohorte'
        }
        detected = "Tipo no especificado"
        for pattern, label in study_types.items():
            if pattern in text.lower():
                detected = label
                break
        return detected

    elif column == "population":
        numbers = re.findall(r'(\d+)\s*(patients?|participants?|subjects?|individuals?|cases)', text.lower())
        if numbers:
            return f"👥 **Muestra:** n = {numbers[0][0]}"
        return "⚠️ Población no especificada"

    elif column == "key_findings":
        stats = extract_numbers_and_stats(abstract)
        result_keywords = ['found', 'showed', 'demonstrated', 'accuracy', 'sensitivity', 'specificity', 'performance']
        result_sentences = extract_key_sentences(abstract, result_keywords, max_sentences=2)
        
        if stats:
            return f"📊 **Resultados:** {', '.join(stats[:4])}"
        if result_sentences:
            return f"🔬 **Hallazgo:** {result_sentences[0][:150]}."
        return "⚠️ Resultados no especificados"

    elif column == "limitations":
        lim_keywords = ['limitation', 'limitations', 'limited', 'however', 'drawback', 'bias']
        lim_sentences = extract_key_sentences(abstract, lim_keywords, max_sentences=2)
        if lim_sentences:
            return f"⚠️ **Limitación:** {lim_sentences[0][:150]}."
        return "ℹ️ Limitaciones no explícitas"

    elif column == "conclusions":
        sentences = [s.strip() for s in abstract.split('.') if len(s.strip()) > 30]
        if sentences:
            return f"✅ **Conclusión:** {sentences[-1][:150]}." # Última oración
        return "⚠️ Conclusiones no disponibles"
    
    return f"⚠️ Columna '{column}' no implementada"


# ==========================
# TRADUCCIÓN (CON AMBAS DIRECCIONES)
# ==========================

def _call_deepl(text: str, target_lang: str) -> Optional[str]:
    """Función helper para llamar a DeepL"""
    if not config.DEEPL_API_KEY:
        logging.warning("⚠️ DEEPL_API_KEY no configurada.")
        return None
        
    try:
        headers = {"Authorization": f"DeepL-Auth-Key {config.DEEPL_API_KEY}", "Content-Type": "application/json"}
        data = {"text": [text[:3000]], "target_lang": target_lang}
        
        response = requests.post(config.DEEPL_API_URL, headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            return response.json()["translations"][0]["text"]
        elif response.status_code == 456:
            logging.warning("⚠️ Cuota DeepL agotada.")
    except Exception as e:
        logging.warning(f"⚠️ DeepL error: {e}")
    return None

def _call_gpt_fallback(text: str, target_lang: str) -> Optional[str]:
    """Función helper para el fallback de traducción con GPT"""
    try:
        headers = {"Authorization": f"Bearer {config.OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        prompt = f"Translate the following text to {target_lang}:\n\n{text[:2000]}"
        data = {
            "model": config.OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1, "max_tokens": 1000
        }
        response = requests.post(f"{config.OPENROUTER_BASE_URL}/chat/completions", headers=headers, json=data, timeout=20)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.error(f"❌ Fallback de traducción (GPT) falló: {e}")
    return None

def translate_abstract_to_spanish(abstract: str) -> str:
    """Traducción (EN -> ES)"""
    if not abstract or len(abstract) < 10: return "Abstract no disponible"
    
    cached = load_from_cache(abstract[:100], "translation_es")
    if cached: return cached
    
    translation = _call_deepl(abstract, "ES") or _call_gpt_fallback(abstract, "Spanish")
    
    if translation:
        save_to_cache(abstract[:100], "translation_es", translation)
        return translation
    return "Error de traducción"

def translate_question_to_english(question: str) -> str:
    """NUEVA FUNCIÓN: Traducción (ES -> EN)"""
    if not question: return ""
    
    cached = load_from_cache(question, "translation_en")
    if cached: return cached

    # DeepL usa "EN-US" o "EN-GB"
    translation = _call_deepl(question, "EN-US") or _call_gpt_fallback(question, "English")
    
    if translation:
        save_to_cache(question, "translation_en", translation)
        return translation
    
    logging.error("❌ Fallo total de traducción a Inglés.")
    return question # Devuelve original como último recurso