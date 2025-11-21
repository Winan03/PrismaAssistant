"""
Pre-Filtrado Inteligente con Grok-3 - VERSIÓN CORREGIDA
CAMBIOS CRÍTICOS:
1. Fallback más permisivo ante rate limit (0.75 en vez de 0.6)
2. Retry automático con backoff exponencial
3. Threshold ajustado a 0.50 (50%) para pre-filtro
"""
import requests
import config
import logging
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re

from modules.screening_ai import translate_question_to_english

def quick_relevance_check(article: Dict, question: str, retry_count: int = 0) -> float:
    """
    Usa Grok-3 para dar un score rápido (0-100) de relevancia.
    Recibe la pregunta (o términos) en INGLÉS para mejor matching.
    
    NUEVO: Retry automático con backoff exponencial
    """
    title = article.get('title', '')
    abstract = article.get('abstract', '')[:600]
    
    if len(abstract) < 50:
        return 0.0
    
    prompt = f"""Act as a systematic reviewer. Rate the relevance (0-100) of this article for the research topic:
    
    TOPIC: "{question}"
    
    ARTICLE:
    Title: {title}
    Abstract: {abstract}...
    
    CRITERIA:
    - >80: Directly addresses the topic.
    - 50-79: Related or potentially useful.
    - <50: Irrelevant or wrong domain.
    
    OUTPUT: Just the number (0-100)."""

    try:
        headers = {
            "Authorization": f"Bearer {config.GITHUB_MODELS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": config.PROMPT_GENERATION_MODEL,
            "messages": [
                {"role": "system", "content": "You are a strict relevance scorer. Output only a number."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 10
        }
        
        response = requests.post(
            f"{config.GITHUB_MODELS_ENDPOINT}/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"].strip()
            
            # Extraer número seguro
            numbers = re.findall(r'\d+', content)
            if numbers:
                score = float(numbers[0]) / 100.0
                return min(max(score, 0.0), 1.0)
        
        elif response.status_code == 429:
            # CAMBIO CRÍTICO: Retry con backoff exponencial
            if retry_count < 3:
                wait_time = 2 ** retry_count  # 1s, 2s, 4s
                logging.warning(f"⚠️ Rate Limit Grok-3. Retry {retry_count + 1}/3 en {wait_time}s...")
                time.sleep(wait_time)
                return quick_relevance_check(article, question, retry_count + 1)
            else:
                # CAMBIO CRÍTICO: Fallback más permisivo
                logging.warning("⚠️ Rate Limit Grok-3 persistente. Asumiendo POTENCIALMENTE relevante (0.75)")
                return 0.75  # ← ANTES ERA 0.6, AHORA 0.75
    
    except Exception as e:
        logging.warning(f"⚠️ Error Grok-3 quick check: {e}")
    
    return 0.65  # ← Fallback neutral más permisivo (antes 0.5)


def batch_filter_articles(articles: List[Dict], question: str, threshold: float = 0.50, max_workers: int = 2) -> List[Dict]:
    """
    Filtra artículos en paralelo con Grok-3.
    
    Args:
        articles: Lista de artículos
        question: Pregunta en CUALQUIER idioma (se traduce automáticamente)
        threshold: Umbral mínimo (0.50 es razonable para pre-filtro)
        max_workers: Hilos paralelos (reducir si hay rate limits)
    """
    if not articles:
        return []
    
    # ✅ NUEVO: Traducir automáticamente a inglés
    original_question = question
    question = translate_question_to_english(question)
    
    if question != original_question:
        logging.info(f"🌍 Pregunta traducida para Grok-3:")
        logging.info(f"   Original: {original_question}")
        logging.info(f"   Inglés: {question}")
    
    logging.info(f"🤖 Pre-filtrando {len(articles)} artículos con Grok-3 (Umbral: {int(threshold*100)}%)...")
    logging.info(f"   Contexto usado: {question[:100]}...")
    
    filtered = []
    processed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_article = {
            executor.submit(quick_relevance_check, article, question): article 
            for article in articles
        }
        
        for future in as_completed(future_to_article):
            article = future_to_article[future]
            try:
                score = future.result()
                article["grok_pre_score"] = score
                
                if score >= threshold:
                    filtered.append(article)
                else:
                    article["grok_exclusion_reason"] = f"Score {int(score*100)}% < {int(threshold*100)}%"
                
                processed += 1
                if processed % 20 == 0:
                    logging.info(f"   Progresando: {processed}/{len(articles)}...")
            
            except Exception as e:
                logging.error(f"   Error en thread: {e}")
                # Si falla completamente, incluir por seguridad
                article["grok_pre_score"] = 0.65
                filtered.append(article)
            
            # CAMBIO CRÍTICO: Pausa más larga entre requests
            time.sleep(0.8) 
    
    # Ordenar por score de Grok
    filtered.sort(key=lambda x: x.get("grok_pre_score", 0), reverse=True)
    
    excluded_count = len(articles) - len(filtered)
    logging.info(f"✅ Pre-filtrado completado: {len(articles)} → {len(filtered)} candidatos ({excluded_count} excluidos).")
    
    return filtered