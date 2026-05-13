"""
Script de Evaluación de Screening - Agnóstico
Mide Recall y Precisión basado en un Ground Truth definido por el investigador.
"""
import logging
import json
from typing import List, Dict

# Configurar logs básicos
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# GROUND TRUTH (Calibración del Investigador)
# 🚨 NOTA: Este objeto DEBE ser editado por el usuario para su RSL específica.
# Sirve como "termómetro" para medir si el motor está capturando lo que el experto sabe que es relevante.
GROUND_TRUTH = {
    # Papers que DEBEN aparecer (True Positives - Ejemplos que conoces de tu tema)
    "must_include": [
        "sifting the noise",           # 10/10
        "secureFalcon",                # 10/10  
        "gPTScan",                     # 9/10
        "new tricks to old codes",     # 10/10
        "can large language models find and fix",  # 10/10
        "analyzing source code vulnerabilities",   # 9/10
    ],
    # Términos o papers que NO deben aparecer (True Negatives - Ruido que quieres purgar)  
    "must_exclude": [
        "baseline defenses for adversarial attacks",
        "towards a robust detection of language model generated text",
        "three bricks to consolidate watermarks",
        "composite backdoor attacks",
        "fine-tuning llms for code mutation",
    ]
}

def evaluate_results(articles: List[Dict]) -> Dict:
    """
    Evalúa una lista de artículos contra el Ground Truth.
    """
    if not articles:
        logging.warning("⚠️ No hay artículos para evaluar.")
        return {'recall': 0, 'noise': 0, 'score': 0}

    titles_lower = [str(p.get('title', '')).lower() for p in articles]
    abstracts_lower = [str(p.get('abstract', '')).lower() for p in articles]
    combined_texts = [f"{t} {a}" for t, a in zip(titles_lower, abstracts_lower)]
    
    # 1. Medir RECALL (¿Cuántos obligatorios encontramos en el Top-100?)
    found_hits = []
    for must in GROUND_TRUTH['must_include']:
        if any(must.lower() in text for text in combined_texts):
            found_hits.append(must)
            
    # 2. Medir RUIDO (¿Cuántos excluidos se colaron en el Top-100?)
    noise_hits = []
    for excl in GROUND_TRUTH['must_exclude']:
        if any(excl.lower() in text for text in combined_texts):
            noise_hits.append(excl)

    # 3. Medir UTILIDAD TOP-20 (v8.0)
    top20_texts = combined_texts[:20]
    found_top20 = [must for must in GROUND_TRUTH['must_include'] 
                    if any(must.lower() in t for t in top20_texts)]
    noise_top20 = [excl for excl in GROUND_TRUTH['must_exclude'] 
                    if any(excl.lower() in t for t in top20_texts)]
            
    recall = len(found_hits) / len(GROUND_TRUTH['must_include'])
    noise_ratio = len(noise_hits) / len(GROUND_TRUTH['must_exclude'])
    
    # 4. Score Combinado (Balance de Calidad)
    # Penalizamos el ruido fuertemente para forzar precisión
    score = (recall * 100) - (noise_ratio * 100)
    
    logging.info("="*50)
    logging.info(f"📊 EVALUACIÓN DE CALIDAD SEMÁNTICA (v8.0 - Top-100 / Top-20)")
    logging.info("="*50)
    logging.info(f"✅ RECALL (Top-100): {len(found_hits)}/{len(GROUND_TRUTH['must_include'])} ({recall*100:.1f}%)")
    if len(found_hits) < len(GROUND_TRUTH['must_include']):
        missing = set(GROUND_TRUTH['must_include']) - set(found_hits)
        logging.info(f"   ❌ Faltantes: {list(missing)}")
        
    logging.info(f"❌ RUIDO (Top-100): {len(noise_hits)}/{len(GROUND_TRUTH['must_exclude'])} ({noise_ratio*100:.1f}%)")
    if noise_hits:
        logging.info(f"   ⚠️ Presentes: {noise_hits}")
    
    logging.info("-" * 30)
    logging.info(f"🏆 UTILIDAD TOP-20: {len(found_top20)} relevantes, {len(noise_top20)} ruido")
    logging.info(f"   ✨ Relevantes Top-20: {found_top20}")
        
    logging.info(f"🎯 SCORE FINAL: {score:.1f}%")
    logging.info("="*50)
    
    return {
        'recall': recall,
        'noise': noise_ratio,
        'score': score,
        'found': found_hits,
        'noise_present': noise_hits,
        'found_top20': found_top20
    }

if __name__ == "__main__":
    # Ejemplo de uso local con un CSV si se desea
    import pandas as pd
    import sys
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        df = pd.read_csv(path)
        articles = df.to_dict('records')
        evaluate_results(articles)
    else:
        print("Uso: python eval_screening.py <ruta_al_csv_de_resultados>")
