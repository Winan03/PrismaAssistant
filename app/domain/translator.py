import requests
import config
import logging
import json
import os
import hashlib

CACHE_FILE = ".cache/translations.json"
os.makedirs(".cache", exist_ok=True)

def _get_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def _save_cache(cache):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"❌ Error guardando caché de traducción: {e}")

def translate_to_english(text: str) -> str:
    """
    Traduce texto al inglés usando DeepL con caché local para ahorrar cuota.
    Si DeepL falla, devuelve el texto original para que el LLM intente traducirlo.
    """
    if not text or len(text.strip()) < 3:
        return text

    # 1. Verificar Caché
    cache = _get_cache()
    text_hash = hashlib.md5(text.strip().lower().encode()).hexdigest()
    
    if text_hash in cache:
        logging.info("♻️ Usando traducción guardada de DeepL (Caché)")
        return cache[text_hash]

    # 2. Intentar DeepL
    if config.DEEPL_API_KEY:
        try:
            logging.info(f"🌍 Traduciendo pregunta con DeepL...")
            headers = {
                "Authorization": f"DeepL-Auth-Key {config.DEEPL_API_KEY}"
            }
            params = {
                "text": text,
                "target_lang": "EN"
            }
            response = requests.post(config.DEEPL_API_URL, headers=headers, data=params, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                translated = result["translations"][0]["text"]
                
                # Guardar en caché
                cache[text_hash] = translated
                _save_cache(cache)
                
                logging.info("✅ Traducción de DeepL exitosa")
                return translated
            else:
                logging.warning(f"⚠️ DeepL devolvió error {response.status_code}: {response.text}")
        except Exception as e:
            logging.error(f"❌ Error conectando con DeepL: {e}")
    
    # 3. Fallback (devuelve el original para que el LLM se encargue)
    logging.warning("🔄 DeepL no disponible o falló. Usando texto original para procesamiento LLM.")
    return text
