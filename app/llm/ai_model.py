# ai_model.py - Sistema Multi-Provider con Fallback Automático
"""
Sistema de generación de texto con fallback automático entre providers gratuitos:
1. Cerebras (Llama 3.3 70B) - 1M tokens/día gratis, ultra-rápido
2. Groq (Llama 3.3 70B) - Rápido, ~500K tokens/día gratis
3. GitHub Models (Grok-3-mini) - Generoso para usuarios GitHub
4. HuggingFace Router (Qwen 72B) - Último recurso

Nunca falla por "créditos agotados" - siempre hay un provider disponible.
"""

import logging
import requests
import os
import re
import time
import threading
from enum import Enum
from typing import Optional, Tuple, List, Dict
import config

# Helper para construir el contenido del usuario de forma segura (v16.0)
def _build_user_content(instruction, input_text):
    """Construye el mensaje del usuario evitando 'None' en el texto."""
    parts = []
    if instruction and instruction != "None":
        parts.append(instruction)
    if input_text and input_text != "None":
        parts.append(input_text)
    return "\n\n".join(parts) or "(sin instrucción)"


# ==============================================================================
# CONFIGURACIÓN DE PROVIDERS
# ==============================================================================

class Provider(Enum):
    DEEPSEEK = "deepseek"               # v19: DeepSeek V4 Flash — priority provider
    GEMINI = "gemini"
    GEMMA_27B = "gemma_27b"
    GEMMA_12B = "gemma_12b"
    CEREBRAS = "cerebras"
    GROQ = "groq"
    GITHUB_GPT4O = "github_gpt4o"
    GITHUB = "github"
    OPENROUTER = "openrouter"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"

# Configuración de cada provider
PROVIDERS_CONFIG = {
    Provider.DEEPSEEK: {
        "name": "DeepSeek (V4 Flash)",
        "endpoint": "https://api.deepseek.com/v1/chat/completions",
        "model": config.DEEPSEEK_MODEL,
        "api_key_getter": lambda: config.DEEPSEEK_API_KEY,
        "timeout": 120,
        "max_tokens_default": 8192,
        "max_input_chars": 60000,
    },
    Provider.GEMINI: {
        "name": "Google Gemini (2.5 Flash)",
        "endpoint": config.GEMINI_ENDPOINT,
        "model": config.GEMINI_MODEL,
        "api_key_getter": lambda: config.GEMINI_API_KEYS,
        "multi_key": True,
        "timeout": 120,
        "max_tokens_default": 8192,
        "max_input_chars": 800000,
    },
    Provider.GEMMA_27B: {
        "name": "Google Gemma 3 27B",
        # v17.4: Gemma no es compatible con endpoint /v1beta/openai/ — usar OpenRouter
        "endpoint": config.OPENROUTER_BASE_URL,
        "model": "google/gemma-3-27b-it:free",
        "api_key_getter": lambda: config.OPENROUTER_API_KEY,
        "timeout": 60,
        "max_tokens_default": 8192,
        "max_input_chars": 100000,
        "extra_headers": {
            "HTTP-Referer": "https://prisma-assistant.local",
            "X-Title": "PRISMA Assistant RSL",
        },
    },
    Provider.GEMMA_12B: {
        "name": "Google Gemma 3 12B",
        # v17.4: Gemma no es compatible con endpoint /v1beta/openai/ — usar OpenRouter
        "endpoint": config.OPENROUTER_BASE_URL,
        "model": "google/gemma-3-12b-it:free",
        "api_key_getter": lambda: config.OPENROUTER_API_KEY,
        "timeout": 60,
        "max_tokens_default": 8192,
        "max_input_chars": 100000,
        "extra_headers": {
            "HTTP-Referer": "https://prisma-assistant.local",
            "X-Title": "PRISMA Assistant RSL",
        },
    },
    Provider.CEREBRAS: {
        "name": "Cerebras Cloud",
        "endpoint": config.CEREBRAS_ENDPOINT,
        "model": config.CEREBRAS_MODEL,
        "api_key_getter": lambda: config.CEREBRAS_API_KEYS,
        "multi_key": True,
        "timeout": 120,
        "max_tokens_default": 8192,
        "max_input_chars": 16000,  # v16.3: Reducido — evita error 400 "9050 > 8192 tokens"
    },
    Provider.GROQ: {
        "name": "Groq Cloud",
        "endpoint": config.GROQ_ENDPOINT,
        "model": config.GROQ_MODEL,
        "api_key_getter": lambda: config.GROQ_API_KEY,
        "timeout": 120,
        "max_tokens_default": 8192,
        "max_input_chars": 24000, # v15.6: Ajustado para evitar error 8192 tokens
    },
    Provider.GITHUB_GPT4O: {
        "name": "GitHub Models (GPT-4o)",
        "endpoint": f"{config.GITHUB_MODELS_ENDPOINT}/chat/completions",
        "model": config.GITHUB_GPT4O_MODEL,
        "api_key_getter": lambda: config.GITHUB_GPT4O_TOKEN,
        "timeout": 180,
        "max_tokens_default": 8192,
        "max_input_chars": 120000,  # gpt-4o soporta 128k
    },
    Provider.GITHUB: {
        "name": "GitHub Models (GPT-4o-mini)",
        "endpoint": f"{config.GITHUB_MODELS_ENDPOINT}/chat/completions",
        "model": config.PROMPT_GENERATION_MODEL,
        "api_key_getter": lambda: config.GITHUB_GPT4O_TOKEN,
        "timeout": 180,
        "max_tokens_default": 8192,
        "max_input_chars": 80000,
    },
    Provider.OPENROUTER: {
        "name": "OpenRouter (GPT-OSS-120B Free)",
        "endpoint": config.OPENROUTER_BASE_URL,
        "model": config.OPENROUTER_MODEL,          # openai/gpt-oss-120b (PRIMARIO)
        "model_alt": config.OPENROUTER_MODEL_ALT,  # qwen/qwen3-next-80b:free (ALT1)
        "model_alt2": config.OPENROUTER_MODEL_ALT2, # arcee-ai/trinity-large-preview:free (ALT2)
        "api_key_getter": lambda: config.OPENROUTER_API_KEY,
        "timeout": 40,   # v11.25: 40s — si gpt-oss-120b no responde rápido, saltar provider
        "max_tokens_default": 4096,
        "max_input_chars": 120000, # v15.5: Aumentado
        "extra_headers": {
            "HTTP-Referer": "https://prisma-assistant.local",
            "X-Title": "PRISMA Assistant RSL",
        },
    },
    Provider.HUGGINGFACE: {
        "name": "HuggingFace Router",
        "endpoint": "https://router.huggingface.co/v1/chat/completions",
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "api_key_getter": lambda: config.HUGGINGFACE_API_KEY,
        "timeout": 180,
        "max_tokens_default": 8192,
        "max_input_chars": 80000,  # v15.5: Aumentado
    },
    Provider.OLLAMA: {
        "name": "Ollama Cloud / Local",
        "endpoint": config.OLLAMA_ENDPOINT,
        "model": config.OLLAMA_MODEL,
        "api_key_getter": lambda: config.OLLAMA_API_KEY if config.OLLAMA_API_KEY else ("ollama" if "localhost" in config.OLLAMA_ENDPOINT or "127.0.0.1" in config.OLLAMA_ENDPOINT else None),
        "timeout": 60,
        "max_tokens_default": 8192,
        "max_input_chars": 100000,
        "extra_headers": {
            "HTTP-Referer": "https://prisma-assistant.local",
            "X-Title": "PRISMA Assistant RSL",
        },
    },
}

# v19: DeepSeek V4 Flash como provider prioritario (más barato que GPT-4o, igual de preciso)
# Fallback: Ollama → Cerebras → Groq → Gemini → GitHub → OpenRouter → HuggingFace → Gemma
PROVIDER_ORDER = [Provider.DEEPSEEK, Provider.OLLAMA, Provider.CEREBRAS, Provider.GROQ, Provider.GEMINI, Provider.GITHUB_GPT4O, Provider.OPENROUTER, Provider.GITHUB, Provider.HUGGINGFACE, Provider.GEMMA_27B, Provider.GEMMA_12B]

_ensured_ollama_models = set()

def ensure_ollama_model(endpoint: str, model_name: str) -> bool:
    """
    Verifica si el modelo está disponible en Ollama.
    Si no lo está, intenta descargarlo (pull) llamando a /api/pull.
    """
    try:
        # Extraer URL base de Ollama (ej: http://localhost:11434)
        base_url = endpoint.replace("/v1/chat/completions", "").replace("/v1/embeddings", "").replace("/v1", "").rstrip("/")
        if not base_url.startswith("http"):
            base_url = "http://localhost:11434"

        # 1. Comprobar si el modelo está presente listando /api/tags
        tags_url = f"{base_url}/api/tags"
        resp = requests.get(tags_url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name") for m in data.get("models", [])]
            model_clean = model_name.strip()
            model_base = model_clean.split(":")[0]
            if any(model_clean == m or m.startswith(model_base) for m in models):
                logging.info(f"✅ [Ollama] El modelo '{model_name}' ya está disponible localmente.")
                return True

        # 2. Si no está en la lista, descargar el modelo automáticamente
        if not bool(getattr(config, "OLLAMA_AUTO_PULL", False)):
            logging.warning(
                "[Ollama] Modelo '%s' no detectado y OLLAMA_AUTO_PULL=False.",
                model_name,
            )
            return False

        pull_url = f"{base_url}/api/pull"
        logging.warning(f"📥 [Ollama] Modelo '{model_name}' no detectado. Iniciando descarga automática vía /api/pull...")

        pull_resp = requests.post(
            pull_url,
            json={"name": model_name, "stream": False},
            timeout=600  # 10 minutos de timeout para descargas lentas
        )
        if pull_resp.status_code == 200:
            logging.info(f"✅ [Ollama] Modelo '{model_name}' descargado y listo.")
            return True
        else:
            logging.error(f"❌ [Ollama] Error al descargar modelo '{model_name}': {pull_resp.status_code} - {pull_resp.text}")
    except Exception as e:
        logging.warning(f"⚠️ [Ollama] No se pudo conectar a Ollama en {endpoint} para verificar/descargar '{model_name}': {e}")
    return False


# ==============================================================================
# CLASE PRINCIPAL
# ==============================================================================

class LocalModel:
    """
    Generador de texto multi-provider con fallback automático.

    Si un provider falla (402, 429, timeout), automáticamente intenta el siguiente.
    Esto garantiza que siempre haya una respuesta disponible.
    """
    _instance = None
    _lock = threading.Lock()
    _providers_logged = False

    # v11.25: Semáforo global — máximo 5 llamadas API simultáneas
    # Cerebras es rápido y tiene 3 keys, puede soportar más concurrencia
    _api_semaphore = threading.Semaphore(5)

    # Tracking de providers con problemas temporales
    _provider_failures = {}  # {provider: timestamp_ultimo_fallo}
    _failure_cooldown = 60  # 1 minuto de cooldown tras fallo de créditos (402)
    _rate_limit_delay = 5  # Segundos a esperar tras rate limit (429)

    # v12.0: Singleton para el extractor local (Qwen 2.5 3B)
    _local_extractor = None
    # v12.2: Si el extractor local falla/timeout una vez, se deshabilita para toda la sesión
    # Esto evita perder 60s por artículo cuando la CPU/GPU no puede con la carga
    _local_extractor_disabled = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # double-checked locking
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._log_available_providers()

    def _log_available_providers(self):
        """Muestra qué providers están configurados."""
        if LocalModel._providers_logged:
            return
        LocalModel._providers_logged = True
        available = []
        for provider in PROVIDER_ORDER:
            cfg = PROVIDERS_CONFIG[provider]
            api_key = cfg["api_key_getter"]()
            if api_key:
                available.append(cfg["name"])
                logging.info(f"✅ Provider disponible: {cfg['name']}")
            else:
                logging.warning(f"⚠️ Provider sin API key: {cfg['name']}")

        if not available:
            logging.error("❌ NINGÚN PROVIDER CONFIGURADO - La generación fallará")
        else:
            logging.info(f"🔄 Orden de fallback: {' → '.join(available)}")

    def _is_provider_available(self, provider: Provider) -> bool:
        """Verifica si un provider está disponible (tiene API key y no está en cooldown)."""
        cfg = PROVIDERS_CONFIG[provider]
        api_key = cfg["api_key_getter"]()

        if not api_key:
            return False

        # Verificar cooldown por fallo reciente
        last_failure = self._provider_failures.get(provider)
        if last_failure:
            elapsed = time.time() - last_failure
            if elapsed < self._failure_cooldown:
                logging.debug(f"⏳ {cfg['name']} en cooldown ({int(self._failure_cooldown - elapsed)}s restantes)")
                return False
            else:
                # Cooldown expirado, limpiar
                del self._provider_failures[provider]

        return True

    def _mark_provider_failed(self, provider: Provider):
        """Marca un provider como fallido temporalmente."""
        self._provider_failures[provider] = time.time()
        cfg = PROVIDERS_CONFIG[provider]
        logging.warning(f"⚠️ {cfg['name']} marcado como fallido por {self._failure_cooldown}s")

    def _get_api_keys(self, provider: Provider) -> list:
        """Obtiene lista de keys para un provider."""
        cfg = PROVIDERS_CONFIG[provider]
        keys = cfg["api_key_getter"]()
        if isinstance(keys, str):
            return [keys]
        return keys if keys else []

    def _is_extraction_task(self, instruction: str) -> bool:
        """Determina si el prompt es una tarea de extracción RSL."""
        # v12.1: Patrón ampliado para capturar prompts que inician con descripción de rol (Scopus Q1)
        # y que mencionan llenar celdas de tabla o extraer columnas.
        patterns = [
            r"Eres un extractor RSL",
            r"llenar UNA SOLA celda",
            r"Extrae\s*\[",
            r"Extract\s*\[",
            r"Identifica\s*\[",
            r"Encuentra\s*\[",
            r"Resumen\s*\["
        ]
        return any(re.search(p, instruction, re.IGNORECASE) for p in patterns)

    def generate(self, instruction: str, input_text: str = "", max_tokens: int = 2048, system_prompt: str = None) -> str:
        """Genera texto con router inteligente: Cloud APIs primero, Local como último salvavidas."""

        # 1. Flujo normal multi-provider (Synthesis, Search, etc.)
        errors = []

        # v11.23: Throttling — esperar si hay más de 3 llamadas simultáneas
        with self._api_semaphore:
            for provider in PROVIDER_ORDER:
                if not self._is_provider_available(provider):
                    continue

                cfg = PROVIDERS_CONFIG[provider]
                keys = self._get_api_keys(provider)

                for i, key in enumerate(keys):
                    logging.info(f"🔄 Intentando con {cfg['name']} (Key {i+1}/{len(keys)})...")

                    limit = cfg.get("max_input_chars", 30000)
                    truncated_input = input_text
                    if input_text and len(input_text) > limit:
                        logging.info(f"✂️ Truncando entrada de {len(input_text)} a {limit} chars para {cfg['name']}")
                        truncated_input = input_text[:limit] + "\n[Texto truncado por límite de ventana...]"

                    # v11.24: Para OpenRouter, rotar internamente entre model, model_alt, model_alt2
                    models_to_try = [cfg["model"]]
                    if cfg.get("model_alt"):
                        models_to_try.append(cfg["model_alt"])
                    if cfg.get("model_alt2"):
                        models_to_try.append(cfg["model_alt2"])
                                   # v15.9: Reintento con Exponential Backoff para Rate Limits (429)
                    last_error = None
                    for model_name in models_to_try:
                        max_retries = 3
                        for attempt in range(max_retries):
                            result, error = self._call_provider(
                                provider, instruction, truncated_input, max_tokens,
                                api_key=key, override_model=model_name,
                                system_prompt=system_prompt
                            )
                            if result:
                                logging.info(f"✅ {cfg['name']} ({model_name.split('/')[-1]}) respondió correctamente ({len(result)} chars)")
                                return result

                            last_error = error
                            # Verifica si fue un Rate Limit o falla de cuota temporal
                            if error and ("429" in error or "rate limit" in error or "too many requests" in error.lower()):
                                if attempt < max_retries - 1:
                                    sleep_seconds = (attempt + 1) * 7  # 7s, 14s...
                                    logging.info(f"⏳ Rate limit en {model_name.split('/')[-1]}. Esperando {sleep_seconds}s para reintentar (Intento {attempt+2}/{max_retries})...")
                                    time.sleep(sleep_seconds)
                                    continue
                                else:
                                    logging.warning(f"⚠️ {model_name.split('/')[-1]} falló repetidamente por Rate Limit. Rotando modelo...")
                                    break
                            else:
                                # Falló por un error distinto a 429 (ej: 400, 401), no reintentar este modelo
                                break

                    error = last_error
                    if error:
                        errors.append(f"{cfg['name']}_K{i+1}: {error}")
                        logging.warning(f"⚠️ {cfg['name']} (Key {i+1}) falló finalmente: {error[:150]}...")

                        # v17.0: Si el error es de contexto demasiado largo (400/413),
                        # saltar TODAS las demás keys de este proveedor — no van a funcionar
                        error_lower = error.lower()
                        is_context_error = (
                            ("length" in error_lower and ("limit" in error_lower or "reduce" in error_lower)) or
                            "too large" in error_lower or
                            "413" in error or
                            "request too large" in error_lower or
                            "maximum context" in error_lower
                        )
                        if is_context_error:
                            logging.warning(f"⏭️ {cfg['name']}: Error de contexto largo. Saltando a siguiente proveedor...")
                            break  # Salir del bucle de keys, ir al siguiente proveedor

                        if "402" in error or "credit" in error or "quota" in error:
                            if i == len(keys) - 1:
                                self._mark_provider_failed(provider)

            error_summary = "; ".join(errors) if errors else "Ningún provider disponible"
            logging.error(f"❌ Todas las APIs cloud fallaron: {error_summary}")

        # 2. Fallback de Emergencia: Si todo falló, usar modelo local
        if config.ENABLE_LOCAL_EXTRACTOR and self._is_extraction_task(instruction):
            if self.__class__._local_extractor_disabled:
                logging.debug("⏭️ Extractor local deshabilitado. No disponible como fallback.")
            else:
                logging.warning("🚨 Activando EXTRACTOR LOCAL (Qwen 2.5 3B) como último recurso...")
                try:
                    if self._local_extractor is None:
                        self._local_extractor = RSLExtractor()

                    local_max_tokens = min(max_tokens, 1024)
                    result = self._local_extractor.extract(instruction, input_text, max_tokens=local_max_tokens)
                    if result and "[ERROR_LOCAL]" not in result:
                        return result

                    logging.warning("⚠️ Extractor local de rescate falló. Deshabilitando para la sesión.")
                    self.__class__._local_extractor_disabled = True
                except Exception as e:
                    logging.error(f"❌ Error en extractor local de rescate: {e}")
                    self.__class__._local_extractor_disabled = True

        return f"⚠️ Error de generación: {error_summary}"

    def generate_ollama_model(
        self,
        instruction: str,
        model_name: str,
        input_text: str = "",
        max_tokens: int = 2048,
        system_prompt: str = None,
    ) -> str:
        """Run a specific Ollama model, then fall back to the regular router."""
        model_name = str(model_name or "").strip()
        if not model_name:
            return self.generate(instruction, input_text, max_tokens, system_prompt)

        errors = []
        with self._api_semaphore:
            keys = self._get_api_keys(Provider.OLLAMA)
            fallback_models = [
                str(model).strip()
                for model in getattr(config, "OLLAMA_ROLE_FALLBACK_MODELS", [])
                if str(model).strip() and str(model).strip() != model_name
            ]
            role_models = [model_name, *fallback_models]
            seen_models = set()

            for role_model in role_models:
                if role_model in seen_models:
                    continue
                seen_models.add(role_model)
                for i, key in enumerate(keys):
                    logging.info(
                        "Intentando Ollama role-model %s (Key %d/%d)...",
                        role_model,
                        i + 1,
                        len(keys),
                    )
                    result, error = self._call_provider(
                        Provider.OLLAMA,
                        instruction,
                        input_text,
                        max_tokens,
                        api_key=key,
                        override_model=role_model,
                        system_prompt=system_prompt,
                    )
                    if result:
                        logging.info(
                            "Ollama role-model %s respondio correctamente (%d chars)",
                            role_model,
                            len(result),
                        )
                        return result
                    if error:
                        errors.append(f"{role_model}: {error}")

        logging.warning(
            "Ollama role-model %s no disponible: %s. Usando router general.",
            model_name,
            "; ".join(errors)[:250] if errors else "sin keys",
        )
        return self.generate(instruction, input_text, max_tokens, system_prompt)

    def _call_provider(self, provider: Provider, instruction: str, input_text: str, max_tokens: int, api_key: str = None, override_model: str = None, system_prompt: str = None) -> Tuple[Optional[str], Optional[str]]:
        """Llama a un provider específico con una clave dada. override_model permite probar modelos alt."""
        cfg = PROVIDERS_CONFIG[provider]
        if not api_key:
            api_key = cfg["api_key_getter"]()
            if isinstance(api_key, list): api_key = api_key[0]

        if not api_key:
            return None, "API key no configurada"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # v11.23: Soporte para headers adicionales por provider (ej: OpenRouter requiere HTTP-Referer)
        extra_headers = cfg.get("extra_headers", {})
        headers.update(extra_headers)

        selected_model = override_model if override_model else cfg["model"]
        if provider == Provider.OLLAMA:
            if selected_model not in _ensured_ollama_models:
                if ensure_ollama_model(cfg["endpoint"], selected_model):
                    _ensured_ollama_models.add(selected_model)
        is_gemma = "gemma" in selected_model.lower()  # v18.0: fix OpenRouter usa google/gemma-* format

        default_system = "Eres un investigador científico senior especializado en revisiones sistemáticas (RSL). Tu mandato absoluto es responder SIEMPRE en ESPAÑOL, incluso si el material de origen o los fragmentos de texto están en inglés u otros idiomas. Tu salida debe ser profesional, académica y 100% en español."
        effective_system = system_prompt if system_prompt else default_system
        user_content = _build_user_content(instruction, input_text)

        if is_gemma:
            # Gemma: System prompt va al inicio del primer mensaje de usuario
            messages = [{"role": "user", "content": f"{effective_system}\n\n{user_content}"}]
        else:
            messages = [
                {"role": "system", "content": effective_system},
                {"role": "user", "content": user_content}
            ]

        payload = {
            "model": selected_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "top_p": 0.9,
        }

        if provider == Provider.DEEPSEEK:
            # deepseek-v4-flash en modo normal (sin reasoning_effort)
            # es suficientemente inteligente y mucho más rápido para screening.
            # Al ser un modelo de razonamiento, consume tokens en la fase de pensamiento (reasoning_content).
            # Si el límite es muy bajo, se corta antes de generar el contenido.
            if payload["max_tokens"] < 2048:
                payload["max_tokens"] = 2048

        try:
            response = requests.post(
                cfg["endpoint"],
                headers=headers,
                json=payload,
                timeout=cfg["timeout"]
            )

            if response.status_code == 200:
                result = response.json()

                # Extraer contenido (formato OpenAI)
                if 'choices' in result and len(result['choices']) > 0:
                    choice = result['choices'][0]
                    message = choice.get('message', {})
                    finish_reason = choice.get('finish_reason', 'unknown')
                    text = message.get('content', '').strip()
                    # v16.2: DEBUG - ver finish_reason para detectar truncación
                    if finish_reason in ('length', 'max_tokens'):
                        logging.warning(
                            f"⚠️ [LLM:{selected_model}] finish_reason={finish_reason} "
                            f"con {len(text)} chars — max_tokens demasiado bajo?"
                        )
                    if text:
                        text = re.sub(r'(\(\w+.*?\d{4}\))\.\.',r'\1.', text)    # Evitar doble punto
                        text = re.sub(r'\bya que\b', 'debido a que', text, flags=re.IGNORECASE)
                        text = re.sub(r'\bfalos\b', 'fallos', text)
                        return text, None

                return None, "Respuesta vacía del modelo"

            # Errores conocidos
            elif response.status_code == 401:
                return None, "401 - API key inválida"

            elif response.status_code == 402:
                return None, "402 - Créditos agotados"

            elif response.status_code == 429:
                return None, "429 - Rate limit excedido"

            elif response.status_code == 503:
                return None, "503 - Modelo cargándose"

            else:
                error_text = response.text[:200] if response.text else "Sin detalles"
                return None, f"{response.status_code} - {error_text}"

        except requests.exceptions.Timeout:
            return None, f"Timeout ({cfg['timeout']}s)"

        except requests.exceptions.ConnectionError:
            return None, "Error de conexión"

        except Exception as e:
            return None, f"Exception: {str(e)[:100]}"


# ==============================================================================
# v12.0: EXTRACTOR LOCAL (QWEN 2.5 3B)
# ==============================================================================

class RSLExtractor:
    """
    Inferencia local usando Qwen 2.5 3B en formato GGUF (vía llama-cpp-python).
    v12.2: Auto-detecta GPU (CUDA) vs CPU. Funciona en local (GTX 1650) y en servidores sin GPU.
    """
    def __init__(self):
        self.model_path = config.LOCAL_EXTRACTOR_PATH
        self.llm = None
        logging.info(f"🧠 Inicializando RSLExtractor GGUF ({self.model_path})...")

    @staticmethod
    def _detect_gpu_layers() -> int:
        """v12.2: Auto-detecta si hay GPU CUDA disponible. Retorna -1 (GPU total) o 0 (CPU)."""
        try:
            from llama_cpp import llama_supports_gpu_offload
            if llama_supports_gpu_offload():
                logging.info("🎮 GPU CUDA detectada → n_gpu_layers=-1 (todas las capas en GPU)")
                return -1  # Todas las capas en GPU — máxima velocidad
            else:
                logging.info("💻 Sin GPU CUDA → n_gpu_layers=0 (inferencia en CPU)")
                return 0
        except Exception:
            return 0  # Fallback seguro a CPU si hay cualquier error

    def _load_model(self):
        """Carga perezosa del modelo GGUF con auto-detección GPU/CPU."""
        if self.llm is not None:
            return

        try:
            from llama_cpp import Llama

            n_gpu_layers = self._detect_gpu_layers()
            backend = "GPU (CUDA)" if n_gpu_layers != 0 else "CPU"
            logging.info(f"📥 Cargando Qwen 2.5 3B GGUF en {backend} desde {self.model_path}...")
            start_t = time.time()

            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=8192,              # v15.6: Aumentado para mayor visibilidad local
                n_threads=os.cpu_count() or 4,
                n_gpu_layers=n_gpu_layers,  # -1=GPU total, 0=CPU pura (auto-detectado)
                verbose=False
            )

            logging.info(f"✅ Qwen 2.5 3B GGUF cargado en {backend} en {time.time() - start_t:.1f}s")
        except Exception as e:
            logging.error(f"❌ Fallo cargando modelo GGUF: {e}. Asegúrate de instalar 'llama-cpp-python'.")
            raise

    def extract(self, instruction: str, input_text: str, max_tokens: int = 1024) -> str:
        """Ejecuta inferencia local con GGUF. v12.3: timeout de 60s y mayor límite de tokens."""
        import concurrent.futures
        try:
            self._load_model()

            # Formatear prompt estilo ChatML (Qwen 2.5)
            # v12.5: Truncar input_text a 18000 chars para mayor capacidad local
            input_truncated = input_text[:18000] if input_text and len(input_text) > 18000 else input_text
            full_prompt = (
                "<|im_start|>system\n"
                "Eres un extractor RSL de alta precisión. Extrae solo lo solicitado basándote exclusivamente en el texto proporcionado. "
                "Si no encuentras la información, responde [SIN INFORMACION].<|im_end|>\n"
                f"<|im_start|>user\n{instruction}\n\nTEXTO:\n{input_truncated}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            logging.info(f"⚡ Inferencia GGUF en curso (max_tokens={max_tokens}, timeout=120s)...")

            # v12.1: Envolver en ThreadPoolExecutor con timeout de 60s
            # Si supera el límite, retorna [ERROR_LOCAL] y el sistema cae al fallback API
            def _run_inference():
                return self.llm(
                    full_prompt,
                    max_tokens=max_tokens,
                    temperature=0.1,
                    top_p=0.9,
                    stop=["<|im_end|>", "<|im_start|>"],
                    echo=False
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_inference)
                try:
                    output = future.result(timeout=120)  # 120 segundos máximo (v12.4)
                except concurrent.futures.TimeoutError:
                    logging.warning("⏱️ Extractor local superó 120s de timeout. Activando fallback API...")
                    return "[ERROR_LOCAL]"

            response = output["choices"][0]["text"].strip()
            logging.info(f"✅ Extractor local completó en tiempo ({len(response)} chars)")
            return response

        except Exception as e:
            logging.error(f"⚠️ Error en inferencia GGUF: {e}")
            return f"[ERROR_LOCAL_GGUF] {str(e)}"


# ==============================================================================
# FUNCIONES DE COMPATIBILIDAD
# ==============================================================================

def init_model():
    """Inicializa el modelo (compatibilidad con código existente)."""
    return LocalModel.get_instance()


def generate_text(instruction: str, input_text: str = "", max_tokens: int = 2048, system_prompt: str = None) -> str:
    """Función helper para generación rápida."""
    model = LocalModel.get_instance()
    return model.generate(instruction, input_text, max_tokens, system_prompt)


def generate_text_with_ollama_model(
    instruction: str,
    model_name: str,
    input_text: str = "",
    max_tokens: int = 2048,
    system_prompt: str = None,
) -> str:
    """Helper for tasks that need a specific Ollama role model."""
    model = LocalModel.get_instance()
    return model.generate_ollama_model(
        instruction=instruction,
        model_name=model_name,
        input_text=input_text,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )
