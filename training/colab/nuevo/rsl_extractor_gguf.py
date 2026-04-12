"""
rsl_extractor_gguf.py — RSLExtractor con GGUF para CPU
=======================================================
Reemplaza la clase RSLExtractor en ai_model.py para usar
el archivo .gguf en lugar de cargar el modelo HuggingFace completo.

VENTAJAS vs float32 HuggingFace:
  - RAM: ~2.5GB vs ~12GB
  - Carga: ~5s vs ~590s
  - Sin GPU necesaria

INSTALACIÓN:
  pip install llama-cpp-python

  # Si querés aceleración con Metal (Mac) o CUDA (Linux GPU):
  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall

USO:
  Reemplazá la clase RSLExtractor en tu ai_model.py con esta.
  El resto del sistema no cambia — misma interfaz extract(instruction, input_text).
"""

import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Ruta al archivo GGUF — configurable por variable de entorno o config.py
# Opciones en orden de prioridad:
#   1. Variable de entorno RSL_GGUF_PATH
#   2. Carpeta 'models/' dentro del proyecto
#   3. Path absoluto hardcodeado como fallback

DEFAULT_GGUF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "qwen2.5-3b-instruct-q4_k_m.gguf"
)

GGUF_PATH = os.getenv("RSL_GGUF_PATH", DEFAULT_GGUF_PATH)


class RSLExtractor:
    """
    Extractor RSL usando llama-cpp-python con modelo GGUF.
    Drop-in replacement de la clase RSLExtractor original en ai_model.py.

    Carga el modelo una sola vez (singleton interno) y lo mantiene en memoria.
    RAM usada: ~2.5GB para Q4_K_M de Qwen 2.5 3B.
    """

    _llm = None
    _lock = threading.Lock()

    def __init__(self, gguf_path: str = GGUF_PATH):
        self.gguf_path = gguf_path
        self._validate_path()

    def _validate_path(self):
        """Verifica que el archivo GGUF exista antes de intentar cargarlo."""
        if not os.path.exists(self.gguf_path):
            raise FileNotFoundError(
                f"Archivo GGUF no encontrado: {self.gguf_path}\n"
                f"Opciones:\n"
                f"  1. Corré el notebook 'convertir_qwen_gguf.ipynb' en Colab\n"
                f"  2. Descargá el .gguf desde Drive y colocalo en: {os.path.dirname(self.gguf_path)}\n"
                f"  3. Configurá RSL_GGUF_PATH=/ruta/al/archivo.gguf en tu .env"
            )

    def _load_model(self):
        """Carga el modelo GGUF en memoria (singleton thread-safe, carga lazy)."""
        if RSLExtractor._llm is not None:
            return

        with RSLExtractor._lock:
            if RSLExtractor._llm is not None:
                return  # Doble check tras adquirir lock

            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python no está instalado.\n"
                    "Instalá con: pip install llama-cpp-python"
                )

            logger.info(f"📥 Cargando GGUF desde {self.gguf_path}...")
            t0 = time.time()

            RSLExtractor._llm = Llama(
                model_path=self.gguf_path,
                n_ctx=4096,        # Ventana de contexto
                n_threads=os.cpu_count() or 4,  # Usar todos los cores disponibles
                n_gpu_layers=0,    # 0 = solo CPU. Cambiá a -1 para usar GPU completa
                verbose=False,
            )

            elapsed = time.time() - t0
            logger.info(f"✅ GGUF cargado en {elapsed:.1f}s (RAM: ~2.5GB)")

    def extract(self, instruction: str, input_text: str, max_tokens: int = 512) -> str:
        """
        Ejecuta inferencia local con el modelo GGUF.
        Misma interfaz que el RSLExtractor original.

        Args:
            instruction : System prompt con la columna RSL a extraer
            input_text  : Contexto del artículo (chunks del PDF)
            max_tokens  : Máximo tokens a generar

        Returns:
            str: Síntesis extraída en español, o '[SIN INFORMACION]' si no aplica
        """
        self._load_model()

        # Formato ChatML que Qwen 2.5 espera
        prompt = (
            "<|im_start|>system\n"
            "Eres un extractor RSL de alta precisión. "
            "Extrae solo lo solicitado basándote exclusivamente en el texto proporcionado. "
            "Si no encuentras la información, responde [SIN INFORMACION].<|im_end|>\n"
            "<|im_start|>user\n"
            f"{instruction}\n\nTEXTO:\n{input_text[:3000]}<|im_end|>\n"  # 3000 chars = ~750 tokens
            "<|im_start|>assistant\n"
        )

        try:
            output = RSLExtractor._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.1,       # Casi determinístico para extracción
                repeat_penalty=1.1,    # Evita bucles de repetición
                stop=["<|im_end|>", "<|im_start|>"],  # Parar en tokens de control
                echo=False,
            )
            result = output["choices"][0]["text"].strip()
            return result if result else "[SIN INFORMACION]"

        except Exception as e:
            logger.error(f"❌ Error en inferencia GGUF: {e}")
            return f"[ERROR_LOCAL] {str(e)}"

    @staticmethod
    def is_loaded() -> bool:
        """Retorna True si el modelo ya está en memoria."""
        return RSLExtractor._llm is not None

    @staticmethod
    def unload():
        """Libera el modelo de memoria (útil para liberar RAM)."""
        with RSLExtractor._lock:
            RSLExtractor._llm = None
        logger.info("🗑️ Modelo GGUF descargado de memoria.")
