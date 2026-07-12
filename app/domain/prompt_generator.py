"""
Generador de Reglas de Extracción (Meta-Prompting)
Usa Grok-3 para crear reglas dinámicas (Keywords + Regex)
"""
import requests
import config
import logging
import json
import re
from typing import Dict, List

class PromptGenerator:
    def __init__(self):
        self.model = config.PROMPT_GENERATION_MODEL
        self.api_key = config.GITHUB_MODELS_TOKEN
        self.base_url = config.GITHUB_MODELS_ENDPOINT
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        logging.info(f"✨ PromptGenerator (Cerebro) conectado a: {self.model}")

    def generate_extraction_rules(self, research_question: str, column_name: str) -> Dict:
        """
        Usa Grok-3 para crear reglas de extracción (JSON) basadas en la pregunta.
        """
        
        # Prompt de Ingeniería para Grok-3
        system_msg = """You are an Expert Data Engineer for Systematic Reviews.
        Your goal is to create EXTRACTION RULES for a Python script to parse scientific abstracts.
        
        Output MUST be a valid JSON with this exact structure:
        {
            "keywords_english": ["list", "of", "8", "specific", "english", "terms"],
            "regex_patterns": ["r'python_regex_1'", "r'python_regex_2'"],
            "extraction_logic": "Instructions in Spanish on how to format the output (e.g., 'Listar valores numéricos primero').",
            "fallback_text": "Text to show if nothing is found"
        }
        """

        # User Prompt: Contextualiza para la columna específica
        user_msg = f"""Research Question: "{research_question}"
        Target Column to Extract: "{column_name}"

        Generate the JSON rules to extract information for this column from an abstract.
        - If '{column_name}' is 'Población', regex should catch 'n=', 'participants', 'sample size'.
        - If '{column_name}' is 'Hallazgos Clave', regex should catch 'p<', '%', 'CI', 'HR', 'OR'.
        - Keywords must be in English (abstracts are in English).
        """

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    "temperature": 0.2, # Precisión
                    "max_tokens": 1000
                },
                timeout=30
            )
            
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                return self._clean_json(content)
            else:
                logging.error(f"❌ Error Grok API {response.status_code}: {response.text}")
                return None

        except Exception as e:
            logging.error(f"❌ Error conectando con Grok: {e}")
            return None

    def _clean_json(self, content: str) -> Dict:
        """Limpia el markdown del JSON que a veces devuelve la IA"""
        try:
            # Buscar contenido entre llaves {}
            match = re.search(r'\{.*\}', content, re.DOTALL)
            json_str = match.group(0) if match else content
            return json.loads(json_str)
        except Exception as e:
            logging.error(f"⚠️ No se pudo parsear el JSON de Grok: {e}")
            # Fallback de emergencia
            return {
                "keywords_english": [], 
                "regex_patterns": [], 
                "extraction_logic": "Error parsing JSON",
                "fallback_text": "No disponible"
            }

# Instancia global
prompt_generator = PromptGenerator()