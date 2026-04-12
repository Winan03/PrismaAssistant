
import os
import sys
import logging
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencias pesadas
sys.modules['dotenv'] = MagicMock()
sys.modules['requests'] = MagicMock()

# Añadir el path raíz para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from modules.ai_model import LocalModel, Provider

class TestModelRouting(unittest.TestCase):
    
    def setUp(self):
        # Resetear el Singleton para cada test
        LocalModel._instance = None
        self.model = LocalModel.get_instance()
        
        # Mocker la disponibilidad de providers para que no falle por falta de API keys
        self.model._is_provider_available = MagicMock(return_value=True)
        self.model._get_api_keys = MagicMock(return_value=["dummy_key"])

    def test_extraction_detection(self):
        """Verifica que el router detecte correctamente las tareas de extracción."""
        extraction_prompts = [
            "Extrae [Metodología]:",
            "Extract [Findings]:",
            "Identifica [Población]:",
            "Encuentra [Variables]:",
            "RESUMEN [Aportes]:",
            "Eres un extractor RSL de nivel Scopus Q1...",
            "Debes llenar UNA SOLA celda de una tabla..."
        ]
        non_extraction_prompts = [
            "Escribe una síntesis de los artículos.",
            "Genera un título para la RSL.",
            "Resume los siguientes párrafos.",
            "Traduce este texto al español."
        ]
        
        for p in extraction_prompts:
            self.assertTrue(self.model._is_extraction_task(p), f"Fallo al detectar extracción: {p}")
            
        for p in non_extraction_prompts:
            self.assertFalse(self.model._is_extraction_task(p), f"Falso positivo de extracción: {p}")

    @patch('modules.ai_model.RSLExtractor')
    @patch('modules.ai_model.LocalModel._call_provider')
    def test_routing_logic(self, mock_call_provider, mock_rslextractor_class):
        """Verifica que las llamadas se ruteen al modelo local o API según el caso."""
        
        # Mock del extractor local
        mock_extractor_instance = mock_rslextractor_class.return_value
        mock_extractor_instance.extract.return_value = "Resultado Local Qwen"
        
        # Mock de la API (Cerebras/GPT-OSS)
        mock_call_provider.return_value = ("Resultado API Cerebras", None)
        
        # TEST 1: Tarea de extracción -> Debería ir por el Local
        instr_ext = "Extrae [Métricas]:"
        res_ext = self.model.generate(instr_ext, "Texto de prueba")
        
        self.assertEqual(res_ext, "Resultado Local Qwen")
        mock_extractor_instance.extract.assert_called_once()
        mock_call_provider.assert_not_called()
        
        # TEST 2: Tarea de síntesis -> Debería ir por la API
        mock_extractor_instance.extract.reset_mock()
        mock_call_provider.reset_mock()
        
        instr_synth = "Genera una síntesis narrativa..."
        res_synth = self.model.generate(instr_synth, "Texto de prueba")
        
        self.assertEqual(res_synth, "Resultado API Cerebras")
        mock_extractor_instance.extract.assert_not_called()
        mock_call_provider.assert_called()

    @patch('modules.ai_model.RSLExtractor')
    @patch('modules.ai_model.LocalModel._call_provider')
    def test_fallback_on_local_error(self, mock_call_provider, mock_rslextractor_class):
        """Verifica que si el local falla, se intente la API como fallback."""
        
        # Mock de fallo en local
        mock_extractor_instance = mock_rslextractor_class.return_value
        mock_extractor_instance.extract.return_value = "[ERROR_LOCAL] CUDA out of memory"
        
        # Mock de éxito en API
        mock_call_provider.return_value = ("Resultado API Fallback", None)
        
        instr = "Extrae [Error Test]:"
        res = self.model.generate(instr, "Texto de prueba")
        
        self.assertEqual(res, "Resultado API Fallback")
        mock_extractor_instance.extract.assert_called_once()
        mock_call_provider.assert_called()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
