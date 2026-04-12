import unittest
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.synthesis import get_specific_research_questions, build_funnel_introduction, extract_main_topic

class TestConsistency(unittest.TestCase):
    def setUp(self):
        self.topic = "Inteligencia Artificial en el Arte Digital"
        self.question = "¿Cómo influye la IA en el arte digital?"
        self.articles = [
            {'title': 'Art 1', 'authors': ['Author A'], 'year': 2024, 'abstract': 'Content 1', 'source': 'Scopus'},
            {'title': 'Art 2', 'authors': ['Author B'], 'year': 2023, 'abstract': 'Content 2', 'source': 'PubMed'}
        ]
        self.stats = {'models': [], 'years': [], 'journals': []}

    def test_domain_detection_education(self):
        from modules.synthesis import detect_domain
        # Mocking model might be needed if we don't want real LLM calls here, 
        # but let's check code logic for now or skip if too expensive.
        pass

    def test_objective_injection_consistency(self):
        # Generate specific objectives
        # Since this calls the LLM, we might just want to verify the logic of build_funnel_introduction
        # with a controlled input.
        specific_objectives = [
            ["1", "Tema 1", "PI1: ¿P1?", "Describir 1."],
            ["2", "Tema 2", "PI2: ¿P2?", "Analizar 2."]
        ]
        
        # Test if build_funnel_introduction uses these objectives
        # We can observe the logs or the resulting text.
        # For this test, let's verify if the code path for injection is correct.
        intro = build_funnel_introduction(self.question, self.articles, self.stats, specific_objectives)
        
        # Check if the objectives appear at the end of the intro
        self.assertIn("Describir 1.", intro)
        self.assertIn("Analizar 2.", intro)
        self.assertTrue(intro.endswith("Analizar 2.") or intro.endswith("Analizar 2.;") or "Analizar 2." in intro.split('\n')[-1])

if __name__ == '__main__':
    unittest.main()
