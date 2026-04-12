
import sys
import os
from collections import Counter
from unittest.mock import MagicMock, patch

# MOCK EVERYTHING HEAVY
sys.modules['dotenv'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.metrics.pairwise'] = MagicMock()
sys.modules['sklearn.cluster'] = MagicMock()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.synthesis import extract_methods_info_corrected, generate_general_description_rag, generate_evidence_based_results

def test_methodology_math_fixed():
    print("Testing methodology math (Hierarchy fix with new labels)...")
    # Article 1: Experiment (Should be Estudios empíricos y experimentales)
    # Article 2: Systematic Review (Should be Artículos de revisión del estado del arte (surveys))
    articles = [
        {"title": "Deep Learning Experiment", "abstract": "We conducted an evaluation of the model architecture.", "year": 2024},
        {"title": "Systematic Review", "abstract": "A review of the literature.", "year": 2023},
    ]
    
    text = extract_methods_info_corrected(articles)
    print(f"Methodology text: {text}")
    
    # Check new labels
    assert "Artículos de revisión del estado del arte (surveys)" in text
    assert "Estudios empíricos y experimentales" in text
    
    # Check percentages (50/50)
    assert "50%" in text
    
    # Count how many percentages are in the text
    import re
    pcts = re.findall(r'(\d+)%', text)
    total_pct = sum(int(p) for p in pcts)
    print(f"Total percentage: {total_pct}%")
    assert total_pct == 100

def test_general_description_rag():
    print("\nTesting General Description RAG flow (Systemic APA)...")
    articles = [{"title": "Art 1", "year": 2024, "abstract": "Evidence about X", "author": "Bommareddy"}]
    stats = {"years": [{"label": "2024", "percentage": 100}], "methodology_types": [{"label": "Estudios empíricos y experimentales", "percentage": 100}]}
    question = "How does X work?"
    
    mock_model = MagicMock()
    # Mock response following APA rules
    mock_model.generate.return_value = "Como se observa en el análisis, Bommareddy (2024) sostiene que el mecanismo de X permite optimizar procesos científicos."
    
    with patch('modules.ai_model.LocalModel.get_instance', return_value=mock_model):
        desc = generate_general_description_rag(articles, question, stats)
        print(f"Generated Description: {desc}")
        
        # Verify it doesn't have the paper title if we mocked correctly
        assert "Bommareddy (2024)" in desc
        
        # Test full results generation
        full_results = generate_evidence_based_results(articles, question, stats)
        assert desc in full_results

if __name__ == "__main__":
    try:
        test_methodology_math_fixed()
        test_general_description_rag()
        print("\n✅ Verification successful: Systemic Dynamism is active!")
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
