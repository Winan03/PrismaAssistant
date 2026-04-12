import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from modules.report_generator import create_prisma_diagram
from modules.synthesis import generate_complete_abstract, ImprovedTranslator

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_prisma_diagram():
    print("Testing PRISMA Diagram Generation...")
    metrics = {
        'total': 100,
        'after_filter': 80,
        'final_included': 20
    }
    output_dir = "tests_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    path = create_prisma_diagram(metrics, output_dir)
    if path and os.path.exists(path):
        print(f"✅ PRISMA diagram generated at: {path}")
    else:
        print("❌ PRISMA diagram generation failed")

def test_abstract_quantitative():
    print("\nTesting Quantitative Abstract...")
    question = "What is the impact of AI in Education?"
    articles = [{'year': 2023, 'source': 'Scopus'}, {'year': 2024, 'source': 'IEEE'}]
    stats = {
        'models': [{'label': 'Random Forest', 'percentage': 80}],
        'years': []
    }
    metrics = {'final_included': 50}
    
    abstract = generate_complete_abstract(question, articles, stats, metrics)
    print(f"Abstract preview: {abstract[:150]}...")
    
    if "Random Forest" in abstract and "80" in abstract: # 80% of 50 is 40
        print("✅ Abstract contains quantitative data (Random Forest)")
    elif "Random Forest" in abstract: 
         print(f"⚠️ Abstract mentions Random Forest but maybe calculation differs. Content: {abstract}")
    else:
        print(f"❌ Abstract missing quantitative data. Content: {abstract}")

def test_translation():
    print("\nTesting Title Translation...")
    title_es = "Inteligencia Artificial en la Educación: Una Revisión Sistemática de la Literatura"
    
    # This might fail if transformers not installed or model not downloaded, 
    # but we want to see if it runs without crashing.
    try:
        translated = ImprovedTranslator.translate_title(title_es)
        print(f"Original: {title_es}")
        print(f"Translated: {translated}")
        
        if "Artificial Intelligence" in translated or "Systematic Literature Review" in translated:
            print("✅ Translation seems to work (at least via fallback/map)")
        else:
            print("❌ Translation failed")
    except Exception as e:
        print(f"❌ Translation crashed: {e}")

if __name__ == "__main__":
    test_prisma_diagram()
    test_abstract_quantitative()
    test_translation()
