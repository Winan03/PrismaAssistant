
import sys
import os
import re
# Añadir el directorio raíz al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.synthesis import _smart_capitalize_title, generate_title_with_blocker, replace_anglicisms

def test_acronyms():
    print("Testing acronyms...")
    text = "la ia en el uso de sast y llms"
    result = _smart_capitalize_title(text)
    print(f"Original: {text}")
    print(f"Result: {result}")
    assert "IA" in result
    assert "SAST" in result
    assert "LLM" in result

def test_title_cleaning():
    print("\nTesting title cleaning...")
    test_title = "IA de los Modelos de Lenguaje Grande (LLMs) Frente a las Herramientas de Análisis Estático Tradicionales (SAST) en la Reducción de Falsos Positivos"
    
    # Logic from generate_title_with_blocker
    clean_title = re.sub(r'\b(IA|Inteligencia\s+Artificial)\s+de\s+los\s+(?=Modelo)', '', test_title, flags=re.IGNORECASE).strip()
    clean_title = clean_title[0].upper() + clean_title[1:] if clean_title else clean_title
    clean_title = re.sub(r'\s*\((LLMs?|SAST|DAST|IAST|IA|GPT)\)', '', clean_title, flags=re.IGNORECASE)
    clean_title = re.sub(r'\s+', ' ', clean_title).strip()
    
    print(f"Original Title: {test_title}")
    print(f"Cleaned Title: {clean_title}")
    
    assert "IA de los" not in clean_title
    assert "(LLMs)" not in clean_title
    assert "(SAST)" not in clean_title

def test_datoss_fix():
    print("\nTesting datoss fix...")
    text = "analizar el conjunto de datoss en la row 3"
    # Logic extracted from synthesis.py
    fixed = re.sub(r'\bdatoss\b', 'datos', text, flags=re.IGNORECASE)
    print(f"Original: {text}")
    print(f"Fixed: {fixed}")
    assert "datos" in fixed
    assert "datoss" not in fixed

def test_table2_filtering():
    print("\nTesting Table 2 filtering...")
    spanish_words = {
        'detección', 'vulnerabilidades', 'seguridad', 'código', 'fuente', 
        'revisión', 'sistemática', 'cuál', 'eficacia', 'frente', 'herramientas',
        'analizar', 'describir', 'identificar', 'según', 'literatura'
    }
    
    queries = [
        "LLM cibersecurity vulnerability detection", # Bien
        "[SEMANTIC] ¿Cuál es la eficacia de los Modelos...", # Mal
        "¿Cuáles son los datasets más usados?", # Mal
        "SAST tools effectiveness in code analysis", # Bien
    ]
    
    unique_eng_queries = []
    for q in queries:
        q_lower = q.lower()
        if any(word in q_lower for word in spanish_words): continue
        if q_lower.startswith('¿') or q_lower.startswith('cuál'): continue
        if '[semantic]' in q_lower or '[openalex]' in q_lower: continue
        unique_eng_queries.append(q)
    
    print(f"Original queries: {len(queries)}")
    print(f"Filtered queries: {len(unique_eng_queries)}")
    assert len(unique_eng_queries) == 2
    assert "LLM cibersecurity" in unique_eng_queries[0]
    assert "SAST tools" in unique_eng_queries[1]

def test_qa_criteria_redesign():
    print("\nTesting QA criteria redesign...")
    # Verificar que QA1 y QA2 ahora son sobre métricas y datasets
    from modules.synthesis import generate_dynamic_qa_criteria
    qa_list = generate_dynamic_qa_criteria("test question", [])
    
    qa_titles = [item[1].lower() for item in qa_list]
    print(f"QA Titles: {qa_titles}")
    
    assert not any("similitud semántica" in title for title in qa_titles)
    assert any("métricas" in title for title in qa_titles)
    assert any("dataset" in title for title in qa_titles)

def test_italics_anglicisms():
    print("\nTesting italics in anglicisms...")
    text = "Usamos un baseline y el state-of-the-art para el prompt engineering."
    result = replace_anglicisms(text)
    print(f"Original: {text}")
    print(f"Result: {result}")
    
    assert "*baseline*" in result
    assert "*state-of-the-art*" in result
    assert "*prompt*" in result

def test_structural_cleaning():
    print("\nTesting structural cleaning (Bug #7)...")
    from modules.synthesis import clean_generated_text
    
    texts = [
        "Elemento 7: Objetivos específicos. Analizar la literatura.",
        "7. Objetivo: Identificar brechas de conocimiento.",
        "Párrafo 1: Contexto general. La IA es útil.",
        "1. Interés: Relevancia global del tema. Los LLMs son potentes."
    ]
    
    for t in texts:
        result = clean_generated_text(t)
        print(f"Original: {t[:30]}...")
        print(f"Cleaned: {result[:30]}...")
        assert "Elemento 7" not in result
        assert "7. Objetivo" not in result
        assert "Párrafo 1" not in result
        assert "1. Interés" not in result

def test_unified_surnames():
    print("\nTesting unified surnames...")
    from modules.synthesis import AuthorGuardian, AuthorPurifier
    
    # Mocking metadata
    metadata = [{'authors': ['Kubiuk, I.'], 'year': 2021, 'title': 'Test'}]
    guardian = AuthorGuardian(metadata)
    
    # AuthorGuardian should use AuthorPurifier or at least the same logic
    surname = guardian._clean_surname("Kubiuk, I.")
    print(f"Surname: {surname}")
    assert surname == "Kubiuk"

if __name__ == "__main__":
    test_acronyms()
    test_title_cleaning()
    test_datoss_fix()
    test_table2_filtering()
    test_qa_criteria_redesign()
    test_italics_anglicisms()
    test_structural_cleaning()
    test_unified_surnames()
    print("\nAll tests passed successfully!")
