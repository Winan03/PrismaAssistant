
import os
import logging
from dotenv import load_dotenv
from modules.synthesis import build_funnel_introduction

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Cargar variables de entorno
load_dotenv()

def generate_sample():
    topic = "Integración de IA (LLMs y SAST) para la detección de vulnerabilidades en el código fuente"
    
    articles = [
        {
            'authors': ['Kubiuk, I.', 'Kyselov, O.'], 'year': 2021, 
            'title': 'Code Representations via Deep Learning', 'source': 'IEEE Xplore',
            'abstract': 'Evaluamos representaciones de AST, PDG y redes BLSTM/BGRU para detección de vulnerabilidades.', # REAL GT
            'full_text': 'Nuestra investigación se centra en AST y PDG usando BLSTM. No usamos LLMs.',
            'full_text_source': 'pdf_download'
        },
        {
            'authors': ['Croft, R.', 'et al.'], 'year': 2021, 'source': 'Scopus',
            'title': 'Empirical Study of ML for Static Analysis', 
            'abstract': 'La integración de SAST con ML no genera sinergias evidentes. Identificamos el problema de las alertas falsas.', # REAL GT
            'full_text_source': 'abstract'
        },
        {
            'authors': ['Charmet, F.', 'et al.'], 'year': 2022, 'source': 'Web of Science',
            'title': 'Explainable AI for Alert Fatigue', 
            'abstract': 'Publicamos una revisión sobre IA Explicable (XAI) para mitigar la fatiga de alertas en seguridad.', # REAL GT
            'full_text_source': 'abstract'
        },
        {
            'authors': ['Akter, S.', 'et al.'], 'year': 2022, 'source': 'IEEE Xplore',
            'title': 'Quantum Learning in Supply Chain', 
            'abstract': 'Evaluamos el aprendizaje automático cuántico (QNN) en el conjunto de datos ClaMP.', # REAL GT
            'full_text_source': 'abstract'
        },
        {
            'authors': ['Harzevili, N.', 'et al.'], 'year': 2023, 'source': 'Scopus',
            'title': 'Systematic Survey of ML in Vulnerability Detection', 
            'abstract': 'Realizamos una encuesta sistemática que identificó brechas en los pipelines y falta de comparabilidad.', # REAL GT
            'full_text_source': 'abstract'
        },
        {
            'authors': ['Rantala, V.', 'et al.'], 'year': 2023, 'source': 'ScienceDirect',
            'title': 'Technical Debt and Static Tools', 
            'abstract': 'Analizamos la deuda técnica etiquetada (SATD) y su falta de coincidencia con herramientas estáticas.', # REAL GT
            'full_text_source': 'abstract'
        },
        {
            'authors': ['Bommareddy, S.'], 'year': 2024, 'source': 'Web of Science',
            'title': 'LLMs and NLP in SDLC', 
            'abstract': 'Investigamos la superioridad de los LLMs y el NLP en el ciclo SDLC para identificar vulnerabilidades CVE.', # REAL GT
            'full_text_source': 'abstract'
        },
        {
            'authors': ['Bersenev, A.', 'et al.'], 'year': 2024, 'source': 'IEEE Xplore',
            'title': 'Multi-agent Script Generation', 
            'abstract': 'Creamos un sistema multiagente que genera scripts a partir de lenguaje natural.', # REAL GT
            'full_text_source': 'abstract'
        },
        {
            'authors': ['Akhtar, N.'], 'year': 2025, 'source': 'arXiv',
            'title': 'PEFT strategies for LLMs', 
            'abstract': 'Evaluamos exhaustivamente técnicas de afinación eficiente (PEFT) como LoRA y QLoRA.', # REAL GT
            'full_text_source': 'abstract'
        },
        {
            'authors': ['Hyla, T.', 'Wawrzyniak, D.'], 'year': 2024, 'source': 'Scopus',
            'title': 'ASCVAS System for Fault Fix', 
            'abstract': 'Propusimos el sistema ASCVAS para localizar y reparar (fix) fallos automáticamente.', # REAL GT
            'full_text_source': 'abstract'
        }
    ]
    
    question = "¿En qué medida la integración de Modelos de Lenguaje Grande (LLMs) y herramientas de análisis estático (SAST) mejora la precisión y reduce los falsos positivos en la detección de vulnerabilidades del código fuente?"

    print("--- GENERANDO RESUMEN ESTRUCTURADO ---")
    try:
        from modules.synthesis import generate_complete_abstract, AcademicRefiner
        stats = {
            'years': [{'label': '2024', 'count': 5, 'percentage': 50.0}, {'label': '2022', 'count': 2, 'percentage': 20.0}, {'label': '2023', 'count': 1, 'percentage': 10.0}, {'label': '2021', 'count': 1, 'percentage': 10.0}, {'label': '2025', 'count': 1, 'percentage': 10.0}],
            'journals': [{'label': 'Future Internet', 'count': 2, 'percentage': 20.0}, {'label': 'International Journal for Multidisciplinary Research', 'count': 1, 'percentage': 10.0}],
            'models': [{'label': 'Familia BERT', 'count': 1, 'percentage': 10.0}, {'label': 'Otros', 'count': 9, 'percentage': 90.0}],
            'metrics': [{'label': 'Exactitud', 'count': 3, 'percentage': 30.0}, {'label': 'Precisión', 'count': 2, 'percentage': 20.0}, {'label': 'Exhaustividad', 'count': 2, 'percentage': 20.0}, {'label': 'Rendimiento', 'count': 2, 'percentage': 20.0}, {'label': 'F1Score', 'count': 1, 'percentage': 10.0}]
        }
        metrics_dict = {'final_included': 10}
        abstract = generate_complete_abstract(question, articles, stats, metrics_dict)
        
        # APLICAR REFINAMIENTO (Simulando generate_synthesis_full)
        refiner = AcademicRefiner()
        abstract = refiner.refine(abstract, context="abstract")
        
        print("\n=== RESUMEN REFINADO ===\n")
        print(abstract)
        
        # PROBAR TRADUCCIÓN (V12.8)
        from modules.synthesis import ImprovedTranslator
        print("\n--- TRADUCIENDO ABSTRACT ---")
        abstract_en = ImprovedTranslator.translate_abstract(abstract)
        abstract_en = refiner.refine(abstract_en, context="abstract")
        
        print("\n=== ABSTRACT EN INGLÉS ===\n")
        print(abstract_en)
        
        with open("sample_abstract_result.txt", "w", encoding="utf-8") as f:
            f.write(f"RESUMEN (ES):\n{abstract}\n\nABSTRACT (EN):\n{abstract_en}")
    except Exception as e:
        print(f"Error en resumen: {e}")

    print("\n--- GENERANDO INTRODUCCIÓN CITE-REFINED ---")
    try:
        from modules.synthesis import build_funnel_introduction
        intro = build_funnel_introduction(topic, articles, question)
        print("\n=== RESULTADO FINAL ===\n")
        print(intro)
        with open("sample_intro_result.txt", "w", encoding="utf-8") as f:
            f.write(intro)
    except Exception as e:
        print(f"Error en intro: {e}")

if __name__ == "__main__":
    generate_sample()
