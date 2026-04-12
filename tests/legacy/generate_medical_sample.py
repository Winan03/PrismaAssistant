import sys
import os
import logging

# Añadir el directorio actual al path para importar módulos
sys.path.append(os.getcwd())

from modules.synthesis import build_funnel_introduction

def test_medical_domain():
    print("--- INICIANDO PRUEBA DE DOMINIO: MEDICINA ---")
    
    question = "¿Cuál es la efectividad de las nuevas intervenciones terapéuticas basadas en IA para el control glucémico en pacientes con Diabetes Tipo 2?"
    
    # Artículos ficticios de medicina
    articles = [
        {
            'title': 'Inteligencia Artificial en el Manejo de la Diabetes',
            'authors': 'Gomez, A. y Perez, B.',
            'year': 2021,
            'source': 'PubMed',
            'abstract': 'Este estudio evaluó el uso de algoritmos de aprendizaje profundo para predecir crisis glucémicas. Se observó que la intervención terapéutica temprana mejora los resultados en pacientes con Diabetes Tipo 2.'
        },
        {
            'title': 'Análisis de Redes Neuronales en Endocrinología',
            'authors': 'Santos et al.',
            'year': 2023,
            'source': 'Scopus',
            'abstract': 'Santos et al. aplicaron redes neuronales convolucionales para el análisis de imágenes de retina en pacientes diabéticos. La precisión diagnóstica fue superior a los métodos tradicionales, evitando falsos negativos en etapas iniciales.'
        },
        {
            'title': 'Metaanálisis de Terapias Digitales',
            'authors': 'Rodriguez, C.',
            'year': 2024,
            'source': 'Web of Science',
            'abstract': 'Se realizó un metaanálisis de 20 ensayos clínicos controlados sobre el impacto de las apps en el control glucémico. Los pacientes mostraron una reducción significativa de la hemoglobina glicosilada.'
        }
    ]
    
    stats = {
        'total_found': 3,
        'selected': 3,
        'years': [2021, 2023, 2024],
        'databases': ['PubMed', 'Google Scholar']
    }
    
    try:
        intro = build_funnel_introduction(question, articles, stats)
        
        # Guardar resultado
        with open("sample_medical_intro.txt", "w", encoding="utf-8") as f:
            f.write(intro)
            
        print("\n✅ Introducción Médica Generada Correctamente.")
        print(f"Resultado guardado en: sample_medical_intro.txt")
        
        # Verificar términos del diccionario medicina.json
        if "sujetos de estudio" in intro.lower():
            print("✨ ÉXITO: Se detectó el dominio y se reemplazó 'pacientes' por 'sujetos de estudio'.")
        else:
            print("⚠️ AVISO: No se detectó el reemplazo de 'sujetos de estudio'.")
            
        if "falsos negativos" in intro.lower() and "las" not in intro:
            print("✨ ÉXITO: Concordancia de género aplicada a 'los falsos negativos'.")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_medical_domain()
