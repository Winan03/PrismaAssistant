
import os
import sys
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_medical_dynamism():
    print("\n--- INICIANDO PRUEBA DE UNIVERSALIDAD (DOMINIO: MEDICINA) ---")
    
    # Datos de un tema totalmente diferente: Alzheimer y IA
    topic = "Aplicación de Inteligencia Artificial para la detección temprana de Alzheimer mediante RM"
    question = "¿Qué algoritmos de Deep Learning presentan mayor precisión en la detección de biomarcadores de Alzheimer en resonancias magnéticas?"
    
    # Autores y hallazgos totalmente nuevos para forzar al sistema a ser dinámico
    articles = [
        {
            'authors': ['García, M.', 'López, R.'], 'year': 2023, 
            'title': 'Convolutional Neural Networks in Neuroimaging', 'source': 'PubMed',
            'abstract': 'Demostramos que las redes CNN de 3D logran un 92% de precisión detectando atrofia hipocampal.',
            'full_text_source': 'abstract'
        },
        {
            'authors': ['Chen, Y.', 'et al.'], 'year': 2024, 'source': 'Nature Medicine', 'database': 'Scopus',
            'title': 'Transformers for Early Dementia Detection', 
            'abstract': 'Los Vision Transformers (ViT) superan a las CNN tradicionales en la detección de cambios micro-estructurales.',
            'full_text_source': 'abstract'
        },
        {
            'authors': ['Smith, J.'], 'year': 2022, 'source': 'Web of Science',
            'title': 'Gated Recurrent Units in Longitudinal Analysis', 
            'abstract': 'El uso de redes GRU permite predecir la conversión de MCI a Alzheimer con 2 años de antelación.',
            'full_text_source': 'abstract'
        }
    ]
    
    try:
        from modules.synthesis import build_funnel_introduction
        
        # Simular estadísticas
        stats = {
            'years': [{'year': 2024, 'count': 5}, {'year': 2022, 'count': 3}],
            'journals': [{'source': 'Nature Medicine', 'count': 4}],
            'models': [{'model': 'ViT', 'count': 3}, {'model': 'CNN 3D', 'count': 2}],
            'metrics': [{'metric': 'Precisión', 'count': 6}]
        }
        
        print("🚀 Generando introducción médica dinámica...")
        intro = build_funnel_introduction(topic, articles, stats)
        
        print("\n=== INTRODUCCIÓN GENERADA (MEDICINA) ===\n")
        print(intro)
        
        # Validar lealtad (check manual en el output)
        print("\n--- VERIFICACIÓN DE ATRIBUCIÓN ---")
        if "García y López (2023)" in intro and ("CNN" in intro or "atrofia" in intro):
            print("✅ García (2023) correctamente vinculado a CNN/Atrofia.")
        if "Chen et al. (2024)" in intro and "Transformers" in intro:
            print("✅ Chen (2024) correctamente vinculado a Transformers.")
        if "Smith (2022)" in intro and ("GRU" in intro or "MCI" in intro):
            print("✅ Smith (2022) correctamente vinculado a GRU/MCI.")
            
    except Exception as e:
        print(f"❌ Error en la prueba de universalidad: {e}")

if __name__ == "__main__":
    # Asegurar que el path sea correcto para importar módulos
    sys.path.append(os.getcwd())
    test_medical_dynamism()
