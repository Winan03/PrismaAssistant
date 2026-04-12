import logging
import time
import os
import sys

# Añadir el directorio raíz al path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.ai_model import RSLExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_gguf_inference():
    print("\n" + "="*50)
    print("🧪 TEST DE INFERENCIA GGUF (QWEN 2.5 3B)")
    print("="*50)
    
    extractor = RSLExtractor()
    
    instruction = "Extrae [Metodología] del texto."
    input_text = """
    En este estudio, utilizamos un diseño experimental con 500 participantes. 
    Se aplicó un análisis estático de código usando la herramienta SonarQube v9.9.
    Los resultados muestran una reducción del 30% en falsos positivos.
    """
    
    print(f"\n📝 Prompt de prueba: {instruction}")
    print(f"📄 Texto de entrada: {input_text.strip()[:100]}...")
    
    start_t = time.time()
    try:
        response = extractor.extract(instruction, input_text)
        duration = time.time() - start_t
        
        print("\n" + "-"*30)
        print("🤖 RESPUESTA DE QWEN GGUF:")
        print(response)
        print("-"*30)
        print(f"⏱️ Tiempo total: {duration:.2f}s")
        
        if response and "[SIN INFORMACION]" not in response and "[ERROR" not in response:
            print("\n✅ TEST EXITOSO: El modelo respondió con contenido.")
        else:
            print("\n⚠️ ALERTA: La respuesta parece vacía o contiene error.")
            
    except Exception as e:
        print(f"\n❌ ERROR FATAL: {e}")

if __name__ == "__main__":
    test_gguf_inference()
