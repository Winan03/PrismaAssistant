import re

def test_cleanup():
    text = """
    I. INTRODUCCIÓN
    La seguridad del código fuente constituye un pilar crítico. Se analizan los pipelines de desarrollo continuo para detectar fallos. Zhu (2020) Evaluaron la homología de firmware (Zhu, 2020). Pooja et al. (2022) Propusieron una hoja de ruta. Akter (2022) contraste redes (Akter, 2022). Se detectó el F1score bajo. Aunque persiste una brecha. Esta carencia justifica este estudio. La metodología usa PRISMA. Los objetivos son: 1) analizar CI/CD.
    """
    
    # 1. CI/CD
    text = re.sub(r'pipelines?\s+de\s+(?:desarrollo\s+continuo|integración\s+y\s+desarrollo\s+continuos?|integracion\s+y\s+despacho\s+continuos?|entrega\s+continua)', 'pipelines de CI/CD', text, flags=re.IGNORECASE)
    print(f"Post CI/CD: {text}")
    
    # 2. Mayúsculas (GENERIC V6)
    # Ejemplo: Zhu (2020). Evaluaron -> Zhu (2020) evaluaron
    text = re.sub(r'(\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+\b(?:\s+et\s+al\.)?)\s+\((\d{4})\)\.\s+([A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+)', 
                  lambda m: f"{m.group(1)} ({m.group(2)}) {m.group(3)[0].lower()}{m.group(3)[1:]}", text)
    
    # También sin punto por si acaso
    text = re.sub(r'(\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+\b(?:\s+et\s+al\.)?)\s+\((\d{4})\)\s+([A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+)', 
                  lambda m: f"{m.group(1)} ({m.group(2)}) {m.group(3)[0].lower()}{m.group(3)[1:]}", text)
    print(f"Post Generic Capitalization: {text}")

if __name__ == "__main__":
    test_cleanup()
