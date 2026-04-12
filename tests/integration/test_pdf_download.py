import logging
import sys
import os

# Añadir el directorio actual al path para importar módulos locales
sys.path.append(os.path.abspath(os.curdir))

from modules.pdf_extractor import extract_text_from_pdf_url

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

urls_to_test = [
    "https://www.mdpi.com/1999-5903/15/10/326/pdf?version=1695912",
    "https://dl.acm.org/doi/pdf/10.1145/3597926.3605233",
    "https://dl.acm.org/doi/pdf/10.1145/3597503.3639202",
    "https://arxiv.org/pdf/2403.10646" # Este funcionaba, sirve de control
]

print("\n--- INICIO DE VERIFICACIÓN DE DESCARGAS PDF ---\n")

for url in urls_to_test:
    print(f"Probando: {url}")
    text = extract_text_from_pdf_url(url)
    if len(text) > 0:
        print(f"✅ ÉXITO: Se extrajeron {len(text)} caracteres.")
    else:
        print(f"❌ FALLO: No se pudo extraer texto (posible 403 o error de red).")
    print("-" * 50)

print("\n--- FIN DE VERIFICACIÓN ---\n")
