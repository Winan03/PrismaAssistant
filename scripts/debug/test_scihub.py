"""Test rápido de _level_scihub con un DOI del Gold Standard."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurar encoding para Windows
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import logging
logging.basicConfig(level=logging.DEBUG, format="%(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

# Suppress ultra-verbose logs from third party libraries
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

from utils.pdf_extractor import _level_scihub, _extract_text

# DOI de un artículo del gold standard (IEEE Access)
test_doi = "10.1109/ACCESS.2021.3095559"
print(f"Testing Sci-Hub for DOI: {test_doi}")
print("=" * 60)

pdf_bytes = _level_scihub(test_doi)

if pdf_bytes:
    print(f"\n*** SUCCESS! PDF downloaded: {len(pdf_bytes)} bytes ***")
    # Intentar extraer texto
    text = _extract_text(pdf_bytes)
    if text:
        print(f"Extracted text: {len(text)} chars")
        print(f"First 500 chars:\n{text[:500]}")
    else:
        print("Could not extract text from PDF")
else:
    print("\n*** FAILED: No PDF returned ***")
