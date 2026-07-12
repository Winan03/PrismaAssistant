"""Proceso aislado para extraer texto de PDFs con timeout real en Windows."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from app.extraction import pdf_extractor


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        return 2

    pdf_path = Path(argv[0])
    mode = argv[1].strip().lower()
    try:
        min_len = int(argv[2])
    except ValueError:
        min_len = 3000

    pdf_bytes = pdf_path.read_bytes()
    if not pdf_bytes.startswith(b"%PDF-"):
        return 3

    logging.getLogger().setLevel(logging.ERROR)
    if mode == "selective":
        text = pdf_extractor.extract_selective_sections(pdf_bytes, min_len=min_len)
    elif mode == "full":
        text = pdf_extractor._extract_text(pdf_bytes)
    else:
        return 4

    sys.stdout.buffer.write((text or "").encode("utf-8", errors="replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
