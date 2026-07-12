"""CSV ingestion helpers for Zotero/Scopus-like exports and gold audits."""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


MAX_CSV_BYTES = 12 * 1024 * 1024

FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "title": ("title", "titulo", "article_title", "item_title"),
    "abstract": ("abstract", "abstract_note", "abstractnote", "resumen", "description"),
    "doi": ("doi", "digital_object_identifier"),
    "year": ("year", "publication_year", "pub_year", "date", "publication_date"),
    "authors": ("author", "authors", "creators", "autor", "autores"),
    "journal": ("publication_title", "journal", "journal_title", "venue", "source_title"),
    "url": ("url", "urls", "link", "document_url"),
    "pdf_path": ("file_attachments", "file", "pdf_path", "pdf", "attachments"),
    "item_type": ("item_type", "type", "publication_type", "document_type"),
    "language": ("language", "idioma"),
    "label": ("manual_decision", "inclusion_manual", "inclusion", "incluido", "label", "gold", "gold_label", "relevant"),
}

POSITIVE_VALUES = {"1", "true", "yes", "y", "si", "sí", "include", "included", "incluir", "relevant", "positivo"}
NEGATIVE_VALUES = {"0", "false", "no", "n", "exclude", "excluded", "excluir", "irrelevant", "negativo"}
POSITIVE_VALUES.update({"incluido"})
NEGATIVE_VALUES.update({"excluido"})


def normalise_header(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def decode_csv_bytes(content: bytes) -> str:
    if len(content) > MAX_CSV_BYTES:
        raise ValueError("El CSV supera el limite permitido de 12 MB.")
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def read_csv_rows(content: bytes) -> Tuple[List[Dict[str, str]], List[str]]:
    text = decode_csv_bytes(content)
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        raise ValueError("El CSV no contiene cabeceras.")
    rows = [dict(row) for row in reader]
    return rows, list(reader.fieldnames)


def detect_column_mapping(fieldnames: Iterable[str], label_column: str = "") -> Dict[str, str]:
    originals = list(fieldnames)
    normalised = {normalise_header(name): name for name in originals}
    mapping: Dict[str, str] = {}

    for target, aliases in FIELD_ALIASES.items():
        if target == "label" and label_column:
            if label_column not in originals:
                raise ValueError(f"No existe la columna de etiqueta '{label_column}'.")
            mapping[target] = label_column
            continue
        for alias in aliases:
            if alias in normalised:
                mapping[target] = normalised[alias]
                break

    if "label" not in mapping:
        for name in originals:
            header = normalise_header(name)
            if "inclus" in header or "manual" in header or "gold" in header:
                mapping["label"] = name
                break

    return mapping


def _value(row: Dict[str, str], mapping: Dict[str, str], key: str) -> str:
    column = mapping.get(key, "")
    return str(row.get(column, "") if column else "").strip()


def _safe_year(value: str) -> Optional[int]:
    match = re.search(r"\b(19|20)\d{2}\b", str(value or ""))
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _first_existing_pdf(value: str) -> str:
    raw = str(value or "").strip().strip('"')
    if not raw:
        return ""
    for part in re.split(r"\s*;\s*", raw):
        candidate = part.strip().strip('"')
        if candidate and Path(candidate).exists():
            return candidate
    return raw


def normalise_label(value: str, positive_values: Optional[Iterable[str]] = None) -> Optional[int]:
    token = normalise_header(str(value or ""))
    positives = {normalise_header(v) for v in (positive_values or POSITIVE_VALUES)}
    negatives = {normalise_header(v) for v in NEGATIVE_VALUES}
    if token in positives:
        return 1
    if token in negatives:
        return 0
    return None


def normalise_csv_articles(
    content: bytes,
    *,
    source_name: str = "CSV/Zotero",
    include_labels: bool = False,
    label_column: str = "",
    positive_values: Optional[Iterable[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows, fieldnames = read_csv_rows(content)
    mapping = detect_column_mapping(fieldnames, label_column=label_column)
    if "title" not in mapping:
        raise ValueError("No pude detectar columna de titulo. Use un CSV con Title/title.")

    articles: List[Dict[str, Any]] = []
    label_counts: Dict[str, int] = {}
    unparsed_label_values: Dict[str, int] = {}
    for idx, row in enumerate(rows, start=1):
        title = _value(row, mapping, "title")
        if not title:
            continue
        abstract = _value(row, mapping, "abstract")
        doi = _value(row, mapping, "doi")
        year = _safe_year(_value(row, mapping, "year"))
        pdf_path = _first_existing_pdf(_value(row, mapping, "pdf_path"))
        article: Dict[str, Any] = {
            "title": title,
            "abstract": abstract,
            "doi": doi,
            "url": _value(row, mapping, "url"),
            "year": year,
            "authors": _value(row, mapping, "authors"),
            "journal": _value(row, mapping, "journal"),
            "venue": _value(row, mapping, "journal"),
            "publication_type": _value(row, mapping, "item_type"),
            "language": _value(row, mapping, "language"),
            "pdf_path": pdf_path,
            "pdf_url": _value(row, mapping, "url"),
            "source": source_name,
            "retrieval_phase": "csv_import",
            "retrieval_query": "csv_upload",
            "retrieval_mode": "csv",
            "open_access": bool(pdf_path),
            "_csv_row_id": f"csv-{idx}",
        }
        if include_labels and mapping.get("label"):
            raw_label = _value(row, mapping, "label")
            label_counts[raw_label] = label_counts.get(raw_label, 0) + 1
            parsed_label = normalise_label(raw_label, positive_values=positive_values)
            if raw_label and parsed_label is None:
                unparsed_label_values[raw_label] = unparsed_label_values.get(raw_label, 0) + 1
            article["_gold_label"] = parsed_label if parsed_label is not None else 0
            article["_gold_label_raw"] = raw_label
        articles.append(article)

    preview = build_csv_preview(articles, fieldnames, mapping, label_counts, unparsed_label_values)
    return articles, preview


def build_csv_preview(
    articles: List[Dict[str, Any]],
    fieldnames: List[str],
    mapping: Dict[str, str],
    label_counts: Optional[Dict[str, int]] = None,
    unparsed_label_values: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    rows_with_title = sum(1 for item in articles if item.get("title"))
    rows_with_abstract = sum(1 for item in articles if item.get("abstract"))
    rows_with_doi = sum(1 for item in articles if item.get("doi"))
    rows_with_year = sum(1 for item in articles if item.get("year"))
    rows_with_pdf = sum(1 for item in articles if item.get("pdf_path"))
    sample = [
        {
            "title": item.get("title", ""),
            "year": item.get("year", ""),
            "doi": item.get("doi", ""),
            "abstract_chars": len(str(item.get("abstract") or "")),
            "gold_label": item.get("_gold_label_raw", ""),
        }
        for item in articles[:5]
    ]
    return {
        "total_rows": len(articles),
        "with_title": rows_with_title,
        "with_abstract": rows_with_abstract,
        "with_doi": rows_with_doi,
        "with_year": rows_with_year,
        "with_pdf_path": rows_with_pdf,
        "fieldnames": fieldnames,
        "mapping": mapping,
        "label_counts": label_counts or {},
        "unparsed_label_values": unparsed_label_values or {},
        "sample": sample,
        "warnings": _preview_warnings(articles, mapping, unparsed_label_values or {}),
    }


def strip_private_labels(article: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in article.items() if not key.startswith("_gold")}


def _preview_warnings(
    articles: List[Dict[str, Any]],
    mapping: Dict[str, str],
    unparsed_label_values: Dict[str, int],
) -> List[str]:
    warnings: List[str] = []
    if not articles:
        warnings.append("No se detectaron articulos con titulo.")
        return warnings
    if "abstract" not in mapping:
        warnings.append("No se detecto columna de abstract; el cribado dependera de metadatos/PDF.")
    if "doi" not in mapping:
        warnings.append("No se detecto DOI; la deduplicacion sera menos precisa.")
    missing_abstract = sum(1 for item in articles if not item.get("abstract"))
    if missing_abstract:
        warnings.append(f"{missing_abstract} articulos no tienen abstract.")
    if unparsed_label_values:
        warnings.append(f"Etiquetas no reconocidas: {unparsed_label_values}.")
    return warnings
