"""
Extrator de PDF v2 para o pipeline RAG do BBSIA.

Gera dois artefatos:
  - textos_extraidos_v2.txt: formato legado, facil de inspecionar.
  - documentos_extraidos_v2.json: formato estruturado para chunking RAG.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import statistics
from typing import Any

import fitz  # PyMuPDF

try:
    from classificador_artigo import classificar_e_registrar
    _HAS_CLASSIFICADOR = True
except ImportError:
    _HAS_CLASSIFICADOR = False


INPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(INPUT_DIR, "docs")
UPLOADS_DIR = os.path.join(INPUT_DIR, "uploads")
QUARANTINE_DIR = os.path.join(UPLOADS_DIR, "quarantine")
APPROVED_DIR = os.path.join(UPLOADS_DIR, "approved")
DATA_DIR = os.path.join(INPUT_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)
LOGGER = logging.getLogger(__name__)

STRUCTURED_OUTPUT_FILE = os.path.join(DATA_DIR, "documentos_extraidos_v2.json")

# Arquivos a ignorar na extracao.
IGNORAR = {"Dataset NICE.xlsx"}


def list_pdf_files() -> list[str]:
    """Lista PDFs da raiz, docs/ e uploads/approved (somente aprovados)."""
    pdf_paths: list[str] = []

    for folder in (INPUT_DIR, DOCS_DIR, APPROVED_DIR):
        if not os.path.exists(folder):
            continue
        for file_name in sorted(os.listdir(folder)):
            if not file_name.lower().endswith(".pdf"):
                continue
            if file_name in IGNORAR:
                continue
            pdf_paths.append(os.path.join(folder, file_name))

    return pdf_paths


def document_label(pdf_path: str) -> str:
    """Retorna um rotulo estavel para metadados."""
    base_name = os.path.basename(pdf_path)
    if os.path.commonpath([os.path.abspath(pdf_path), os.path.abspath(UPLOADS_DIR)]) == os.path.abspath(UPLOADS_DIR):
        return f"uploads/{base_name}"
    return base_name


def clean_extracted_text(text: str) -> str:
    """Limpa artefatos comuns da extracao PDF."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    cleaned = []
    blank_count = 0

    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            blank_count += 1
            if blank_count <= 2:
                cleaned.append("")
        else:
            blank_count = 0
            cleaned.append(stripped)

    return "\n".join(cleaned).strip()


def _table_to_markdown(rows: list[list[Any]]) -> str:
    """Converte tabela extraida pelo PyMuPDF para Markdown."""
    cleaned_rows: list[list[str]] = []
    for row in rows:
        values = [str(cell or "").strip().replace("\n", " ") for cell in row]
        if any(values):
            cleaned_rows.append(values)

    if not cleaned_rows:
        return ""

    width = max(len(row) for row in cleaned_rows)
    normalized = [row + [""] * (width - len(row)) for row in cleaned_rows]
    header = normalized[0]
    body = normalized[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _extract_tables(page: fitz.Page) -> list[dict[str, Any]]:
    """Extrai tabelas em elementos separados, quando disponivel."""
    if not hasattr(page, "find_tables"):
        return []

    try:
        tables = page.find_tables()
    except Exception:
        return []

    elements: list[dict[str, Any]] = []
    for table_index, table in enumerate(getattr(tables, "tables", []), start=1):
        try:
            markdown = _table_to_markdown(table.extract())
        except Exception:
            continue
        if markdown:
            elements.append(
                {
                    "tipo": "table",
                    "texto": markdown,
                    "secao": None,
                    "table_index": table_index,
                }
            )
    return elements


def _span_lines(page: fitz.Page) -> list[dict[str, Any]]:
    """Extrai linhas com informacoes de fonte para inferir titulos."""
    try:
        payload = page.get_text("dict")
    except Exception:
        return []

    lines: list[dict[str, Any]] = []
    for block in payload.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = " ".join(str(span.get("text", "")).strip() for span in spans).strip()
            text = re.sub(r"\s{2,}", " ", text)
            if not text:
                continue
            sizes = [float(span.get("size", 0)) for span in spans if span.get("text", "").strip()]
            fonts = " ".join(str(span.get("font", "")) for span in spans)
            lines.append(
                {
                    "texto": text,
                    "font_size": max(sizes) if sizes else 0.0,
                    "is_bold": "bold" in fonts.lower(),
                }
            )
    return lines


def _looks_like_heading(text: str, size: float, median_size: float, is_bold: bool) -> bool:
    """Heuristica conservadora para detectar titulos e secoes."""
    normalized = text.strip()
    if not normalized or len(normalized) > 140:
        return False
    if normalized.endswith((".", ",", ";", ":")) and len(normalized.split()) > 8:
        return False

    numbered = bool(re.match(r"^(\d+(\.\d+)*|[A-Z]\.)\s+.+", normalized))
    mostly_upper = normalized.upper() == normalized and len(normalized.split()) <= 12
    bigger = median_size > 0 and size >= median_size + 1.0
    return numbered or mostly_upper or (bigger and (is_bold or len(normalized.split()) <= 10))


def _extract_text_elements(page: fitz.Page) -> list[dict[str, Any]]:
    """Extrai texto em elementos com a secao corrente."""
    lines = _span_lines(page)
    if not lines:
        text = clean_extracted_text(page.get_text("text"))
        return [{"tipo": "text", "texto": text, "secao": None}] if text else []

    font_sizes = [line["font_size"] for line in lines if line["font_size"] > 0]
    median_size = statistics.median(font_sizes) if font_sizes else 0.0

    elements: list[dict[str, Any]] = []
    current_section: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        text = clean_extracted_text("\n".join(buffer))
        if text:
            elements.append({"tipo": "text", "texto": text, "secao": current_section})
        buffer.clear()

    for line in lines:
        text = line["texto"]
        if _looks_like_heading(text, line["font_size"], median_size, line["is_bold"]):
            flush()
            current_section = text
            elements.append({"tipo": "section", "texto": text, "secao": current_section})
        else:
            buffer.append(text)

    flush()
    return elements


def _ocr_page_if_needed(page: fitz.Page) -> str | None:
    """Executa OCR opcional quando a pagina nao possui texto extraivel."""
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except Exception:
        return None

    try:
        pix = page.get_pixmap(dpi=200)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(image, lang="por+eng")
    except Exception:
        return None

    cleaned = clean_extracted_text(text)
    return cleaned or None


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.document import DocItemLabel
    except ImportError:
        LOGGER.warning("Docling nao instalado. Usando fallback PyMuPDF.")
        return _fallback_extract_text_from_pdf(pdf_path)
        
    try:
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        
        pages_dict = {}
        for item, level in result.document.iterate_items():
            page_no = 1
            if getattr(item, "prov", None) and len(item.prov) > 0:
                page_no = item.prov[0].page_no
                
            if page_no not in pages_dict:
                pages_dict[page_no] = {"elementos": [], "ocr_usado": False}
                
            label = getattr(item, "label", None)
            
            tipo = "text"
            secao = None
            texto = ""
            
            if label in [DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER, DocItemLabel.PAGE_HEADER]:
                tipo = "section"
                texto = getattr(item, "text", "")
            elif label == DocItemLabel.TABLE:
                tipo = "table"
                texto = item.export_to_markdown() if hasattr(item, "export_to_markdown") else ""
            elif getattr(item, "text", None):
                tipo = "text"
                texto = item.text
            
            texto = texto.strip() if isinstance(texto, str) else ""
            if not texto:
                continue
                
            pages_dict[page_no]["elementos"].append({
                "tipo": tipo,
                "texto": texto,
                "secao": secao
            })
            
        for page_no, page_data in pages_dict.items():
            curr_secao = None
            for el in page_data["elementos"]:
                if el["tipo"] == "section":
                    curr_secao = el["texto"]
                el["secao"] = curr_secao
                
        pages_list = []
        for page_no in sorted(pages_dict.keys()):
            elements = pages_dict[page_no]["elementos"]
            page_text = "\n\n".join(e["texto"] for e in elements if e.get("tipo") != "section")
            if not page_text.strip():
                page_text = "\n\n".join(e["texto"] for e in elements)
            
            pages_list.append({
                "pagina": page_no,
                "texto": clean_extracted_text(page_text),
                "elementos": elements,
                "ocr_usado": pages_dict[page_no]["ocr_usado"]
            })
            
        if not pages_list:
            return _fallback_extract_text_from_pdf(pdf_path)
            
        return pages_list
        
    except Exception as exc:
        LOGGER.error(f"Erro no Docling para {pdf_path}: {exc}")
        return _fallback_extract_text_from_pdf(pdf_path)

def _fallback_extract_text_from_pdf(pdf_path: str) -> list[dict[str, Any]]:
    """
    Extrai texto, titulos e tabelas de um PDF.

    Retorna uma lista de paginas com os campos:
      {pagina, texto, elementos, ocr_usado}
    """
    doc = fitz.open(pdf_path)
    pages: list[dict[str, Any]] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text_elements = _extract_text_elements(page)
        table_elements = _extract_tables(page)

        ocr_used = False
        has_text = any(e.get("texto", "").strip() for e in text_elements if e.get("tipo") != "section")
        if not has_text:
            ocr_text = _ocr_page_if_needed(page)
            if ocr_text:
                ocr_used = True
                text_elements.append({"tipo": "ocr_text", "texto": ocr_text, "secao": None})

        elements = text_elements + table_elements
        page_text = "\n\n".join(
            e["texto"] for e in elements if e.get("tipo") != "section" and e.get("texto")
        )
        if not page_text.strip():
            page_text = "\n\n".join(e["texto"] for e in elements if e.get("texto"))

        if page_text.strip() or elements:
            pages.append(
                {
                    "pagina": page_num + 1,
                    "texto": clean_extracted_text(page_text),
                    "elementos": elements,
                    "ocr_usado": ocr_used,
                }
            )

    doc.close()
    return pages


def _write_legacy_page(out_f, page: dict[str, Any]) -> None:
    out_f.write(f"--- Pagina {page['pagina']} ---\n")
    if page.get("ocr_usado"):
        out_f.write("[OCR aplicado]\n")

    for element in page.get("elementos", []):
        kind = element.get("tipo")
        text = str(element.get("texto", "")).strip()
        if not text:
            continue
        if kind == "section":
            out_f.write(f"\n### SECAO: {text}\n")
        elif kind == "table":
            out_f.write("\n[TABELA EXTRAIDA]\n")
            out_f.write(text + "\n")
        else:
            out_f.write(text + "\n")
    out_f.write("\n")


def run_extraction() -> dict[str, Any]:
    """Pipeline principal de extracao."""
    pdf_files = list_pdf_files()
    doc_errors: list[dict[str, str]] = []
    metadata_errors: list[dict[str, str]] = []

    if not pdf_files:
        LOGGER.warning("event=extraction_no_pdfs_found")
        return {
            "total_pdfs_encontrados": 0,
            "pdfs_processados": 0,
            "total_paginas": 0,
            "total_caracteres": 0,
            "documento_erros": [],
            "metadado_erros": [],
            "structured_output_file": STRUCTURED_OUTPUT_FILE,
        }

    structured_output_path = os.path.join(INPUT_DIR, STRUCTURED_OUTPUT_FILE)
    LOGGER.info("event=extraction_started total_pdfs=%s", len(pdf_files))

    total_pages = 0
    total_chars = 0
    structured_documents: list[dict[str, Any]] = []

    for pdf_path in pdf_files:
        doc_label = document_label(pdf_path)
        LOGGER.info("event=extraction_document_started documento=%s", doc_label)

        try:
            pages = extract_text_from_pdf(pdf_path)

            # Classifica o documento e atualiza biblioteca.json
            doc_metadados = None
            if _HAS_CLASSIFICADOR:
                try:
                    metadado = classificar_e_registrar(pdf_path, usar_llm=True)
                    doc_metadados = metadado.model_dump()
                    t = metadado.titulo
                    LOGGER.info(
                        "event=extraction_metadata_ok documento=%s titulo=%s",
                        doc_label,
                        t[:60] + "..." if len(t) > 60 else t,
                    )
                except Exception as meta_exc:
                    metadata_errors.append({"documento": doc_label, "erro": str(meta_exc)})
                    LOGGER.warning(
                        "event=extraction_metadata_failed documento=%s erro=%s",
                        doc_label,
                        meta_exc,
                    )

            entry: dict[str, Any] = {"documento": doc_label, "paginas": pages}
            if doc_metadados:
                entry["metadados"] = doc_metadados
            structured_documents.append(entry)

            for page in pages:
                total_chars += len(page.get("texto", ""))

            total_pages += len(pages)
            LOGGER.info("event=extraction_document_completed documento=%s paginas=%s", doc_label, len(pages))

        except Exception as exc:
            doc_errors.append({"documento": doc_label, "erro": str(exc)})
            LOGGER.exception("event=extraction_document_failed documento=%s", doc_label)

    with open(structured_output_path, "w", encoding="utf-8") as structured_f:
        json.dump(
            {"versao": 2, "documentos": structured_documents},
            structured_f,
            ensure_ascii=False,
            indent=2,
        )

    processed = len(structured_documents)
    LOGGER.info(
        "event=extraction_completed total_pdfs=%s processados=%s total_paginas=%s total_caracteres=%s erro_docs=%s erro_meta=%s output=%s",
        len(pdf_files),
        processed,
        total_pages,
        total_chars,
        len(doc_errors),
        len(metadata_errors),
        STRUCTURED_OUTPUT_FILE,
    )
    return {
        "total_pdfs_encontrados": len(pdf_files),
        "pdfs_processados": processed,
        "total_paginas": total_pages,
        "total_caracteres": total_chars,
        "documento_erros": doc_errors,
        "metadado_erros": metadata_errors,
        "structured_output_file": STRUCTURED_OUTPUT_FILE,
    }


if __name__ == "__main__":
    run_extraction()


