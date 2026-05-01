"""
Chunking estruturado para o pipeline RAG do BBSIA.

Le documentos_extraidos_v2.json, gerado pelo extrator v2, e cria chunks
parent-child com metadados de documento, pagina, secao e tipo de conteudo.
O arquivo estruturado e entrada obrigatoria deste modulo.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Iterable

# Parâmetros de chunking:
# - CHILD_CHUNK_SIZE : tamanho (em palavras) dos child chunks (usados no embedding)
# - CHILD_CHUNK_OVERLAP: sobreposição entre child chunks consecutivos
# - PARENT_MAX_WORDS : limite máximo de palavras no parent_text enviado ao contexto

CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 35
PARENT_MAX_WORDS = 900

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
STRUCTURED_INPUT_FILE = os.path.join(DATA_DIR, "documentos_extraidos_v2.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "chunks.json")
PARENTS_FILE = os.path.join(DATA_DIR, "parents.json")
UPLOAD_METADATA_FILE = os.path.join("uploads", "metadata_uploads.json")
BIBLIOTECA_FILE = os.path.join(DATA_DIR, "biblioteca.json")
LOGGER = logging.getLogger(__name__)

CATEGORIAS_DOCUMENTOS = {}
try:
    import yaml
    import os
    cat_path = os.path.join(_REPO_ROOT, 'bbsia', 'core', 'categorias.yaml')
    if os.path.exists(cat_path):
        with open(cat_path, 'r', encoding='utf-8') as f:
            CATEGORIAS_DOCUMENTOS = yaml.safe_load(f) or {}
except Exception as e:
    LOGGER.warning(f'Failed to load bbsia/core/categorias.yaml: {e}')


def _script_dir() -> str:
    return _REPO_ROOT


def load_upload_metadata(filepath: str | None = None) -> dict:
    """Carrega metadados de uploads (area/assuntos), se existir."""
    resolved_path = os.path.join(_script_dir(), filepath or UPLOAD_METADATA_FILE)
    if not os.path.exists(resolved_path):
        return {}

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def normalize_upload_doc_name(doc_name: str) -> str:
    """Normaliza chave de documento para o padrao uploads/<arquivo.pdf>."""
    doc = (doc_name or "").strip().replace("\\", "/")
    if doc.startswith("uploads/"):
        return f"uploads/{doc.split('/')[-1]}"
    return doc


def _load_biblioteca() -> dict:
    """Carrega o catálogo biblioteca.json indexado por nome de arquivo."""
    path = os.path.join(_script_dir(), BIBLIOTECA_FILE)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("documentos"), list):
            index: dict[str, dict] = {}
            for doc in data["documentos"]:
                orig = doc.get("documento_original", "")
                if orig:
                    index[orig] = doc
                    index[os.path.basename(orig)] = doc
            return index
    except Exception:
        pass
    return {}


def _is_specific_metadata(area: str, assuntos: list[str]) -> bool:
    area_norm = str(area or "").strip().lower()
    assuntos_norm = [str(a or "").strip().lower() for a in assuntos]
    if area_norm and area_norm != "geral":
        return True
    return any(a and a != "geral" for a in assuntos_norm)


def _metadata_from_biblioteca_item(bib_item: dict, authorship: dict) -> dict:
    area = str(bib_item.get("area_tematica", "geral")).strip() or "geral"
    assuntos = bib_item.get("assuntos", [])
    if not isinstance(assuntos, list):
        assuntos = ["geral"]
    assuntos = [str(a).strip() for a in assuntos if str(a).strip()] or ["geral"]
    return {"area": area, "assuntos": assuntos, **authorship}


def get_doc_metadata(doc_name: str) -> dict:
    """Retorna area, assuntos e metadados de autoria para um documento.

    Prioridade para area/assuntos:
      1. upload_metadata (vem do usuário)
      2. biblioteca.json, se a classificação automática for específica
      3. CATEGORIAS_DOCUMENTOS, como fallback legado/manual
      4. biblioteca.json genérico, se não houver fonte melhor
      5. Fallback genérico

    Campos de autoria (doc_titulo, doc_autores, doc_ano) vêm de
    biblioteca.json quando disponíveis.
    """
    normalized_name = normalize_upload_doc_name(doc_name)
    base_name = os.path.basename(doc_name)

    # Metadados de autoria (sempre de biblioteca.json)
    bib = _load_biblioteca()
    bib_item = bib.get(normalized_name) or bib.get(doc_name) or bib.get(base_name)
    doc_titulo = ""
    doc_autores: list[str] = []
    doc_ano: int | None = None
    if isinstance(bib_item, dict):
        doc_titulo = bib_item.get("titulo", "")
        doc_autores = bib_item.get("autores", [])
        doc_ano = bib_item.get("ano")

    authorship = {"doc_titulo": doc_titulo, "doc_autores": doc_autores, "doc_ano": doc_ano}

    # 1. Upload metadata
    upload_metadata = load_upload_metadata()
    upload_item = upload_metadata.get(normalized_name)
    if isinstance(upload_item, dict):
        area = str(upload_item.get("area", "geral")).strip() or "geral"
        assuntos = upload_item.get("assuntos", [])
        if not isinstance(assuntos, list):
            assuntos = [str(assuntos)]
        assuntos = [str(a).strip() for a in assuntos if str(a).strip()]
        if not assuntos:
            assuntos = ["geral"]
        return {"area": area, "assuntos": assuntos, **authorship}

    # 2. biblioteca.json, when the automatic classification is specific.
    bib_meta = None
    if isinstance(bib_item, dict):
        bib_meta = _metadata_from_biblioteca_item(bib_item, authorship)
        if _is_specific_metadata(bib_meta["area"], bib_meta["assuntos"]):
            return bib_meta

    # 3. CATEGORIAS_DOCUMENTOS (legado)
    cat = CATEGORIAS_DOCUMENTOS.get(doc_name) or CATEGORIAS_DOCUMENTOS.get(normalized_name)
    if isinstance(cat, dict):
        return {**cat, **authorship}

    # 4. biblioteca.json generic classification, only when no better source exists.
    if bib_meta is not None:
        return bib_meta

    # 5. Fallback
    return {"area": "geral", "assuntos": ["geral"], **authorship}


def clean_text(text: str) -> str:
    """Limpa texto fragmentado pelo extrator PDF."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("### SECAO:"):
            lines.append(stripped)

    joined = " ".join(lines)
    joined = re.sub(r"\s+(\d{1,2})\s+", " ", joined)
    joined = re.sub(r"\s{2,}", " ", joined)
    return joined.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Divide texto em chunks de palavras com overlap.
    Usado como splitter child dentro de parents estruturais.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        if end < len(words):
            last_period = chunk_text_str.rfind(".")
            if last_period > len(chunk_text_str) * 0.5:
                chunk_text_str = chunk_text_str[: last_period + 1]
                actual_words = len(chunk_text_str.split())
                if actual_words > overlap:
                    end = start + actual_words

        chunks.append(chunk_text_str)

        if end >= len(words):
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = start + max(1, chunk_size - overlap)
        start = next_start

    return chunks


def _load_structured_payload(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and isinstance(payload.get("documentos"), list):
            return payload
    except Exception:
        return None
    return None


def _word_count(text: str) -> int:
    return len((text or "").split())


def _trim_parent_text(text: str, max_words: int = PARENT_MAX_WORDS) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip() + " ..."


def _make_parent(
    *,
    documento: str,
    pagina: int,
    section_heading: str | None,
    content_type: str,
    text_parts: Iterable[str],
    ocr_usado: bool = False,
    table_index: int | None = None,
) -> dict[str, Any] | None:
    text = clean_text("\n".join(part for part in text_parts if part))
    if content_type == "table":
        text = "\n".join(part.strip() for part in text_parts if part and part.strip())
    min_words = 3 if content_type == "table" else 20
    if not text or _word_count(text) < min_words:
        return None

    return {
        "documento": documento,
        "pagina": pagina,
        "section_heading": section_heading,
        "content_type": content_type,
        "texto": text,
        "ocr_usado": ocr_usado,
        "table_index": table_index,
    }


def _structured_parent_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    parents: list[dict[str, Any]] = []

    for doc in payload.get("documentos", []):
        documento = str(doc.get("documento", "")).strip()
        if not documento:
            continue

        for page in doc.get("paginas", []):
            pagina = int(page.get("pagina", 0) or 0)
            ocr_usado = bool(page.get("ocr_usado"))
            current_section: str | None = None
            buffer: list[str] = []
            buffer_section: str | None = None
            buffer_type = "text"

            def flush_text() -> None:
                nonlocal buffer, buffer_section, buffer_type
                parent = _make_parent(
                    documento=documento,
                    pagina=pagina,
                    section_heading=buffer_section,
                    content_type=buffer_type,
                    text_parts=buffer,
                    ocr_usado=ocr_usado,
                )
                if parent:
                    parents.append(parent)
                buffer = []
                buffer_section = current_section
                buffer_type = "text"

            for element in page.get("elementos", []):
                kind = str(element.get("tipo", "text"))
                text = str(element.get("texto", "")).strip()
                if not text:
                    continue

                if kind == "section":
                    flush_text()
                    current_section = text
                    buffer_section = current_section
                    continue

                if kind == "table":
                    flush_text()
                    parent = _make_parent(
                        documento=documento,
                        pagina=pagina,
                        section_heading=current_section or element.get("secao"),
                        content_type="table",
                        text_parts=[text],
                        ocr_usado=ocr_usado,
                        table_index=element.get("table_index"),
                    )
                    if parent:
                        parents.append(parent)
                    continue

                next_type = "ocr_text" if kind == "ocr_text" else "text"
                next_section = element.get("secao") or current_section
                if buffer and (next_section != buffer_section or next_type != buffer_type):
                    flush_text()
                buffer_section = next_section
                buffer_type = next_type
                buffer.append(text)

            flush_text()

    return parents


def _child_chunks_for_parent(parent: dict[str, Any]) -> list[str]:
    if parent["content_type"] == "table":
        return chunk_text(parent["texto"], max(CHILD_CHUNK_SIZE, 260), CHILD_CHUNK_OVERLAP)
    return chunk_text(parent["texto"], CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP)


def _materialize_chunks(parents: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    all_chunks: list[dict[str, Any]] = []
    parents_map: dict[str, str] = {}
    chunk_id = 0

    for parent_index, parent in enumerate(parents):
        meta = get_doc_metadata(parent["documento"])
        parent_id = f"parent-{parent_index}"
        parent_text = _trim_parent_text(parent["texto"])
        parents_map[parent_id] = parent_text

        for child_index, child in enumerate(_child_chunks_for_parent(parent)):
            if not child or _word_count(child) < 3:
                continue
            all_chunks.append(
                {
                    "id": chunk_id,
                    "parent_id": parent_id,
                    "documento": parent["documento"],
                    "pagina": parent["pagina"],
                    "area": meta["area"],
                    "assuntos": meta["assuntos"],
                    "doc_titulo": meta.get("doc_titulo", ""),
                    "doc_autores": meta.get("doc_autores", []),
                    "doc_ano": meta.get("doc_ano"),
                    "section_heading": parent.get("section_heading"),
                    "content_type": parent.get("content_type", "text"),
                    "ocr_usado": bool(parent.get("ocr_usado")),
                    "table_index": parent.get("table_index"),
                    "chunk_index": child_index,
                    "texto": child,
                    "num_palavras": _word_count(child),
                    "parent_num_palavras": _word_count(parent["texto"]),
                }
            )
            chunk_id += 1

    return all_chunks, parents_map


def run_chunking() -> dict[str, Any]:
    """Pipeline principal de chunking."""
    script_dir = _script_dir()
    structured_path = os.path.join(script_dir, STRUCTURED_INPUT_FILE)
    output_path = os.path.join(script_dir, OUTPUT_FILE)
    parents_path = os.path.join(script_dir, PARENTS_FILE)

    structured_payload = _load_structured_payload(structured_path)
    if not structured_payload:
        msg = f"'{STRUCTURED_INPUT_FILE}' nao encontrado. Execute extrator_pdf_v2.py primeiro."
        LOGGER.error("event=chunking_missing_source error=%s", msg)
        raise FileNotFoundError(msg)

    LOGGER.info("event=chunking_started source=%s", STRUCTURED_INPUT_FILE)
    parents = _structured_parent_blocks(structured_payload)
    source = STRUCTURED_INPUT_FILE
    all_chunks, parents_map = _materialize_chunks(parents)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    with open(parents_path, "w", encoding="utf-8") as f:
        json.dump(parents_map, f, ensure_ascii=False, indent=2)

    total_palavras = sum(c["num_palavras"] for c in all_chunks)
    docs_unicos = {c["documento"] for c in all_chunks}
    areas_unicas = {c["area"] for c in all_chunks}
    tipos = sorted({c["content_type"] for c in all_chunks})
    parents_unicos = {c["parent_id"] for c in all_chunks}

    LOGGER.info(
        "event=chunking_completed source=%s documentos=%s parents=%s chunks=%s total_palavras=%s output=%s",
        source,
        len(docs_unicos),
        len(parents_unicos),
        len(all_chunks),
        total_palavras,
        OUTPUT_FILE,
    )

    return {
        "source": source,
        "documentos_processados": len(docs_unicos),
        "parents_gerados": len(parents_unicos),
        "total_chunks": len(all_chunks),
        "total_palavras": total_palavras,
        "areas": sorted(areas_unicas),
        "tipos_conteudo": tipos,
        "output_file": OUTPUT_FILE,
        "parents_file": PARENTS_FILE,
    }


if __name__ == "__main__":
    run_chunking()

