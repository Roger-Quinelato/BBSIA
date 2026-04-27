from __future__ import annotations

import fitz

import chunking
import extrator_pdf_v2
from extrator_pdf_v2 import extract_text_from_pdf


def test_extract_text_from_small_pdf(tmp_path):
    pdf_path = tmp_path / "mini.pdf"

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "BBSIA teste de extracao PDF")
    doc.save(pdf_path)
    doc.close()

    pages = extract_text_from_pdf(str(pdf_path))
    assert len(pages) == 1
    assert pages[0]["pagina"] == 1
    assert "BBSIA" in pages[0]["texto"]


def test_chunk_text_with_overlap():
    text = " ".join([f"w{i}" for i in range(30)])
    chunks = chunking.chunk_text(text=text, chunk_size=10, overlap=2)

    assert len(chunks) >= 3
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert first_words[-2:] == second_words[:2]


def test_upload_metadata_overrides_default(monkeypatch):
    monkeypatch.setattr(
        chunking,
        "load_upload_metadata",
        lambda filepath=None: {
            "uploads/novo_doc.pdf": {
                "area": "infraestrutura",
                "assuntos": ["kubernetes", "seguranca"],
            }
        },
    )

    meta = chunking.get_doc_metadata("uploads/novo_doc.pdf")
    assert meta["area"] == "infraestrutura"
    assert meta["assuntos"] == ["kubernetes", "seguranca"]


def test_extrator_pdf_vazio(tmp_path):
    """PDF válido mas sem texto deve retornar lista vazia ou páginas sem conteúdo."""
    pdf_path = tmp_path / "vazio.pdf"

    doc = fitz.open()
    doc.new_page()  # Página em branco, sem texto inserido.
    doc.save(pdf_path)
    doc.close()

    pages = extract_text_from_pdf(str(pdf_path))
    # Pode retornar lista vazia (sem conteúdo) ou página com texto vazio.
    if pages:
        for page in pages:
            texto = page.get("texto", "").strip()
            # Se retornou algo, o texto deve ser vazio ou mínimo.
            assert len(texto) < 10, f"Texto inesperado em PDF vazio: {texto[:50]}"


def test_extrator_pdf_multiplas_paginas(tmp_path):
    """PDF com múltiplas páginas deve retornar uma entrada por página com conteúdo."""
    pdf_path = tmp_path / "multi.pdf"

    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"Página {i + 1} do documento de teste BBSIA")
    doc.save(pdf_path)
    doc.close()

    pages = extract_text_from_pdf(str(pdf_path))
    assert len(pages) == 3
    for i, page in enumerate(pages):
        assert page["pagina"] == i + 1
        assert "BBSIA" in page["texto"]


def test_get_doc_metadata_fallback_generico():
    """Documentos desconhecidos devem receber area='geral' (não 'ia')."""
    meta = chunking.get_doc_metadata("arquivo_totalmente_desconhecido.pdf")
    assert meta["area"] == "geral"
    assert meta["assuntos"] == ["geral"]


def test_list_pdf_files_so_considera_uploads_approved(tmp_path, monkeypatch):
    input_dir = tmp_path / "repo"
    docs_dir = input_dir / "docs"
    uploads_dir = input_dir / "uploads"
    quarantine_dir = uploads_dir / "quarantine"
    approved_dir = uploads_dir / "approved"
    for path in (docs_dir, quarantine_dir, approved_dir):
        path.mkdir(parents=True, exist_ok=True)

    (quarantine_dir / "bloqueado.pdf").write_bytes(b"%PDF-1.4")
    (approved_dir / "aprovado.pdf").write_bytes(b"%PDF-1.4")
    (docs_dir / "doc.pdf").write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(extrator_pdf_v2, "INPUT_DIR", str(input_dir))
    monkeypatch.setattr(extrator_pdf_v2, "DOCS_DIR", str(docs_dir))
    monkeypatch.setattr(extrator_pdf_v2, "UPLOADS_DIR", str(uploads_dir))
    monkeypatch.setattr(extrator_pdf_v2, "QUARANTINE_DIR", str(quarantine_dir))
    monkeypatch.setattr(extrator_pdf_v2, "APPROVED_DIR", str(approved_dir))

    files = extrator_pdf_v2.list_pdf_files()
    as_text = "\n".join(files)
    assert "aprovado.pdf" in as_text
    assert "doc.pdf" in as_text
    assert "bloqueado.pdf" not in as_text
