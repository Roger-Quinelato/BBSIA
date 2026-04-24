from __future__ import annotations

import json
import os

import pytest

from embedding import _load_chunks


def test_load_chunks_raises_on_missing_file(tmp_path):
    """Verifica que FileNotFoundError é levantado quando chunks.json não existe."""
    fake_path = str(tmp_path / "inexistente.json")
    with pytest.raises(FileNotFoundError, match="nao encontrado"):
        _load_chunks(fake_path)


def test_load_chunks_raises_on_empty_list(tmp_path):
    """Verifica que ValueError é levantado quando chunks.json é uma lista vazia."""
    chunks_file = tmp_path / "chunks.json"
    chunks_file.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="vazio ou em formato invalido"):
        _load_chunks(str(chunks_file))


def test_load_chunks_raises_on_invalid_format(tmp_path):
    """Verifica que ValueError é levantado quando chunks.json não contém lista."""
    chunks_file = tmp_path / "chunks.json"
    chunks_file.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="vazio ou em formato invalido"):
        _load_chunks(str(chunks_file))


def test_load_chunks_raises_on_missing_texto_field(tmp_path):
    """Verifica que ValueError é levantado quando um chunk não possui campo 'texto'."""
    chunks_file = tmp_path / "chunks.json"
    chunks_file.write_text(
        json.dumps([{"id": 0, "documento": "doc.pdf"}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="campo 'texto'"):
        _load_chunks(str(chunks_file))


def test_load_chunks_success(tmp_path):
    """Verifica que chunks válidos são carregados corretamente."""
    chunks_data = [
        {"id": 0, "texto": "Trecho A sobre infraestrutura."},
        {"id": 1, "texto": "Trecho B sobre IA."},
        {"id": 2, "texto": "Trecho C sobre ética."},
    ]
    chunks_file = tmp_path / "chunks.json"
    chunks_file.write_text(json.dumps(chunks_data, ensure_ascii=False), encoding="utf-8")

    result = _load_chunks(str(chunks_file))
    assert len(result) == 3
    assert result[0]["texto"] == "Trecho A sobre infraestrutura."
