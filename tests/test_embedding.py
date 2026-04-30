from __future__ import annotations

import json
import os

import pytest

from embedding import _load_chunks, _load_parents_map, _split_lean_chunks_and_parents


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


def test_load_parents_map_ignores_missing_file(tmp_path):
    assert _load_parents_map(str(tmp_path / "parents.json")) == {}


def test_split_lean_chunks_moves_legacy_parent_text_to_map():
    chunks = [
        {"id": 0, "parent_id": "parent-0", "texto": "filho", "parent_text": "parent completo"},
        {"id": 1, "parent_id": "parent-1", "texto": "outro filho"},
    ]

    lean_chunks, parents_map = _split_lean_chunks_and_parents(chunks, {"parent-1": "parent vindo do arquivo"})

    assert lean_chunks == [
        {"id": 0, "parent_id": "parent-0", "texto": "filho"},
        {"id": 1, "parent_id": "parent-1", "texto": "outro filho"},
    ]
    assert parents_map == {
        "parent-0": "parent completo",
        "parent-1": "parent vindo do arquivo",
    }
