from __future__ import annotations

import copy
import json

import pytest

from bbsia.domain.catalogo import service as catalogo


def _valid_solution() -> dict:
    return {
        "id": "solucao-teste",
        "nome": "Solucao de Teste",
        "descricao": "Descricao suficientemente detalhada da solucao piloto para validacao.",
        "orgao": "BBSIA",
        "area": "ia",
        "problema": "Problema operacional que precisa de apoio com inteligencia artificial.",
        "solucao": "Solucao baseada em catalogo estruturado, validacao e busca semantica.",
        "sintomas": ["Sintoma A", "Sintoma B"],
        "causa_raiz": "Falta de integracao no sistema X.",
        "pre_condicoes": ["Acesso admin"],
        "passos_implantacao": ["Passo 1", "Passo 2"],
        "riscos": ["Risco de lentidao"],
        "tecnologias": ["Python", "FAISS"],
        "status": "piloto",
        "conformidade": {
            "dados_pessoais": "a_confirmar",
            "base_legal_lgpd": "Uso interno em avaliacao.",
            "hospedagem": "local",
            "modelo": "qwen3.5:7b-instruct",
            "licenca_modelo": "a confirmar",
            "dependencias_open_source": ["fastapi", "faiss-cpu"],
        },
    }


def test_validate_solution_accepts_valid_payload():
    catalogo.validate_solution(_valid_solution())


def test_validate_solution_rejects_missing_required_field():
    payload = _valid_solution()
    payload.pop("problema")

    with pytest.raises(ValueError, match="problema"):
        catalogo.validate_solution(payload)


def test_validate_solution_rejects_invalid_enum():
    payload = _valid_solution()
    payload["status"] = "desconhecido"

    with pytest.raises(ValueError, match="status"):
        catalogo.validate_solution(payload)


def test_validate_catalog_rejects_duplicate_ids():
    solution = _valid_solution()

    with pytest.raises(ValueError, match="duplicado"):
        catalogo.validate_catalog([solution, copy.deepcopy(solution)])


def test_materialize_solution_chunks_writes_embedding_input(tmp_path):
    catalog_path = tmp_path / "solucoes.json"
    output_path = tmp_path / "chunks.json"
    catalog_path.write_text(json.dumps([_valid_solution()], ensure_ascii=False), encoding="utf-8")

    result = catalogo.materialize_solution_chunks(catalog_path=catalog_path, output_path=output_path)
    chunks = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["total_solucoes"] == 1
    assert result["total_chunks"] == 1
    assert chunks[0]["content_type"] == "solucao_piloto"
    assert chunks[0]["solution_id"] == "solucao-teste"
    assert "Base legal LGPD" in chunks[0]["texto"]
