"""
Catalogo estruturado de solucoes piloto BBSIA.

Valida entradas com JSON Schema e materializa chunks textuais para embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

BASE_DIR = Path(__file__).resolve().parents[3]
SCHEMA_FILE = BASE_DIR / "schemas" / "solucao_piloto.schema.json"
CATALOGO_FILE = BASE_DIR / "catalogo" / "solucoes_piloto.json"
OUTPUT_CHUNKS_FILE = BASE_DIR / "data" / "solucoes_piloto_chunks.json"


def load_schema(schema_path: Path = SCHEMA_FILE) -> dict[str, Any]:
    return json.loads(schema_path.read_text(encoding="utf-8"))


def load_catalog(catalog_path: Path = CATALOGO_FILE) -> list[dict[str, Any]]:
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Catalogo de solucoes deve ser uma lista JSON.")
    return payload


def validate_solution(solution: dict[str, Any], schema: dict[str, Any] | None = None) -> None:
    validator = Draft202012Validator(schema or load_schema())
    errors = sorted(validator.iter_errors(solution), key=lambda error: list(error.path))
    if errors:
        first = errors[0]
        path = ".".join(str(part) for part in first.path) or "<raiz>"
        raise ValueError(f"Solucao piloto invalida em {path}: {first.message}")


def validate_catalog(catalog: list[dict[str, Any]], schema: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    schema = schema or load_schema()
    seen_ids: set[str] = set()
    for item in catalog:
        validate_solution(item, schema)
        solution_id = str(item["id"])
        if solution_id in seen_ids:
            raise ValueError(f"ID de solucao duplicado: {solution_id}")
        seen_ids.add(solution_id)
    return catalog


def _solution_text(solution: dict[str, Any]) -> str:
    conformidade = solution.get("conformidade", {})
    tecnologias = ", ".join(solution.get("tecnologias", []))
    dependencias = ", ".join(conformidade.get("dependencias_open_source", []))
    return "\n".join(
        [
            f"Nome: {solution['nome']}",
            f"Descricao: {solution['descricao']}",
            f"Orgao: {solution['orgao']}",
            f"Area: {solution['area']}",
            f"Problema: {solution['problema']}",
            f"Sintomas: {', '.join(solution.get('sintomas', []))}",
            f"Causa raiz: {solution.get('causa_raiz', '')}",
            f"Pre-condicoes: {', '.join(solution.get('pre_condicoes', []))}",
            f"Solucao: {solution['solucao']}",
            f"Passos de implantacao: {', '.join(solution.get('passos_implantacao', []))}",
            f"Riscos: {', '.join(solution.get('riscos', []))}",
            f"Tecnologias: {tecnologias}",
            f"Status: {solution['status']}",
            f"Dados pessoais: {conformidade.get('dados_pessoais')}",
            f"Base legal LGPD: {conformidade.get('base_legal_lgpd')}",
            f"Hospedagem: {conformidade.get('hospedagem')}",
            f"Modelo: {conformidade.get('modelo')}",
            f"Licenca do modelo: {conformidade.get('licenca_modelo')}",
            f"Dependencias open-source: {dependencias}",
        ]
    )


def catalog_to_chunks(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for idx, solution in enumerate(catalog):
        conformidade = solution["conformidade"]
        chunks.append(
            {
                "id": idx,
                "solution_id": solution["id"],
                "documento": f"catalogo/solucoes_piloto.json#{solution['id']}",
                "pagina": None,
                "area": solution["area"],
                "assuntos": sorted(set([solution["area"], solution["status"], *solution["tecnologias"]])),
                "content_type": "solucao_piloto",
                "doc_titulo": solution["nome"],
                "doc_autores": [solution["orgao"]],
                "doc_ano": None,
                "section_heading": "Catalogo de solucoes piloto",
                "chunk_index": idx,
                "texto": _solution_text(solution),
                "conformidade": conformidade,
            }
        )
    return chunks


def materialize_solution_chunks(
    catalog_path: Path = CATALOGO_FILE,
    output_path: Path = OUTPUT_CHUNKS_FILE,
) -> dict[str, Any]:
    catalog = validate_catalog(load_catalog(catalog_path))
    chunks = catalog_to_chunks(catalog)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "total_solucoes": len(catalog),
        "total_chunks": len(chunks),
        "output_path": str(output_path),
    }


if __name__ == "__main__":
    result = materialize_solution_chunks()
    print(json.dumps(result, ensure_ascii=False, indent=2))
