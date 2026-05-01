from __future__ import annotations

import os
import re


def format_source_label(item: dict) -> str:
    """Gera rotulo academico: 'Sobrenome, Ano - Titulo' quando disponivel."""
    autores = item.get("doc_autores", [])
    ano = item.get("doc_ano")
    titulo = item.get("doc_titulo", "")
    documento = item.get("documento", "desconhecido")

    if autores and isinstance(autores, list) and autores[0]:
        primeiro_autor = autores[0].strip()
        partes_nome = primeiro_autor.split()
        sobrenome = partes_nome[-1] if partes_nome else primeiro_autor
        label = sobrenome
        if ano:
            label += f", {ano}"
        if titulo:
            label += f' \u2014 "{titulo}"'
        return label

    nome_base = os.path.splitext(os.path.basename(documento))[0]
    if ano:
        return f"{nome_base} ({ano})"
    return nome_base


def format_citation_label(item: dict) -> str:
    autores = item.get("doc_autores", [])
    ano = item.get("doc_ano") or "s.d."

    if autores and isinstance(autores, list) and autores[0]:
        primeiro_autor = str(autores[0]).strip()
        partes_nome = primeiro_autor.split()
        sobrenome = partes_nome[-1] if partes_nome else primeiro_autor
    else:
        documento = str(item.get("documento", "documento"))
        sobrenome = os.path.splitext(os.path.basename(documento))[0]

    sobrenome = re.sub(r"\s+", " ", sobrenome).strip() or "Documento"
    return f"{sobrenome}, {ano}"


_format_source_label = format_source_label
_format_citation_label = format_citation_label
