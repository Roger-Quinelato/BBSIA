from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
sys.path.insert(0, str(PROJECT_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception as exc:
        print(f"[aviso] Nao foi possivel ler {path}: {exc}")
        return default


def _snippet(text: str, max_chars: int = 420) -> str:
    cleaned = " ".join((text or "").split())
    return cleaned if len(cleaned) <= max_chars else cleaned[:max_chars].rstrip() + "..."


def _document_inventory() -> list[dict[str, Any]]:
    chunks = _load_json(DATA_DIR / "chunks.json", [])
    if not isinstance(chunks, list):
        chunks = []

    docs: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "documento": "",
            "chunks": 0,
            "paginas": set(),
            "areas": Counter(),
            "assuntos": Counter(),
            "titulo": "",
            "ano": None,
        }
    )

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        documento = str(chunk.get("documento") or "desconhecido")
        item = docs[documento]
        item["documento"] = documento
        item["chunks"] += 1
        if chunk.get("pagina") is not None:
            item["paginas"].add(chunk.get("pagina"))
        area = str(chunk.get("area") or "geral")
        item["areas"][area] += 1
        for assunto in chunk.get("assuntos") or []:
            if isinstance(assunto, str) and assunto.strip():
                item["assuntos"][assunto.strip()] += 1
        if not item["titulo"] and chunk.get("doc_titulo"):
            item["titulo"] = str(chunk.get("doc_titulo"))
        if item["ano"] is None and chunk.get("doc_ano"):
            item["ano"] = chunk.get("doc_ano")

    inventory: list[dict[str, Any]] = []
    for item in docs.values():
        paginas = sorted(item["paginas"])
        inventory.append(
            {
                "documento": item["documento"],
                "chunks": item["chunks"],
                "paginas": len(paginas),
                "primeira_pagina": paginas[0] if paginas else None,
                "ultima_pagina": paginas[-1] if paginas else None,
                "areas": [area for area, _ in item["areas"].most_common(3)],
                "assuntos": [assunto for assunto, _ in item["assuntos"].most_common(5)],
                "titulo": item["titulo"],
                "ano": item["ano"],
            }
        )

    return sorted(inventory, key=lambda row: row["documento"].lower())


def _solution_inventory() -> list[dict[str, Any]]:
    chunks = _load_json(DATA_DIR / "solucoes_piloto_chunks.json", [])
    if not isinstance(chunks, list):
        return []

    docs: dict[str, dict[str, Any]] = defaultdict(lambda: {"documento": "", "chunks": 0, "titulo": ""})
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        documento = str(chunk.get("documento") or "bbsia/domain/catalogo/data/solucoes_piloto.json")
        docs[documento]["documento"] = documento
        docs[documento]["chunks"] += 1
        if not docs[documento]["titulo"]:
            docs[documento]["titulo"] = str(chunk.get("doc_titulo") or "")
    return sorted(docs.values(), key=lambda row: row["documento"].lower())


def print_inventory(limit: int = 50) -> None:
    docs = _document_inventory()
    solutions = _solution_inventory()
    total_chunks = sum(int(doc["chunks"]) for doc in docs)

    print("\nArquivos ja consumidos/indexados")
    print("=" * 88)
    print(f"Documentos: {len(docs)} | chunks: {total_chunks}")
    if solutions:
        print(f"Solucoes do catalogo: {len(solutions)} arquivo(s) | chunks: {sum(s['chunks'] for s in solutions)}")
    print()

    if not docs:
        print("Nenhum documento encontrado em data/chunks.json.")
        return

    for idx, doc in enumerate(docs[:limit], start=1):
        page_range = "-"
        if doc["primeira_pagina"] is not None:
            page_range = f"{doc['primeira_pagina']}-{doc['ultima_pagina']}"
        titulo = f" | {doc['titulo']}" if doc["titulo"] else ""
        ano = f" ({doc['ano']})" if doc["ano"] else ""
        print(f"{idx:02d}. {doc['documento']}{ano}")
        print(f"    chunks={doc['chunks']} | paginas={doc['paginas']} [{page_range}] | areas={', '.join(doc['areas']) or '-'}")
        if doc["assuntos"]:
            print(f"    assuntos={', '.join(doc['assuntos'])}")
        if titulo:
            print(f"    titulo={doc['titulo']}")

    if len(docs) > limit:
        print(f"\n... {len(docs) - limit} documento(s) ocultos. Use :docs --todos para listar tudo.")


def print_answer(payload: dict[str, Any], show_chunks: bool) -> None:
    print("\nResposta")
    print("-" * 88)
    print(payload.get("resposta", ""))

    fontes = payload.get("fontes") or []
    if fontes:
        print("\nFontes citadas")
        for fonte in fontes:
            print(f"- {fonte}")

    resultados = payload.get("resultados") or []
    if resultados:
        print("\nTrechos recuperados nesta pergunta")
        for idx, item in enumerate(resultados, start=1):
            print(
                f"{idx}. {item.get('documento')} | p. {item.get('pagina')} | "
                f"score={float(item.get('score', 0.0)):.4f} | area={item.get('area')}"
            )
            if show_chunks:
                print(f"   {_snippet(str(item.get('parent_text') or item.get('texto') or ''))}")


def run_search(pergunta: str, top_k: int, area: str, assunto: str, show_chunks: bool) -> None:
    from bbsia.rag.retrieval.retriever import search

    results = search(
        query=pergunta,
        top_k=top_k,
        filtro_area=[area] if area else [],
        filtro_assunto=[assunto] if assunto else [],
    )
    print(f"\nResultados recuperados: {len(results)}")
    for idx, item in enumerate(results, start=1):
        print(
            f"{idx}. {item.get('documento')} | p. {item.get('pagina')} | "
            f"score={float(item.get('score', 0.0)):.4f} "
            f"dense={float(item.get('score_dense', 0.0)):.4f} "
            f"sparse={float(item.get('score_sparse', 0.0)):.4f}"
        )
        if show_chunks:
            print(f"   {_snippet(str(item.get('parent_text') or item.get('texto') or ''))}")


def run_answer(pergunta: str, model: str, top_k: int, area: str, assunto: str, show_chunks: bool) -> None:
    from bbsia.rag.orchestration.pipeline import answer_question

    payload = answer_question(
        pergunta=pergunta,
        model=model,
        top_k=top_k,
        filtro_area=[area] if area else [],
        filtro_assunto=[assunto] if assunto else [],
    )
    print_answer(payload, show_chunks=show_chunks)


def print_help() -> None:
    print(
        """
Comandos:
  :docs              lista os arquivos ja consumidos/indexados
  :docs --todos      lista todos os arquivos, sem limite
  :buscar pergunta   mostra apenas os trechos recuperados, sem chamar o LLM
  :ajuda             mostra esta ajuda
  :sair              encerra

Qualquer outro texto e enviado ao modelo/RAG como pergunta.
""".strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat interativo do BBSIA para PowerShell.")
    parser.add_argument("--modelo", default=os.getenv("DEFAULT_MODEL", "qwen3.5:7b-instruct"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--area", default="")
    parser.add_argument("--assunto", default="")
    parser.add_argument("--sem-lista", action="store_true", help="Nao lista documentos indexados ao iniciar.")
    parser.add_argument("--mostrar-trechos", action="store_true", help="Mostra o texto dos trechos recuperados.")
    parser.add_argument("--sem-faithfulness", action="store_true", help="Desliga checagem de fidelidade sincrona.")
    args = parser.parse_args()

    if args.sem_faithfulness:
        os.environ["ENABLE_SYNC_FAITHFULNESS"] = "false"

    if not args.sem_lista:
        print_inventory()

    print("\nChat BBSIA")
    print("=" * 88)
    print(f"Modelo: {args.modelo} | top_k={args.top_k}")
    if args.area or args.assunto:
        print(f"Filtros: area={args.area or '-'} | assunto={args.assunto or '-'}")
    print("Digite :ajuda para comandos ou :sair para encerrar.\n")

    while True:
        try:
            pergunta = input("bbsia> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not pergunta:
            continue
        lowered = pergunta.lower()
        if lowered in {":sair", "sair", "exit", "quit"}:
            return
        if lowered in {":ajuda", "ajuda", "help", ":help"}:
            print_help()
            continue
        if lowered.startswith(":docs"):
            print_inventory(limit=10_000 if "--todos" in lowered else 50)
            continue
        if lowered.startswith(":buscar"):
            query = pergunta[len(":buscar") :].strip()
            if not query:
                print("Informe uma pergunta depois de :buscar.")
                continue
            run_search(query, args.top_k, args.area, args.assunto, args.mostrar_trechos)
            continue

        run_answer(pergunta, args.modelo, args.top_k, args.area, args.assunto, args.mostrar_trechos)
        print()


if __name__ == "__main__":
    main()
