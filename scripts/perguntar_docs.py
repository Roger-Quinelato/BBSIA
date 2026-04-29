from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

def _snippet(text: str, max_chars: int = 900) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def _print_search(pergunta: str, top_k: int, filtro_area: list[str], filtro_assunto: list[str]) -> None:
    from retriever import search

    results = search(
        query=pergunta,
        top_k=top_k,
        filtro_area=filtro_area,
        filtro_assunto=filtro_assunto,
    )

    print(f"\nPergunta: {pergunta}")
    print(f"Resultados: {len(results)}")
    if not results:
        print("Nenhum trecho recuperado.")
        return

    for idx, item in enumerate(results, start=1):
        texto = str(item.get("parent_text") or item.get("texto") or "")
        assuntos = ", ".join(item.get("assuntos") or [])
        print("\n" + "-" * 88)
        print(
            f"#{idx} score={float(item.get('score', 0.0)):.4f} "
            f"dense={float(item.get('score_dense', 0.0)):.4f} "
            f"sparse={float(item.get('score_sparse', 0.0)):.4f}"
        )
        print(f"Documento: {item.get('documento')} | pagina: {item.get('pagina')} | area: {item.get('area')}")
        if assuntos:
            print(f"Assuntos: {assuntos}")
        print(f"Trecho: {_snippet(texto)}")


def _print_answer(pergunta: str, top_k: int, filtro_area: list[str], filtro_assunto: list[str]) -> None:
    from pipeline import answer_question

    payload = answer_question(
        pergunta=pergunta,
        top_k=top_k,
        filtro_area=filtro_area,
        filtro_assunto=filtro_assunto,
    )

    print(f"\nPergunta: {pergunta}")
    print("\nResposta:")
    print(payload.get("resposta", ""))
    fontes = payload.get("fontes", [])
    if fontes:
        print("\nFontes:")
        for fonte in fontes:
            print(f"- {fonte}")


def _run_once(args: argparse.Namespace, pergunta: str) -> None:
    filtro_area = [args.area] if args.area else []
    filtro_assunto = [args.assunto] if args.assunto else []
    if args.modo == "answer":
        _print_answer(pergunta, args.top_k, filtro_area, filtro_assunto)
    else:
        _print_search(pergunta, args.top_k, filtro_area, filtro_assunto)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pergunte aos documentos indexados do BBSIA pelo terminal.")
    parser.add_argument("--modo", choices=["search", "answer"], default="search", help="search mostra trechos; answer usa o LLM/fallback.")
    parser.add_argument("--pergunta", default="", help="Pergunta unica. Se omitida, abre modo interativo.")
    parser.add_argument("--top-k", type=int, default=3, help="Quantidade de resultados recuperados.")
    parser.add_argument("--area", default="", help="Filtro opcional de area, ex: ia, infraestrutura, juridico.")
    parser.add_argument("--assunto", default="", help="Filtro opcional de assunto, ex: lgpd, rag, kubernetes.")
    parser.add_argument(
        "--sem-faithfulness",
        action="store_true",
        help="Desliga a checagem NLI sincrona para testes locais de resposta.",
    )
    args = parser.parse_args()

    if args.sem_faithfulness:
        os.environ["ENABLE_SYNC_FAITHFULNESS"] = "false"

    if args.pergunta.strip():
        _run_once(args, args.pergunta.strip())
        return

    print("Pergunte aos documentos indexados do BBSIA.")
    print("Digite uma pergunta e pressione Enter. Use 'sair' para encerrar.")
    print(f"Modo atual: {args.modo} | top_k={args.top_k}\n")

    while True:
        try:
            pergunta = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not pergunta:
            continue
        if pergunta.lower() in {"sair", "exit", "quit"}:
            return
        _run_once(args, pergunta)
        print()


if __name__ == "__main__":
    main()
