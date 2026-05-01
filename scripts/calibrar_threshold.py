"""
Script de calibracao do threshold MIN_DENSE_SCORE_FOR_ANSWER.

Executa queries representativas contra a base vetorial, mede qualidade do
retrieval e sugere um valor para MIN_DENSE_SCORE_PERCENT.

Uso:
    python scripts/calibrar_threshold.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Garante que o diretorio raiz do projeto esteja no path.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from bbsia.rag.retrieval.retriever import calibrate_dense_threshold  # noqa: E402

DEFAULT_OUTPUT_PATH = Path(PROJECT_DIR) / "benchmarks" / "results" / "threshold_calibration_latest.json"
CURRENT_THRESHOLD_PERCENT = int(os.getenv("MIN_DENSE_SCORE_PERCENT", "18"))

QUERIES_DE_TESTE = [
    {"query": "O que e o BBSIA?", "area_esperada": "ia"},
    {"query": "Qual e o objetivo do MVP do Banco Brasileiro de Solucoes de IA?", "area_esperada": "ia"},
    {"query": "Como funciona o pipeline RAG do chatbot?", "area_esperada": "ia"},
    {"query": "Quais modelos de linguagem sao usados no projeto?", "area_esperada": "ia"},
    {"query": "Quais sao os requisitos de infraestrutura do projeto?", "area_esperada": "infraestrutura"},
    {"query": "O projeto utiliza Kubernetes ou containers?", "area_esperada": "infraestrutura"},
    {"query": "Como o framework de etica em IA aborda a LGPD?", "area_esperada": "juridico"},
    {"query": "Quais sao os principios eticos de IA do projeto?", "area_esperada": "juridico"},
    {"query": "O que e o InovaLabs e qual sua metodologia?", "area_esperada": "tecnologia"},
    {"query": "Como funcionam as oficinas de inovacao do LIIA?", "area_esperada": "tecnologia"},
    {"query": "Qual a relacao entre IA e saude no projeto?", "area_esperada": "saude"},
    {"query": "O que e o Framework de Prontidao em IA?", "area_esperada": "ia"},
    {"query": "Quais sao as fases do projeto BBSIA?", "area_esperada": "ia"},
    {"query": "Como e feita a avaliacao de maturidade em IA?", "area_esperada": "ia"},
    {"query": "Qual a receita de bolo de chocolate?", "area_esperada": "nenhuma"},
]



def run_calibration(output_path: str | os.PathLike[str] = DEFAULT_OUTPUT_PATH) -> dict:
    print("=" * 70)
    print("  CALIBRACAO DE THRESHOLD - MIN_DENSE_SCORE_FOR_ANSWER")
    print("=" * 70)
    print(f"\n  Total de queries de teste: {len(QUERIES_DE_TESTE)}\n")

    payload = calibrate_dense_threshold(QUERIES_DE_TESTE, top_k=5)
    estatisticas = payload["estatisticas"]
    qualidade = payload["qualidade"]
    dense_informativo = bool(
        estatisticas.get("in_scope_count", 0)
        and (
            float(estatisticas.get("dense_max", 0.0)) > 0.0
            or float(estatisticas.get("out_scope_dense_max", 0.0)) > 0.0
        )
    )
    recomendacao_status = "acionavel" if dense_informativo else "nao_acionavel_dense_zero"
    decisao = (
        "avaliar ajuste manual com base no threshold_sugerido_percent"
        if dense_informativo
        else "manter threshold atual; amostra nao separa scores dense"
    )
    payload["calibracao"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/calibrar_threshold.py",
        "query_count": len(QUERIES_DE_TESTE),
        "top_k": 5,
        "threshold_atual_percent": CURRENT_THRESHOLD_PERCENT,
        "threshold_adotado_percent": CURRENT_THRESHOLD_PERCENT,
        "recomendacao_status": recomendacao_status,
        "decisao": decisao,
        "observacao": (
            "Este script registra evidencia e sugestao. "
            "A alteracao de MIN_DENSE_SCORE_PERCENT deve ser decisao documentada, nao automatica."
        ),
    }

    for i, item in enumerate(payload["resultados"], start=1):
        if item.get("erro"):
            print(f"  [{i:02d}] ERRO ao buscar: {item['erro']}")
            continue
        if not item.get("total_resultados"):
            print(f"  [{i:02d}] Nenhum resultado para: {item['query'][:50]}...")
            continue

        match = "-"
        if item.get("area_match") is not None:
            match = "OK" if item.get("area_match") else "FALHA"
        print(
            f"  [{i:02d}] dense={item['score_dense']:.4f}  "
            f"sparse={item['score_sparse']:.4f}  final={item['score_final']:.4f}  "
            f"area={item.get('area_retornada')} {match}  "
            f"doc={str(item.get('documento') or '?')[:40]}"
        )

    print(f"\n{'=' * 70}")
    print("  ESTATISTICAS DOS SCORES DENSE")
    print(f"{'=' * 70}")

    if estatisticas.get("in_scope_count", 0):
        print(f"  Amostras dentro do escopo : {estatisticas['in_scope_count']}")
        print(f"  Amostras fora do escopo   : {estatisticas['out_scope_count']}")
        print(f"  Minimo                   : {estatisticas['dense_min']:.4f}")
        print(f"  Percentil 10             : {estatisticas['dense_p10']:.4f}")
        print(f"  Percentil 25             : {estatisticas['dense_p25']:.4f}")
        print(f"  Mediana                  : {estatisticas['dense_mediana']:.4f}")
        print(f"  Media                    : {estatisticas['dense_media']:.4f}")
        print(f"  Maximo                   : {estatisticas['dense_max']:.4f}")
        print(f"  Maior fora do escopo     : {estatisticas['out_scope_dense_max']:.4f}")

        sugestao_percent = int(estatisticas["threshold_sugerido_percent"])
        print(f"\n{'=' * 70}")
        print("  SUGESTAO DE THRESHOLD")
        print(f"{'=' * 70}")
        print("  Baseado em queries esperadas dentro e fora do escopo:")
        print(f"  -> MIN_DENSE_SCORE_PERCENT = {sugestao_percent}")
        print(f"     (equivale a {estatisticas['threshold_sugerido_float']:.4f} em escala 0-1)")
        print(f"  Qualidade top-k: {qualidade['passed']}/{qualidade['total']} ({qualidade['pass_rate']:.0%})")
        print("\n  Para aplicar, adicione no .env:")
        print(f"    MIN_DENSE_SCORE_PERCENT={sugestao_percent}")
    else:
        print("  Nenhum score dense coletado.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  Resultados salvos em: {output_path}")
    print(f"{'=' * 70}\n")
    return payload


if __name__ == "__main__":
    run_calibration()
