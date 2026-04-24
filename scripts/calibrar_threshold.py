"""
Script de calibração do threshold MIN_DENSE_SCORE_FOR_ANSWER.

Executa queries representativas contra a base vetorial e coleta métricas
de score para sugerir um valor calibrado do threshold.

Uso:
    python scripts/calibrar_threshold.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys

# Garante que o diretório raiz do projeto esteja no path.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from rag_engine import search  # noqa: E402


QUERIES_DE_TESTE = [
    # Área: ia
    {"query": "O que é o BBSIA?", "area_esperada": "ia"},
    {"query": "Qual é o objetivo do MVP do Banco Brasileiro de Soluções de IA?", "area_esperada": "ia"},
    {"query": "Como funciona o pipeline RAG do chatbot?", "area_esperada": "ia"},
    {"query": "Quais modelos de linguagem são usados no projeto?", "area_esperada": "ia"},
    # Área: infraestrutura
    {"query": "Quais são os requisitos de infraestrutura do projeto?", "area_esperada": "infraestrutura"},
    {"query": "O projeto utiliza Kubernetes ou containers?", "area_esperada": "infraestrutura"},
    # Área: juridico
    {"query": "Como o framework de ética em IA aborda a LGPD?", "area_esperada": "juridico"},
    {"query": "Quais são os princípios éticos de IA do projeto?", "area_esperada": "juridico"},
    # Área: tecnologia
    {"query": "O que é o InovaLabs e qual sua metodologia?", "area_esperada": "tecnologia"},
    {"query": "Como funcionam as oficinas de inovação do LIIA?", "area_esperada": "tecnologia"},
    # Área: saude
    {"query": "Qual a relação entre IA e saúde no projeto?", "area_esperada": "saude"},
    # Perguntas genéricas / cross-area
    {"query": "O que é o Framework de Prontidão em IA?", "area_esperada": "ia"},
    {"query": "Quais são as fases do projeto BBSIA?", "area_esperada": "ia"},
    {"query": "Como é feita a avaliação de maturidade em IA?", "area_esperada": "ia"},
    # Pergunta fora do escopo (deve gerar score baixo)
    {"query": "Qual a receita de bolo de chocolate?", "area_esperada": "nenhuma"},
]


def run_calibration() -> None:
    print("=" * 70)
    print("  CALIBRAÇÃO DE THRESHOLD — MIN_DENSE_SCORE_FOR_ANSWER")
    print("=" * 70)
    print(f"\n  Total de queries de teste: {len(QUERIES_DE_TESTE)}\n")

    resultados: list[dict] = []
    dense_scores: list[float] = []
    sparse_scores: list[float] = []

    for i, item in enumerate(QUERIES_DE_TESTE, start=1):
        query = item["query"]
        area_esperada = item["area_esperada"]

        try:
            results = search(query=query, top_k=5)
        except Exception as exc:
            print(f"  [{i:02d}] ERRO ao buscar: {exc}")
            resultados.append({
                "query": query,
                "area_esperada": area_esperada,
                "erro": str(exc),
            })
            continue

        if not results:
            print(f"  [{i:02d}] Nenhum resultado para: {query[:50]}...")
            resultados.append({
                "query": query,
                "area_esperada": area_esperada,
                "sem_resultados": True,
            })
            continue

        top = results[0]
        score_dense = float(top.get("score_dense", 0.0))
        score_sparse = float(top.get("score_sparse", 0.0))
        score_final = float(top.get("score", 0.0))
        documento = top.get("documento", "?")
        pagina = top.get("pagina", "?")
        area_retornada = top.get("area", "?")

        dense_scores.append(score_dense)
        sparse_scores.append(score_sparse)

        match = "✓" if area_retornada == area_esperada else "✗"

        registro = {
            "query": query,
            "area_esperada": area_esperada,
            "area_retornada": area_retornada,
            "area_match": area_retornada == area_esperada,
            "score_dense": round(score_dense, 4),
            "score_sparse": round(score_sparse, 4),
            "score_final": round(score_final, 4),
            "documento": documento,
            "pagina": pagina,
            "total_resultados": len(results),
        }
        resultados.append(registro)

        print(
            f"  [{i:02d}] dense={score_dense:.4f}  sparse={score_sparse:.4f}  "
            f"final={score_final:.4f}  area={area_retornada} {match}  "
            f"doc={documento[:40]}"
        )

    # --- Estatísticas ---
    print(f"\n{'=' * 70}")
    print("  ESTATÍSTICAS DOS SCORES DENSE")
    print(f"{'=' * 70}")

    if dense_scores:
        dense_sorted = sorted(dense_scores)
        media = statistics.mean(dense_scores)
        mediana = statistics.median(dense_scores)
        minimo = min(dense_scores)
        maximo = max(dense_scores)
        desvio = statistics.stdev(dense_scores) if len(dense_scores) > 1 else 0.0

        # Percentil 25 (Q1)
        n = len(dense_sorted)
        p25_idx = max(0, int(n * 0.25) - 1)
        percentil_25 = dense_sorted[p25_idx]

        # Percentil 10 (mais conservador)
        p10_idx = max(0, int(n * 0.10) - 1)
        percentil_10 = dense_sorted[p10_idx]

        print(f"  Amostras    : {len(dense_scores)}")
        print(f"  Mínimo      : {minimo:.4f}")
        print(f"  Percentil 10: {percentil_10:.4f}")
        print(f"  Percentil 25: {percentil_25:.4f}")
        print(f"  Mediana     : {mediana:.4f}")
        print(f"  Média       : {media:.4f}")
        print(f"  Máximo      : {maximo:.4f}")
        print(f"  Desvio pad. : {desvio:.4f}")

        # Sugestão de threshold
        sugestao_percent = max(1, int(percentil_25 * 100))
        print(f"\n{'=' * 70}")
        print("  SUGESTÃO DE THRESHOLD")
        print(f"{'=' * 70}")
        print(f"  Baseado no percentil 25 dos scores dense observados:")
        print(f"  → MIN_DENSE_SCORE_PERCENT = {sugestao_percent}")
        print(f"    (equivale a {percentil_25:.4f} em escala 0–1)")
        print(f"\n  Para aplicar, adicione no .env:")
        print(f"    MIN_DENSE_SCORE_PERCENT={sugestao_percent}")

        estatisticas = {
            "total_queries": len(QUERIES_DE_TESTE),
            "queries_com_resultado": len(dense_scores),
            "dense_min": round(minimo, 4),
            "dense_p10": round(percentil_10, 4),
            "dense_p25": round(percentil_25, 4),
            "dense_mediana": round(mediana, 4),
            "dense_media": round(media, 4),
            "dense_max": round(maximo, 4),
            "dense_desvio": round(desvio, 4),
            "threshold_sugerido_percent": sugestao_percent,
            "threshold_sugerido_float": round(percentil_25, 4),
        }
    else:
        print("  Nenhum score dense coletado.")
        estatisticas = {"total_queries": len(QUERIES_DE_TESTE), "queries_com_resultado": 0}

    # --- Salvar resultados ---
    output_path = os.path.join(SCRIPT_DIR, "resultados_calibracao.json")
    payload = {
        "estatisticas": estatisticas,
        "resultados": resultados,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n  Resultados salvos em: {output_path}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    run_calibration()
