import json
from pathlib import Path

from datasets import Dataset
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from bbsia.rag.generation.generator import DEFAULT_LLM_MODEL, OLLAMA_URL
from bbsia.rag.orchestration.pipeline import answer_question

llm = ChatOllama(model=DEFAULT_LLM_MODEL, base_url=OLLAMA_URL)
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cpu"},
)


def run_benchmark(dataset_path: str):
    if not Path(dataset_path).exists():
        print(f"Erro: {dataset_path} nao encontrado. Gere o dataset primeiro com generate_eval.py.")
        return

    print(f"Carregando dataset de: {dataset_path}")
    with open(dataset_path, encoding="utf-8") as f:
        eval_samples = json.load(f)

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    print(f"Executando RAG para {len(eval_samples)} perguntas...")
    for i, sample in enumerate(eval_samples):
        question = sample["question"]
        print(f"[{i + 1}/{len(eval_samples)}] Questao: {question}")

        result = answer_question(question)

        data["question"].append(question)
        data["answer"].append(result["resposta"])
        data["contexts"].append([res.get("texto", "") for res in result.get("resultados", [])])
        data["ground_truth"].append(sample["ground_truth"])

    dataset = Dataset.from_dict(data)

    print("\nIniciando avaliacao com RAGAS. Isso pode demorar dependendo da velocidade do Ollama.")
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=llm,
        embeddings=embeddings,
    )

    print("\n" + "=" * 30)
    print("RESULTADOS DO BENCHMARK")
    print("=" * 30)
    df = result.to_pandas()

    summary = df.mean(numeric_only=True)
    print(summary)

    output_csv = Path("bbsia/evaluation/benchmarks/results/results.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\nResultados detalhados salvos em: {output_csv}")

    summary_path = Path("bbsia/evaluation/benchmarks/results/summary.json")
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    run_benchmark("bbsia/evaluation/benchmarks/eval_dataset.json")
