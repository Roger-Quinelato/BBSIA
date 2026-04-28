import json
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
import sys

# Adiciona o diretório raiz ao path para importar pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import answer_question
from generator import DEFAULT_LLM_MODEL, OLLAMA_URL
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# Configuração para usar Ollama local com RAGAS
# Nota: RAGAS usa LLMs do LangChain para avaliar as métricas
llm = ChatOllama(model=DEFAULT_LLM_MODEL, base_url=OLLAMA_URL)

# Configuração do modelo de embeddings (o mesmo usado no projeto)
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cpu"}
)

def run_benchmark(dataset_path: str):
    if not os.path.exists(dataset_path):
        print(f"Erro: {dataset_path} não encontrado. Gere o dataset primeiro com generate_eval.py.")
        return

    print(f"Carregando dataset de: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        eval_samples = json.load(f)

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    print(f"Executando RAG para {len(eval_samples)} perguntas...")
    for i, sample in enumerate(eval_samples):
        question = sample["question"]
        print(f"[{i+1}/{len(eval_samples)}] Questão: {question}")
        
        # Chama o pipeline real do BBSIA
        result = answer_question(question)
        
        data["question"].append(question)
        data["answer"].append(result["resposta"])
        
        # Ragas espera lista de strings para contextos (chunks recuperados)
        contexts = [res.get("texto", "") for res in result.get("resultados", [])]
        data["contexts"].append(contexts)
        
        # O ground_truth é a resposta esperada gerada pelo LLM no script de criação do dataset
        data["ground_truth"].append(sample["ground_truth"])

    # Converte para Dataset do HuggingFace (necessário para Ragas)
    dataset = Dataset.from_dict(data)

    print("\nIniciando avaliação com RAGAS (pode demorar dependendo da velocidade do Ollama)...")
    
    # Executa a avaliação
    # As métricas serão calculadas usando o Ollama local
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

    print("\n" + "="*30)
    print("RESULTADOS DO BENCHMARK")
    print("="*30)
    df = result.to_pandas()
    
    # Exibe as médias das métricas
    summary = df.mean(numeric_only=True)
    print(summary)
    
    # Salva os resultados detalhados
    output_csv = "benchmarks/results.csv"
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\nResultados detalhados salvos em: {output_csv}")
    
    # Salva um resumo em JSON
    with open("benchmarks/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)

if __name__ == "__main__":
    run_benchmark("benchmarks/eval_dataset.json")
