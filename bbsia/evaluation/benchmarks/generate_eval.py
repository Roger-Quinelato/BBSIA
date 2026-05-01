import json
import random
import os
import requests
from typing import List, Dict

# Configurações do Ollama (reutilizando lógica do generator.py)
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:7b-instruct"

def query_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3}
        }
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()

def generate_eval_dataset(chunks_path: str, output_path: str, num_samples: int = 10):
    if not os.path.exists(chunks_path):
        print(f"Erro: {chunks_path} não encontrado.")
        return

    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # Filtrar chunks que tenham texto relevante (mais de 100 caracteres)
    useful_chunks = [c for c in chunks if len(c.get("texto", "")) > 200]
    
    if not useful_chunks:
        print("Nenhum chunk útil encontrado para gerar perguntas.")
        return

    selected_chunks = random.sample(useful_chunks, min(num_samples, len(useful_chunks)))
    dataset = []

    print(f"Gerando {len(selected_chunks)} amostras de avaliação...")

    for i, chunk in enumerate(selected_chunks):
        texto = chunk["texto"]
        doc = chunk["documento"]
        
        prompt = f"""
Com base no texto abaixo, extraído do documento '{doc}', crie:
1. Uma pergunta direta cuja resposta esteja contida no texto.
2. A resposta correta e completa baseada apenas no texto.

TEXTO:
{texto}

Responda APENAS no formato JSON abaixo, sem explicações:
{{
  "question": "sua pergunta aqui",
  "ground_truth": "sua resposta aqui"
}}
"""
        try:
            raw_response = query_ollama(prompt)
            # Tentar extrair JSON da resposta (às vezes o LLM coloca markdown)
            json_str = raw_response
            if "```json" in raw_response:
                json_str = raw_response.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_response:
                json_str = raw_response.split("```")[1].split("```")[0].strip()
            
            sample = json.loads(json_str)
            sample["context"] = [texto]  # O contexto original do qual a pergunta foi gerada
            sample["metadata"] = {
                "documento": doc,
                "pagina": chunk.get("pagina"),
                "chunk_id": chunk.get("id")
            }
            dataset.append(sample)
            print(f"[{i+1}/{len(selected_chunks)}] Pergunta gerada com sucesso.")
        except Exception as e:
            print(f"[{i+1}/{len(selected_chunks)}] Erro ao gerar amostra: {e}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset de avaliação salvo em: {output_path}")

if __name__ == "__main__":
    generate_eval_dataset("data/chunks.json", "bbsia/evaluation/benchmarks/eval_dataset.json", num_samples=15)
