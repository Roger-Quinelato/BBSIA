import os
import re

def update_api():
    path = "api.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    content = content.replace(
        "from fastapi.responses import JSONResponse, RedirectResponse",
        "from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse"
    )
    
    content = content.replace(
        "    answer_question,",
        "    answer_question,\n    answer_question_stream,"
    )

    old_chat = """@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        selected_model = validate_ollama_model(payload.modelo)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = answer_question(
            pergunta=payload.pergunta,
            model=selected_model,
            top_k=payload.top_k,
            filtro_area=payload.filtro_area,
            filtro_assunto=payload.filtro_assunto,
        )
        resultados = [_normalize_chunk(x) for x in result.get("resultados", [])]
        fontes = [str(v) for v in result.get("fontes", [])]
        resposta = str(result.get("resposta", "")).strip()
        if not resposta:
            resposta = "NÃ£o foi possÃ­vel gerar resposta no momento."

        return ChatResponse(
            resposta=resposta,
            fontes=fontes,
            resultados=resultados,
            modelo_usado=selected_model,
            total_fontes=len(fontes),
            total_chunks_recuperados=len(resultados),
        )
    except Exception as exc:
        # /chat precisa ser tolerante a falhas do LLM.
        try:
            resultados = search(
                query=payload.pergunta,
                top_k=payload.top_k,
                filtro_area=payload.filtro_area,
                filtro_assunto=payload.filtro_assunto,
            )
            chunks = [_normalize_chunk(x) for x in resultados]
            fontes = []
            seen = set()
            for item in chunks:
                src = f"{item['documento']} (p. {item['pagina']})"
                if src not in seen:
                    seen.add(src)
                    fontes.append(src)

            return ChatResponse(
                resposta=(
                    "RecuperaÃ§Ã£o concluÃ­da, mas o modelo de linguagem nÃ£o respondeu. "
                    f"Verifique o Ollama em {OLLAMA_URL} e tente novamente."
                ),
                fontes=fontes,
                resultados=chunks,
                modelo_usado=payload.modelo,
                total_fontes=len(fontes),
                total_chunks_recuperados=len(chunks),
            )
        except Exception as inner_exc:
            _raise_http_exception(inner_exc if not isinstance(exc, FileNotFoundError) else exc)
            raise  # pragma: no cover"""

    new_chat = """@app.post("/chat")
async def chat(payload: ChatRequest) -> StreamingResponse:
    try:
        selected_model = validate_ollama_model(payload.modelo)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def generate():
        try:
            async for chunk in answer_question_stream(
                pergunta=payload.pergunta,
                model=selected_model,
                top_k=payload.top_k,
                filtro_area=payload.filtro_area,
                filtro_assunto=payload.filtro_assunto,
            ):
                if chunk["type"] == "metadata":
                    chunk["resultados"] = [_normalize_chunk(x) for x in chunk.get("resultados", [])]
                    chunk["modelo_usado"] = selected_model
                yield json.dumps(chunk) + "\\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")"""
    
    # Also change /search to async def
    old_search = """@app.post("/search", response_model=SearchResponse)
def semantic_search(payload: SearchRequest) -> SearchResponse:"""
    new_search = """@app.post("/search", response_model=SearchResponse)
async def semantic_search(payload: SearchRequest) -> SearchResponse:"""

    content = content.replace(old_chat, new_chat)
    content = content.replace(old_search, new_search)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

update_api()
