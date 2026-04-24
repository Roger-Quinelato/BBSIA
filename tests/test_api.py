from __future__ import annotations

import requests
from fastapi.testclient import TestClient

import api
import rag_engine


def _sample_chunk() -> dict:
    return {
        "score": 0.91,
        "id": 88,
        "documento": "LIIA BBSIA - Infra-estrutura.pdf",
        "pagina": 3,
        "area": "infraestrutura",
        "assuntos": ["servidores", "kubernetes"],
        "doc_titulo": "Infraestrutura BBSIA",
        "doc_autores": ["Equipe Técnica"],
        "doc_ano": 2026,
        "texto": "Trecho de teste",
        "chunk_index": 2,
    }


def _build_client(monkeypatch, tmp_path) -> TestClient:
    uploads_dir = tmp_path / "uploads"
    metadata_file = uploads_dir / "metadata_uploads.json"
    monkeypatch.setattr(api, "UPLOADS_DIR", uploads_dir)
    monkeypatch.setattr(api, "UPLOAD_METADATA_FILE", metadata_file)

    monkeypatch.setattr(rag_engine, "_load_resources", lambda: {"chunks": [{"id": 1}, {"id": 2}]})
    monkeypatch.setattr(api, "list_available_areas", lambda: ["arquitetura", "infraestrutura"])
    monkeypatch.setattr(api, "list_available_assuntos", lambda: ["RAG", "kubernetes"])
    monkeypatch.setattr(api, "_check_ollama", lambda: (True, ["llama3.1:8b"]))
    monkeypatch.setattr(api, "search", lambda query, top_k, filtro_area, filtro_assunto: [_sample_chunk()])
    monkeypatch.setattr(
        api,
        "answer_question",
        lambda pergunta, model, top_k, filtro_area, filtro_assunto: {
            "resposta": "Resposta de teste",
            "fontes": ["LIIA BBSIA - Infra-estrutura.pdf (p. 3)"],
            "resultados": [_sample_chunk()],
            "prompt": "prompt",
        },
    )

    return TestClient(api.app)


def test_status_endpoint(monkeypatch, tmp_path):
    with _build_client(monkeypatch, tmp_path) as client:
        response = client.get("/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["indice_carregado"] is True
    assert body["ollama_online"] is True


def test_search_endpoint(monkeypatch, tmp_path):
    with _build_client(monkeypatch, tmp_path) as client:
        response = client.post(
            "/search",
            json={
                "query": "infraestrutura kubernetes",
                "top_k": 3,
                "filtro_area": ["infraestrutura"],
                "filtro_assunto": [],
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["resultados"][0]["area"] == "infraestrutura"


def test_chat_fallback_when_ollama_fails(monkeypatch, tmp_path):
    with _build_client(monkeypatch, tmp_path) as client:
        monkeypatch.setattr(api, "answer_question", lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout("timeout")))

        response = client.post(
            "/chat",
            json={
                "pergunta": "Quais requisitos de infraestrutura?",
                "modelo": "llama3.1:8b",
                "top_k": 3,
                "filtro_area": ["infraestrutura"],
                "filtro_assunto": [],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert "Recuperação concluída" in body["resposta"]
    assert body["total_chunks_recuperados"] == 1


def test_reprocessar_retorna_iniciado(monkeypatch, tmp_path):
    """POST /reprocessar deve retornar status 'iniciado' (execução em background)."""
    with _build_client(monkeypatch, tmp_path) as client:
        # Mocka as funções do pipeline para não executarem nada.
        monkeypatch.setattr(api, "run_extraction", lambda: None)
        monkeypatch.setattr(api, "run_chunking", lambda: None)
        monkeypatch.setattr(api, "run_embedding", lambda: None)
        monkeypatch.setattr(api, "reload_resources", lambda: None)
        # Garante que não há reprocessamento em andamento.
        api._reprocessamento_status["rodando"] = False

        response = client.post("/reprocessar")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "iniciado"
    assert "background" in body["mensagem"].lower()


def test_reprocessar_retorna_409_quando_em_andamento(monkeypatch, tmp_path):
    """POST /reprocessar deve retornar 409 se já houver reprocessamento rodando."""
    with _build_client(monkeypatch, tmp_path) as client:
        # Simula reprocessamento em andamento.
        api._reprocessamento_status["rodando"] = True

        response = client.post("/reprocessar")

    assert response.status_code == 409
    assert "andamento" in response.json()["detail"].lower()

    # Limpa o estado para não afetar outros testes.
    api._reprocessamento_status["rodando"] = False


def test_upload_rejeita_arquivo_maior_que_limite(monkeypatch, tmp_path):
    """POST /upload deve retornar 413 para arquivos acima de MAX_UPLOAD_SIZE_MB."""
    # Define limite de 1 MB para facilitar o teste.
    monkeypatch.setattr(api, "MAX_UPLOAD_SIZE_MB", 1)

    with _build_client(monkeypatch, tmp_path) as client:
        # Cria um conteúdo de ~1.5 MB (acima do limite de 1 MB).
        big_content = b"%PDF-1.4 fake" + b"X" * (1_500_000)

        response = client.post(
            "/upload",
            files=[("files", ("grande.pdf", big_content, "application/pdf"))],
        )

    assert response.status_code == 413
    body = response.json()
    assert "grande.pdf" in body["detail"]
    assert "1 MB" in body["detail"]


def test_upload_aceita_arquivo_dentro_do_limite(monkeypatch, tmp_path):
    """POST /upload deve aceitar arquivos dentro do limite."""
    monkeypatch.setattr(api, "MAX_UPLOAD_SIZE_MB", 10)

    with _build_client(monkeypatch, tmp_path) as client:
        small_content = b"%PDF-1.4 fake content"

        response = client.post(
            "/upload",
            files=[("files", ("pequeno.pdf", small_content, "application/pdf"))],
        )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert "pequeno.pdf" in body["arquivos_salvos"]


def test_status_inclui_reprocessamento(monkeypatch, tmp_path):
    """GET /status deve incluir o campo 'reprocessamento'."""
    with _build_client(monkeypatch, tmp_path) as client:
        response = client.get("/status")

    assert response.status_code == 200
    body = response.json()
    assert "reprocessamento" in body
    assert "rodando" in body["reprocessamento"]


def test_get_biblioteca_retorna_lista(monkeypatch, tmp_path):
    """GET /biblioteca deve retornar documentos filtrados ou completos."""
    with _build_client(monkeypatch, tmp_path) as client:
        # Mock classificador_artigo.carregar_biblioteca
        fake_bib = {
            "documentos": [
                {"id": "doc1", "titulo": "Adoção de IA", "area_tematica": "ia", "tipo_documento": "artigo"},
                {"id": "doc2", "titulo": "Manual de Infra", "area_tematica": "infra", "tipo_documento": "manual"}
            ]
        }
        monkeypatch.setattr("api.carregar_biblioteca", lambda: fake_bib)

        # Testa sem filtros
        resp_all = client.get("/biblioteca")
        assert resp_all.status_code == 200
        assert resp_all.json()["total"] == 2

        # Testa com filtro
        resp_filtered = client.get("/biblioteca?area=ia")
        assert resp_filtered.status_code == 200
        assert resp_filtered.json()["total"] == 1
        assert resp_filtered.json()["documentos"][0]["id"] == "doc1"


def test_get_filtros_retorna_areas_unicas(monkeypatch, tmp_path):
    """GET /filtros deve retornar listas ordenadas de valores únicos."""
    with _build_client(monkeypatch, tmp_path) as client:
        fake_bib = {
            "documentos": [
                {"id": "doc1", "area_tematica": "ia", "ano": 2022, "assuntos": ["etica"]},
                {"id": "doc2", "area_tematica": "ia", "ano": 2023, "assuntos": ["lgpd"]},
                {"id": "doc3", "area_tematica": "infra", "ano": 2022, "assuntos": ["kubernetes", "etica"]}
            ]
        }
        monkeypatch.setattr("api.carregar_biblioteca", lambda: fake_bib)

        response = client.get("/filtros")
        assert response.status_code == 200
        data = response.json()
        assert data["areas"] == ["ia", "infra"]
        assert data["anos"] == [2022, 2023]
        assert "etica" in data["assuntos"]
        assert "lgpd" in data["assuntos"]


def test_chunks_contem_autoria_no_search(monkeypatch, tmp_path):
    """Verifica se os campos de autoria (doc_titulo, doc_autores, doc_ano) são retornados pela API."""
    with _build_client(monkeypatch, tmp_path) as client:
        response = client.post(
            "/search",
            json={
                "query": "qualquer coisa",
                "top_k": 1
            },
        )
        assert response.status_code == 200
        body = response.json()
        chunk = body["resultados"][0]
        
        assert "doc_titulo" in chunk
        assert "doc_autores" in chunk
        assert "doc_ano" in chunk
        assert chunk["doc_titulo"] == "Infraestrutura BBSIA"
        assert chunk["doc_autores"] == ["Equipe Técnica"]
        assert chunk["doc_ano"] == 2026
