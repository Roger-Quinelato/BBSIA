from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from bbsia.app.routers.biblioteca import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)
MOCK_BIBLIOTECA = {
    "documentos": [
        {
            "id": "doc_1",
            "titulo": "Manual de Infraestrutura",
            "autores": ["Alan Turing"],
            "ano": 2020,
            "area_tematica": "TI",
            "assuntos": ["redes", "hardware"],
            "tipo_documento": "manual",
            "paginas_total": 120,
            "conteudo_completo": "Este campo não deve aparecer na listagem resumida."
        },
        {
            "id": "doc_2",
            "titulo": "Relatório de Contratações",
            "autores": ["Ada Lovelace"],
            "ano": 2022,
            "area_tematica": "RH",
            "assuntos": ["recrutamento"],
            "tipo_documento": "relatorio",
            "paginas_total": 45
        },
        {
            "id": "doc_3",
            "titulo": "Política de Senhas Fortes",
            "autores": ["Grace Hopper"],
            "ano": 2023,
            "area_tematica": "ti",
            "assuntos": ["segurança", "redes"],
            "tipo_documento": "politica",
            "paginas_total": 15
        }
    ]
}

@patch("bbsia.app.routers.biblioteca.carregar_biblioteca")
def test_get_biblioteca_sem_filtros(mock_carregar):
    mock_carregar.return_value = MOCK_BIBLIOTECA
    
    response = client.get("/biblioteca")
    
    assert response.status_code == 200
    dados = response.json()
    assert dados["total"] == 3
    assert len(dados["documentos"]) == 3
    assert "conteudo_completo" not in dados["documentos"][0]
    assert "titulo" in dados["documentos"][0]

@patch("bbsia.app.routers.biblioteca.carregar_biblioteca")
def test_get_biblioteca_filtro_area(mock_carregar):
    mock_carregar.return_value = MOCK_BIBLIOTECA
    
    response = client.get("/biblioteca?area=TI")
    
    assert response.status_code == 200
    dados = response.json()
    assert dados["total"] == 2
    ids_retornados = [d["id"] for d in dados["documentos"]]
    assert "doc_1" in ids_retornados
    assert "doc_3" in ids_retornados

@patch("bbsia.app.routers.biblioteca.carregar_biblioteca")
def test_get_biblioteca_filtro_tipo(mock_carregar):
    mock_carregar.return_value = MOCK_BIBLIOTECA
    
    response = client.get("/biblioteca?tipo=relatorio")
    
    assert response.status_code == 200
    dados = response.json()
    assert dados["total"] == 1
    assert dados["documentos"][0]["id"] == "doc_2"

@patch("bbsia.app.routers.biblioteca.carregar_biblioteca")
def test_get_biblioteca_filtro_ano_range(mock_carregar):
    mock_carregar.return_value = MOCK_BIBLIOTECA
    
    response = client.get("/biblioteca?ano_min=2021&ano_max=2023")
    
    assert response.status_code == 200
    dados = response.json()
    assert dados["total"] == 2
    ids_retornados = [d["id"] for d in dados["documentos"]]
    assert "doc_1" not in ids_retornados


@patch("bbsia.app.routers.biblioteca.carregar_biblioteca")
def test_get_biblioteca_doc_encontrado(mock_carregar):
    mock_carregar.return_value = MOCK_BIBLIOTECA
    
    response = client.get("/biblioteca/doc_1")
    
    assert response.status_code == 200
    dados = response.json()
    assert dados["id"] == "doc_1"
    assert "conteudo_completo" in dados

@patch("bbsia.app.routers.biblioteca.carregar_biblioteca")
def test_get_biblioteca_doc_nao_encontrado(mock_carregar):
    mock_carregar.return_value = MOCK_BIBLIOTECA
    
    response = client.get("/biblioteca/doc_inexistente")
    
    assert response.status_code == 404
    assert "nao encontrado" in response.json()["detail"].lower()


@patch("bbsia.app.routers.biblioteca.carregar_biblioteca")
def test_get_filtros(mock_carregar):
    mock_carregar.return_value = MOCK_BIBLIOTECA
    
    response = client.get("/filtros")
    
    assert response.status_code == 200
    dados = response.json()
    
    assert dados["areas"] == ["RH", "TI", "ti"] 
    assert dados["tipos"] == ["manual", "politica", "relatorio"]
    assert dados["anos"] == [2020, 2022, 2023]
    assert dados["assuntos"] == ["hardware", "recrutamento", "redes", "segurança"]