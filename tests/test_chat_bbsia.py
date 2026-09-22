import argparse
import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch
import pytest
from bbsia.cli import chat_bbsia


def test_load_json_sucesso():
    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.read_text", return_value='[{"teste": "ok"}]'):
        resultado = chat_bbsia._load_json(Path("dummy.json"), [])
        assert resultado == [{"teste": "ok"}]

@patch("builtins.print")
def test_load_json_falha(mock_print):
    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.read_text", side_effect=Exception("Simulação de erro IO")):
        resultado = chat_bbsia._load_json(Path("dummy.json"), ["valor_padrao"])
        
        assert resultado == ["valor_padrao"]
        mock_print.assert_called_once()
        assert "[aviso] Nao foi possivel ler" in mock_print.call_args[0][0]


@patch("bbsia.cli.chat_bbsia._load_json")
def test_document_inventory(mock_load_json):
    mock_load_json.return_value = [
        {"documento": "doc1.pdf", "pagina": 1, "area": "ia", "assuntos": ["rag"], "doc_titulo": "Titulo 1", "doc_ano": 2023},
        {"documento": "doc1.pdf", "pagina": 2, "area": "ia", "assuntos": ["rag", "llm"]},
        {"documento": "doc2.pdf", "pagina": None, "area": "rh", "assuntos": []},
        "item_invalido_deve_ser_ignorado"
    ]
    
    inventario = chat_bbsia._document_inventory()

    assert len(inventario) == 2
    
    doc1 = next(d for d in inventario if d["documento"] == "doc1.pdf")
    assert doc1["chunks"] == 2
    assert doc1["paginas"] == 2
    assert doc1["areas"] == ["ia"]
    assert doc1["assuntos"] == ["rag", "llm"]
    assert doc1["titulo"] == "Titulo 1"
    
    doc2 = next(d for d in inventario if d["documento"] == "doc2.pdf")
    assert doc2["chunks"] == 1
    assert doc2["paginas"] == 0


@patch("bbsia.rag.retrieval.retriever.search")
@patch("builtins.print")
def test_run_search(mock_print, mock_search):
    mock_search.return_value = [
        {"documento": "doc.pdf", "pagina": 1, "score": 0.9, "texto": "trecho de teste"}
    ]
    
    chat_bbsia.run_search("teste?", 3, "ia", "", show_chunks=True)
    
    mock_search.assert_called_once_with(query="teste?", top_k=3, filtro_area=["ia"], filtro_assunto=[])
    mock_print.assert_any_call("1. doc.pdf | p. 1 | score=0.9000 dense=0.0000 sparse=0.0000")
    mock_print.assert_any_call("   trecho de teste")


@patch("sys.argv", ["chat_bbsia.py", "--sem-lista"])
@patch("builtins.input")
@patch("bbsia.cli.chat_bbsia.print_help")
@patch("bbsia.cli.chat_bbsia.print_inventory")
@patch("bbsia.cli.chat_bbsia.run_search")
@patch("bbsia.cli.chat_bbsia.run_answer")
def test_main_loop_roteamento_de_comandos(mock_answer, mock_search, mock_inv, mock_help, mock_input):
    mock_input.side_effect = [
        "",              # Enter vazio -> ignora e continua
        ":ajuda",        # Comando -> chama print_help
        ":docs",         # Comando -> chama print_inventory(50)
        ":docs --todos", # Comando -> chama print_inventory(10000)
        ":buscar",       # Comando malformado -> printa erro e continua
        ":buscar teste", # Comando -> extrai query e chama run_search
        "pergunta RAG",  # Texto livre -> chama pipeline de LLM (run_answer)
        "sair"           # Interrompe o loop
    ]
    
    chat_bbsia.main()
    
    assert mock_help.call_count == 1
    assert mock_inv.call_count == 2
    mock_search.assert_called_once_with("teste", 5, "", "", False)
    mock_answer.assert_called_once_with("pergunta RAG", os.getenv("DEFAULT_MODEL", "qwen3.5:7b-instruct"), 5, "", "", False)

@patch("sys.argv", ["chat_bbsia.py", "--sem-lista"])
@patch("builtins.input", side_effect=KeyboardInterrupt())
def test_main_keyboard_interrupt(mock_input):
    chat_bbsia.main()