import argparse
import os
from unittest.mock import MagicMock, call, patch
import pytest
from bbsia.cli import perguntar_docs

def test_snippet_curto_e_vazio():
    assert perguntar_docs._snippet("") == ""
    assert perguntar_docs._snippet(None) == ""
    
    assert perguntar_docs._snippet("   Texto   com  espaços   ") == "Texto com espaços"

def test_snippet_longo():
    texto_longo = "A" * 1000
    resultado = perguntar_docs._snippet(texto_longo, max_chars=10)
    assert resultado == "AAAAAAAAAA..."
    assert len(resultado) == 13


@patch("bbsia.rag.retrieval.retriever.search")
@patch("builtins.print")
def test_print_search_com_resultados(mock_print, mock_search):
    mock_search.return_value = [
        {
            "parent_text": "Texto pai",
            "assuntos": ["teste", "cli"],
            "score": 0.95,
            "score_dense": 0.90,
            "score_sparse": 0.10,
            "documento": "doc1.pdf",
            "pagina": 5,
            "area": "ia"
        },
        {
            "texto": "Texto filho isolado"
        }
    ]
    
    perguntar_docs._print_search("O que é BBSIA?", 2, ["ia"], ["teste"])
    
    mock_search.assert_called_once_with(
        query="O que é BBSIA?", top_k=2, filtro_area=["ia"], filtro_assunto=["teste"]
    )
    
    mock_print.assert_any_call("\nPergunta: O que é BBSIA?")
    mock_print.assert_any_call("Resultados: 2")
    mock_print.assert_any_call("Documento: doc1.pdf | pagina: 5 | area: ia")
    mock_print.assert_any_call("Trecho: Texto filho isolado")


@patch("bbsia.rag.retrieval.retriever.search")
@patch("builtins.print")
def test_print_search_sem_resultados(mock_print, mock_search):
    mock_search.return_value = []
    
    perguntar_docs._print_search("Busca vazia", 1, [], [])
    
    mock_print.assert_any_call("Nenhum trecho recuperado.")


@patch("bbsia.rag.orchestration.pipeline.answer_question")
@patch("builtins.print")
def test_print_answer(mock_print, mock_answer_question):
    mock_answer_question.return_value = {
        "resposta": "Esta é a resposta simulada.",
        "fontes": ["fonte A", "fonte B"]
    }
    
    perguntar_docs._print_answer("Qual é o sentido da vida?", 3, [], [])
    
    mock_answer_question.assert_called_once_with(
        pergunta="Qual é o sentido da vida?", top_k=3, filtro_area=[], filtro_assunto=[]
    )
    mock_print.assert_any_call("\nResposta:")
    mock_print.assert_any_call("Esta é a resposta simulada.")
    mock_print.assert_any_call("- fonte A")


@patch("bbsia.cli.perguntar_docs._print_answer")
@patch("bbsia.cli.perguntar_docs._print_search")
def test_run_once_roteamento(mock_print_search, mock_print_answer):
    args_answer = argparse.Namespace(modo="answer", area="ia", assunto="rag", top_k=3)
    perguntar_docs._run_once(args_answer, "Teste Answer")
    mock_print_answer.assert_called_once_with("Teste Answer", 3, ["ia"], ["rag"])
    mock_print_search.assert_not_called()
    
    mock_print_answer.reset_mock()
    args_search = argparse.Namespace(modo="search", area="", assunto="", top_k=5)
    perguntar_docs._run_once(args_search, "Teste Search")
    mock_print_search.assert_called_once_with("Teste Search", 5, [], [])


@patch("sys.argv", ["perguntar_docs.py", "--pergunta", "pergunta direta", "--sem-faithfulness"])
@patch("bbsia.cli.perguntar_docs._run_once")
@patch.dict(os.environ, clear=True)
def test_main_execucao_direta(mock_run_once):
    perguntar_docs.main()
    
    assert os.environ.get("ENABLE_SYNC_FAITHFULNESS") == "false"
    mock_run_once.assert_called_once()
    arg_namespace, pergunta = mock_run_once.call_args[0]
    assert pergunta == "pergunta direta"

@patch("sys.argv", ["perguntar_docs.py", "--modo", "search"])
@patch("bbsia.cli.perguntar_docs._run_once")
@patch("builtins.input")
@patch("builtins.print")
def test_main_loop_interativo_saida_graciosa(mock_print, mock_input, mock_run_once):
    mock_input.side_effect = ["", "Primeira pergunta", "sair"]
    
    perguntar_docs.main()
    
    mock_run_once.assert_called_once()
    arg_namespace, pergunta = mock_run_once.call_args[0]
    assert pergunta == "Primeira pergunta"

@patch("sys.argv", ["perguntar_docs.py"])
@patch("builtins.input")
def test_main_loop_interativo_interrupcao_teclado(mock_input):
    mock_input.side_effect = KeyboardInterrupt()

    perguntar_docs.main()