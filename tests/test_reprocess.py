import logging
from unittest.mock import MagicMock, call, patch
import pytest
from bbsia.app.runtime import reprocess

@pytest.fixture
def mock_stages():
    with patch("bbsia.app.runtime.reprocess.run_extraction") as m_ext, \
         patch("bbsia.app.runtime.reprocess.run_chunking") as m_chk, \
         patch("bbsia.app.runtime.reprocess.run_embedding") as m_emb, \
         patch("bbsia.app.runtime.reprocess.reload_resources") as m_rel:
        
        m_ext.return_value = {}
        m_chk.return_value = {}
        m_emb.return_value = {}
        m_rel.return_value = {}
        
        yield m_ext, m_chk, m_emb, m_rel



@patch("bbsia.app.runtime.reprocess._record_event")
def test_run_reprocess_pipeline_sucesso(mock_record_event, mock_stages):
    m_ext, m_chk, m_emb, m_rel = mock_stages
    
    m_ext.return_value = {
        "documento_erros": [
            {"documento": "arquivo1.pdf", "erro": "falha no OCR"},
            "uma_string_perdida_que_deve_ser_ignorada",
            {"documento": "arquivo2.pdf"}
        ]
    }
    
    m_chk.return_value = None 
    mark_step = MagicMock()
    reprocess._run_reprocess_pipeline(mark_step)
    
    assert mark_step.call_count == 4
    mark_step.assert_has_calls([
        call("extracao"), call("chunking"), call("embedding"), call("reload")
    ])
    
    mock_record_event.assert_any_call("reprocess_stage_started", None, stage="extracao")
    mock_record_event.assert_any_call(
        "reprocess_document_error",
        None,
        level=logging.ERROR,
        stage="extracao",
        documento="arquivo1.pdf",
        error="falha no OCR"
    )
    mock_record_event.assert_any_call(
        "reprocess_document_error",
        None,
        level=logging.ERROR,
        stage="extracao",
        documento="arquivo2.pdf",
        error="erro nao informado"
    )


@patch("bbsia.app.runtime.reprocess._record_event")
def test_run_reprocess_pipeline_falha(mock_record_event, mock_stages):
    m_ext, _, _, _ = mock_stages
    
    m_ext.side_effect = RuntimeError("erro ao estabelecer conexão com o BD")
    
    mark_step = MagicMock()
    
    with pytest.raises(RuntimeError, match="erro ao estabelecer conexão com o BD"):
        reprocess._run_reprocess_pipeline(mark_step)
        
    called_args = mock_record_event.call_args_list[-1]
    assert called_args.args[0] == "reprocess_stage_failed"
    assert called_args.kwargs["stage"] == "extracao"
    assert called_args.kwargs["error"] == "erro ao estabelecer conexão com o BD"
    assert "duration_ms" in called_args.kwargs



@patch("bbsia.app.runtime.reprocess.ReprocessWorker")
@patch("bbsia.app.runtime.reprocess._record_event")
def test_build_reprocess_manager(mock_record_event, mock_worker_class):
    manager = reprocess._build_reprocess_manager()
    
    mock_worker_class.assert_called_once()

    argumentos_da_instanciacao = mock_worker_class.call_args.kwargs
    funcao_on_event = argumentos_da_instanciacao["on_event"]
    
    funcao_on_event("worker_parado", {"motivo": "timeout", "id": 42})
    
    mock_record_event.assert_called_once_with("worker_parado", None, motivo="timeout", id=42)