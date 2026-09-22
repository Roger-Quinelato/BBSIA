import json
import logging
import threading
from unittest.mock import MagicMock, patch
import pytest
from fastapi import Request
from bbsia.app.runtime import audit


@pytest.fixture
def mock_state(tmp_path, monkeypatch):
    """
    Fixture para garantir que os testes não escrevam no diretório real da aplicação.
    Redirecionamos DATA_DIR e AUDIT_LOG_FILE para uma pasta temporária do pytest.
    """
    mock_dir = tmp_path / "test_data"
    mock_file = mock_dir / "audit.log"
    mock_lock = threading.Lock()

    monkeypatch.setattr(audit, "DATA_DIR", mock_dir)
    monkeypatch.setattr(audit, "AUDIT_LOG_FILE", mock_file)
    monkeypatch.setattr(audit, "_AUDIT_LOCK", mock_lock)
    
    return mock_dir, mock_file



def test_client_ip_sem_request():
    assert audit._client_ip(None) == "desconhecido"


def test_client_ip_sem_client_no_request():
    req = MagicMock(spec=Request)
    req.client = None
    assert audit._client_ip(req) == "desconhecido"


def test_client_ip_com_host_valido():
    req = MagicMock(spec=Request)
    req.client.host = "192.168.1.42"
    assert audit._client_ip(req) == "192.168.1.42"


def test_audit_event_sem_request(mock_state):
    _, mock_file = mock_state
    
    audit._audit_event("EVENTO_SIMPLES", detail_key="foobar")
    
    assert mock_file.exists()
    content = mock_file.read_text(encoding="utf-8").strip()
    payload = json.loads(content)
    
    assert payload["event"] == "EVENTO_SIMPLES"
    assert payload["client_ip"] == "desconhecido"
    assert payload["detail_key"] == "foobar"
    assert "ts" in payload


def test_audit_event_com_request(mock_state):
    _, mock_file = mock_state
    
    req = MagicMock(spec=Request)
    req.client.host = "10.0.0.5"
    req.method = "POST"
    req.url.path = "/api/v1/auth"
    
    audit._audit_event("LOGIN_TENTATIVA", request=req, user_id=99)
    
    content = mock_file.read_text(encoding="utf-8").strip()
    payload = json.loads(content)
    
    assert payload["event"] == "LOGIN_TENTATIVA"
    assert payload["client_ip"] == "10.0.0.5"
    assert payload["method"] == "POST"
    assert payload["path"] == "/api/v1/auth"
    assert payload["user_id"] == 99

@patch("bbsia.app.runtime.audit.log_event")
@patch("bbsia.app.runtime.audit._audit_event")
def test_record_event(mock_audit_event, mock_log_event):
    req = MagicMock(spec=Request)
    
    audit._record_event(
        "EVENTO_COMPLEXO", 
        request=req, 
        level=logging.WARNING, 
        extra_data=123
    )
    
    mock_audit_event.assert_called_once_with("EVENTO_COMPLEXO", req, extra_data=123)
    mock_log_event.assert_called_once_with(
        audit.LOGGER, 
        "app.audit", 
        "EVENTO_COMPLEXO", 
        level=logging.WARNING, 
        extra_data=123
    )