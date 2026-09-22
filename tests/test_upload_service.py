import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi import HTTPException

from bbsia.app.uploads_service.service import (
    PdfValidationResult,
    _metadata_key_for_stored_filename,
    _resolve_quarantine_source_path,
    _safe_approved_path,
    _safe_quarantine_path,
    _sha256_bytes,
    validate_pdf_upload,
)

def test_sha256_bytes():
    conteudo = b"teste de conteudo"
    hash_esperado = "2b2391e0e91f453305987f5b9f3807c7a3cf4231bc973a3ea5ee031caebf8daf"
    resultado = _sha256_bytes(conteudo)
    assert resultado == hash_esperado

def test_metadata_key_for_stored_filename():
    mock_metadata = {
        "uploads/quarantine/doc1.pdf": {"stored_filename": "doc1.pdf"},
        "uploads/approved/doc2.pdf": {"outro_campo": "valor"},
        "apenas_string": "nao_sou_dict"
    }

    assert _metadata_key_for_stored_filename("doc1.pdf", mock_metadata) == "uploads/quarantine/doc1.pdf"
    
    assert _metadata_key_for_stored_filename("doc2.pdf", mock_metadata) == "uploads/approved/doc2.pdf"
    
    assert _metadata_key_for_stored_filename("doc3.pdf", mock_metadata) is None

@patch("bbsia.app.uploads_service.service.UPLOAD_QUARANTINE_DIR")
@patch("bbsia.app.uploads_service.service.secrets.token_hex")
def test_safe_quarantine_path_sucesso(mock_token, mock_dir):
    mock_token.return_value = "token_seguro_123"
    
    mock_dir_path = MagicMock(spec=Path)
    mock_dir.resolve.return_value = mock_dir_path
    mock_dir.__truediv__.return_value.resolve.return_value.parents = [mock_dir_path]
    
    caminho = _safe_quarantine_path("arquivo_perigoso.exe")
    
    mock_dir.__truediv__.assert_called_with("token_seguro_123.pdf")

def test_safe_approved_path_extensao_invalida():
    with pytest.raises(HTTPException) as exc_info:
        _safe_approved_path("documento_malicioso.sh")
    
    assert exc_info.value.status_code == 422
    assert "invalido" in exc_info.value.detail.lower()


@patch("bbsia.app.uploads_service.service.subprocess.run")
def test_validate_pdf_upload_sucesso(mock_subprocess_run):
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps({
        "page_count": 10,
        "extracted_chars": 5000,
        "prompt_injection_findings": ["ignore all previous instructions"]
    })
    mock_subprocess_run.return_value = mock_result

    caminho_fake = Path("/tmp/fake.pdf")
    resultado = validate_pdf_upload(caminho_fake)

    assert isinstance(resultado, PdfValidationResult)
    assert resultado.page_count == 10
    assert resultado.extracted_chars == 5000
    assert "ignore all previous instructions" in resultado.prompt_injection_findings
    
    mock_subprocess_run.assert_called_once()
    args_chamada = mock_subprocess_run.call_args[0][0]
    assert "-c" in args_chamada
    assert str(caminho_fake) in args_chamada

@patch("bbsia.app.uploads_service.service.subprocess.run")
def test_validate_pdf_upload_falha_execucao(mock_subprocess_run):
    mock_result = MagicMock()
    mock_result.returncode = 4
    mock_result.stderr = "PDF excede o limite de paginas."
    mock_subprocess_run.return_value = mock_result

    with pytest.raises(ValueError) as exc_info:
        validate_pdf_upload(Path("/tmp/fake.pdf"))
    
    assert "limite de paginas" in str(exc_info.value)

@patch("bbsia.app.uploads_service.service.subprocess.run")
def test_validate_pdf_upload_timeout(mock_subprocess_run):
    mock_subprocess_run.side_effect = subprocess.TimeoutExpired(cmd="python -c ...", timeout=10)

    with pytest.raises(TimeoutError) as exc_info:
        validate_pdf_upload(Path("/tmp/fake.pdf"), timeout_sec=10)
    
    assert "excedeu 10s" in str(exc_info.value)

@patch("bbsia.app.uploads_service.service.subprocess.run")
def test_validate_pdf_upload_saida_invalida(mock_subprocess_run):
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Isto nao e um JSON"
    mock_subprocess_run.return_value = mock_result

    with pytest.raises(ValueError) as exc_info:
        validate_pdf_upload(Path("/tmp/fake.pdf"))
    
    assert "saida invalida" in str(exc_info.value).lower()