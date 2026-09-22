from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from bbsia.app.routers.admin import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)

@patch("bbsia.app.routers.admin._reprocess_manager.enqueue")
@patch("bbsia.app.routers.admin._record_event")
def test_reprocessar_base_enfileirado(mock_record, mock_enqueue):
    mock_enqueue.return_value = {"status": "queued", "run_id": "123", "queue_size": 2}
    
    response = client.post("/reprocessar")
    
    assert response.status_code == 409
    assert response.json()["status"] == "enfileirado"
    assert response.json()["run_id"] == "123"
    mock_record.assert_called_once()

@patch("bbsia.app.routers.admin._reprocess_manager.enqueue")
@patch("bbsia.app.routers.admin._record_event")
def test_reprocessar_base_iniciado(mock_record, mock_enqueue):
    mock_enqueue.return_value = {"status": "started", "run_id": "456", "queue_size": 1}
    
    response = client.post("/reprocessar")
    
    assert response.status_code == 200
    assert response.json()["status"] == "iniciado"
    mock_record.assert_called_once()

@patch("bbsia.app.routers.admin.reload_resources")
@patch("bbsia.app.routers.admin._record_event")
def test_recarregar_indice_sucesso(mock_record, mock_reload):
    response = client.post("/recarregar")
    
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    mock_reload.assert_called_once()
    mock_record.assert_called_once()

def test_upload_legacy_desativado():
    files = {"files": ("dummy.pdf", b"dummy", "application/pdf")}
    response = client.post("/upload-legacy-disabled", files=files)
    
    assert response.status_code == 410
    assert "legado" in response.json()["detail"].lower()

@patch("bbsia.app.routers.admin.MAX_UPLOAD_SIZE_MB", 1)
def test_upload_tamanho_excedido():
    conteudo_gigante = b"A" * (2 * 1024 * 1024) 
    files = {"files": ("gigante.pdf", conteudo_gigante, "application/pdf")}
    
    response = client.post("/upload", files=files)
    
    assert response.status_code == 413
    assert "excede o limite" in response.json()["detail"]

def test_upload_extensao_invalida():
    files = {"files": ("arquivo.txt", b"conteudo fake", "text/plain")}
    response = client.post("/upload", files=files)
    
    assert response.status_code == 422
    assert "Apenas .pdf" in response.json()["detail"]

def test_upload_magic_bytes_ausente():
    files = {"files": ("falso.pdf", b"nao sou um pdf real", "application/pdf")}
    response = client.post("/upload", files=files)
    
    assert response.status_code == 422
    assert "Assinatura PDF ausente" in response.json()["detail"]

@patch("pathlib.Path.mkdir")
@patch("bbsia.app.routers.admin._safe_quarantine_path")
@patch("bbsia.app.routers.admin.validate_pdf_upload")
@patch("bbsia.app.routers.admin.update_upload_metadata_entry")
@patch("bbsia.app.routers.admin._audit_event")
def test_upload_sucesso(mock_audit, mock_update, mock_validate, mock_safe_path, mock_mkdir):
    mock_path_obj = MagicMock()
    mock_path_obj.name = "123_seguro.pdf"
    mock_path_obj.relative_to.return_value.as_posix.return_value = "uploads/quarantine/123_seguro.pdf"
    mock_safe_path.return_value = mock_path_obj
    
    mock_validation = MagicMock()
    mock_validation.page_count = 5
    mock_validation.extracted_chars = 1500
    mock_validation.prompt_injection_findings = ["Injeção detectada"]
    mock_validate.return_value = mock_validation

    conteudo_valido = b"%PDF-1.4\nConteudo real de teste"
    files = {"files": ("real.pdf", conteudo_valido, "application/pdf")}
    
    response = client.post("/upload", files=files, data={"area": "ti", "assuntos": "seguranca, nuvem"})
    
    assert response.status_code == 200
    resultado = response.json()
    assert resultado["total"] == 1
    assert resultado["arquivos_salvos"][0] == "123_seguro.pdf"
    
    mock_path_obj.write_bytes.assert_called_once_with(conteudo_valido)
    
    args_update = mock_update.call_args[1]
    assert args_update["extra"]["status"] == "quarantined_prompt_review"
    assert args_update["assuntos"] == ["seguranca", "nuvem"]


@patch("bbsia.app.routers.admin.load_upload_metadata")
@patch("bbsia.app.routers.admin._resolve_quarantine_source_path")
def test_list_quarantine(mock_resolve, mock_load):
    mock_load.return_value = {
        "uploads/quarantine/doc1.pdf": {
            "status": "quarantined_pending_review",
            "stored_filename": "doc1.pdf",
            "uploaded_at": "2023-10-01T10:00:00Z"
        },
        "uploads/approved/doc2.pdf": {
            "status": "approved_pending_index"
        }
    }
    
    mock_resolved_path = MagicMock()
    mock_resolved_path.exists.return_value = True
    mock_resolved_path.is_file.return_value = True
    mock_resolve.return_value = mock_resolved_path
    
    response = client.get("/admin/quarantine")
    
    assert response.status_code == 200
    itens = response.json()["itens"]
    assert len(itens) == 1
    assert itens[0]["documento"] == "uploads/quarantine/doc1.pdf"
    assert itens[0]["file_exists"] is True

@patch("bbsia.app.routers.admin.load_upload_metadata")
@patch("bbsia.app.routers.admin.save_upload_metadata")
@patch("bbsia.app.routers.admin.shutil.move")
@patch("bbsia.app.routers.admin._resolve_quarantine_source_path")
@patch("bbsia.app.routers.admin._safe_approved_path")
def test_approve_quarantine_file_sucesso(mock_safe, mock_resolve, mock_move, mock_save, mock_load):
    mock_load.return_value = {
        "uploads/quarantine/doc1.pdf": {
            "status": "quarantined_pending_review",
            "stored_filename": "doc1.pdf",
            "quarantine_path": "uploads/quarantine/doc1.pdf"
        }
    }
    
    mock_source = MagicMock()
    mock_source.parents = [MagicMock(), MagicMock()] 
    
    with patch("pathlib.Path.resolve", return_value=mock_source.parents[0]):
        mock_source.exists.return_value = True
        mock_resolve.return_value = mock_source
        
        mock_approved = MagicMock()
        mock_safe.return_value = mock_approved
        
        response = client.post("/admin/quarantine/doc1.pdf/approve")
        
        assert response.status_code == 200
        assert response.json()["arquivo"] == "doc1.pdf"
        
        mock_move.assert_called_once_with(str(mock_source), str(mock_approved))
        
        mock_save.assert_called_once()
        dado_salvo = mock_save.call_args[0][0]
        assert "uploads/approved/doc1.pdf" in dado_salvo
        assert dado_salvo["uploads/approved/doc1.pdf"]["status"] == "approved_pending_index"