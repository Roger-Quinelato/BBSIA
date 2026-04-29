from __future__ import annotations

import json

import scripts.calibrar_threshold as calibrar_threshold


def test_run_calibration_writes_auditable_output(monkeypatch, tmp_path):
    payload = {
        "estatisticas": {
            "in_scope_count": 1,
            "out_scope_count": 1,
            "dense_min": 0.31,
            "dense_p10": 0.31,
            "dense_p25": 0.31,
            "dense_mediana": 0.31,
            "dense_media": 0.31,
            "dense_max": 0.31,
            "out_scope_dense_max": 0.05,
            "threshold_sugerido_percent": 31,
            "threshold_sugerido_float": 0.31,
        },
        "qualidade": {"passed": 2, "total": 2, "pass_rate": 1.0},
        "resultados": [],
    }

    output_path = tmp_path / "threshold_calibration_latest.json"
    legacy_path = tmp_path / "resultados_calibracao.json"

    monkeypatch.setattr(calibrar_threshold, "calibrate_dense_threshold", lambda queries, top_k=5: payload)
    monkeypatch.setattr(calibrar_threshold, "LEGACY_OUTPUT_PATH", legacy_path)
    result = calibrar_threshold.run_calibration(output_path)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))

    assert result["calibracao"]["threshold_adotado_percent"] == calibrar_threshold.CURRENT_THRESHOLD_PERCENT
    assert result["calibracao"]["recomendacao_status"] == "acionavel"
    assert saved["estatisticas"]["threshold_sugerido_percent"] == 31
    assert saved["calibracao"]["query_count"] == len(calibrar_threshold.QUERIES_DE_TESTE)
    assert legacy == saved
