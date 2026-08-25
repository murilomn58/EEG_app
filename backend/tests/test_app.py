import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# IMPORTANTE: TestClient(app) SEM "with" não dispara o lifespan (que
# baixaria o fsaverage e demoraria minutos) — os testes setam app.state
# manualmente e testam só a camada HTTP, sem rodar MNE de verdade.
from app import app
from csv_data import CANAIS_19


def _df_teste():
    linhas = []
    for i in range(300):  # 300/128 ≈ 2.3s, o bastante pra uma janela de 2s
        linha = {c: float(i) for c in CANAIS_19}
        linha["ID"] = "suj1"
        linha["Class"] = "ADHD"
        linhas.append(linha)
    return pd.DataFrame(linhas)


@pytest.fixture
def client():
    app.state.df = _df_teste()
    app.state.inverse_operator = object()  # não usado diretamente — apply_source_localization é mockado
    app.state.n_vertices = 4
    return TestClient(app)


def test_get_subjects(client):
    resp = client.get("/subjects")
    assert resp.status_code == 200
    assert resp.json() == [{"id": "suj1", "classe": "ADHD", "duracao_s": 300 / 128}]


def test_source_localization_sucesso(client, monkeypatch):
    def fake_apply(janela, inverse_operator, method="dSPM"):
        assert janela.shape == (19, 256)  # 2s * 128Hz
        return np.array([1.0, 2.0, 3.0, 4.0])

    monkeypatch.setattr("app.mne_infer.apply_source_localization", fake_apply)

    resp = client.post(
        "/source-localization",
        json={"subject_id": "suj1", "t_start": 0.0, "t_end": 2.0, "method": "dSPM"},
    )
    assert resp.status_code == 200
    corpo = resp.json()
    assert corpo["values"] == [1.0, 2.0, 3.0, 4.0]
    assert corpo["n_vertices"] == 4
    assert corpo["method"] == "dSPM"


def test_source_localization_sujeito_inexistente(client):
    resp = client.post(
        "/source-localization",
        json={"subject_id": "naoexiste", "t_start": 0.0, "t_end": 2.0},
    )
    assert resp.status_code == 400
    assert "não encontrado" in resp.json()["detail"]


def test_raw_data_sucesso(client):
    resp = client.get("/raw-data", params={"subject_id": "suj1"})
    assert resp.status_code == 200
    corpo = resp.json()
    assert corpo["subject_id"] == "suj1"
    assert corpo["fs"] == 128
    assert set(corpo["channels"].keys()) == set(CANAIS_19)
    assert len(corpo["channels"]["Fp1"]) == 300


def test_raw_data_sujeito_inexistente(client):
    resp = client.get("/raw-data", params={"subject_id": "naoexiste"})
    assert resp.status_code == 400
    assert "não encontrado" in resp.json()["detail"]


def test_source_localization_erro_mne(client, monkeypatch):
    def fake_apply_com_erro(janela, inverse_operator, method="dSPM"):
        raise RuntimeError("falha simulada")

    monkeypatch.setattr("app.mne_infer.apply_source_localization", fake_apply_com_erro)

    resp = client.post(
        "/source-localization",
        json={"subject_id": "suj1", "t_start": 0.0, "t_end": 2.0},
    )
    assert resp.status_code == 400
    assert "falha simulada" in resp.json()["detail"]
