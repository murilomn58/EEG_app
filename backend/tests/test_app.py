import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# IMPORTANTE: TestClient(app) SEM "with" não dispara o lifespan (que
# baixaria o fsaverage e demoraria minutos) — os testes setam app.state
# manualmente e testam só a camada HTTP, sem rodar MNE de verdade.
from app import app
# o mesmo objeto de modulo que app.py importou: alterar aqui altera la
import config
from csv_data import CANAIS_19


def _df_teste():
    linhas = []
    for i in range(300):  # 300/128 ≈ 2.3s, o bastante pra uma janela de 2s
        linha = {c: float(i) for c in CANAIS_19}
        linha["ID"] = "suj1"
        linha["Class"] = "ADHD"
        linhas.append(linha)
    return pd.DataFrame(linhas)


def _df_filtravel():
    """Sinal longo o bastante para o passa-alta de 0,5 Hz do MNE rodar, com
    offset e rede de 50 Hz embutidos — a rampa de _df_teste é curta demais
    e o filtro levantaria erro de comprimento mínimo."""
    fs = 128.0
    t = np.arange(int(fs * 40)) / fs
    sinal = 140.0 + 20.0 * np.sin(2 * np.pi * 10.0 * t) + 6.0 * np.sin(2 * np.pi * 50.0 * t)

    linhas = []
    for i in range(len(t)):
        linha = {c: float(sinal[i]) for c in CANAIS_19}
        linha["ID"] = "suj1"
        linha["Class"] = "ADHD"
        linhas.append(linha)
    return pd.DataFrame(linhas)


@pytest.fixture
def client():
    app.state.df = _df_teste()
    app.state.inverse_operator = object()  # não usado diretamente — apply_source_localization é mockado
    app.state.n_vertices = 4
    app.state.cache_filtrado = {}
    return TestClient(app)


@pytest.fixture
def client_filtravel():
    app.state.df = _df_filtravel()
    app.state.cache_filtrado = {}
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


def test_raw_data_preproc_nenhum_e_o_default(client):
    resp = client.get("/raw-data", params={"subject_id": "suj1"})
    assert resp.json()["preproc"] == "nenhum"
    assert "freq_rede_hz" not in resp.json()


def test_raw_data_preproc_invalido(client):
    resp = client.get("/raw-data", params={"subject_id": "suj1", "preproc": "avancado"})
    assert resp.status_code == 400
    assert "preproc inválido" in resp.json()["detail"]


def test_raw_data_preproc_basico_filtra_e_informa_a_rede(client_filtravel):
    """Os metadados da decisão viajam junto com o dado: sem eles a
    interface mostraria um traçado diferente sem poder dizer o que saiu."""
    resp = client_filtravel.get(
        "/raw-data", params={"subject_id": "suj1", "preproc": "basico"}
    )
    assert resp.status_code == 200
    corpo = resp.json()

    assert corpo["preproc"] == "basico"
    assert corpo["freq_rede_hz"] == 50.0
    assert corpo["harmonicos_notchados"] == [50.0]
    assert corpo["l_freq"] == 0.5
    assert set(corpo["channels"].keys()) == set(CANAIS_19)

    # o offset de 140 saiu; o sinal continua na escala de µV
    media = np.mean(corpo["channels"]["Fp1"])
    assert abs(media) < 1.0


def test_raw_data_preproc_usa_cache(client_filtravel, monkeypatch):
    """Filtrar com MNE leva segundos — sem cache, alternar bruto/filtrado
    no frontend refiltraria a gravação inteira a cada clique."""
    client_filtravel.get("/raw-data", params={"subject_id": "suj1", "preproc": "basico"})
    assert ("suj1", "basico") in app.state.cache_filtrado

    def nao_deve_ser_chamado(subject_id, canais):
        raise AssertionError("refiltrou apesar do cache")

    monkeypatch.setattr("app._preprocessar_sujeito", nao_deve_ser_chamado)

    resp = client_filtravel.get(
        "/raw-data", params={"subject_id": "suj1", "preproc": "basico"}
    )
    assert resp.status_code == 200
    assert resp.json()["freq_rede_hz"] == 50.0


def test_source_localization_recusa_banco_nao_coberto(client):
    """O modelo de fonte e construido uma vez, a 128 Hz em µV. Rodar o HBN
    por ele DARIA numero — dSPM e normalizado, o mapa sairia plausivel — e
    seria errado, sem nada na tela denunciando. Recusar e a resposta honesta
    ate o modelo passar a ser construido por banco."""
    resp = client.post(
        "/source-localization",
        json={"subject_id": "sub-X", "t_start": 0.0, "t_end": 2.0,
              "dataset_id": "hbn_ds005505"},
    )
    assert resp.status_code == 400
    assert "ainda não cobre" in resp.json()["detail"]


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


def test_datasets_lista_com_disponibilidade(client):
    resp = client.get("/datasets")
    assert resp.status_code == 200
    ids = {d["id"] for d in resp.json()["datasets"]}
    assert "adhdata" in ids and "hbn_ds005505" in ids
    # todo item declara disponibilidade, e quando indisponivel diz por que
    for d in resp.json()["datasets"]:
        assert "disponivel" in d
        if not d["disponivel"]:
            assert d["motivo_indisponivel"]


def test_dataset_config_desconhecido(client):
    resp = client.get("/dataset-config", params={"dataset_id": "nao_existe"})
    assert resp.status_code == 400
    assert "dataset desconhecido" in resp.json()["detail"]


def test_dataset_config_mede_e_se_configura_pelo_medido(client_filtravel):
    """O passo 2 do wizard: o app mede em vez de assumir, e se configura
    para o que mediu em vez de exigir que o banco bata com uma premissa."""
    resp = client_filtravel.get("/dataset-config", params={"dataset_id": "adhdata"})
    assert resp.status_code == 200
    corpo = resp.json()

    assert corpo["detectado"]["fs_hz"] == 128.0
    assert corpo["detectado"]["n_canais"] == 19
    assert corpo["detectado"]["freq_rede_hz"] == 50.0
    # o que o app VAI USAR sai do medido, nao de uma constante global
    assert corpo["esperado"]["fs_hz"] == 128.0
    assert corpo["esperado"]["n_canais"] == 19
    assert corpo["compativel"]["ok"] is True

    # a referencia nunca sai como medida: ela nao e detectavel com seguranca
    assert corpo["detectado"]["referencia"]["origem"] == "declarada"
    assert corpo["detectado"]["referencia"]["evidencia_no_sinal"]


def test_dataset_config_resolve_canais_por_nome_direto(client_filtravel):
    """O adhdata ja nomeia no padrao 10-20, entao a resolucao e identidade e
    nada aparece como traduzido."""
    resp = client_filtravel.get("/dataset-config", params={"dataset_id": "adhdata"})
    canais = resp.json()["canais"]
    assert canais["faltantes"] == []
    assert canais["traduzidos"] == []
    assert canais["resolvido"]["Fz"] == "Fz"
    assert len(canais["resolvido"]) == len(canais["analise"])


def test_dataset_config_avisa_rede_por_eliminacao(client_filtravel):
    """A 128 Hz o teto de Nyquist exclui 60 Hz antes de medir, e o wizard
    precisa dizer isso em vez de mostrar '50 Hz' como se fosse medicao."""
    resp = client_filtravel.get("/dataset-config", params={"dataset_id": "adhdata"})
    codigos = {a["codigo"] for a in resp.json()["compativel"]["avisos"]}
    assert "rede_por_eliminacao" in codigos
    # aviso de atencao nao bloqueia a entrada
    assert resp.json()["compativel"]["ok"] is True


def test_dataset_config_bloqueia_abaixo_da_fs_minima(client_filtravel, monkeypatch):
    """O unico limite de taxa que ainda barra e o FISICO: abaixo dele a banda
    mais alta nao cabe sob o Nyquist e o filtro sai degenerado."""
    monkeypatch.setattr("config.FS_MINIMA", 200.0)
    resp = client_filtravel.get("/dataset-config", params={"dataset_id": "adhdata"})
    corpo = resp.json()
    assert corpo["compativel"]["ok"] is False
    bloqueios = [a for a in corpo["compativel"]["avisos"] if a["severidade"] == "bloqueio"]
    assert any(a["codigo"] == "fs_incompativel" for a in bloqueios)


def test_dataset_config_fs_alta_nao_bloqueia(client_filtravel, monkeypatch):
    """Taxa ACIMA da declarada nao barra mais: o app toca no relogio do sinal
    em vez de reamostrar, entao 500 Hz e configuracao, nao incompatibilidade.
    Este e o teste que prova que a premissa global de 128 Hz saiu."""
    monkeypatch.setitem(config.DATASETS["adhdata"], "fs_declarada", 500.0)
    resp = client_filtravel.get("/dataset-config", params={"dataset_id": "adhdata"})
    corpo = resp.json()

    assert corpo["compativel"]["ok"] is True
    codigos = {a["codigo"] for a in corpo["compativel"]["avisos"]}
    # a divergencia contra a doc vira ATENCAO, e o app segue o sinal medido
    assert "fs_divergente_da_declarada" in codigos
    assert "fs_incompativel" not in codigos
    assert corpo["esperado"]["fs_hz"] == 128.0


def test_dataset_config_avisa_canal_de_analise_plano(monkeypatch):
    """No HBN o Cz E a referencia fisica e vem com desvio ~0, mas entra no
    conjunto de 19 mesmo assim. Nao bloqueia (re-referenciar conserta), mas
    tem de aparecer: um Cz achatado no tracado e plausivel e errado."""
    fs = 128.0
    t = np.arange(int(fs * 40)) / fs
    sinal = 20.0 * np.sin(2 * np.pi * 10.0 * t)

    linhas = []
    for i in range(len(t)):
        linha = {c: float(sinal[i] + j) for j, c in enumerate(CANAIS_19)}
        linha["Cz"] = 0.0  # a referencia fisica: referenciada contra si mesma
        linha["ID"] = "suj1"
        linha["Class"] = "ADHD"
        linhas.append(linha)

    app.state.df = pd.DataFrame(linhas)
    app.state.cache_filtrado = {}
    resp = TestClient(app).get("/dataset-config", params={"dataset_id": "adhdata"})
    corpo = resp.json()

    avisos = {a["codigo"]: a for a in corpo["compativel"]["avisos"]}
    assert "canal_analise_plano" in avisos
    assert "Cz" in avisos["canal_analise_plano"]["texto"]
    # atencao, nao bloqueio: tem conserto conhecido
    assert avisos["canal_analise_plano"]["severidade"] == "atencao"
    assert corpo["compativel"]["ok"] is True


def test_dataset_config_bloqueia_canal_sem_traducao(client_filtravel, monkeypatch):
    """Ter MAIS canais que os 19 nao e problema — malha densa e o caso normal.
    Problema e faltar um canal que a analise precisa e nenhum mapa alcanca."""
    monkeypatch.setitem(
        config.DATASETS["adhdata"], "canais_analise",
        list(config.CANAIS_10_20) + ["NaoExiste"],
    )
    resp = client_filtravel.get("/dataset-config", params={"dataset_id": "adhdata"})
    corpo = resp.json()

    assert corpo["compativel"]["ok"] is False
    bloqueios = [a for a in corpo["compativel"]["avisos"] if a["severidade"] == "bloqueio"]
    assert any(a["codigo"] == "canais_incompativel" for a in bloqueios)
    assert corpo["canais"]["faltantes"] == ["NaoExiste"]


# --- janela pedida: teto e tempo negativo -------------------------------
# Os dois pedidos abaixo devolviam 200 antes do conserto. O primeiro
# devolvia outro trecho da gravação sob o carimbo do tempo pedido; o
# segundo devolvia um mapa correto e levava o processo junto.


def test_source_localization_recusa_tempo_negativo(client):
    """MEDIDO contra o backend vivo, sujeito v10p: t=[-10,-5] voltou 200 com
    valores BIT A BIT idênticos aos de t=[101,75; 106,75] — o fim da
    gravação — e "time": -10.0 na resposta. O médico leria o mapa de fonte
    de um instante que ele não pediu, sem nada na tela avisando."""
    resp = client.post(
        "/source-localization",
        json={"subject_id": "suj1", "t_start": -10.0, "t_end": -5.0},
    )
    assert resp.status_code == 400
    assert "negativo" in resp.json()["detail"]


def test_source_localization_recusa_janela_gigante(client):
    """MEDIDO: t_end=1e9 no v10p devolveu 200 em 17,1 s e levou o pico de
    working set do uvicorn de 1.878 MB para 10.334 MB (20.484 vértices x
    14.304 amostras x 8 bytes = 2,3 GB só no stc)."""
    resp = client.post(
        "/source-localization",
        json={"subject_id": "suj1", "t_start": 0.0, "t_end": 1e9},
    )
    assert resp.status_code == 400
    assert "janela longa demais" in resp.json()["detail"]


def test_source_localization_nao_chama_o_mne_com_janela_recusada(client, monkeypatch):
    """O ponto do teto não é a mensagem de erro: é o trabalho que não
    acontece. Se a recusa viesse depois da solução inversa, o 400 sairia
    bonito e a memória teria subido do mesmo jeito."""
    def nao_deve_ser_chamado(*args, **kwargs):
        raise AssertionError("a solução inversa rodou apesar da janela recusada")

    monkeypatch.setattr("app.mne_infer.apply_source_localization", nao_deve_ser_chamado)

    resp = client.post(
        "/source-localization",
        json={"subject_id": "suj1", "t_start": 0.0, "t_end": 1e9},
    )
    assert resp.status_code == 400
    # a conferência do texto não é preciosismo: sem ela, o teste passaria
    # também no código ANTIGO, onde o mock estoura dentro da chamada ao MNE e
    # o except do endpoint transforma o estouro num 400 "falha no MNE" — 400
    # pelo motivo errado, depois de a memória já ter sido gasta
    assert "janela longa demais" in resp.json()["detail"]


# --- sujeito inexistente num banco BIDS ---------------------------------
# A raiz falsa abaixo é só a ÁRVORE de diretórios: os .set ficam vazios de
# propósito, porque a escolha do sujeito acontece antes de qualquer leitura
# de arquivo. Um .set de verdade aqui testaria o MNE, não a escolha.


def _raiz_bids_falsa(tmp_path, nomes):
    for nome in nomes:
        pasta = tmp_path / "ds_falso" / nome / "eeg"
        pasta.mkdir(parents=True)
        (pasta / f"{nome}_task-RestingState_eeg.set").write_bytes(b"")
    return tmp_path


@pytest.fixture
def bids_falso(tmp_path, monkeypatch):
    """Registra um banco BIDS de mentira em config.DATASETS, com dois
    sujeitos, e devolve o id dele."""
    _raiz_bids_falsa(tmp_path, ["sub-PRIMEIRO", "sub-SEGUNDO"])
    monkeypatch.setattr(config, "RAIZ_DADOS", tmp_path)
    monkeypatch.setitem(
        config.DATASETS,
        "bids_falso",
        {
            "nome": "banco de teste",
            "tipo": "bids",
            "accession": "ds_falso",
            "referencia_declarada": "Cz",
            "fs_declarada": 500.0,
            "n_canais_declarado": 129,
            "canais_analise": config.CANAIS_10_20,
            "mapa_canais": {},
            "procedencia_mapa": "teste",
        },
    )
    return "bids_falso"


def test_set_do_sujeito_sem_id_escolhe_o_primeiro(bids_falso):
    """O comportamento LEGÍTIMO que o conserto não pode quebrar: quem abre
    a tela sem ter escolhido ninguém recebe o primeiro sujeito."""
    from app import _set_do_sujeito

    arquivo, escolhido, _ = _set_do_sujeito(config.DATASETS[bids_falso], None)
    assert escolhido == "sub-PRIMEIRO"
    assert arquivo.parent.parent.name == "sub-PRIMEIRO"


def test_set_do_sujeito_recusa_id_inexistente(bids_falso):
    """O que não pode: id PRESENTE e inexistente caindo no primeiro."""
    from app import _set_do_sujeito

    with pytest.raises(ValueError, match="sujeito não encontrado"):
        _set_do_sujeito(config.DATASETS[bids_falso], "sub-INEXISTENTE")


def test_eventos_sujeito_inexistente_nao_devolve_o_primeiro(client, bids_falso):
    """MEDIDO contra o backend vivo antes do conserto:
    /eventos?subject_id=sub-INEXISTENTE voltou 200 com "subject_id":
    "sub-INEXISTENTE", 36 eventos e os onsets (0.0, 1.288, 47.84) de
    sub-NDARAC904DMU — o dado de um sujeito servido sob o nome de outro."""
    resp = client.get(
        "/eventos", params={"dataset_id": bids_falso, "subject_id": "sub-INEXISTENTE"}
    )
    assert resp.status_code == 400
    assert "sujeito não encontrado" in resp.json()["detail"]


def test_dataset_config_sujeito_inexistente_nao_devolve_o_primeiro(client, bids_falso):
    """MEDIDO: /dataset-config?subject_id=sub-INEXISTENTE voltou 200 com
    "amostra": {"subject_id": "sub-INEXISTENTE"} e as medidas de outro
    sujeito. O wizard confirmaria a configuração do sujeito errado."""
    resp = client.get(
        "/dataset-config",
        params={"dataset_id": bids_falso, "subject_id": "sub-INEXISTENTE"},
    )
    assert resp.status_code == 400
    assert "sujeito não encontrado" in resp.json()["detail"]


def test_source_localization_recusa_infinito(client):
    """O único 500 que sobrou deste caminho: `1e999` é JSON válido, vira inf,
    e int(round(inf * 128)) levanta OverflowError — que não é ValueError e
    atravessa o except do endpoint. MEDIDO contra o backend vivo: HTTP 500."""
    # `content=` e não `json=`: o encoder do httpx recusa serializar inf, e é
    # justamente por isso que o pedido hostil vem por texto cru. Um cliente
    # HTTP qualquer manda `1e999` sem reclamar — o servidor é que precisa
    # aguentar
    resp = client.post(
        "/source-localization",
        content='{"subject_id": "suj1", "t_start": 0.0, "t_end": 1e999}',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400
    assert "finitos" in resp.json()["detail"]
