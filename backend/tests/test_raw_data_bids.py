"""O caminho que a tela de conferencia dizia que faltava: servir um banco de
malha densa ja reduzido ao conjunto de analise, no relogio do proprio sinal.

Le o HBN de verdade, como test_inventario_hbn ja faz.
"""
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import app as app_mod
import config

DATASET = "hbn_ds005505"


@pytest.fixture
def limpo():
    # OrderedDict e nao dict: e o que o lifespan cria de verdade, e o cache_raw
    # agora tem politica de despejo que depende da ordem. Um fixture que
    # entrega um tipo diferente do de producao testa outra coisa.
    app_mod.app.state.cache_raw = OrderedDict()
    app_mod.app.state.cache_filtrado = {}
    return app_mod


def test_reduz_129_para_o_conjunto_de_analise(limpo):
    """A reducao acontece no BACKEND, que era exatamente o que a tela pedia."""
    r = limpo.get_raw_data(subject_id=None, dataset_id=DATASET)
    assert list(r["channels"]) == list(config.CANAIS_10_20)
    assert len(r["channels"]) == 19


def test_serve_no_relogio_do_sinal_e_nao_em_128(limpo):
    """O teste que prova que a premissa global de 128 Hz saiu do caminho do
    dado: o HBN sai a 500 Hz, e a duracao bate com a gravacao real."""
    r = limpo.get_raw_data(subject_id=None, dataset_id=DATASET)
    assert r["fs"] == 500.0
    duracao = len(r["channels"]["Fz"]) / r["fs"]
    assert duracao == pytest.approx(403.5, abs=1.0)


def test_declara_a_unidade_como_uv_com_a_ressalva_que_a_evidencia_sustenta(limpo):
    """Trava a redacao NOVA, depois da retratacao.

    A versao antiga deste teste exigia a palavra "arbitraria": ela vinha da
    conclusao de que o HBN nao estaria em microvolt, e essa conclusao foi
    retratada. A evidencia que a sustentava era o percentil 99 do sinal BRUTO
    (~89.000), que e dominado pelo offset DC — depois do passa-alta o pico do
    HBN fica ABAIXO do adhdata e dentro da faixa fisiologica.

    O que continua de pe, e o que este teste trava, e que a calibracao NAO
    ESTA CONFIRMADA: ninguem verificou o ganho deste banco contra sinal de
    referencia conhecido. O assert e por ausencia tambem, de proposito — se
    alguem reescrever "arbitraria" ou "nao calibrado" aqui, isto acusa."""
    r = limpo.get_raw_data(subject_id=None, dataset_id=DATASET)
    assert r["unidade"] == "µV (calibração não confirmada neste banco)"
    assert "arbitrária" not in r["unidade"]
    assert "não calibrado" not in r["unidade"]
    assert r["amplitude_p99"] > 0


def test_o_ramo_bids_devolve_microvolt_e_nao_volt_cru(limpo):
    """O bug que a unidade escondia: este ramo devolvia volt cru enquanto o
    ramo do CSV devolvia µV — 1e6 de diferenca no MESMO endpoint, invisivel
    porque o frontend auto-escala pelo amplitude_p99.

    O limiar e frouxo de proposito: nao se trava o valor de um sujeito, trava-se
    a ORDEM DE GRANDEZA. Em volt o p99 do bruto daria ~0,089; em µV da ~89.000.
    Qualquer coisa acima de 1 so e possivel se a conversao aconteceu."""
    r = limpo.get_raw_data(subject_id=None, dataset_id=DATASET)
    assert r["amplitude_p99"] > 1.0


def test_clinico_aplica_passa_baixa_e_devolve_o_corte_verdadeiro(limpo):
    """Sem passa-baixa, contracao de musculo temporal entra inteira no tracado.
    O corte pedido e 70 Hz e a 500 Hz ele cabe inteiro — o campo h_freq existe
    para a tela escrever o valor aplicado, nao o pedido."""
    r = limpo.get_raw_data(subject_id=None, dataset_id=DATASET, preproc="clinico")
    assert r["h_freq"] == 70.0
    assert r["l_freq"] == 0.5


def test_basico_continua_sem_passa_baixa(limpo):
    """`basico` e o que o qc_relatorio.py mede. Se ele ganhasse um passa-baixa,
    o relatorio deixaria de ser evidencia sobre o que o usuario ve."""
    r = limpo.get_raw_data(subject_id=None, dataset_id=DATASET, preproc="basico")
    assert r["h_freq"] is None


def test_preproc_invalido_e_recusado(limpo):
    with pytest.raises(Exception) as e:
        limpo.get_raw_data(subject_id=None, dataset_id=DATASET, preproc="chute")
    assert "preproc inválido" in str(e.value)


def test_cache_raw_nao_cresce_sem_limite(limpo):
    """Cada entrada aqui e a gravacao INTEIRA, ~208 MB por sujeito do HBN.
    Medido no servidor de pe: o RSS do uvicorn subiu ~273 MB a cada sujeito
    novo. Sem despejo, trocar de sujeito no seletor e vazamento de memoria com
    outro nome."""
    raiz = config.caminho_release(config.DATASETS[DATASET]["accession"])
    sets = sorted(raiz.glob("sub-*/eeg/*task-RestingState*.set"))
    sujeitos = [s.parent.parent.name for s in sets][: limpo.LIMITE_CACHE_RAW + 1]
    if len(sujeitos) <= limpo.LIMITE_CACHE_RAW:
        pytest.skip(f"release com menos de {limpo.LIMITE_CACHE_RAW + 1} sujeitos no disco")

    for s in sujeitos:
        limpo.get_raw_data(subject_id=s, dataset_id=DATASET)

    assert len(limpo.app.state.cache_raw) == limpo.LIMITE_CACHE_RAW
    # o primeiro foi despejado, os dois ultimos ficaram
    assert sujeitos[0] not in limpo.app.state.cache_raw
    assert sujeitos[-1] in limpo.app.state.cache_raw


def test_diz_de_qual_eletrodo_cada_canal_veio(limpo):
    """Sem isso o usuario ve 'Fz' e nao tem como saber que o dado veio do
    E11 — e a correspondencia EGI e aproximada, nao exata."""
    r = limpo.get_raw_data(subject_id=None, dataset_id=DATASET)
    assert r["canais_origem"]["Fz"] == "E11"
    assert r["canais_origem"]["Pz"] == "E62"


def test_cache_evita_reler_o_set(limpo):
    r1 = limpo.get_raw_data(subject_id=None, dataset_id=DATASET)
    assert r1["subject_id"] in limpo.app.state.cache_raw

    def nao_deve_ser_chamado(*a, **k):
        raise AssertionError("releu o .set apesar do cache")

    original = limpo.preproc_basico.carregar_raw
    limpo.preproc_basico.carregar_raw = nao_deve_ser_chamado
    try:
        r2 = limpo.get_raw_data(subject_id=r1["subject_id"], dataset_id=DATASET)
    finally:
        limpo.preproc_basico.carregar_raw = original
    assert r2["fs"] == r1["fs"]


def test_dataset_desconhecido(limpo):
    with pytest.raises(Exception) as e:
        limpo.get_raw_data(subject_id=None, dataset_id="nao_existe")
    assert "dataset desconhecido" in str(e.value)


def test_o_cz_do_hbn_chega_plano_como_a_tela_avisa(limpo):
    """Confirma no DADO o que /dataset-config anuncia: o Cz e a referencia
    fisica e vem com desvio zero. Servimos assim de proposito — quem decide
    re-referenciar e o usuario —, mas se um dia deixar de ser verdade, o
    aviso da tela vira mentira e este teste acusa."""
    r = limpo.get_raw_data(subject_id=None, dataset_id=DATASET)
    assert np.array(r["channels"]["Cz"]).std() == 0.0
