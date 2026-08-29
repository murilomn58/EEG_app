"""Ancora a convencao de coordenadas entre o MNE e o app 3D.

Traduzir errado nao levanta erro: os eletrodos so aparecem espelhados ou
girados, e a tela fica plausivel e errada. Estes testes sao o que transforma
esse erro silencioso em falha barulhenta.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import config
import eletrodos


def test_topo_da_cabeca_e_theta_zero():
    """Cz fica no topo: theta 0. Se isto virar 90, todo mundo desce pro
    equador e o mapa inteiro deita."""
    theta, _ = eletrodos.para_esfericas([0.0, 0.0, 0.09])
    assert theta == pytest.approx(0.0, abs=0.1)


def test_frente_e_phi_zero():
    """+y do MNE e a frente da cabeca, e a frente e phi 0 no app."""
    theta, phi = eletrodos.para_esfericas([0.0, 0.09, 0.0])
    assert theta == pytest.approx(90.0, abs=0.1)
    assert phi == pytest.approx(0.0, abs=0.1)


def test_direita_e_phi_noventa():
    """+x do MNE e a direita. Trocar o sinal aqui espelha a cabeca inteira
    — e um cerebro espelhado continua parecendo um cerebro."""
    _, phi = eletrodos.para_esfericas([0.09, 0.0, 0.0])
    assert phi == pytest.approx(90.0, abs=0.1)


def test_esquerda_e_phi_negativo():
    _, phi = eletrodos.para_esfericas([-0.09, 0.0, 0.0])
    assert phi == pytest.approx(-90.0, abs=0.1)


def test_origem_nao_vira_topo_da_cabeca():
    """Ponto na origem tem direcao indefinida. Devolver (0,0) o colocaria
    exatamente sobre o Cz, como se fosse medicao."""
    assert eletrodos.para_esfericas([0.0, 0.0, 0.0]) is None


def test_nan_nao_vira_posicao():
    assert eletrodos.para_esfericas([np.nan, 0.0, 0.09]) is None


def test_bate_com_a_montagem_que_o_app_ja_usava():
    """A prova real: converter o standard_1020 tem de reproduzir o
    POS_10_20 que o frontend traz cravado desde antes deste trabalho.
    Se divergir, os eletrodos do adhdata saem do lugar."""
    import mne
    montagem = mne.channels.make_standard_montage("standard_1020")
    ch_pos = montagem.get_positions()["ch_pos"]

    # o que o eeg-cerebro-3d.html tem em POS_10_20
    esperado = {
        "Fz": (45, 0), "Cz": (0, 0), "Fp1": (90, -18), "Fp2": (90, 18),
        "T7": (90, -90), "T8": (90, 90), "O1": (90, -162), "O2": (90, 162),
    }
    for nome, (theta_ref, phi_ref) in esperado.items():
        theta, phi = eletrodos.para_esfericas(ch_pos[nome])
        # tolerancia larga: o POS_10_20 do app e um esquema idealizado, nao a
        # montagem exata. O que se afirma aqui e que estao no MESMO hemisferio
        # e na mesma faixa — nao que sejam iguais ao grau.
        assert abs(theta - theta_ref) < 35, f"{nome}: theta {theta} vs {theta_ref}"
        if nome != "Cz":
            assert np.sign(phi) == np.sign(phi_ref) or abs(phi - phi_ref) < 35, \
                f"{nome}: phi {phi} vs {phi_ref} — lado trocado?"


def test_descrever_lista_todos_os_canais_mesmo_sem_posicao():
    """Esconder canal sem posicao faria a tela mostrar 127 e dizer 129."""
    import mne
    info = mne.create_info(["Fz", "Cz", "InventadoXY"], 128.0, "eeg", verbose=False)
    raw = mne.io.RawArray(np.zeros((3, 10)), info, verbose=False)

    d = eletrodos.descrever(raw)
    assert len(d["eletrodos"]) == 3
    nomes = [e["nome"] for e in d["eletrodos"]]
    assert "InventadoXY" in nomes
    inventado = next(e for e in d["eletrodos"] if e["nome"] == "InventadoXY")
    assert inventado["pos"] is None


def test_descrever_marca_o_papel_de_cada_eletrodo():
    import mne
    info = mne.create_info(["Fz", "Cz"], 128.0, "eeg", verbose=False)
    raw = mne.io.RawArray(np.zeros((2, 10)), info, verbose=False)

    d = eletrodos.descrever(raw, resolvido={"Fz": "Fz"})
    porNome = {e["nome"]: e for e in d["eletrodos"]}
    assert porNome["Fz"]["sugerido_para"] == "Fz"
    assert porNome["Cz"]["sugerido_para"] is None


# ---------------------------------------------------------------------------
# ESCOLHA DA MONTAGEM TEMPLATE: por cobertura medida, nao por contagem
# ---------------------------------------------------------------------------

def _raw_falso(nomes, fs=160.0):
    """Um Raw minimo com os nomes pedidos e sem montagem.

    Sem montagem de proposito: e o caso em que `posicoes` tem de cair no
    template, que e justamente o caminho que estes testes exercitam."""
    import mne
    import numpy as np
    info = mne.create_info(list(nomes), fs, "eeg", verbose=False)
    return mne.io.RawArray(np.zeros((len(nomes), 10)), info, verbose=False)


NOMES_64_COM_PONTO = [
    "Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.",
    "C5..", "C3..", "C1..", "Cz..", "C2..", "C4..", "C6..",
    "Cp5.", "Cp3.", "Cp1.", "Cpz.", "Cp2.", "Cp4.", "Cp6.",
    "Fp1.", "Fpz.", "Fp2.", "Af7.", "Af3.", "Afz.", "Af4.", "Af8.",
    "F7..", "F5..", "F3..", "F1..", "Fz..", "F2..", "F4..", "F6..", "F8..",
    "Ft7.", "Ft8.", "T7..", "T8..", "T9..", "T10.",
    "Tp7.", "Tp8.", "P7..", "P5..", "P3..", "P1..", "Pz..", "P2..", "P4..",
    "P6..", "P8..", "Po7.", "Po3.", "Poz.", "Po4.", "Po8.",
    "O1..", "Oz..", "O2..", "Iz..",
]


def test_sessenta_e_quatro_canais_nao_ficam_sem_posicao():
    """O caso que motivou a mudanca, com os nomes reais do eegmmidb.

    O criterio antigo escolhia a montagem pela CONTAGEM de canais: 64 nao
    estava na tabela, caia no standard_1020 de 19 nomes, e a intersecao dava
    ZERO. Na tela isso nao vira erro — vira a cabeca de conferencia
    desenhada vazia, sem dizer que a causa foi um ponto no fim do nome."""
    ch_pos, origem = eletrodos.posicoes(_raw_falso(NOMES_64_COM_PONTO))
    assert len(ch_pos) == 64, "canal ficou sem posicao"
    assert "standard_1005" in origem


def test_a_chave_devolvida_e_o_nome_do_arquivo():
    """`descrever` cruza `ch_pos` com `raw.ch_names` por igualdade de
    string. Devolver `CZ` normalizado faria o cruzamento falhar e o eletrodo
    sair com `pos: None`, sem erro nenhum — a tela mostraria 64 canais e
    nenhuma posicao."""
    ch_pos, _ = eletrodos.posicoes(_raw_falso(NOMES_64_COM_PONTO))
    assert "Cz.." in ch_pos
    assert "Cz" not in ch_pos


def test_a_malha_egi_continua_vencendo_pela_contagem():
    """Contagem nao foi abandonada, so deixou de ser o unico criterio: para
    a malha EGI ela e a informacao certa, porque `E11` nao existe em
    montagem 10-05 nenhuma."""
    nomes = [f"E{i}" for i in range(1, 129)] + ["Cz"]
    ch_pos, origem = eletrodos.posicoes(_raw_falso(nomes, fs=500.0))
    assert "GSN-HydroCel-129" in origem
    assert len(ch_pos) == 129


def test_dez_vinte_puro_continua_resolvendo():
    """O caso mais simples nao pode ter regredido: 19 nomes 10-20 exatos.
    Aqui o standard_1005 tambem cobriria os 19, e qualquer uma das duas
    montagens serve — o que se exige e que TODOS tenham posicao."""
    ch_pos, origem = eletrodos.posicoes(_raw_falso(config.CANAIS_10_20, fs=128.0))
    assert len(ch_pos) == 19
    assert origem is not None


def test_arquivo_sem_canal_conhecido_nao_inventa_posicao():
    """Nomes que nao existem em montagem nenhuma tem de sair SEM posicao.
    Cair num template qualquer aqui poria eletrodos plausiveis na cabeca
    para canais que ninguem sabe onde estao."""
    ch_pos, origem = eletrodos.posicoes(_raw_falso(["MISC1", "MISC2", "STATUS"]))
    assert ch_pos == {}
    assert origem is None


def test_descrever_marca_como_template_e_nao_como_medicao():
    """Template e media de populacao. Uma tela que apresenta isso como
    digitalizacao do sujeito faz o usuario confirmar o que nao conferiu."""
    d = eletrodos.descrever(_raw_falso(NOMES_64_COM_PONTO))
    assert "template" in d["origem_posicoes"]
    assert "media de populacao" in d["origem_posicoes"].replace("é", "e").replace("ç", "c").replace("ã", "a")


def test_malha_egi_de_128_canais_tambem_resolve():
    """O caso que o usuario nomeou: 'uma configuracao de EEG que tem 128'.

    O HBN tem 129 (o vertice de referencia entra como canal). Uma malha EGI
    publicada com 128 e outro arquivo e outra montagem, e tem de resolver
    sozinha — nao por acidente de cair na de 129."""
    nomes = [f"E{i}" for i in range(1, 129)]
    ch_pos, origem = eletrodos.posicoes(_raw_falso(nomes, fs=250.0))
    assert "GSN-HydroCel-128" in origem
    assert len(ch_pos) == 128
