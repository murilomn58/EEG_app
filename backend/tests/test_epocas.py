# -*- coding: utf-8 -*-
"""Trava a garantia central da epocagem: uma época nunca atravessa fronteira.

POR QUE ESTA É A GARANTIA QUE IMPORTA

Uma época que cruza uma descontinuidade não dá erro. Ela dá um número — e o
número é plausível. No adhdata, cujas gravações são linhas concatenadas de
121 crianças num CSV só, uma janela que cruza a fronteira de `ID` emenda
**duas crianças numa época só**, e essa época recebe o rótulo de uma delas.
No HBN, o evento `boundary` do EEGLAB marca onde o registro foi cortado e
remendado; uma janela em cima dele contém um degrau que não é fisiologia.

Nos dois casos o resultado entra no classificador como se fosse sinal. É
exatamente o tipo de defeito que este projeto persegue: não levanta exceção,
não aparece na tela, e contamina a medida que o experimento existe para
produzir.

Por isso o teste principal aqui não verifica contagem nem formato: ele
constrói dois "sujeitos" de amplitude conhecida, concatena, epoca, e exige
que **nenhuma época contenha amostra dos dois**.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import epocas


# ---------------------------------------------------------------------------
# de onde vêm as fronteiras
# ---------------------------------------------------------------------------

def test_cortes_por_mudanca_acha_a_troca_de_sujeito():
    """No adhdata a coluna ID é a única fronteira que existe."""
    ids = ["a", "a", "a", "b", "b", "c"]
    assert epocas.cortes_por_mudanca(ids) == [3, 5]


def test_cortes_por_mudanca_sem_troca_nao_inventa_fronteira():
    assert epocas.cortes_por_mudanca(["a"] * 10) == []


def test_cortes_por_mudanca_lista_vazia():
    assert epocas.cortes_por_mudanca([]) == []


def test_cortes_de_eventos_converte_segundos_em_amostra():
    """O events.tsv traz onset em segundos; o corte é em amostra."""
    eventos = [
        {"onset": 0.0, "valor": "resting_start"},
        {"onset": 2.0, "valor": "boundary"},
        {"onset": 5.5, "valor": "instructed_toOpenEyes"},
    ]
    assert epocas.cortes_de_eventos(eventos, fs=100.0) == [200]


def test_cortes_de_eventos_ignora_onset_zero():
    """Um `boundary` em t=0 não corta nada: já é o começo da gravação, e
    aceitá-lo criaria um segmento vazio na frente."""
    eventos = [{"onset": 0.0, "valor": "boundary"}]
    assert epocas.cortes_de_eventos(eventos, fs=100.0) == []


# ---------------------------------------------------------------------------
# blocos ancorados em evento (olhos abertos / fechados do HBN)
# ---------------------------------------------------------------------------

ALVOS = ("instructed_toOpenEyes", "instructed_toCloseEyes")


def _eventos_oc():
    """O padrão medido nos events.tsv do ds005505: abre em t, fecha em t+20,
    abre de novo em t+60. Aqui com t = 10 s, e o último "abre" sem par."""
    return [
        {"onset": 10.0, "valor": "instructed_toOpenEyes"},
        {"onset": 30.0, "valor": "instructed_toCloseEyes"},
        {"onset": 70.0, "valor": "instructed_toOpenEyes"},
        {"onset": 90.0, "valor": "instructed_toCloseEyes"},
        {"onset": 130.0, "valor": "instructed_toOpenEyes"},
    ]


def test_blocos_alternam_e_medem_a_duracao_pelo_proximo_evento():
    blocos, dec = epocas.blocos_entre_eventos(_eventos_oc(), 100.0, 15000, ALVOS)
    assert blocos == [
        (1000, 3000, "instructed_toOpenEyes"),
        (3000, 7000, "instructed_toCloseEyes"),
        (7000, 9000, "instructed_toOpenEyes"),
        (9000, 13000, "instructed_toCloseEyes"),
    ]
    # 20 s abertos, 40 s fechados: medido, não fixado
    assert dec["duracao_s_por_valor"]["instructed_toOpenEyes"] == [20.0, 20.0]
    assert dec["duracao_s_por_valor"]["instructed_toCloseEyes"] == [40.0, 40.0]
    assert dec["n_blocos_por_valor"] == {"instructed_toOpenEyes": 2,
                                         "instructed_toCloseEyes": 2}


def test_ultimo_alvo_sem_fechamento_e_declarado_e_nao_vira_bloco():
    blocos, dec = epocas.blocos_entre_eventos(_eventos_oc(), 100.0, 15000, ALVOS)
    assert all(fim <= 13000 for _, fim, _ in blocos)
    assert dec["ultimo_alvo_sem_fechamento"] == 130.0
    assert dec["n_marcos"] == 5


def test_evento_de_outra_tarefa_nao_abre_nem_fecha_bloco():
    """O sub-NDARAC904DMU carrega eventos de seqLearning dentro do
    RestingState. Eles não podem mexer nas bordas dos blocos de repouso."""
    eventos = _eventos_oc() + [
        {"onset": 50.0, "valor": "dot_no1_ON"},
        {"onset": 55.0, "valor": "seqLearning_start"},
    ]
    com, _ = epocas.blocos_entre_eventos(eventos, 100.0, 15000, ALVOS)
    sem, _ = epocas.blocos_entre_eventos(_eventos_oc(), 100.0, 15000, ALVOS)
    assert com == sem


def test_corte_dentro_do_bloco_recorta_sem_perder_o_rotulo():
    """Um boundary em 50 s cai dentro do bloco fechado (30 a 70 s): o bloco
    vira dois segmentos, os dois ainda rotulados como fechados, e a duração
    declarada continua sendo a do bloco inteiro."""
    blocos, dec = epocas.blocos_entre_eventos(
        _eventos_oc(), 100.0, 15000, ALVOS, cortes=[5000]
    )
    fechados = [(a, b) for a, b, v in blocos if v == "instructed_toCloseEyes"]
    assert (3000, 5000) in fechados and (5000, 7000) in fechados
    assert dec["n_blocos_por_valor"]["instructed_toCloseEyes"] == 2
    assert dec["duracao_s_por_valor"]["instructed_toCloseEyes"] == [40.0, 40.0]
    assert dec["blocos_recortados_por_corte"] == 1


def test_nenhuma_epoca_mistura_olhos_abertos_com_fechados():
    """A garantia central do módulo, aplicada ao caso novo: sinal 1,0 nos
    blocos abertos e 100,0 nos fechados; epocar cada condição pelos seus
    segmentos; nenhuma época pode conter as duas amplitudes."""
    fs = 100.0
    n = 15000
    dado = np.zeros((2, n))
    blocos, _ = epocas.blocos_entre_eventos(_eventos_oc(), fs, n, ALVOS)
    for a, b, v in blocos:
        dado[:, a:b] = 1.0 if v == "instructed_toOpenEyes" else 100.0

    for condicao, esperado in (("instructed_toOpenEyes", 1.0),
                               ("instructed_toCloseEyes", 100.0)):
        segs = [(a, b) for a, b, v in blocos if v == condicao]
        janelas, dec = epocas.epocar_janela_fixa(dado, fs, 2.0, 2.0, segs)
        assert len(janelas) > 0
        for ep in janelas:
            assert set(np.unique(ep).tolist()) == {esperado}
    # 2 blocos abertos de 20 s -> 20 épocas de 2 s; 2 fechados de 40 s -> 40
    abertos = [(a, b) for a, b, v in blocos if v == "instructed_toOpenEyes"]
    assert epocas.epocar_janela_fixa(dado, fs, 2.0, 2.0, abertos)[1]["n_epocas"] == 20


# ---------------------------------------------------------------------------
# os segmentos contínuos
# ---------------------------------------------------------------------------

def test_segmentos_sem_corte_e_a_gravacao_inteira():
    assert epocas.segmentos_continuos(1000, []) == [(0, 1000)]


def test_segmentos_com_um_corte():
    assert epocas.segmentos_continuos(1000, [400]) == [(0, 400), (400, 1000)]


def test_segmentos_com_cortes_fora_de_ordem_ou_repetidos():
    """O chamador junta cortes de duas fontes (mudança de ID e eventos); eles
    podem chegar desordenados e repetidos, e isso não é erro dele."""
    assert epocas.segmentos_continuos(100, [60, 20, 60]) == [(0, 20), (20, 60), (60, 100)]


def test_segmentos_ignora_corte_fora_da_faixa():
    assert epocas.segmentos_continuos(100, [0, 100, 150, -5]) == [(0, 100)]


# ---------------------------------------------------------------------------
# A GARANTIA CENTRAL
# ---------------------------------------------------------------------------

def test_nenhuma_epoca_atravessa_a_fronteira_entre_dois_sujeitos():
    """Dois sujeitos de amplitude conhecida, concatenados.

    O sujeito A vale 1,0 em toda amostra; o B vale 100,0. Uma época que
    cruzasse a fronteira teria as duas amplitudes dentro. Nenhuma pode ter.

    A fronteira cai em 250, que NÃO é múltiplo do passo (100) — de propósito:
    com fronteira alinhada ao passo, o defeito não apareceria mesmo se a
    guarda fosse removida.
    """
    fs = 100.0
    a = np.ones((3, 250)) * 1.0
    b = np.ones((3, 250)) * 100.0
    dado = np.concatenate([a, b], axis=1)

    segmentos = epocas.segmentos_continuos(dado.shape[1], [250])
    janelas, _ = epocas.epocar_janela_fixa(dado, fs, duracao_s=1.0, passo_s=1.0,
                                           segmentos=segmentos)

    assert len(janelas) > 0, "o teste não vale nada se não epocar nada"
    for i, ep in enumerate(janelas):
        valores = set(np.unique(ep).tolist())
        assert valores in ({1.0}, {100.0}), (
            f"a época {i} contém amostras dos DOIS sujeitos: {sorted(valores)}"
        )


def test_a_fronteira_tambem_vale_com_sobreposicao():
    """Sobreposição é onde o descuido reaparece: com passo menor que a
    duração, uma janela pode começar dentro de um segmento e terminar fora."""
    fs = 100.0
    dado = np.concatenate([np.ones((2, 250)), np.ones((2, 250)) * 100.0], axis=1)
    segmentos = epocas.segmentos_continuos(dado.shape[1], [250])
    janelas, _ = epocas.epocar_janela_fixa(dado, fs, duracao_s=1.0, passo_s=0.25,
                                           segmentos=segmentos)
    assert len(janelas) > 0
    for ep in janelas:
        assert len(set(np.unique(ep).tolist())) == 1


# ---------------------------------------------------------------------------
# geometria da janela
# ---------------------------------------------------------------------------

def test_formato_e_contagem_sem_sobreposicao():
    """1000 amostras a 100 Hz, janela de 1 s sem sobreposição = 10 épocas."""
    dado = np.arange(2 * 1000, dtype=float).reshape(2, 1000)
    janelas, dec = epocas.epocar_janela_fixa(dado, 100.0, 1.0, 1.0)
    assert janelas.shape == (10, 2, 100)
    assert dec["n_epocas"] == 10


def test_contagem_com_50_por_cento_de_sobreposicao():
    dado = np.zeros((1, 1000))
    janelas, _ = epocas.epocar_janela_fixa(dado, 100.0, 1.0, 0.5)
    # inícios em 0,50,...,900 -> 19 janelas de 100 amostras cabem em 1000
    assert len(janelas) == 19


def test_a_epoca_traz_as_amostras_certas_e_na_ordem():
    """Não basta o formato: os valores têm de ser a fatia certa do sinal."""
    dado = np.arange(10, dtype=float).reshape(1, 10)
    janelas, _ = epocas.epocar_janela_fixa(dado, 10.0, 0.5, 0.5)
    assert np.array_equal(janelas[0][0], np.array([0., 1., 2., 3., 4.]))
    assert np.array_equal(janelas[1][0], np.array([5., 6., 7., 8., 9.]))


def test_sobra_que_nao_completa_a_janela_e_descartada_e_contada():
    """1050 amostras com janela de 100: as 50 do fim não viram época pela
    metade — viram descarte declarado."""
    dado = np.zeros((1, 1050))
    janelas, dec = epocas.epocar_janela_fixa(dado, 100.0, 1.0, 1.0)
    assert len(janelas) == 10
    assert dec["amostras_descartadas"] == 50


def test_segmento_curto_demais_nao_vira_epoca_e_e_declarado():
    """Um segmento menor que a janela produz zero épocas — e o motivo aparece
    nas decisões, em vez de o segmento sumir sem explicação."""
    dado = np.zeros((1, 130))
    segmentos = epocas.segmentos_continuos(130, [100])   # (0,100) e (100,130)
    janelas, dec = epocas.epocar_janela_fixa(dado, 100.0, 1.0, 1.0, segmentos)
    assert len(janelas) == 1
    assert dec["segmentos_curtos_demais"] == 1


# ---------------------------------------------------------------------------
# as decisões — o registro que faz a receita ser possível
# ---------------------------------------------------------------------------

def test_decisoes_registram_os_parametros_usados():
    """`decisoes` é o que viaja para a receita e permite refazer igual. O
    comprimento e o passo escolhidos aqui são os que o tokenizador do
    transformer vai receber, então eles têm de sair nomeados."""
    dado = np.zeros((2, 500))
    _, dec = epocas.epocar_janela_fixa(dado, 100.0, 2.0, 1.0)
    assert dec["duracao_s"] == 2.0
    assert dec["passo_s"] == 1.0
    assert dec["fs"] == 100.0
    assert dec["amostras_por_epoca"] == 200
    assert dec["n_segmentos"] == 1


def test_parametro_invalido_recusa_em_vez_de_devolver_vazio():
    """Lista vazia silenciosa é o modo de falha que este projeto proíbe."""
    dado = np.zeros((1, 100))
    with pytest.raises(ValueError, match="duracao_s"):
        epocas.epocar_janela_fixa(dado, 100.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="passo_s"):
        epocas.epocar_janela_fixa(dado, 100.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="fs"):
        epocas.epocar_janela_fixa(dado, 0.0, 1.0, 1.0)


def test_janela_maior_que_a_gravacao_devolve_zero_com_motivo():
    dado = np.zeros((1, 50))
    janelas, dec = epocas.epocar_janela_fixa(dado, 100.0, 1.0, 1.0)
    assert len(janelas) == 0
    assert dec["n_epocas"] == 0
    assert dec["segmentos_curtos_demais"] == 1
