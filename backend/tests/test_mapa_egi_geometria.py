"""Confere o mapa EGI->10-20 pela GEOMETRIA, contra a digitalizacao real.

Isto e verificacao INDEPENDENTE da fonte. O mapa em config.MAPA_EGI_1020 foi
transcrito de dois documentos do fabricante; transcrever 19 pares a mao e
exatamente o tipo de tarefa onde um digito troca sem ninguem notar, e um
E92 no lugar de E62 nao levanta erro nenhum — so move um eletrodo para o
outro lado da cabeca e deixa a analise plausivel e errada.

O metodo e o mesmo de figuras_limpeza.conferir_mapa, generalizado dos 4
eletrodos de linha media para os 19: se o mapa esta certo, os canais caem
onde o nome deles promete.

ATENCAO ao que estes testes NAO afirmam: nao provam que o mapa e o oficial,
so que e geometricamente coerente. Um mapa errado mas simetrico passaria.
Servem para pegar erro de transcricao, nao para dispensar a fonte.
"""
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import config
import eletrodos
import preproc_basico
from canais import resolver

# Tolerancia larga de proposito: a correspondencia EGI e aproximada (o
# fabricante informa desvios de ate 2,5 cm). O que se afirma e lado e faixa,
# nao precisao angular.
TOL_LINHA_MEDIA = 15.0   # graus de |phi| aceitos como "sobre a linha media"
TOL_SIMETRIA = 20.0      # graus de diferenca aceitos entre um par esquerda/direita


@pytest.fixture(scope="module")
def posicoes():
    raiz = config.caminho_release("ds005505")
    sets = sorted(raiz.glob("sub-*/eeg/*task-RestingState*.set"))
    if not sets:
        pytest.skip("release ds005505 nao disponivel")
    raw = preproc_basico.carregar_raw(sets[0])

    resolvido, faltantes = resolver(
        raw.ch_names, config.CANAIS_10_20, config.MAPA_EGI_1020
    )
    assert faltantes == [], f"mapa nao cobre: {faltantes}"

    d = eletrodos.descrever(raw, resolvido)
    por_nome = {e["nome"]: e for e in d["eletrodos"]}
    return {alvo: por_nome[arq]["pos"] for alvo, arq in resolvido.items()}


@pytest.mark.parametrize("canal", ["Fz", "Cz", "Pz"])
def test_canais_de_linha_media_ficam_na_linha_media(posicoes, canal):
    """Fz, Cz e Pz sao sagitais por definicao. Se um deles sai da linha
    media, o par que o produziu esta trocado."""
    phi = posicoes[canal]["phi"]
    # a linha media e phi ~0 (frente) ou ~180 (tras); Cz no topo tem phi
    # indefinido na pratica, entao o teste de theta abaixo e que vale por ele
    desvio = min(abs(phi), abs(abs(phi) - 180.0))
    assert desvio < TOL_LINHA_MEDIA, f"{canal} em phi={phi}: fora da linha media"


def test_cz_fica_no_topo(posicoes):
    assert posicoes["Cz"]["theta"] < 10.0


@pytest.mark.parametrize("esq,dir_", [
    ("Fp1", "Fp2"), ("F3", "F4"), ("F7", "F8"),
    ("C3", "C4"), ("T7", "T8"),
    ("P3", "P4"), ("P7", "P8"), ("O1", "O2"),
])
def test_pares_sao_espelhados(posicoes, esq, dir_):
    """Todo par 10-20 impar/par e esquerda/direita. Este e o teste que pega
    troca de digito: um E92 no lugar de E52 quebra a simetria na hora."""
    pe, pd = posicoes[esq], posicoes[dir_]
    assert pe["phi"] < 0, f"{esq} deveria estar a esquerda (phi<0), esta em {pe['phi']}"
    assert pd["phi"] > 0, f"{dir_} deveria estar a direita (phi>0), esta em {pd['phi']}"
    assert abs(abs(pe["phi"]) - abs(pd["phi"])) < TOL_SIMETRIA, \
        f"{esq}/{dir_} assimetricos: {pe['phi']} vs {pd['phi']}"
    assert abs(pe["theta"] - pd["theta"]) < TOL_SIMETRIA, \
        f"{esq}/{dir_} em alturas diferentes: {pe['theta']} vs {pd['theta']}"


@pytest.mark.parametrize("frontal,posterior", [
    ("Fp1", "O1"), ("Fp2", "O2"), ("F3", "P3"), ("F4", "P4"), ("Fz", "Pz"),
])
def test_frontais_ficam_na_frente_dos_posteriores(posicoes, frontal, posterior):
    """|phi| cresce da frente (0) para tras (180). Trocar um F por um P
    inverteria a topografia inteira sem quebrar simetria nenhuma."""
    assert abs(posicoes[frontal]["phi"]) < abs(posicoes[posterior]["phi"]), \
        f"{frontal} nao esta na frente de {posterior}"
