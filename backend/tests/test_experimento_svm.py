# -*- coding: utf-8 -*-
"""Trava as quatro condições do experimento de SVM.

POR QUE QUATRO E NÃO UMA

A condição A é a que foi pedida: SVM sobre TBR. Sozinha ela produz um número
que não se pode interpretar. B diz se a TBR está jogando informação fora; C diz
se o SVM ganha algo sobre a regressão logística que já existia; D diz se o
número bate o acaso.

O teste roda com poucos sujeitos e poucas permutações, porque o que ele trava é
a ESTRUTURA — que as quatro condições existem, que cada uma reporta AUC e n —
e não os valores, que dependem do banco inteiro.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import csv_data
import experimento_svm


@pytest.fixture
def df_sintetico():
    """Quatro sujeitos, 40 s cada, com theta deslocado num dos grupos.

    Sintético e não amostra do banco real: o CSV real tem 267 MB, e um teste que
    o lê deixa de ser teste e vira experimento."""
    rng = np.random.default_rng(0)
    fs = int(csv_data.FS)
    n = fs * 40
    t = np.arange(n) / fs
    linhas = []
    for i, (sid, classe) in enumerate(
        [("s1", "ADHD"), ("s2", "ADHD"), ("s3", "Control"), ("s4", "Control")]
    ):
        amp_theta = 3.0 if classe == "ADHD" else 1.0
        bloco = {}
        for c in csv_data.CANAIS_19:
            sinal = (amp_theta * np.sin(2 * np.pi * 6 * t)
                     + 1.0 * np.sin(2 * np.pi * 20 * t)
                     + rng.normal(0, 0.5, n))
            bloco[c] = sinal
        bloco["ID"] = sid
        bloco["Class"] = classe
        linhas.append(pd.DataFrame(bloco))
    return pd.concat(linhas, ignore_index=True)


def test_rodar_produz_as_quatro_condicoes(df_sintetico):
    resultados, receita = experimento_svm.rodar(
        df=df_sintetico, duracao_s=4.0, passo_s=4.0,
        n_permutacoes=3, n_dobras_internas=2, semente=0,
    )

    condicoes = {r["condicao"] for r in resultados}
    assert condicoes == {"A", "B", "C", "D"}
    for r in resultados:
        assert 0.0 <= r["auc"] <= 1.0
        assert r["n_sujeitos"] == 4
    assert receita["banco"] == "adhdata"
    assert "licença" in receita["notas"]


def test_a_condicao_d_reporta_p_empirico(df_sintetico):
    """O nulo sem p-valor é um histograma, não um teste."""
    resultados, _ = experimento_svm.rodar(
        df=df_sintetico, duracao_s=4.0, passo_s=4.0,
        n_permutacoes=3, n_dobras_internas=2, semente=0,
    )
    d = next(r for r in resultados if r["condicao"] == "D")
    assert 0.0 <= d["p_empirico"] <= 1.0


def test_a_e_b_rodam_sobre_o_mesmo_conjunto(df_sintetico):
    """A comparação entre TBR e potência de banda só vale no mesmo n.

    Se um sujeito falhar só numa das duas montagens, as duas condições passam a
    medir conjuntos diferentes, e a pergunta que a condição B existe para
    responder — a TBR joga informação fora? — deixa de ter resposta."""
    resultados, receita = experimento_svm.rodar(
        df=df_sintetico, duracao_s=4.0, passo_s=4.0,
        n_permutacoes=3, n_dobras_internas=2, semente=0,
    )
    a = next(r for r in resultados if r["condicao"] == "A")
    b = next(r for r in resultados if r["condicao"] == "B")
    assert a["n_sujeitos"] == b["n_sujeitos"], (
        "A e B mediram conjuntos de tamanhos diferentes: a comparação entre "
        "elas não é válida"
    )
