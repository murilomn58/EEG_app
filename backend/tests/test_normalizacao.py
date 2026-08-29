# -*- coding: utf-8 -*-
"""Trava a separação entre ajustar e aplicar.

POR QUE ISSO É UM TESTE E NÃO UMA CONVENÇÃO

`fit_transform` sobre o conjunto inteiro é uma linha só, parece organizado, e
vaza: a média e o desvio usados para normalizar o TREINO passam a carregar
informação do TESTE. O classificador vê, indiretamente, dados que não deveria
ver, e a acurácia sobe.

Esse é o segundo eixo da figura do vazamento — o primeiro é o split. A
diferença entre eles importa: o vazamento por split é grosseiro e conhecido;
o de normalização é sutil e sobrevive em código que parece correto.

A defesa deste módulo é de TIPO, não de disciplina: `aplicar` exige um
`params`, e `params` só nasce de `ajustar`. Ajustar no conjunto inteiro
continua possível — e continua sendo a condição de controle da figura — mas
passa a ser uma linha visível que alguém escreveu de propósito.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import normalizacao


# ---------------------------------------------------------------------------
# a separação
# ---------------------------------------------------------------------------

def test_ajustar_devolve_parametros_e_nao_dados():
    """Se `ajustar` devolvesse os dados transformados, ele seria um
    `fit_transform` com outro nome e a separação não existiria."""
    X = np.array([[1.0, 10.0], [3.0, 30.0]])
    p = normalizacao.ajustar(X)
    assert isinstance(p, dict)
    assert "media" in p and "desvio" in p
    assert np.allclose(p["media"], [2.0, 20.0])


def test_aplicar_exige_parametros():
    """Não há caminho que normalize sem alguém ter ajustado antes."""
    X = np.array([[1.0], [3.0]])
    with pytest.raises(TypeError):
        normalizacao.aplicar(X)


def test_nao_existe_fit_transform():
    """A ausência é a defesa. Se alguém acrescentar o atalho, este teste cai
    e a conversa acontece antes de o atalho virar hábito."""
    assert not hasattr(normalizacao, "fit_transform")
    assert not hasattr(normalizacao, "ajustar_e_aplicar")


# ---------------------------------------------------------------------------
# a matemática
# ---------------------------------------------------------------------------

def test_z_score_por_caracteristica():
    X = np.array([[1.0], [2.0], [3.0]])
    Z = normalizacao.aplicar(X, normalizacao.ajustar(X))
    assert np.isclose(Z.mean(), 0.0)
    assert np.isclose(Z.std(), 1.0)


def test_cada_coluna_e_normalizada_por_si():
    """Colunas com escalas muito diferentes — potência de delta contra
    potência de gamma é exatamente esse caso — não podem partilhar média."""
    X = np.array([[1.0, 1000.0], [2.0, 2000.0], [3.0, 3000.0]])
    Z = normalizacao.aplicar(X, normalizacao.ajustar(X))
    assert np.allclose(Z[:, 0], Z[:, 1])


def test_desvio_zero_nao_vira_divisao_por_zero():
    """Canal morto tem desvio zero. Dividir por ele dá inf/NaN, que atravessa
    o classificador inteiro e só aparece no resultado."""
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    Z = normalizacao.aplicar(X, normalizacao.ajustar(X))
    assert np.all(np.isfinite(Z))
    assert np.allclose(Z[:, 0], 0.0)


# ---------------------------------------------------------------------------
# o que a separação existe para permitir
# ---------------------------------------------------------------------------

def test_parametros_do_treino_aplicados_ao_teste():
    """O uso correto: ajusta no treino, aplica nos dois. O teste é
    normalizado por estatística que NÃO veio dele."""
    treino = np.array([[0.0], [2.0]])          # média 1, desvio 1
    teste = np.array([[4.0]])
    p = normalizacao.ajustar(treino)
    assert np.isclose(normalizacao.aplicar(teste, p)[0, 0], 3.0)


def test_ajustar_no_conjunto_todo_da_resultado_diferente():
    """A prova de que o vazamento MUDA o número — sem isso, a figura estaria
    comparando duas coisas iguais.

    Mesmo dado de teste, duas origens de parâmetro: só do treino, ou de
    treino+teste juntos. Os valores normalizados têm de diferir."""
    treino = np.array([[0.0], [2.0]])
    teste = np.array([[10.0]])
    so_treino = normalizacao.aplicar(teste, normalizacao.ajustar(treino))
    vazado = normalizacao.aplicar(teste, normalizacao.ajustar(np.vstack([treino, teste])))
    assert not np.isclose(so_treino[0, 0], vazado[0, 0])


def test_decisoes_registram_a_origem_dos_parametros():
    """A receita precisa dizer de ONDE saiu a estatística. Sem esse campo, os
    dois braços do experimento produzem receitas idênticas — e a figura
    perderia a única coisa que a distingue."""
    X = np.random.RandomState(0).randn(20, 4)
    p = normalizacao.ajustar(X, origem="treino da dobra 2")
    dec = normalizacao.decisoes(p)
    assert dec["origem"] == "treino da dobra 2"
    assert dec["n_amostras_do_ajuste"] == 20
    assert dec["n_caracteristicas"] == 4
    assert dec["metodo"] == "z-score por característica"


def test_aplicar_recusa_formato_incompativel():
    """Ajustar com 4 características e aplicar em 5 é erro de quem chama, e
    silenciar isso produziria números sem significado."""
    p = normalizacao.ajustar(np.zeros((10, 4)))
    with pytest.raises(ValueError, match="característica"):
        normalizacao.aplicar(np.zeros((3, 5)), p)
