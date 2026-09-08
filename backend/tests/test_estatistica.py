# -*- coding: utf-8 -*-
"""O delta de Cliff e o Mann-Whitney têm de dar o que a definição diz."""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import estatistica


def test_delta_zero_para_amostras_iguais():
    assert estatistica.delta_de_cliff([1, 2, 3], [1, 2, 3]) == 0.0


def test_delta_mais_um_quando_todo_a_e_maior():
    assert estatistica.delta_de_cliff([10, 11, 12], [1, 2, 3]) == 1.0


def test_delta_menos_um_quando_todo_a_e_menor():
    assert estatistica.delta_de_cliff([1, 2, 3], [10, 11, 12]) == -1.0


def test_delta_e_antissimetrico():
    a, b = [1, 5, 9, 2], [3, 3, 7]
    assert estatistica.delta_de_cliff(a, b) == -estatistica.delta_de_cliff(b, a)


def test_delta_de_um_exemplo_contado_a_mao():
    """a=[1,2,3], b=[2,3,4]: um par com a>b (3>2), seis com a<b, dois empates;
    (1 − 6) / 9 = −5/9."""
    assert np.isclose(estatistica.delta_de_cliff([1, 2, 3], [2, 3, 4]), -5 / 9)


def test_mann_whitney_da_mesma_distribuicao_nao_e_significativo():
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, 60)
    b = rng.normal(0, 1, 60)
    r = estatistica.mann_whitney(a, b)
    assert r["p"] > 0.05
    assert r["n_a"] == 60 and r["n_b"] == 60
    assert r["nan_descartados"] == 0


def test_mann_whitney_de_distribuicoes_separadas_e_significativo():
    r = estatistica.mann_whitney(np.arange(30), np.arange(30) + 100)
    assert r["p"] < 1e-6


def test_nan_e_descartado_e_contado():
    r = estatistica.mann_whitney([1.0, np.nan, 3.0], [2.0, 4.0, np.nan, np.nan])
    assert r["nan_descartados"] == 3
    assert r["n_a"] == 2 and r["n_b"] == 2
    assert np.isnan(estatistica.delta_de_cliff([np.nan], [1.0]))


def test_mediana_iqr():
    r = estatistica.mediana_iqr([1, 2, 3, 4, 5, np.nan])
    assert r == {"mediana": 3.0, "q1": 2.0, "q3": 4.0, "n": 5}
    assert estatistica.mediana_iqr([np.nan])["n"] == 0
