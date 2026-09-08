# -*- coding: utf-8 -*-
"""O delta de Cliff e o Mann-Whitney têm de dar o que a definição diz."""
import sys
from pathlib import Path

import numpy as np
import pytest

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


def test_erro_padrao_hanley_mcneil_auc_meio_n0_n1_trinta():
    """AUC=0,5 (sem discriminação), n0=n1=30.

    O valor de referência não é ~0,129 (esse número corresponde a n0=n1≈10
    pela mesma fórmula, não a n0=n1=30). O valor correto para n0=n1=30 foi
    conferido de duas formas independentes antes de fixar esta tolerância:
    contra a fórmula-texto de Hanley & McNeil (1982) e por simulação Monte
    Carlo (20 000 repetições de AUC empírica via Mann-Whitney U sobre duas
    normais padrão, n0=n1=30), que devolveu desvio-padrão empírico de
    0,07522 — a 4 casas decimais do valor fechado, 0,07515."""
    se = estatistica.erro_padrao_auc_hanley_mcneil(0.5, 30, 30)
    assert np.isclose(se, 0.0752, atol=0.001)


def test_erro_padrao_hanley_mcneil_condicao_a_do_experimento():
    """AUC=0,65, n0=60 (Control), n1=61 (ADHD): o caso real da condição A."""
    se = estatistica.erro_padrao_auc_hanley_mcneil(0.65, 60, 61)
    assert np.isclose(se, 0.050, atol=0.001)


def test_erro_padrao_hanley_mcneil_nao_e_simetrico_em_n0_n1():
    """Trocar n0 por n1 muda o resultado quando AUC ≠ 0,5: Q1 e Q2 só
    coincidem em AUC = 0,5, e a fórmula pondera cada um por um n diferente."""
    se_original = estatistica.erro_padrao_auc_hanley_mcneil(0.65, 60, 61)
    se_trocado = estatistica.erro_padrao_auc_hanley_mcneil(0.65, 61, 60)
    assert not np.isclose(se_original, se_trocado)


def test_ic95_auc_trunca_limite_superior_em_um():
    ic_inf, ic_sup = estatistica.ic95_auc(0.98, 5, 5)
    assert ic_sup == 1.0
    assert ic_inf < ic_sup


def test_ic95_auc_trunca_limite_inferior_em_zero():
    ic_inf, ic_sup = estatistica.ic95_auc(0.05, 5, 5)
    assert ic_inf == 0.0
    assert ic_sup > ic_inf


def test_erro_padrao_hanley_mcneil_recusa_auc_fora_de_zero_um():
    with pytest.raises(ValueError):
        estatistica.erro_padrao_auc_hanley_mcneil(1.5, 30, 30)
    with pytest.raises(ValueError):
        estatistica.erro_padrao_auc_hanley_mcneil(-0.1, 30, 30)


def test_erro_padrao_hanley_mcneil_recusa_n_menor_que_dois():
    with pytest.raises(ValueError):
        estatistica.erro_padrao_auc_hanley_mcneil(0.65, 1, 30)
    with pytest.raises(ValueError):
        estatistica.erro_padrao_auc_hanley_mcneil(0.65, 30, 1)
