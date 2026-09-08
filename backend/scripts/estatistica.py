# -*- coding: utf-8 -*-
"""Três medidas de comparação entre grupos, sem nenhuma escolha escondida.

POR QUE UM MÓDULO PRÓPRIO E NÃO TRÊS CHAMADAS SOLTAS

O frontend afirma, num texto, que "a TBR em Cz saiu menor no grupo TDAH,
mediana 3,35 contra 3,57, p = 0,19 (Mann-Whitney), delta de Cliff −0,139".
Nenhum script do repositório produz esse número. Este módulo existe para que
o número seguinte tenha um lugar de origem que se possa reler: o teste, a
alternativa, o tratamento do NaN e o sinal do tamanho de efeito estão
escritos aqui, uma vez, e os experimentos só chamam.

NaN é descartado e CONTADO, nunca silenciado: um canal sem beta devolve NaN
em `razao_theta_beta` de propósito, e a estatística tem de dizer quantos
sujeitos saíram da conta por isso.

O delta de Cliff é calculado pela contagem direta de pares, e não por uma
biblioteca: com 61 × 60 = 3 660 pares a conta é trivial, e a definição fica
visível. Sinal positivo significa que `a` tende a ser MAIOR que `b`.
"""
import numpy as np
from scipy import stats


def _limpar(x):
    """(valores sem NaN, quantos NaN saíram)."""
    arr = np.asarray(x, dtype=float).ravel()
    ok = ~np.isnan(arr)
    return arr[ok], int((~ok).sum())


def mann_whitney(a, b):
    """U bilateral de Mann-Whitney entre duas amostras independentes."""
    a, na_nan = _limpar(a)
    b, nb_nan = _limpar(b)
    saida = {
        "n_a": int(len(a)), "n_b": int(len(b)),
        "nan_descartados": na_nan + nb_nan,
        "metodo": "scipy.stats.mannwhitneyu, bilateral",
    }
    if len(a) == 0 or len(b) == 0:
        saida.update({"U": None, "p": None})
        return saida
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    saida.update({"U": float(res.statistic), "p": float(res.pvalue)})
    return saida


def delta_de_cliff(a, b):
    """(#pares a>b − #pares a<b) / (n_a · n_b), em [−1, 1]. NaN para vazio."""
    a, _ = _limpar(a)
    b, _ = _limpar(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(np.sign(a[:, None] - b[None, :]).mean())


def mediana_iqr(x):
    """Mediana, primeiro e terceiro quartis, e n sem NaN."""
    v, _ = _limpar(x)
    if len(v) == 0:
        return {"mediana": None, "q1": None, "q3": None, "n": 0}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"mediana": float(med), "q1": float(q1), "q3": float(q3), "n": int(len(v))}
