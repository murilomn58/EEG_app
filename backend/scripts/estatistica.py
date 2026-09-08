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


def erro_padrao_auc_hanley_mcneil(auc, n0, n1):
    """Erro-padrão da AUC de uma validação LOSO já rodada, pela aproximação de
    Hanley & McNeil (1982).

    POR QUE HANLEY-MCNEIL E NÃO BOOTSTRAP

    A validação LOSO deste experimento já levou ~8h. Um bootstrap sobre o
    resultado exigiria refazer a validação centenas de vezes para reamostrar
    os sujeitos — refazer o experimento, não analisar o que ele já produziu.
    Hanley-McNeil dá o erro-padrão em forma fechada, a partir só da AUC e das
    contagens de cada classe, sem retreinar nada.

    O QUE A APROXIMAÇÃO É E O QUE ELA NÃO É

    É assintótica (vale melhor com n grande) e trata a AUC como uma estatística
    U de Mann-Whitney, não como uma proporção simples: por isso ela NÃO é
    `auc*(1-auc)/n` (isso subestimaria a variância, porque ignora que a mesma
    AUC entra em muitos pares comparados, correlacionados entre si). Q1 e Q2
    capturam exatamente essa correlação entre pares que compartilham um dos
    dois sujeitos.

    n0 E n1 NÃO SÃO INTERCAMBIÁVEIS

    `n1` é a contagem da classe POSITIVA (aqui, sujeitos ADHD) e `n0` da classe
    NEGATIVA (Control). Trocar os dois muda o resultado: a fórmula não é
    simétrica em n0/n1 quando AUC ≠ 0,5, porque Q1 pondera os "empates do lado
    positivo" e Q2 os do lado negativo, e as duas quantidades só coincidem
    quando AUC = 0,5. Errar a ordem não quebra visivelmente (o número sai
    plausível), o que o torna o tipo de engano que passa despercebido — daí
    a validação cruzada ser explícita nos testes.
    """
    if not (0.0 <= auc <= 1.0):
        raise ValueError(
            f"AUC={auc!r} fora de [0, 1]: a fórmula de Hanley-McNeil não tem "
            "sentido para uma AUC que não é uma probabilidade."
        )
    if n0 < 2 or n1 < 2:
        raise ValueError(
            f"n0={n0!r}, n1={n1!r}: a aproximação assintótica de Hanley-McNeil "
            "exige pelo menos 2 sujeitos em cada classe para que Q1 e Q2 façam "
            "sentido como estimativas — com n menor que isso o erro-padrão "
            "resultante não sustenta interpretação."
        )
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    se2 = (
        auc * (1 - auc)
        + (n1 - 1) * (q1 - auc ** 2)
        + (n0 - 1) * (q2 - auc ** 2)
    ) / (n0 * n1)
    return float(np.sqrt(se2))


def ic95_auc(auc, n0, n1):
    """IC95% da AUC (`auc ± 1,96·erro-padrão`), truncado a [0, 1].

    O truncamento existe porque a aproximação normal de Hanley-McNeil pode
    devolver um limite fora de [0, 1] perto dos extremos (AUC alta com n
    pequeno, por exemplo) — e uma AUC, sendo uma probabilidade, não pode ter
    intervalo de confiança fora dessa faixa por definição."""
    erro_padrao = erro_padrao_auc_hanley_mcneil(auc, n0, n1)
    limite_inferior = max(0.0, auc - 1.96 * erro_padrao)
    limite_superior = min(1.0, auc + 1.96 * erro_padrao)
    return limite_inferior, limite_superior
