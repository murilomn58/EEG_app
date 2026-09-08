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


def test_a_b_c_tem_ic95_e_d_nao(df_sintetico):
    """A, B e C reportam erro-padrão e IC95% da AUC (Hanley-McNeil); D não.

    D não tem IC porque o "AUC" ali é a média da distribuição nula por
    permutação, não uma AUC de classificação normal."""
    resultados, _ = experimento_svm.rodar(
        df=df_sintetico, duracao_s=4.0, passo_s=4.0,
        n_permutacoes=3, n_dobras_internas=2, semente=0,
    )
    for cond in ("A", "B", "C"):
        r = next(x for x in resultados if x["condicao"] == cond)
        assert isinstance(r["erro_padrao"], float)
        assert isinstance(r["auc_ic95_inf"], float)
        assert isinstance(r["auc_ic95_sup"], float)
        assert r["auc_ic95_inf"] <= r["auc"] <= r["auc_ic95_sup"]

    d = next(x for x in resultados if x["condicao"] == "D")
    assert d["erro_padrao"] is None
    assert d["auc_ic95_inf"] is None
    assert d["auc_ic95_sup"] is None


def test_rodar_sem_progresso_nao_imprime(df_sintetico, capsys):
    """`rodar()` nunca imprime por conta própria, só via callback `progresso`.

    O padrão de `progresso=None` é o que preserva o comportamento de quem já
    chama `rodar()` sem esse parâmetro — e isso só é verdade se a função for
    muda quando ninguém pede sinal."""
    experimento_svm.rodar(
        df=df_sintetico, duracao_s=4.0, passo_s=4.0,
        n_permutacoes=2, n_dobras_internas=2, semente=0,
        progresso=None,
    )

    capturado = capsys.readouterr()
    assert capturado.out == ""


def test_progresso_anuncia_as_quatro_condicoes(df_sintetico):
    """A, B, C e D aparecem nas linhas de progresso, na ordem em que rodam.

    Uma rodada de horas sem esse sinal já foi lida, uma vez, como processo
    travado quando na verdade progredia — é exatamente isso que este teste
    impede de silenciar de novo."""
    linhas = []

    experimento_svm.rodar(
        df=df_sintetico, duracao_s=4.0, passo_s=4.0,
        n_permutacoes=2, n_dobras_internas=2, semente=0,
        progresso=linhas.append,
    )

    texto = "\n".join(linhas)
    pos_a = texto.find("condição A")
    pos_b = texto.find("condição B")
    pos_c = texto.find("condição C")
    pos_d = texto.find("condição D")
    assert pos_a != -1 and pos_b != -1 and pos_c != -1 and pos_d != -1, (
        "nem todas as quatro condições anunciaram progresso"
    )
    assert pos_a < pos_b < pos_c < pos_d, (
        "as condições não apareceram na ordem A, B, C, D"
    )


def test_checkpoint_existe_depois_da_primeira_condicao(df_sintetico, monkeypatch, tmp_path):
    """O checkpoint tem as linhas de A e B mesmo se C explodir no meio.

    Monkeypatcha `LogisticRegression` (o modelo específico da condição C) para
    levantar: é o ponto de falha mais robusto, porque não depende de contar
    chamadas de `avaliar_loso` na ordem certa — só depende de C ser a única
    condição que usa regressão logística."""
    import sklearn.linear_model as linear_model_mod

    def _explode(*a, **kw):
        raise RuntimeError("falha simulada na condição C")

    monkeypatch.setattr(linear_model_mod, "LogisticRegression", _explode)
    # `experimento_svm.rodar` importa `LogisticRegression` localmente de
    # `sklearn.linear_model` a cada chamada, então corrigir o atributo no
    # módulo de origem é suficiente.

    caminho_checkpoint = tmp_path / "checkpoint.csv"

    with pytest.raises(RuntimeError, match="falha simulada na condição C"):
        experimento_svm.rodar(
            df=df_sintetico, duracao_s=4.0, passo_s=4.0,
            n_permutacoes=2, n_dobras_internas=2, semente=0,
            checkpoint=caminho_checkpoint,
        )

    assert caminho_checkpoint.exists(), "o checkpoint não foi escrito antes da falha em C"
    with caminho_checkpoint.open(encoding="utf-8") as f:
        conteudo = f.read()
    condicoes_presentes = {
        linha.split(",")[0] for linha in conteudo.splitlines()[1:] if linha
    }
    assert condicoes_presentes == {"A", "B"}, (
        f"esperava só A e B no checkpoint, achou {condicoes_presentes}"
    )
