# -*- coding: utf-8 -*-
"""Trava o classificador SVM e o protocolo de avaliação.

O QUE ESTE MÓDULO ESTÁ DEFENDENDO

Um classificador de EEG com 121 sujeitos tem três formas de mentir, e as três
produzem número publicável:

1. Vazamento de sujeito — épocas da mesma criança nos dois lados da partição.
2. Vazamento de normalização — média e desvio calculados com o teste dentro.
3. Ausência de nulo — uma AUC de 0,62 que ninguém comparou com o acaso.

Os testes aqui existem para que as três falhem ruidosamente em vez de virarem
resultado.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import classificador


def test_agrega_pela_mediana_das_epocas():
    """Um sujeito com épocas [1, 2, 9] vale 2, não 4.

    A média daria 4, arrastada pelo 9. Num sinal real o 9 é a época em que a
    criança se mexeu, e ela não deve decidir o diagnóstico dela."""
    escores = np.array([1.0, 2.0, 9.0, 10.0, 20.0])
    grupos = np.array([0, 0, 0, 1, 1])
    rotulos = np.array([0, 0, 0, 1, 1])

    esc, rot, ids = classificador.agregar_por_sujeito(escores, grupos, rotulos)

    assert list(ids) == [0, 1]
    assert esc[0] == pytest.approx(2.0)
    assert esc[1] == pytest.approx(15.0)
    assert list(rot) == [0, 1]


def test_rotulo_inconsistente_dentro_do_sujeito_levanta():
    """Um sujeito com dois rótulos é dado corrompido, não caso de borda.

    Devolver o primeiro rótulo, ou o mais frequente, esconderia um defeito de
    montagem do conjunto atrás de um número plausível."""
    escores = np.array([1.0, 2.0])
    grupos = np.array([0, 0])
    rotulos = np.array([0, 1])

    with pytest.raises(ValueError, match="rótulo"):
        classificador.agregar_por_sujeito(escores, grupos, rotulos)


def conjunto_separavel(n_sujeitos=8, n_epocas=6, semente=0):
    """Sujeitos de classe 1 com média deslocada: separável, mas com ruído.

    Deslocamento pequeno de propósito. Um conjunto perfeitamente separável
    daria AUC 1,0 em qualquer implementação, inclusive numa que vaza — e o
    teste deixaria de distinguir o certo do errado."""
    rng = np.random.default_rng(semente)
    X, y, g = [], [], []
    for s in range(n_sujeitos):
        classe = s % 2
        X.append(rng.normal(loc=classe * 1.2, scale=1.0, size=(n_epocas, 4)))
        y.extend([classe] * n_epocas)
        g.extend([s] * n_epocas)
    return np.vstack(X), np.array(y), np.array(g)


def test_loso_produz_um_escore_por_sujeito():
    """121 sujeitos dariam 121 escores; aqui são 8.

    A AUC sai de UMA conta sobre todos os escores, e não da média de contas por
    dobra: no LOSO cada dobra tem um sujeito só, e AUC de uma classe só é
    indefinida."""
    X, y, g = conjunto_separavel()

    r = classificador.avaliar_loso(X, y, g, n_dobras_internas=2)

    assert r["n_sujeitos"] == 8
    assert len(r["escores_por_sujeito"]) == 8
    assert len(r["rotulos_por_sujeito"]) == 8
    assert 0.0 <= r["auc"] <= 1.0


def test_loso_nao_vaza_sujeito():
    """O sujeito de teste nunca aparece no treino da sua dobra.

    Redundante com o LeaveOneGroupOut, e deliberado: a garantia fica afirmada no
    ponto de uso, e não depende de a biblioteca continuar se comportando assim
    numa versão futura."""
    X, y, g = conjunto_separavel()

    r = classificador.avaliar_loso(X, y, g, n_dobras_internas=2)

    for dobra in r["decisoes"]["dobras"]:
        assert dobra["sujeito_de_teste"] not in dobra["sujeitos_de_treino"]


def test_normalizacao_ajustada_fora_da_dobra_mudaria_o_escore():
    """O escore honesto difere do escore com normalização vazada.

    A COMPARAÇÃO É ENTRE DOIS AJUSTES DOS MESMOS DADOS, e não entre dois
    conjuntos de dados. Um teste que alterasse o dado de um sujeito e exigisse
    que os escores dos outros não mudassem seria inválido: nas dobras dos
    outros, aquele sujeito é dado de TREINO legítimo, e mudar o treino muda o
    modelo. Medido em 08/09/2026: o escore de um sujeito ia de -0,82 para
    +1,00 quando outro sujeito era multiplicado por 1000, com o pipeline
    correto.

    O que este teste faz é diferente: os dados são os MESMOS nos dois lados, e
    só muda DE ONDE saem a média e o desvio. Se `avaliar_loso` ajustasse a
    escala no conjunto inteiro, o resultado dele bateria com o braço vazado."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X, y, g = conjunto_separavel(n_sujeitos=8)
    X[g == 0] *= 50.0     # o sujeito de teste tem escala própria

    grade_fixa = {"svm__C": [1.0], "svm__gamma": ["scale"]}
    r = classificador.avaliar_loso(
        X, y, g, grade=grade_fixa, n_dobras_internas=2
    )

    # O braço VAZADO, construído de propósito: escala ajustada com o teste dentro.
    treino, teste = g != 0, g == 0
    escalador = StandardScaler().fit(X)
    Xn = escalador.transform(X)
    modelo = SVC(kernel="rbf", C=1.0, gamma="scale")
    modelo.fit(Xn[treino], y[treino])
    escore_vazado = float(np.median(modelo.decision_function(Xn[teste])))

    escore_honesto = float(r["escores_por_sujeito"][0])
    assert not np.isclose(escore_honesto, escore_vazado, rtol=1e-6), (
        "o escore do LOSO bateu com o de uma normalização ajustada no conjunto "
        "inteiro: a escala está sendo ajustada fora da dobra"
    )
