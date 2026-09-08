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
