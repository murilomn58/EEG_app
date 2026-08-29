import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from verificar_referencia import diagnosticar

CANAIS = ["Fz", "Cz", "Pz", "O1"]


def _sinal(n_canais=4, n=2000, amp=1.5e-4, comum=1.1e-4, semente=0):
    """Sinal em VOLTS, que é a unidade que o MNE entrega. A escala importa:
    foi justamente ela que quebrou a primeira versão desta função, escrita
    com limiares em microvolts."""
    rng = np.random.RandomState(semente)
    d = rng.randn(n_canais, n) * amp
    if comum:
        d = d + comum * np.sin(np.linspace(0, 50, n))
    return d


def test_detecta_referencia_fisica_entre_os_canais():
    """O caso do HBN: o Cz é a referência e vem identicamente zero."""
    d = _sinal()
    d[1] = 0.0
    resultado = diagnosticar(d, CANAIS)
    assert resultado["referencia"] == "Cz"
    assert resultado["canais_flat"] == ["Cz"]


def test_detecta_referencia_externa():
    """O caso do adhdata: a referência é linked-ears, que não está entre os
    canais gravados, então nenhum canal é plano e a média instantânea é
    grande."""
    resultado = diagnosticar(_sinal(), CANAIS)
    assert resultado["referencia"] is None
    assert "externa" in resultado["evidencia"]


def test_detecta_car_ja_aplicado():
    d = _sinal()
    resultado = diagnosticar(d - d.mean(axis=0), CANAIS)
    assert resultado["referencia"] == "average"


def test_criterio_independe_da_escala():
    """A armadilha que motivou a refatoração: o mesmo sinal multiplicado por
    um fator grande precisa dar o MESMO veredito. Com limiar absoluto, dava
    respostas diferentes.

    O 1400 aqui é só um fator arbitrário de teste, e não uma afirmação sobre
    o HBN: a conclusão de que o HBN estaria numa escala arbitrária foi
    RETRATADA (o percentil que a sustentava é dominado por offset DC). O que
    este teste trava é a invariância de escala do critério, que vale para
    qualquer fator."""
    d = _sinal()
    assert diagnosticar(d, CANAIS)["referencia"] == diagnosticar(d * 1400, CANAIS)["referencia"]

    car = d - d.mean(axis=0)
    assert diagnosticar(car, CANAIS)["referencia"] == diagnosticar(car * 1400, CANAIS)["referencia"]


def test_sinal_todo_zero_recusa_o_veredito():
    """Gravacao morta nao recebe diagnostico de referencia.

    A VERSAO ANTERIOR DESTE TESTE NAO PODIA FALHAR. Ela era:

        assert resultado["referencia"] is None or resultado["canais_flat"] == []

    Com sinal todo zero, `tipico` e 0, entao `flat` sai vazio e o SEGUNDO
    termo do `or` e sempre verdadeiro — a assercao valia para qualquer
    resposta. E escondia o defeito real: `diagnosticar` devolvia
    referencia='average' com a evidencia "CAR aparentemente ja aplicado"
    para uma gravacao identicamente morta.

    Agora afirma as duas coisas separadamente, e a evidencia tem de explicar
    a causa em vez de afirmar um veredito."""
    resultado = diagnosticar(np.zeros((4, 100)), CANAIS)
    assert resultado["referencia"] is None, (
        f"sinal morto recebeu veredito {resultado['referencia']!r}: {resultado['evidencia']}"
    )
    assert resultado["canais_flat"] == []
    assert "desvio zero" in resultado["evidencia"]
    assert "CAR" not in resultado["evidencia"], "nao pode afirmar CAR sobre sinal sem conteudo"


def test_sinal_quase_zero_tambem_recusa():
    """Nao basta tratar o zero exato: ruido de quantizacao residual daria
    `tipico` minusculo e a razao voltaria a cair na faixa do CAR."""
    quase = np.zeros((4, 100))
    quase[0, 0] = 1e-30
    resultado = diagnosticar(quase, CANAIS)
    assert resultado["referencia"] is None
