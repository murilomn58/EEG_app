# -*- coding: utf-8 -*-
"""Trava a extração de características: potência de banda por canal.

POR QUE ESTES TESTES SÃO NUMÉRICOS E NÃO DE FORMATO

Uma extração de característica errada não quebra nada. Ela devolve uma matriz
do tamanho certo, o classificador treina, e a acurácia sai — só que o número
descreve outra coisa. Num experimento cujo resultado é uma comparação entre
condições, um viés constante nas características ainda produziria barras, e
as barras pareceriam certas.

Por isso cada teste aqui ancora num fato conhecido de fora: uma senoide de
frequência conhecida põe a potência na banda certa; a integral da densidade
espectral bate com a variância do sinal (Parseval); ruído com desvio conhecido
dá potência total conhecida.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import caracteristicas


FS = 128.0


def _epoca_senoide(freq, n_canais=2, dur_s=4.0, amp=1.0):
    t = np.arange(int(dur_s * FS)) / FS
    onda = amp * np.sin(2 * np.pi * freq * t)
    return np.tile(onda, (n_canais, 1))


# ---------------------------------------------------------------------------
# a potência cai na banda certa
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("freq,banda", [
    (2.0, "delta"), (6.0, "theta"), (10.0, "alpha"), (20.0, "beta"), (38.0, "gamma"),
])
def test_senoide_concentra_potencia_na_banda_dela(freq, banda):
    """O teste mais direto que existe: uma senoide de 10 Hz tem de aparecer
    em alpha, e não em delta."""
    epocas = np.array([_epoca_senoide(freq)])
    X, nomes = caracteristicas.potencia_de_banda(epocas, FS)

    # coluna de cada banda para o canal 0
    por_banda = {b: X[0, nomes.index(f"ch0_{b}")] for b in caracteristicas.BANDAS}
    dominante = max(por_banda, key=por_banda.get)
    assert dominante == banda, (
        f"senoide de {freq} Hz caiu em {dominante} e não em {banda}: {por_banda}"
    )


def test_a_banda_dominante_concentra_a_maior_parte_da_potencia():
    """Não basta ser a maior: uma senoide pura tem de pôr a maior parte da
    potência numa banda só. Se estiver espalhada, o janelamento ou a
    integração estão errados."""
    epocas = np.array([_epoca_senoide(10.0)])
    X, nomes = caracteristicas.potencia_de_banda(epocas, FS)
    por_banda = np.array([X[0, nomes.index(f"ch0_{b}")] for b in caracteristicas.BANDAS])
    assert por_banda.max() / por_banda.sum() > 0.9


# ---------------------------------------------------------------------------
# a escala é a certa
# ---------------------------------------------------------------------------

def test_potencia_total_bate_com_a_variancia_do_sinal():
    """Parseval. A soma da potência sobre todas as bandas de um ruído de
    banda larga tem de se aproximar da variância — é o teste que pega erro de
    normalização da janela, que é o defeito mais silencioso de uma PSD.

    Tolerância larga de propósito: as cinco bandas cobrem 1 a 45 Hz e não o
    espectro inteiro até o Nyquist de 64 Hz, então a soma é MENOR que a
    variância por construção. O que se afirma é a ordem de grandeza."""
    rs = np.random.RandomState(0)
    sinal = rs.randn(1, int(20 * FS)) * 3.0     # desvio 3 -> variancia 9
    X, nomes = caracteristicas.potencia_de_banda(np.array([sinal]), FS)
    total = sum(X[0, nomes.index(f"ch0_{b}")] for b in caracteristicas.BANDAS)
    variancia = sinal.var()
    assert 0.3 * variancia < total < 1.1 * variancia, (
        f"potência somada {total:.3f} contra variância {variancia:.3f}"
    )


def test_dobrar_a_amplitude_quadruplica_a_potencia():
    """Potência vai com o quadrado da amplitude. Se esta relação não valer, o
    que está sendo medido é amplitude com nome de potência — o mesmo erro que
    já foi encontrado e corrigido no frontend deste projeto."""
    um = caracteristicas.potencia_de_banda(np.array([_epoca_senoide(10.0, amp=1.0)]), FS)[0]
    dois = caracteristicas.potencia_de_banda(np.array([_epoca_senoide(10.0, amp=2.0)]), FS)[0]
    assert np.isclose(dois.max() / um.max(), 4.0, rtol=0.05)


# ---------------------------------------------------------------------------
# formato e nomes
# ---------------------------------------------------------------------------

def test_formato_e_19_canais_por_5_bandas():
    epocas = np.zeros((7, 19, int(2 * FS)))
    X, nomes = caracteristicas.potencia_de_banda(epocas, FS)
    assert X.shape == (7, 95)
    assert len(nomes) == 95


def test_os_nomes_identificam_canal_e_banda():
    """Sem nome de coluna, uma matriz 95 é ilegível — e a figura precisa
    poder dizer qual característica pesou."""
    epocas = np.zeros((1, 2, int(2 * FS)))
    _, nomes = caracteristicas.potencia_de_banda(epocas, FS, nomes_canais=["Fp1", "Cz"])
    assert nomes[0] == "Fp1_delta"
    assert "Cz_gamma" in nomes


def test_a_ordem_das_colunas_e_estavel():
    """A ordem entra na receita e no CSV. Se ela variasse entre execuções, um
    modelo treinado numa rodada não valeria na outra."""
    e = np.zeros((1, 3, int(2 * FS)))
    a = caracteristicas.potencia_de_banda(e, FS)[1]
    b = caracteristicas.potencia_de_banda(e, FS)[1]
    assert a == b


# ---------------------------------------------------------------------------
# recusa em vez de número sem sentido
# ---------------------------------------------------------------------------

def test_epoca_curta_demais_para_a_banda_mais_baixa_recusa():
    """Delta começa em 1 Hz: uma época de 0,5 s não contém um ciclo inteiro, e
    a potência de delta nela é um artefato do janelamento. Recusar é mais
    honesto que devolver o número."""
    epocas = np.zeros((1, 1, int(0.5 * FS)))
    with pytest.raises(ValueError, match="curta"):
        caracteristicas.potencia_de_banda(epocas, FS)


def test_sem_epoca_nenhuma_recusa():
    with pytest.raises(ValueError, match="vazio"):
        caracteristicas.potencia_de_banda(np.empty((0, 19, 256)), FS)


def test_decisoes_registram_as_bandas_e_a_janela():
    epocas = np.zeros((2, 2, int(4 * FS)))
    _, _, dec = caracteristicas.potencia_de_banda(epocas, FS, com_decisoes=True)
    assert dec["bandas"] == dict(caracteristicas.BANDAS)
    assert dec["fs"] == FS
    assert dec["n_caracteristicas"] == 10
    assert dec["metodo"] == "Welch/Hann, integração trapezoidal por banda"
