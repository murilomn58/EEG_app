# -*- coding: utf-8 -*-
"""Transforma épocas em características: potência de banda por canal.

O QUE ENTRA E O QUE SAI

Entra (n_epocas, n_canais, n_amostras). Sai (n_epocas, n_canais × n_bandas),
mais a lista de nomes das colunas. Com os 19 canais de análise e as 5 bandas,
são 95 características por época.

POR QUE A PSD VEM DO MNE E NÃO É ESCRITA AQUI

`psd_array_welch` é a mesma função que `preproc_basico` usa para detectar a
frequência da rede e que `figuras_limpeza` usa para os espectros das figuras.
Reimplementar Welch aqui criaria uma segunda PSD no mesmo projeto — e o
projeto já aprendeu, no `qc_relatorio`, que medir com outra implementação faz
o resultado deixar de ser evidência sobre o que roda de verdade.

POTÊNCIA, E NÃO AMPLITUDE

A integral da densidade espectral sobre a faixa é potência, em µV². Isso não
é preciosismo de nomenclatura: este projeto já teve um defeito em que o RMS
da saída de um filtro era chamado de potência, e o efeito medido foi a razão
theta/beta sair comprimida a ponto de o limiar clássico de 4,0 ficar
inalcançável. O teste `test_dobrar_a_amplitude_quadruplica_a_potencia` existe
para que esse erro não volte por outra porta.
"""
import numpy as np
from mne.time_frequency import psd_array_welch

# `np.trapz` foi renomeada para `np.trapezoid` no numpy 2.0, e a antiga passou
# a emitir DeprecationWarning. O `requirements.txt` nao fixa versao de numpy,
# entao as duas aparecem em maquinas diferentes — e a diferenca so aparece
# quando alguem roda o experimento, nao quando escreve o codigo.
_integral = getattr(np, "trapezoid", None) or np.trapz

# As mesmas cinco faixas que o resto do projeto usa. Dicionário ordenado — a
# ordem entra nos nomes das colunas, no CSV e na receita, e um modelo
# treinado com uma ordem não vale com outra.
BANDAS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _integrar(psd, freqs, lo, hi):
    """Potência na faixa [lo, hi], por integração trapezoidal da densidade.

    Trapézio e não soma simples: a densidade é por Hz, e somar as raias
    ignoraria o espaçamento entre elas — o que faz o resultado depender do
    comprimento da janela em vez de só do sinal."""
    dentro = (freqs >= lo) & (freqs <= hi)
    if not np.any(dentro):
        return 0.0
    return float(_integral(psd[dentro], freqs[dentro]))


def potencia_de_banda(epocas, fs, nomes_canais=None, com_decisoes=False):
    """(X, nomes) ou (X, nomes, decisoes) com a potência de cada banda.

    Recusa época curta demais em vez de devolver o número: delta começa em
    1 Hz, e uma janela que não contém um ciclo inteiro produz uma potência de
    delta que é artefato do janelamento, não do sinal. Devolver esse número
    seria pior que parar, porque ele entra na matriz sem nenhuma marca."""
    epocas = np.asarray(epocas, dtype=float)
    if epocas.ndim != 3:
        raise ValueError(
            f"epocas tem de ser (n_epocas, n_canais, n_amostras), recebi {epocas.shape}"
        )
    n_epocas, n_canais, n_amostras = epocas.shape
    if n_epocas == 0:
        raise ValueError("conjunto de épocas vazio: não há o que caracterizar")

    menor_freq = min(lo for lo, _ in BANDAS.values())
    dur_s = n_amostras / fs
    if dur_s < 1.0 / menor_freq:
        raise ValueError(
            f"época curta demais: {dur_s:.3f} s não contém um ciclo de "
            f"{menor_freq} Hz, e a potência de delta nela seria artefato do "
            f"janelamento"
        )

    if nomes_canais is None:
        nomes_canais = [f"ch{i}" for i in range(n_canais)]
    if len(nomes_canais) != n_canais:
        raise ValueError(
            f"{len(nomes_canais)} nomes para {n_canais} canais"
        )

    fmax = min(fs / 2.0, max(hi for _, hi in BANDAS.values()) + 5.0)
    psd, freqs = psd_array_welch(
        epocas, sfreq=fs, fmin=0.0, fmax=fmax,
        n_fft=n_amostras, n_per_seg=n_amostras, window="hann", verbose=False,
    )

    nomes = [f"{c}_{b}" for c in nomes_canais for b in BANDAS]
    X = np.empty((n_epocas, n_canais * len(BANDAS)), dtype=float)
    for e in range(n_epocas):
        col = 0
        for c in range(n_canais):
            for lo, hi in BANDAS.values():
                X[e, col] = _integrar(psd[e, c], freqs, lo, hi)
                col += 1

    if not com_decisoes:
        return X, nomes

    return X, nomes, {
        "metodo": "Welch/Hann, integração trapezoidal por banda",
        "fs": float(fs),
        "bandas": dict(BANDAS),
        "unidade": "µV² (se a entrada estiver em µV)",
        "duracao_epoca_s": float(dur_s),
        "n_epocas": int(n_epocas),
        "n_canais": int(n_canais),
        "n_caracteristicas": int(n_canais * len(BANDAS)),
        "resolucao_hz": float(freqs[1] - freqs[0]) if len(freqs) > 1 else None,
    }
