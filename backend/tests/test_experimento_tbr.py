# -*- coding: utf-8 -*-
"""Os dois experimentos de TBR sobre sinal sintético de resposta conhecida."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import csv_data
import experimento_tbr_eventos as ev
import experimento_tbr_grupos as gr

FS = 128.0


def _seno(f_hz, dur_s, fs, rng, amp=20.0):
    t = np.arange(int(dur_s * fs)) / fs
    return amp * np.sin(2 * np.pi * f_hz * t) + rng.normal(0, 0.5, len(t))


def _df_sintetico(dur_s=30.0):
    """4 crianças: as 2 ADHD com 6 Hz forte (theta), as 2 controle com 20 Hz
    (beta). A TBR tem de sair maior no grupo ADHD em todo canal."""
    rng = np.random.default_rng(1)
    blocos = []
    for sid, classe, f in (("a1", "ADHD", 6.0), ("a2", "ADHD", 6.0),
                           ("c1", "Control", 20.0), ("c2", "Control", 20.0)):
        sinal = _seno(f, dur_s, FS, rng)
        df = pd.DataFrame({c: sinal + j for j, c in enumerate(csv_data.CANAIS_19)})
        df["ID"] = sid
        df["Class"] = classe
        blocos.append(df)
    return pd.concat(blocos, ignore_index=True)


def test_grupos_tbr_maior_no_adhd_sintetico():
    linhas, dec = gr.medir_sujeitos(_df_sintetico(), duracao_s=2.0, passo_s=2.0)
    assert len(linhas) == 4 * 19
    assert dec["n_por_classe"] == {"ADHD": 2, "Control": 2}
    assert dec["sujeitos_que_falharam"] == []
    comp = gr.comparar_grupos(linhas)
    assert len(comp) == 19
    for c in comp:
        assert c["adhd_mediana"] > c["control_mediana"], c["canal"]
        assert c["delta_cliff"] == 1.0
        assert c["n_adhd"] == 2 and c["n_control"] == 2


def test_grupos_sujeito_curto_aparece_como_falha():
    df = _df_sintetico()
    curto = df[df["ID"] == "a1"].iloc[:64].copy()   # 0,5 s: nenhuma época de 2 s cabe
    curto["ID"] = "curto"
    df = pd.concat([df, curto], ignore_index=True)
    _, dec = gr.medir_sujeitos(df)
    ids = {f["id"] for f in dec["sujeitos_que_falharam"]}
    assert "curto" in ids


def test_grupos_amostra_equilibrada_por_classe():
    sujeitos = [{"id": f"a{i}", "classe": "ADHD"} for i in range(5)] + \
               [{"id": f"c{i}", "classe": "Control"} for i in range(5)]
    escolhidos = gr.escolher_sujeitos(sujeitos, 2)
    assert {s["classe"] for s in escolhidos} == {"ADHD", "Control"}


def _eventos_hbn(fs):
    """Padrão medido: abre em t, fecha em t+20, abre em t+60; 2 ciclos e um
    último 'abre' sem par, mais um boundary dentro de um bloco fechado e
    eventos de outra tarefa no meio."""
    return [
        {"onset": 10.0, "valor": "resting_start"},
        {"onset": 10.0, "valor": "instructed_toOpenEyes"},
        {"onset": 30.0, "valor": "instructed_toCloseEyes"},
        {"onset": 50.0, "valor": "boundary"},
        {"onset": 55.0, "valor": "dot_no1_ON"},
        {"onset": 70.0, "valor": "instructed_toOpenEyes"},
        {"onset": 90.0, "valor": "instructed_toCloseEyes"},
        {"onset": 130.0, "valor": "instructed_toOpenEyes"},
    ]


def test_eventos_tbr_fechado_maior_que_aberto_e_contagens():
    fs = 100.0
    n = int(150 * fs)
    rng = np.random.default_rng(2)
    nomes = list(csv_data.CANAIS_19)
    dado = np.zeros((19, n))
    # abertos (10-30, 70-90) com 20 Hz; fechados (30-70, 90-130) com 6 Hz
    for a, b, f in ((10, 30, 20.0), (70, 90, 20.0), (30, 70, 6.0), (90, 130, 6.0)):
        i0, i1 = int(a * fs), int(b * fs)
        for c in range(19):
            dado[c, i0:i1] = _seno(f, b - a, fs, rng)

    res, dec = ev.medir_por_condicao(dado, fs, _eventos_hbn(fs), 2.0, 2.0, nomes)
    ab, fe = res[ev.ABERTO], res[ev.FECHADO]
    # 2 blocos abertos de 20 s -> 20 épocas; 2 fechados de 40 s -> 40 (o
    # boundary em 50 s recorta um bloco em dois de 20 s, sem perder época)
    assert ab["n_epocas"] == 20 and fe["n_epocas"] == 40
    assert ab["n_blocos"] == 2 and fe["n_blocos"] == 2
    assert dec["ultimo_alvo_sem_fechamento"] == 130.0
    med_ab = np.nanmedian(ab["tbr"], axis=0)
    med_fe = np.nanmedian(fe["tbr"], axis=0)
    assert np.all(med_fe > med_ab)


def test_eventos_comparar_condicoes_conta_pares_e_direcao():
    nomes = ["Cz", "Fz"]
    linhas = []
    for sid, f, a in (("s1", 5.0, 2.0), ("s2", 4.0, 3.0), ("s3", 1.0, 2.0)):
        for canal in nomes:
            linhas.append({"sujeito": sid, "condicao": ev.FECHADO, "canal": canal, "tbr_mediana": f})
            linhas.append({"sujeito": sid, "condicao": ev.ABERTO, "canal": canal, "tbr_mediana": a})
    comp = ev.comparar_condicoes(linhas, nomes)
    cz = next(c for c in comp if c["canal"] == "Cz")
    assert cz["n_pares"] == 3
    assert cz["k_fechado_maior"] == 2
    assert cz["dif_mediana"] == 1.0
    assert cz["p_wilcoxon"] is None   # n < 6: não se reporta p
