import numpy as np
import pandas as pd
import pytest

from csv_data import CANAIS_19, list_subjects, get_window, get_subject_raw


def _df_teste():
    linhas = []
    for sid, classe, n in [("suj1", "ADHD", 5), ("suj2", "Control", 3)]:
        for i in range(n):
            linha = {c: float(i) for c in CANAIS_19}
            linha["ID"] = sid
            linha["Class"] = classe
            linhas.append(linha)
    return pd.DataFrame(linhas)


def test_list_subjects():
    df = _df_teste()
    assert list_subjects(df) == [
        {"id": "suj1", "classe": "ADHD", "duracao_s": 5 / 128},
        {"id": "suj2", "classe": "Control", "duracao_s": 3 / 128},
    ]


def test_get_window_janela_valida():
    df = _df_teste()
    janela = get_window(df, "suj1", 0.0, 3 / 128)
    assert janela.shape == (19, 3)
    assert list(janela[0]) == [0.0, 1.0, 2.0]


def test_get_window_sujeito_inexistente():
    df = _df_teste()
    with pytest.raises(ValueError, match="sujeito não encontrado"):
        get_window(df, "suj999", 0.0, 1.0)


def test_get_window_fora_do_intervalo():
    df = _df_teste()
    with pytest.raises(ValueError, match="fora do intervalo"):
        get_window(df, "suj2", 10.0, 11.0)


def test_get_window_t_end_menor_que_t_start():
    df = _df_teste()
    with pytest.raises(ValueError, match="janela inválida"):
        get_window(df, "suj1", 1.0, 0.5)


def test_get_subject_raw():
    df = _df_teste()
    dados = get_subject_raw(df, "suj1")
    assert set(dados.keys()) == set(CANAIS_19)
    assert dados["Fp1"] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_get_subject_raw_inexistente():
    df = _df_teste()
    with pytest.raises(ValueError, match="sujeito não encontrado"):
        get_subject_raw(df, "naoexiste")
