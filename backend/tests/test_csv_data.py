import numpy as np
import pandas as pd
import pytest

from csv_data import (
    CANAIS_19,
    MAX_JANELA_S,
    list_subjects,
    get_window,
    get_subject_raw,
)


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


# --- teto de janela e tempo negativo ------------------------------------
# Os dois testes abaixo travam consertos que vieram de uma auditoria hostil
# contra o backend vivo. Nos dois casos o pedido devolvia 200 e um número
# plausível, que é a razão de eles terem sobrevivido tanto tempo.


def _df_longo(n_amostras):
    """Um sujeito só, com n_amostras linhas numeradas — o valor de cada
    linha é o próprio índice, e é isso que permite provar QUAL trecho
    voltou (e não só que voltou alguma coisa)."""
    linhas = []
    for i in range(n_amostras):
        linha = {c: float(i) for c in CANAIS_19}
        linha["ID"] = "suj1"
        linha["Class"] = "ADHD"
        linhas.append(linha)
    return pd.DataFrame(linhas)


def test_get_window_recusa_t_start_negativo():
    """A auditoria pediu t=[-10,-5] e recebeu 200 com o FIM da gravação,
    carimbado como "time": -10.0. É indexação negativa de lista Python."""
    df = _df_longo(1280)  # 10 s a 128 Hz
    with pytest.raises(ValueError, match="não pode ser negativo"):
        get_window(df, "suj1", -10.0, -5.0)


def test_get_window_negativo_nao_devolve_o_trecho_final():
    """O teste que prova QUAL era o dano, e não só que agora dá erro.

    Sem a recusa, iloc[-1280:-640] devolve os segundos 5 a 7,5 de uma
    gravação de 10 s: sinal real, do sujeito certo, do instante errado, e
    nada na resposta denunciando. Aqui a asserção é que essa igualdade não
    tem mais como acontecer — o pedido negativo morre antes de fatiar."""
    df = _df_longo(1280)
    fim_da_gravacao = get_window(df, "suj1", 5.0, 7.5)
    assert fim_da_gravacao[0][0] == 640.0  # o trecho que a fatia negativa devolvia

    with pytest.raises(ValueError):
        get_window(df, "suj1", -5.0, -2.5)


def test_get_window_recusa_janela_acima_do_teto():
    """O pedido de t_end=1e9 que levou o pico de working set do uvicorn de
    1.878 MB para 10.334 MB — e devolveu 200."""
    df = _df_longo(1280)
    with pytest.raises(ValueError, match="janela longa demais"):
        get_window(df, "suj1", 0.0, 1e9)


def test_teto_e_medido_no_pedido_e_nao_no_que_existe():
    """A armadilha do conserto: se o fim for aparado em n_amostras ANTES da
    conferência, t_end=1e9 vira "a gravação inteira" e passa calado. Este
    sujeito tem 10 s — o pedido gigante tem de morrer mesmo assim."""
    df = _df_longo(1280)
    with pytest.raises(ValueError, match="janela longa demais"):
        get_window(df, "suj1", 0.0, MAX_JANELA_S * 2)


def test_get_window_no_teto_ainda_passa():
    """O teto não pode reprovar o uso legítimo: 30 s é folga sobre os 2 s
    que o frontend pede, e uma janela exatamente no teto é válida."""
    df = _df_longo(int(MAX_JANELA_S * 128) + 10)
    janela = get_window(df, "suj1", 0.0, MAX_JANELA_S)
    assert janela.shape == (19, int(MAX_JANELA_S * 128))


def test_get_window_corta_o_fim_no_que_existe():
    """Janela dentro do teto, mas que termina depois do fim da gravação:
    devolve o que existe, sem inventar amostra e sem levantar erro."""
    df = _df_longo(300)
    janela = get_window(df, "suj1", 0.0, 5.0)  # 640 amostras pedidas, 300 existem
    assert janela.shape == (19, 300)
    assert janela[0][-1] == 299.0


def test_get_window_recusa_infinito_e_nan():
    """MEDIDO contra o backend vivo, já com o teto de janela instalado: um
    POST com t_end=1e999 devolveu HTTP 500. `1e999` vira inf no parser de
    JSON, o pydantic aceita, e int(round(inf * 128)) levanta OverflowError —
    que não é ValueError e escapa do except do endpoint. NaN escapa das duas
    conferências pelo mesmo motivo: nenhuma comparação com NaN é verdadeira."""
    df = _df_longo(1280)
    for valor in (float("inf"), float("nan")):
        with pytest.raises(ValueError, match="números finitos"):
            get_window(df, "suj1", 0.0, valor)
        with pytest.raises(ValueError, match="números finitos"):
            get_window(df, "suj1", valor, 2.0)
