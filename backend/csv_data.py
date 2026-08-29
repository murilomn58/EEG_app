import numpy as np
import pandas as pd

FS = 128.0

# Teto EXPLÍCITO de duração para uma janela pedida, em segundos.
#
# Existe porque o custo de uma janela não está nas amostras que ela devolve —
# são poucos MB — e sim no que a solução inversa constrói a partir delas:
# n_vertices x n_amostras em float64, com as cópias intermediárias do MNE por
# cima. Sem teto, o tamanho do pedido é escolhido por quem chama e o servidor
# obedece até acabar a memória.
#
# MEDIDO nesta máquina, backend de pé, RSS lido do Win32_Process: um POST
# /source-localization com t_end=1e9 no sujeito v10p (111,75 s = 14.304
# amostras) devolveu 200 em 17,1 s e levou o working set do uvicorn de 1.878 MB
# para um pico de 10.334 MB. A conta explica o número: 20.484 vértices x 14.304
# amostras x 8 bytes já são 2,3 GB só no stc. Note que o 200 é o problema —
# ninguém vê nada de errado na tela, e o processo é que fica machucado.
#
# 30 s é folga deliberada sobre o uso real: o frontend pede 2 s, e 2 s são 256
# amostras (20.484 x 256 x 8 = 42 MB de stc).
#
# O CUSTO DO PRÓPRIO TETO, também medido, porque ele não é de graça: uma janela
# EXATAMENTE no teto (30 s, 3.840 amostras) responde em 2,2 s e leva o pico de
# working set do processo de 1.207 MB para 3.712 MB. Ou seja, ~2,5 GB
# transitórios, cerca de QUATRO vezes o stc de 20.484 x 3.840 x 8 = 629 MB — o
# MNE materializa cópias intermediárias, e a conta do stc sozinha subestima o
# custo real por esse fator. Quem for mexer neste número mexa sabendo disso:
# 30 s é o pior caso que este processo aceita, não um valor confortável.
#
# Um teto derivado do tamanho do arquivo não serviria: o maior sujeito do
# adhdata tem 338 s, e 338 s seguem sendo o cenário de 10 GB medido acima.
MAX_JANELA_S = 30.0

CANAIS_19 = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]


def load_csv(caminho):
    """Lê adhdata.csv inteiro (colunas dos 19 canais + ID + Class) uma vez
    só — o resultado fica guardado em app.state pelo FastAPI, não é
    relido a cada request."""
    return pd.read_csv(caminho, usecols=CANAIS_19 + ["ID", "Class"])


def list_subjects(df):
    """Um item por sujeito único, na ordem de primeira aparição no CSV."""
    contagens = df.groupby("ID", sort=False).size()
    classes = df.groupby("ID", sort=False)["Class"].first()
    return [
        {"id": sid, "classe": classes[sid], "duracao_s": contagens[sid] / FS}
        for sid in contagens.index
    ]


def get_window(df, subject_id, t_start, t_end):
    """Linhas do subject_id no intervalo [t_start, t_end) segundos,
    assumindo FS=128Hz. Devolve array (19, n_amostras) na ordem de
    CANAIS_19. Lança ValueError com mensagem descritiva em qualquer
    entrada inválida.

    POR QUE t_start NEGATIVO É RECUSADO, E NÃO APENAS "ESTRANHO". A fatia é
    feita com `.iloc`, e em Python índice negativo conta a partir do FIM. Um
    pedido de t=[-10, -5] não devolve erro nem devolve vazio: devolve os
    últimos 10 a 5 segundos da gravação, um trecho de sinal real e
    perfeitamente plausível, carimbado na resposta HTTP com "time": -10.0.
    MEDIDO contra o backend de pé, sujeito v10p (111,75 s): o POST com
    t=[-10, -5] voltou 200 com valores BIT A BIT idênticos aos de
    t=[101,75; 106,75]. Ou seja, o erro não aparece em lugar nenhum na tela —
    o médico lê um mapa de fonte de um instante que ele não pediu. Recusar na
    entrada é a única forma de isso não virar número na tela.

    O teto de janela (MAX_JANELA_S) é conferido sobre o que foi PEDIDO, e não
    sobre o que existe no arquivo, de propósito: quem pede 1e9 segundos
    precisa ouvir "não", e não receber a gravação inteira em silêncio."""
    # inf e NaN chegam por JSON como número perfeitamente válido — `1e999` vira
    # inf no parser, e o pydantic aceita os dois em campo float. Sem esta
    # conferência eles atravessam as duas checagens abaixo (NaN não é menor que
    # zero, e `t_end <= t_start` é False com NaN) e morrem em int(round(...)):
    # OverflowError com inf, ValueError com NaN. MEDIDO contra o backend de pé,
    # depois do teto de janela já instalado: um POST com t_end=1e999 devolveu
    # HTTP 500, porque OverflowError não é ValueError e escapa do except do
    # endpoint. É o único 500 que sobrou deste caminho.
    if not (np.isfinite(t_start) and np.isfinite(t_end)):
        raise ValueError(
            f"janela inválida: t_start ({t_start}) e t_end ({t_end}) têm de ser "
            f"números finitos"
        )

    if t_start < 0:
        raise ValueError(
            f"janela inválida: t_start ({t_start}) não pode ser negativo — "
            f"o tempo é contado do início da gravação"
        )

    if t_end <= t_start:
        raise ValueError(
            f"janela inválida: t_end ({t_end}) deve ser maior que t_start ({t_start})"
        )

    sujeito = df[df["ID"] == subject_id]
    if sujeito.empty:
        raise ValueError(f"sujeito não encontrado: {subject_id}")

    idx_start = int(round(t_start * FS))
    idx_end = int(round(t_end * FS))
    n_amostras = len(sujeito)

    if idx_start >= n_amostras:
        raise ValueError(
            f"janela fora do intervalo gravado: sujeito tem {n_amostras} amostras "
            f"({n_amostras / FS:.2f}s), pedido começa em {t_start:.2f}s"
        )

    # o teto olha o PEDIDO, antes de qualquer corte: se ele for aparado
    # primeiro, t_end=1e9 vira "a gravação inteira" e passa calado
    if idx_end - idx_start > MAX_JANELA_S * FS:
        raise ValueError(
            f"janela longa demais: pedidos {(idx_end - idx_start) / FS:.2f}s, "
            f"e o teto é {MAX_JANELA_S:.0f}s — a reconstrução de fonte custa "
            f"memória proporcional à duração da janela"
        )

    # e só então cortar o fim no que existe de verdade
    idx_end = min(idx_end, n_amostras)

    janela = sujeito.iloc[idx_start:idx_end]
    if janela.empty:
        raise ValueError("janela pedida não contém nenhuma amostra")

    return np.array([janela[c].to_numpy(dtype=float) for c in CANAIS_19])


def get_subject_raw(df, subject_id):
    """Toda a gravação do sujeito, um array por canal, na ordem de
    CANAIS_19 — usado por /raw-data pra tocar o EEG real (não sintético)
    no frontend. Lança ValueError se o sujeito não existir."""
    sujeito = df[df["ID"] == subject_id]
    if sujeito.empty:
        raise ValueError(f"sujeito não encontrado: {subject_id}")
    return {c: sujeito[c].tolist() for c in CANAIS_19}
