import numpy as np
import pandas as pd

FS = 128.0
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
    entrada inválida."""
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
