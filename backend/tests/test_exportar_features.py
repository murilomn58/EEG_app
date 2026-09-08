# -*- coding: utf-8 -*-
"""Trava o contrato do CSV de features entre o app e o eeg_transformer.

O QUE ESTE CONTRATO EXISTE PARA GARANTIR

O README do eeg_transformer diz que o app "prepara os dados que este projeto
consome", e nenhum formato tinha sido definido. Um CSV que pareça certo e não
traga a fronteira de sujeito é pior que nenhum: o outro projeto treina, publica
um número, e o vazamento só aparece quando alguém pergunta.

A coluna que carrega essa garantia é `sujeito_id`, e ela tem de trazer o ID
ORIGINAL do banco, não o índice interno da montagem.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import csv_data
import exportar_features


@pytest.fixture
def df_sintetico():
    """Dois sujeitos, 40 s cada — o bastante para montar o conjunto TBR.

    Sintético e não amostra do banco real: o CSV real tem 267 MB, e um teste
    que o lê deixa de ser teste e vira experimento."""
    rng = np.random.default_rng(0)
    fs = int(csv_data.FS)
    n = fs * 40
    t = np.arange(n) / fs
    linhas = []
    for sid, classe in [("s1", "ADHD"), ("s2", "Control")]:
        amp_theta = 3.0 if classe == "ADHD" else 1.0
        bloco = {}
        for c in csv_data.CANAIS_19:
            sinal = (amp_theta * np.sin(2 * np.pi * 6 * t)
                     + 1.0 * np.sin(2 * np.pi * 20 * t)
                     + rng.normal(0, 0.5, n))
            bloco[c] = sinal
        bloco["ID"] = sid
        bloco["Class"] = classe
        linhas.append(pd.DataFrame(bloco))
    return pd.concat(linhas, ignore_index=True)


def test_csv_traz_o_id_original_do_sujeito(tmp_path):
    """`v10p`, e não `0`.

    O índice interno é o que o GroupKFold usa, e é local à montagem: rodar com
    --max-sujeitos muda o índice do mesmo sujeito. Um CSV com índice não permite
    juntar duas exportações nem rastrear um sujeito até o banco."""
    destino = tmp_path / "features.csv"
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    exportar_features.escrever_csv(
        destino, X,
        y=np.array(["ADHD", "Control"]),
        ids_originais=np.array(["v10p", "v20c"]),
        indices_epoca=np.array([0, 0]),
        nomes=["Cz_tbr", "Pz_tbr"],
    )

    linhas = list(csv.DictReader(destino.open(encoding="utf-8")))
    assert linhas[0]["sujeito_id"] == "v10p"
    assert linhas[1]["sujeito_id"] == "v20c"


def test_csv_tem_o_cabecalho_do_contrato(tmp_path):
    """As três colunas de identificação vêm primeiro, e nesta ordem."""
    destino = tmp_path / "features.csv"

    exportar_features.escrever_csv(
        destino, np.array([[1.0]]),
        y=np.array(["ADHD"]),
        ids_originais=np.array(["v10p"]),
        indices_epoca=np.array([0]),
        nomes=["Cz_tbr"],
    )

    cabecalho = destino.open(encoding="utf-8").readline().strip().split(",")
    assert cabecalho[:3] == ["sujeito_id", "rotulo", "epoca_idx"]
    assert cabecalho[3:] == ["Cz_tbr"]


def test_rotulo_sai_como_texto_e_nao_como_codigo(tmp_path):
    """`ADHD`, não `1`.

    Um CSV com 0 e 1 exige um dicionário externo para ser lido, e esse
    dicionário é justamente o que se perde entre dois projetos."""
    destino = tmp_path / "features.csv"

    exportar_features.escrever_csv(
        destino, np.array([[1.0], [2.0]]),
        y=np.array(["ADHD", "Control"]),
        ids_originais=np.array(["a", "b"]),
        indices_epoca=np.array([0, 0]),
        nomes=["Cz_tbr"],
    )

    linhas = list(csv.DictReader(destino.open(encoding="utf-8")))
    assert {l["rotulo"] for l in linhas} == {"ADHD", "Control"}


def test_recusa_comprimentos_incompativeis(tmp_path):
    """X com 2 linhas e 3 ids é defeito de montagem, não caso de borda."""
    with pytest.raises(ValueError, match="comprimento"):
        exportar_features.escrever_csv(
            tmp_path / "f.csv", np.array([[1.0], [2.0]]),
            y=np.array(["ADHD", "Control"]),
            ids_originais=np.array(["a", "b", "c"]),
            indices_epoca=np.array([0, 0]),
            nomes=["Cz_tbr"],
        )


def test_cli_grava_csv_e_receita_juntos(monkeypatch, tmp_path, df_sintetico):
    """O CLI existe para que ninguém gere o CSV sem a receita ao lado.

    Um comando que produz o dado sem a procedência contradiz o propósito do
    receita.py — este teste tranca que `main()` sempre grava os dois arquivos,
    não só o CSV."""
    monkeypatch.setattr(csv_data, "load_csv", lambda caminho: df_sintetico)

    destino = tmp_path / "features.csv"
    argv = ["exportar_features.py", "--epoca", "4.0", "--passo", "4.0",
            "--saida", str(destino)]
    monkeypatch.setattr(sys, "argv", argv)

    exportar_features.main()

    receita_path = destino.with_suffix(".receita.json")
    assert destino.is_file()
    assert receita_path.is_file()

    linhas = list(csv.DictReader(destino.open(encoding="utf-8")))
    assert len(linhas) > 0

    receita = json.loads(receita_path.read_text(encoding="utf-8"))
    assert "licença" in receita["notas"]
