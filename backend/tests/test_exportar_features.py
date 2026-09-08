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
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import exportar_features


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
