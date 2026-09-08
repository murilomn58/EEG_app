# -*- coding: utf-8 -*-
"""A figura de dois painéis lê os CSVs e grava PDF; sem CSV, diz o que rodar."""
import csv
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import csv_data
import figura_tbr


def _escrever(caminho, linhas):
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)


def _csvs(tmp_path):
    grupos = [{
        "canal": c, "n_adhd": 2, "n_control": 2,
        "adhd_mediana": 3.0 + i * 0.05, "adhd_q1": 2.5, "adhd_q3": 3.5,
        "control_mediana": 2.5, "control_q1": 2.0, "control_q3": 3.0,
        "U": 3.0, "p": 0.4, "delta_cliff": 0.5, "nan_descartados": 0,
    } for i, c in enumerate(csv_data.CANAIS_19)]
    resultados = []
    for sid in ("s1", "s2", "s3"):
        for cond, v in ((figura_tbr.FECHADO, 4.0), (figura_tbr.ABERTO, 3.0)):
            for c in csv_data.CANAIS_19:
                resultados.append({"sujeito": sid, "condicao": cond, "canal": c,
                                   "tbr_mediana": v, "tbr_q1": v - 0.5, "tbr_q3": v + 0.5,
                                   "n_epocas": 10, "n_blocos": 5, "duracao_blocos_s": "20;20"})
    estat = [{"canal": c, "n_pares": 3, "dif_mediana": 1.0, "dif_q1": 0.8, "dif_q3": 1.2,
              "k_fechado_maior": 3, "p_wilcoxon": ""} for c in csv_data.CANAIS_19]
    g, r, e = tmp_path / "g.csv", tmp_path / "r.csv", tmp_path / "e.csv"
    _escrever(g, grupos)
    _escrever(r, resultados)
    _escrever(e, estat)
    return g, r, e


def test_desenhar_grava_pdf(tmp_path):
    g, r, e = _csvs(tmp_path)
    grupos = figura_tbr.ler_grupos(g)
    resumo, estat = figura_tbr.ler_eventos(r, e)
    assert set(grupos) == set(csv_data.CANAIS_19)
    assert resumo["Cz"][figura_tbr.FECHADO]["mediana"] == 4.0
    destino = figura_tbr.desenhar(grupos, resumo, estat, None, None, tmp_path / "f.pdf")
    assert destino.is_file() and destino.stat().st_size > 1000


def test_sem_csv_diz_qual_experimento_rodar(tmp_path):
    r = subprocess.run(
        [sys.executable, str(RAIZ / "scripts" / "figura_tbr.py"),
         "--csv-grupos", str(tmp_path / "nao_existe.csv")],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "experimento_tbr_grupos.py" in (r.stderr + r.stdout)
