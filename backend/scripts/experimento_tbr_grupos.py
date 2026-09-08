# -*- coding: utf-8 -*-
"""TBR θ/β por sujeito e canal no adhdata, e a comparação TDAH × Controle.

O QUE ESTE SCRIPT TORNA REPRODUTÍVEL

O frontend afirma que "medindo neste próprio app os 121 sujeitos, o TBR em Cz
saiu MENOR no grupo TDAH: mediana 3,35 contra 3,57, p = 0,19 (Mann-Whitney),
delta de Cliff −0,139". Nenhum script versionado produzia esse número. Este
produz UM número com origem: a mesma `razao_theta_beta` que o endpoint
/features/tbr usa, sobre épocas de janela fixa, com o pré-processamento
`basico` do app (passa-alta 0,5 Hz + notch na rede detectada, sem passa-baixa).

A UNIDADE ESTATÍSTICA É O SUJEITO, não a época. Uma criança com gravação de
338 s gera três vezes mais épocas de 2 s que uma de 112 s; comparar épocas
entre grupos pesaria os sujeitos compridos e pseudo-replicaria n = 61 em
milhares. Por isso cada sujeito vira UM valor por canal (a mediana das suas
épocas), e o teste é feito sobre 61 contra 60 valores.

O que este número NÃO é: o adhdata é registro de TAREFA de atenção visual, não
de repouso, e o grupo TDAH estava medicado com metilfenidato (ficha do
dataset, ver Referencia - Tecnicas Kaggle EEG for ADHD no vault). O limiar
clássico de 4,0 da TBR vem de repouso e não se aplica aqui. A comparação vale
como descrição deste banco, não como marcador.

Uso:
    python scripts/experimento_tbr_grupos.py
    python scripts/experimento_tbr_grupos.py --max-sujeitos 6   (ensaio)
    python scripts/experimento_tbr_grupos.py --base car --epoca 4 --passo 2
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import csv_data
import caracteristicas
import epocas as mod_epocas
import estatistica
import preproc_basico
import receita

RESSALVAS = (
    "adhdata: registro de tarefa de atenção visual, não de repouso; grupo TDAH "
    "medicado com metilfenidato (ficha do dataset). O limiar clássico de 4,0 da "
    "TBR vem de repouso e não se aplica. Unidade estatística: um valor por "
    "sujeito e canal (mediana das épocas)."
)

CLASSES = ("ADHD", "Control")


def escolher_sujeitos(sujeitos, max_sujeitos=None):
    """Amostra EQUILIBRADA por classe, e não os N primeiros.

    A lista do adhdata chega ORDENADA POR CLASSE: 61 ADHD seguidos de 60
    controle, com exatamente uma troca. Pegar os N primeiros dá uma classe só
    (medido em experimento_vazamento: `--max-sujeitos 10` produzia dez ADHD)."""
    if not max_sujeitos:
        return list(sujeitos)
    por_classe = {}
    for s in sujeitos:
        por_classe.setdefault(s["classe"], []).append(s)
    metade = max(1, max_sujeitos // max(1, len(por_classe)))
    escolhidos = []
    for lista in por_classe.values():
        escolhidos.extend(lista[:metade])
    return escolhidos[:max_sujeitos]


def medir_sujeitos(df, duracao_s=2.0, passo_s=2.0, base="nativa", max_sujeitos=None):
    """(linhas, decisoes): uma linha por sujeito e canal.

    Cada linha traz a mediana e os quartis da TBR entre as épocas daquele
    sujeito. Sujeito que falha APARECE em `sujeitos_que_falharam`, com o
    motivo, em vez de sumir do denominador."""
    sujeitos = escolher_sujeitos(csv_data.list_subjects(df), max_sujeitos)
    linhas, falhas = [], []
    dec_preproc = dec_epocas = dec_tbr = None
    nomes = list(csv_data.CANAIS_19)

    for s in sujeitos:
        sid = s["id"]
        try:
            raw = preproc_basico.raw_de_dataframe(df, sid)
            filtrado, dec_preproc = preproc_basico.preprocessar(
                raw, l_freq=0.5, h_freq=None, base=base
            )
            dados = filtrado.get_data() * 1e6
            janelas, dec_epocas = mod_epocas.epocar_janela_fixa(
                dados, csv_data.FS, duracao_s, passo_s
            )
            if len(janelas) == 0:
                falhas.append((sid, "nenhuma época coube na gravação"))
                continue
            tbr, _, dec_tbr = caracteristicas.razao_theta_beta(
                janelas, csv_data.FS, nomes_canais=nomes, com_decisoes=True
            )
            for i, canal in enumerate(nomes):
                r = estatistica.mediana_iqr(tbr[:, i])
                nan = float("nan")
                linhas.append({
                    "sujeito": sid, "classe": s["classe"], "canal": canal,
                    "tbr_mediana": r["mediana"] if r["mediana"] is not None else nan,
                    "tbr_q1": r["q1"] if r["q1"] is not None else nan,
                    "tbr_q3": r["q3"] if r["q3"] is not None else nan,
                    "n_epocas": r["n"],
                })
        except Exception as e:            # noqa: BLE001 — sujeito que falha aparece
            falhas.append((sid, str(e)))

    usados = {l["sujeito"] for l in linhas}
    decisoes = {
        "preproc": dec_preproc,
        "epocas": dec_epocas,
        "tbr": dec_tbr,
        "base": base,
        "n_sujeitos_pedidos": len(sujeitos),
        "n_sujeitos_usados": len(usados),
        "n_por_classe": {
            c: len({l["sujeito"] for l in linhas if l["classe"] == c}) for c in CLASSES
        },
        "sujeitos_que_falharam": [{"id": s, "motivo": m} for s, m in falhas],
    }
    return linhas, decisoes


def comparar_grupos(linhas):
    """Por canal: mediana e quartis de cada grupo, Mann-Whitney e delta de Cliff."""
    por_canal = {}
    for l in linhas:
        por_canal.setdefault(l["canal"], {}).setdefault(l["classe"], []).append(l["tbr_mediana"])

    saida = []
    for canal in csv_data.CANAIS_19:
        grupos = por_canal.get(canal, {})
        a = grupos.get("ADHD", [])
        b = grupos.get("Control", [])
        ma, mb = estatistica.mediana_iqr(a), estatistica.mediana_iqr(b)
        mw = estatistica.mann_whitney(a, b)
        saida.append({
            "canal": canal,
            "n_adhd": ma["n"], "n_control": mb["n"],
            "adhd_mediana": ma["mediana"], "adhd_q1": ma["q1"], "adhd_q3": ma["q3"],
            "control_mediana": mb["mediana"], "control_q1": mb["q1"], "control_q3": mb["q3"],
            "U": mw["U"], "p": mw["p"],
            "delta_cliff": estatistica.delta_de_cliff(a, b),
            "nan_descartados": mw["nan_descartados"],
        })
    return saida


def _escrever_csv(caminho, linhas):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    if not linhas:
        raise RuntimeError(f"nada a escrever em {caminho}")
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    return caminho


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epoca", type=float, default=2.0, help="duração da época, s (padrão 2)")
    p.add_argument("--passo", type=float, default=2.0, help="passo entre épocas, s (padrão 2)")
    p.add_argument("--base", default="nativa", choices=preproc_basico.BASES)
    p.add_argument("--max-sujeitos", type=int, default=None, help="ensaio: N sujeitos, equilibrados")
    p.add_argument("--saida", default=str(config.caminho_relatorios()))
    args = p.parse_args()

    t0 = time.time()
    df = csv_data.load_csv(config.CAMINHO_ADHDATA)
    linhas, dec = medir_sujeitos(df, args.epoca, args.passo, args.base, args.max_sujeitos)
    comparacao = comparar_grupos(linhas)

    saida = Path(args.saida)
    _escrever_csv(saida / "tbr_grupos_por_sujeito.csv", linhas)
    _escrever_csv(saida / "tbr_grupos_estatistica.csv", comparacao)
    rec = receita.montar(
        banco="adhdata",
        sujeitos=sorted({l["sujeito"] for l in linhas}),
        etapas={"preproc": dec["preproc"], "epocas": dec["epocas"], "tbr": dec["tbr"],
                "amostra": {k: dec[k] for k in ("base", "n_sujeitos_pedidos",
                                                  "n_sujeitos_usados", "n_por_classe",
                                                  "sujeitos_que_falharam")}},
        notas=RESSALVAS,
    )
    receita.salvar(rec, saida / "tbr_grupos_receita.json")

    cz = next(c for c in comparacao if c["canal"] == "Cz")
    print(f"[tbr_grupos] {dec['n_sujeitos_usados']} sujeitos "
          f"(ADHD {dec['n_por_classe']['ADHD']}, Control {dec['n_por_classe']['Control']}), "
          f"{len(dec['sujeitos_que_falharam'])} falharam, {time.time() - t0:.0f} s")
    print(f"[tbr_grupos] Cz: mediana ADHD {cz['adhd_mediana']:.2f} x Control "
          f"{cz['control_mediana']:.2f} | p = {cz['p']:.3f} | delta de Cliff {cz['delta_cliff']:+.3f}")
    print(f"[tbr_grupos] arquivos em {saida}")

    if args.max_sujeitos is None and dec["n_por_classe"] != {"ADHD": 61, "Control": 60}:
        print("[tbr_grupos] AVISO: contagem por classe diferente de 61/60; ver "
              "sujeitos_que_falharam na receita antes de usar a estatística")
        sys.exit(2)


if __name__ == "__main__":
    main()
