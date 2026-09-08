# -*- coding: utf-8 -*-
"""TBR θ/β no RestingState do HBN: olhos fechados × abertos, dentro do sujeito.

O QUE ESTE SCRIPT MEDE, E O QUE NÃO MEDE

O HBN tem o que o adhdata não tem: eventos. O protocolo de repouso alterna
`instructed_toOpenEyes` (20 s) e `instructed_toCloseEyes` (40 s), cinco
ciclos, e isso permite comparar a MESMA criança em dois estados, ancorando
as épocas no evento. É o "mesmo evento, dois estados" que a orientadora
pediu, com uma diferença que não pode ficar implícita: o contraste é entre
CONDIÇÕES de um sujeito, não entre sujeitos com e sem diagnóstico. O HBN não
traz diagnóstico (só escores contínuos, e nos 10 sujeitos baixados apenas 2
têm `attention` acima de zero), então nenhum agrupamento clínico é feito aqui.

BASE CAR, declarada. Sob a referência nativa o Cz do HBN é a própria
referência, vem identicamente zero, e a TBR dele é NaN (0/0). O CAR sobre os
19 canais de análise devolve sinal ao Cz e é a mesma escolha que o app faz
em /raw-data (sobre os 19, não sobre os 129: a média de 19 eletrodos 10-20
não é a média da malha inteira).

OS BLOCOS SÃO MEDIDOS, NÃO FIXADOS. `blocos_entre_eventos` fecha cada bloco
no onset do próximo evento de instrução e descarta o último `toOpenEyes`,
que não tem par (4,5 s depois vem um `boundary` ou `break cnt`, e a tarefa
seguinte). Cortes em `boundary` E `break cnt`: o sub-NDARAC904DMU não tem
`boundary`, e é o que tem eventos de seqLearning dentro do RestingState.

Uso:
    python scripts/experimento_tbr_eventos.py
    python scripts/experimento_tbr_eventos.py --max-sujeitos 2   (ensaio)
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
import caracteristicas
import epocas as mod_epocas
import estatistica
import eventos as mod_eventos
import preproc_basico
import receita
from canais import resolver as resolver_canais

ALVOS = ("instructed_toOpenEyes", "instructed_toCloseEyes")
CORTES = ("boundary", "break cnt")
ABERTO, FECHADO = ALVOS

RESSALVAS = (
    "HBN ds005505 RestingState, 10 sujeitos, base CAR sobre os 19 canais de "
    "análise. Contraste DENTRO do sujeito (olhos fechados x abertos), sem "
    "diagnóstico: o participants.tsv só traz escores contínuos. Blocos medidos "
    "pelo próximo evento de instrução; o último toOpenEyes, sem par, não entra."
)


def preparar_hbn(caminho_set, meta):
    """O Raw reduzido aos 19 canais de análise, nomeados em 10-20, na ordem
    de `meta["canais_analise"]`. É o que `_raw_data_bids` faz no app."""
    raw = preproc_basico.carregar_raw(str(caminho_set))
    resolvido, faltantes = resolver_canais(
        raw.ch_names, meta["canais_analise"], meta.get("mapa_canais")
    )
    if faltantes:
        raise ValueError(f"canais sem correspondente: {', '.join(faltantes)}")
    nomes_arquivo = [resolvido[c] for c in meta["canais_analise"]]
    reduzido = raw.copy().pick(nomes_arquivo).reorder_channels(nomes_arquivo)
    return reduzido


def medir_por_condicao(dado, fs, eventos, duracao_s, passo_s, nomes,
                       alvos=ALVOS, cortes_valores=CORTES):
    """{condicao: {"tbr": (n_epocas, n_canais) ou None, "n_epocas", "n_blocos",
    "duracoes_s"}} e as decisões da ancoragem. Função pura: recebe o sinal já
    filtrado em µV e a lista de eventos; é o que o teste exercita."""
    n = dado.shape[1]
    cortes = mod_epocas.cortes_de_eventos(eventos, fs, valores=cortes_valores)
    blocos, dec_blocos = mod_epocas.blocos_entre_eventos(eventos, fs, n, alvos, cortes)

    saida = {}
    for condicao in alvos:
        segs = [(a, b) for a, b, v in blocos if v == condicao]
        janelas, _ = mod_epocas.epocar_janela_fixa(dado, fs, duracao_s, passo_s, segs)
        item = {
            "tbr": None, "n_epocas": int(len(janelas)),
            "n_blocos": dec_blocos["n_blocos_por_valor"].get(condicao, 0),
            "duracoes_s": dec_blocos["duracao_s_por_valor"].get(condicao, []),
        }
        if len(janelas):
            tbr, _, _ = caracteristicas.razao_theta_beta(
                janelas, fs, nomes_canais=nomes, com_decisoes=True
            )
            item["tbr"] = tbr
        saida[condicao] = item
    return saida, dec_blocos


def medir_sujeito(caminho_set, meta, duracao_s=2.0, passo_s=2.0, base="car"):
    """Linhas (uma por condição e canal) e decisões de um sujeito do HBN."""
    reduzido = preparar_hbn(caminho_set, meta)
    fs = float(reduzido.info["sfreq"])
    filtrado, dec_preproc = preproc_basico.preprocessar(
        reduzido, l_freq=0.5, h_freq=None, base=base
    )
    dado = filtrado.get_data() * 1e6
    ev = mod_eventos.detectar_eventos(
        str(caminho_set), raiz=config.caminho_release(meta["accession"])
    )["eventos"]
    nomes = list(meta["canais_analise"])
    resultado, dec_blocos = medir_por_condicao(dado, fs, ev, duracao_s, passo_s, nomes)

    sid = Path(caminho_set).parent.parent.name
    linhas = []
    for condicao, item in resultado.items():
        for i, canal in enumerate(nomes):
            r = estatistica.mediana_iqr(item["tbr"][:, i]) if item["tbr"] is not None \
                else {"mediana": None, "q1": None, "q3": None, "n": 0}
            nan = float("nan")
            linhas.append({
                "sujeito": sid, "condicao": condicao, "canal": canal,
                "tbr_mediana": r["mediana"] if r["mediana"] is not None else nan,
                "tbr_q1": r["q1"] if r["q1"] is not None else nan,
                "tbr_q3": r["q3"] if r["q3"] is not None else nan,
                "n_epocas": item["n_epocas"], "n_blocos": item["n_blocos"],
                "duracao_blocos_s": ";".join(f"{d:g}" for d in item["duracoes_s"]),
            })
    return linhas, {"preproc": dec_preproc, "blocos": dec_blocos, "fs": fs}


def comparar_condicoes(linhas, nomes):
    """Por canal: diferença pareada fechado − aberto por sujeito, mediana e
    quartis entre sujeitos, em quantos o fechado é maior, e Wilcoxon pareado
    (com a ressalva de n pequeno escrita na receita, não escondida aqui)."""
    from scipy import stats

    por = {}
    for l in linhas:
        por.setdefault((l["sujeito"], l["canal"]), {})[l["condicao"]] = l["tbr_mediana"]

    saida = []
    for canal in nomes:
        difs = []
        for (sid, c), v in por.items():
            if c != canal:
                continue
            f, a = v.get(FECHADO), v.get(ABERTO)
            if f is None or a is None or np.isnan(f) or np.isnan(a):
                continue
            difs.append(f - a)
        r = estatistica.mediana_iqr(difs)
        k = int(sum(1 for d in difs if d > 0))
        p = None
        if len(difs) >= 6 and any(d != 0 for d in difs):
            try:
                p = float(stats.wilcoxon(difs, alternative="two-sided").pvalue)
            except ValueError:
                p = None
        saida.append({
            "canal": canal, "n_pares": r["n"],
            "dif_mediana": r["mediana"], "dif_q1": r["q1"], "dif_q3": r["q3"],
            "k_fechado_maior": k, "p_wilcoxon": p,
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
    p.add_argument("--dataset", default="hbn_ds005505")
    p.add_argument("--epoca", type=float, default=2.0)
    p.add_argument("--passo", type=float, default=2.0)
    p.add_argument("--base", default="car", choices=preproc_basico.BASES)
    p.add_argument("--max-sujeitos", type=int, default=None)
    p.add_argument("--saida", default=str(config.caminho_relatorios()))
    args = p.parse_args()

    meta = config.DATASETS[args.dataset]
    raiz = config.caminho_release(meta["accession"])
    sets = sorted(raiz.glob("sub-*/eeg/*task-RestingState*.set"))
    if args.max_sujeitos:
        sets = sets[:args.max_sujeitos]
    if not sets:
        raise SystemExit(f"nenhuma gravação em {raiz}")

    t0 = time.time()
    linhas, falhas, decisoes = [], [], {}
    for caminho in sets:
        sid = caminho.parent.parent.name
        try:
            l, d = medir_sujeito(caminho, meta, args.epoca, args.passo, args.base)
            linhas.extend(l)
            decisoes[sid] = {"blocos": d["blocos"], "fs": d["fs"]}
            print(f"[tbr_eventos] {sid}: blocos {d['blocos']['n_blocos_por_valor']} "
                  f"({time.time() - t0:.0f} s)")
        except Exception as e:            # noqa: BLE001 — sujeito que falha aparece
            falhas.append((sid, str(e)))
            print(f"[tbr_eventos] {sid}: FALHOU: {e}")

    nomes = list(meta["canais_analise"])
    comparacao = comparar_condicoes(linhas, nomes)
    saida = Path(args.saida)
    _escrever_csv(saida / "tbr_eventos_resultados.csv", linhas)
    _escrever_csv(saida / "tbr_eventos_estatistica.csv", comparacao)
    rec = receita.montar(
        banco=args.dataset,
        sujeitos=sorted(decisoes),
        etapas={"preproc": {"l_freq": 0.5, "h_freq": None, "base": args.base},
                "epocas": {"duracao_s": args.epoca, "passo_s": args.passo,
                           "alvos": list(ALVOS), "cortes": list(CORTES)},
                "por_sujeito": decisoes,
                "sujeitos_que_falharam": [{"id": s, "motivo": m} for s, m in falhas]},
        notas=RESSALVAS,
    )
    receita.salvar(rec, saida / "tbr_eventos_receita.json")

    cz = next(c for c in comparacao if c["canal"] == "Cz")
    print(f"[tbr_eventos] {len(decisoes)} sujeitos, {len(falhas)} falharam, "
          f"{time.time() - t0:.0f} s")
    print(f"[tbr_eventos] Cz: fechado − aberto mediana {cz['dif_mediana']} | "
          f"fechado > aberto em {cz['k_fechado_maior']}/{cz['n_pares']} | "
          f"Wilcoxon p = {cz['p_wilcoxon']}")
    print(f"[tbr_eventos] arquivos em {saida}")


if __name__ == "__main__":
    main()
