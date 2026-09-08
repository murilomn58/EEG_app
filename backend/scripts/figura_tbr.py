# -*- coding: utf-8 -*-
"""A figura de dois painéis da TBR θ/β, lida dos CSVs dos dois experimentos.

POR QUE DOIS PAINÉIS, E POR QUE ELES NÃO SÃO A MESMA COISA

A orientadora pediu "o mesmo evento num sujeito sem e num com a
neurodivergência". Nenhum banco disponível permite isso: o adhdata tem o
rótulo (TDAH/controle) e nenhum evento; o HBN tem os eventos (olhos abertos/
fechados) e nenhum diagnóstico. A figura mostra as duas metades e escreve a
diferença em vez de fingir que são uma só:

  A  adhdata, ENTRE sujeitos: 61 TDAH × 60 controle, registro inteiro em
     épocas de 2 s, base nativa (linked-ears). Tarefa de atenção visual,
     grupo TDAH medicado.
  B  HBN RestingState, DENTRO do sujeito: olhos fechados × abertos, épocas
     ancoradas nos eventos de instrução, base CAR (sob nativa o Cz é zero).

A figura LÊ os CSVs e não roda nada, pela mesma razão de figura_vazamento:
refazer um rótulo não pode custar dez minutos de CPU nem trocar um número.

Uso:
    python scripts/figura_tbr.py
    python scripts/figura_tbr.py --saida ../relatorios/tbr_dois_paineis.pdf
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import csv_data
import estatistica

ABERTO = "instructed_toOpenEyes"
FECHADO = "instructed_toCloseEyes"
LIMIAR_CLASSICO = 4.0

# Duas cores da paleta de Okabe e Ito (distinguíveis nos três tipos de
# daltonismo), mais marcador diferente por série: a figura tem de continuar
# legível impressa em preto e branco.
CORES_A = {"ADHD": "#d55e00", "Control": "#0072b2"}
CORES_B = {FECHADO: "#d55e00", ABERTO: "#0072b2"}
MARCADORES = {"ADHD": "o", "Control": "s", FECHADO: "o", ABERTO: "s"}
ROTULOS = {"ADHD": "TDAH", "Control": "controle",
           FECHADO: "olhos fechados", ABERTO: "olhos abertos"}

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 8,
    "axes.linewidth": 0.5, "figure.dpi": 150, "axes.axisbelow": True,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
}


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def ler_grupos(caminho_estatistica):
    """{canal: linha} da estatística por grupo do adhdata."""
    saida = {}
    with open(caminho_estatistica, newline="", encoding="utf-8") as f:
        for l in csv.DictReader(f):
            saida[l["canal"]] = {k: (_f(v) if k not in ("canal",) else v) for k, v in l.items()}
    return saida


def ler_eventos(caminho_resultados, caminho_estatistica):
    """{canal: {condicao: mediana_iqr entre sujeitos}} e {canal: estatística pareada}."""
    por = defaultdict(lambda: defaultdict(list))
    with open(caminho_resultados, newline="", encoding="utf-8") as f:
        for l in csv.DictReader(f):
            por[l["canal"]][l["condicao"]].append(_f(l["tbr_mediana"]))
    resumo = {c: {cond: estatistica.mediana_iqr(v) for cond, v in conds.items()}
              for c, conds in por.items()}
    estat = {}
    with open(caminho_estatistica, newline="", encoding="utf-8") as f:
        for l in csv.DictReader(f):
            estat[l["canal"]] = {k: (_f(v) if k != "canal" else v) for k, v in l.items()}
    return resumo, estat


def _rodape(receita_a, receita_b):
    partes = []
    for rotulo, rec in (("A", receita_a), ("B", receita_b)):
        if not rec:
            partes.append(f"{rotulo}: receita ausente")
            continue
        et = rec.get("etapas", {})
        ep = et.get("epocas", {}) or {}
        partes.append(
            f"{rotulo}: {rec.get('banco', '?')}, {len(rec.get('sujeitos', []))} sujeitos, "
            f"épocas de {ep.get('duracao_s', '?')} s"
        )
    return " · ".join(partes)


def _painel(ax, canais, series, cores, deslocamento=0.16):
    """Mediana entre sujeitos como marcador, quartis como haste vertical, uma
    série ao lado da outra em cada canal. Barras não: a TBR é razão, e uma
    barra a partir de zero afirmaria uma área que não significa nada."""
    x = np.arange(len(canais))
    ordem = list(series.keys())
    for j, nome in enumerate(ordem):
        dx = (j - (len(ordem) - 1) / 2) * deslocamento
        med = np.array([series[nome][c]["mediana"] if series[nome][c]["mediana"] is not None
                        else np.nan for c in canais])
        q1 = np.array([series[nome][c]["q1"] if series[nome][c]["q1"] is not None
                       else np.nan for c in canais])
        q3 = np.array([series[nome][c]["q3"] if series[nome][c]["q3"] is not None
                       else np.nan for c in canais])
        ax.vlines(x + dx, q1, q3, color=cores[nome], lw=1.0, alpha=0.85, zorder=2)
        ax.plot(x + dx, med, MARCADORES[nome], color=cores[nome], ms=3.6,
                mec="white", mew=0.4, zorder=3, label=ROTULOS[nome])
    ax.set_xticks(x)
    ax.set_xticklabels(canais, rotation=0)
    ax.set_xlim(-0.6, len(canais) - 0.4)
    # folga no topo para a caixa de anotação e a legenda não caírem sobre
    # dado nem sobre o título: o maior quartil define o teto, mais 30 %
    topo = max(np.nanmax([series[n][c]["q3"] if series[n][c]["q3"] is not None else np.nan
                          for n in ordem for c in canais]), LIMIAR_CLASSICO)
    ax.set_ylim(0, topo * 1.32)
    ax.grid(axis="y", alpha=0.18, lw=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("TBR θ/β (mediana e quartis entre sujeitos)")
    return x


def desenhar(grupos, eventos_resumo, eventos_estat, receita_a, receita_b, destino):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC)

    canais = list(csv_data.CANAIS_19)
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(6.3, 7.0))
    fig.subplots_adjust(hspace=0.55, top=0.93, bottom=0.12, left=0.11, right=0.98)

    # ---- A: adhdata, entre sujeitos ------------------------------------
    series_a = {
        "ADHD": {c: {"mediana": grupos[c]["adhd_mediana"], "q1": grupos[c]["adhd_q1"],
                     "q3": grupos[c]["adhd_q3"]} for c in canais},
        "Control": {c: {"mediana": grupos[c]["control_mediana"], "q1": grupos[c]["control_q1"],
                        "q3": grupos[c]["control_q3"]} for c in canais},
    }
    x = _painel(ax_a, canais, series_a, CORES_A)
    n_a = int(grupos["Cz"]["n_adhd"]); n_c = int(grupos["Cz"]["n_control"])
    ax_a.set_title(f"A. adhdata, entre sujeitos: TDAH (n = {n_a}) × controle (n = {n_c}), "
                   "registro inteiro, base nativa (A1/A2)", loc="left", fontsize=8.5)
    ax_a.axhline(LIMIAR_CLASSICO, color="#777", ls="--", lw=0.7)
    ax_a.text(len(canais) - 0.5, LIMIAR_CLASSICO, "limiar clássico 4,0 (repouso; não se aplica a tarefa)",
              fontsize=6, color="#777", ha="right", va="bottom")
    i_cz = canais.index("Cz")
    ax_a.axvspan(i_cz - 0.5, i_cz + 0.5, color="#c8a86a", alpha=0.18, lw=0, zorder=0)
    cz = grupos["Cz"]
    # caixa DENTRO do painel, no canto superior direito, que a folga de 32 %
    # do teto deixa livre; ancorada ao canal por uma linha fina
    ax_a.annotate(
        f"Cz: {cz['adhd_mediana']:.2f} × {cz['control_mediana']:.2f}\n"
        f"p = {cz['p']:.2f} (Mann-Whitney) · δ Cliff = {cz['delta_cliff']:+.3f}",
        xy=(i_cz, max(cz["adhd_q3"], cz["control_q3"])),
        xytext=(0.985, 0.97), textcoords="axes fraction", ha="right", va="top",
        fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbb", lw=0.5),
        arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.5, shrinkB=3))
    ax_a.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.2)

    # ---- B: HBN, dentro do sujeito -------------------------------------
    series_b = {
        FECHADO: {c: eventos_resumo.get(c, {}).get(FECHADO, {"mediana": None, "q1": None, "q3": None})
                  for c in canais},
        ABERTO: {c: eventos_resumo.get(c, {}).get(ABERTO, {"mediana": None, "q1": None, "q3": None})
                 for c in canais},
    }
    _painel(ax_b, canais, series_b, CORES_B)
    n_b = int(eventos_estat["Cz"]["n_pares"]) if "Cz" in eventos_estat else 0
    ax_b.set_title(f"B. HBN RestingState, dentro do sujeito (n = {n_b}): olhos fechados × abertos, "
                   "épocas ancoradas no evento, base CAR", loc="left", fontsize=8.5)
    ax_b.axvspan(i_cz - 0.5, i_cz + 0.5, color="#c8a86a", alpha=0.18, lw=0, zorder=0)
    if "Cz" in eventos_estat:
        e = eventos_estat["Cz"]
        p_txt = f"Wilcoxon p = {e['p_wilcoxon']:.2f}" if not np.isnan(e.get("p_wilcoxon", np.nan)) else "sem p (n pequeno)"
        topo = max(v for v in (series_b[FECHADO]["Cz"]["q3"], series_b[ABERTO]["Cz"]["q3"]) if v is not None)
        ax_b.annotate(
            f"Cz: fechado > aberto em {int(e['k_fechado_maior'])}/{int(e['n_pares'])} sujeitos\n"
            f"diferença mediana {e['dif_mediana']:+.2f} · {p_txt}",
            xy=(i_cz, topo), xytext=(0.985, 0.97), textcoords="axes fraction",
            ha="right", va="top", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbb", lw=0.5),
            arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.5, shrinkB=3))
    ax_b.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.2)
    ax_b.set_xlabel("canal (montagem 10-20)")

    fig.suptitle("Razão θ/β por canal: o contraste diagnóstico (A) e o contraste de estado (B) "
                 "não vêm do mesmo banco", fontsize=9, y=0.985)
    fig.text(0.5, 0.012, _rodape(receita_a, receita_b), ha="center", va="bottom",
             fontsize=6.5, color="#555")

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return destino


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = config.caminho_relatorios()
    p.add_argument("--csv-grupos", default=str(base / "tbr_grupos_estatistica.csv"))
    p.add_argument("--receita-grupos", default=str(base / "tbr_grupos_receita.json"))
    p.add_argument("--csv-eventos", default=str(base / "tbr_eventos_resultados.csv"))
    p.add_argument("--csv-eventos-estat", default=str(base / "tbr_eventos_estatistica.csv"))
    p.add_argument("--receita-eventos", default=str(base / "tbr_eventos_receita.json"))
    p.add_argument("--saida", default=str(base / "tbr_dois_paineis.pdf"))
    args = p.parse_args()

    for caminho, script in ((args.csv_grupos, "experimento_tbr_grupos.py"),
                            (args.csv_eventos, "experimento_tbr_eventos.py"),
                            (args.csv_eventos_estat, "experimento_tbr_eventos.py")):
        if not Path(caminho).is_file():
            raise SystemExit(f"não achei {caminho}. Rode antes: python scripts/{script}")

    grupos = ler_grupos(args.csv_grupos)
    resumo, estat = ler_eventos(args.csv_eventos, args.csv_eventos_estat)
    rec_a = json.loads(Path(args.receita_grupos).read_text(encoding="utf-8")) \
        if Path(args.receita_grupos).is_file() else None
    rec_b = json.loads(Path(args.receita_eventos).read_text(encoding="utf-8")) \
        if Path(args.receita_eventos).is_file() else None
    saida = desenhar(grupos, resumo, estat, rec_a, rec_b, args.saida)
    print(f"[figura_tbr] {saida}")


if __name__ == "__main__":
    main()
