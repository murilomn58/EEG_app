# -*- coding: utf-8 -*-
"""Desenha as quatro barras do experimento de vazamento, a partir do CSV.

POR QUE A FIGURA LÊ O CSV EM VEZ DE RODAR O EXPERIMENTO

Separar as duas coisas é o que permite refazer a figura sem refazer o
experimento — mudar uma cor, um rótulo ou o formato do papel não deveria
custar dez minutos de CPU nem, pior, produzir números ligeiramente
diferentes por causa de outra semente.

É a mesma razão pela qual o CSV existe: ele é o dado da figura, e alguém
pode conferir a média que cada barra mostra somando as linhas dele à mão.

O QUE A LEGENDA TEM DE DIZER, E POR QUÊ

A figura afirma uma DIFERENÇA entre condições, não uma acurácia. Um
classificador melhor levantaria as quatro barras e a diferença continuaria
lá. Sem essa frase, o número 0,71 da barra D vira, na leitura de outra
pessoa, "o app classifica TDAH com 71%" — que é uma afirmação que este
experimento não faz.

Uso:
    python scripts/figura_vazamento.py
    python scripts/figura_vazamento.py --csv outro.csv --receita outra.json
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

NOMES = {
    "A": "split por segmento\nnormalização global",
    "B": "split por segmento\nnormalização por dobra",
    "C": "split por sujeito\nnormalização global",
    "D": "split por sujeito\nnormalização por dobra",
}

# Âmbar para as condições que vazam, verde para a honesta. A cor carrega a
# leitura: as três primeiras barras são o que NÃO se deve reportar.
CORES = {"A": "#e8a33d", "B": "#e8a33d", "C": "#e8a33d", "D": "#3dbb6e"}


def ler_resultados(caminho):
    """Média e desvio por condição, direto do CSV do experimento."""
    por_condicao = defaultdict(list)
    with open(caminho, newline="", encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            valor = float(linha["acuracia"])
            if not np.isnan(valor):
                por_condicao[linha["condicao"]].append(valor)
    return {
        c: {"media": float(np.mean(v)), "desvio": float(np.std(v)), "n": len(v)}
        for c, v in por_condicao.items()
    }


def _rodape(receita):
    """A procedência, em texto, sob a figura.

    Uma figura de tese sem n, sem banco e sem o método de janelamento obriga
    quem lê a procurar no texto — e a figura acaba viajando sozinha, colada
    num slide, sem nada disso."""
    if not receita:
        return "procedência não disponível: receita ausente"

    e = receita.get("etapas", {})
    ep = e.get("epocas", {}) or {}
    sp = e.get("split", {}) or {}
    partes = [
        f"banco: {receita.get('banco', '?')}",
        f"{e.get('n_sujeitos_usados', '?')} sujeitos",
        f"{sp.get('n_epocas', '?')} épocas de {ep.get('duracao_s', '?')} s",
        f"{sp.get('n_dobras', '?')} dobras",
        "regressão logística sobre potência de banda (19 canais x 5 bandas)",
    ]
    linhas = [" · ".join(partes)]

    # O desequilíbrio de épocas por sujeito é declarado, sempre. Gravações de
    # duração diferente geram números diferentes de épocas, e o sujeito com
    # mais épocas pesa mais — quem lê precisa do número para saber se está
    # vendo patologia ou o sujeito mais comprido.
    mn, mx = sp.get("min_epocas_por_sujeito"), sp.get("max_epocas_por_sujeito")
    if mn is not None and mx is not None:
        linhas.append(f"épocas por sujeito: {mn} a {mx} (não equalizado)")

    if receita.get("notas"):
        linhas.append(receita["notas"])
    linhas.append(
        "a figura afirma a DIFERENÇA entre condições, não a acurácia absoluta"
    )
    return "\n".join(linhas)


def desenhar(resultados, receita, destino):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordem = [c for c in ("A", "B", "C", "D") if c in resultados]
    medias = [resultados[c]["media"] for c in ordem]
    desvios = [resultados[c]["desvio"] for c in ordem]

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    barras = ax.bar(range(len(ordem)), medias, yerr=desvios, capsize=5,
                    color=[CORES[c] for c in ordem], edgecolor="#333", linewidth=0.8)

    for i, (b, c) in enumerate(zip(barras, ordem)):
        ax.text(b.get_x() + b.get_width() / 2, medias[i] + desvios[i] + 0.02,
                f"{medias[i]:.3f}", ha="center", fontsize=11, fontweight="bold")

    # O acaso é a régua: sem esta linha, 0,71 parece bom em vez de parecer o
    # que é — pouco acima de chutar.
    ax.axhline(0.5, color="#888", linestyle="--", linewidth=1)
    ax.text(len(ordem) - 0.45, 0.515, "acaso", fontsize=9, color="#888")

    ax.set_xticks(range(len(ordem)))
    ax.set_xticklabels([f"{c}\n{NOMES[c]}" for c in ordem], fontsize=9)
    ax.set_ylabel("acurácia média entre dobras")
    ax.set_ylim(0, 1.08)
    ax.set_title("Quanto da acurácia é vazamento, e não patologia", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)

    if "A" in resultados and "D" in resultados:
        queda = resultados["A"]["media"] - resultados["D"]["media"]
        ax.annotate(f"queda de {queda:.3f}", xy=(0, medias[0]), xytext=(1.5, 1.0),
                    ha="center", fontsize=11,
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4))

    fig.text(0.5, 0.005, _rodape(receita), ha="center", va="bottom",
             fontsize=7.5, color="#555")
    fig.subplots_adjust(bottom=0.32)

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    # PDF vetorial: a dissertação exige figura vetorial, nunca captura de tela
    fig.savefig(destino, format="pdf")
    plt.close(fig)
    return destino


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = config.caminho_relatorios()
    p.add_argument("--csv", default=str(base / "vazamento_resultados.csv"))
    p.add_argument("--receita", default=str(base / "vazamento_receita.json"))
    p.add_argument("--saida", default=str(base / "vazamento.pdf"))
    args = p.parse_args()

    if not Path(args.csv).is_file():
        raise SystemExit(
            f"não achei {args.csv}. Rode antes: "
            f"python scripts/experimento_vazamento.py"
        )

    resultados = ler_resultados(args.csv)
    receita = None
    if Path(args.receita).is_file():
        receita = json.loads(Path(args.receita).read_text(encoding="utf-8"))

    saida = desenhar(resultados, receita, args.saida)
    for c in ("A", "B", "C", "D"):
        if c in resultados:
            r = resultados[c]
            print(f"{c}  {r['media']:.3f} ± {r['desvio']:.3f}  ({r['n']} dobras)")
    print(f"[figura] {saida}")


if __name__ == "__main__":
    main()
