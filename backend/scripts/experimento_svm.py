# -*- coding: utf-8 -*-
"""SVM sobre a razão theta/beta do adhdata, sob validação aninhada por sujeito.

A MISSÃO

Folha manuscrita da reunião de 08/09/2026: "Executar pipeline de classificador e
extração de feature para θ/β e com SKLEARN. SVM".

POR QUE QUATRO CONDIÇÕES E NÃO UMA

A condição A é a pedida. Sozinha, ela produz um número que ninguém consegue
interpretar: 0,62 é bom? É melhor que o que já havia? Bate o acaso?

    A   TBR (19)              SVM-RBF     o que foi pedido
    B   potência de banda(95) SVM-RBF     a TBR joga informação fora?
    C   TBR (19)              LogReg      o SVM ganha sobre o que existia?
    D   TBR (19)              SVM, nulo   o número bate o acaso?

O QUE ESTE EXPERIMENTO NÃO AFIRMA

Nada sobre o estado da arte. A TBR é literatura contestada — efeito declinante
com o ano de publicação, parecer negativo de sociedade médica para uso
diagnóstico, e reanálise multiverso atribuindo boa parte do efeito ao componente
aperiódico e à frequência individual de alfa. Ela entra como linha de base a
bater, não como marcador em que o projeto aposta.

EXPECTATIVA REGISTRADA ANTES DE MEDIR

A medição de 07/09/2026 encontrou a TBR do grupo TDAH MENOR que a do controle no
adhdata (direção contrária à literatura), p = 0,132 em Cz. Uma AUC entre 0,55 e
0,70 é o resultado coerente com isso. AUC acima de 0,90 deve ser tratada como
suspeita de defeito, e investigada antes de reportada.

Uso:
    python scripts/experimento_svm.py
    python scripts/experimento_svm.py --max-sujeitos 10 --permutacoes 10
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import caracteristicas
import classificador
import config
import csv_data
import epocas as mod_epocas
import exportar_features
import preproc_basico
import receita as mod_receita


def _potencia_de_banda_do_conjunto(df, duracao_s, passo_s, max_sujeitos):
    """As 95 características de potência, para a condição B.

    Repete a montagem em vez de generalizar `montar_conjunto_tbr` com um
    parâmetro de tipo de característica: a montagem é curta, e um parâmetro que
    troca o que é extraído esconderia, numa flag, a diferença entre duas
    condições do experimento."""
    sujeitos = csv_data.list_subjects(df)
    if max_sujeitos:
        por_classe = {}
        for s in sujeitos:
            por_classe.setdefault(s["classe"], []).append(s)
        metade = max(1, max_sujeitos // max(1, len(por_classe)))
        escolhidos = []
        for lista in por_classe.values():
            escolhidos.extend(lista[:metade])
        sujeitos = escolhidos[:max_sujeitos]

    bX, by, bg = [], [], []
    for idx, s in enumerate(sujeitos):
        try:
            raw = preproc_basico.raw_de_dataframe(df, s["id"])
            filtrado, _ = preproc_basico.preprocessar(raw, l_freq=0.5, h_freq=None)
            janelas, _ = mod_epocas.epocar_janela_fixa(
                filtrado.get_data() * 1e6, csv_data.FS, duracao_s, passo_s
            )
            if len(janelas) == 0:
                continue
            X, _ = caracteristicas.potencia_de_banda(
                janelas, csv_data.FS, nomes_canais=list(csv_data.CANAIS_19)
            )
            bX.append(X)
            by.append(np.full(len(X), 1 if s["classe"] == "ADHD" else 0))
            bg.append(np.full(len(X), idx))
        except Exception:                 # noqa: BLE001
            continue
    return np.vstack(bX), np.concatenate(by), np.concatenate(bg)


def rodar(df=None, duracao_s=4.0, passo_s=4.0, max_sujeitos=None,
          n_permutacoes=100, n_dobras_internas=5, semente=0):
    """As quatro condições, e a receita para refazê-las."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if df is None:
        df = csv_data.load_csv(config.CAMINHO_ADHDATA)

    Xt, yt_txt, gt, _, _, _, dec = exportar_features.montar_conjunto_tbr(
        df, duracao_s, passo_s, max_sujeitos
    )
    yt = (yt_txt == "ADHD").astype(int)

    resultados = []

    a = classificador.avaliar_loso(
        Xt, yt, gt, n_dobras_internas=n_dobras_internas, semente=semente
    )
    resultados.append({"condicao": "A", "features": "TBR (19)", "modelo": "SVM-RBF",
                       "auc": a["auc"], "acuracia": a["acuracia"],
                       "n_sujeitos": a["n_sujeitos"], "p_empirico": None})

    Xb, yb, gb = _potencia_de_banda_do_conjunto(df, duracao_s, passo_s, max_sujeitos)
    b = classificador.avaliar_loso(
        Xb, yb, gb, n_dobras_internas=n_dobras_internas, semente=semente
    )
    resultados.append({"condicao": "B", "features": "potência de banda (95)",
                       "modelo": "SVM-RBF", "auc": b["auc"],
                       "acuracia": b["acuracia"], "n_sujeitos": b["n_sujeitos"],
                       "p_empirico": None})

    logreg = Pipeline([("escala", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=2000,
                                                  random_state=semente))])
    c = classificador.avaliar_loso(
        Xt, yt, gt, estimador=logreg, grade={"clf__C": [0.1, 1.0, 10.0]},
        n_dobras_internas=n_dobras_internas, semente=semente,
    )
    resultados.append({"condicao": "C", "features": "TBR (19)",
                       "modelo": "LogisticRegression", "auc": c["auc"],
                       "acuracia": c["acuracia"], "n_sujeitos": c["n_sujeitos"],
                       "p_empirico": None})

    d = classificador.nulo_por_permutacao(
        Xt, yt, gt, n_permutacoes=n_permutacoes,
        n_dobras_internas=n_dobras_internas, semente=semente,
    )
    resultados.append({"condicao": "D", "features": "TBR (19)",
                       "modelo": "SVM-RBF, rótulos permutados",
                       "auc": d["media_nula"], "acuracia": None,
                       "n_sujeitos": a["n_sujeitos"],
                       "p_empirico": d["p_empirico"]})

    dec["avaliacao"] = a["decisoes"]
    dec["nulo"] = d["decisoes"]
    receita = mod_receita.montar(
        banco="adhdata",
        sujeitos=[f"{dec['n_sujeitos_usados']} sujeitos"],
        etapas=dec,
        notas=exportar_features.RESSALVA_LICENCA,
    )
    return resultados, receita


def salvar_csv(resultados, caminho):
    """Uma linha por condição."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    campos = ["condicao", "features", "modelo", "auc", "acuracia",
              "n_sujeitos", "p_empirico"]
    with caminho.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(resultados)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epoca", type=float, default=4.0)
    p.add_argument("--passo", type=float, default=4.0)
    p.add_argument("--max-sujeitos", type=int, default=None)
    p.add_argument("--permutacoes", type=int, default=100)
    p.add_argument("--dobras-internas", type=int, default=5)
    p.add_argument("--saida", default="relatorios/experimento_svm.csv")
    args = p.parse_args()

    resultados, receita = rodar(
        duracao_s=args.epoca, passo_s=args.passo,
        max_sujeitos=args.max_sujeitos, n_permutacoes=args.permutacoes,
        n_dobras_internas=args.dobras_internas,
    )
    salvar_csv(resultados, args.saida)
    mod_receita.salvar(receita, Path(args.saida).with_suffix(".receita.json"))

    print(f"{'cond':<5}{'features':<24}{'modelo':<30}{'AUC':>7}{'p':>8}")
    for r in resultados:
        pe = "" if r["p_empirico"] is None else f"{r['p_empirico']:.3f}"
        print(f"{r['condicao']:<5}{r['features']:<24}{r['modelo']:<30}"
              f"{r['auc']:>7.3f}{pe:>8}")
    print(f"\n{exportar_features.RESSALVA_LICENCA}")


if __name__ == "__main__":
    main()
