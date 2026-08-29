# -*- coding: utf-8 -*-
"""Mede quanto da acurácia de classificação de EEG é vazamento, não patologia.

A PERGUNTA

Trabalhos de classificação de TDAH por EEG reportam acurácias altas. Parte
disso é real. Parte é o classificador aprendendo a reconhecer o SUJEITO em
vez do diagnóstico — porque épocas da mesma criança são parecidas entre si
por impedância de eletrodo, formato do crânio e quanto ela se mexeu, e nada
disso tem a ver com o rótulo.

Este experimento separa as duas fontes de vazamento em vez de misturá-las:

                     | normalização global | normalização por dobra
    split por segmento |         A          |          B
    split por sujeito  |         C          |     D  (o honesto)

A é o número mais otimista; D é o que sobrevive à avaliação rigorosa. A
diferença entre eles é o resultado.

O QUE ESTE EXPERIMENTO NÃO AFIRMA

Nada sobre a acurácia ABSOLUTA. O classificador é uma regressão logística
sobre potência de banda — deliberadamente simples e transparente. Um modelo
melhor daria números melhores nas quatro células, e a comparação entre elas
continuaria valendo. É a comparação que é o resultado, e isso vai escrito na
legenda da figura.

SOBRE O BANCO

Roda no adhdata: 121 sujeitos, rótulo binário, sem necessidade de DUA, e a
coluna `ID` dá a fronteira de sujeito explícita. É o único dado onde o
experimento fecha hoje.

A licença do adhdata NÃO PÔDE SER CONFIRMADA — o README do projeto registra
que ausência de licença não é licença permissiva. A ressalva sai na figura e
no CSV, não fica só no repositório.

Uso:
    python scripts/experimento_vazamento.py                 # padrão
    python scripts/experimento_vazamento.py --dobras 5 --epoca 4 --passo 4
    python scripts/experimento_vazamento.py --max-sujeitos 20   # ensaio rápido
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import caracteristicas
import config
import csv_data
import epocas as mod_epocas
import normalizacao
import preproc_basico
import receita as mod_receita
import split as mod_split

RESSALVA_LICENCA = (
    "adhdata: a licença deste banco não pôde ser confirmada; ausência de "
    "licença não é licença permissiva"
)


def montar_conjunto(df, duracao_s, passo_s, max_sujeitos=None):
    """(X, y, grupos, nomes, decisoes) a partir do CSV inteiro.

    Um sujeito de cada vez, e as épocas de cada um recebem o índice dele em
    `grupos` — é esse vetor que o split por sujeito usa para não deixar a
    mesma criança nos dois lados.

    A fronteira é dupla e as duas importam: `epocar_janela_fixa` recebe o
    sinal de UM sujeito, então não há como uma janela cruzar de criança para
    criança; e dentro do sujeito não há descontinuidade conhecida no adhdata.
    A separação por sujeito acontece aqui, na montagem, e não é delegada."""
    sujeitos = csv_data.list_subjects(df)
    if max_sujeitos:
        # AMOSTRA EQUILIBRADA, e nao os N primeiros.
        #
        # A lista do adhdata chega ORDENADA POR CLASSE: 61 ADHD seguidos de
        # 60 controle, com exatamente uma troca. Pegar os N primeiros da uma
        # amostra de uma classe so — medido: `--max-sujeitos 10` produzia dez
        # ADHD e a regressao logistica recusava treinar.
        #
        # Isso nao e detalhe do modo de ensaio: e a mesma ordenacao que
        # tornaria FALSO qualquer cegamento feito escondendo o rotulo sem
        # embaralhar.
        por_classe = {}
        for s in sujeitos:
            por_classe.setdefault(s["classe"], []).append(s)
        metade = max(1, max_sujeitos // max(1, len(por_classe)))
        escolhidos = []
        for lista in por_classe.values():
            escolhidos.extend(lista[:metade])
        sujeitos = escolhidos[:max_sujeitos]

    blocos_X, blocos_y, blocos_g = [], [], []
    nomes = None
    dec_epocas = dec_carac = dec_preproc = None
    falhas = []

    for idx, s in enumerate(sujeitos):
        sid = s["id"]
        try:
            bruto = df[df["ID"] == sid]
            dados = np.array([bruto[c].to_numpy(dtype=float)
                              for c in csv_data.CANAIS_19])

            # µV -> V para o MNE, e de volta; a mesma conversão que o app faz
            raw = preproc_basico.raw_de_dataframe(df, sid)
            filtrado, dec_preproc = preproc_basico.preprocessar(
                raw, l_freq=0.5, h_freq=None
            )
            dados = filtrado.get_data() * 1e6

            janelas, dec_epocas = mod_epocas.epocar_janela_fixa(
                dados, csv_data.FS, duracao_s, passo_s
            )
            if len(janelas) == 0:
                falhas.append((sid, "nenhuma época coube na gravação"))
                continue

            X, nomes, dec_carac = caracteristicas.potencia_de_banda(
                janelas, csv_data.FS, nomes_canais=list(csv_data.CANAIS_19),
                com_decisoes=True,
            )
            blocos_X.append(X)
            blocos_y.append(np.full(len(X), 1 if s["classe"] == "ADHD" else 0))
            blocos_g.append(np.full(len(X), idx))
        except Exception as e:            # noqa: BLE001 — sujeito que falha aparece
            falhas.append((sid, str(e)))

    if not blocos_X:
        raise RuntimeError("nenhum sujeito produziu época: nada a medir")

    decisoes = {
        "preproc": dec_preproc,
        "epocas": dec_epocas,
        "caracteristicas": dec_carac,
        "n_sujeitos_pedidos": len(sujeitos),
        "n_sujeitos_usados": len(blocos_X),
        # Sujeito que falhou APARECE. Sumir da contagem faria o denominador
        # mentir sem que nada na tela dissesse.
        "sujeitos_que_falharam": [{"id": s, "motivo": m} for s, m in falhas],
    }
    return (np.vstack(blocos_X), np.concatenate(blocos_y),
            np.concatenate(blocos_g), nomes, decisoes)


def _uma_condicao(X, y, grupos, tipo_split, normalizacao_global, n_dobras, semente):
    """Acurácia média e AUC média de uma célula da tabela.

    A diferença entre as células está inteira em duas linhas: qual gerador de
    split, e de ONDE saem os parâmetros de normalização."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    if tipo_split == "sujeito":
        dobras = list(mod_split.split_por_sujeito(grupos, n_dobras))
    else:
        dobras = list(mod_split.split_por_segmento(len(y), n_dobras, semente))

    # O VAZAMENTO DELIBERADO: ajustar no conjunto inteiro, teste incluído.
    # É uma linha visível, escrita de propósito, com `origem` declarada — que
    # é exatamente o que a separação ajustar/aplicar existe para tornar
    # possível sem virar descuido.
    params_global = (normalizacao.ajustar(X, origem="conjunto inteiro (vazado)")
                     if normalizacao_global else None)

    linhas = []
    for i, (treino, teste) in enumerate(dobras):
        p = params_global or normalizacao.ajustar(
            X[treino], origem=f"treino da dobra {i}"
        )
        Xtr = normalizacao.aplicar(X[treino], p)
        Xte = normalizacao.aplicar(X[teste], p)

        if len(np.unique(y[treino])) < 2:
            # Dobra sem as duas classes no treino nao produz modelo. Registrar
            # e seguir e mais honesto que estourar (perde-se o experimento
            # inteiro) ou que fingir 0,5 (inventa um numero).
            linhas.append({
                "dobra": i, "acuracia": float("nan"), "auc": float("nan"),
                "n_treino": int(len(treino)), "n_teste": int(len(teste)),
            })
            continue

        modelo = LogisticRegression(max_iter=2000, random_state=semente)
        modelo.fit(Xtr, y[treino])
        pred = modelo.predict(Xte)

        try:
            auc = roc_auc_score(y[teste], modelo.predict_proba(Xte)[:, 1])
        except ValueError:
            auc = float("nan")   # dobra com uma classe só; não inventa 0,5
        linhas.append({
            "dobra": i,
            "acuracia": float(accuracy_score(y[teste], pred)),
            "auc": float(auc),
            "n_treino": int(len(treino)),
            "n_teste": int(len(teste)),
        })
    return linhas


def rodar(duracao_s=4.0, passo_s=4.0, n_dobras=5, semente=0, max_sujeitos=None):
    """As quatro condições, e tudo o que é preciso para refazê-las."""
    df = csv_data.load_csv(config.CAMINHO_ADHDATA)
    X, y, grupos, nomes, dec = montar_conjunto(df, duracao_s, passo_s, max_sujeitos)

    condicoes = [
        ("A", "segmento", True),
        ("B", "segmento", False),
        ("C", "sujeito", True),
        ("D", "sujeito", False),
    ]
    resultados = []
    for rotulo, tipo, glob in condicoes:
        for linha in _uma_condicao(X, y, grupos, tipo, glob, n_dobras, semente):
            resultados.append({
                "condicao": rotulo,
                "split": tipo,
                "normalizacao": "global" if glob else "por dobra",
                **linha,
            })

    dec["split"] = mod_split.decisoes_do_split("sujeito", grupos, n_dobras, semente)
    dec["classificador"] = {
        "modelo": "LogisticRegression",
        "max_iter": 2000,
        "semente": semente,
        "observacao": "escolhido por ser transparente; o resultado é a DIFERENÇA "
                      "entre condições, não a acurácia absoluta",
    }
    receita = mod_receita.montar(
        banco="adhdata",
        sujeitos=[f"{dec['n_sujeitos_usados']} sujeitos"],
        etapas=dec,
        notas=RESSALVA_LICENCA,
    )
    return resultados, receita, nomes


def resumo(resultados):
    """Média e desvio por condição — o que vira barra."""
    saida = {}
    for cond in ("A", "B", "C", "D"):
        acs = [r["acuracia"] for r in resultados if r["condicao"] == cond
               and not np.isnan(r["acuracia"])]
        aucs = [r["auc"] for r in resultados if r["condicao"] == cond
                and not np.isnan(r["auc"])]
        saida[cond] = {
            "acuracia_media": float(np.mean(acs)) if acs else float("nan"),
            "acuracia_desvio": float(np.std(acs)) if acs else float("nan"),
            "auc_media": float(np.mean(aucs)) if aucs else float("nan"),
            "n_dobras": len(acs),
        }
    return saida


def salvar_csv(resultados, caminho):
    """Uma linha por dobra e condição.

    O CSV é o que permite refazer a FIGURA sem refazer o experimento — e é
    também o que permite alguém conferir a média que a barra mostra."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    campos = ["condicao", "split", "normalizacao", "dobra",
              "acuracia", "auc", "n_treino", "n_teste"]
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(resultados)
    return caminho


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epoca", type=float, default=4.0, help="duração da época em s")
    p.add_argument("--passo", type=float, default=4.0, help="passo entre épocas em s")
    p.add_argument("--dobras", type=int, default=5)
    p.add_argument("--semente", type=int, default=0)
    p.add_argument("--max-sujeitos", type=int, default=None,
                   help="ensaio rápido com os N primeiros")
    p.add_argument("--saida", default=None)
    args = p.parse_args()

    destino = Path(args.saida) if args.saida else config.caminho_relatorios()

    print(f"[experimento] epocando em janelas de {args.epoca} s, passo {args.passo} s…",
          flush=True)
    resultados, receita, _ = rodar(args.epoca, args.passo, args.dobras,
                                   args.semente, args.max_sujeitos)

    dec = receita["etapas"]
    print(f"[experimento] {dec['n_sujeitos_usados']} de {dec['n_sujeitos_pedidos']} "
          f"sujeitos, {dec['split']['n_epocas']} épocas", flush=True)
    for f in dec["sujeitos_que_falharam"]:
        print(f"[experimento]   FALHOU {f['id']}: {f['motivo']}", flush=True)

    r = resumo(resultados)
    print()
    print(f"{'':22} {'acurácia':>18} {'AUC':>8}")
    nomes_cond = {
        "A": "segmento + global",
        "B": "segmento + por dobra",
        "C": "sujeito + global",
        "D": "sujeito + por dobra",
    }
    for cond in ("A", "B", "C", "D"):
        v = r[cond]
        print(f"{cond}  {nomes_cond[cond]:19} "
              f"{v['acuracia_media']:.3f} ± {v['acuracia_desvio']:.3f}   "
              f"{v['auc_media']:.3f}")
    print()
    print(f"queda de A para D: {r['A']['acuracia_media'] - r['D']['acuracia_media']:+.3f}")
    print(f"[ressalva] {RESSALVA_LICENCA}")

    csv_saida = salvar_csv(resultados, destino / "vazamento_resultados.csv")
    rec_saida = mod_receita.salvar(receita, destino / "vazamento_receita.json")
    print(f"[experimento] {csv_saida}")
    print(f"[experimento] {rec_saida}")


if __name__ == "__main__":
    main()
