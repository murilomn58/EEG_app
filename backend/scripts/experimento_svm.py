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
    python scripts/experimento_svm.py --n-jobs 4   # limita núcleos; padrão -1 (todos)
"""
import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import caracteristicas
import classificador
import config
import csv_data
import epocas as mod_epocas
import estatistica
import exportar_features
import preproc_basico
import receita as mod_receita


def _potencia_de_banda_do_conjunto(df, duracao_s, passo_s, max_sujeitos):
    """As 95 características de potência, para a condição B.

    Repete a montagem em vez de generalizar `montar_conjunto_tbr` com um
    parâmetro de tipo de característica: a montagem é curta, e um parâmetro que
    troca o que é extraído esconderia, numa flag, a diferença entre duas
    condições do experimento.

    Sujeito que falha APARECE, e não some da contagem: a condição B só é
    comparável com a A se as duas rodarem sobre o mesmo conjunto, e um n que
    encolhe em silêncio faz a comparação mentir."""
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
    falhas = []
    for idx, s in enumerate(sujeitos):
        try:
            raw = preproc_basico.raw_de_dataframe(df, s["id"])
            filtrado, _ = preproc_basico.preprocessar(raw, l_freq=0.5, h_freq=None)
            janelas, _ = mod_epocas.epocar_janela_fixa(
                filtrado.get_data() * 1e6, csv_data.FS, duracao_s, passo_s
            )
            if len(janelas) == 0:
                falhas.append((s["id"], "nenhuma época coube na gravação"))
                continue
            X, _ = caracteristicas.potencia_de_banda(
                janelas, csv_data.FS, nomes_canais=list(csv_data.CANAIS_19)
            )
            bX.append(X)
            by.append(np.full(len(X), 1 if s["classe"] == "ADHD" else 0))
            bg.append(np.full(len(X), idx))
        except Exception as e:            # noqa: BLE001
            falhas.append((s["id"], str(e)))
    return np.vstack(bX), np.concatenate(by), np.concatenate(bg), falhas


def _agora_monotonica():
    """`time.time()`, isolado numa função só para o cronômetro de cada condição
    ser trivial de substituir num teste, se algum dia precisar."""
    return time.time()


def _ic_da_condicao(resultado):
    """Erro-padrão e IC95% da AUC de uma condição, via Hanley-McNeil.

    `n1` é a contagem de sujeitos ADHD (rótulo positivo, 1) e `n0` a de
    Control (rótulo negativo, 0) — a mesma convenção de `rotulos_por_sujeito`,
    que já vem como 0/1 desde `yt = (yt_txt == "ADHD").astype(int)` em
    `rodar()`. Não se aplica à condição D: o "AUC" ali é a média da
    distribuição nula por permutação, não uma AUC de classificação normal, e
    o erro-padrão de Hanley-McNeil pressupõe a segunda coisa."""
    rotulos = resultado["rotulos_por_sujeito"]
    n1 = int(np.sum(rotulos == 1))
    n0 = len(rotulos) - n1
    erro_padrao = estatistica.erro_padrao_auc_hanley_mcneil(
        resultado["auc"], n0, n1
    )
    ic_inf, ic_sup = estatistica.ic95_auc(resultado["auc"], n0, n1)
    return erro_padrao, ic_inf, ic_sup


def rodar(df=None, duracao_s=4.0, passo_s=4.0, max_sujeitos=None,
          n_permutacoes=100, n_dobras_internas=5, semente=0,
          progresso=None, checkpoint=None, retomar_nulo=False,
          pular_nulo=False, n_jobs=1):
    """As quatro condições, e a receita para refazê-las.

    POR QUE PROGRESSO E CHECKPOINT AQUI, E NÃO SÓ NO NULO

    O nulo (condição D) é o que mais demora, mas A, B e C já passam minutos
    cada, e a rodada inteira leva ~8h sem imprimir nada e sem salvar nada até
    o fim — isso já causou um diagnóstico errado, lido como processo travado
    quando na verdade progredia. `progresso`, se não `None`, anuncia o início
    e o fim de cada condição; `checkpoint`, se não `None`, grava o CSV das
    condições já concluídas depois de cada uma, para que matar o processo no
    meio não jogue fora o que já rodou.

    `n_jobs` propaga para as quatro chamadas de `avaliar_loso` (via `A`, `B`,
    `C` diretamente, e via `D` dentro de `nulo_por_permutacao`, que repassa
    por `**kw`). O padrão aqui também é `1`, pela mesma razão de
    `avaliar_loso`: não mudar o comportamento de quem já chama `rodar()` sem
    esse argumento. Quem quer velocidade real usa o `--n-jobs` do CLI, cujo
    padrão é `-1`."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if df is None:
        df = csv_data.load_csv(config.CAMINHO_ADHDATA)

    Xt, yt_txt, gt, _, _, _, dec = exportar_features.montar_conjunto_tbr(
        df, duracao_s, passo_s, max_sujeitos
    )
    yt = (yt_txt == "ADHD").astype(int)

    if progresso is not None:
        progresso(
            f"montagem TBR: {dec['n_sujeitos_usados']} sujeitos, "
            f"{len(yt)} épocas, {Xt.shape[1]} features"
        )

    resultados = []

    def _salvar_checkpoint():
        if checkpoint is not None:
            salvar_csv(resultados, checkpoint)

    if progresso is not None:
        progresso(
            f"condição A (TBR + SVM-RBF): iniciando, {len(np.unique(gt))} "
            "dobras LOSO"
        )
    t0 = _agora_monotonica()
    a = classificador.avaliar_loso(
        Xt, yt, gt, n_dobras_internas=n_dobras_internas, semente=semente,
        n_jobs=n_jobs,
    )
    erro_padrao_a, ic_inf_a, ic_sup_a = _ic_da_condicao(a)
    resultados.append({"condicao": "A", "features": "TBR (19)", "modelo": "SVM-RBF",
                       "auc": a["auc"], "acuracia": a["acuracia"],
                       "n_sujeitos": a["n_sujeitos"], "p_empirico": None,
                       "erro_padrao": erro_padrao_a,
                       "auc_ic95_inf": ic_inf_a, "auc_ic95_sup": ic_sup_a})
    if progresso is not None:
        progresso(
            f"condição A: AUC {a['auc']:.3f}, acurácia {a['acuracia']:.3f}, "
            f"{_agora_monotonica() - t0:.0f}s"
        )
    _salvar_checkpoint()

    if progresso is not None:
        progresso(
            f"condição B (potência de banda + SVM-RBF): iniciando, "
            f"{len(np.unique(gt))} dobras LOSO"
        )
    t0 = _agora_monotonica()
    Xb, yb, gb, falhas_b = _potencia_de_banda_do_conjunto(
        df, duracao_s, passo_s, max_sujeitos
    )
    if progresso is not None:
        for sid, motivo in falhas_b:
            progresso(f"FALHOU {sid}: {motivo}")
    b = classificador.avaliar_loso(
        Xb, yb, gb, n_dobras_internas=n_dobras_internas, semente=semente,
        n_jobs=n_jobs,
    )
    erro_padrao_b, ic_inf_b, ic_sup_b = _ic_da_condicao(b)
    resultados.append({"condicao": "B", "features": "potência de banda (95)",
                       "modelo": "SVM-RBF", "auc": b["auc"],
                       "acuracia": b["acuracia"], "n_sujeitos": b["n_sujeitos"],
                       "p_empirico": None,
                       "erro_padrao": erro_padrao_b,
                       "auc_ic95_inf": ic_inf_b, "auc_ic95_sup": ic_sup_b})
    if progresso is not None:
        progresso(
            f"condição B: AUC {b['auc']:.3f}, acurácia {b['acuracia']:.3f}, "
            f"{_agora_monotonica() - t0:.0f}s"
        )
    _salvar_checkpoint()

    if progresso is not None:
        progresso(
            f"condição C (TBR + LogisticRegression): iniciando, "
            f"{len(np.unique(gt))} dobras LOSO"
        )
    t0 = _agora_monotonica()
    logreg = Pipeline([("escala", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=2000,
                                                  random_state=semente))])
    c = classificador.avaliar_loso(
        Xt, yt, gt, estimador=logreg, grade={"clf__C": [0.1, 1.0, 10.0]},
        n_dobras_internas=n_dobras_internas, semente=semente, n_jobs=n_jobs,
    )
    erro_padrao_c, ic_inf_c, ic_sup_c = _ic_da_condicao(c)
    resultados.append({"condicao": "C", "features": "TBR (19)",
                       "modelo": "LogisticRegression", "auc": c["auc"],
                       "acuracia": c["acuracia"], "n_sujeitos": c["n_sujeitos"],
                       "p_empirico": None,
                       "erro_padrao": erro_padrao_c,
                       "auc_ic95_inf": ic_inf_c, "auc_ic95_sup": ic_sup_c})
    if progresso is not None:
        progresso(
            f"condição C: AUC {c['auc']:.3f}, acurácia {c['acuracia']:.3f}, "
            f"{_agora_monotonica() - t0:.0f}s"
        )
    _salvar_checkpoint()

    d = None
    if pular_nulo:
        if progresso is not None:
            progresso(
                "condição D pulada por --sem-nulo: o resultado não terá p-valor"
            )
    else:
        if progresso is not None:
            progresso(
                f"condição D (nulo, {n_permutacoes} permutações): iniciando; "
                f"custa {n_permutacoes}+1 avaliações LOSO completas — pode "
                "levar horas"
            )
        checkpoint_nulo = (
            Path(checkpoint).with_suffix(".nulo.json")
            if checkpoint is not None else None
        )
        d = classificador.nulo_por_permutacao(
            Xt, yt, gt, n_permutacoes=n_permutacoes,
            n_dobras_internas=n_dobras_internas, semente=semente,
            progresso=progresso, checkpoint=checkpoint_nulo,
            retomar=retomar_nulo, n_jobs=n_jobs,
        )
        resultados.append({"condicao": "D", "features": "TBR (19)",
                           "modelo": "SVM-RBF, rótulos permutados",
                           "auc": d["media_nula"], "acuracia": None,
                           "n_sujeitos": a["n_sujeitos"],
                           "p_empirico": d["p_empirico"],
                           "erro_padrao": None,
                           "auc_ic95_inf": None, "auc_ic95_sup": None})
        _salvar_checkpoint()

    dec["avaliacao"] = a["decisoes"]
    if d is not None:
        dec["nulo"] = d["decisoes"]
    dec["sujeitos_que_falharam_na_potencia_de_banda"] = [
        {"id": s, "motivo": m} for s, m in falhas_b
    ]
    receita = mod_receita.montar(
        banco="adhdata",
        sujeitos=[f"{dec['n_sujeitos_usados']} sujeitos"],
        etapas=dec,
        notas=exportar_features.RESSALVA_LICENCA,
    )
    return resultados, receita


def salvar_csv(resultados, caminho):
    """Uma linha por condição. Escrita ATÔMICA.

    POR QUE ATÔMICA

    Esta função é chamada como checkpoint intermediário, depois de CADA
    condição — e também como gravação final de `main()`. Escrever direto no
    caminho final deixa uma janela em que uma morte do processo NO MEIO da
    escrita produz um CSV truncado que parece válido (abre, tem cabeçalho) e
    não é. A correção: escrever num arquivo temporário e então renomear por
    cima do destino com `os.replace`, que é atômico no mesmo volume — inclusive
    no Windows. Um leitor nunca vê o arquivo final em estado parcial: ou lê a
    versão anterior completa, ou já lê a nova completa."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    campos = ["condicao", "features", "modelo", "auc", "acuracia",
              "n_sujeitos", "p_empirico", "erro_padrao",
              "auc_ic95_inf", "auc_ic95_sup"]
    temporario = caminho.with_suffix(caminho.suffix + ".parcial")
    with temporario.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(resultados)
    os.replace(temporario, caminho)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epoca", type=float, default=4.0)
    p.add_argument("--passo", type=float, default=4.0)
    p.add_argument("--max-sujeitos", type=int, default=None)
    p.add_argument("--permutacoes", type=int, default=100)
    p.add_argument("--dobras-internas", type=int, default=5)
    p.add_argument("--saida", default="relatorios/experimento_svm.csv")
    p.add_argument("--sem-progresso", action="store_true",
                   help="não imprime progresso; a rodada volta a ficar muda "
                        "até o fim")
    p.add_argument("--retomar", action="store_true",
                   help="retoma o nulo (condição D) a partir do checkpoint "
                        "existente, em vez de recomeçar do zero")
    p.add_argument("--sem-nulo", action="store_true",
                   help="pula a condição D inteira; o resultado sai sem "
                        "p-valor")
    # Padrão -1 aqui, e não 1 como em `rodar()`/`avaliar_loso`: quem chama a
    # função de código quer o comportamento antigo por padrão (nenhum teste
    # existente passa `n_jobs`), mas quem roda da linha de comando está aqui
    # justamente para uma rodada real e quer todos os núcleos por padrão.
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="núcleos para as dobras externas do LOSO (padrão -1: "
                        "todos os núcleos disponíveis)")
    args = p.parse_args()

    def diga(t):
        print(f"[svm] {t}", flush=True)

    resultados, receita = rodar(
        duracao_s=args.epoca, passo_s=args.passo,
        max_sujeitos=args.max_sujeitos, n_permutacoes=args.permutacoes,
        n_dobras_internas=args.dobras_internas,
        progresso=None if args.sem_progresso else diga,
        checkpoint=args.saida,
        retomar_nulo=args.retomar,
        pular_nulo=args.sem_nulo,
        n_jobs=args.n_jobs,
    )
    salvar_csv(resultados, args.saida)
    mod_receita.salvar(receita, Path(args.saida).with_suffix(".receita.json"))

    print(f"{'cond':<5}{'features':<24}{'modelo':<30}{'AUC':>7}{'p':>8}{'IC95%':>14}")
    for r in resultados:
        pe = "" if r["p_empirico"] is None else f"{r['p_empirico']:.3f}"
        ic = ("" if r["auc_ic95_inf"] is None else
              f"[{r['auc_ic95_inf']:.2f};{r['auc_ic95_sup']:.2f}]")
        print(f"{r['condicao']:<5}{r['features']:<24}{r['modelo']:<30}"
              f"{r['auc']:>7.3f}{pe:>8}{ic:>14}")

    resultado_a = next(r for r in resultados if r["condicao"] == "A")
    if resultado_a["erro_padrao"] is not None:
        # Diferença mínima detectável (95%) entre duas AUCs independentes com
        # erros-padrão próximos: 1,96 * sqrt(SE_a² + SE_b²) ≈ 1,96 * sqrt(2) *
        # SE quando SE_a ≈ SE_b. Usa o erro-padrão de A (a condição principal)
        # como referência das quatro condições, que têm n de sujeitos parecido.
        diferenca_minima = 1.96 * math.sqrt(2) * resultado_a["erro_padrao"]
        print(
            "\nDiferença mínima de AUC detectável entre condições (95%, a "
            f"partir do erro-padrão de A): ~{diferenca_minima:.2f}"
        )
    print(f"\n{exportar_features.RESSALVA_LICENCA}")


if __name__ == "__main__":
    main()
