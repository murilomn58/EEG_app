# -*- coding: utf-8 -*-
"""Classificação com SVM sob validação cruzada aninhada por sujeito.

POR QUE A AGREGAÇÃO POR SUJEITO EXISTE

O classificador prediz por época, e a pergunta clínica é por criança. Entre a
época e a criança há uma escolha que muda o resultado e costuma ficar
implícita: como as N épocas de um sujeito viram um número.

Aqui é a MEDIANA. O adhdata tem sujeitos com 62 s e sujeitos com 338 s de
gravação — uma razão de 5,4 —, e a média deixaria uma única época contaminada
por artefato deslocar o escore de um sujeito inteiro. A mediana não.
"""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import (GridSearchCV, LeaveOneGroupOut,
                                     StratifiedGroupKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# A grade do interno. Pequena de propósito: cada valor a mais multiplica 121
# dobras externas por 5 internas, e uma grade generosa aqui compra decimais de
# AUC ao custo de horas.
GRADE_SVM = {
    "svm__C": [0.1, 1.0, 10.0, 100.0],
    "svm__gamma": ["scale", 0.01, 0.1],
}


def estimador_svm():
    """SVM RBF com a escala DENTRO do pipeline.

    A escala tem de ser um passo do Pipeline, e não uma transformação aplicada
    antes: é isso que faz o sklearn reajustá-la a cada dobra, em vez de usar uma
    média calculada com o sujeito de teste dentro.

    `probability` fica no default (False), e a AUC sai de `decision_function`.
    Ligá-lo acrescentaria um Platt scaling com CV próprio, multiplicando o custo
    por cerca de cinco para produzir a mesma ordenação. Passá-lo explicitamente,
    mesmo como False, dispara aviso de depreciação no sklearn 1.9 e deixa de
    funcionar na 1.11."""
    return Pipeline([
        ("escala", StandardScaler()),
        ("svm", SVC(kernel="rbf")),
    ])


def agregar_por_sujeito(escores, grupos, rotulos):
    """(escores_sujeito, rotulos_sujeito, ids_sujeito), na ordem de `np.unique`.

    Recusa sujeito com mais de um rótulo em vez de escolher um. Um sujeito com
    dois rótulos significa que a montagem do conjunto está errada, e devolver o
    primeiro esconderia o defeito atrás de um número plausível."""
    escores = np.asarray(escores, dtype=float)
    grupos = np.asarray(grupos)
    rotulos = np.asarray(rotulos)

    if not (len(escores) == len(grupos) == len(rotulos)):
        raise ValueError(
            f"escores ({len(escores)}), grupos ({len(grupos)}) e rotulos "
            f"({len(rotulos)}) têm de ter o mesmo comprimento"
        )

    ids = np.unique(grupos)
    esc_s = np.empty(len(ids), dtype=float)
    rot_s = np.empty(len(ids), dtype=rotulos.dtype)

    for i, sid in enumerate(ids):
        dentro = grupos == sid
        rot_unicos = np.unique(rotulos[dentro])
        if len(rot_unicos) > 1:
            raise ValueError(
                f"sujeito {sid!r} tem mais de um rótulo ({rot_unicos.tolist()}): "
                f"o conjunto foi montado errado"
            )
        esc_s[i] = float(np.median(escores[dentro]))
        rot_s[i] = rot_unicos[0]

    return esc_s, rot_s, ids


def avaliar_loso(X, y, grupos, estimador=None, grade=None,
                 n_dobras_internas=5, semente=0):
    """Nested CV: LOSO externo, StratifiedGroupKFold interno para os hiperparâmetros.

    POR QUE A AUC SAI DE UMA CONTA SÓ, NO FIM

    Cada dobra externa testa UM sujeito. Uma dobra com um sujeito tem uma classe
    só, e a AUC de uma classe só é indefinida — o sklearn avisa e devolve NaN.
    Uma 'AUC média das dobras' aqui seria a média de 121 NaN.

    O que se faz: acumular o escore de cada sujeito ao longo das dobras e
    calcular a AUC uma vez, sobre o vetor de escores e o vetor de rótulos.

    O escore de cada época sai de `decision_function`, que é a distância
    assinada ao hiperplano. Não é probabilidade e não precisa ser: a AUC depende
    só da ordenação."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    grupos = np.asarray(grupos)

    if len(np.unique(grupos)) < 2:
        raise ValueError(
            f"LOSO precisa de pelo menos 2 sujeitos, há {len(np.unique(grupos))}"
        )

    estimador = estimador if estimador is not None else estimador_svm()
    grade = grade if grade is not None else GRADE_SVM

    escores = np.empty(len(y), dtype=float)
    dobras = []

    for treino, teste in LeaveOneGroupOut().split(X, y, groups=grupos):
        g_treino = grupos[treino]
        sujeito_teste = np.unique(grupos[teste])[0]

        # O interno nunca vê o sujeito de teste: ele parte de `treino`.
        #
        # StratifiedGroupKFold, e NÃO GroupKFold. Medido em 08/09/2026: o
        # GroupKFold respeita a fronteira de sujeito mas ignora a classe, e com
        # poucos sujeitos ele produz dobra interna com UMA CLASSE SÓ — o SVC
        # levanta "The number of classes has to be greater than one; got 1
        # class" e o experimento inteiro morre no meio. O estratificado respeita
        # as duas restrições ao mesmo tempo.
        n_int = min(n_dobras_internas, len(np.unique(g_treino)))
        busca = GridSearchCV(
            estimador, grade,
            cv=StratifiedGroupKFold(n_splits=n_int, shuffle=True,
                                    random_state=semente),
            scoring="roc_auc",
            n_jobs=1,
        )
        busca.fit(X[treino], y[treino], groups=g_treino)
        escores[teste] = busca.best_estimator_.decision_function(X[teste])

        dobras.append({
            "sujeito_de_teste": sujeito_teste.item()
            if hasattr(sujeito_teste, "item") else sujeito_teste,
            "sujeitos_de_treino": [
                s.item() if hasattr(s, "item") else s
                for s in np.unique(g_treino)
            ],
            "melhores_parametros": busca.best_params_,
            "n_dobras_internas": int(n_int),
        })

    esc_s, rot_s, ids = agregar_por_sujeito(escores, grupos, y)

    if len(np.unique(rot_s)) < 2:
        raise ValueError(
            "todos os sujeitos têm o mesmo rótulo: a AUC não é definida"
        )

    auc = float(roc_auc_score(rot_s, esc_s))
    predito = (esc_s > 0).astype(int)

    return {
        "auc": auc,
        "acuracia": float(accuracy_score(rot_s, predito)),
        "n_sujeitos": int(len(ids)),
        "escores_por_sujeito": esc_s,
        "rotulos_por_sujeito": rot_s,
        "ids_sujeito": ids,
        "hiperparametros_por_dobra": [d["melhores_parametros"] for d in dobras],
        "decisoes": {
            "externo": "LeaveOneGroupOut (LOSO)",
            "interno": f"StratifiedGroupKFold({n_dobras_internas}) com scoring roc_auc",
            "agregacao": "mediana dos escores das épocas do sujeito",
            "escore": "decision_function (distância assinada ao hiperplano)",
            "auc": "calculada UMA vez sobre os escores de sujeito; a AUC por "
                   "dobra é indefinida no LOSO porque a dobra tem uma classe só",
            "grade": {k: list(v) for k, v in grade.items()},
            "semente": int(semente),
            "n_epocas": int(len(y)),
            "dobras": dobras,
        },
    }


def permutar_rotulos_por_sujeito(y, grupos, semente=0):
    """Embaralha os rótulos ENTRE sujeitos, mantendo-os constantes DENTRO.

    A distinção é o nulo inteiro. Permutar por época daria a cada sujeito as
    duas classes; nenhum classificador aprenderia nada, a AUC nula cairia bem
    abaixo de 0,5 e qualquer resultado observado passaria por significativo.

    Permutando por sujeito, o nulo preserva tudo o que não é o rótulo — o
    desequilíbrio de épocas, a estrutura de correlação dentro do sujeito — e
    mede exatamente o que se quer medir."""
    y = np.asarray(y)
    grupos = np.asarray(grupos)

    ids = np.unique(grupos)
    rot_por_sujeito = np.array([np.unique(y[grupos == s])[0] for s in ids])

    embaralhados = rot_por_sujeito.copy()
    np.random.default_rng(semente).shuffle(embaralhados)

    saida = np.empty_like(y)
    for sid, rot in zip(ids, embaralhados):
        saida[grupos == sid] = rot
    return saida


def nulo_por_permutacao(X, y, grupos, n_permutacoes=100, semente=0,
                        progresso=None, checkpoint=None, retomar=False, **kw):
    """AUC observada, distribuição sob o nulo e p-valor empírico.

    O p-valor usa a correção de Phipson e Smyth (2010), `(b + 1) / (m + 1)`:
    com m permutações, um p-valor de zero é impossível de sustentar, e o
    estimador ingênuo `b / m` produz exatamente isso quando nenhuma permutação
    supera a observação.

    POR QUE ESTA FUNÇÃO PRECISA DE PROGRESSO E CHECKPOINT

    Cada permutação é uma avaliação LOSO completa — minutos, não segundos —, e
    com `n_permutacoes=100` a rodada inteira leva horas sem imprimir nada e
    sem salvar nada até o fim. Isso já causou diagnóstico errado: um processo
    que progride normalmente foi lido como travado por falta de sinal
    externo. `progresso` (um chamável `f(texto)`) e `checkpoint` (um caminho
    de JSON) existem para que a rodada seja observável e retomável sem mudar
    o resultado matemático de quem não usa nenhum dos dois — por isso os dois
    são `None` por padrão e nada muda para quem já chama esta função sem
    eles.

    `progresso` e `checkpoint` são extraídos como parâmetros NOMEADOS, antes
    de `**kw` seguir para `avaliar_loso`. `avaliar_loso` não conhece esses
    dois nomes: se vazassem dentro do `**kw` repassado adiante, o resultado
    seria um `TypeError` que só aparece no dia em que alguém finalmente passa
    `progresso=` — no pior momento possível, no meio de uma rodada de horas.

    POR QUE A RETOMADA É MATEMATICAMENTE CORRETA

    Cada permutação usa `semente=semente + i + 1`: função pura do índice `i`,
    sem estado compartilhado entre iterações. Retomar do índice k e continuar
    até n produz EXATAMENTE as mesmas n permutações que uma execução contínua
    produziria. Isso deixaria de valer se a fonte de aleatoriedade virasse
    algo com estado (um único `np.random.default_rng` reaproveitado entre
    iterações, por exemplo) — por isso o laço abaixo constrói a semente de
    cada permutação a partir do índice, nunca de um gerador que carrega
    memória de uma iteração para a próxima."""
    if n_permutacoes < 1:
        raise ValueError(
            f"n_permutacoes={n_permutacoes} produziria uma média sobre lista "
            "vazia (nan). Para pular a condição do nulo, use --sem-nulo em "
            "vez de --permutacoes 0."
        )

    caminho_checkpoint = Path(checkpoint) if checkpoint is not None else None

    r_obs = avaliar_loso(X, y, grupos, semente=semente, **kw)
    auc_obs = r_obs["auc"]

    if progresso is not None:
        progresso(
            f"permutação 0/{n_permutacoes} | AUC observada {auc_obs:.3f} | "
            "avaliação de referência concluída"
        )

    aucs_salvas = []
    inicio_laco = 0

    if retomar and caminho_checkpoint is not None and caminho_checkpoint.exists():
        with caminho_checkpoint.open("r", encoding="utf-8") as f:
            estado = json.load(f)

        if estado.get("semente") != semente:
            raise ValueError(
                f"checkpoint tem semente={estado.get('semente')!r}, a chamada "
                f"atual pede semente={semente!r}: retomar misturaria dois "
                "nulos de sementes diferentes, o que produz uma distribuição "
                "sem procedência"
            )
        aucs_salvas = list(estado["aucs_nulas"])

        # Continuar para um n_permutacoes MAIOR que o do checkpoint é o uso
        # normal da retomada — é assim que uma rodada interrompida em 2 de
        # 100 volta a rodar até 100. O que não faz sentido, e por isso é
        # recusado, é o checkpoint já ter MAIS permutações computadas do que
        # a chamada atual pede: aceitar isso exigiria truncar a lista salva,
        # e uma distribuição truncada às cegas não tem a mesma procedência de
        # uma gerada com aquele n_permutacoes desde o início.
        if len(aucs_salvas) > n_permutacoes:
            raise ValueError(
                f"checkpoint já tem {len(aucs_salvas)} permutações computadas, "
                f"a chamada atual pede apenas n_permutacoes={n_permutacoes!r}: "
                "retomar exigiria truncar o checkpoint, o que produz uma "
                "distribuição sem a mesma procedência"
            )

        inicio_laco = len(aucs_salvas)

    aucs = list(aucs_salvas)
    tempos = []

    for i in range(inicio_laco, n_permutacoes):
        t0 = time.time()
        yp = permutar_rotulos_por_sujeito(y, grupos, semente=semente + i + 1)
        auc_i = avaliar_loso(X, yp, grupos, semente=semente, **kw)["auc"]
        aucs.append(auc_i)
        dt = time.time() - t0
        tempos.append(dt)

        if progresso is not None:
            media_ate_aqui = float(np.mean(aucs))
            tempo_medio = float(np.mean(tempos))
            restantes = n_permutacoes - (i + 1)
            fim_estimado = datetime.now() + timedelta(seconds=tempo_medio * restantes)
            progresso(
                f"permutação {i + 1}/{n_permutacoes} | AUC nula {auc_i:.3f} | "
                f"média até aqui {media_ate_aqui:.3f} | {dt:.1f}s | "
                f"fim estimado {fim_estimado:%H}:{fim_estimado:%M}"
            )

        if caminho_checkpoint is not None:
            with caminho_checkpoint.open("w", encoding="utf-8") as f:
                json.dump({
                    "semente": semente,
                    "n_permutacoes": n_permutacoes,
                    "auc_observada": auc_obs,
                    "aucs_nulas": aucs,
                }, f)

    aucs = np.array(aucs, dtype=float)

    b = int(np.sum(aucs >= auc_obs))
    return {
        "auc_observada": auc_obs,
        "aucs_nulas": aucs,
        "media_nula": float(aucs.mean()),
        "desvio_nulo": float(aucs.std()),
        "p_empirico": float((b + 1) / (n_permutacoes + 1)),
        "n_permutacoes": int(n_permutacoes),
        "decisoes": {
            "permutacao": "rótulo por SUJEITO, nunca por época",
            "p_valor": "(b+1)/(m+1), Phipson e Smyth 2010",
            "avaliacao": r_obs["decisoes"],
        },
    }
