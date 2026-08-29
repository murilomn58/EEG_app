# -*- coding: utf-8 -*-
"""Normaliza características, com AJUSTE e APLICAÇÃO separados.

A SEPARAÇÃO É O MÓDULO

Normalizar com `fit_transform` sobre o conjunto inteiro é uma linha só,
parece organizado, e vaza: a média e o desvio usados no treino carregam
informação do teste. O classificador vê, por via indireta, dados que não
deveria ver, e a acurácia sobe.

É um vazamento diferente do de split, e mais perigoso justamente por ser
discreto. O de split é grosseiro — épocas do mesmo sujeito dos dois lados —
e qualquer revisor pergunta por ele. O de normalização sobrevive em código
que parece correto, e não deixa rastro no resultado.

Este módulo não impede o vazamento por disciplina, e sim por TIPO: `aplicar`
exige um `params`, e `params` só nasce de `ajustar`. Ajustar no conjunto
inteiro continua possível — precisa continuar, porque é a condição de
controle da figura — mas passa a ser uma linha que alguém escreveu de
propósito, com `origem` declarada, em vez de um descuido.

NÃO ACRESCENTE `fit_transform` AQUI. Existe um teste que falha se alguém o
acrescentar, e ele existe para que a conversa aconteça antes de o atalho
virar hábito.
"""
import numpy as np

# Piso para o desvio, para não dividir por zero.
#
# Canal morto tem desvio exatamente zero — o `verificar_referencia` já
# encontra esse caso em dado real. Sem piso, a divisão dá inf ou NaN, que
# atravessa o classificador inteiro sem levantar nada e só aparece como
# resultado estranho no fim.
#
# O valor normaliza a coluna constante para zero, que é a resposta correta:
# uma característica sem variância não informa nada, e zero é o que ela vale
# depois de centrada.
_PISO_DESVIO = 1e-12


def ajustar(X, origem=None):
    """Calcula os parâmetros de normalização. NÃO transforma nada.

    Devolver os dados transformados aqui faria disto um `fit_transform` com
    outro nome, e a separação — que é o ponto do módulo — deixaria de
    existir.

    `origem` é texto livre dizendo de onde saiu esta estatística ("treino da
    dobra 2", "conjunto inteiro"). Vai para a receita, e é o único campo que
    distingue os dois braços do experimento: sem ele, a condição vazada e a
    honesta produzem receitas idênticas."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X tem de ser (n_amostras, n_caracteristicas), recebi {X.shape}")
    return {
        "media": X.mean(axis=0),
        "desvio": X.std(axis=0),
        "origem": origem,
        "n_amostras_do_ajuste": int(X.shape[0]),
        "n_caracteristicas": int(X.shape[1]),
    }


def aplicar(X, params):
    """Aplica parâmetros JÁ AJUSTADOS. `params` é obrigatório.

    A obrigatoriedade é a defesa: não existe caminho neste módulo que
    normalize sem alguém ter ajustado antes e ter dito onde."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X tem de ser (n_amostras, n_caracteristicas), recebi {X.shape}")
    if X.shape[1] != params["n_caracteristicas"]:
        raise ValueError(
            f"os parâmetros foram ajustados com {params['n_caracteristicas']} "
            f"característica(s) e X tem {X.shape[1]}"
        )
    desvio = np.maximum(params["desvio"], _PISO_DESVIO)
    return (X - params["media"]) / desvio


def decisoes(params):
    """O registro que vai para a receita.

    `origem` é o campo que importa: é ele que permite a outra pessoa saber se
    aquela barra foi produzida ajustando só no treino ou no conjunto todo."""
    return {
        "metodo": "z-score por característica",
        "origem": params.get("origem"),
        "n_amostras_do_ajuste": params["n_amostras_do_ajuste"],
        "n_caracteristicas": params["n_caracteristicas"],
        "piso_desvio": _PISO_DESVIO,
        "caracteristicas_sem_variancia": int(
            np.sum(np.asarray(params["desvio"]) <= _PISO_DESVIO)
        ),
    }
