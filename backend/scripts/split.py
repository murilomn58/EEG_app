# -*- coding: utf-8 -*-
"""Separa treino e teste, e recusa o split que vaza sujeito.

O QUE ESTÁ EM JOGO

Épocas de uma mesma criança são parecidas entre si por razões que não têm
nada a ver com o diagnóstico: impedância dos eletrodos, formato do crânio,
quanto ela se mexeu, a hora do dia. Um classificador treinado com algumas
épocas de uma criança e testado com OUTRAS épocas da mesma criança aprende a
reconhecer a criança. A acurácia sobe de verdade, e o número é publicável.

Medir quanto dessa subida some quando a fronteira de sujeito é respeitada é o
experimento inteiro que este módulo serve.

POR QUE EXISTEM DOIS SPLITS AQUI

`split_por_segmento` **não é um erro que ficou no código**. É a condição de
controle: o número inflado que a figura precisa mostrar ao lado do honesto.
Um experimento que só rodasse o split certo não teria com o que comparar, e a
afirmação "o split por sujeito derruba a acurácia" seria uma citação de
literatura em vez de uma medida deste projeto.

Por isso os dois são funções públicas com nomes que dizem o que fazem, e a
verificação de vazamento roda apenas no primeiro.
"""
import numpy as np
from sklearn.model_selection import GroupKFold, KFold


def verificar_sem_vazamento(indices_treino, indices_teste, grupos):
    """Levanta se algum sujeito aparece dos dois lados.

    Pública de propósito: quem montar um split à mão — por exemplo um
    leave-one-subject-out escrito na mão — passa por aqui antes de treinar.

    A mensagem NOMEIA os sujeitos que cruzaram. Um erro que diz apenas "houve
    vazamento" obriga a caçar qual foi, e a caçada acontece justamente na hora
    em que se está com pressa para ver o resultado."""
    grupos = np.asarray(grupos)
    nos_dois = set(grupos[np.asarray(indices_treino)].tolist()) & \
               set(grupos[np.asarray(indices_teste)].tolist())
    if nos_dois:
        raise ValueError(
            f"vazamento de sujeito: {sorted(map(str, nos_dois))} aparece(m) em "
            f"treino E teste. Épocas do mesmo sujeito nos dois lados fazem o "
            f"classificador aprender o sujeito em vez do rótulo."
        )


def split_por_sujeito(grupos, n_dobras):
    """Gera (treino, teste) com todas as épocas de um sujeito do mesmo lado.

    Cada dobra passa por `verificar_sem_vazamento` antes de sair. É
    redundante — o GroupKFold já garante — e é deliberado: a garantia fica
    afirmada no ponto de saída, e não depende de a biblioteca continuar se
    comportando como se espera numa versão futura."""
    grupos = np.asarray(grupos)
    n_sujeitos = len(np.unique(grupos))
    if n_sujeitos < 2:
        raise ValueError(
            f"split por sujeito precisa de pelo menos 2 sujeitos, há {n_sujeitos}"
        )
    if n_dobras > n_sujeitos:
        raise ValueError(
            f"não dá para fazer {n_dobras} dobras com {n_sujeitos} sujeitos: "
            f"alguma dobra ficaria sem sujeito para testar"
        )

    for treino, teste in GroupKFold(n_splits=n_dobras).split(np.zeros(len(grupos)),
                                                            groups=grupos):
        verificar_sem_vazamento(treino, teste, grupos)
        yield treino, teste


def split_por_segmento(n_epocas, n_dobras, semente=0):
    """Gera (treino, teste) embaralhando ÉPOCAS, ignorando de quem elas são.

    É a condição de controle, e vaza por construção. Não chame
    `verificar_sem_vazamento` aqui: ela levantaria, e o levantamento é o
    comportamento esperado desta função, não um defeito dela.

    A semente é obrigatória na prática porque sem ela dois runs do
    experimento produzem barras diferentes e ninguém sabe se a diferença veio
    do método ou do sorteio."""
    for treino, teste in KFold(n_splits=n_dobras, shuffle=True,
                               random_state=semente).split(np.zeros(n_epocas)):
        yield treino, teste


def decisoes_do_split(tipo, grupos, n_dobras, semente=0):
    """O registro que vai para a receita.

    `epocas_por_sujeito` entra porque o desequilíbrio importa para ler o
    resultado: gravações de duração diferente geram números diferentes de
    épocas, e o sujeito com mais épocas pesa mais. Quem lê a figura precisa
    do número para saber se está vendo patologia ou o sujeito mais comprido.

    As chaves saem como string porque este dicionário é serializado em JSON,
    e JSON não tem chave inteira — converter aqui evita que o mesmo dado
    volte diferente do arquivo."""
    grupos = np.asarray(grupos)
    unicos, contagens = np.unique(grupos, return_counts=True)
    return {
        "tipo": tipo,
        "n_dobras": int(n_dobras),
        "semente": int(semente),
        "n_epocas": int(len(grupos)),
        "n_sujeitos": int(len(unicos)),
        "epocas_por_sujeito": {str(u): int(c) for u, c in zip(unicos, contagens)},
        "min_epocas_por_sujeito": int(contagens.min()) if len(contagens) else 0,
        "max_epocas_por_sujeito": int(contagens.max()) if len(contagens) else 0,
    }
