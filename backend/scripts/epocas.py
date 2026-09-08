# -*- coding: utf-8 -*-
"""Recorta o sinal contínuo em épocas que NUNCA atravessam descontinuidade.

POR QUE ESTE MÓDULO EXISTE, E POR QUE A FRONTEIRA É O PONTO DELE

Epocar é dividir o sinal em janelas. A parte difícil não é dividir: é saber
onde NÃO se pode dividir. Uma janela que atravessa uma descontinuidade não
levanta erro nenhum — ela devolve um array do tamanho certo, com números
plausíveis, e segue viagem para o classificador.

As duas descontinuidades deste projeto são reais e medidas:

  adhdata — o arquivo é UM csv com as linhas de 121 crianças concatenadas, e
  a coluna `ID` é a única fronteira que existe. Uma janela que a cruza emenda
  **duas crianças numa época só**, e essa época recebe o rótulo de uma delas.
  Num experimento que mede vazamento entre sujeitos, é o pior defeito
  possível: ele fabrica exatamente o efeito que se quer medir.

  HBN — o evento `boundary` do EEGLAB marca onde o registro foi cortado e
  remendado. Está presente em 9 dos 10 sujeitos inventariados. Uma janela em
  cima dele contém um degrau que não é fisiologia, e que os filtros de banda
  transformam em transiente com energia em todas as bandas.

O módulo é deliberadamente burro: ele não sabe o que é um sujeito nem o que é
um evento. Recebe uma lista de CORTES em índice de amostra e respeita. Quem
sabe traduzir ID e `boundary` em corte são as duas funções auxiliares no topo,
e elas ficam separadas para que um banco novo entre acrescentando um tradutor,
não mexendo na epocagem.

O comprimento e o passo escolhidos aqui não são detalhe de implementação: são
os parâmetros que o tokenizador do transformer vai receber, porque a época é a
unidade que vira patch. Por isso eles saem nomeados em `decisoes`.
"""
import numpy as np


def cortes_por_mudanca(rotulos):
    """Índices onde o valor da sequência muda.

    É a fronteira de sujeito do adhdata: `rotulos` é a coluna `ID` linha a
    linha, e o corte cai na primeira amostra de cada criança nova."""
    cortes = []
    anterior = None
    for i, r in enumerate(rotulos):
        if i > 0 and r != anterior:
            cortes.append(i)
        anterior = r
    return cortes


def cortes_de_eventos(eventos, fs, valores=("boundary",)):
    """Índices de amostra dos eventos que marcam descontinuidade.

    O onset vem em SEGUNDOS no events.tsv; o corte é em amostra, e a
    conversão mora aqui num lugar só.

    Onset zero é ignorado de propósito: um `boundary` no instante 0 não corta
    nada — já é o começo da gravação — e aceitá-lo criaria um segmento vazio
    na frente que só serviria para confundir a contagem."""
    alvo = set(valores)
    saida = []
    for ev in eventos or ():
        if ev.get("valor") not in alvo:
            continue
        amostra = int(round(float(ev.get("onset", 0.0)) * fs))
        if amostra > 0:
            saida.append(amostra)
    return saida


def blocos_entre_eventos(eventos, fs, n_amostras, alvos, cortes=()):
    """[(inicio, fim, valor)] e decisoes: os blocos delimitados por eventos.

    `cortes_de_eventos` CORTA; esta função ANCORA. Um bloco vai do onset de
    um evento em `alvos` ao onset do PRÓXIMO evento em `alvos`, e carrega o
    valor do evento que o abriu. É o que permite epocar "olhos fechados" e
    "olhos abertos" separadamente no RestingState do HBN, em que
    `instructed_toOpenEyes` e `instructed_toCloseEyes` se alternam.

    Três decisões, todas medidas nos 10 events.tsv do ds005505:

      O ÚLTIMO alvo não fecha bloco. O 6.º `toOpenEyes` cai ~4,5 s antes de
      um `boundary` ou de um `break cnt`, e a tarefa seguinte começa logo
      depois; inventar um fim para ele seria epocar o intervalo entre
      tarefas como se fosse repouso. Ele é contado, não usado.

      Evento fora de `alvos` é ignorado. O sub-NDARAC904DMU carrega 21
      eventos de seqLearning dentro do arquivo de RestingState; nenhum deles
      pode abrir nem fechar bloco de repouso.

      `cortes` (boundary, break cnt) recortam o bloco em segmentos
      contínuos que PRESERVAM o rótulo. A duração declarada é a do bloco
      inteiro, medida pelo próximo onset, nunca fixada em 20 ou 40 s.

    A saída já é a lista de segmentos que `epocar_janela_fixa` recebe, e o
    chamador filtra por valor antes de epocar cada condição."""
    if not fs or fs <= 0:
        raise ValueError(f"fs tem de ser positiva, recebi {fs!r}")
    alvo = set(alvos)
    marcos = []
    for ev in eventos or ():
        if ev.get("valor") not in alvo:
            continue
        amostra = int(round(float(ev.get("onset", 0.0)) * fs))
        if 0 <= amostra < int(n_amostras):
            marcos.append((amostra, ev.get("valor")))
    marcos.sort(key=lambda m: m[0])

    validos = sorted({int(c) for c in (cortes or ()) if 0 < int(c) < int(n_amostras)})

    blocos = []
    n_blocos = {}
    duracoes = {}
    recortados = 0
    for (ini, valor), (fim, _) in zip(marcos, marcos[1:]):
        if fim <= ini:
            continue
        bordas = [ini] + [c for c in validos if ini < c < fim] + [fim]
        if len(bordas) > 2:
            recortados += 1
        for a, b in zip(bordas, bordas[1:]):
            blocos.append((a, b, valor))
        n_blocos[valor] = n_blocos.get(valor, 0) + 1
        duracoes.setdefault(valor, []).append((fim - ini) / fs)

    decisoes = {
        "alvos": list(alvos),
        "n_blocos_por_valor": n_blocos,
        "duracao_s_por_valor": duracoes,
        "blocos_recortados_por_corte": recortados,
        # o marco que ficou sem par, em segundos: declarado, não inventado
        "ultimo_alvo_sem_fechamento": (marcos[-1][0] / fs) if marcos else None,
        "n_marcos": len(marcos),
    }
    return blocos, decisoes


def segmentos_continuos(n_amostras, cortes):
    """Faixas `[inicio, fim)` que não atravessam nenhum corte.

    Ordena e deduplica porque quem chama junta cortes de DUAS fontes (mudança
    de rótulo e eventos), e elas chegam desordenadas por natureza — exigir
    ordem do chamador seria transferir para ele um cuidado que é daqui.

    Corte fora da faixa é descartado em silêncio: 0 e `n_amostras` não
    separam nada, e um índice fora dos limites é ruído do tradutor, não
    informação."""
    validos = sorted({int(c) for c in (cortes or ()) if 0 < int(c) < n_amostras})
    bordas = [0] + validos + [int(n_amostras)]
    return [(bordas[i], bordas[i + 1]) for i in range(len(bordas) - 1)]


def epocar_janela_fixa(dado, fs, duracao_s, passo_s, segmentos=None):
    """Épocas de comprimento fixo, e as decisões que as produziram.

    `dado` tem forma (n_canais, n_amostras). A saída tem forma
    (n_epocas, n_canais, amostras_por_epoca) — a convenção do MNE, para que
    o resultado atravesse o resto do ecossistema sem transposição.

    Uma janela só é aceita se couber INTEIRA dentro de um segmento. A sobra
    do fim de cada segmento é descartada e contada; ela não vira época pela
    metade, nem é completada com zero. Completar com zero seria inventar
    sinal, e num experimento de classificação o zero é um valor como outro
    qualquer — o classificador aprenderia a reconhecer o preenchimento.

    Parâmetro inválido levanta em vez de devolver lista vazia. Lista vazia
    silenciosa é o modo de falha que este projeto proíbe: quem recebe zero
    épocas não sabe se o sinal era curto, se o passo estava errado, ou se
    alguém passou zero sem perceber."""
    if not fs or fs <= 0:
        raise ValueError(f"fs tem de ser positiva, recebi {fs!r}")
    if not duracao_s or duracao_s <= 0:
        raise ValueError(f"duracao_s tem de ser positiva, recebi {duracao_s!r}")
    if not passo_s or passo_s <= 0:
        raise ValueError(f"passo_s tem de ser positivo, recebi {passo_s!r}")

    dado = np.asarray(dado)
    if dado.ndim != 2:
        raise ValueError(f"dado tem de ser (n_canais, n_amostras), recebi {dado.shape}")

    n_canais, n_amostras = dado.shape
    largura = int(round(duracao_s * fs))
    passo = int(round(passo_s * fs))
    if largura <= 0 or passo <= 0:
        raise ValueError(
            f"duracao_s={duracao_s} e passo_s={passo_s} a {fs} Hz dão janela de "
            f"{largura} e passo de {passo} amostras; ambos precisam ser >= 1"
        )

    if segmentos is None:
        segmentos = segmentos_continuos(n_amostras, [])

    janelas = []
    descartadas = 0
    curtos = 0
    for inicio, fim in segmentos:
        disponivel = fim - inicio
        if disponivel < largura:
            curtos += 1
            descartadas += disponivel
            continue
        # último início que ainda deixa a janela caber inteira no segmento
        ultimo = fim - largura
        pos = inicio
        while pos <= ultimo:
            janelas.append(dado[:, pos:pos + largura])
            pos += passo
        # o que sobrou depois do último início aceito
        n_desta = ((ultimo - inicio) // passo) + 1
        consumido = (n_desta - 1) * passo + largura
        descartadas += disponivel - consumido

    if janelas:
        saida = np.stack(janelas, axis=0)
    else:
        # forma coerente mesmo vazia: quem consome pode olhar .shape sem
        # tratar o caso vazio como especial
        saida = np.empty((0, n_canais, largura), dtype=float)

    decisoes = {
        "fs": float(fs),
        "duracao_s": float(duracao_s),
        "passo_s": float(passo_s),
        "amostras_por_epoca": largura,
        "passo_amostras": passo,
        "sobreposicao": max(0.0, 1.0 - passo / largura),
        "n_segmentos": len(segmentos),
        "n_epocas": int(saida.shape[0]),
        "amostras_descartadas": int(descartadas),
        "segmentos_curtos_demais": curtos,
    }
    return saida, decisoes
