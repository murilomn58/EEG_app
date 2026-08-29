"""Traduz o conjunto de canais de análise para os nomes de um banco.

Existe porque bancos diferentes chamam o mesmo eletrodo por nomes
diferentes: o adhdata já nomeia no padrão 10-20, e a malha EGI do HBN chama
o mesmo Fz de E11. Sem esta camada, cada rota que toca dado teria de saber
de que banco veio o que está lendo.

A estratégia de duas vias é a mesma que figuras_limpeza.escolher_canais já
usava para escolher 3 eletrodos de linha média; aqui ela serve os 19.

A TERCEIRA VIA, E POR QUE ELA PRECISOU EXISTIR
----------------------------------------------
Duas vias bastavam enquanto os bancos eram dois. Um terceiro banco baixado
de fora mostrou o buraco: o EEG Motor Movement/Imagery do PhysioNet
(eegmmidb) nomeia os mesmos eletrodos 10-10 como `Cz..`, `Fc5.` e `C3..` —
com ponto de preenchimento e caixa própria, herdados do cabeçalho EDF, que
tem campo de tamanho fixo.

Medido sobre a gravação S001R03: das duas vias antigas, `resolver` casava
ZERO dos 19 canais de análise, e `eletrodos.posicoes` devolvia ZERO
posições. O efeito na tela não é uma mensagem de erro: é a cabeça de
conferência desenhada VAZIA, sem um eletrodo sequer, e sem dizer que a causa
foi um ponto no fim do nome.

A normalização é deliberadamente conservadora: tira espaço e ponto das
bordas e compara sem caixa. Ela NÃO tenta adivinhar equivalência entre
nomes diferentes (T3 e T7, por exemplo, que são o mesmo eletrodo em
convenções distintas do 10-20). Isso é tradução, não normalização de
grafia, e tradução tem de ser declarada em `mapa_canais` com procedência —
inferir aqui faria o app afirmar equivalência que ninguém conferiu.
"""


def normalizar(nome):
    """A forma canônica de um nome de canal, para comparação apenas.

    O valor devolvido NUNCA sai desta camada para o resto do app: quem
    responde é sempre o nome original do arquivo, porque é ele que o MNE
    aceita em `pick_channels`. Confundir os dois faria a leitura falhar
    depois, longe daqui, com o nome já reescrito."""
    return str(nome).strip().strip(".").upper()


def resolver_detalhado(nomes_no_arquivo, canais_analise, mapa=None):
    """(resolvido, faltantes, por_grafia) para um conjunto de canais.

    Três vias, nesta ordem de autoridade:

      1. o arquivo já traz o canal com o nome 10-20, exatamente: usa direto;
      2. o mapa do fabricante traduz o nome 10-20 para o do arquivo;
      3. o arquivo traz o nome a menos de grafia (caixa, espaço, ponto de
         preenchimento do EDF): casa por forma normalizada.

    A ordem importa duas vezes. O nome exato vence o mapa, para que um banco
    que nomeia corretamente não seja reescrito por uma tabela feita para
    outra malha. E o mapa vence a grafia, porque o mapa é declaração de
    fonte publicada e a grafia é palpite tipográfico: onde os dois
    discordarem, quem manda é quem tem procedência.

    Quando a via 3 é a que casa, o nome entra em `por_grafia` — a tela
    precisa poder dizer que aquele canal veio por aproximação de escrita, e
    não por identidade.

    `resolvido` mapeia nome-de-análise para nome-no-arquivo, na ordem de
    `canais_analise`. `faltantes` lista os que nenhuma das três vias
    alcançou — quem chama decide se isso é bloqueio ou aviso, porque a
    resposta depende de para que os canais seriam usados."""
    disponiveis = set(nomes_no_arquivo)
    mapa = mapa or {}

    # O primeiro nome vence em caso de empate na forma normalizada. Empate
    # é raro e é sinal de arquivo estranho ("Cz" e "cz" no mesmo cabeçalho);
    # escolher em silêncio o último seria escolher pela ordem de iteração.
    por_forma = {}
    for nome in nomes_no_arquivo:
        por_forma.setdefault(normalizar(nome), nome)

    resolvido, faltantes, por_grafia = {}, [], []
    for alvo in canais_analise:
        if alvo in disponiveis:
            resolvido[alvo] = alvo
        elif mapa.get(alvo) in disponiveis:
            resolvido[alvo] = mapa[alvo]
        elif normalizar(alvo) in por_forma:
            resolvido[alvo] = por_forma[normalizar(alvo)]
            por_grafia.append(alvo)
        else:
            faltantes.append(alvo)

    return resolvido, faltantes, por_grafia


def resolver(nomes_no_arquivo, canais_analise, mapa=None):
    """Como `resolver_detalhado`, sem a lista de casados por grafia.

    Existe para não mudar a assinatura que três rotas do app já consomem.
    É um invólucro e não uma segunda implementação de propósito: duas
    cópias da regra de resolução é como elas divergem.

    Note que o detalhe é DEVOLVIDO e não guardado em atributo de módulo. O
    backend atende em várias threads, e um `ultimo_por_grafia` global faria
    a resposta de uma requisição descrever a resolução de outra — erro que
    só aparece sob carga, exatamente quando ninguém está olhando."""
    resolvido, faltantes, _ = resolver_detalhado(nomes_no_arquivo, canais_analise, mapa)
    return resolvido, faltantes
