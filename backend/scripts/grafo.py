#!/usr/bin/env python3
"""As dependências entre tarefas, declaradas aqui porque o vault não as tem.

POR QUE ESTE ARQUIVO EXISTE
---------------------------
Levantei as dependências em `Tarefas.md`, `Rota.md` e `Entregaveis.md`. Em
todo o vault existe **uma única** dependência legível por máquina:

    Tarefas.md:54 — "· 🔒 bloqueada por: split por sujeito"

Todas as outras estão em prosa, em quatro dialetos diferentes: `🔒 bloqueada
por:` na linha da tarefa, `destrava` no corpo, `destravada em DD/MM` dentro
da nota de execução, e parágrafos narrativos em `Rota.md` e `Entregaveis.md`.
Nenhuma tarefa tem identificador, e o texto do bloqueador ("split por
sujeito") não bate literalmente com o rótulo do alvo ("Implementar split por
sujeito (GroupKFold/LOSO) nos dados carregados pelo app").

Havia duas saídas. Mudar o formato do `Tarefas.md` daria fonte única de
verdade, mas mexeria no que o workflow de estudo escreve toda sessão. A
escolhida foi esta: as arestas moram no repositório e **cada uma cita o
arquivo e a linha do vault de onde saiu**. O vault não muda.

O QUE ISSO CUSTA, DITO SEM ROUPA
--------------------------------
Uma aresta declarada aqui pode divergir do vault sem que nada avise, que é
exatamente o defeito que a citação existe para mitigar e o teste
`test_grafo.py` existe para pegar: ele exige que as duas pontas de toda
aresta ainda casem com uma tarefa real. Renomeie um rótulo no vault e a
suíte falha nomeando a aresta órfã, em vez de a seta sumir do desenho em
silêncio.

Onde a ligação for julgamento meu e não citação, a fonte diz
`atribuição minha`. Não é o mesmo estatuto e não pode parecer o mesmo.

COMO AS TAREFAS SÃO REFERENCIADAS
---------------------------------
Por TRECHO DISTINTIVO do rótulo, não por posição nem por rótulo inteiro.
Posição quebra quando alguém acrescenta uma tarefa; rótulo inteiro quebra
quando alguém corrige uma vírgula. O trecho é o meio-termo que sobrevive a
edição normal e falha alto quando a tarefa deixa de existir.
"""
import re
import unicodedata

# ---------------------------------------------------------------------------
# identidade
# ---------------------------------------------------------------------------

# Palavras que não distinguem tarefa nenhuma e só gastam espaço no slug.
VAZIAS = {
    "a", "à", "ao", "aos", "as", "às", "com", "da", "das", "de", "do", "dos",
    "e", "em", "na", "nas", "no", "nos", "o", "os", "para", "por", "que",
    "um", "uma", "sobre",
}
PALAVRAS_NO_ID = 5


def _sem_acento(texto):
    return "".join(c for c in unicodedata.normalize("NFD", texto)
                   if unicodedata.category(c) != "Mn")


def identificador(rotulo):
    """Slug curto e determinístico a partir do rótulo, para a tela usar como
    chave de nó.

    Não é identidade forte: rótulo editado gera slug diferente. É aceitável
    porque o slug só vive dentro de uma resposta de `/estado` — o desenho
    inteiro é remontado a cada carga. O que precisa sobreviver a edição é a
    ARESTA, e ela é resolvida por trecho, não por slug.

    O teste `test_identificador_nao_colide` trava a única propriedade que
    importa aqui: dois rótulos diferentes não podem gerar o mesmo slug, ou
    duas tarefas viram um nó só no desenho."""
    limpo = _sem_acento(rotulo).lower()
    limpo = re.sub(r"[^a-z0-9\s-]", " ", limpo)
    palavras = [p for p in limpo.split() if p and p not in VAZIAS]
    return "-".join(palavras[:PALAVRAS_NO_ID]) or "sem-rotulo"


# ---------------------------------------------------------------------------
# as arestas
# ---------------------------------------------------------------------------
#
# tipo:
#   bloqueia  o destino NÃO PODE começar antes de a origem terminar
#   alimenta  o destino usa o que a origem produziu, mas poderia começar antes
#   destrava  a origem já terminou E era o que faltava para o destino andar
#
# A distinção não é decorativa: `bloqueia` é afirmação sobre o futuro, e
# errar nela faz o desenho mentir sobre o que dá para atacar agora.

ARESTAS = [
    # --- App EEG: a cadeia do dado -----------------------------------------
    {"de": "script de inventário de release HBN", "para": "download do ds005505",
     "tipo": "bloqueia",
     "fonte": "Rota.md:50 — 'Baixar o ds005505 depende do script de inventário rodar antes (é ele que diz o que pesa)'"},

    {"de": "script de inventário de release HBN", "para": "quais sujeitos do Release 1 têm rótulo",
     "tipo": "alimenta",
     "fonte": "Rota.md:56-57 — 'É a raiz de quase tudo: alimenta o fenótipo, o split e o critério de aptidão'"},

    {"de": "script de inventário de release HBN", "para": "split por sujeito (GroupKFold/LOSO)",
     "tipo": "alimenta", "fonte": "Rota.md:56-57 — idem, 'o split'"},

    {"de": "script de inventário de release HBN", "para": "critérios de seleção automatizados",
     "tipo": "alimenta", "fonte": "Rota.md:56-57 — idem, 'o critério de aptidão'"},

    {"de": "quais sujeitos do Release 1 têm rótulo", "para": "critérios de seleção automatizados",
     "tipo": "alimenta",
     "fonte": "Tarefas.md:48 — 'o fenotipo_hbn.py já é o insumo; falta a decisão automatizada'"},

    {"de": "default do caminho nos dois scripts", "para": "download do ds005505",
     "tipo": "bloqueia",
     "fonte": "atribuição minha — o download grava no caminho que a centralização definiu"},

    # --- App EEG: a cadeia do filtro ---------------------------------------
    {"de": "filtro passa-alta e notch no pipeline", "para": "passa-baixa e CAR",
     "tipo": "bloqueia",
     "fonte": "atribuição minha — as quatro etapas são o mesmo módulo, na ordem da folha"},

    {"de": "filtro passa-alta e notch no pipeline", "para": "Implementar ICA",
     "tipo": "destrava",
     "fonte": "Tarefas.md:51 — 'destravada em 27/08: o passa-alta que ela exigia já está no pipeline'"},

    {"de": "passa-baixa e CAR", "para": "Tornar a referência",
     "tipo": "destrava",
     "fonte": "Tarefas.md:52 — 'O CAR já existe como aplicar_car no backend e como botão Average no frontend; falta unificar'"},

    {"de": "filtro passa-alta e notch no pipeline", "para": "rejeição automática de época",
     "tipo": "destrava",
     "fonte": "Entregaveis.md:26-27 — 'As tarefas de ICA, referência configurável e rejeição de época ficam bloqueadas por este entregável'"},

    {"de": "passa-baixa e CAR", "para": "caderno de entrega do exercício de limpeza",
     "tipo": "bloqueia",
     "fonte": "atribuição minha — o caderno reporta os números que as quatro etapas produziram"},

    {"de": "filtro passa-alta e notch no pipeline", "para": "veredito silencioso do qc_relatorio",
     "tipo": "alimenta",
     "fonte": "atribuição minha — o defeito estava no relatório que mede esse filtro"},

    # --- App EEG: split e prova de vazamento -------------------------------
    {"de": "split por sujeito (GroupKFold/LOSO)", "para": "teste que PROVA ausência de vazamento",
     "tipo": "bloqueia",
     "fonte": "Tarefas.md:54 — '🔒 bloqueada por: split por sujeito' (a única dependência legível por máquina no vault)"},

    # --- App EEG: a tela ----------------------------------------------------
    {"de": "Recuperar a tela", "para": "Marcadores de evento sobre o eixo",
     "tipo": "bloqueia",
     "fonte": "atribuição minha — os marcadores são desenhados na tela recuperada"},

    {"de": "Wizard de setup do app", "para": "Marcadores de evento sobre o eixo",
     "tipo": "bloqueia",
     "fonte": "atribuição minha — a lista editável de eventos é um passo do wizard"},

    {"de": "Reduzir o HBN no backend", "para": "Aposentar o csv_data.py",
     "tipo": "destrava",
     "fonte": "Tarefas.md — a redução em _raw_data_bids é o que tornou o csv_data desnecessário "
              "no caminho BIDS; o destino foi renomeado quando ficou claro que a saída era "
              "contornar o arquivo, e não adaptá-lo"},

    {"de": "Tornar a referência", "para": "Um dono só para o CAR",
     "tipo": "alimenta",
     "fonte": "atribuição minha — foi ao ligar a referência configurável que os dois CARs "
              "ficaram visíveis: o testado (MNE, backend) não é o entregue (laço JS)"},

    {"de": "Recuperar a tela", "para": "pulsação e o travamento do traçado",
     "tipo": "bloqueia",
     "fonte": "atribuição minha — a tela precisava existir com altura utilizável antes de o "
              "defeito de escala ficar visível"},

    {"de": "Painel de progresso em localhost", "para": "mapa de três trilhas",
     "tipo": "bloqueia",
     "fonte": "atribuição minha — o coletor de /estado do painel anterior é o que o mapa consome"},

    # --- entre trilhas ------------------------------------------------------
    {"de": "Criar o projeto", "para": "Reduzir o HBN no backend",
     "tipo": "alimenta",
     "fonte": "Rota.md:21 — 'sem pipeline de dados resolvido no app, treinar aqui produz número que não se sustenta'"},

    {"de": "quais sujeitos do Release 1 têm rótulo", "para": "acrescentar subseção sobre a anatomia do HBN-EEG",
     "tipo": "alimenta",
     "fonte": "Painel-Estudos.md — a viabilidade do TDAH medida (132/136 com attention) é o que a subseção afirma"},

    {"de": "caderno de entrega do exercício de limpeza", "para": "Bibliografia e apontamentos ao código",
     "tipo": "bloqueia",
     "fonte": "atribuição minha — a bibliografia foi acrescentada ao caderno que já existia"},

    # --- Dissertação --------------------------------------------------------
    {"de": "acrescentar subseção sobre a anatomia do HBN-EEG", "para": "justificativa de escolha de tarefa-alvo",
     "tipo": "alimenta",
     "fonte": "Tarefas.md:29 — 'o release já está justificado na 2.1.4, falta a tarefa e o critério'"},

    {"de": "Ler Bastos_2016", "para": "Bibliografia e apontamentos ao código",
     "tipo": "alimenta",
     "fonte": "Tarefas.md:30 — 'antes de citá-la com confiança no Cap. 2'"},

    {"de": "Conferir a contagem de sujeitos", "para": "Bibliografia e apontamentos ao código",
     "tipo": "alimenta",
     "fonte": "atribuição minha — a procedência dos números entrou junto com as referências"},
]


# ---------------------------------------------------------------------------
# os portões: bloqueadores que não são tarefa
# ---------------------------------------------------------------------------
#
# Sem eles o desenho afirmaria que só falta código, e é falso: três dos
# bloqueios mais pesados do mestrado não se resolvem escrevendo nada.

PORTOES = [
    {"id": "portao-linha-pesquisa", "trilha": "Dissertação",
     "titulo": "Linha de pesquisa em disputa",
     "texto": "EEG-signal (o rascunho atual) contra LLM sobre linguagem espontânea (a Proposta v2). "
              "Pendente com a orientadora. Muda o escopo inteiro.",
     "fonte": "Rota.md:38-40",
     "trava": ["justificativa de escolha de tarefa-alvo"]},

    {"id": "portao-metodologia", "trilha": "Dissertação",
     "titulo": "Metodologia não estabelecida",
     "texto": "Capítulos 3, 4 e 5 estão fechados para escrita por decisão, até a metodologia existir. "
              "Blocos que tocam esses capítulos ainda geram teoria e tarefa de código, só não geram "
              "tarefa de escrita.",
     "fonte": "Rota.md:41-42 e Curriculo.md:154-156",
     "trava": []},

    {"id": "portao-dua", "trilha": "App EEG",
     "titulo": "DUA do HBN não submetido",
     "texto": "Sem o Data Use Agreement não há diagnóstico clínico, só as quatro dimensões contínuas "
              "do fenótipo aberto. É o que impede transformar `attention` em rótulo binário com "
              "lastro clínico.",
     "fonte": "Rota.md:43-44",
     "trava": ["critérios de seleção automatizados"]},
]


# ---------------------------------------------------------------------------
# o que cada tarefa construiu
# ---------------------------------------------------------------------------
#
# Mapa trecho-da-tarefa -> id do nó de ARQUITETURA em painel_progresso.py.
# É isto que pendura o arquivo produzido embaixo da tarefa que o produziu, e
# que faz o desenho responder "as tarefas vão construindo o quê".

ARTEFATOS = {
    "default do caminho nos dois scripts": "config",
    "script de inventário de release HBN": "inventario",
    "quais sujeitos do Release 1 têm rótulo": "fenotipo",
    "filtro passa-alta e notch no pipeline": "preproc",
    "veredito silencioso do qc_relatorio": "qc",
    "Marcadores de evento sobre o eixo": "eventos",
    "caderno de entrega do exercício de limpeza": "figuras",
    "Wizard de setup do app": "api",
    "Recuperar a tela": "app",
}


# ---------------------------------------------------------------------------
# resolução
# ---------------------------------------------------------------------------

class ArestaOrfa(Exception):
    """Uma ponta de aresta que não casa com nenhuma tarefa, ou que casa com
    mais de uma.

    Levantada e NÃO engolida de propósito. Silenciar aqui faria a seta
    desaparecer do desenho sem ninguém notar, que é o pior modo de falha
    possível para um grafo declarado à mão longe da sua fonte."""


def _casar(trecho, tarefas):
    """O índice da única tarefa cujo rótulo contém `trecho`.

    Comparação sem acento e sem caixa, porque o vault é editado à mão e a
    acentuação de um rótulo já mudou entre sessões."""
    alvo = _sem_acento(trecho).lower()
    achados = [i for i, t in enumerate(tarefas)
               if alvo in _sem_acento(t["rotulo"]).lower()]
    if len(achados) == 1:
        return achados[0]
    if not achados:
        raise ArestaOrfa(f"nenhuma tarefa contém {trecho!r} — ela foi renomeada ou removida do vault")
    rotulos = [tarefas[i]["rotulo"][:60] for i in achados]
    raise ArestaOrfa(f"{trecho!r} casa com {len(achados)} tarefas, precisa ser mais específico: {rotulos}")


def resolver(tarefas):
    """Traduz as declarações acima para índices na lista de tarefas.

    Devolve `{arestas, portoes, artefatos, erros}`. Os três primeiros já vêm
    com índice resolvido, prontos para o desenho. `erros` traz as declarações
    órfãs em texto.

    NÃO levanta: o painel precisa abrir mesmo com o vault fora do lugar, e um
    grafo parcial com a lista de erros na tela é mais útil que uma página em
    branco. Quem levanta é o teste, que chama `resolver` e exige `erros` vazio
    — assim o desenho degrada e a suíte não."""
    erros = []

    arestas = []
    for a in ARESTAS:
        try:
            arestas.append({
                "de": _casar(a["de"], tarefas),
                "para": _casar(a["para"], tarefas),
                "tipo": a["tipo"],
                "fonte": a["fonte"],
            })
        except ArestaOrfa as e:
            erros.append(f"aresta {a['de']!r} -> {a['para']!r}: {e}")

    portoes = []
    for p in PORTOES:
        trava = []
        for trecho in p["trava"]:
            try:
                trava.append(_casar(trecho, tarefas))
            except ArestaOrfa as e:
                erros.append(f"portão {p['id']}: {e}")
        portoes.append({**p, "trava": trava})

    artefatos = {}
    for trecho, no in ARTEFATOS.items():
        try:
            artefatos[_casar(trecho, tarefas)] = no
        except ArestaOrfa as e:
            erros.append(f"artefato {no!r}: {e}")

    return {"arestas": arestas, "portoes": portoes, "artefatos": artefatos, "erros": erros}


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import painel_progresso

    lista = painel_progresso.ler_tarefas()["lista"]
    r = resolver(lista)
    print(f"{len(lista)} tarefas · {len(r['arestas'])}/{len(ARESTAS)} arestas resolvidas · "
          f"{len(r['artefatos'])}/{len(ARTEFATOS)} artefatos")
    for e in r["erros"]:
        print("  ERRO:", e)
    if "--json" in sys.argv:
        print(json.dumps(r, ensure_ascii=False, indent=2))
