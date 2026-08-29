#!/usr/bin/env python3
"""Painel de progresso: lê o estado REAL do projeto e serve como página web.

    python scripts/painel_progresso.py            # sobe em http://localhost:8002
    python scripts/painel_progresso.py --json     # só imprime o estado, sem servidor

O QUE ESTE SCRIPT É
-------------------
Um coletor mais um servidor estático. Ele não guarda estado próprio, não tem
banco e não escreve nada: cada requisição a /estado relê os arquivos e devolve
o que encontrou. Por isso o painel nunca "fica desatualizado" — no pior caso
ele fica *errado junto com os arquivos*, que é uma falha visível, e não a
falha silenciosa de um cache que ninguém sabe que existe.

DE ONDE VEM CADA NÚMERO
-----------------------
Todo campo do JSON carrega a sua procedência, e é isso que separa este painel
de um slide bonito:

  * `tarefas`        do quadro do vault (10 - Estudos e Gamificação/Tarefas.md)
  * `painel`         do Painel-Estudos.md do vault (XP, nível, streak)
  * `git`            de `git log` no próprio repositório
  * `testes`         contados por `def test_` nos arquivos de teste
  * `arquitetura`    da existência e do tamanho real dos arquivos em disco
  * `medidas`        do relatorios/figuras/medidas.json, gerado por figuras_limpeza.py
  * `qc`             do relatorios/qc_relatorio.txt, gerado por qc_relatorio.py
  * `caderno`        da contagem de páginas do PDF em _Anexos
  * `curriculo`      das fases e blocos do Curriculo.md do vault
  * `grafo`          das arestas declaradas em grafo.py, resolvidas contra as tarefas

Quando uma fonte não existe, o campo vem com `disponivel: false` e um motivo em
texto. NUNCA vem zero: zero é um número, e um número errado no painel é pior
que um vazio declarado. A tela sabe desenhar as duas coisas de forma diferente.

O QUE A TELA FAZ COM ISSO
-------------------------
Desenha um fluxograma de três faixas horizontais, uma por trilha, com o eixo
horizontal em ORDEM DE DEPENDÊNCIA e não em data. Acima delas, a espinha dos
13 blocos do currículo.

Duas fontes de aresta, com estatutos diferentes e ditos como diferentes:

  * ENTRE trilhas, derivadas: o campo `**Trilha:**` do Curriculo.md é
    multivalorado em 7 dos 13 blocos, e cada bloco liga as faixas que nomeia.
  * ENTRE tarefas, declaradas: vivem em grafo.py, cada uma citando o
    arquivo:linha do vault de onde saiu. Foi preciso porque o vault tem UMA
    única dependência legível por máquina em todo o texto (Tarefas.md:54).

Uma versão anterior desta página desenhava o avanço como um registro de EEG.
Era bonito e não respondia "o que destrava o quê", que é a pergunta que se faz
olhando de longe.
"""
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grafo

RAIZ_REPO = Path(__file__).resolve().parent.parent.parent
PASTA_SCRIPTS = Path(__file__).resolve().parent
PORTA_PADRAO = 8002

# O vault não vive no repositório e não pode ser hard-coded no meio do módulo:
# entra por variável de ambiente, com o valor lido num lugar só, que é a
# mesma regra que config.py aplica ao caminho do dado.
RAIZ_VAULT = Path(os.environ.get("EEG_VAULT", r"C:\Obsidian\MESTRADO_ITA"))
PASTA_JOGO = RAIZ_VAULT / "10 - Estudos e Gamificação"
PASTA_CADERNO = RAIZ_VAULT / "_Anexos" / "limpeza-car-notch"

# Cor e ordem das trilhas. Fixadas aqui, e não no HTML, porque o mesmo mapa
# alimenta o desenho e o resumo — dois lugares divergiriam na primeira vez que
# alguém acrescentasse uma trilha.
TRILHAS = [
    {"chave": "Dissertação", "cor": "#c9a227"},
    {"chave": "App EEG", "cor": "#4ea1d3"},
    {"chave": "Transformer", "cor": "#7ec07a"},
    {"chave": "Correções", "cor": "#b98ad6"},
]

# Um nó da arquitetura por etapa real do pipeline, na ordem em que o dado
# atravessa. `produz` é o artefato que prova que a etapa rodou: é a diferença
# entre "o script existe" e "o script foi executado".
ARQUITETURA = [
    {"id": "config", "titulo": "onde o dado mora", "arquivo": "backend/config.py",
     "funcoes": ["caminho_release", "descrever_datasets"], "produz": None,
     "papel": "caminho do dado num ponto único, lido de variável de ambiente"},
    {"id": "inventario", "titulo": "o que existe no release", "arquivo": "backend/scripts/inventario_hbn.py",
     "funcoes": ["inventariar"], "produz": "relatorios/inventario_hbn.csv",
     "papel": "uma linha por sujeito e tarefa do release BIDS"},
    {"id": "fenotipo", "titulo": "quem tem rótulo", "arquivo": "backend/scripts/fenotipo_hbn.py",
     "funcoes": ["fenotipo"], "produz": "relatorios/fenotipo_hbn.txt",
     "papel": "idade, gênero e as 4 dimensões de psicopatologia, com os faltantes contados"},
    {"id": "preproc", "titulo": "as quatro etapas", "arquivo": "backend/scripts/preproc_basico.py",
     "funcoes": ["detectar_frequencia_rede", "aplicar_passa_alta", "aplicar_passa_baixa",
                 "aplicar_notch", "aplicar_car", "preprocessar"], "produz": None,
     "papel": "passa-alta, passa-baixa, notch com a rede medida do espectro, e CAR"},
    {"id": "qc", "titulo": "mede antes e depois", "arquivo": "backend/scripts/qc_relatorio.py",
     "funcoes": ["metricas_de_raw", "comparar_antes_depois", "relatorio_qc"],
     "produz": "relatorios/qc_relatorio.txt",
     "papel": "três métricas com veredito, e o veredito reprova de verdade"},
    {"id": "eventos", "titulo": "os marcadores do protocolo", "arquivo": "backend/scripts/eventos.py",
     "funcoes": ["detectar_eventos"], "produz": None,
     "papel": "lê o events.tsv do release e sinaliza divergência no dicionário"},
    {"id": "figuras", "titulo": "as figuras do caderno", "arquivo": "backend/scripts/figuras_limpeza.py",
     "funcoes": ["escolher_canais", "grade_2x2", "cascata_de_etapas", "gerar_figuras"],
     "produz": None,
     "papel": "grade 2x2 na mesma escala e a cascata etapa a etapa"},
    {"id": "api", "titulo": "o backend do app", "arquivo": "backend/app.py",
     "funcoes": ["/subjects", "/raw-data", "/datasets", "/dataset-config", "/eventos"],
     "produz": None, "papel": "serve sinal bruto e filtrado, e a configuração medida do banco"},
    {"id": "app", "titulo": "a tela", "arquivo": "eeg-cerebro-3d.html",
     "funcoes": ["desenharTracosClinico", "marcadoresDaTarefa", "abrirWizard"],
     "produz": None, "papel": "wizard, traçado clínico com marcadores e o cérebro 3D"},
]


# ---------------------------------------------------------------------------
# leitura do vault
# ---------------------------------------------------------------------------

# `- [x] descrição · estágio · tamanho, 25 XP — *nota*`
LINHA_TAREFA = re.compile(r"^- \[( |x|X)\] (.+)$")
XP_DA_TAREFA = re.compile(r"·\s*(pequena|média|media|grande)\s*,\s*(\d+)\s*XP")
ESTAGIO_DA_TAREFA = re.compile(r"·\s*(teoria|método|metodo|implementação|implementacao|"
                               r"experimento|escrita|teoria/implementação|teoria/implementacao)\s*·")
# O nome do bloqueador, que o parser antigo descartava ao guardar só um bool.
BLOQUEADA_POR = re.compile(r"bloqueada por:\s*([^·\n]+)")
# "destravada em 27/08" — escrito dentro da nota de execução, em prosa.
DESTRAVADA_EM = re.compile(r"destravad[ao] em (\d{2}/\d{2})")



def _texto(caminho):
    """Devolve o conteúdo, ou None se o arquivo não existe.

    Sem exceção e sem string vazia: os dois se confundem com "arquivo vazio",
    e quem chama precisa distinguir ausência de vazio para escrever o motivo
    certo no painel."""
    try:
        return Path(caminho).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _limpar_markdown(texto):
    """Tira wikilink, negrito, itálico e crase do rótulo que vai para a tela.

    O quadro do vault é escrito para o Obsidian, então a descrição vem cheia de
    marcação. Renderizá-la crua num canvas mostraria os asteriscos."""
    texto = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", texto)   # [[alvo|rótulo]]
    texto = re.sub(r"\[\[([^\]]+)\]\]", r"\1", texto)
    texto = re.sub(r"`([^`]+)`", r"\1", texto)
    texto = re.sub(r"\*\*([^*]+)\*\*", r"\1", texto)
    texto = re.sub(r"\*([^*]+)\*", r"\1", texto)
    return texto.strip()


def _trilha_da_secao(titulo):
    """Casa o cabeçalho `## ...` do quadro com uma das trilhas conhecidas.

    Casamento por prefixo e não por igualdade: no vault a seção se chama
    'Transformer (Proposta v2)' e 'Correções pendentes no material', e exigir
    o nome exato faria a trilha sumir do painel na primeira vez que alguém
    reescrevesse o cabeçalho."""
    limpo = titulo.strip().lower()
    for t in TRILHAS:
        if limpo.startswith(t["chave"].lower()):
            return t["chave"]
    return None


def ler_tarefas():
    """O quadro de tarefas do vault, tarefa a tarefa, com trilha e XP.

    Devolve um dicionário com `disponivel`, `motivo` e a lista. A lista traz,
    por tarefa: trilha, feita, rotulo, xp, tamanho, estagio, bloqueada, nota.

    O XP sai do texto da própria linha e não de uma tabela de tamanhos: o
    quadro é editado à mão entre sessões, e já houve tarefa com XP fora do
    padrão do tamanho. Ler o número escrito é o que respeita o arquivo."""
    caminho = PASTA_JOGO / "Tarefas.md"
    bruto = _texto(caminho)
    if bruto is None:
        return {"disponivel": False,
                "motivo": f"não achei {caminho}. Aponte EEG_VAULT para a raiz do vault.",
                "lista": []}

    lista = []
    trilha = None
    for linha in bruto.split("\n"):
        if linha.startswith("## "):
            trilha = _trilha_da_secao(linha[3:])
            continue
        m = LINHA_TAREFA.match(linha)
        if not m or trilha is None:
            continue

        feita = m.group(1).lower() == "x"
        linha_inteira = m.group(2)

        # Os marcadores de dependência são procurados na linha INTEIRA, antes
        # de a nota ser cortada. Antes o corte vinha primeiro, e por isso o
        # "destravada em 27/08" que mora dentro da nota era invisível ao
        # parser: três tarefas do quadro declaravam destrave e o painel não
        # enxergava nenhuma.
        bloqueada = "bloqueada por:" in linha_inteira
        mbloq = BLOQUEADA_POR.search(linha_inteira)
        mdestrave = DESTRAVADA_EM.search(linha_inteira)

        # a nota de execução vem depois de " — *" e é o que prova o que foi
        # feito; separada aqui para não poluir o rótulo curto do desenho
        corpo = linha_inteira
        nota = ""
        if " — *" in corpo:
            corpo, _, nota = corpo.partition(" — *")
            nota = _limpar_markdown(nota.rstrip("*"))

        mxp = XP_DA_TAREFA.search(corpo)
        mest = ESTAGIO_DA_TAREFA.search(corpo)
        rotulo = _limpar_markdown(corpo.split(" · ")[0])

        lista.append({
            "trilha": trilha,
            "feita": feita,
            "rotulo": rotulo,
            "id": grafo.identificador(rotulo),
            "xp": int(mxp.group(2)) if mxp else None,
            "tamanho": mxp.group(1) if mxp else None,
            "estagio": mest.group(1) if mest else None,
            # `bloqueada` continua bool porque é contrato de teste; o nome do
            # bloqueador, que o parser antigo jogava fora, agora vem junto
            "bloqueada": bloqueada,
            "bloqueada_por": mbloq.group(1).strip() if mbloq else None,
            "destravada_em": mdestrave.group(1) if mdestrave else None,
            "nota": nota,
        })

    return {"disponivel": True, "motivo": "", "arquivo": str(caminho), "lista": lista}


def ler_painel():
    """XP, nível e streak do Painel-Estudos.md.

    Lidos do arquivo, e NÃO recalculados aqui a partir das tarefas. O painel do
    vault é escrito pelo workflow de estudo, que conhece regras que este script
    não conhece (streak, XP de sessão, bônus). Recalcular daria um segundo
    número autoritativo para a mesma coisa, e os dois divergiriam."""
    caminho = PASTA_JOGO / "Painel-Estudos.md"
    bruto = _texto(caminho)
    if bruto is None:
        return {"disponivel": False, "motivo": f"não achei {caminho}"}

    def numero(rotulo):
        """O número da coluna direita da linha cujo rótulo casa.

        `rotulo` entra como trecho de expressão regular, e não como texto
        literal, porque um dos rótulos do painel carrega o número do nível
        seguinte dentro de si e portanto muda a cada subida de nível."""
        m = re.search(r"\|\s*\**\s*" + rotulo + r"\s*\**\s*\|\s*\**\s*(\d+)", bruto)
        return int(m.group(1)) if m else None

    m_at = re.search(r"^atualizado:\s*(\S+)", bruto, re.M)
    return {
        "disponivel": True,
        "arquivo": str(caminho),
        "xp": numero("XP Total"),
        "nivel": numero("Nível"),
        # o rótulo é "Faltam para o nível 5", "... 6", e assim por diante:
        # fixar o número faria a barra de XP zerar exatamente na sessão em
        # que ele subisse de nível, que é a pior hora para o painel falhar
        "falta_proximo": numero(r"Faltam para o nível \d+"),
        "streak": numero("Streak Atual"),
        "streak_recorde": numero("Streak Recorde"),
        "atualizado": m_at.group(1) if m_at else None,
    }


# ---------------------------------------------------------------------------
# o currículo, que é o único grafo que o vault já tinha
# ---------------------------------------------------------------------------

FASE_DO_CURRICULO = re.compile(r"^## Fase (\d+)\s*[—-]\s*(.+)$")
BLOCO_DO_CURRICULO = re.compile(r"^### (\d+)\.\s*(.+)$")
TRILHA_DO_BLOCO = re.compile(r"^- \*\*Trilha:\*\*\s*(.+)$")

# Como o currículo escreve o nome da trilha, e como o resto do projeto
# escreve. Sem esta tradução o campo não casa com TRILHAS e as ligações
# entre faixas somem sem nenhum aviso.
NOME_DE_TRILHA = {
    "app eeg": "App EEG",
    "dissertacao": "Dissertação",
    "dissertação": "Dissertação",
    "transformer": "Transformer",
}


def ler_curriculo():
    """Os 13 blocos do currículo, em 4 fases, com as trilhas que cada um toca.

    É a descoberta que torna o fluxograma possível. O `Tarefas.md` tem uma
    única dependência legível por máquina em todo o vault; o `Curriculo.md`,
    ao contrário, tem estrutura rígida:

        ## Fase 1 — Dados
        ### 2. Pré-processamento de EEG
        - **Trilha:** app EEG, dissertação (item 1 do TODO do Cap. 3)

    e o campo `Trilha:` é MULTIVALORADO em 7 dos 13 blocos. É daí que saem as
    ligações entre as faixas do desenho, derivadas de dado real, sem nada
    declarado à mão.

    O parêntese explicativo é cortado antes de separar por vírgula: sem isso,
    "dissertação (itens 3 e 5 do TODO do Cap. 3 — protocolo OOF por sujeito e
    avaliação inter-sujeitos)" viraria três trilhas inventadas.

    `concluido` sai da tabela `## Currículo` do Painel-Estudos.md. Quando ela
    não existe, todos vêm como não concluídos e o painel diz isso — o que é
    diferente de afirmar que nada foi feito."""
    caminho = RAIZ_VAULT / "10 - Estudos e Gamificação" / "Curriculo.md"
    bruto = _texto(caminho)
    if bruto is None:
        return {"disponivel": False, "motivo": f"não achei {caminho}", "fases": []}

    fases = []
    bloco = None
    for linha in bruto.split("\n"):
        mf = FASE_DO_CURRICULO.match(linha)
        if mf:
            fases.append({"n": int(mf.group(1)), "nome": mf.group(2).strip(), "blocos": []})
            bloco = None
            continue
        mb = BLOCO_DO_CURRICULO.match(linha)
        if mb and fases:
            bloco = {"n": int(mb.group(1)), "titulo": mb.group(2).strip(),
                     "trilhas": [], "concluido": False}
            fases[-1]["blocos"].append(bloco)
            continue
        mt = TRILHA_DO_BLOCO.match(linha)
        if mt and bloco is not None:
            texto = re.sub(r"\(.*", "", mt.group(1), flags=re.S)
            for pedaco in texto.split(","):
                nome = NOME_DE_TRILHA.get(pedaco.strip().lower())
                if nome and nome not in bloco["trilhas"]:
                    bloco["trilhas"].append(nome)

    _marcar_blocos_concluidos(fases)
    return {"disponivel": True, "motivo": "", "arquivo": str(caminho), "fases": fases}


def _marcar_blocos_concluidos(fases):
    """Cruza os blocos com a tabela `## Currículo` do Painel-Estudos.md.

    A tabela lista `| Bloco | Título | Sessão |` só dos blocos já dados. Ler o
    número da primeira coluna é o bastante, e é mais robusto que casar título:
    o título do quadro é abreviado em relação ao do currículo."""
    bruto = _texto(PASTA_JOGO / "Painel-Estudos.md")
    if bruto is None:
        return
    trecho = bruto.split("## Currículo", 1)
    if len(trecho) < 2:
        return
    dados = trecho[1].split("\n## ", 1)[0]
    feitos = {int(m.group(1)) for m in re.finditer(r"^\|\s*(\d+)\s*\|", dados, re.M)}
    for f in fases:
        for b in f["blocos"]:
            b["concluido"] = b["n"] in feitos


# ---------------------------------------------------------------------------
# leitura do repositório
# ---------------------------------------------------------------------------

def ler_git(n=12):
    """Os últimos commits, com data ISO. Devolve lista vazia com motivo se o
    git não estiver disponível — o painel precisa abrir num clone sem git
    tanto quanto num com."""
    try:
        saida = subprocess.run(
            ["git", "log", f"-{n}", "--date=short", "--format=%h\x1f%ad\x1f%s"],
            cwd=RAIZ_REPO, capture_output=True, text=True, encoding="utf-8", timeout=15)
    except (OSError, subprocess.SubprocessError) as e:
        return {"disponivel": False, "motivo": f"git indisponível: {e}", "commits": []}
    if saida.returncode != 0:
        return {"disponivel": False, "motivo": saida.stderr.strip()[:200], "commits": []}

    commits = []
    for linha in saida.stdout.strip().split("\n"):
        if not linha:
            continue
        partes = linha.split("\x1f")
        if len(partes) == 3:
            commits.append({"hash": partes[0], "data": partes[1], "assunto": partes[2]})
    return {"disponivel": True, "motivo": "", "commits": commits}


def contar_testes():
    """Quantos `def test_` existem, por arquivo.

    Contagem estática de propósito: rodar o pytest daria o número verdadeiro
    de testes COLETADOS, mas custaria segundos a cada carga da página e faria
    o painel depender de o ambiente conseguir importar mne. O rótulo na tela
    diz 'definidos', que é exatamente o que este número é."""
    pasta = RAIZ_REPO / "backend" / "tests"
    if not pasta.is_dir():
        return {"disponivel": False, "motivo": f"não achei {pasta}", "total": None, "por_arquivo": []}

    por_arquivo = []
    total = 0
    for arq in sorted(pasta.glob("test_*.py")):
        texto = _texto(arq) or ""
        n = len(re.findall(r"^def test_", texto, re.M))
        total += n
        por_arquivo.append({"arquivo": arq.name, "n": n})
    return {"disponivel": True, "motivo": "", "total": total, "por_arquivo": por_arquivo}


def ler_arquitetura():
    """Um nó por etapa do pipeline, com o estado real de cada uma.

    `estado` tem três valores e a diferença entre eles é o ponto do painel:
      * `ausente`  — o arquivo do script não existe
      * `escrito`  — o script existe mas o artefato que ele produz, não
      * `rodado`   — o script existe e o artefato está em disco, com data

    Um quadro que só mostrasse verde e vermelho esconderia justamente a etapa
    que foi codificada e nunca executada, que é a mais fácil de esquecer."""
    nos = []
    for meta in ARQUITETURA:
        caminho = RAIZ_REPO / meta["arquivo"]
        existe = caminho.is_file()
        linhas = None
        if existe:
            texto = _texto(caminho)
            linhas = texto.count("\n") + 1 if texto else None

        saida = None
        if meta["id"] == "figuras":
            # caso especial declarado, e não silencioso: a saída deste nó mora
            # no vault, fora do repositório, então o caminho não é relativo a
            # RAIZ_REPO como o dos outros
            med = ler_medidas()
            saida = {"caminho": med.get("arquivo", "medidas.json"), "existe": med["disponivel"]}
            if med["disponivel"]:
                p = Path(med["arquivo"])
                saida["bytes"] = p.stat().st_size
                saida["quando"] = datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
        elif meta["produz"]:
            p = RAIZ_REPO / meta["produz"]
            if p.exists():
                saida = {"caminho": meta["produz"], "existe": True,
                         "bytes": p.stat().st_size,
                         "quando": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")}
            else:
                saida = {"caminho": meta["produz"], "existe": False}

        if not existe:
            estado = "ausente"
        elif saida is None or saida["existe"]:
            estado = "rodado" if saida else "escrito"
        else:
            estado = "escrito"

        nos.append({**meta, "existe": existe, "linhas": linhas, "saida": saida, "estado": estado})
    return nos


# O medidas.json é gravado ao lado das figuras, e as figuras do caderno moram
# no vault, não no repositório: quem chama figuras_limpeza.py passa a pasta de
# saída. Procurar nos dois lugares, nesta ordem, evita que o painel diga
# "ainda não geradas" para figuras que existem há dias na outra pasta.
CANDIDATOS_MEDIDAS = [
    PASTA_CADERNO / "figuras" / "medidas.json",
    RAIZ_REPO / "relatorios" / "figuras" / "medidas.json",
]


def ler_medidas():
    """Os números que o caderno cita, direto do medidas.json das figuras."""
    for caminho in CANDIDATOS_MEDIDAS:
        bruto = _texto(caminho)
        if bruto is None:
            continue
        try:
            return {"disponivel": True, "arquivo": str(caminho), "bancos": json.loads(bruto)}
        except json.JSONDecodeError as e:
            return {"disponivel": False, "motivo": f"{caminho} ilegível: {e}"}
    return {"disponivel": False,
            "motivo": "figuras ainda não geradas: rode scripts/figuras_limpeza.py"}


def ler_qc():
    """O veredito do relatório de qualidade, incluindo o que REPROVOU.

    A linha de reprovação é extraída explicitamente porque é a informação que
    um painel de progresso tende a esconder. Aqui ela é campo de primeira
    classe: `reprovadas` sai não-vazia quando alguma métrica falhou."""
    caminho = RAIZ_REPO / "relatorios" / "qc_relatorio.txt"
    bruto = _texto(caminho)
    if bruto is None:
        return {"disponivel": False,
                "motivo": "QC ainda não rodado: rode scripts/qc_relatorio.py"}

    reprovadas = []
    capturando = False
    for linha in bruto.split("\n"):
        if linha.startswith("sujeitos com métrica reprovada"):
            capturando = True
            continue
        if capturando:
            if not linha.strip():
                break
            partes = linha.split()
            if len(partes) >= 2:
                reprovadas.append({"sujeito": partes[0], "metricas": " ".join(partes[1:])})

    linhas_metrica = [l for l in bruto.split("\n") if " OK" in l or "FALHOU" in l]
    return {"disponivel": True, "arquivo": str(caminho),
            "quando": datetime.fromtimestamp(caminho.stat().st_mtime).isoformat(timespec="seconds"),
            "reprovadas": reprovadas, "linhas": linhas_metrica[:12]}


def ler_caderno():
    """Páginas do PDF do caderno de entrega.

    Conta os objetos /Type /Page do PDF em vez de chamar pdfinfo: o painel não
    pode exigir poppler instalado só para mostrar um número."""
    pdf = PASTA_CADERNO / "pdf" / "limpeza-car-notch-teoria.pdf"
    if not pdf.is_file():
        return {"disponivel": False, "motivo": f"não achei {pdf}"}
    try:
        dados = pdf.read_bytes()
    except OSError as e:
        return {"disponivel": False, "motivo": str(e)}

    paginas = len(re.findall(rb"/Type\s*/Page[^s]", dados))
    figuras = len(list((PASTA_CADERNO / "figuras").glob("*.pdf"))) if (PASTA_CADERNO / "figuras").is_dir() else 0
    capitulos = len(list((PASTA_CADERNO / "teoria" / "cap").glob("*.tex"))) if (PASTA_CADERNO / "teoria" / "cap").is_dir() else 0

    refs = _texto(PASTA_CADERNO / "teoria" / "cap" / "07-referencias.tex") or ""
    return {"disponivel": True, "arquivo": str(pdf),
            "quando": datetime.fromtimestamp(pdf.stat().st_mtime).isoformat(timespec="seconds"),
            "paginas": paginas or None, "figuras": figuras, "capitulos": capitulos,
            "referencias": len(re.findall(r"\\bibitem\{", refs)) or None,
            "kb": round(pdf.stat().st_size / 1024)}


def coletar():
    """Todo o estado, num objeto só. É o que /estado devolve."""
    tarefas = ler_tarefas()
    # o grafo é resolvido contra as tarefas desta leitura, não contra uma
    # cópia: renomear uma tarefa no vault tem que aparecer como aresta órfã
    # em `grafo.erros`, e não como aresta apontando para o vazio
    g = grafo.resolver(tarefas["lista"])
    return {
        "curriculo": ler_curriculo(),
        "grafo": g,
        "gerado": datetime.now().isoformat(timespec="seconds"),
        "hoje": date.today().isoformat(),
        "repo": str(RAIZ_REPO),
        "vault": str(RAIZ_VAULT),
        "trilhas": TRILHAS,
        "tarefas": tarefas,
        "painel": ler_painel(),
        "git": ler_git(),
        "testes": contar_testes(),
        "arquitetura": ler_arquitetura(),
        "medidas": ler_medidas(),
        "qc": ler_qc(),
        "caderno": ler_caderno(),
    }


# ---------------------------------------------------------------------------
# servidor
# ---------------------------------------------------------------------------

class Manipulador(BaseHTTPRequestHandler):
    def _responder(self, corpo, tipo, codigo=200):
        dados = corpo if isinstance(corpo, bytes) else corpo.encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(dados)))
        # sem cache: o painel relê o disco a cada carga, e um 304 do navegador
        # anularia exatamente a propriedade que o torna confiável
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(dados)

    def do_GET(self):
        caminho = self.path.split("?")[0]
        if caminho in ("/", "/index.html"):
            html = _texto(PASTA_SCRIPTS / "painel_progresso.html")
            if html is None:
                self._responder("painel_progresso.html não encontrado ao lado do script",
                                "text/plain; charset=utf-8", 500)
            else:
                self._responder(html, "text/html; charset=utf-8")
        elif caminho == "/estado":
            self._responder(json.dumps(coletar(), ensure_ascii=False, default=str),
                            "application/json; charset=utf-8")
        else:
            self._responder("não existe", "text/plain; charset=utf-8", 404)

    # Conexão parada mais que isto é derrubada. Navegador abre conexão
    # especulativa e não manda nada nela; sem timeout, cada uma dessas
    # segura uma thread até a página fechar.
    timeout = 20

    def handle_one_request(self):
        """Igual ao da base, mas tratando o timeout como fim de conexão.

        Sem isto, o socket.timeout de uma conexão ociosa sobe como exceção e
        o http.server imprime um traceback por conexão especulativa do
        navegador — ruído puro, num console que também tem o backend."""
        try:
            super().handle_one_request()
        except TimeoutError:
            self.close_connection = True

    def log_message(self, formato, *args):
        # o log padrão do http.server escreve uma linha por requisição e
        # afoga o console de quem subiu isto junto com o backend
        pass


def servir(porta=PORTA_PADRAO):
    """Sobe o painel e serve até Ctrl+C.

    ThreadingHTTPServer, e NÃO HTTPServer. A diferença não é otimização: o
    HTTPServer atende uma conexão por vez, e todo navegador abre uma segunda
    conexão especulativa que fica aberta sem mandar byte nenhum. O servidor
    bloqueia lendo dessa conexão, e a página carrega uma vez e depois trava
    em qualquer requisição seguinte — inclusive no /estado que ela mesma
    pede. Foi exatamente o que aconteceu na primeira versão deste script.

    `daemon_threads` garante que Ctrl+C encerre de fato, sem esperar as
    conexões ociosas do navegador expirarem uma a uma."""
    servidor = ThreadingHTTPServer(("127.0.0.1", porta), Manipulador)
    servidor.daemon_threads = True

    # flush=True em TODAS estas linhas, e não por elegância. Quando o painel
    # sobe por iniciar.py ele é um subprocesso com stdout em pipe, e o
    # Python passa a bufferizar em blocos de alguns KB. Estas cinco linhas
    # não enchem o buffer, então ficavam presas nele até o processo morrer:
    # na prática, um servidor que subia sem nunca dizer que subiu — e o
    # AVISO do vault ausente, que é a mensagem mais útil das cinco, nunca
    # chegava a quem precisava dela.
    def diga(texto):
        print(texto, flush=True)

    diga(f"[painel] http://localhost:{porta}  (Ctrl+C encerra)")
    diga(f"[painel] repo:  {RAIZ_REPO}")
    diga(f"[painel] vault: {RAIZ_VAULT}")
    if not PASTA_JOGO.is_dir():
        diga(f"[painel] AVISO: não achei {PASTA_JOGO} — o quadro de tarefas vai aparecer vazio.")
        diga("[painel] Aponte a variável EEG_VAULT para a raiz do vault.")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\n[painel] encerrado.")
    finally:
        servidor.server_close()


if __name__ == "__main__":
    if "--json" in sys.argv:
        print(json.dumps(coletar(), ensure_ascii=False, indent=2, default=str))
    else:
        porta = PORTA_PADRAO
        for i, a in enumerate(sys.argv):
            if a == "--porta" and i + 1 < len(sys.argv):
                porta = int(sys.argv[i + 1])
        servir(porta)
