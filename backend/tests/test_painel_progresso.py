"""Testes do coletor do painel de progresso.

O que se está travando aqui é a LEITURA do quadro do vault, que é a parte
frágil: o quadro é um arquivo markdown editado à mão entre sessões, e o
parser precisa sobreviver a reescritas de cabeçalho, a marcação do Obsidian
no meio da descrição e a linhas que não são tarefa.

Não se testa o desenho: ele vive no HTML e é verificado no navegador.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import painel_progresso as pp


# ---------------------------------------------------------------------------
# limpeza de markdown
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entrada,esperado", [
    ("`backend/config.py`: centralizar", "backend/config.py: centralizar"),
    ("**Wizard de setup** do app", "Wizard de setup do app"),
    ("ver [[Fonte - Folha Manuscrita]]", "ver Fonte - Folha Manuscrita"),
    ("ver [[arquivo.pdf|Caderno de limpeza]]", "ver Caderno de limpeza"),
    ("*ênfase* no meio", "ênfase no meio"),
    ("sem marcação nenhuma", "sem marcação nenhuma"),
])
def test_limpar_markdown(entrada, esperado):
    assert pp._limpar_markdown(entrada) == esperado


# ---------------------------------------------------------------------------
# casamento de trilha
# ---------------------------------------------------------------------------

def test_trilha_casa_por_prefixo():
    """O vault chama a seção de 'Transformer (Proposta v2)'. Exigir o nome
    exato faria a trilha sumir do painel na primeira reescrita do cabeçalho."""
    assert pp._trilha_da_secao("Transformer (Proposta v2)") == "Transformer"
    assert pp._trilha_da_secao("Correções pendentes no material") == "Correções"
    assert pp._trilha_da_secao("App EEG") == "App EEG"


def test_trilha_desconhecida_vira_none():
    """Seção que não é trilha não pode virar uma trilha vazia no painel."""
    assert pp._trilha_da_secao("Notas soltas") is None


# ---------------------------------------------------------------------------
# leitura do quadro
# ---------------------------------------------------------------------------

QUADRO = """---
tipo: estado-de-jogo
---

# Tarefas

> Instruções que não são tarefa e não podem ser lidas como tarefa.

## Dissertação

- [x] `Cap2/sec-2-1.tex`: acrescentar subseção · escrita · média, 25 XP — *três parágrafos*
- [ ] Ler **Bastos 2016** antes de citar · escrita · pequena, 10 XP

## App EEG

- [x] Escrever inventário do release · implementação · média, 25 XP
- [ ] Escrever teste de vazamento · implementação · grande, 60 XP · 🔒 bloqueada por: split por sujeito

## Notas soltas

- [ ] Isto não está em trilha nenhuma e deve ser ignorado · escrita · pequena, 10 XP
"""


@pytest.fixture
def vault(tmp_path, monkeypatch):
    pasta = tmp_path / "10 - Estudos e Gamificação"
    pasta.mkdir(parents=True)
    monkeypatch.setattr(pp, "PASTA_JOGO", pasta)
    return pasta


def test_ler_tarefas_extrai_trilha_xp_e_estado(vault):
    (vault / "Tarefas.md").write_text(QUADRO, encoding="utf-8")
    r = pp.ler_tarefas()

    assert r["disponivel"] is True
    assert len(r["lista"]) == 4, "a seção fora das trilhas conhecidas não entra"

    t = r["lista"][0]
    assert t["trilha"] == "Dissertação"
    assert t["feita"] is True
    assert t["xp"] == 25
    assert t["tamanho"] == "média"
    assert t["estagio"] == "escrita"
    assert t["nota"] == "três parágrafos"
    assert "`" not in t["rotulo"], "a crase do Obsidian não pode chegar ao canvas"


def test_ler_tarefas_marca_bloqueada(vault):
    (vault / "Tarefas.md").write_text(QUADRO, encoding="utf-8")
    lista = pp.ler_tarefas()["lista"]
    travada = [t for t in lista if t["xp"] == 60][0]
    assert travada["bloqueada"] is True
    assert travada["feita"] is False


def test_ler_tarefas_sem_arquivo_diz_o_motivo(vault):
    """Ausência de fonte NUNCA vira lista vazia silenciosa: o painel tem que
    conseguir dizer na tela por que não há nada."""
    r = pp.ler_tarefas()
    assert r["disponivel"] is False
    assert r["lista"] == []
    assert "Tarefas.md" in r["motivo"]


def test_ler_tarefas_ignora_linha_antes_da_primeira_trilha(vault):
    """Uma tarefa escrita acima de qualquer `## Trilha` não tem trilha, e
    atribuí-la à primeira seria inventar dado."""
    (vault / "Tarefas.md").write_text(
        "# Tarefas\n\n- [x] órfã · escrita · pequena, 10 XP\n\n## App EEG\n\n"
        "- [x] com trilha · escrita · pequena, 10 XP\n", encoding="utf-8")
    lista = pp.ler_tarefas()["lista"]
    assert [t["rotulo"] for t in lista] == ["com trilha"]


def test_ler_tarefas_sem_xp_devolve_none_e_nao_zero(vault):
    """Zero é um número e seria somado; None é a ausência, e a tela sabe
    desenhar '?' para ela."""
    (vault / "Tarefas.md").write_text(
        "## App EEG\n\n- [ ] tarefa sem sufixo de tamanho\n", encoding="utf-8")
    t = pp.ler_tarefas()["lista"][0]
    assert t["xp"] is None
    assert t["tamanho"] is None


# ---------------------------------------------------------------------------
# painel de gamificação
# ---------------------------------------------------------------------------

PAINEL = """---
atualizado: 2026-08-27
---

| Métrica | Valor |
|---|---|
| **XP Total** | **305** |
| **Nível** | **4** |
| Faltam para o nível 5 | 95 XP |
| **Streak Atual** | 2 |
| **Streak Recorde** | 7 |
"""


def test_ler_painel(vault):
    (vault / "Painel-Estudos.md").write_text(PAINEL, encoding="utf-8")
    p = pp.ler_painel()
    assert (p["xp"], p["nivel"], p["streak"], p["streak_recorde"]) == (305, 4, 2, 7)
    assert p["falta_proximo"] == 95
    assert p["atualizado"] == "2026-08-27"


def test_ler_painel_sem_arquivo(vault):
    p = pp.ler_painel()
    assert p["disponivel"] is False
    assert "Painel-Estudos.md" in p["motivo"]


# ---------------------------------------------------------------------------
# repositório
# ---------------------------------------------------------------------------

def test_contar_testes_conta_este_arquivo():
    r = pp.contar_testes()
    assert r["disponivel"] is True
    nomes = [a["arquivo"] for a in r["por_arquivo"]]
    assert "test_painel_progresso.py" in nomes
    assert r["total"] == sum(a["n"] for a in r["por_arquivo"])


def test_arquitetura_distingue_escrito_de_rodado():
    """Os três estados são o ponto do quadro: um painel binário esconderia
    justamente a etapa codificada e nunca executada."""
    nos = pp.ler_arquitetura()
    assert {n["id"] for n in nos} >= {"config", "preproc", "qc", "app"}
    assert all(n["estado"] in ("ausente", "escrito", "rodado") for n in nos)
    # o próprio preproc_basico.py tem que existir, senão o repositório está quebrado
    preproc = [n for n in nos if n["id"] == "preproc"][0]
    assert preproc["existe"] is True
    assert preproc["linhas"] > 100


def test_coletar_serializa_em_json(vault):
    """O servidor devolve isto por json.dumps: um objeto não serializável
    aqui viraria erro 500 em produção, e não falha de teste."""
    (vault / "Tarefas.md").write_text(QUADRO, encoding="utf-8")
    (vault / "Painel-Estudos.md").write_text(PAINEL, encoding="utf-8")
    bruto = json.dumps(pp.coletar(), ensure_ascii=False, default=str)
    de_volta = json.loads(bruto)
    assert de_volta["tarefas"]["disponivel"] is True
    assert de_volta["painel"]["xp"] == 305
    assert "arquitetura" in de_volta


def test_ler_painel_sobrevive_a_subir_de_nivel(vault):
    """O rótulo do quadro carrega o número do nível seguinte dentro de si.

    Fixar esse número no parser faria a barra de XP zerar exatamente na
    sessão em que o nível sobe, que é a pior hora possível para o painel
    falhar.
    """
    (vault / "Painel-Estudos.md").write_text(
        PAINEL.replace("nível 5", "nível 9").replace("**305**", "**860**"),
        encoding="utf-8")
    p = pp.ler_painel()
    assert p["xp"] == 860
    assert p["falta_proximo"] == 95


# ---------------------------------------------------------------------------
# servidor
# ---------------------------------------------------------------------------

import socket
import threading
import urllib.error
import urllib.request


@pytest.fixture
def servidor_do_painel():
    """Sobe o painel numa porta livre, numa thread, e devolve a URL base.

    Porta 0 deixa o sistema escolher: fixar um número faria o teste falhar
    quando o painel de verdade já estivesse rodando na máquina de quem
    executa a suíte.
    """
    servidor = pp.ThreadingHTTPServer(("127.0.0.1", 0), pp.Manipulador)
    servidor.daemon_threads = True
    t = threading.Thread(target=servidor.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{servidor.server_address[1]}"
    finally:
        servidor.shutdown()
        servidor.server_close()
        t.join(timeout=5)


def test_serve_a_pagina_e_o_estado(servidor_do_painel):
    with urllib.request.urlopen(servidor_do_painel + "/", timeout=10) as r:
        assert r.status == 200
        assert "text/html" in r.headers["Content-Type"]
        # O elemento raiz do mapa. Antes esta assercao citava o
        # <canvas id="registro">, do painel que desenhava o progresso como um
        # registro de EEG; o painel novo e um fluxograma de divs mais um SVG.
        # A INTENCAO nao mudou: garantir que o servidor serve a pagina certa,
        # e nao um 200 com corpo vazio.
        assert b'<div id="mapa">' in r.read()

    with urllib.request.urlopen(servidor_do_painel + "/estado", timeout=20) as r:
        assert r.status == 200
        estado = json.loads(r.read().decode("utf-8"))
    assert "arquitetura" in estado and "tarefas" in estado


def test_caminho_desconhecido_da_404(servidor_do_painel):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(servidor_do_painel + "/nao-existe", timeout=10)
    assert e.value.code == 404


def test_conexao_ociosa_nao_trava_o_servidor(servidor_do_painel):
    """O bug que derrubou o painel na primeira versão.

    Todo navegador abre uma conexão especulativa e não manda byte nenhum
    nela. Com `HTTPServer`, que atende uma conexão por vez, o servidor
    bloqueava lendo dessa conexão: a página carregava uma vez e depois
    travava em tudo, inclusive no `/estado` que ela mesma pede.

    Este teste reproduz a conexão muda e exige que uma requisição de
    verdade, feita em paralelo, ainda responda. Com o servidor
    single-threaded ele estoura o timeout; com threads, passa.
    """
    host, porta = servidor_do_painel.rsplit(":", 1)
    muda = socket.create_connection(("127.0.0.1", int(porta)), timeout=10)
    try:
        # conectada e calada, exatamente como o preconnect do navegador
        with urllib.request.urlopen(servidor_do_painel + "/estado", timeout=25) as r:
            assert r.status == 200
            assert json.loads(r.read().decode("utf-8"))["tarefas"] is not None
    finally:
        muda.close()


def test_estado_nao_e_cacheavel(servidor_do_painel):
    """O painel relê o disco a cada carga; um 304 do navegador anularia
    justamente a propriedade que o torna confiável."""
    with urllib.request.urlopen(servidor_do_painel + "/estado", timeout=20) as r:
        assert r.headers["Cache-Control"] == "no-store"


# ---------------------------------------------------------------------------
# currículo: o único grafo que o vault já tinha
# ---------------------------------------------------------------------------

CURRICULO = """---
tipo: plano
---

# Currículo

## Fase 1 — Dados

### 1. Anatomia do HBN
- **Fontes:** o paper
- **Trilha:** app EEG

### 2. Pré-processamento de EEG
- **Trilha:** app EEG, dissertação (item 1 do TODO do Cap. 3)

## Fase 2 — Representação

### 3. Patches e tokens
- **Trilha:** transformer

### 4. Bloco sem trilha conhecida
- **Trilha:** astrologia, transformer
"""


@pytest.fixture
def vault_curriculo(tmp_path, monkeypatch):
    pasta = tmp_path / "10 - Estudos e Gamificação"
    pasta.mkdir(parents=True)
    monkeypatch.setattr(pp, "RAIZ_VAULT", tmp_path)
    monkeypatch.setattr(pp, "PASTA_JOGO", pasta)
    return pasta


def test_ler_curriculo_estrutura(vault_curriculo):
    (vault_curriculo / "Curriculo.md").write_text(CURRICULO, encoding="utf-8")
    c = pp.ler_curriculo()

    assert c["disponivel"] is True
    assert [f["n"] for f in c["fases"]] == [1, 2]
    assert c["fases"][0]["nome"] == "Dados"
    assert [b["n"] for b in c["fases"][0]["blocos"]] == [1, 2]
    assert c["fases"][1]["blocos"][0]["titulo"] == "Patches e tokens"


def test_ler_curriculo_trilha_multivalorada(vault_curriculo):
    """É daqui que saem as ligações entre as faixas do desenho. Sem separar
    a vírgula, o bloco 2 tocaria uma trilha só e a ligação sumiria."""
    (vault_curriculo / "Curriculo.md").write_text(CURRICULO, encoding="utf-8")
    c = pp.ler_curriculo()
    assert c["fases"][0]["blocos"][1]["trilhas"] == ["App EEG", "Dissertação"]


def test_ler_curriculo_descarta_parentese(vault_curriculo):
    """'dissertação (itens 3 e 5 do TODO do Cap. 3 — protocolo OOF...)' tem
    vírgulas dentro do parêntese. Sem cortá-lo antes, elas virariam trilhas
    inventadas."""
    (vault_curriculo / "Curriculo.md").write_text(CURRICULO, encoding="utf-8")
    trilhas = pp.ler_curriculo()["fases"][0]["blocos"][1]["trilhas"]
    assert all("item" not in t for t in trilhas)


def test_ler_curriculo_ignora_trilha_desconhecida(vault_curriculo):
    (vault_curriculo / "Curriculo.md").write_text(CURRICULO, encoding="utf-8")
    assert pp.ler_curriculo()["fases"][1]["blocos"][1]["trilhas"] == ["Transformer"]


def test_ler_curriculo_marca_concluidos(vault_curriculo):
    (vault_curriculo / "Curriculo.md").write_text(CURRICULO, encoding="utf-8")
    (vault_curriculo / "Painel-Estudos.md").write_text(
        "# x\n\n## Currículo\n\n| Bloco | Título | Sessão |\n|---|---|---|\n"
        "| 1 | Anatomia | 2026-08-26 |\n| 2 | Pré-proc | 2026-08-26 |\n\n## Outra\n",
        encoding="utf-8")
    c = pp.ler_curriculo()
    assert [b["concluido"] for b in c["fases"][0]["blocos"]] == [True, True]
    assert [b["concluido"] for b in c["fases"][1]["blocos"]] == [False, False]


def test_ler_curriculo_sem_arquivo_diz_o_motivo(vault_curriculo):
    c = pp.ler_curriculo()
    assert c["disponivel"] is False
    assert c["fases"] == []
    assert "Curriculo.md" in c["motivo"]


def test_ler_curriculo_sem_tabela_de_concluidos(vault_curriculo):
    """Sem a tabela, todos vêm como não concluídos — que é diferente de
    afirmar que nada foi feito, e é o que a tela mostra."""
    (vault_curriculo / "Curriculo.md").write_text(CURRICULO, encoding="utf-8")
    c = pp.ler_curriculo()
    assert all(not b["concluido"] for f in c["fases"] for b in f["blocos"])


# ---------------------------------------------------------------------------
# os campos novos de ler_tarefas
# ---------------------------------------------------------------------------

def test_ler_tarefas_guarda_o_nome_do_bloqueador(vault):
    """O parser antigo guardava só um bool e jogava fora o nome, então o
    desenho não tinha como ligar as duas tarefas."""
    (vault / "Tarefas.md").write_text(QUADRO, encoding="utf-8")
    t = [x for x in pp.ler_tarefas()["lista"] if x["bloqueada"]][0]
    assert t["bloqueada"] is True
    assert t["bloqueada_por"] == "split por sujeito"


def test_ler_tarefas_enxerga_destrave_dentro_da_nota(vault):
    """Este era o furo: a nota era cortada ANTES da busca por marcador, e as
    três tarefas que declaram 'destravada em 27/08' dentro dela passavam
    batido."""
    (vault / "Tarefas.md").write_text(
        "## App EEG\n\n- [ ] Implementar ICA · implementação · média, 25 XP — "
        "*destravada em 27/08: o passa-alta que ela exigia já está no pipeline*\n",
        encoding="utf-8")
    t = pp.ler_tarefas()["lista"][0]
    assert t["destravada_em"] == "27/08"
    assert "passa-alta" in t["nota"]


def test_ler_tarefas_da_id_unico(vault):
    (vault / "Tarefas.md").write_text(QUADRO, encoding="utf-8")
    lista = pp.ler_tarefas()["lista"]
    ids = [t["id"] for t in lista]
    assert len(set(ids)) == len(ids)
    assert all(ids)
