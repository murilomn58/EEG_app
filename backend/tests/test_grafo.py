"""Testes do grafo declarado.

O teste que dá razão a este arquivo é `test_toda_aresta_resolve`. As arestas
moram no repositório e as tarefas moram no vault, longe uma da outra: sem uma
verificação automática, renomear um rótulo no `Tarefas.md` faria a seta sumir
do desenho em silêncio, e ninguém saberia por quê.

Com ele, a suíte falha nomeando a declaração órfã.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import grafo
import painel_progresso as pp


@pytest.fixture(scope="module")
def tarefas():
    """As tarefas REAIS do vault. Se o vault não estiver acessível, os testes
    que dependem dele são pulados em vez de falharem: quem roda a suíte num
    clone sem o Obsidian não quebrou nada."""
    r = pp.ler_tarefas()
    if not r["disponivel"] or not r["lista"]:
        pytest.skip("vault indisponível: " + r.get("motivo", ""))
    return r["lista"]


# ---------------------------------------------------------------------------
# identidade
# ---------------------------------------------------------------------------

def test_identificador_e_deterministico():
    a = grafo.identificador("Implementar filtro passa-alta e notch no pipeline")
    b = grafo.identificador("Implementar filtro passa-alta e notch no pipeline")
    assert a == b


def test_identificador_tira_acento_e_pontuacao():
    assert grafo.identificador("Correções: no `Cap2/sec-2-1.tex`!") == "correcoes-cap2-sec-2-1-tex"


def test_identificador_descarta_palavra_vazia():
    """Sem isso o slug viraria 'a-de-do-para-o' e duas tarefas diferentes
    colidiriam num nó só."""
    assert grafo.identificador("A remoção de artefatos do sinal") == "remocao-artefatos-sinal"


def test_identificador_nao_colide_nas_tarefas_reais(tarefas):
    ids = [grafo.identificador(t["rotulo"]) for t in tarefas]
    duplicados = {i for i in ids if ids.count(i) > 1}
    assert not duplicados, f"slugs repetidos viram um nó só no desenho: {duplicados}"


def test_identificador_de_rotulo_vazio():
    assert grafo.identificador("") == "sem-rotulo"


# ---------------------------------------------------------------------------
# resolução das declarações
# ---------------------------------------------------------------------------

def test_toda_aresta_resolve(tarefas):
    """O teste mais importante do arquivo.

    As arestas são declaradas no repositório e as tarefas vivem no vault. Um
    rótulo reescrito lá deixa a declaração órfã, e sem este teste a seta
    simplesmente sumiria do desenho, sem erro e sem aviso."""
    r = grafo.resolver(tarefas)
    assert r["erros"] == [], "declarações órfãs:\n  " + "\n  ".join(r["erros"])
    assert len(r["arestas"]) == len(grafo.ARESTAS)
    assert len(r["artefatos"]) == len(grafo.ARTEFATOS)


def test_toda_aresta_tem_fonte():
    """Fonte vazia é aresta sem procedência, e o painel exibe a citação na
    tela. Uma linha em branco ali passaria por afirmação sem lastro."""
    for a in grafo.ARESTAS:
        assert a.get("fonte", "").strip(), f"aresta sem fonte: {a['de']} -> {a['para']}"


def test_tipo_de_aresta_e_do_vocabulario():
    validos = {"bloqueia", "alimenta", "destrava"}
    for a in grafo.ARESTAS:
        assert a["tipo"] in validos, f"tipo desconhecido: {a['tipo']}"


def test_nenhuma_aresta_aponta_para_si_mesma(tarefas):
    r = grafo.resolver(tarefas)
    for a in r["arestas"]:
        assert a["de"] != a["para"], "aresta de uma tarefa para ela mesma"


def test_portao_tem_texto_e_fonte():
    for p in grafo.PORTOES:
        assert p["texto"].strip() and p["fonte"].strip(), f"portão incompleto: {p['id']}"
        assert p["trilha"] in {"Dissertação", "App EEG", "Transformer"}


def test_artefato_aponta_para_no_de_arquitetura_existente():
    """O mapa liga tarefa -> id de ARQUITETURA. Um id inventado deixaria o
    nó pendurado invisível, sem erro."""
    ids = {n["id"] for n in pp.ARQUITETURA}
    for trecho, no in grafo.ARTEFATOS.items():
        assert no in ids, f"artefato {trecho!r} aponta para nó inexistente: {no}"


# ---------------------------------------------------------------------------
# o casamento por trecho, que é o mecanismo frágil
# ---------------------------------------------------------------------------

FALSAS = [
    {"rotulo": "Implementar split por sujeito (GroupKFold/LOSO) nos dados"},
    {"rotulo": "Escrever teste que PROVA ausência de vazamento"},
    {"rotulo": "Corrigir a acentuação de Pré-processamento"},
]


def test_casar_encontra_por_trecho():
    assert grafo._casar("split por sujeito", FALSAS) == 0


def test_casar_ignora_acento_e_caixa():
    """O vault é editado à mão e a acentuação de um rótulo já mudou entre
    sessões. Casar com acento tornaria a aresta refém disso."""
    assert grafo._casar("PRE-PROCESSAMENTO", FALSAS) == 2


def test_casar_recusa_trecho_ambiguo():
    with pytest.raises(grafo.ArestaOrfa, match="casa com 2 tarefas"):
        grafo._casar("e", [{"rotulo": "teste um"}, {"rotulo": "teste dois"}])


def test_casar_recusa_trecho_ausente():
    with pytest.raises(grafo.ArestaOrfa, match="nenhuma tarefa"):
        grafo._casar("não existe em lugar nenhum", FALSAS)


def test_resolver_nao_levanta_com_declaracao_orfa(monkeypatch):
    """O painel precisa abrir com o vault fora do lugar: grafo parcial mais
    lista de erros é mais útil que página em branco. Quem levanta é o teste,
    não o desenho."""
    monkeypatch.setattr(grafo, "ARESTAS", [
        {"de": "não existe", "para": "também não", "tipo": "bloqueia", "fonte": "x"},
    ])
    monkeypatch.setattr(grafo, "ARTEFATOS", {})
    monkeypatch.setattr(grafo, "PORTOES", [])
    r = grafo.resolver(FALSAS)
    assert r["arestas"] == []
    assert len(r["erros"]) == 1
    assert "não existe" in r["erros"][0]


# ---------------------------------------------------------------------------
# o grafo tem que ser desenhável
# ---------------------------------------------------------------------------

def test_grafo_real_e_aciclico(tarefas):
    """Ciclo no grafo declarado quebraria a atribuição de colunas do desenho.
    O JS detecta e corta, mas o certo é não haver."""
    r = grafo.resolver(tarefas)
    entram = {}
    for a in r["arestas"]:
        entram.setdefault(a["para"], []).append(a["de"])

    estado = {}

    def visitar(i, caminho):
        if estado.get(i) == 2:
            return
        assert estado.get(i) != 1, f"ciclo passando por {tarefas[i]['rotulo'][:50]!r}"
        estado[i] = 1
        for p in entram.get(i, []):
            visitar(p, caminho + [i])
        estado[i] = 2

    for i in range(len(tarefas)):
        visitar(i, [])


def test_aresta_nao_liga_tarefa_feita_a_predecessor_aberto(tarefas):
    """Se A bloqueia B, A feita e B feita é normal; A ABERTA e B FEITA quer
    dizer que a declaração está invertida, ou que o quadro e o grafo
    discordam. Vale saber, mas não vale reprovar: no mundo real a ordem é
    quebrada de propósito às vezes."""
    r = grafo.resolver(tarefas)
    suspeitas = [
        (tarefas[a["de"]]["rotulo"][:40], tarefas[a["para"]]["rotulo"][:40])
        for a in r["arestas"]
        if a["tipo"] == "bloqueia" and not tarefas[a["de"]]["feita"] and tarefas[a["para"]]["feita"]
    ]
    if suspeitas:
        print("\nbloqueios cumpridos fora de ordem (revisar a declaração):")
        for de, para in suspeitas:
            print(f"  {de!r} (aberta) bloqueia {para!r} (feita)")
