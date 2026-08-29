# -*- coding: utf-8 -*-
"""Trava o reinicio ao trocar de sujeito e o batimento que alimenta a popup.

O QUE ESTES TESTES PODEM E NAO PODEM FAZER
------------------------------------------
Sao testes de GREP sobre o HTML, e isso e uma limitacao declarada: eles nao
executam o app e nao provam que a tela faz a coisa certa. Quem provou isso
foi a medicao no navegador, e ela esta registrada no plano da sessao.

O que estes testes fazem e outra coisa, e vale por si: impedir que uma
correcao seja DESFEITA em silencio. Os dois defeitos consertados aqui eram
invisiveis — trocar de sujeito deixava o sinal do anterior na tela, e a
popup congelava sem que nada dissesse por que. Se alguem remover uma destas
chamadas, o app volta a parecer funcionar, e e a suite que tem de gritar.

Um teste de grep so vale se recortar o CORPO da funcao, e nao um numero fixo
de caracteres depois dela. A primeira versao dos testes de evento usava 1200
caracteres, e quebrou no dia em que o codigo certo cresceu — acusando uma
regressao que nao existia. Ver `_corpo_da_funcao`.
"""
import re
from pathlib import Path

import pytest

HTML = Path(__file__).resolve().parent.parent.parent / "eeg-cerebro-3d.html"


@pytest.fixture(scope="module")
def html():
    return HTML.read_text(encoding="utf-8")


def _corpo_da_funcao(html, assinatura):
    """O texto entre uma declaracao de funcao e a proxima do arquivo."""
    assert assinatura in html, f"funcao sumiu do arquivo: {assinatura}"
    depois = html.split(assinatura, 1)[1]
    fim = depois.find("\nfunction ")
    return depois if fim == -1 else depois[:fim]


# ---------------------------------------------------------------------------
# Parte 1 — trocar de sujeito zera o que era do anterior
# ---------------------------------------------------------------------------

def test_existe_uma_funcao_unica_de_reinicio(html):
    """A limpeza vive num lugar so. Espalhada em cada chamador, ela diverge:
    era exatamente o que acontecia antes, com o handler da popup zerando o
    `relogioPopup` que os tres chamadores da janela principal nao zeravam."""
    assert "function reiniciarSinal()" in html


@pytest.mark.parametrize("alvo,porque", [
    ("limparHistorico()", "o buffer do sinal"),
    ("reconstruirFiltros()", "a memoria x1/x2/y1/y2 dos 95 biquads e a energia do envelope"),
    ("potenciaBanda[c][b] = 0", "a potencia que colore o cerebro 3D"),
    ("picoAdaptativo = 1", "o pico que normaliza brilho e cor do 3D"),
    ("ultimoMapeamentoClinico = null", "o mapeamento pixel-tempo do desenho anterior"),
    ("relogioPopup = null", "o relogio de exibicao da popup"),
])
def test_o_reinicio_cobre_cada_item_que_sobrevivia(html, alvo, porque):
    """Cada item desta lista foi encontrado AINDA VIVO depois de uma troca de
    sujeito. Nenhum e precaucao teorica."""
    corpo = _corpo_da_funcao(html, "function reiniciarSinal() {")
    assert alvo in corpo, f"o reinicio deixou de zerar {porque}"


@pytest.mark.parametrize("alvo,porque", [
    ("reiniciarSinal()", "zerar o pipeline local"),
    ("publicarLimpeza()", "mandar a popup jogar o buffer dela fora"),
    ("aplicarTempoPausado(null, false)", "soltar um instante congelado que nao existe mais"),
    ("window.clearEEGData()", "parar de servir a gravacao anterior enquanto a nova baixa"),
    ("aguardandoGravacao = true", "impedir o sinal sintetico de encher o buffer durante o download"),
])
def test_trocar_de_sujeito_faz_a_limpeza_completa(html, alvo, porque):
    """`tocarSujeito` montava um `gravacaoAtual` novo e nao limpava NADA.

    Nao era esquecimento obvio: `configurarTaxa` sai no primeiro `if` quando
    a taxa nao muda, entao trocar de sujeito dentro do mesmo banco passava
    por um caminho que parecia limpar e nao limpava."""
    corpo = _corpo_da_funcao(html, "function tocarSujeito(sujeito) {")
    assert alvo in corpo, f"tocarSujeito deixou de {porque}"


def test_a_espera_e_solta_no_sucesso_e_no_erro(html):
    """`aguardandoGravacao` trava `passoSinal`. Ficar presa em true depois de
    uma falha de rede congelaria o app para sempre, e a causa (o download)
    nao apareceria em lugar nenhum da tela."""
    corpo = _corpo_da_funcao(html, "function tocarSujeito(sujeito) {")
    assert corpo.count("aguardandoGravacao = false") >= 2, (
        "a guarda tem de ser solta nos DOIS ramos: chegada e erro"
    )


def test_a_guarda_de_espera_esta_no_passo_de_sinal(html):
    """Sem esta guarda o app gerava SINAL SINTETICO durante os 20 a 30 s de
    download, e a gravacao real chegava depois dele. Medido: a primeira
    amostra desenhada caia em t=343,5 s de uma gravacao de 348,5 s."""
    corpo = _corpo_da_funcao(html, "function passoSinal(dt, lote) {")
    # A EXPRESSAO da guarda, e nao so a palavra.
    #
    # A primeira versao deste teste procurava "aguardandoGravacao" em
    # qualquer lugar do corpo, e por isso NAO pegou uma mutacao de teste que
    # trocava `if (pausado || aguardandoGravacao) return;` de volta por
    # `if (pausado) return;` — o nome continuava aparecendo no comentario
    # logo acima. Um teste que casa com a prosa mede a prosa.
    assert re.search(r"if \(pausado \|\| aguardandoGravacao\) return;", corpo), (
        "a guarda de espera saiu do `if` de passoSinal"
    )
    # tem de sair ANTES de mexer no relogio, senao tGlobal avanca na espera
    # e o `offsetReal = tGlobal` da chegada aterrissa fora do lugar
    guarda = corpo.index("if (pausado || aguardandoGravacao) return;")
    relogio = corpo.index("tGlobal += dt")
    assert guarda < relogio, "a guarda tem de vir antes do avanco de tGlobal"


def test_parar_dado_real_tambem_limpa(html):
    """Parar deixava o buffer cheio do ultimo sujeito: o tracado seguia
    desenhado, agora sem gravacao nenhuma por tras dele."""
    corpo = _corpo_da_funcao(html, "function pararDadoReal() {")
    assert "reiniciarSinal()" in corpo
    assert "publicarLimpeza()" in corpo


# ---------------------------------------------------------------------------
# Parte 2 — a geracao do sinal nao depende do desenho
# ---------------------------------------------------------------------------

def test_gerar_e_desenhar_sao_funcoes_separadas(html):
    """A popup so recebe sinal pela mensagem `amostras`, postada pela janela
    principal. Enquanto isso morava dentro do laco de `requestAnimationFrame`,
    ocultar a principal (que e o que maximizar a popup faz) suspendia o rAF e
    matava o sinal da popup junto."""
    assert "function gerarSinal(agora)" in html
    assert "function iniciarBatimento()" in html


def test_animar_nao_gera_mais_sinal(html):
    """Se `animar` voltar a chamar `gerarSinal`, a taxa do sinal fica atada de
    novo a taxa de desenho — que e exatamente o defeito desfeito aqui."""
    corpo = _corpo_da_funcao(html, "function animar() {")
    assert "gerarSinal(" not in corpo


def test_o_batimento_usa_worker_e_tem_reserva(html):
    """Worker e nao `setInterval` na pagina: o Chrome estrangula temporizador
    de pagina oculta a 1 Hz, e a 1 POR MINUTO depois de 5 minutos. Cinco
    minutos e menos que uma sessao de leitura com a popup maximizada, entao o
    `setInterval` resolveria o comeco do problema e devolveria o congelamento
    no meio dele."""
    corpo = _corpo_da_funcao(html, "function iniciarBatimento() {")
    assert "new Worker(" in corpo
    assert "setInterval(" in corpo, "a reserva para navegador sem Worker sumiu"


def test_o_batimento_e_ligado_no_boot(html):
    """Uma funcao de batimento que ninguem chama e o modo de falha mais
    silencioso possivel: o conserto esta no arquivo e o defeito segue na
    tela.

    Exige a chamada DESCOMENTADA. A primeira versao deste teste procurava
    `iniciarBatimento();` no arquivo inteiro, e por isso nao pegou uma
    mutacao que simplesmente comentou a linha: `// iniciarBatimento();`
    contem a string procurada."""
    chamada = [
        linha for linha in html.split("\n")
        if linha.strip().startswith("iniciarBatimento();")
    ]
    assert chamada, "o batimento existe mas ninguem o liga (chamada ausente ou comentada)"


# ---------------------------------------------------------------------------
# Parte 2 — a popup nao pode mentir sobre a taxa nem sobre o silencio
# ---------------------------------------------------------------------------

def test_a_popup_aplica_a_taxa_que_recebe(html):
    """Este bloco escrevia `info.fs` dentro do texto da barra e mais nada: a
    popup exibia "500Hz" rodando buffers, biquads e ALPHA_EMA na taxa com que
    tinha nascido. O rotulo afirmava uma coisa e a maquina rodava outra."""
    corpo = _corpo_da_funcao(html, "function aplicarInfoGravacaoUI(info) {")
    assert "configurarTaxa(info.fs)" in corpo
    # trocar a taxa realoca o buffer vazio; sem pedir o historico de novo, a
    # popup ficaria esperando encher do zero
    assert "pedir-historico" in corpo


def test_a_popup_avisa_quando_para_de_receber(html):
    """"Congelei porque voce mandou" e "parei de receber" se leem igual na
    tela — tracado imovel — e confundir os dois foi metade do tempo perdido
    diagnosticando esta janela."""
    assert "ultimaChegadaDeAmostras" in html
    corpo = _corpo_da_funcao(html, "function desenharTracosClinico() {")
    assert "sem sinal da janela principal" in corpo


# ---------------------------------------------------------------------------
# O instante congelado tem de existir no buffer
# ---------------------------------------------------------------------------

def test_instante_fora_do_buffer_nao_vira_amostra_da_borda(html):
    """`indiceMaisProximoDoTempo` devolvia o candidato mais proximo por mais
    longe que ele estivesse: pedir um instante fora da janela devolvia a
    amostra da BORDA, e o cerebro 3D pintava aquele momento como se fosse o
    pedido, sem nada na tela dizendo.

    E o modo de falha mais caro que existe aqui, porque o app inteiro se
    sustenta em cerebro, dipolo e tracado descreverem o MESMO instante."""
    corpo = _corpo_da_funcao(html, "function indiceMaisProximoDoTempo(tempoAlvo) {")
    assert "melhorDist > 2 / FS" in corpo, "a guarda de distancia sumiu"
    assert re.search(r"if \(melhorDist > 2 / FS\) return null;", corpo)


def test_a_tela_diz_quando_o_instante_saiu_do_buffer(html):
    """Cair no valor ao vivo sem avisar mostraria o sinal corrente sob um HUD
    que diz "pausado em t=X" — as duas afirmacoes ao mesmo tempo, e nenhuma
    pista de qual acreditar."""
    assert "function atualizarAvisoInstanteForaDoBuffer()" in html
    assert "saiu do buffer" in html
    assert "fora-do-buffer" in html
