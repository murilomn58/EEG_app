"""
Gera as figuras do exercício de limpeza de sinal: a grade 2x2 com o sinal
sem CAR e com CAR, no tempo e em frequência, e as figuras auxiliares que
mostram o que cada etapa do filtro removeu.

A grade 2x2 é o pedido literal da folha manuscrita:

        |  domínio do tempo  |  |FFT|
  ------+--------------------+-------------
  None  |  C1 C2 C3          |  C1 C2 C3
  CAR   |  C1 C2 C3          |  C1 C2 C3

com a exigência, também literal, de "mesma escala, mesmo t, mesma
atividade". Por isso os limites dos eixos são calculados UMA vez, sobre
os dois conjuntos, e aplicados aos quatro painéis. Sem isso a comparação
mente: o CAR reduz a amplitude, o matplotlib reescala sozinho, e as duas
linhas do painel parecem ter a mesma energia.

CANAIS: a folha pede 2 a 3 canais de índice z, os da linha média. Eles
não têm o mesmo nome nos dois bancos:

  adhdata (19 canais, 10-20)      Fz, Cz, Pz  --  NÃO tem Oz
  HBN (GSN-HydroCel-129, EGI)     E11=Fz, Cz, E62=Pz, E75=Oz

A correspondência do HBN vem do MAPA OFICIAL do fabricante (EGI, doc.
8403486-52), nao de inferencia. Inferir pela geometria foi tentado e
falha: o eletrodo mais proximo do Fz padrao e o E6, nao o E11, porque a
touca EGI assenta diferente. A geometria fica como conferencia, em
conferir_mapa, que confirma que os tres escolhidos estao sobre o plano
sagital.

FORMATO: PDF vetorial, para o \\includegraphics do caderno LaTeX. Os
cadernos existentes desenham tudo em TikZ/pgfplots, e diagrama conceitual
continua sendo desenhado assim; mas um traço de EEG real tem milhares de
pontos por canal, o que inviabiliza pgfplots na prática.

Uso: cd backend && python scripts/figuras_limpeza.py <arquivo> [subject_id]
Saída: as figuras .pdf na pasta pedida, uma por assunto
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sem display: isto roda em terminal
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FuncFormatter, MultipleLocator, ScalarFormatter
import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preproc_basico
from preproc_basico import _psd_medio, razao_pico_vizinhanca_db

# Os eletrodos de linha média que a folha pede, em nomenclatura 10-20.
# A busca é POSICIONAL e não por nome: numa malha EGI eles se chamam E11,
# E62, E75, e nomear à mão seria escrever a resposta em vez de medi-la.
ALVOS_LINHA_MEDIA = ["Fz", "Cz", "Pz", "Oz"]

# Meia-largura da faixa sagital, em metros. 6 mm é folgado o bastante para
# aceitar o desvio de fabricação da touca e estreito o bastante para não
# pegar o anel vizinho.
TOLERANCIA_SAGITAL = 0.006

JANELA_S = 4.0   # quanto do sinal entra no painel de tempo
INICIO_S = 20.0  # pula o começo da gravação, que costuma ter transitório

# Paleta CATEGÓRICA. Escala sequencial (viridis, plasma) implicaria uma
# ordem entre Fz, Pz e Oz que não existe, e faz dois canais saírem com
# cores quase iguais. Os traços também variam de estilo, para o caderno
# continuar legível impresso em preto e branco.
CORES = ["#1b4e79", "#c1553b", "#4c7a34", "#6b4c9a", "#8a8a8a"]
TRACOS = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

# Faixa dinâmica fixa do eixo de dB. Sem isto o piso vira o mínimo
# absoluto do periodograma, que num sinal com offset gigante desce tanto
# que esmaga todo o resto contra o topo.
FAIXA_DB = 80.0

# Piso do eixo log de frequência, em Hz. Precisa ficar ABAIXO do corte do
# passa-alta (0,5 Hz), senão a única faixa que essa etapa ataca fica fora
# do gráfico e ela aparece como se não tivesse feito nada.
F_MIN_LOG = 0.25

# Espessura padrão do traço de dado. Uma constante e não um literal
# repetido: a grade usava 1,0 e a cascata 0,9, sem que nada justificasse a
# diferença, e as duas aparecem lado a lado no mesmo capítulo.
LW_DADO = 0.9
LW_FANTASMA = 0.7   # a curva do estado anterior, por baixo
LW_MARCA = 0.6      # as verticais do corte e da rede

# Fundo dos rótulos que nomeiam as verticais. Eles ficam na base do
# painel, que é o lugar livre depois que o passa-baixa derruba o sinal —
# mas no espectro do sinal BRUTO o ruído chega até lá, e sem um fundo o
# rótulo fica ilegível justamente na figura que o introduz.
FUNDO_ROTULO = dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.75)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.5, "figure.dpi": 150,
    # a grade é referência de leitura, não conteúdo: fica ATRÁS do dado.
    # Sem isto ela cruza por cima dos traços, e a cascata precisava
    # contornar o problema com zorder à mão em cada chamada de plot
    "axes.axisbelow": True,
    # marcas na mesma espessura dos eixos. No padrão elas saem em 0,8
    # contra os 0,5 do eixo, e ficam mais pesadas que a moldura que as
    # ancora
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})


def _milhar(x, _pos=None):
    """Formata o eixo de amplitude com ponto separando milhar.

    O HBN chega a dezenas de milhares de unidades de arquivo, e "40000"
    no eixo destoa do resto do caderno, que usa o separador do siunitx
    configurado no papirolatex.cls."""
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", ".")
    return f"{x:g}"


# Mapa 10-20 da malha EGI, do MAPA OFICIAL DO FABRICANTE: Electrical
# Geodesics, Inc. (2011), "HydroCel Geodesic Sensor Net: 128-Channel Map",
# documento 8403486-52. Não é palpite nem inferência geométrica.
#
# A inferência geométrica foi tentada e NÃO serve: comparando as
# coordenadas da montagem, o eletrodo mais próximo do Fz do sistema 10-20
# é o E6 (y = +41,2 mm) e não o E11 (y = +86,2 mm), porque a touca EGI
# assenta na cabeça de forma diferente do modelo 10-20. Quem resolve é a
# fonte publicada; a geometria fica como conferência, em conferir_mapa.
MAPA_EGI_1020 = {"Fz": "E11", "Cz": "Cz", "Pz": "E62", "Oz": "E75"}


def conferir_mapa(raw, nomes):
    """Confere que os canais escolhidos estão mesmo sobre o plano sagital,
    segundo as coordenadas da montagem. É verificação independente do mapa
    do fabricante: se um dia um release trocar a malha, isto acusa."""
    montagem = raw.get_montage()
    if montagem is None:
        return {"conferido": False, "motivo": "arquivo sem montagem"}
    posicoes = montagem.get_positions()["ch_pos"]
    fora = [n for n in nomes
            if n in posicoes and abs(posicoes[n][0]) >= TOLERANCIA_SAGITAL]
    return {
        "conferido": True,
        "fora_da_linha_media": fora,
        "desvio_max_mm": max((abs(posicoes[n][0]) * 1000 for n in nomes if n in posicoes), default=0.0),
    }


def escolher_canais(raw):
    """(nomes, rótulos) dos eletrodos de linha média pedidos pela folha.

    Duas vias, nesta ordem de autoridade:

      1. o canal já se chama Fz/Cz/Pz/Oz (montagem 10-20): usa direto;
      2. a malha é EGI e os nomes são E*: usa o mapa publicado pelo
         fabricante (MAPA_EGI_1020).

    Cai para os três primeiros canais se nenhuma das duas se aplicar, para
    nunca abortar por causa de um banco novo. Devolve no máximo três, que
    é o que a folha pede, e tira o Cz quando há alternativa: no HBN ele é
    a referência e vem plano, então não ilustra nada no painel de tempo."""
    nomes, rotulos = [], []
    for alvo in ALVOS_LINHA_MEDIA:
        if alvo in raw.ch_names:
            nomes.append(alvo)
            rotulos.append(alvo)
        elif MAPA_EGI_1020.get(alvo) in raw.ch_names:
            nomes.append(MAPA_EGI_1020[alvo])
            rotulos.append(alvo)

    if not nomes:
        primeiros = raw.ch_names[:3]
        return primeiros, list(primeiros)

    if len(nomes) > 3 and "Cz" in rotulos:
        i = rotulos.index("Cz")
        nomes.pop(i)
        rotulos.pop(i)
    return nomes[:3], rotulos[:3]


def _indices_trecho(raw):
    """(i0, i1) do trecho analisado. Extraído para que o painel de tempo e
    o de frequência usem exatamente as MESMAS amostras: a folha exige
    'mesmo t', e antes o espectro usava os primeiros 8 s, que é justamente
    o transitório que o painel de tempo evita."""
    fs = raw.info["sfreq"]
    i0 = int(INICIO_S * fs)
    i1 = i0 + int(JANELA_S * fs)
    if i1 > raw.n_times:
        i0, i1 = 0, min(raw.n_times, int(JANELA_S * fs))
    return i0, i1


def _trecho(raw, nomes):
    """(t, matriz) do trecho analisado.

    `t` em segundos desde o início da gravação, e a matriz com um canal por
    linha, na ordem de `nomes`, já multiplicada por 1e6 — o MNE guarda em
    volts, e todo o resto deste módulo raciocina na escala do arquivo.

    O trecho é o MESMO para todas as chamadas dentro de uma execução (ver
    `_indices_trecho`). Isso não é detalhe de implementação: a folha exige
    "mesmo t", e recortar um trecho novo a cada painel destruiria a
    comparação sem que nada na figura denunciasse."""
    fs = raw.info["sfreq"]
    i0, i1 = _indices_trecho(raw)
    idx = [raw.ch_names.index(n) for n in nomes]
    dados = raw.get_data(picks=idx)[:, i0:i1] * 1e6
    t = np.arange(i0, i1) / fs
    return t, dados


def _espectro(raw, nomes, janela_s=8.0, fmin=0.5, n_janelas=4, escala="db"):
    """(freqs, PSD) dos canais pedidos, por Welch.

    `escala` decide a unidade da saída, e a escolha é do chamador:

      "db"     — 10·log10(PSD), a escala das figuras do caderno de 26/08.
      "linear" — µV²/Hz, a PSD como Welch a devolve, sem logaritmo.

    A escala linear entrou por pedido da orientadora em 01/09/2026 ("deixar
    linear", "trocar dB p/ FFT pura"). O default continua em dB porque as
    figuras do caderno JÁ ENTREGUE são desenhadas nele, com faixa e limites
    de eixo calibrados em dB: trocar o default reescreveria um entregável
    aprovado para atender um pedido que é sobre o que vem depois dele.

    Por que a escolha existe em vez de uma resposta só. O logaritmo comprime
    a dinâmica, e é isso que faz o pico de rede a +43 dB caber no mesmo eixo
    que o alfa — útil para VER o que o notch removeu. Mas potência de banda,
    razão entre bandas e ajuste do componente aperiódico são contas sobre a
    potência, e fazê-las sobre dB é fazê-las sobre o logaritmo dela: a média
    de dB é a média geométrica da potência, não a aritmética. Para a extração
    de features que vem a seguir (missão 2, TBR θ/β), é a linear que vale.

    Welch e não periodograma cru: o periodograma de uma janela só é
    ruidoso e, pior, produzia um número em dB que não era comparável com
    a tabela do caderno, que usa Welch. Dois estimadores diferentes com o
    mesmo rótulo 'dB' convidam o leitor a comparar o que não se compara.

    `fmin` é parâmetro e não constante porque a cascata precisa enxergar
    ABAIXO do corte do passa-alta: com o 0,5 Hz que serve às outras
    figuras, a única faixa que o passa-alta ataca fica fora do espectro, e
    a etapa apareceria como se não tivesse feito nada.

    `n_janelas` controla a resolução: n_fft = fs*n_janelas. Mais janela é
    mais resolução em frequência e menos suavização — a cascata usa uma
    janela longa para separar 0,25 Hz de 0,5 Hz, que é onde o passa-alta
    trabalha."""
    fs = raw.info["sfreq"]
    i0, _ = _indices_trecho(raw)
    n = min(int(fs * janela_s), raw.n_times - i0)
    idx = [raw.ch_names.index(n_) for n_ in nomes]
    dados = raw.get_data(picks=idx, start=i0, stop=i0 + n) * 1e6

    n_fft = min(int(fs * n_janelas), n)
    psd, freqs = mne.time_frequency.psd_array_welch(
        dados, sfreq=fs, fmin=fmin, fmax=fs / 2 * 0.9, n_fft=n_fft, verbose=False
    )
    if escala == "linear":
        return freqs, psd
    if escala != "db":
        raise ValueError(f"escala desconhecida: {escala} (use db ou linear)")
    return freqs, 10 * np.log10(psd + 1e-20)


def _teto_linear(espectros, f, f_lo, f_hi, percentil=99.0):
    """Topo do eixo y em escala LINEAR: percentil da PSD em [f_lo, f_hi].

    Não é o máximo. Em µV²/Hz o pico de rede a +43 dB é ~20 000 vezes o
    alfa: um teto no máximo transforma o alfa numa linha no zero, e a figura
    passa a mostrar só a rede que o notch vai tirar. O percentil deixa o
    pico sair pelo topo do quadro, e `_picos_fora_da_escala` escreve ao
    lado quanto ele valia.

    `f_lo = 2 Hz` e não 1 Hz: em linear o joelho 1/f do bruto abaixo de
    2 Hz ainda domina o percentil e esmaga o alfa; o log da versão em dB
    comprimia isso, e aqui não há log para ajudar."""
    dentro = (f >= f_lo) & (f <= f_hi)
    valores = np.concatenate([e[:, dentro].ravel() for e in espectros])
    teto = float(np.percentile(valores, percentil))
    return teto if teto > 0 else float(valores.max() or 1.0)


def _picos_fora_da_escala(e, f, f_hi, teto, rotulos, maximo=3):
    """As curvas que passam do teto, com o valor e a frequência medidos."""
    dentro = (f >= 0.0) & (f <= f_hi)
    picos = []
    for i, rot in enumerate(rotulos):
        curva = e[i][dentro]
        j = int(np.argmax(curva))
        if curva[j] > teto:
            picos.append({"canal": rot, "f_hz": float(f[dentro][j]),
                          "valor": float(curva[j])})
    picos.sort(key=lambda p: -p["valor"])
    return picos[:maximo]


def _fmt_ciencia(v):
    """1,2×10⁴, em mathtext, para caber num rótulo de 5 pt."""
    if v == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(v))))
    mant = v / 10 ** exp
    return rf"${mant:.1f}".replace(".", "{,}") + rf"\times10^{{{exp}}}$"


def _anotar_fora_da_escala(ax, picos, f_hi, y=0.96):
    """Caixa com até três picos que saíram do quadro, e onde eles estavam.

    Sem ela, um leitor concluiria que o pico de rede não existe no painel em
    que ele é, de longe, o maior valor medido. A caixa vai para o VÃO mais
    largo entre os picos (em fração do eixo x), e não para um canto fixo: no
    adhdata a rede está a 87 % da largura e o canto direito a cobriria; no
    HBN está a 62 % e o centro a cobriria."""
    if not picos:
        return
    marcos = sorted({0.04, 0.96} | {min(0.96, max(0.04, p["f_hz"] / f_hi)) for p in picos})
    vaos = [(b - a, (a + b) / 2) for a, b in zip(marcos, marcos[1:])]
    _, centro = max(vaos)
    # uma casa abaixo de 10 Hz: "0 Hz" para o resíduo em 0,5 Hz seria dizer
    # que sobrou DC depois do passa-alta, o que é falso
    linhas = ["fora da escala:"] + [
        f"{p['f_hz']:.1f} Hz: {_fmt_ciencia(p['valor'])} ({p['canal']})" if p["f_hz"] < 10
        else f"{p['f_hz']:.0f} Hz: {_fmt_ciencia(p['valor'])} ({p['canal']})"
        for p in picos
    ]
    ax.text(centro, y, "\n".join(linhas), transform=ax.transAxes,
            ha="center", va="top", fontsize=5, color="#333", linespacing=1.3,
            bbox=FUNDO_ROTULO, zorder=5)


def _rotulo_psd(escala, unidade):
    if escala == "db":
        return "PSD (dB)"
    curta = "un." if unidade != "µV" else "µV"
    return f"PSD ({curta}²/Hz)"


def _formatador_linear(ax):
    """Notação científica no eixo y com o expoente pequeno e fora do título."""
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 3))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.get_offset_text().set_fontsize(6)


def grade_2x2(raw_sem, raw_com, nomes, rotulos, caminho_pdf, titulo,
              unidade="µV", freq_rede=None, h_freq=None, zoom=False,
              escala="db"):
    """A figura central do exercício. Linhas: sem CAR / com CAR. Colunas:
    tempo / frequência. Os limites de eixo são compartilhados dentro de
    cada coluna, que é o que torna a comparação honesta: o CAR reduz a
    amplitude, e deixar cada painel se reescalar faria as duas linhas
    parecerem ter a mesma energia.

    zoom=True acrescenta um painel ampliado dentro do painel de tempo.
    Serve para o sinal BRUTO de bancos com offset gigante, em que os
    traços viram retas na escala compartilhada. O inset mostra a forma sem
    mentir sobre a escala, que é o que reescalar o painel inteiro faria.
    NÃO serve a banco cujo bruto já tem forma visível: ali ele repete o
    que o painel grande mostra, e repete pior.

    A LEGENDA DOS CANAIS é da figura, e não de um painel: em `lower left`
    ela caía sobre os traços em todas as quatro figuras do capítulo."""
    fig, eixos = plt.subplots(2, 2, figsize=(6.1, 4.6))
    fig.subplots_adjust(hspace=0.38, wspace=0.30, top=0.84)

    t_sem, d_sem = _trecho(raw_sem, nomes)
    t_com, d_com = _trecho(raw_com, nomes)
    # MESMO trecho nos dois painéis. O padrão de _espectro é 8 s, e o
    # painel de tempo mostra JANELA_S = 4 s: a figura afirmava "o mesmo
    # trecho" na legenda do caderno enquanto o espectro descrevia o dobro
    # da duração, a partir do mesmo início.
    f_sem, e_sem = _espectro(raw_sem, nomes, janela_s=JANELA_S, escala=escala)
    f_com, e_com = _espectro(raw_com, nomes, janela_s=JANELA_S, escala=escala)

    # o eixo de frequência precisa passar DA rede, senão a ausência do
    # pico depois do notch cai em cima da borda e não se vê
    f_max = min(f_sem.max(), max(80.0, (freq_rede or 50.0) * 1.6))

    # UMA escala para as duas linhas, calculada sobre os dois conjuntos.
    # Em dB, a faixa fixa abaixo do máximo; em linear, o percentil 99 acima
    # de 2 Hz (ver _teto_linear), com os picos que saem do quadro anotados
    lim_amp = np.abs(np.concatenate([d_sem, d_com])).max() * 1.05
    if escala == "linear":
        teto = _teto_linear([e_sem, e_com], f_sem, 2.0, f_max)
        lim_y = (0.0, teto * 1.10)
    else:
        teto_db = max(e_sem.max(), e_com.max())
        lim_y = (teto_db - FAIXA_DB, teto_db + 5)
    picos_registro = []

    linhas_legenda = []
    for linha, (t, d, f, e, nome_linha) in enumerate([
        (t_sem, d_sem, f_sem, e_sem, "sem CAR"),
        (t_com, d_com, f_com, e_com, "com CAR"),
    ]):
        ax = eixos[linha][0]
        for i, rot in enumerate(rotulos):
            (ln,) = ax.plot(t, d[i], lw=LW_DADO, color=CORES[i],
                            ls=TRACOS[i], label=rot)
            if linha == 0:
                linhas_legenda.append(ln)
        ax.set_ylim(-lim_amp, lim_amp)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylabel(f"{nome_linha}\n({unidade})")
        ax.yaxis.set_major_formatter(FuncFormatter(_milhar))

        if zoom:
            # cada canal MENOS a sua propria media: os offsets sao
            # diferentes entre canais, entao centrar todos num valor so
            # deixaria o painel vazio. Aqui a forma aparece; a escala
            # verdadeira continua sendo a do painel grande.
            ac = d - d.mean(axis=1, keepdims=True)
            # janela curta tambem no TEMPO: em 4 s a oscilacao da rede nao
            # se resolve e o inset vira uma faixa cheia. Em 0,25 s ela
            # aparece como onda, e o leitor VE do que o sinal bruto e feito
            n_zoom = max(8, int(0.25 * (len(t) / (t[-1] - t[0]))))
            # `escala_inset`, e não `escala`: este nome sombreava o parâmetro
            # `escala` da função, e nos bancos com inset (o HBN) a coluna da
            # frequência perdia a anotação e o formatador da escala linear
            escala_inset = np.abs(ac[:, :n_zoom]).max() * 1.15
            fatia = ax.inset_axes([0.40, 0.08, 0.57, 0.34])
            for i in range(len(rotulos)):
                fatia.plot(t[:n_zoom], ac[i, :n_zoom], lw=LW_FANTASMA,
                           color=CORES[i], ls=TRACOS[i])
            fatia.set_ylim(-escala_inset, escala_inset)
            fatia.set_xlim(t[0], t[n_zoom - 1])
            # SEM rótulos de tempo: eram eles que caíam em cima dos do
            # painel grande, e a duração já está dita no título do inset
            fatia.tick_params(labelbottom=False, labelsize=4.5, length=1.5,
                              pad=1)
            fatia.set_facecolor("#fbfbfb")
            for lado in fatia.spines.values():
                lado.set_linewidth(0.4)
                lado.set_color("#999")
            fatia.set_title(
                f"0,25 s, cada canal sem a sua média ({lim_amp / escala_inset:.0f}×)",
                fontsize=4.5, pad=1.5)

        if linha == 1:
            ax.set_xlabel("tempo (s)")
        if linha == 0:
            ax.set_title("domínio do tempo", pad=4)

        ax = eixos[linha][1]
        for i, rot in enumerate(rotulos):
            ax.plot(f, e[i], lw=LW_DADO, color=CORES[i], ls=TRACOS[i])
        # as duas marcas que a legenda do caderno manda o leitor olhar
        if h_freq:
            ax.axvline(h_freq, color="#555", ls="--", lw=LW_MARCA)
        if freq_rede:
            ax.axvline(freq_rede, color="#b03030", ls=":", lw=LW_MARCA)
        ax.set_ylim(*lim_y)
        ax.set_xlim(0, f_max)
        ax.set_ylabel(_rotulo_psd(escala, unidade))
        if escala == "linear":
            _formatador_linear(ax)
            picos = _picos_fora_da_escala(e, f, f_max, lim_y[1], rotulos)
            _anotar_fora_da_escala(ax, picos, f_max)
            picos_registro.append({"linha": nome_linha, "picos": picos})
        if linha == 1:
            ax.set_xlabel("frequência (Hz)")
        if linha == 0:
            ax.set_title("domínio da frequência", pad=4)
        # os rótulos das verticais vão na BASE, onde o espectro já caiu, e
        # não no topo, que é do título da coluna
        piso = lim_y[0]
        if h_freq:
            ax.annotate(f"corte {h_freq:g} Hz", (h_freq, piso), fontsize=5.5,
                        color="#555", ha="right", va="bottom", rotation=90,
                        xytext=(-1.5, 2), textcoords="offset points",
                        bbox=FUNDO_ROTULO)
        if freq_rede:
            ax.annotate(f"rede {freq_rede:g} Hz", (freq_rede, piso),
                        fontsize=5.5, color="#b03030", ha="left", va="bottom",
                        rotation=90, xytext=(1.5, 2), textcoords="offset points",
                        bbox=FUNDO_ROTULO)

    for eixo in eixos.ravel():
        # grade nos DOIS eixos: sem as verticais, localizar 50 ou 60 Hz no
        # painel de frequência depende de seguir o tick com o dedo
        eixo.grid(alpha=0.18, lw=0.4)
        eixo.spines[["top", "right"]].set_visible(False)

    # Legenda ÚNICA, da figura e acima dos painéis. Dentro de um painel ela
    # cobria dado em todas as quatro figuras, e repeti-la nos quatro seria
    # gastar espaço para dizer quatro vezes a mesma coisa.
    fig.legend(handles=linhas_legenda, labels=list(rotulos),
               loc="upper center", bbox_to_anchor=(0.5, 0.925),
               ncol=len(rotulos), frameon=False, fontsize=8,
               handlelength=1.8, columnspacing=1.6)

    fig.suptitle(titulo, fontsize=9, y=0.985)
    fig.savefig(caminho_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    saida = {"lim_amplitude": float(lim_amp), "canais": rotulos, "unidade": unidade,
             "escala": escala, "faixa_y": [float(lim_y[0]), float(lim_y[1])],
             "f_max_hz": float(f_max)}
    if escala == "db":
        saida["faixa_db"] = saida["faixa_y"]   # o nome antigo, para quem o cita
    else:
        saida["picos_fora_da_escala"] = picos_registro
    return saida


def _potencia_faixa_db(raw, nomes, f_lo, f_hi):
    """Potência média da PSD na faixa [f_lo, f_hi), em dB.

    É a métrica do passa-alta. A razão pico/vizinhança da rede, que serve
    às outras etapas, não serve a esta: mede 45,99 -> 45,98 dB, porque o
    passa-alta não toca em 60 Hz. Anotar esse par embaixo da linha do
    passa-alta diria ao leitor que a etapa não fez nada, que é o oposto do
    que ela faz e o oposto do que esta figura existe para mostrar."""
    # os MESMOS parâmetros com que a cascata desenha o espectro: o número
    # anotado na figura tem de descrever a curva que está ao lado dele
    f, e = _espectro(raw, nomes, janela_s=32.0, fmin=0.0, n_janelas=8, escala="db")
    dentro = (f >= f_lo) & (f < f_hi)
    if not dentro.any():
        return None
    linear = 10 ** (e[:, dentro] / 10.0)
    return float(10 * np.log10(linear.mean() + 1e-20))


def cascata_de_etapas(raw, nomes, rotulos, freq_rede, caminho_pdf, titulo,
                      l_freq=0.5, h_freq=45.0, unidade="µV",
                      escala_y="db", escala_x="log", f_min=None, f_max=None,
                      etapas_visiveis=("pa", "pb", "notch", "car")):
    """A cascata, uma etapa por LINHA, cada linha um par antes/depois.

    A versão anterior sobrepunha os cinco estados num eixo só. Com cinco
    curvas que coincidem em quase toda a banda e divergem em trechos
    estreitos, o resultado é ilegível: dá para ver QUE mudou, não O QUE
    mudou nem por obra de qual etapa. Como a legenda da figura prometia
    exatamente isso ("mostrar só o resultado final esconde qual etapa fez
    o quê"), a figura contradizia a própria legenda.

    Agora cada linha isola uma etapa. A coluna da direita de uma linha é a
    da esquerda da seguinte, e é isso que faz da figura uma cascata e não
    quatro comparações soltas.

    TRÊS CANAIS em todos os painéis, e não um. Num canal só, o CAR mede
    0,27 -> 0,32 dB e parece não fazer nada, porque o CAR é uma operação
    ESPACIAL: ele subtrai a média entre canais. O que ele faz só existe na
    relação entre canais, e some quando se olha um canal isolado.

    FANTASMA: o painel da direita repete a curva da esquerda em cinza por
    baixo. Sem isso o leitor compara dois gráficos movendo os olhos e
    memorizando forma, que é justamente o que ele não consegue fazer.

    ESCALA DE dB ÚNICA nos oito painéis, pela mesma razão que a grade 2x2
    documenta: deixar cada painel se autoescalar faria etapas que derrubam
    energia parecerem não ter mexido em nada."""
    etapas = [("bruto", raw)]
    pa = preproc_basico.aplicar_passa_alta(raw, l_freq=l_freq)
    etapas.append((f"+ passa-alta {l_freq:g} Hz", pa))
    pb = preproc_basico.aplicar_passa_baixa(pa, h_freq=h_freq)
    etapas.append((f"+ passa-baixa {h_freq:g} Hz", pb))
    nt, harmonicos = preproc_basico.aplicar_notch(pb, freq_rede)
    etapas.append((f"+ notch {freq_rede:g} Hz" if freq_rede else "+ notch (sem rede)", nt))
    etapas.append(("+ CAR", preproc_basico.aplicar_car(nt)))

    # o eixo vai até o último harmônico notchado, senão o trabalho que o
    # pipeline faz em 120 e 180 Hz fica invisível
    if f_max is None:
        f_max = min(raw.info["sfreq"] / 2 * 0.9, max(harmonicos or [90.0]) * 1.2)
    f_min_plot = F_MIN_LOG if f_min is None else float(f_min)
    # os harmônicos que CABEM na faixa pedida; os de fora vão para a nota
    # da linha do notch, em vez de sumirem sem aviso
    harmonicos_visiveis = [h for h in (harmonicos or []) if f_min_plot <= h <= f_max]
    harmonicos_fora = [h for h in (harmonicos or []) if h not in harmonicos_visiveis]

    # Espectro de cada estado, uma vez só. Três parâmetros, três razões:
    #
    # fmin=0     a faixa que o passa-alta ataca vive abaixo de 0,5 Hz e
    #            ficaria fora do gráfico com o padrão.
    # n_janelas=8  resolução de 0,125 Hz, necessária para separar o que
    #            está abaixo do corte do que está logo acima.
    # janela_s=32  e é aqui que estava o defeito: com os 8 s do padrão,
    #            n_fft == n, ou seja UMA janela, sem sobreposição nem
    #            média. Era um periodograma cru se dizendo Welch, e é a
    #            origem do aspecto trêmulo das curvas. Alongar o trecho,
    #            e não encurtar a FFT, é o que resolve sem sacrificar a
    #            resolução de que a primeira linha depende.
    espectros = [_espectro(r, nomes, janela_s=32.0, fmin=0.0, n_janelas=8,
                           escala=escala_y)
                 for _, r in etapas]

    # (título da linha, índice do estado antes, índice depois, tipo)
    todas = [
        (f"passa-alta {l_freq:g} Hz", 0, 1, "pa"),
        (f"passa-baixa {h_freq:g} Hz", 1, 2, "pb"),
        (f"notch {freq_rede:g} Hz" if freq_rede else "notch (sem rede)", 2, 3, "notch"),
        ("CAR", 3, 4, "car"),
    ]
    linhas = [l for l in todas if l[3] in etapas_visiveis]
    if not linhas:
        raise ValueError(f"nenhuma etapa visível em {etapas_visiveis!r}")

    # TETO da escala. Em dB: medido acima de 1 Hz, porque incluir o DC
    # faria o offset do sinal bruto, que é o que o passa-alta remove,
    # definir sozinho o topo e achatar contra o piso tudo o que interessa.
    # Em linear: percentil 99 acima de 2 Hz sobre os estados que aparecem
    # (ver _teto_linear); o DC não entra porque a versão linear não desenha
    # a linha do passa-alta.
    f_ref = espectros[0][0]
    if escala_y == "linear":
        usados = sorted({i for _, a, d, _ in linhas for i in (a, d)})
        teto = _teto_linear([espectros[i][1] for i in usados], f_ref, 2.0, f_max)
        lim_y = (0.0, teto * 1.10)
    else:
        util = (f_ref >= 1.0) & (f_ref <= f_max)
        teto = max(e[:, util].max() for _, e in espectros) + 3.0
        lim_y = (teto - FAIXA_DB, teto)

    # hspace generoso: o painel da direita leva um título de DUAS linhas
    # (o estado e a métrica da etapa), e com o espaçamento padrão a
    # segunda linha invade a moldura do painel de cima
    # em linear os rótulos do eixo x são mais largos ("10 20 30…") e o título
    # de duas linhas da fileira de baixo caía em cima deles; mais folga
    hspace = 0.42 if escala_x == "log" else 0.62
    fig, eixos = plt.subplots(len(linhas), 2, figsize=(6.1, 1.95 * len(linhas) + 0.2),
                              sharex=True, sharey=True, squeeze=False,
                              gridspec_kw={"hspace": hspace, "wspace": 0.13})

    for k, (nome_etapa, i_antes, i_depois, tipo) in enumerate(linhas):
        f_a, e_a = espectros[i_antes]
        f_d, e_d = espectros[i_depois]

        for coluna in (0, 1):
            ax = eixos[k][coluna]
            f, e = (f_a, e_a) if coluna == 0 else (f_d, e_d)

            # a faixa que ESTA etapa ataca, sombreada nos dois painéis
            if tipo == "pa":
                ax.axvspan(f_min_plot, l_freq, color="#c8a86a", alpha=0.30, lw=0, zorder=0)
            elif tipo == "pb":
                ax.axvspan(h_freq, f_max, color="#c8a86a", alpha=0.22, lw=0, zorder=0)
            elif tipo == "notch":
                for h in harmonicos_visiveis:
                    ax.axvspan(h * 0.94, h * 1.06, color="#c8a86a", alpha=0.45,
                               lw=0, zorder=0)

            # fantasma do "antes" por baixo, só na coluna da direita
            if coluna == 1:
                for i in range(len(rotulos)):
                    ax.plot(f_a, e_a[i], lw=LW_FANTASMA, color="#b9b9b9",
                            ls="-", zorder=1)

            for i, rot in enumerate(rotulos):
                ax.plot(f, e[i], lw=LW_DADO, color=CORES[i], ls=TRACOS[i],
                        zorder=3, label=rot if (k == 0 and coluna == 0) else None)

            if h_freq:
                ax.axvline(h_freq, color="#555", ls="--", lw=LW_MARCA, zorder=2)
            for h in harmonicos_visiveis:
                ax.axvline(h, color="#b03030", ls=":", lw=LW_MARCA, zorder=2)

            # EIXO LOG em frequência, no padrão. Em escala linear até a
            # Nyquist, a faixa que o passa-alta ataca (0 a 0,5 Hz) ocupa
            # 0,2 % da largura e some, enquanto a banda acima do corte do
            # passa-baixa, que fica vazia depois da segunda linha, ocupa
            # três quartos do painel. O log corrige os dois de uma vez.
            # A versão LINEAR (cascata_linear) existe para outro pedido:
            # ver a banda de interesse, 0,5 a 60 Hz, sem compressão.
            ax.set_xscale(escala_x)
            ax.set_xlim(f_min_plot, f_max)
            ax.set_ylim(*lim_y)
            if escala_y == "linear":
                _formatador_linear(ax)
                # só o maior pico por painel: o painel é pequeno e três
                # linhas cobririam dado
                _anotar_fora_da_escala(
                    ax, _picos_fora_da_escala(e, f, f_max, lim_y[1], rotulos, maximo=1), f_max)
            ax.grid(axis="both", alpha=0.15, lw=0.4, which="major")
            ax.spines[["top", "right"]].set_visible(False)
            # sharex/sharey garantem escala idêntica nos oito painéis, mas
            # por padrão escondem os rótulos de todas as linhas menos a
            # última e de todas as colunas menos a primeira. O leitor
            # ficava sem saber em que frequência estava sem descer até o
            # rodapé da figura. Escala continua compartilhada; os números
            # voltam a aparecer em cada painel.
            ax.tick_params(labelsize=7, labelbottom=True, labelleft=True)

            estado = etapas[i_antes][0] if coluna == 0 else etapas[i_depois][0]
            rotulo_col = "antes" if coluna == 0 else "depois"
            ax.set_title(f"{rotulo_col}: {estado}", fontsize=7, pad=3,
                         loc="left", color="#333")

        eixos[k][0].set_ylabel(_rotulo_psd(escala_y, unidade), fontsize=8)

        # o nome do tratamento, à esquerda da linha inteira
        eixos[k][0].text(-0.30, 0.5, nome_etapa, transform=eixos[k][0].transAxes,
                         fontsize=9, fontweight="bold", color="#1b4e79",
                         rotation=90, va="center", ha="center")

        # a seta que faz a figura ser uma cascata e não quatro comparações
        seta = ConnectionPatch(
            xyA=(1.015, 0.5), coordsA=eixos[k][0].transAxes,
            xyB=(-0.015, 0.5), coordsB=eixos[k][1].transAxes,
            arrowstyle="-|>", mutation_scale=9, lw=0.8,
            color="#1b4e79", shrinkA=1, shrinkB=1,
        )
        fig.add_artist(seta)

    eixos[-1][0].set_xlabel("frequência (Hz)", fontsize=8)
    eixos[-1][1].set_xlabel("frequência (Hz)", fontsize=8)
    # em log o canto inferior esquerdo é livre (o espectro cai abaixo do
    # corte); em linear ele é justamente onde o 1/f e o alfa vivem, e o vão
    # livre é o meio da banda, à direita, entre o alfa e o corte
    eixos[0][0].legend(ncol=len(rotulos), fontsize=6.5,
                       loc="lower left" if escala_x == "log" else "center right",
                       framealpha=0.85, handlelength=1.4, columnspacing=0.8)

    # Ticks espaçados o bastante para não colidirem. Em log, 45 e 60 Hz
    # ficam a 1,2 mm um do outro e os rótulos viram "4560".
    if escala_x == "log":
        marcas = [m for m in (0.5, 1, 5, 10, 50, 200) if f_min_plot <= m <= f_max]
        for ax in eixos.ravel():
            ax.set_xticks(marcas)
            ax.set_xticklabels([f"{m:g}" for m in marcas])
            ax.minorticks_off()
    else:
        # em linear os ticks regulares servem; 10 em 10 Hz com traço menor
        # a cada 5, e nada de rótulo apinhado
        for ax in eixos.ravel():
            ax.xaxis.set_major_locator(MultipleLocator(10))
            ax.xaxis.set_minor_locator(MultipleLocator(5))

    # As frequências que a discussão cita nominalmente — o corte e os
    # harmônicos da rede — ganham rótulo JUNTO DA LINHA VERTICAL, e não no
    # eixo. É a mesma solução da grade 2x2, e resolve o que os ticks não
    # resolviam: a legenda manda o leitor olhar 120 e 180 Hz, e não havia
    # nada na figura dizendo onde eles ficam.
    # Rotacionados e na BASE do painel: em pé eles ocupam pouca largura, e
    # 120 e 180 Hz ficam a 10 pt um do outro em escala log — deitados, os
    # rótulos se sobrepõem. A base é o lugar livre nessas linhas, porque é
    # exatamente ali que o passa-baixa já derrubou o sinal; o topo é do
    # título de duas linhas.
    piso = lim_y[0]

    def _lado(freq):
        """Alinhamento do rótulo, para ele cair sempre DENTRO do painel.

        Numa banda que termina logo depois da marca, o rótulo ancorado à
        esquerda sai pela borda. É o caso do adhdata: com Nyquist em
        ​64 Hz, o corte de 45 e a rede de 50 ficam a 97 % da largura do
        eixo log, e ambos os rótulos escapavam do quadro."""
        if escala_x == "log":
            pos = np.log10(freq / f_min_plot) / np.log10(f_max / f_min_plot)
        else:
            pos = (freq - f_min_plot) / (f_max - f_min_plot)
        return ("right", -1.5) if pos > 0.85 else ("left", 1.5)

    for k, (_, _, _, tipo) in enumerate(linhas):
        if tipo == "pb" and h_freq:
            ha, dx = _lado(h_freq)
            for coluna in (0, 1):
                eixos[k][coluna].annotate(
                    f"corte {h_freq:g} Hz", (h_freq, piso), fontsize=5.5,
                    color="#555", ha=ha, va="bottom", rotation=90,
                    xytext=(dx, 2), textcoords="offset points",
                    bbox=FUNDO_ROTULO)
        elif tipo == "notch":
            for coluna in (0, 1):
                for h in harmonicos_visiveis:
                    ha, dx = _lado(h)
                    eixos[k][coluna].annotate(
                        f"{h:g} Hz", (h, piso), fontsize=5.5, color="#b03030",
                        ha=ha, va="bottom", rotation=90, xytext=(dx, 2),
                        textcoords="offset points", bbox=FUNDO_ROTULO)

    # A MÉTRICA DE CADA LINHA é a que corresponde ao que a etapa faz. A
    # mesma métrica nas quatro linhas repetiria, em número, o erro de
    # leitura que esta figura veio corrigir: a razão da rede mede
    # 45,99 -> 45,98 no passa-alta e 0,27 -> 0,32 no CAR, e as duas etapas
    # pareceriam inúteis.
    razoes = {}
    for nome, r in etapas:
        psd, freqs = _psd_medio(r)
        razoes[nome] = razao_pico_vizinhanca_db(psd, freqs, freq_rede) if freq_rede else None

    p_antes = _potencia_faixa_db(etapas[0][1], nomes, 0.0, l_freq)
    p_depois = _potencia_faixa_db(etapas[1][1], nomes, 0.0, l_freq)

    # As notas ficam indexadas pelo TIPO da etapa, não pela posição da
    # linha: a cascata linear omite a linha do passa-alta, e uma lista
    # posicional colaria a nota do passa-alta na linha do passa-baixa.
    notas = {}
    if p_antes is None:
        notas["pa"] = ""
    elif escala_y == "linear":
        # um NÍVEL em dB sobre um eixo linear confundiria; o fator de queda
        # diz a mesma coisa na linguagem do eixo. O par em dB fica no medidas
        fator = 10 ** ((p_antes - p_depois) / 10.0)
        # uma casa quando o fator é pequeno: "caiu 1×" diria que nada mudou
        # quando a queda medida no adhdata é de 1,1×
        notas["pa"] = f"potência abaixo de {l_freq:g} Hz caiu {fator:.1f}×" if fator < 10 \
            else f"potência abaixo de {l_freq:g} Hz caiu {fator:.0f}×"
    else:
        notas["pa"] = f"potência abaixo de {l_freq:g} Hz: {p_antes:.1f} -> {p_depois:.1f} dB"
    for tipo, i_antes, i_depois in (("pb", 1, 2), ("notch", 2, 3)):
        a, d = razoes[etapas[i_antes][0]], razoes[etapas[i_depois][0]]
        notas[tipo] = (f"razão pico/vizinhança em {freq_rede:g} Hz: {a:.2f} -> {d:.2f} dB"
                       if a is not None else "sem rede detectada")
    if harmonicos_fora:
        # a razão é em dB por ser RAZÃO, mesmo quando o eixo é linear; e
        # os harmônicos que a faixa pedida não alcança são ditos, não omitidos
        # em linha própria: junto da razão, o título passava da borda direita
        # da figura e obrigava o LaTeX a encolher tudo para caber
        notas["notch"] += ("\nharmônicos fora da faixa: "
                           + ", ".join(f"{h:g}" for h in harmonicos_fora)
                           + " Hz (ver versão log)")

    m_antes = float(etapas[3][1].get_data().mean(axis=0).std() * 1e6)
    m_depois = float(etapas[4][1].get_data().mean(axis=0).std() * 1e6)
    notas["car"] = (f"média entre canais: {m_antes:.2f} -> {m_depois:.0e} "
                    f"(efeito espacial, não espectral)")

    # a nota vai para o TÍTULO do painel da direita, e não para dentro da
    # área de plotagem: em cima das curvas ela cobria justamente a banda
    # alta, que é onde o notch e o passa-baixa fazem o seu trabalho
    for k, (_, _, _, tipo) in enumerate(linhas):
        nota = notas.get(tipo)
        if nota:
            atual = eixos[k][1].get_title(loc="left")
            eixos[k][1].set_title(f"{atual}\n{nota}", fontsize=7, pad=3,
                                  loc="left", color="#333", linespacing=1.5)

    fig.suptitle(titulo, fontsize=9)
    fig.savefig(caminho_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # os números que o caderno cita, medidos aqui e não à mão. As chaves da
    # cascata continuam sendo os rótulos de etapa, porque o medidas.json é
    # copiado linha a linha para a tabela do capítulo 4.
    medidas = dict(razoes)
    medidas["_potencia_abaixo_do_corte"] = {
        "antes": p_antes, "depois": p_depois, "f_corte_hz": l_freq,
    }
    medidas["_figura"] = {
        "escala_x": escala_x, "escala_y": escala_y,
        "linhas": [tipo for _, _, _, tipo in linhas],
        "f_min_hz": float(f_min_plot), "f_max_hz": float(f_max),
        "faixa_y": [float(lim_y[0]), float(lim_y[1])],
        "harmonicos_fora_da_faixa": [float(h) for h in harmonicos_fora],
    }
    return medidas


def cascata_linear(raw, nomes, rotulos, freq_rede, caminho_pdf, titulo,
                   l_freq=0.5, h_freq=45.0, unidade="µV", f_max=60.0):
    """A cascata irmã: eixo x LINEAR de 0,5 Hz a `f_max`, y em µV²/Hz.

    Pedido da orientadora em 07/09/2026: ver a banda que interessa, depois
    de passa-baixa, notch e CAR, sem a compressão do log e sem dB. Não
    substitui a versão log: aquela é a única em que a faixa do passa-alta
    (abaixo de 0,5 Hz) e os harmônicos de 120 e 180 Hz do HBN cabem no
    mesmo quadro. Por isso a linha do passa-alta fica de fora aqui: acima
    de 0,5 Hz ela mostraria dois painéis idênticos."""
    return cascata_de_etapas(
        raw, nomes, rotulos, freq_rede, caminho_pdf, titulo,
        l_freq=l_freq, h_freq=h_freq, unidade=unidade,
        escala_y="linear", escala_x="linear", f_min=0.5, f_max=f_max,
        etapas_visiveis=("pb", "notch", "car"),
    )


# As mesmas faixas de caracteristicas.BANDAS, com o símbolo que o leitor de
# EEG usa. Gama (30-45 Hz) fica de fora do espectro final de propósito: a
# figura existe para ver o sinal onde ele tem energia, e acima de 30 Hz o
# que resta depois do passa-baixa de 45 Hz é pouco.
BANDAS_FINAL = [("δ", 1.0, 4.0), ("θ", 4.0, 8.0), ("α", 8.0, 13.0), ("β", 13.0, 30.0)]


def espectro_final(raw_sem, raw_com, nomes, rotulos, caminho_pdf, titulo,
                   unidade="µV", f_min=0.5, f_max=30.0):
    """O espectro do sinal já tratado, ampliado na faixa em que ele vive.

    Pedido da orientadora em 08/09/2026: nas figuras de eixo linear, ver o
    sinal centralizado, de 0 a 30 Hz, "os que têm valor". A cascata não
    serve para isso, porque o passa-baixa (45 Hz) e o notch (50 ou 60 Hz)
    atuam fora dessa faixa e as linhas deles ficariam iguais antes e
    depois. Esta figura mostra só o resultado: sem CAR e com CAR, lado a
    lado, mesma escala, com as quatro bandas de interesse sombreadas para o
    leitor localizar theta (4 a 8 Hz) e beta (13 a 30 Hz), que são as da
    razão do capítulo seguinte.

    Welch de 32 s com n_fft de 8 s (resolução de 0,125 Hz), como na
    cascata: para VER o espectro, a janela curta da grade 2x2 é ruidosa
    demais."""
    f_a, e_a = _espectro(raw_sem, nomes, janela_s=32.0, fmin=0.0, n_janelas=8, escala="linear")
    f_b, e_b = _espectro(raw_com, nomes, janela_s=32.0, fmin=0.0, n_janelas=8, escala="linear")
    teto = _teto_linear([e_a, e_b], f_a, 2.0, f_max)
    lim_y = (0.0, teto * 1.10)

    fig, eixos = plt.subplots(1, 2, figsize=(6.1, 2.9), sharey=True,
                              gridspec_kw={"wspace": 0.10})
    fig.subplots_adjust(top=0.80, bottom=0.17)
    for ax, (f, e, nome) in zip(eixos, ((f_a, e_a, "sem CAR"), (f_b, e_b, "com CAR"))):
        # bandas por baixo de tudo, em cinza alternado, com a letra no topo
        for k, (simbolo, lo, hi) in enumerate(BANDAS_FINAL):
            ax.axvspan(lo, min(hi, f_max), color="#000000", alpha=0.035 if k % 2 else 0.07,
                       lw=0, zorder=0)
            ax.text((lo + min(hi, f_max)) / 2, 0.985, simbolo, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color="#555", zorder=4)
        for i, rot in enumerate(rotulos):
            ax.plot(f, e[i], lw=LW_DADO, color=CORES[i], ls=TRACOS[i], zorder=3,
                    label=rot if nome == "sem CAR" else None)
        ax.set_xlim(f_min, f_max)
        ax.set_ylim(*lim_y)
        _formatador_linear(ax)
        # abaixo da linha das letras de banda, que ocupa o topo do painel
        _anotar_fora_da_escala(ax, _picos_fora_da_escala(e, f, f_max, lim_y[1], rotulos, maximo=2),
                               f_max, y=0.86)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(alpha=0.18, lw=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(nome, fontsize=8, loc="left", color="#333", pad=10)
        ax.set_xlabel("frequência (Hz)")
    eixos[0].set_ylabel(_rotulo_psd("linear", unidade))
    eixos[0].legend(loc="upper right", fontsize=7, frameon=True, framealpha=0.85,
                    handlelength=1.6, bbox_to_anchor=(1.0, 0.90))
    fig.suptitle(titulo, fontsize=9, y=0.99)
    fig.savefig(caminho_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return {"f_min_hz": float(f_min), "f_max_hz": float(f_max),
            "faixa_y": [float(lim_y[0]), float(lim_y[1])],
            "bandas": {s: [lo, hi] for s, lo, hi in BANDAS_FINAL},
            "janela_s": 32.0, "resolucao_hz": 0.125}


def gerar_figuras(caminho, pasta_saida, subject_id=None, rotulo=None,
                  l_freq=0.5, h_freq=45.0, escala="linear",
                  gerar_cascata_linear=False, f_max_linear=60.0,
                  f_max_final=30.0):
    """Roda o conjunto todo sobre um arquivo e devolve as medidas medidas.

    `caminho`     .set do BIDS ou o adhdata.csv
    `pasta_saida` onde gravar os PDFs. Criada se não existir
    `subject_id`  qual sujeito extrair, quando o arquivo tem vários
    `rotulo`      prefixo dos nomes de arquivo e chave no medidas.json. Sem
                  ele, o stem do arquivo — o que faria dois bancos com
                  arquivos de mesmo nome se sobrescreverem
    `l_freq`      corte do passa-alta, em Hz
    `h_freq`      corte do passa-baixa, em Hz
    `escala`      "linear" (µV²/Hz, o pedido de 01/09/2026) ou "db" para as
                  grades 2x2. A cascata log fica sempre em dB: em linear a
                  raia DC do bruto do HBN vira um muro e a linha do
                  passa-alta, que existe para mostrar a remoção dele,
                  fica ilegível
    `gerar_cascata_linear`  acrescenta a cascata irmã, x linear 0,5 a
                  `f_max_linear` Hz, só com passa-baixa, notch e CAR

    Grava três ou quatro PDFs (grade bruta, grade tratada, cascata log e,
    se pedida, cascata linear) e devolve o
    dicionário de medidas que o caderno cita número a número: a cascata da
    razão de rede etapa a etapa, o desvio do canal de referência antes e
    depois do CAR, e a média instantânea entre canais.

    A UNIDADE DO EIXO É DECIDIDA AQUI, e não no desenho: bancos com mais de
    32 canais recebem "un. do arquivo" em vez de µV. O motivo é honestidade
    e não estética — o release do HBN não declara calibração verificada, e
    escrever µV no eixo afirmaria uma coisa que ninguém conferiu.

    A GRADE 2x2 usa limites de eixo calculados UMA vez sobre os dois
    conjuntos. É exigência da folha ("mesma escala") e é o que impede a
    comparação de mentir: o CAR reduz a amplitude, o matplotlib reescalaria
    sozinho, e os dois painéis passariam a parecer ter a mesma energia."""
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    raw = preproc_basico.carregar_raw(caminho, subject_id)
    nomes, rotulos = escolher_canais(raw)
    freq_rede, diagnostico = preproc_basico.detectar_frequencia_rede(raw)
    rotulo = rotulo or Path(caminho).stem

    # a escala do HBN não é µV fisiológico verificado; dizer "µV" no eixo
    # seria afirmar uma calibração que o release não declara
    unidade = "µV" if len(raw.ch_names) <= 32 else "un. do arquivo"

    print(f"[figuras_limpeza] {rotulo}: {len(raw.ch_names)} canais @ "
          f"{raw.info['sfreq']:.0f} Hz | linha média: {', '.join(rotulos)} "
          f"({', '.join(nomes)}) | rede: {freq_rede}")

    # o "tratado" da folha: as três etapas de filtro, sem CAR
    tratado, decisoes = preproc_basico.preprocessar(
        raw, l_freq=l_freq, h_freq=h_freq, freq_rede=freq_rede
    )
    tratado_car = preproc_basico.aplicar_car(tratado)

    medidas = {
        "rotulo": rotulo, "arquivo": str(caminho), "n_canais": len(raw.ch_names),
        "fs": float(raw.info["sfreq"]), "canais": nomes, "rotulos": rotulos,
        "unidade": unidade, "l_freq": l_freq, "h_freq": h_freq,
        "escala": escala,
        "janela_s": [float(JANELA_S), float(INICIO_S)],
        "freq_rede": freq_rede, "motivo_rede": diagnostico.get("motivo"),
        "harmonicos": decisoes["harmonicos_notchados"], "ordem": decisoes["ordem"],
        "conferencia_montagem": conferir_mapa(raw, nomes),
    }

    # O inset de zoom só serve onde o offset é grande a ponto de achatar o
    # traço até virar reta. Mede-se isso, não se presume: a razão entre a
    # excursão total e a variação em torno da média de cada canal diz se
    # há forma visível no painel grande. No HBN ela passa de 100; no
    # adhdata fica perto de 4, e lá o inset repetiria o que já se vê.
    bruto = raw.get_data(picks=[raw.ch_names.index(n) for n in nomes]) * 1e6
    lim = np.abs(bruto).max()
    # que FRAÇÃO da altura do painel o offset consome. É essa fração que
    # decide se sobra espaço para a forma do sinal: a primeira métrica
    # tentada aqui foi pico/desvio, que mede outra coisa (o quanto o maior
    # artefato se destaca do ruído) e dava 12,8 contra 8,6 nos dois bancos,
    # perto demais para separar.
    ocupacao = np.abs(bruto.mean(axis=1)).max() / max(lim, 1e-12)
    precisa_zoom = ocupacao > 0.5
    print(f"[figuras_limpeza] offset ocupa {ocupacao:.0%} do painel "
          f"-> inset {'sim' if precisa_zoom else 'nao'}")

    medidas["grade_bruto"] = grade_2x2(
        raw, preproc_basico.aplicar_car(raw), nomes, rotulos,
        pasta_saida / f"grade-2x2-bruto-{rotulo}.pdf",
        "sinal bruto, sem CAR e com CAR",
        unidade=unidade, freq_rede=freq_rede, zoom=precisa_zoom, escala=escala,
    )
    medidas["grade_tratado"] = grade_2x2(
        tratado, tratado_car, nomes, rotulos,
        pasta_saida / f"grade-2x2-tratado-{rotulo}.pdf",
        "sinal tratado (passa-alta, passa-baixa, notch), sem CAR e com CAR",
        unidade=unidade, freq_rede=freq_rede, h_freq=h_freq, escala=escala,
    )
    # A cascata segue a MESMA escala das grades. Em linear, x vai de 0 a
    # 60 Hz: começa em zero para a linha do passa-alta mostrar a faixa que
    # ela ataca, e vai até 60 porque é onde passa-baixa (45 Hz) e notch (50
    # ou 60 Hz) trabalham. A raia DC do bruto sai pelo topo e é anotada, como
    # qualquer pico. Harmônicos acima de 60 Hz vão para a nota do notch. A
    # versão dB com x logarítmico só sai com --escala db (08/09/2026: a
    # orientadora pediu FFT pura no y e frequência linear no x em TODAS).
    if escala == "linear":
        cascata_kw = dict(escala_y="linear", escala_x="linear", f_min=0.0, f_max=60.0)
    else:
        cascata_kw = dict(escala_y="db", escala_x="log")
    medidas["cascata"] = cascata_de_etapas(
        raw, nomes, rotulos, freq_rede,
        pasta_saida / f"cascata-etapas-{rotulo}.pdf",
        f"o que cada etapa removeu, canais {', '.join(rotulos)}",
        l_freq=l_freq, h_freq=h_freq, unidade=unidade, **cascata_kw,
    )
    if gerar_cascata_linear:
        medidas["cascata_linear"] = cascata_linear(
            raw, nomes, rotulos, freq_rede,
            pasta_saida / f"cascata-linear-{rotulo}.pdf",
            f"passa-baixa, notch e CAR em escala linear, 0,5 a {f_max_linear:g} Hz, "
            f"canais {', '.join(rotulos)}",
            l_freq=l_freq, h_freq=h_freq, unidade=unidade, f_max=f_max_linear,
        )
    if f_max_final:
        medidas["espectro_final"] = espectro_final(
            tratado, tratado_car, nomes, rotulos,
            pasta_saida / f"espectro-final-{rotulo}.pdf",
            f"sinal tratado, {l_freq:g} a {f_max_final:g} Hz, canais {', '.join(rotulos)}",
            unidade=unidade, f_min=l_freq, f_max=f_max_final,
        )

    # o efeito do CAR sobre o canal de referência, que só aparece quando a
    # referência física está entre os canais (o Cz do HBN)
    if "Cz" in raw.ch_names:
        i = raw.ch_names.index("Cz")
        medidas["cz"] = {
            "desvio_antes": float(raw.get_data(picks=[i]).std() * 1e6),
            "desvio_depois": float(preproc_basico.aplicar_car(raw).get_data(picks=[i]).std() * 1e6),
        }

    # ANTES é o sinal já filtrado, não o bruto: comparar bruto com
    # filtrado+CAR creditaria ao CAR a remoção de offset que foi trabalho
    # do passa-alta. Assim o par isola o CAR, que é o que a tabela afirma.
    medidas["media_instantanea"] = {
        "antes": float(tratado.get_data().mean(axis=0).std() * 1e6),
        "depois": float(tratado_car.get_data().mean(axis=0).std() * 1e6),
        "antes_do_bruto": float(raw.get_data().mean(axis=0).std() * 1e6),
        "nota": "antes/depois isolam o CAR (os dois já filtrados); "
                "antes_do_bruto inclui o offset que o passa-alta remove",
    }

    print(f"[figuras_limpeza] média instantânea entre canais (isolando o CAR): "
          f"{medidas['media_instantanea']['antes']:.3f} -> "
          f"{medidas['media_instantanea']['depois']:.3e}")
    print(f"[figuras_limpeza] figuras em {pasta_saida}")
    return medidas


if __name__ == "__main__":
    import argparse

    # Os quatro posicionais são os de sempre (o caderno de 01/09 documenta a
    # chamada); as opções novas vêm depois. `subject_id` vazio ("") continua
    # significando "o padrão do arquivo", como antes.
    p = argparse.ArgumentParser(
        description="figuras do exercício de limpeza; ver docstring do módulo")
    p.add_argument("arquivo")
    p.add_argument("subject_id", nargs="?", default=None)
    p.add_argument("pasta_saida", nargs="?", default=str(
        Path(__file__).resolve().parent.parent.parent / "relatorios" / "figuras"))
    p.add_argument("rotulo", nargs="?", default=None)
    p.add_argument("--escala", choices=("linear", "db"), default="linear",
                   help="escala do eixo y das grades 2x2 (padrão: linear, µV²/Hz)")
    p.add_argument("--com-cascata-linear", action="store_true",
                   help="gerar também a cascata de 3 linhas (PB, notch, CAR) de 0,5 Hz a --fmax-linear; "
                        "desde 08/09/2026 a cascata principal já é linear e esta é redundante")
    p.add_argument("--fmax-linear", type=float, default=60.0,
                   help="limite superior, em Hz, da cascata linear (padrão 60)")
    p.add_argument("--fmax-final", type=float, default=30.0,
                   help="limite superior, em Hz, do espectro final ampliado (padrão 30; 0 desliga)")
    args = p.parse_args()

    caminho = args.arquivo
    sujeito = args.subject_id or None
    pasta = Path(args.pasta_saida)
    rotulo = args.rotulo

    medidas = gerar_figuras(caminho, pasta, sujeito, rotulo, escala=args.escala,
                            gerar_cascata_linear=args.com_cascata_linear,
                            f_max_linear=args.fmax_linear, f_max_final=args.fmax_final)

    # grava o medidas.json que o caderno cita: antes o dicionário era
    # devolvido e descartado aqui, e o arquivo só existia porque alguém o
    # tinha gerado à mão por outro caminho
    destino = Path(pasta) / "medidas.json"
    anterior = {}
    if destino.is_file():
        anterior = json.loads(destino.read_text(encoding="utf-8"))
    anterior[medidas["rotulo"]] = medidas
    destino.write_text(json.dumps(anterior, indent=2, ensure_ascii=False, default=float),
                       encoding="utf-8")
    print(f"[figuras_limpeza] medidas em {destino}")
