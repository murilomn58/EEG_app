"""
Pré-processamento básico de EEG: passa-alta em ~0,5 Hz e notch na
frequência da rede elétrica, NESTA ORDEM, com a rede DETECTADA do
espectro em vez de fixada no código.

Existe porque o app mede bandas sobre sinal cru, e o dano disso está
medido no README: offset de canal de +130 a +145 (deveria ser ~0),
delta absorvendo 47-72% da potência total (muito disso é deriva, não
atividade delta), e um pico de rede de +6 a +16 dB acima do pico alfa
— ou seja, cerca de 40x mais forte que o ritmo que se quer medir.
Normalizar por potência total nesse sinal é normalizar pela tomada da
parede.

POR QUE A REDE É DETECTADA, NÃO FIXADA: o adhdata.csv tem interferência
em 50 Hz; o HBN-EEG foi gravado em Nova York, onde a rede é 60 Hz.
Fixar o número no código é o tipo de detalhe que vira bug silencioso
quando o pipeline migra de um dataset para o outro — o notch erraria o
alvo, deixando a rede intacta E abrindo um buraco espectral no meio do
gamma.

POR QUE A ORDEM IMPORTA: filtros digitais têm resposta transitória nas
bordas, e um offset DC grande na entrada faz o transitório do notch
durar mais e contaminar mais amostras. Removida a deriva, o notch opera
sobre um sinal já centrado.

UNIDADE: o MNE trabalha em VOLTS. O adhdata.csv está em microvolts, e
por isso carregar_raw converte na entrada (* 1e-6). Quem for reportar
número para humano converte de volta (* 1e6) — é assim que o
qc_relatorio reproduz os +130 a +145 do README literalmente.

Nos .set do HBN o MNE já faz essa conversão sozinho (EEGLAB grava em µV
por convenção), e está certo em fazê-la. Mas o número resultante não pode
ser lido como microvolt conferido: medido no Release 1, o percentil 99 do
sinal BRUTO dá ~136.000 µV.

RETRATAÇÃO, e ela é deste próprio cabeçalho. Uma versão anterior deste
texto concluía, daquele ~136.000, que o dado estaria numa escala
arbitrária de um amplificador não calibrado. A evidência não sustenta
essa conclusão: o percentil 99 do sinal BRUTO é dominado pelo offset DC,
não pela amplitude do EEG. Depois do tratamento, o pico do HBN fica em
76,9 contra 1240,3 do adhdata (números medidos e registrados no README) —
dezesseis vezes MENOR, e dentro da faixa fisiológica. A leitura
parcimoniosa é microvolt com offset DC grande. O que fica de pé é só
isto: a CALIBRAÇÃO NÃO ESTÁ CONFIRMADA. Ninguém conferiu o ganho da
cadeia de aquisição contra um sinal de referência conhecido, então o
fator de escala do arquivo é desconhecido — o que é diferente de ser
arbitrário, e a diferença não é retórica: "desconhecido" é uma pendência
de verificação, "arbitrário" seria uma afirmação sobre o dado que a
medida não autoriza.

A consequência prática é a mesma de antes, e é ela que muda o que se pode
dizer com o número na mão: a média por canal em µV fica incomparável
entre HBN e adhdata. Isso não afeta filtro nenhum (passa-alta e notch são
lineares e não dependem de escala) nem a razão pico/rede (é uma razão).
Leia a média como "distância do zero na escala do dataset", não como
microvolt conferido.

Este módulo NÃO decide se o filtro funcionou: quem mede é o
qc_relatorio.py, que importa daqui justamente para medir o mesmo filtro
que roda em produção.

Uso: cd backend && python scripts/preproc_basico.py <arquivo> [subject_id]
Saída: imprime as decisões tomadas (rede detectada, harmônicos, ordem)
"""
import sys
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import csv_data

CANDIDATAS_REDE = (50.0, 60.0)

# Um pico precisa erguer isto acima da vizinhança para contar como rede,
# e vencer a outra candidata por esta margem. 3 dB = o dobro da potência:
# abaixo disso não é pico, é ondulação do espectro.
LIMIAR_PICO_DB = 3.0
MARGEM_DECISAO_DB = 3.0

# Fração da Nyquist acima da qual não se confia em medir nem filtrar.
FRACAO_NYQUIST_UTIL = 0.9

# Piso de conteúdo AC, em µV RMS (desvio padrão por canal), abaixo do qual o
# sinal não tem espectro que se possa interrogar.
#
# Existe porque a razão pico/vizinhança é um QUOCIENTE, e quociente entre dois
# números que são só ruído de arredondamento continua devolvendo um número
# perfeitamente apresentável. MEDIDO: 19 canais constantes em 50 µV, 60 s a
# 128 Hz — sinal sem UMA oscilação — saem daqui como "rede detectada: 50,0 Hz"
# com razão de 7,071 dB contra o limiar de 3,0. O desvio padrão desse sinal é
# 1,4e-14 µV, ou seja, o "pico de 50 Hz" é o último bit da mantissa do float
# encostado no offset DC.
#
# 1e-3 µV (1 nV RMS) é o piso, e ele é declarado e não medido do dado: fica
# quatro ordens de grandeza ABAIXO do ruído próprio de um amplificador de EEG
# (~0,1 µV RMS, que é a ordem do passo de digitalização) e onze ordens ACIMA
# do resíduo de ponto flutuante medido acima. Nessa faixa não cabe nenhuma
# gravação real: o que cai abaixo dela é canal morto, sinal sintético
# constante ou arquivo zerado — casos em que a resposta honesta é "não dá para
# medir", e não um número de rede.
PISO_AC_UV = 1e-3

# Largura do notch, em Hz. O default do MNE é f/200 (0,25 Hz em 50 Hz) —
# estreito demais na prática: medindo a razão pico/vizinhança depois de
# filtrar, sobra resíduo nos "ombros" da banda rejeitada, a meio Hz do
# centro. 1 Hz remove a rede sem chegar perto de banda de interesse
# (alfa termina em 13 Hz, gamma em 45).
LARGURA_NOTCH_HZ = 1.0


def carregar_raw(caminho, subject_id=None):
    """Devolve um mne.io.Raw a partir de um .set (HBN, formato EEGLAB) ou
    de um .csv (adhdata). É o ÚNICO ponto deste módulo que conhece
    formato de arquivo — daqui para cima tudo opera sobre Raw e não sabe
    de onde o sinal veio, que é o que evita duplicar o pipeline."""
    caminho = Path(caminho)
    if not caminho.is_file():
        raise ValueError(f"arquivo não encontrado: {caminho}")

    if caminho.suffix == ".set":
        return mne.io.read_raw_eeglab(str(caminho), preload=True, verbose=False)
    if caminho.suffix == ".csv":
        return _raw_de_adhdata(caminho, subject_id)

    raise ValueError(
        f"extensão não suportada: {caminho.suffix} (esperado .set do HBN ou .csv do adhdata)"
    )


def raw_de_dataframe(df, subject_id):
    """Um sujeito de um DataFrame JÁ CARREGADO, como Raw.

    Existe separada de carregar_raw por um motivo de custo, não de estilo: o
    backend mantém o adhdata.csv inteiro em memória (267 MB) desde o
    startup, e qualquer caminho que passe por carregar_raw releria o arquivo
    do disco a cada request. Num endpoint isso são dezenas de segundos e
    centenas de MB por chamada, invisíveis em teste unitário porque o
    fixture tem 40 s de sinal."""
    canais = csv_data.get_subject_raw(df, subject_id)
    # microvolts -> volts, que é o que o MNE assume internamente
    dados = np.array([canais[c] for c in csv_data.CANAIS_19], dtype=float) * 1e-6
    info = mne.create_info(list(csv_data.CANAIS_19), csv_data.FS, "eeg", verbose=False)
    return mne.io.RawArray(dados, info, verbose=False)


def _raw_de_adhdata(caminho_csv, subject_id):
    """Um sujeito do adhdata.csv como Raw, lendo o arquivo. Reaproveita
    csv_data em vez de reler o CSV à mão: a ordem dos canais precisa ser
    exatamente CANAIS_19, que é a mesma que o resto do backend assume."""
    if not subject_id:
        raise ValueError("adhdata.csv exige subject_id (o CSV tem todos os sujeitos juntos)")
    return raw_de_dataframe(csv_data.load_csv(caminho_csv), subject_id)


def _psd_medio(raw, fmax=None, n_canais=20, dur_s=60.0, pular_s=10.0):
    """(psd médio entre canais, freqs). Subamostra canais e tempo porque
    detectar rede não precisa da gravação inteira, e um .set do HBN tem
    129 canais a 500 Hz. Pula os primeiros segundos: início de gravação
    costuma trazer transiente que não representa o sinal."""
    sfreq = raw.info["sfreq"]
    nyquist = sfreq / 2.0
    if fmax is None:
        fmax = min(nyquist * 0.95, 100.0)

    dados = raw.get_data(picks="eeg")
    n_total = dados.shape[0]
    if n_total == 0:
        raise ValueError("nenhum canal de eeg no arquivo")

    # passo fixo, não sorteio: a detecção precisa ser reproduzível
    passo = max(1, n_total // n_canais)
    dados = dados[::passo][:n_canais]

    inicio = int(pular_s * sfreq)
    if inicio >= dados.shape[1]:
        inicio = 0
    fim = inicio + int(dur_s * sfreq)
    dados = dados[:, inicio:fim]

    # 4 s de janela -> resolução de 0,25 Hz, folga enorme pra separar 50 de 60
    n_fft = min(int(sfreq * 4), dados.shape[1])
    if n_fft < 16:
        raise ValueError(
            f"sinal curto demais para estimar espectro: {dados.shape[1]} amostras"
        )

    espectro = mne.time_frequency.psd_array_welch(
        dados, sfreq=sfreq, fmin=1.0, fmax=fmax, n_fft=n_fft, verbose=False
    )
    psd, freqs = espectro
    return psd.mean(axis=0), freqs


def razao_pico_vizinhanca_db(psd, freqs, f0, meia_largura=1.0, guarda=2.0, lateral=5.0):
    """Quanto o pico em f0 se ergue acima da vizinhança, em dB. Usada
    tanto pela detecção quanto pelo relatório de QC — é a MESMA medida
    antes e depois do notch, e é isso que torna o veredito comparável.

    Usa MEDIANA na vizinhança, não média: um harmônico ou um artefato
    dentro da janela lateral inflaria o denominador e mascararia a rede
    justamente no caso em que ela é pior."""
    dentro = (freqs >= f0 - meia_largura) & (freqs <= f0 + meia_largura)
    fora = (
        ((freqs >= f0 - lateral) & (freqs <= f0 - guarda))
        | ((freqs >= f0 + guarda) & (freqs <= f0 + lateral))
    )
    if not dentro.any() or not fora.any():
        return None

    pico = float(psd[dentro].max())
    vizinha = float(np.median(psd[fora]))
    # canal flat ou banda vazia: log10(0) = -inf, que polui tabela e veredito
    if vizinha <= 0 or pico <= 0:
        return None
    return 10.0 * np.log10(pico / vizinha)


def detectar_frequencia_rede(raw, candidatas=CANDIDATAS_REDE):
    """(frequencia, diagnostico) — a rede medida do espectro, ou None com
    o motivo. Devolve None em QUATRO situações DIFERENTES, e a diferença
    importa: sem pico algum significa que o dado já veio notchado (e
    notchar às cegas removeria sinal neural de graça); ambíguo significa
    que a medida não decide e escolher no cara-ou-coroa seria pior que
    admitir; nyquist insuficiente é limitação física do registro; e
    sinal_sem_conteudo_ac significa que não há oscilação nenhuma para
    interrogar — ver PISO_AC_UV, que traz o número medido que motivou o
    motivo novo.

    A ordem das recusas é do mais barato para o mais caro: nyquist sai do
    cabeçalho, o conteúdo AC lê um trecho do sinal, e só quem passa pelos dois
    paga o espectro de Welch.

    O diagnóstico volta inteiro de propósito — o relatório imprime as
    razões de cada candidata, porque uma frequência que aparece do nada
    não é auditável."""
    sfreq = float(raw.info["sfreq"])
    nyquist = sfreq / 2.0
    teto = nyquist * FRACAO_NYQUIST_UTIL

    diagnostico = {
        "sfreq": sfreq,
        "nyquist": nyquist,
        "candidatas": list(candidatas),
        "candidatas_viaveis": [],
        "razoes_db": {},
        "margem_db": None,
        "desvio_padrao_uv": None,
        "motivo": None,
    }

    viaveis = [f for f in candidatas if f < teto]
    diagnostico["candidatas_viaveis"] = viaveis

    if not viaveis:
        diagnostico["motivo"] = "nyquist_insuficiente"
        return None, diagnostico

    # Só um trecho do começo, e não a gravação inteira: a pergunta aqui é "este
    # sinal oscila?", que não precisa de 400 s de HBN em memória (129 canais x
    # 500 Hz x 403 s = 208 MB) para ser respondida. O `stop` limita a leitura na
    # origem, então esta checagem não duplica o custo do _psd_medio logo abaixo.
    fim = min(raw.n_times, int(sfreq * 60.0))
    dados = raw.get_data(picks="eeg", start=0, stop=fim)
    if dados.size:
        # o MAIOR desvio entre canais, e não a média deles: um canal morto no
        # meio de dezoito vivos não é motivo para recusar a medida, mas
        # dezenove canais mortos são
        desvio_uv = float(np.max(dados.std(axis=1)) * 1e6)
        diagnostico["desvio_padrao_uv"] = desvio_uv
        if desvio_uv < PISO_AC_UV:
            diagnostico["motivo"] = "sinal_sem_conteudo_ac"
            return None, diagnostico
    # dados vazio (nenhum canal de eeg) cai adiante: quem levanta o erro com a
    # mensagem certa é o _psd_medio, e duplicá-la aqui é criar duas verdades

    psd, freqs = _psd_medio(raw)
    for f in viaveis:
        diagnostico["razoes_db"][f] = razao_pico_vizinhanca_db(psd, freqs, f)

    medidas = {f: r for f, r in diagnostico["razoes_db"].items() if r is not None}
    if not medidas:
        diagnostico["motivo"] = "sem_pico_de_rede"
        return None, diagnostico

    ordenadas = sorted(medidas.items(), key=lambda par: par[1], reverse=True)
    melhor, razao_melhor = ordenadas[0]

    if len(ordenadas) > 1:
        diagnostico["margem_db"] = razao_melhor - ordenadas[1][1]

    if razao_melhor < LIMIAR_PICO_DB:
        diagnostico["motivo"] = "sem_pico_de_rede"
        return None, diagnostico

    if diagnostico["margem_db"] is not None and diagnostico["margem_db"] < MARGEM_DECISAO_DB:
        diagnostico["motivo"] = "ambiguo"
        return None, diagnostico

    if len(viaveis) == 1:
        # 50 venceu por eliminação, não por medição: a 128 Hz o teto de
        # Nyquist já exclui 60 Hz antes de olhar o espectro. Está certo no
        # adhdata (a rede é 50 mesmo), mas seria errado num dataset de
        # 128 Hz gravado nos EUA — por isso a marca fica no relatório.
        diagnostico["motivo"] = "candidata_unica_por_nyquist"
    else:
        diagnostico["motivo"] = "detectada"

    return melhor, diagnostico


def aplicar_passa_alta(raw, l_freq=0.5):
    """Passa-alta sobre o sinal contínuo, ANTES de qualquer segmentação —
    assim o transitório do filtro contamina só as duas pontas do
    registro, não as bordas de cada segmento.

    O corte fica em 0,5 Hz porque abaixo disso não há ritmo cerebral de
    interesse: há deriva de eletrodo, suor, movimento lento da cabeça e
    offset DC do amplificador. Cortes acima de 1 Hz distorceriam
    componentes lentas de potencial evocado.

    Devolve uma CÓPIA: raw.filter() modifica in place, e sem a cópia o
    relatório compararia o sinal filtrado consigo mesmo e reportaria
    delta zero em tudo — falha silenciosa e plausível."""
    saida = raw.copy()
    saida.filter(l_freq=l_freq, h_freq=None, verbose=False)
    return saida


def aplicar_notch(raw, freq_rede, n_harmonicos=3):
    """Rejeita-faixa na rede e nos harmônicos que couberem abaixo da
    Nyquist útil. Devolve uma cópia, pelo mesmo motivo do passa-alta.

    freq_rede None é caso legítimo (detecção não decidiu): devolve a
    cópia intacta e uma lista vazia de harmônicos, em vez de chutar um
    número."""
    saida = raw.copy()
    if freq_rede is None:
        return saida, []

    teto = raw.info["sfreq"] / 2.0 * FRACAO_NYQUIST_UTIL
    harmonicos = [freq_rede * k for k in range(1, n_harmonicos + 1) if freq_rede * k < teto]
    if not harmonicos:
        return saida, []

    saida.notch_filter(freqs=harmonicos, notch_widths=LARGURA_NOTCH_HZ, verbose=False)
    return saida, harmonicos


def limitar_h_freq(sfreq, h_freq):
    """O corte de passa-baixa que DÁ para aplicar nesta taxa de amostragem,
    ou None quando não há passa-baixa a fazer.

    Existe porque pedir 70 Hz a um sinal de 128 Hz não é um pedido
    ligeiramente exagerado: é impossível. A 128 Hz a Nyquist é 64 Hz, e o
    MNE não corrige nem avisa — ele levanta ValueError ("h_freq must be
    less than the Nyquist frequency 64.0") e derruba o pipeline inteiro.
    Quem chama preprocessar quase nunca sabe de cor a taxa do arquivo que
    acabou de abrir, e um h_freq que serve para o HBN (500 Hz) estoura no
    adhdata (128 Hz). Saber disso é responsabilidade do pipeline, não de
    cada chamador.

    O teto é a Nyquist ÚTIL — FRACAO_NYQUIST_UTIL da Nyquist, o MESMO teto
    que a detecção de rede e o notch já usam — e não a Nyquist exata. Um
    FIR precisa de banda de transição: um corte encostado na Nyquist não
    sai apenas apertado, sai degenerado. A 128 Hz isso dá 57,6 Hz.

    Baixar o corte em silêncio seria pior que o erro que ele evita, porque
    o relatório passaria a medir uma banda diferente da pedida sem que
    nada denunciasse. Por isso preprocessar devolve em decisoes["h_freq"] o
    valor EFETIVAMENTE aplicado, e guarda o pedido original em
    decisoes["h_freq_pedido"] para que a diferença fique auditável."""
    if h_freq is None:
        return None
    teto = float(sfreq) / 2.0 * FRACAO_NYQUIST_UTIL
    return float(min(float(h_freq), teto))


def aplicar_passa_baixa(raw, h_freq=45.0):
    """Passa-baixa sobre o dado contínuo. Tira o que está acima da banda
    de interesse: atividade muscular (EMG), que domina acima de ~30 Hz e
    é a contaminação mais comum em criança acordada, e ruído de alta
    frequência do amplificador.

    O corte fica em 45 Hz porque é onde a banda gama termina, na definição
    que o resto do app usa.

    O QUE ELE NÃO FAZ, medido em vez de suposto: um passa-baixa em 45 Hz
    NÃO substitui o notch, nem numa rede de 50 Hz. Sobre sinal sintético
    com rede em 50 Hz, a queda de potência em 50 Hz foi de 4,5 dB pelo
    passa-baixa contra 43,2 dB pelo notch. A razão é a banda de transição:
    50 Hz fica a 5 Hz do corte, ainda dentro da rampa do FIR, longe da
    região onde o filtro de fato atenua. As duas etapas são necessárias, e
    é por isso que existem separadas.

    Devolve cópia: raw.filter() age in place."""
    saida = raw.copy()
    saida.filter(l_freq=None, h_freq=h_freq, verbose=False)
    return saida


def aplicar_car(raw):
    """Referência média comum (CAR): de cada canal subtrai a média
    instantânea de todos os canais. O que é comum a todos não pode ser
    atividade local de um deles, então é ruído captado por igual, e tirar
    a média o remove sem precisar saber de onde veio.

    Duas consequências que se medem depois e valem registrar:

      - a média instantânea entre canais passa a ser exatamente zero, por
        construção. É o teste de que o CAR foi aplicado.
      - o eletrodo de referência física, se estiver entre os canais, DEIXA
        de ser plano. No HBN o Cz é a referência e vem identicamente zero
        no dado bruto; depois do CAR ele passa a ter sinal. Não é artefato:
        é o sinal que sempre esteve em Cz e estava escondido no zero.

    projection=False aplica de verdade, em vez de pendurar uma projeção
    que só valeria quando alguém a ativasse."""
    saida = raw.copy()
    saida.set_eeg_reference("average", projection=False, verbose=False)
    return saida


def preprocessar(raw, l_freq=0.5, h_freq=None, freq_rede=None, car=False):
    """(raw_filtrado, decisoes). A ordem é fixa e não é arbitrária:

        passa-alta -> passa-baixa -> notch -> CAR

    O passa-alta vem primeiro porque filtro digital tem transitório nas
    bordas, e um offset DC grande na entrada faz o transitório das etapas
    seguintes durar mais e contaminar mais amostras. O CAR vem por último
    porque ele é uma operação entre canais, e faz sentido calcular a média
    comum sobre sinal já limpo: uma deriva enorme num canal só entraria na
    média e se espalharia por todos os outros.

    h_freq=None pula o passa-baixa, e car=False pula o CAR, para que o
    relatório de qualidade consiga medir o espectro depois de cada etapa
    isolada em vez de só no fim.

    O h_freq PEDIDO não é necessariamente o aplicado: ele passa por
    limitar_h_freq, que o rebaixa ao teto de Nyquist útil do próprio
    arquivo. Sem isso, preprocessar(raw, h_freq=70.0) sobre o adhdata, que
    é de 128 Hz, morreria com ValueError do MNE em vez de filtrar o que
    dava para filtrar. O valor que sai em decisoes["h_freq"] é sempre o
    EFETIVAMENTE aplicado, e decisoes["h_freq_pedido"] guarda o que foi
    pedido — comparar os dois é como se descobre que houve rebaixamento,
    sem precisar refazer a conta da Nyquist do lado de fora.

    Se freq_rede não for informada, é detectada do espectro do sinal
    ORIGINAL: medir depois dos filtros também funcionaria, mas medir antes
    deixa o número comparável com o que o relatório reporta como 'antes'."""
    # limita ANTES de montar as decisoes: o que o dicionario reporta tem de
    # ser o que o filtro de fato usou, nunca o que foi pedido
    h_freq_pedido = h_freq
    h_freq = limitar_h_freq(raw.info["sfreq"], h_freq)

    decisoes = {
        "l_freq": l_freq, "h_freq": h_freq, "h_freq_pedido": h_freq_pedido,
        "car": car, "freq_rede": freq_rede, "diagnostico_rede": None,
    }

    if freq_rede is None:
        freq_rede, diagnostico = detectar_frequencia_rede(raw)
        decisoes["freq_rede"] = freq_rede
        decisoes["diagnostico_rede"] = diagnostico

    ordem = ["passa-alta"]
    saida = aplicar_passa_alta(raw, l_freq=l_freq)

    if h_freq is not None:
        saida = aplicar_passa_baixa(saida, h_freq=h_freq)
        ordem.append("passa-baixa")

    saida, harmonicos = aplicar_notch(saida, freq_rede)
    ordem.append("notch")

    if car:
        saida = aplicar_car(saida)
        ordem.append("car")

    decisoes["harmonicos_notchados"] = harmonicos
    decisoes["ordem"] = ordem
    return saida, decisoes


def _imprimir_decisoes(caminho, raw, decisoes):
    print(f"[preproc_basico] {caminho}")
    print(
        f"  {len(raw.ch_names)} canais @ {raw.info['sfreq']:.1f} Hz, "
        f"{raw.n_times / raw.info['sfreq']:.1f}s"
    )

    diagnostico = decisoes["diagnostico_rede"] or {}
    razoes = diagnostico.get("razoes_db", {})
    if razoes:
        print("\n  razão pico/vizinhança por candidata:")
        for f, r in sorted(razoes.items()):
            print(f"    {f:>6.1f} Hz  {'n/a' if r is None else f'{r:>7.2f} dB'}")

    print("\n--- veredito ---")
    if decisoes["freq_rede"] is None:
        print(f"rede NÃO detectada (motivo: {diagnostico.get('motivo')}) — notch não aplicado")
    else:
        print(f"rede detectada: {decisoes['freq_rede']:.1f} Hz (motivo: {diagnostico.get('motivo')})")
        if diagnostico.get("margem_db") is not None:
            print(f"margem sobre a segunda candidata: {diagnostico['margem_db']:.2f} dB")
        if diagnostico.get("motivo") == "candidata_unica_por_nyquist":
            print(
                "ATENÇÃO: única candidata viável a esta taxa de amostragem — venceu por "
                "eliminação, não por medição. Confira a rede elétrica do local de gravação."
            )
    print(f"passa-alta: {decisoes['l_freq']} Hz | harmônicos notchados: {decisoes['harmonicos_notchados']}")

    # o rebaixamento por Nyquist é impresso porque é uma decisão do
    # pipeline, não do usuário: quem pediu 70 Hz e recebeu 57,6 precisa
    # saber disso ANTES de comparar o resultado com o de outro banco
    if decisoes.get("h_freq_pedido") is not None and decisoes["h_freq"] != decisoes["h_freq_pedido"]:
        print(
            f"ATENÇÃO: passa-baixa pedido em {decisoes['h_freq_pedido']:.1f} Hz, aplicado em "
            f"{decisoes['h_freq']:.1f} Hz — teto de Nyquist útil a {raw.info['sfreq']:.1f} Hz"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raiz = Path(__file__).resolve().parent.parent.parent
        caminho_padrao = raiz / "adhdata.csv"
        print(f"uso: python scripts/preproc_basico.py <arquivo> [subject_id]")
        print(f"exemplo: python scripts/preproc_basico.py {caminho_padrao} v10p")
        sys.exit(1)

    caminho = sys.argv[1]
    sujeito = sys.argv[2] if len(sys.argv) > 2 else None
    raw = carregar_raw(caminho, sujeito)
    _filtrado, decisoes = preprocessar(raw)
    _imprimir_decisoes(caminho, raw, decisoes)
