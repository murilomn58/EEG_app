"""
Mede, por sujeito e ANTES/DEPOIS do pré-processamento, quatro métricas.
Três dizem se o filtro fez o serviço; a quarta diz se ele fez o serviço
sem destruir o sinal no caminho.

Existe porque "olhar o traçado e achar bonito" não é verificação.

AS METAS QUE ESTE SCRIPT DE FATO APLICA — são estas, e não outras. Antes
elas estavam escritas como "cair para ~0" e "cair muito", que não são
critérios: não dá para reprovar nada com eles, e o relatório não reprovava
por eles. As constantes logo abaixo são o contrato real:

  média |máx| por canal      residual < MAX_FRACAO_MEDIA_RESIDUAL da média
                             de entrada (hoje 1%). REPROVA se passar.
                             Relativo, não absoluto — o porquê está no
                             comentário das constantes.

  razão pico-de-rede sobre
  a vizinhança, DEPOIS       < MAX_RAZAO_REDE_DB (hoje 3 dB). REPROVA se
                             passar. 3 dB é o dobro da potência: abaixo
                             disso não é mais pico, é ondulação.
                             Vale "n/a", e n/a NÃO é reprovação, quando
                             não houve rede a detectar.

  potência alfa preservada   razão depois/antes >= MIN_RAZAO_ALPHA (hoje
                             0,9). REPROVA se ficar abaixo. É a métrica de
                             NÃO-DANO, a que a maioria esquece: um filtro
                             que zera a média e mata o alfa junto passa nas
                             outras duas e destruiu o dado.

  fração de potência em
  delta (1-4 Hz)             REPORTADA, com a variação antes->depois. NÃO
                             REPROVA, e não tem limiar. O passa-alta corta
                             em 0,5 Hz, então ele só derruba delta quando a
                             potência ali era deriva vazando de baixo; se a
                             energia em 1-4 Hz for atividade delta real, o
                             filtro está CERTO em não tocá-la e reprovar
                             aqui seria exigir que ele destruísse sinal.
                             O que a fração diagnostica é quanto sobrou de
                             deriva. Quem prova que o DC saiu é a média.

Os números de referência do README (média por canal de +130 a +145,
delta absorvendo 47% a 72%, rede de +6 a +16 dB acima do pico alfa) são
o retrato do problema que motivou o filtro. Eles descrevem a ENTRADA
esperada, e é por isso que não aparecem como limiar: o relatório mede o
sujeito que recebeu, não o sujeito do README.

Este script IMPORTA preproc_basico em vez de reimplementar os filtros.
É isso que garante que o relatório mede exatamente o pipeline que roda
em produção — um QC que mede outra implementação não é evidência sobre
nada.

DETALHES DE MEDIDA QUE MUDAM O RESULTADO:

  - A fração delta é medida a partir de 1 Hz, não de 0. Incluindo o bin
    DC, a fração iria a ~0,99 em qualquer sinal com offset e a queda
    mediria a remoção do degrau, não da deriva.
  - A vizinhança do pico de rede usa MEDIANA, não média: um harmônico
    dentro da janela lateral inflaria o denominador e mascararia a rede
    justamente onde ela é pior.
  - Descarta 2 s de cada ponta antes de medir: filtro FIR de fase zero
    deixa transitório nas bordas, e medir ali reportaria como resíduo
    algo que o filtro removeu no miolo.
  - Amostras não finitas (NaN/inf) são CONTADAS e denunciadas antes da
    tabela, e nenhum veredito pode sair OK com elas dentro. O caso que
    motivou isso está na docstring de `metricas_de_raw`.

A UNIDADE DA POTÊNCIA ALFA, E POR QUE A CHAVE MUDOU DE NOME. A chave
chamava-se `potencia_alpha`, a documentação dizia µV²/Hz e o valor saía em
V²/Hz — 1e12 de diferença entre o que estava escrito e o que estava lá.
Escolhido: converter o NÚMERO para µV²/Hz e pôr a unidade no NOME
(`potencia_alpha_uv2_hz`), em vez de corrigir a documentação para V²/Hz.

Três razões, nesta ordem. (1) O relatório inteiro fala em µV — a própria
métrica vizinha se chama `media_uv_abs_max` e já converte de volts —, e um
relatório com duas escalas de tensão é um convite a comparar números que não
se comparam. (2) O leitor é médico ou cientista da saúde: potência alfa em
µV²/Hz é da ordem de dezenas a centenas, uma faixa que ele reconhece; em
V²/Hz o mesmo dado sai como 3e-11, que não diz nada a ninguém. (3) A unidade
no nome sobrevive a quem lê só o JSON de saída, sem abrir esta docstring —
que foi exatamente como o erro sobreviveu tanto tempo.

Isto NÃO move nenhum veredito: o que decide a aprovação é `razao_alpha`,
depois/antes, e o fator 1e12 cancela nos dois lados.

Uso: cd backend && python scripts/qc_relatorio.py <arquivo|pasta_bids> [subject_id]
Saída: ../relatorios/qc_relatorio.txt
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preproc_basico
from preproc_basico import _psd_medio, razao_pico_vizinhanca_db

# Bordas descartadas antes de medir (transitório do filtro FIR).
BORDA_S = 2.0

# Critérios de aprovação por métrica. O de alfa é o mais frouxo dos
# quatro de propósito: o FIR do MNE tem ripple na banda de passagem, e
# exigir 1% geraria falso alarme num filtro que está correto.
#
# A média é avaliada em termos RELATIVOS, não absolutos. Um limiar fixo
# em µV só funciona num dataset cuja calibração alguém conferiu: o adhdata
# parte de ~152 e chega a 0,24, mas o HBN parte de ~194.000 (calibração
# não confirmada, e o valor bruto é dominado pelo offset DC — ver a
# retratação no cabeçalho de preproc_basico) e chega a 5 — uma redução de
# 99,997% que um limiar de "1 µV" reprovaria como falha. O que prova que
# o DC saiu é a queda proporcional, que independe da escala do dataset.
MAX_FRACAO_MEDIA_RESIDUAL = 0.01
MAX_RAZAO_REDE_DB = 3.0
MIN_RAZAO_ALPHA = 0.9


def _aparar_bordas(raw):
    """Recorta BORDA_S de cada ponta. Devolve o próprio raw se a gravação
    for curta demais para aparar — medir com transitório é melhor que não
    medir."""
    duracao = raw.n_times / raw.info["sfreq"]
    if duracao <= 3 * BORDA_S:
        return raw
    return raw.copy().crop(tmin=BORDA_S, tmax=duracao - BORDA_S)


def metricas_de_raw(raw, freq_rede):
    """As quatro métricas de um sinal, num dicionário.

    `raw`        objeto Raw do MNE, em volts (a convenção do MNE).
    `freq_rede`  frequência da rede em Hz, ou None quando não houve pico a
                 detectar. Com None, `razao_rede_db` volta None e NÃO zero:
                 zero significaria "notch perfeito", que é o oposto de
                 "não deu para medir".

    Devolve as chaves:

      n_canais            quantos canais de EEG entraram na conta
      media_uv_abs_max    o pior canal, em µV. É o máximo e não a média das
                          médias: um único canal com offset enorme é um
                          problema, e a média das médias o esconderia
      media_uv_mediana    o canal típico, para comparar com o pior
      fracao_delta        potência em 1-4 Hz sobre a potência em 1 Hz até o
                          teto, adimensional. None se a potência total for
                          zero (canal plano zeraria o denominador)
      razao_rede_db       quanto o pico da rede se destaca da vizinhança
      potencia_alpha_uv2_hz
                          potência em 8-13 Hz, em µV²/Hz — o nome carrega a
                          unidade porque o número já foi convertido de V²/Hz
                          e o par nome/valor precisa concordar sozinho
      n_amostras_nao_finitas
                          quantas amostras não são finitas (NaN ou inf),
                          somadas sobre os canais. Ver abaixo por que ela é
                          contada aqui e não deixada para quem lê o número

    POR QUE A CONTAGEM DE NÃO-FINITOS ENTROU NAS MÉTRICAS. Uma única amostra
    NaN não fica onde está: o FIR de fase zero a espalha por toda a sua
    extensão, e o CAR espalha o canal para todos os outros. MEDIDO com sinal
    sintético de 19 canais a 128 Hz, UMA amostra NaN plantada no canal 3:
    depois de `preprocessar` sem CAR o canal 3 tinha 2.830 amostras não
    finitas; com CAR, os DEZENOVE canais tinham 2.830 cada. Nenhuma exceção foi
    levantada em momento algum.

    O que isso fazia no relatório é o motivo de a contagem existir: a média por
    canal virava `nan`, `nan < 0.01` é False em Python, mas o texto impresso
    saía como `nan  nan  OK  (residual 0.000%)` porque o residual era calculado
    de outro ramo. Ou seja, o QC assinava embaixo de um dado destruído. NaN
    silencioso é pior que exceção justamente por isso: ele vira número na
    tela.

    DUAS DECISÕES QUE MUDAM OS NÚMEROS, e por isso ficam declaradas:

    As bordas são aparadas antes de medir (`_aparar_bordas`). Filtro digital
    tem transitório nas pontas, e medir dentro dele mede o filtro, não o
    sinal.

    A faixa começa em 1 Hz e não em 0. O bin DC levaria a fração de delta a
    ~0,99 em qualquer sinal com offset, e a queda depois do passa-alta
    mediria a remoção do degrau, não a da deriva."""
    aparado = _aparar_bordas(raw)
    dados = aparado.get_data(picks="eeg")

    medias_uv = dados.mean(axis=1) * 1e6
    psd, freqs = _psd_medio(aparado)

    teto = min(45.0, aparado.info["sfreq"] / 2.0 * preproc_basico.FRACAO_NYQUIST_UTIL)
    # de 1 Hz, não de 0: o bin DC levaria a fração a ~0,99 em qualquer
    # sinal com offset e a queda mediria o degrau, não a deriva
    faixa_total = (freqs >= 1.0) & (freqs < teto)
    faixa_delta = (freqs >= 1.0) & (freqs < 4.0)
    faixa_alpha = (freqs >= 8.0) & (freqs < 13.0)

    potencia_total = float(psd[faixa_total].sum())
    potencia_delta = float(psd[faixa_delta].sum())
    # o sufixo _v2_hz na variável local não é redundância: é o passo em que a
    # unidade ainda é volt, e foi a ausência dessa marca que deixou o valor
    # em V²/Hz sair por uma chave documentada em µV²/Hz
    potencia_alpha_v2_hz = float(psd[faixa_alpha].sum())

    # contado por canal e somado: o que importa para o veredito é "existe
    # amostra não finita neste sinal", e a soma responde isso sem esconder o
    # tamanho do estrago (2.830 amostras num canal e 2.830 em dezenove são
    # diagnósticos bem diferentes)
    nao_finitas_por_canal = (~np.isfinite(dados)).sum(axis=1)

    return {
        "n_canais": dados.shape[0],
        "media_uv_abs_max": float(np.abs(medias_uv).max()),
        "media_uv_mediana": float(np.median(medias_uv)),
        # canal flat zeraria o denominador
        "fracao_delta": potencia_delta / potencia_total if potencia_total > 0 else None,
        "razao_rede_db": razao_pico_vizinhanca_db(psd, freqs, freq_rede) if freq_rede else None,
        # em µV²/Hz, como o nome diz: o MNE entrega volts, e 1e12 é o
        # (1e6)² que leva V² a µV². Ver a nota de unidade no cabeçalho
        "potencia_alpha_uv2_hz": potencia_alpha_v2_hz * 1e12,
        "n_amostras_nao_finitas": int(nao_finitas_por_canal.sum()),
    }


def comparar_antes_depois(raw_antes, raw_depois, freq_rede):
    """As métricas antes e depois, mais a razão de alfa preservado.

    A razão de alfa é a métrica de NÃO-DANO, e é a que a maioria dos
    relatórios esquece. Um filtro que zera a média e mata o ritmo alfa junto
    passa nas outras três e destruiu o dado. `razao_alpha` perto de 1
    significa que o alfa sobreviveu; bem abaixo de 1, que o filtro foi longe
    demais.

    Volta None quando o sinal de entrada não tinha alfa nenhum: dividir por
    zero daria infinito, e infinito lido como "preservou muito" seria a
    leitura exatamente invertida."""
    antes = metricas_de_raw(raw_antes, freq_rede)
    depois = metricas_de_raw(raw_depois, freq_rede)

    # a razão é adimensional: multiplicar as duas pontas por 1e12 (a conversão
    # de V²/Hz para µV²/Hz) não move este número em bit nenhum, e é por isso
    # que a correção de unidade não pode virar nem desvirar nenhum veredito
    razao_alpha = None
    if antes["potencia_alpha_uv2_hz"] > 0:
        razao_alpha = depois["potencia_alpha_uv2_hz"] / antes["potencia_alpha_uv2_hz"]

    return {"antes": antes, "depois": depois, "razao_alpha": razao_alpha}


def _fmt(valor, casas=3, sufixo=""):
    """None vira '-' e n/a vira 'n/a' — nunca 0.0. Um zero no lugar de
    'não medido' é indistinguível de 'notch perfeito' e inverteria o
    veredito da tabela.

    NaN e inf caem no mesmo 'n/a' pelo mesmo motivo: a formatação de um float
    NaN sai como a string `nan`, que alinhada numa coluna de números parece
    uma medida e não um buraco. Quem diz que houve NaN, e quantos, é a linha de
    ATENÇÃO no topo do bloco do sujeito."""
    if valor is None or not np.isfinite(valor):
        return "n/a"
    return f"{valor:.{casas}f}{sufixo}"


def _finito(*valores):
    """True só se TODOS os valores forem números finitos.

    None conta como não-finito de propósito: quem chama aqui está decidindo se
    pode escrever OK, e "não medi" não autoriza OK do mesmo jeito que NaN não
    autoriza. A distinção entre os dois continua viva onde ela importa — no
    veredito, que guarda None para não-medido e False para reprovado."""
    for v in valores:
        if v is None or not np.isfinite(v):
            return False
    return True


def _linhas_do_sujeito(nome, raw, comparacao, decisoes):
    """Monta o bloco de texto de um sujeito e devolve (linhas, veredito).

    O `veredito` é um dicionário com quatro chaves e TRÊS valores possíveis
    por chave: True (passou), False (reprovou) e None (não foi medido). A
    distinção entre False e None é o que permite ao resumo do fim dizer
    "reprovou" sem confundir com "não deu para medir".

    Os bool() na hora de montá-lo não são decorativos: veja o comentário
    junto do return."""
    antes, depois = comparacao["antes"], comparacao["depois"]
    freq_rede = decisoes["freq_rede"]
    diagnostico = decisoes["diagnostico_rede"] or {}

    linhas = []
    linhas.append("")
    linhas.append(f"=== {nome} ===")
    linhas.append(
        f"{antes['n_canais']} canais @ {raw.info['sfreq']:.1f} Hz, "
        f"{raw.n_times / raw.info['sfreq']:.1f}s"
    )

    if freq_rede is None:
        linhas.append(f"rede NÃO detectada ({diagnostico.get('motivo')}) — notch não aplicado")
    else:
        linhas.append(
            f"rede detectada: {freq_rede:.1f} Hz ({diagnostico.get('motivo')})"
            + (f", margem {diagnostico['margem_db']:.1f} dB" if diagnostico.get("margem_db") else "")
        )
        linhas.append(f"harmônicos notchados: {decisoes['harmonicos_notchados']}")

    # A linha de amostras não finitas vem ANTES das quatro métricas porque ela
    # explica o resto: com NaN dentro, os números abaixo saem "n/a" e os
    # vereditos saem FALHOU, e quem lê precisa saber que a causa é o dado e não
    # o filtro. Ela não entra no dicionário de veredito de propósito — o que
    # reprova o sujeito é a média, logo abaixo, e uma linha que imprime FALHOU
    # sem aparecer no resumo do fim é exatamente o bug que o comentário do
    # `return` desta função descreve.
    if antes["n_amostras_nao_finitas"] or depois["n_amostras_nao_finitas"]:
        linhas.append(
            f"ATENÇÃO: amostras não finitas (NaN/inf) — "
            f"antes {antes['n_amostras_nao_finitas']}, "
            f"depois {depois['n_amostras_nao_finitas']}. "
            f"Uma só amostra NaN se espalha pelo comprimento do FIR e, com CAR, "
            f"para todos os canais; as métricas abaixo não são interpretáveis."
        )

    linhas.append("")
    linhas.append(f"{'métrica':<30} {'antes':>12} {'depois':>12} {'veredito':>10}")

    if not _finito(antes["media_uv_abs_max"], depois["media_uv_abs_max"]):
        # Este ramo é o conserto do `nan  nan  OK  (residual 0.000%)`. Com média
        # NaN, `antes > 0` dava False, o residual caía no ramo do 0.0, e o
        # relatório imprimia o número mais tranquilizador possível — zero por
        # cento de resíduo — sobre um sinal que estava destruído. Não há
        # residual a calcular aqui: NaN é a resposta honesta, e _fmt o
        # transforma em "n/a".
        residual = float("nan")
    elif antes["media_uv_abs_max"] > 0:
        residual = depois["media_uv_abs_max"] / antes["media_uv_abs_max"]
    else:
        residual = 0.0
    ok_media = _finito(residual) and residual < MAX_FRACAO_MEDIA_RESIDUAL
    linhas.append(
        f"{'média |máx| por canal':<30} {_fmt(antes['media_uv_abs_max'], 2):>12} "
        f"{_fmt(depois['media_uv_abs_max'], 2):>12} {'OK' if ok_media else 'FALHOU':>10}"
        f"  (residual {_fmt(residual * 100, 3, '%')})"
    )

    # A fração delta é REPORTADA, não reprovada: o passa-alta corta em
    # 0,5 Hz, então ele só derruba delta quando a potência ali era deriva
    # vazando de baixo. Se a energia em 1-4 Hz for atividade delta real —
    # que é o caso do adhdata, onde o espectro acima de 1 Hz fica
    # idêntico antes e depois — o filtro está CERTO em não tocá-la, e
    # reprovar aqui seria exigir que ele destruísse sinal.
    # O que a fração delta diagnostica é o quanto sobrou de deriva; a
    # prova de que o DC saiu é a média por canal, logo acima.
    # `_finito` e não `is not None`: a fração delta é um quociente de somas do
    # espectro, e com NaN no sinal ela sai NaN em vez de sair None — o
    # denominador nunca chega a ser zero, então o ramo do None não protege daqui
    ok_delta = True if _finito(antes["fracao_delta"]) else None
    if _finito(antes["fracao_delta"], depois["fracao_delta"]):
        variacao_delta = depois["fracao_delta"] - antes["fracao_delta"]
    else:
        variacao_delta = None
    variacao = "" if variacao_delta is None else f"{variacao_delta:+.3f}"
    linhas.append(
        f"{'fração delta (1-4 Hz)':<30} {_fmt(antes['fracao_delta']):>12} "
        f"{_fmt(depois['fracao_delta']):>12} {variacao:>10}"
    )

    if not _finito(depois["razao_rede_db"]):
        # None (não houve rede a medir) e NaN (o sinal tinha buraco) viram os
        # dois "n/a", e n/a não é reprovação: nenhum dos dois é evidência de
        # que o notch falhou
        ok_rede = None
        veredito_rede = "n/a"
    else:
        ok_rede = depois["razao_rede_db"] < MAX_RAZAO_REDE_DB
        veredito_rede = "OK" if ok_rede else "FALHOU"
    rotulo_rede = f"razão rede @{freq_rede:.0f}Hz (dB)" if freq_rede else "razão rede (dB)"
    linhas.append(
        f"{rotulo_rede:<30} {_fmt(antes['razao_rede_db'], 2):>12} "
        f"{_fmt(depois['razao_rede_db'], 2):>12} {veredito_rede:>10}"
    )

    razao_alpha = comparacao["razao_alpha"]
    # com NaN na potência de entrada, a razão sai NaN, e toda comparação com
    # NaN é False — inclusive `NaN >= 0.9`. O veredito já sairia FALHOU por
    # acidente; `_finito` é o que o faz sair FALHOU por decisão
    ok_alpha = _finito(razao_alpha) and razao_alpha >= MIN_RAZAO_ALPHA
    linhas.append(
        f"{'potência alfa preservada':<30} {'1.000':>12} "
        f"{_fmt(razao_alpha):>12} {'OK' if ok_alpha else 'FALHOU':>10}"
    )

    # bool() em cada um NÃO é decorativo. As métricas saem do numpy, então
    # `ok_rede` é um numpy.bool_, e `numpy.bool_(False) is False` dá False.
    # Sem a conversão, o filtro de reprovadas em `relatorio_qc` não enxerga a
    # falha e o resumo final imprime "nenhuma métrica reprovada" enquanto a
    # linha da tabela, logo acima, imprime FALHOU. Foi exatamente o que
    # aconteceu com sub-NDARAC904DMU, e é o tipo de bug que só é descoberto
    # quando alguém confere a tabela contra o resumo.
    return linhas, {
        "media": None if ok_media is None else bool(ok_media),
        "delta": None if ok_delta is None else bool(ok_delta),
        "rede": None if ok_rede is None else bool(ok_rede),
        "alpha": None if ok_alpha is None else bool(ok_alpha),
    }


def _listar_entradas(caminho, subject_id):
    """As gravações a medir, como lista de (nome, caminho, subject_id).

    Aceita as duas formas de chamar o script:

      * um arquivo solto (.set ou .csv), e aí `subject_id` diz qual sujeito
        extrair de dentro dele — que é como o adhdata.csv funciona, com os
        121 sujeitos num arquivo só;
      * uma raiz BIDS, e aí varre `sub-*/eeg/*task-RestingState*.set` e
        ignora `subject_id`, porque cada arquivo já é de um sujeito.

    Levanta ValueError com o caminho no texto quando não acha nada. Devolver
    lista vazia faria o relatório sair com "0 sujeitos medidos" e veredito
    de aprovação, que é a pior saída possível: silêncio que parece sucesso."""
    caminho = Path(caminho)
    if caminho.is_file():
        return [(caminho.stem, caminho, subject_id)]

    if caminho.is_dir():
        entradas = []
        for pasta in sorted(p for p in caminho.iterdir() if p.is_dir() and p.name.startswith("sub-")):
            for arquivo in sorted((pasta / "eeg").glob("*task-RestingState*.set")):
                entradas.append((pasta.name, arquivo, None))
        if not entradas:
            raise ValueError(f"nenhum .set de RestingState encontrado em: {caminho}")
        return entradas

    raise ValueError(f"caminho não encontrado: {caminho}")


def relatorio_qc(caminho, caminho_saida, subject_id=None):
    """Mede antes e depois de cada sujeito, grava o relatório e devolve os
    resumos.

    O relatório é gravado em `caminho_saida` (UTF-8, criando a pasta se
    preciso) e a mesma informação volta como lista de dicionários, um por
    sujeito, com `decisoes`, `comparacao` e `veredito`.

    O QUE ESTE RELATÓRIO MEDE, E O QUE ELE NÃO MEDE: ele chama
    `preproc_basico.preprocessar` com o padrão do app, ou seja, passa-alta
    mais notch e SEM passa-baixa, porque é isso que o app aplica no traçado
    que o usuário vê. As figuras do caderno aplicam as quatro etapas da
    folha, passa-baixa incluído, e por isso chegam a uma razão de rede bem
    menor no MESMO sujeito. Os dois números estão certos e medem coisas
    diferentes; comparar um com o outro sem essa frase leva à conclusão
    errada de que um dos dois está com bug."""
    entradas = _listar_entradas(caminho, subject_id)

    linhas = [f"[qc_relatorio] {caminho} — {len(entradas)} sujeito(s)"]
    resumos = []
    contagem = {"media": 0, "delta": 0, "rede": 0, "alpha": 0}
    falhas = []

    for nome, arquivo, sujeito in entradas:
        raw = preproc_basico.carregar_raw(arquivo, sujeito)
        filtrado, decisoes = preproc_basico.preprocessar(raw)
        comparacao = comparar_antes_depois(raw, filtrado, decisoes["freq_rede"])

        novas, veredito = _linhas_do_sujeito(nome, raw, comparacao, decisoes)
        linhas.extend(novas)

        for chave, ok in veredito.items():
            if ok:
                contagem[chave] += 1
        # `not ok` em vez de `ok is False`: métrica não medida vale None e
        # não é reprovação, mas False de qualquer tipo é
        reprovadas = [c for c, ok in veredito.items() if ok is not None and not ok]
        if reprovadas:
            falhas.append((nome, reprovadas))

        resumos.append({"sujeito": nome, "decisoes": decisoes, "comparacao": comparacao,
                        "veredito": veredito})

    n = len(entradas)
    linhas.append("")
    linhas.append("--- veredito ---")
    linhas.append(f"sujeitos medidos: {n}")
    linhas.append(f"média por canal zerada:      {contagem['media']}/{n}")
    linhas.append(f"rede removida (<{MAX_RAZAO_REDE_DB:.0f} dB):      {contagem['rede']}/{n}")
    linhas.append(f"potência alfa preservada:    {contagem['alpha']}/{n}")
    linhas.append(
        "(a fração delta é reportada, não reprovada — o passa-alta corta em 0,5 Hz e "
        "só derruba delta quando havia deriva vazando de baixo)"
    )

    redes = {}
    for r in resumos:
        f = r["decisoes"]["freq_rede"]
        redes[f] = redes.get(f, 0) + 1
    linhas.append(f"frequências de rede detectadas: { {k: v for k, v in redes.items()} }")

    if falhas:
        linhas.append("")
        linhas.append("sujeitos com métrica reprovada:")
        for nome, reprovadas in falhas:
            linhas.append(f"  {nome:<24} {', '.join(reprovadas)}")
    else:
        linhas.append("nenhuma métrica reprovada")

    texto = "\n".join(linhas)

    caminho_saida = Path(caminho_saida)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    # encoding explícito: o cp1252 default do Windows engasga em "fração"
    caminho_saida.write_text(texto + "\n", encoding="utf-8")

    print(texto)
    print(f"\n[qc_relatorio] relatório gravado em {caminho_saida}")
    return resumos


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("uso: python scripts/qc_relatorio.py <arquivo|pasta_bids> [subject_id]")
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config

    caminho = sys.argv[1]
    sujeito = sys.argv[2] if len(sys.argv) > 2 else None
    saida = config.caminho_relatorios() / "qc_relatorio.txt"
    relatorio_qc(caminho, saida, sujeito)
