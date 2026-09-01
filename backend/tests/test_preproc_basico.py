import sys
from pathlib import Path

import mne
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from preproc_basico import (
    PISO_AC_UV,
    aplicar_base,
    aplicar_car,
    bases_disponiveis,
    diagnosticar_referencia_fisica,
    aplicar_passa_baixa,
    aplicar_notch,
    aplicar_passa_alta,
    carregar_raw,
    detectar_frequencia_rede,
    preprocessar,
    razao_pico_vizinhanca_db,
    _psd_medio,
)


def _raw_sintetico(fs=500.0, dur=60.0, offset_uv=140.0, f_rede=60.0,
                   amp_rede_uv=6.0, amp_alpha_uv=8.0, amp_deriva_uv=400.0,
                   f_deriva=0.15, n_canais=4, semente=0):
    """Sinal com os problemas do README embutidos e conhecidos: offset DC,
    deriva lenta, pico de rede e um ritmo alfa a preservar. Em volts, que
    é a unidade que o MNE assume.

    Os três parâmetros são calibrados para reproduzir os números medidos,
    e cada um resolve uma armadilha:

    - A deriva fica em 0,15 Hz, ABAIXO do corte de 0,5 Hz. Offset DC
      sozinho não serve (o PSD começa em 1 Hz e um degrau constante some
      da medida), e deriva acima do corte também não (o filtro a preserva
      legitimamente, e com razão). É a deriva sub-corte que vaza para
      dentro de delta por leakage espectral — o sintoma que o README mede
      como delta absorvendo 47-72% da potência.

    - A deriva é grande (400 µV) e o alfa modesto (8 µV) porque é essa a
      proporção real: com alfa dominante, delta ficaria em 0,2% e a queda
      seria invisível.

    - A rede fica em 6 µV, não 80: o README mede a rede entre +6 e +16 dB
      acima do pico alfa. Um tom puro de 80 µV sobre ruído de 2 µV daria
      +57 dB — cenário que nenhum EEG real apresenta, e que faria o teste
      medir a cauda do filtro em vez da rede.

    A duração de 60 s também é escolhida, não acidental: a deriva de
    0,15 Hz tem período de 6,7 s, e em 30 s cabem só ~4,5 ciclos. O ciclo
    incompleto deixa média residual de ~5 µV depois do filtro — que
    pareceria falha do passa-alta e é só o fixture curto demais."""
    rng = np.random.RandomState(semente)
    t = np.arange(int(fs * dur)) / fs

    dados = np.zeros((n_canais, len(t)))
    for i in range(n_canais):
        sinal = offset_uv + amp_alpha_uv * np.sin(2 * np.pi * 10.0 * t + i)
        if amp_deriva_uv:
            sinal = sinal + amp_deriva_uv * np.sin(2 * np.pi * f_deriva * t + i)
        if amp_rede_uv:
            sinal = sinal + amp_rede_uv * np.sin(2 * np.pi * f_rede * t)
        sinal = sinal + rng.randn(len(t)) * 2.0
        dados[i] = sinal * 1e-6

    info = mne.create_info([f"E{i + 1}" for i in range(n_canais)], fs, "eeg", verbose=False)
    return mne.io.RawArray(dados, info, verbose=False)


def _media_uv(raw):
    return np.abs(raw.get_data().mean(axis=1)).max() * 1e6


def _potencia_faixa(raw, lo, hi):
    psd, freqs = _psd_medio(raw)
    dentro = (freqs >= lo) & (freqs < hi)
    return float(psd[dentro].sum())


# --- detecção da rede ---------------------------------------------------

def test_detecta_50_hz():
    raw = _raw_sintetico(fs=500.0, f_rede=50.0)
    freq, diagnostico = detectar_frequencia_rede(raw)
    assert freq == 50.0
    assert diagnostico["motivo"] == "detectada"


def test_detecta_60_hz():
    """O caso do HBN — gravado em Nova York, rede de 60 Hz."""
    raw = _raw_sintetico(fs=500.0, f_rede=60.0)
    freq, diagnostico = detectar_frequencia_rede(raw)
    assert freq == 60.0
    assert diagnostico["motivo"] == "detectada"
    assert diagnostico["margem_db"] > 3.0


def test_detecta_none_sem_pico_de_rede():
    """Dado já notchado na origem: notchar às cegas removeria sinal
    neural de graça."""
    raw = _raw_sintetico(fs=500.0, amp_rede_uv=0.0)
    freq, diagnostico = detectar_frequencia_rede(raw)
    assert freq is None
    assert diagnostico["motivo"] == "sem_pico_de_rede"


def test_detecta_none_quando_nyquist_nao_cobre():
    raw = _raw_sintetico(fs=100.0, f_rede=0.0, amp_rede_uv=0.0)
    freq, diagnostico = detectar_frequencia_rede(raw)
    assert freq is None
    assert diagnostico["motivo"] == "nyquist_insuficiente"
    assert diagnostico["candidatas_viaveis"] == []


def test_detecta_marca_candidata_unica_por_nyquist():
    """A 128 Hz (adhdata) o teto de Nyquist exclui 60 Hz antes de medir:
    50 vence por eliminação, e isso precisa ficar registrado."""
    raw = _raw_sintetico(fs=128.0, f_rede=50.0)
    freq, diagnostico = detectar_frequencia_rede(raw)
    assert freq == 50.0
    assert diagnostico["motivo"] == "candidata_unica_por_nyquist"
    assert diagnostico["candidatas_viaveis"] == [50.0]


def test_detecta_none_quando_ambiguo():
    """Dois picos igualmente fortes: falhar em voz alta é melhor que
    escolher no cara-ou-coroa."""
    raw = _raw_sintetico(fs=500.0, f_rede=50.0)
    dados = raw.get_data()
    t = np.arange(dados.shape[1]) / 500.0
    dados = dados + 6e-6 * np.sin(2 * np.pi * 60.0 * t)
    ambiguo = mne.io.RawArray(dados, raw.info, verbose=False)

    freq, diagnostico = detectar_frequencia_rede(ambiguo)
    assert freq is None
    assert diagnostico["motivo"] == "ambiguo"


def test_detecta_none_em_sinal_sem_conteudo_ac():
    """DC puro não é "rede de 50 Hz". MEDIDO antes do conserto: 19 canais
    constantes em 50 µV a 128 Hz saíam daqui como 50,0 Hz com razão de
    7,071 dB contra o limiar de 3,0 — e o relatório de QC imprimia
    "rede detectada: 50,0 Hz" embaixo. O "pico" era o último bit da
    mantissa do float: o desvio padrão desse sinal é 1,4e-14 µV.

    A razão pico/vizinhança é um quociente, e quociente entre dois ruídos
    de arredondamento continua devolvendo um número apresentável — por isso
    a checagem tem de vir ANTES da medida, e não depois."""
    dados = np.full((19, int(128 * 60)), 50e-6)
    info = mne.create_info([f"c{i}" for i in range(19)], 128.0, "eeg", verbose=False)
    raw = mne.io.RawArray(dados, info, verbose=False)

    freq, diagnostico = detectar_frequencia_rede(raw)
    assert freq is None
    assert diagnostico["motivo"] == "sinal_sem_conteudo_ac"
    assert diagnostico["desvio_padrao_uv"] < PISO_AC_UV


def test_piso_de_ac_nao_barra_sinal_fraco_de_verdade():
    """O contrapeso do teste acima: o piso existe para pegar sinal MORTO, e
    não sinal pequeno. 1 µV RMS é fraco para EEG (o normal é dezenas) e
    ainda assim fica mil vezes acima do piso — a rede continua sendo
    detectada nele."""
    t = np.arange(int(128 * 60)) / 128.0
    rng = np.random.default_rng(3)
    dados = rng.normal(0, 1e-6, (19, len(t))) + 0.6e-6 * np.sin(2 * np.pi * 50.0 * t)
    info = mne.create_info([f"c{i}" for i in range(19)], 128.0, "eeg", verbose=False)
    raw = mne.io.RawArray(dados, info, verbose=False)

    freq, diagnostico = detectar_frequencia_rede(raw)
    assert diagnostico["desvio_padrao_uv"] > PISO_AC_UV
    assert freq == 50.0


def test_um_canal_vivo_entre_mortos_ainda_e_medido():
    """O desvio olhado é o MAIOR entre canais, não a média deles: dezoito
    canais mortos não podem impedir a medida do décimo nono. Recusar aqui
    seria trocar um falso positivo por um falso negativo."""
    t = np.arange(int(128 * 60)) / 128.0
    rng = np.random.default_rng(4)
    dados = np.full((19, len(t)), 50e-6)
    dados[7] = rng.normal(0, 20e-6, len(t)) + 6e-6 * np.sin(2 * np.pi * 50.0 * t)
    info = mne.create_info([f"c{i}" for i in range(19)], 128.0, "eeg", verbose=False)
    raw = mne.io.RawArray(dados, info, verbose=False)

    freq, diagnostico = detectar_frequencia_rede(raw)
    assert diagnostico["motivo"] != "sinal_sem_conteudo_ac"
    assert freq == 50.0


def test_detecta_candidata_unica_sem_pico_nao_aceita():
    """Única candidata viável ainda precisa ter pico de verdade."""
    raw = _raw_sintetico(fs=128.0, amp_rede_uv=0.0)
    freq, diagnostico = detectar_frequencia_rede(raw)
    assert freq is None
    assert diagnostico["motivo"] == "sem_pico_de_rede"


# --- passa-alta ---------------------------------------------------------

def test_passa_alta_zera_a_media():
    raw = _raw_sintetico(offset_uv=140.0)
    assert _media_uv(raw) > 100.0
    assert _media_uv(aplicar_passa_alta(raw)) < 1.0


def test_passa_alta_preserva_potencia_alfa():
    """O teste que a maioria esquece: o filtro tem que remover deriva SEM
    mexer no que interessa."""
    raw = _raw_sintetico(offset_uv=140.0)
    antes = _potencia_faixa(raw, 8.0, 13.0)
    depois = _potencia_faixa(aplicar_passa_alta(raw), 8.0, 13.0)
    assert depois / antes == pytest.approx(1.0, abs=0.1)


def test_passa_alta_derruba_fracao_delta():
    """A deriva de 0,2 Hz vaza para dentro de delta; o passa-alta tem que
    tirá-la de lá. É o sintoma que o README mede como delta absorvendo
    47-72% da potência total."""
    raw = _raw_sintetico(offset_uv=140.0)
    filtrado = aplicar_passa_alta(raw)
    fracao_antes = _potencia_faixa(raw, 1.0, 4.0) / _potencia_faixa(raw, 1.0, 45.0)
    fracao_depois = _potencia_faixa(filtrado, 1.0, 4.0) / _potencia_faixa(filtrado, 1.0, 45.0)
    assert fracao_antes > 0.4
    assert fracao_depois < fracao_antes / 2


def test_passa_alta_nao_modifica_o_original():
    """raw.filter() age in place — sem a cópia, o relatório compararia o
    sinal filtrado consigo mesmo e reportaria delta zero em tudo."""
    raw = _raw_sintetico(offset_uv=140.0)
    media_antes = _media_uv(raw)
    aplicar_passa_alta(raw)
    assert _media_uv(raw) == pytest.approx(media_antes)


# --- notch --------------------------------------------------------------

def test_notch_zera_razao_pico_vizinhanca():
    raw = _raw_sintetico(fs=500.0, f_rede=60.0)
    psd, freqs = _psd_medio(raw)
    assert razao_pico_vizinhanca_db(psd, freqs, 60.0) > 6.0

    filtrado, harmonicos = aplicar_notch(raw, 60.0)
    psd, freqs = _psd_medio(filtrado)
    assert razao_pico_vizinhanca_db(psd, freqs, 60.0) < 3.0
    assert 60.0 in harmonicos


def test_notch_preserva_potencia_alfa():
    raw = _raw_sintetico(fs=500.0, f_rede=60.0)
    antes = _potencia_faixa(raw, 8.0, 13.0)
    filtrado, _ = aplicar_notch(raw, 60.0)
    assert _potencia_faixa(filtrado, 8.0, 13.0) / antes == pytest.approx(1.0, abs=0.1)


def test_notch_freq_none_devolve_copia_intacta():
    raw = _raw_sintetico()
    saida, harmonicos = aplicar_notch(raw, None)
    assert harmonicos == []
    assert np.allclose(saida.get_data(), raw.get_data())


def test_notch_so_inclui_harmonicos_abaixo_da_nyquist():
    """A 128 Hz sobra só o fundamental de 50; a 500 Hz cabem 60/120/180."""
    _, harmonicos_lento = aplicar_notch(_raw_sintetico(fs=128.0, f_rede=50.0), 50.0)
    assert harmonicos_lento == [50.0]

    _, harmonicos_rapido = aplicar_notch(_raw_sintetico(fs=500.0, f_rede=60.0), 60.0)
    assert harmonicos_rapido == [60.0, 120.0, 180.0]


# --- orquestração -------------------------------------------------------

def test_preprocessar_registra_ordem_e_decisoes():
    raw = _raw_sintetico(fs=500.0, f_rede=60.0)
    filtrado, decisoes = preprocessar(raw)

    assert decisoes["ordem"] == ["passa-alta", "notch"]
    assert decisoes["freq_rede"] == 60.0
    assert decisoes["l_freq"] == 0.5
    assert decisoes["diagnostico_rede"]["motivo"] == "detectada"
    assert _media_uv(filtrado) < 1.0


def test_preprocessar_nao_modifica_o_original():
    raw = _raw_sintetico(offset_uv=140.0, f_rede=60.0)
    dados_antes = raw.get_data().copy()
    preprocessar(raw)
    assert np.allclose(raw.get_data(), dados_antes)


def test_preprocessar_sem_rede_detectada_pula_notch():
    raw = _raw_sintetico(fs=500.0, amp_rede_uv=0.0)
    _filtrado, decisoes = preprocessar(raw)
    assert decisoes["freq_rede"] is None
    assert decisoes["harmonicos_notchados"] == []


def test_preprocessar_aceita_freq_rede_explicita():
    raw = _raw_sintetico(fs=500.0, f_rede=60.0)
    _filtrado, decisoes = preprocessar(raw, freq_rede=50.0)
    assert decisoes["freq_rede"] == 50.0
    assert decisoes["diagnostico_rede"] is None


# --- carregamento -------------------------------------------------------

def test_carregar_raw_extensao_desconhecida(tmp_path):
    caminho = tmp_path / "sinal.edf"
    caminho.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="extensão não suportada"):
        carregar_raw(caminho)


def test_carregar_raw_arquivo_inexistente(tmp_path):
    with pytest.raises(ValueError, match="arquivo não encontrado"):
        carregar_raw(tmp_path / "nao_existe.set")


def test_carregar_raw_csv_sem_subject_id(tmp_path):
    caminho = tmp_path / "adhdata.csv"
    caminho.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="exige subject_id"):
        carregar_raw(caminho)


def test_carregar_raw_csv_converte_para_volts(tmp_path):
    """O adhdata está em µV e o MNE assume volts — sem a conversão os
    números do relatório sairiam 1e6 vezes errados."""
    import csv_data

    caminho = tmp_path / "adhdata.csv"
    cabecalho = ",".join(list(csv_data.CANAIS_19) + ["ID", "Class"])
    linhas = [cabecalho]
    for _ in range(256):
        linhas.append(",".join(["100.0"] * len(csv_data.CANAIS_19) + ["v1", "ADHD"]))
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")

    raw = carregar_raw(caminho, "v1")
    assert raw.info["sfreq"] == csv_data.FS
    assert len(raw.ch_names) == len(csv_data.CANAIS_19)
    # 100 µV no CSV -> 100e-6 V no Raw -> 100 µV de volta no relatório
    assert raw.get_data().mean() * 1e6 == pytest.approx(100.0)


# --- razão pico/vizinhança ---------------------------------------------

def test_razao_pico_vizinhanca_none_em_canal_flat():
    """log10(0) = -inf polui tabela e veredito."""
    freqs = np.arange(1.0, 100.0, 0.25)
    assert razao_pico_vizinhanca_db(np.zeros_like(freqs), freqs, 50.0) is None


def test_razao_pico_vizinhanca_none_fora_do_espectro():
    freqs = np.arange(1.0, 40.0, 0.25)
    assert razao_pico_vizinhanca_db(np.ones_like(freqs), freqs, 60.0) is None


# --- passa-baixa --------------------------------------------------------

def test_passa_baixa_derruba_acima_do_corte():
    """Um tom de 80 Hz esta acima do corte de 45 Hz e tem que sumir."""
    raw = _raw_sintetico(fs=500.0, f_rede=80.0, amp_rede_uv=20.0)
    antes = _potencia_faixa(raw, 70.0, 90.0)
    depois = _potencia_faixa(aplicar_passa_baixa(raw, h_freq=45.0), 70.0, 90.0)
    assert depois < antes / 100


def test_passa_baixa_preserva_potencia_alfa():
    raw = _raw_sintetico(fs=500.0)
    antes = _potencia_faixa(raw, 8.0, 13.0)
    depois = _potencia_faixa(aplicar_passa_baixa(raw), 8.0, 13.0)
    assert depois / antes == pytest.approx(1.0, abs=0.1)


def test_passa_baixa_nao_modifica_o_original():
    raw = _raw_sintetico(fs=500.0, f_rede=80.0, amp_rede_uv=20.0)
    antes = _potencia_faixa(raw, 70.0, 90.0)
    aplicar_passa_baixa(raw)
    assert _potencia_faixa(raw, 70.0, 90.0) == pytest.approx(antes)


def test_passa_baixa_nao_substitui_o_notch():
    """Trava um resultado contraintuitivo e fácil de errar: um passa-baixa
    em 45 Hz NÃO remove a rede de 50 Hz, porque 50 fica a 5 Hz do corte,
    ainda dentro da banda de transição do FIR. Medido: 4,5 dB de queda
    pelo passa-baixa contra 43,2 dB pelo notch. Se este teste passar a
    falhar, é porque alguém mudou o corte ou a largura de transição, e a
    conclusão do relatório precisa ser refeita junto."""
    raw = _raw_sintetico(fs=500.0, f_rede=50.0, amp_rede_uv=20.0)

    def potencia_em_50(r):
        psd, freqs = _psd_medio(r)
        return float(psd[(freqs >= 49.5) & (freqs <= 50.5)].max())

    bruto = potencia_em_50(raw)
    queda_passa_baixa = 10 * np.log10(bruto / potencia_em_50(aplicar_passa_baixa(raw, h_freq=45.0)))
    queda_notch = 10 * np.log10(bruto / potencia_em_50(aplicar_notch(raw, 50.0)[0]))

    assert queda_passa_baixa < 10.0     # mal encosta na rede
    assert queda_notch > 30.0           # o notch é quem a remove
    assert queda_notch > queda_passa_baixa * 3


# --- CAR ----------------------------------------------------------------

def _media_instantanea_uv(raw):
    """Desvio da media entre canais, amostra a amostra. O CAR tem que
    zerar isto por construcao."""
    return float(raw.get_data().mean(axis=0).std() * 1e6)


def test_car_zera_a_media_instantanea_entre_canais():
    raw = _raw_sintetico(fs=500.0)
    assert _media_instantanea_uv(raw) > 1.0
    assert _media_instantanea_uv(aplicar_car(raw)) == pytest.approx(0.0, abs=1e-9)


def test_car_tira_o_canal_de_referencia_do_zero():
    """O caso do Cz no HBN: ele e a referencia fisica, entao vem
    identicamente zero no dado bruto. Depois do CAR passa a ter sinal."""
    raw = _raw_sintetico(fs=500.0, n_canais=4)
    dados = raw.get_data()
    dados[3] = 0.0  # o canal de referencia
    ref = mne.io.RawArray(dados, raw.info, verbose=False)

    assert ref.get_data()[3].std() == 0.0
    assert aplicar_car(ref).get_data()[3].std() > 0.0


def test_car_nao_modifica_o_original():
    raw = _raw_sintetico(fs=500.0)
    antes = _media_instantanea_uv(raw)
    aplicar_car(raw)
    assert _media_instantanea_uv(raw) == pytest.approx(antes)


# --- orquestracao com as etapas novas -----------------------------------

def test_preprocessar_registra_a_ordem_completa():
    raw = _raw_sintetico(fs=500.0, f_rede=60.0)
    _filtrado, decisoes = preprocessar(raw, h_freq=45.0, car=True)
    assert decisoes["ordem"] == ["passa-alta", "passa-baixa", "notch", "car"]
    assert decisoes["h_freq"] == 45.0
    assert decisoes["car"] is True


def test_preprocessar_pula_etapas_opcionais_por_padrao():
    """h_freq=None e car=False mantem o comportamento anterior, para o
    relatorio conseguir medir cada etapa isolada."""
    raw = _raw_sintetico(fs=500.0, f_rede=60.0)
    _filtrado, decisoes = preprocessar(raw)
    assert decisoes["ordem"] == ["passa-alta", "notch"]


def test_preprocessar_completo_zera_media_instantanea():
    raw = _raw_sintetico(fs=500.0, f_rede=60.0)
    filtrado, _ = preprocessar(raw, h_freq=45.0, car=True)
    assert _media_instantanea_uv(filtrado) == pytest.approx(0.0, abs=1e-9)


# --- troca de base (re-referenciamento parametrizado) -------------------
#
# Missao 1 da folha de 01/09/2026: "Troca de Base - ouvido". A base deixa de
# ser decidida pelo booleano car=True/False e passa a ser NOMEADA, porque o
# app precisa oferecer a escolha ao usuario e refazer o pre-processamento
# conforme ela.

def _raw_com_orelhas(fs=128.0, n_canais=4):
    """Sinal com dois canais de orelha (A1/A2) alem dos de analise, que e o
    que o adhdata tem e o HBN nao."""
    raw = _raw_sintetico(fs=fs, n_canais=n_canais)
    dados = raw.get_data()
    nomes = [f"E{i + 1}" for i in range(n_canais)] + ["A1", "A2"]
    # as orelhas carregam um offset comum, que e o que a re-referencia remove
    orelhas = np.vstack([dados[0] * 0.1 + 30e-6, dados[0] * 0.1 + 26e-6])
    info = mne.create_info(nomes, fs, "eeg", verbose=False)
    return mne.io.RawArray(np.vstack([dados, orelhas]), info, verbose=False)


def test_bases_disponiveis_lista_orelha_so_quando_a1_a2_existem():
    """A rede GSN-HydroCel do HBN nao tem A1/A2. Oferecer 'orelha' ali seria
    oferecer um botao que so pode falhar."""
    com = bases_disponiveis(_raw_com_orelhas().ch_names)
    sem = bases_disponiveis(_raw_sintetico(fs=500.0).ch_names)

    assert "orelha" in com
    assert "orelha" not in sem
    # nativa e car nao dependem de canal nenhum: valem em qualquer banco
    for base in ("nativa", "car"):
        assert base in com and base in sem


def test_aplicar_base_car_zera_a_media_instantanea():
    raw = _raw_sintetico(fs=500.0)
    saida, info = aplicar_base(raw, "car")
    assert _media_instantanea_uv(saida) == pytest.approx(0.0, abs=1e-9)
    assert info["base"] == "car"


def test_aplicar_base_nativa_devolve_copia_intacta():
    """'nativa' e a ausencia de re-referencia, nao uma re-referencia para a
    referencia declarada: o sinal ja esta nela."""
    raw = _raw_sintetico(fs=500.0)
    saida, info = aplicar_base(raw, "nativa")
    assert np.allclose(saida.get_data(), raw.get_data())
    assert info["base"] == "nativa"


def test_aplicar_base_orelha_remove_o_que_e_comum_as_orelhas():
    raw = _raw_com_orelhas()
    saida, info = aplicar_base(raw, "orelha")
    assert info["canais_referencia"] == ["A1", "A2"]
    # a media das duas orelhas, depois da re-referencia, e zero por construcao
    dados = saida.get_data()
    idx = [saida.ch_names.index(c) for c in ("A1", "A2")]
    assert np.abs(dados[idx].mean(axis=0)).max() == pytest.approx(0.0, abs=1e-12)


def test_aplicar_base_orelha_recusa_banco_sem_orelhas():
    """Falha declarada, e nao silenciosa: sem A1/A2 nao ha o que fazer."""
    raw = _raw_sintetico(fs=500.0)
    with pytest.raises(ValueError, match="orelha"):
        aplicar_base(raw, "orelha")


def test_aplicar_base_cz_recusa_banco_sem_cz():
    raw = _raw_sintetico(fs=500.0)
    with pytest.raises(ValueError, match="Cz"):
        aplicar_base(raw, "cz")


def test_aplicar_base_desconhecida_recusa():
    raw = _raw_sintetico(fs=500.0)
    with pytest.raises(ValueError, match="base desconhecida"):
        aplicar_base(raw, "mastoide_esquerda")


def test_aplicar_base_nao_modifica_o_original():
    raw = _raw_sintetico(fs=500.0)
    antes = _media_instantanea_uv(raw)
    aplicar_base(raw, "car")
    assert _media_instantanea_uv(raw) == pytest.approx(antes)


def test_preprocessar_com_base_registra_a_ordem_e_a_base():
    """A base entra na MESMA posicao que o CAR ocupava: por ultimo, depois
    dos filtros."""
    raw = _raw_sintetico(fs=500.0, f_rede=60.0)
    _filtrado, decisoes = preprocessar(raw, h_freq=45.0, base="car")
    assert decisoes["ordem"] == ["passa-alta", "passa-baixa", "notch", "car"]
    assert decisoes["base"] == "car"


def test_preprocessar_base_nativa_nao_acrescenta_etapa():
    raw = _raw_sintetico(fs=500.0, f_rede=60.0)
    _filtrado, decisoes = preprocessar(raw, base="nativa")
    assert decisoes["ordem"] == ["passa-alta", "notch"]
    assert decisoes["base"] == "nativa"


def test_preprocessar_car_true_continua_funcionando():
    """Compatibilidade: car=True e o mesmo que base='car'. O qc_relatorio e as
    figuras do caderno chamam assim, e quebra-los para renomear um parametro
    seria trocar trabalho entregue por estetica."""
    raw = _raw_sintetico(fs=500.0, f_rede=60.0)
    _f1, d1 = preprocessar(raw, h_freq=45.0, car=True)
    _f2, d2 = preprocessar(raw, h_freq=45.0, base="car")
    assert d1["ordem"] == d2["ordem"]
    assert d1["base"] == d2["base"] == "car"
    assert d1["car"] is True


def test_preprocessar_recusa_car_e_base_em_conflito():
    """car=True com base='nativa' e um pedido contraditorio. Escolher um lado
    em silencio seria decidir pelo chamador."""
    raw = _raw_sintetico(fs=500.0, f_rede=60.0)
    with pytest.raises(ValueError, match="conflito"):
        preprocessar(raw, car=True, base="nativa")


# --- tratamento do Cz ---------------------------------------------------
#
# "θ Tratamento · Cz" na folha. O Cz do HBN e a referencia fisica: vem
# identicamente zero no bruto e ganha sinal depois do CAR. Ate aqui isso era
# comportamento emergente do MNE; passa a ser decisao declarada e medida.

def _raw_com_cz_de_referencia(fs=500.0):
    raw = _raw_sintetico(fs=fs, n_canais=4)
    dados = raw.get_data()
    dados[3] = 0.0
    nomes = ["Fz", "Pz", "C3", "Cz"]
    info = mne.create_info(nomes, fs, "eeg", verbose=False)
    return mne.io.RawArray(dados, info, verbose=False)


def test_diagnosticar_referencia_fisica_encontra_o_cz_zerado():
    achados = diagnosticar_referencia_fisica(_raw_com_cz_de_referencia())
    assert achados["canais_flat"] == ["Cz"]
    assert achados["referencia_fisica_provavel"] == "Cz"


def test_diagnosticar_referencia_fisica_nao_inventa_quando_nao_ha():
    achados = diagnosticar_referencia_fisica(_raw_sintetico(fs=500.0))
    assert achados["canais_flat"] == []
    assert achados["referencia_fisica_provavel"] is None


def test_preprocessar_declara_o_cz_recuperado_pelo_car():
    """O numero que o caderno reporta: desvio 0,000 antes, > 0 depois."""
    raw = _raw_com_cz_de_referencia()
    _filtrado, decisoes = preprocessar(raw, h_freq=45.0, base="car")

    cz = decisoes["referencia_fisica"]
    assert cz["referencia_fisica_provavel"] == "Cz"
    assert cz["recuperada_pela_base"] is True


def test_preprocessar_nao_declara_recuperacao_quando_a_base_e_nativa():
    """Sem re-referenciar, o Cz continua morto — e a decisao tem de dizer
    isso, porque um Cz plano no tracado e plausivel e errado."""
    raw = _raw_com_cz_de_referencia()
    _filtrado, decisoes = preprocessar(raw, base="nativa")

    cz = decisoes["referencia_fisica"]
    assert cz["referencia_fisica_provavel"] == "Cz"
    assert cz["recuperada_pela_base"] is False
