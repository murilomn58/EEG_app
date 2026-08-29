import sys
from pathlib import Path

import mne
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from qc_relatorio import (
    comparar_antes_depois,
    metricas_de_raw,
    relatorio_qc,
    _fmt,
    _listar_entradas,
)

from test_preproc_basico import _raw_sintetico


def _raw_constante(valor_uv, fs=500.0, dur=20.0, n_canais=3):
    """Sinal sem nada além de um degrau — para checar a métrica de média
    contra um número conhecido de antemão."""
    dados = np.full((n_canais, int(fs * dur)), valor_uv * 1e-6)
    info = mne.create_info([f"E{i + 1}" for i in range(n_canais)], fs, "eeg", verbose=False)
    return mne.io.RawArray(dados, info, verbose=False)


def _raw_seno(freq, fs=500.0, dur=20.0, amp_uv=50.0, n_canais=3):
    t = np.arange(int(fs * dur)) / fs
    dados = np.tile(amp_uv * 1e-6 * np.sin(2 * np.pi * freq * t), (n_canais, 1))
    info = mne.create_info([f"E{i + 1}" for i in range(n_canais)], fs, "eeg", verbose=False)
    return mne.io.RawArray(dados, info, verbose=False)


# --- métricas -----------------------------------------------------------

def test_media_reproduz_offset_conhecido():
    """O número que o README mede como +130 a +145."""
    metricas = metricas_de_raw(_raw_constante(140.0), freq_rede=None)
    assert metricas["media_uv_abs_max"] == pytest.approx(140.0, abs=0.5)


def test_fracao_delta_de_sinal_puramente_delta():
    """Seno de 2 Hz: quase toda a potência dentro de [1,4)."""
    metricas = metricas_de_raw(_raw_seno(2.0), freq_rede=None)
    assert metricas["fracao_delta"] > 0.9


def test_fracao_delta_de_sinal_puramente_alfa():
    metricas = metricas_de_raw(_raw_seno(10.0), freq_rede=None)
    assert metricas["fracao_delta"] < 0.1


def test_razao_rede_de_sinal_sem_rede():
    metricas = metricas_de_raw(_raw_sintetico(amp_rede_uv=0.0), freq_rede=60.0)
    assert metricas["razao_rede_db"] < 3.0


def test_razao_rede_none_quando_freq_none():
    """n/a e 0.0 não são a mesma coisa: zero seria lido como notch
    perfeito e inverteria o veredito."""
    metricas = metricas_de_raw(_raw_sintetico(), freq_rede=None)
    assert metricas["razao_rede_db"] is None


def test_metricas_de_canal_flat_nao_levanta():
    """log10(0) e divisão por zero em canal sem sinal nenhum."""
    metricas = metricas_de_raw(_raw_constante(0.0), freq_rede=60.0)
    assert metricas["media_uv_abs_max"] == pytest.approx(0.0, abs=1e-6)
    assert metricas["razao_rede_db"] is None


def test_metricas_gravacao_curta_nao_levanta():
    """Curta demais para aparar as bordas — mede assim mesmo."""
    metricas = metricas_de_raw(_raw_seno(10.0, dur=3.0), freq_rede=None)
    assert metricas["n_canais"] == 3


# --- comparação ---------------------------------------------------------

def test_comparar_antes_depois_calcula_razao_alpha():
    import preproc_basico

    raw = _raw_sintetico(fs=500.0, f_rede=60.0)
    filtrado, decisoes = preproc_basico.preprocessar(raw)
    comparacao = comparar_antes_depois(raw, filtrado, decisoes["freq_rede"])

    assert comparacao["razao_alpha"] == pytest.approx(1.0, abs=0.1)
    assert comparacao["depois"]["media_uv_abs_max"] < comparacao["antes"]["media_uv_abs_max"]
    assert comparacao["depois"]["razao_rede_db"] < comparacao["antes"]["razao_rede_db"]


# --- formatação ---------------------------------------------------------

def test_fmt_none_vira_na_e_nao_zero():
    assert _fmt(None) == "n/a"
    assert _fmt(0.0) == "0.000"


# --- relatório ----------------------------------------------------------

def _tmp_set(tmp_path, nome="sub-001", f_rede=60.0):
    """Um .set mínimo com rede conhecida, via o mesmo caminho do fixture
    do inventário."""
    from test_inventario_hbn import _escrever_set

    caminho = tmp_path / nome / "eeg" / f"{nome}_task-RestingState_eeg.set"
    _escrever_set(caminho, n_canais=4, n_amostras=10000, srate=500.0)
    return caminho


def test_relatorio_grava_arquivo_utf8(tmp_path):
    raw_path = tmp_path / "sinal.csv"
    import csv_data

    cabecalho = ",".join(list(csv_data.CANAIS_19) + ["ID", "Class"])
    linhas = [cabecalho]
    rng = np.random.RandomState(0)
    t = np.arange(128 * 40) / 128.0
    for i in range(len(t)):
        valores = [f"{140.0 + 20.0 * np.sin(2 * np.pi * 10 * t[i]) + rng.randn():.4f}"
                   for _ in csv_data.CANAIS_19]
        linhas.append(",".join(valores + ["v1", "ADHD"]))
    raw_path.write_text("\n".join(linhas) + "\n", encoding="utf-8")

    saida = tmp_path / "out" / "qc.txt"
    resumos = relatorio_qc(raw_path, saida, subject_id="v1")

    texto = saida.read_text(encoding="utf-8")
    assert "--- veredito ---" in texto
    assert "fração delta" in texto
    assert "potência alfa preservada" in texto
    assert len(resumos) == 1


def test_relatorio_varre_arvore_bids(tmp_path):
    _tmp_set(tmp_path, "sub-001")
    _tmp_set(tmp_path, "sub-002")

    resumos = relatorio_qc(tmp_path, tmp_path / "qc.txt")
    assert len(resumos) == 2
    assert {r["sujeito"] for r in resumos} == {"sub-001", "sub-002"}


def test_listar_entradas_pasta_sem_sujeitos(tmp_path):
    vazia = tmp_path / "vazia"
    vazia.mkdir()
    with pytest.raises(ValueError, match="nenhum .set de RestingState"):
        _listar_entradas(vazia, None)


def test_listar_entradas_caminho_inexistente(tmp_path):
    with pytest.raises(ValueError, match="caminho não encontrado"):
        _listar_entradas(tmp_path / "nao_existe", None)


def test_veredito_reprovado_aparece_no_resumo(tmp_path, monkeypatch):
    """A tabela por sujeito e o resumo do fim têm que contar a mesma história.

    Este teste existe por causa de um bug real: as métricas saem do numpy, e
    `numpy.bool_(False) is False` dá False. O filtro de reprovadas usava
    `ok is False`, então a linha da tabela imprimia FALHOU enquanto o resumo,
    logo abaixo, imprimia "nenhuma métrica reprovada". Um sujeito do HBN
    passou por esse buraco.
    """
    raw_path = _tmp_set(tmp_path, "sub-ruim")

    # força a métrica de alfa a reprovar sem depender do conteúdo do sinal
    # sintético: o que se está travando aqui é o caminho do veredito, não o
    # filtro. Alfa é a métrica certa para isso porque é sempre medida, ao
    # contrário da rede, que vale None quando não há pico a detectar.
    monkeypatch.setattr("qc_relatorio.MIN_RAZAO_ALPHA", 1e9)

    resumos = relatorio_qc(raw_path, tmp_path / "qc.txt", subject_id="sub-ruim")
    texto = (tmp_path / "qc.txt").read_text(encoding="utf-8")

    assert resumos[0]["veredito"]["alpha"] is False, "veredito deve ser bool puro, não numpy.bool_"
    assert "sujeitos com métrica reprovada:" in texto
    assert "nenhuma métrica reprovada" not in texto


# --- NaN não pode virar veredito OK -------------------------------------

def _raw_com_nan(amostras_nan=1, fs=128.0, dur=60.0, n_canais=19, semente=0):
    """Sinal saudável com um punhado de amostras NaN plantadas num canal —
    o dano que um eletrodo desconectado ou uma conversão malfeita deixa."""
    rng = np.random.default_rng(semente)
    dados = rng.normal(0, 20e-6, (n_canais, int(fs * dur))) + 150e-6
    dados[3, 1000:1000 + amostras_nan] = np.nan
    info = mne.create_info([f"c{i}" for i in range(n_canais)], fs, "eeg", verbose=False)
    return mne.io.RawArray(dados, info, verbose=False)


def test_metricas_contam_amostras_nao_finitas():
    metricas = metricas_de_raw(_raw_com_nan(amostras_nan=3), freq_rede=None)
    assert metricas["n_amostras_nao_finitas"] == 3


def test_metricas_de_sinal_limpo_contam_zero():
    """A contagem só serve se ela for zero no caso normal — uma métrica que
    acusa sempre não acusa nada."""
    metricas = metricas_de_raw(_raw_seno(10.0), freq_rede=None)
    assert metricas["n_amostras_nao_finitas"] == 0


def test_uma_amostra_nan_contamina_o_canal_e_o_car_espalha():
    """MEDIDO: UMA amostra NaN no canal 3, sinal de 19 canais a 128 Hz.
    Depois de `preprocessar` sem CAR, o canal 3 tinha 2.830 amostras não
    finitas (o comprimento do FIR); com CAR, os DEZENOVE canais tinham
    2.830 cada. Nenhuma exceção em momento nenhum — é este silêncio que o
    relatório precisa denunciar."""
    import preproc_basico

    raw = _raw_com_nan()
    assert metricas_de_raw(raw, freq_rede=None)["n_amostras_nao_finitas"] == 1

    sem_car, _ = preproc_basico.preprocessar(raw, car=False)
    com_car, _ = preproc_basico.preprocessar(raw, car=True)

    nf_sem = (~np.isfinite(sem_car.get_data(picks="eeg"))).sum(axis=1)
    nf_com = (~np.isfinite(com_car.get_data(picks="eeg"))).sum(axis=1)

    assert (nf_sem > 0).sum() == 1
    assert (nf_com > 0).sum() == 19


def test_veredito_da_media_nao_sai_ok_com_nan():
    """O bug exato que a auditoria pegou: a linha da média imprimia
    `nan  nan  OK  (residual 0.000%)`. O residual caía no ramo do 0.0
    porque `nan > 0` é False, e o relatório assinava embaixo de um dado
    destruído."""
    import preproc_basico
    from qc_relatorio import _linhas_do_sujeito

    raw = _raw_com_nan()
    filtrado, decisoes = preproc_basico.preprocessar(raw, car=True)
    comparacao = comparar_antes_depois(raw, filtrado, decisoes["freq_rede"])

    linhas, veredito = _linhas_do_sujeito("suj_nan", raw, comparacao, decisoes)
    texto = "\n".join(linhas)

    assert veredito["media"] is False
    linha_media = [l for l in linhas if l.startswith("média |máx|")][0]
    assert "OK" not in linha_media
    assert "residual 0.000%" not in linha_media
    # e o leitor precisa saber POR QUE, não só que reprovou
    assert "não finitas" in texto


def test_relatorio_com_nan_aparece_no_resumo_de_reprovadas(tmp_path, monkeypatch):
    """O outro lado do mesmo bug: tabela e resumo têm de contar a mesma
    história. Antes, a tabela dizia OK e o resumo dizia "nenhuma métrica
    reprovada" — os dois errados, de acordo entre si."""
    import preproc_basico

    raw = _raw_com_nan()
    monkeypatch.setattr(preproc_basico, "carregar_raw", lambda *a, **k: raw)
    monkeypatch.setattr("qc_relatorio.preproc_basico.carregar_raw", lambda *a, **k: raw)

    arquivo = tmp_path / "suj_nan.set"
    arquivo.write_bytes(b"")
    saida = tmp_path / "qc.txt"
    resumos = relatorio_qc(arquivo, saida)

    assert resumos[0]["veredito"]["media"] is False
    texto = saida.read_text(encoding="utf-8")
    assert "nenhuma métrica reprovada" not in texto
    assert "media" in texto.split("sujeitos com métrica reprovada:")[1]


def test_fmt_transforma_nan_em_na():
    """`f"{float('nan'):.2f}"` sai como a string `nan`, que numa coluna de
    números parece uma medida e não um buraco."""
    assert _fmt(float("nan")) == "n/a"
    assert _fmt(float("inf")) == "n/a"
    assert _fmt(None) == "n/a"
    assert _fmt(1.5, 2) == "1.50"


# --- unidade da potência alfa -------------------------------------------

def test_potencia_alpha_sai_em_uv2_por_hz_como_o_nome_diz():
    """Nome e número concordando, com o número conferível na mão.

    Um seno de 50 µV a 10 Hz tem variância A²/2 = 1.250 µV². O PSD de Welch
    aqui usa janela de 4 s a 500 Hz, ou seja, Δf = 0,25 Hz, e a soma dos
    bins vale variância/Δf = 1.250/0,25 = 5.000 µV²/Hz.

    Antes do conserto a mesma chave devolvia 5e-9, que é o mesmo número em
    V²/Hz — 1e12 de distância da unidade que a documentação prometia."""
    metricas = metricas_de_raw(_raw_seno(10.0, amp_uv=50.0), freq_rede=None)
    assert metricas["potencia_alpha_uv2_hz"] == pytest.approx(5000.0, rel=0.01)


def test_correcao_de_unidade_nao_move_a_razao_de_alfa():
    """A razão é adimensional: o fator 1e12 cancela nos dois lados. Se este
    teste quebrar, a correção de unidade virou correção de veredito."""
    import preproc_basico

    raw = _raw_sintetico(fs=500.0, f_rede=60.0)
    filtrado, decisoes = preproc_basico.preprocessar(raw)
    comparacao = comparar_antes_depois(raw, filtrado, decisoes["freq_rede"])

    em_volts = comparacao["depois"]["potencia_alpha_uv2_hz"] * 1e-12
    em_volts_antes = comparacao["antes"]["potencia_alpha_uv2_hz"] * 1e-12
    assert comparacao["razao_alpha"] == pytest.approx(em_volts / em_volts_antes)
