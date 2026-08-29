# -*- coding: utf-8 -*-
"""Trava a receita: o registro que permite refazer uma análise igual.

O BURACO QUE ELA FECHA

Hoje o `decisoes` do pré-processamento — filtro aplicado, rede detectada,
rebaixamento por Nyquist — viaja na resposta HTTP e morre no selo de texto da
tela. Nenhuma das medidas quantitativas do app sai. Outra pessoa não consegue
repetir uma análise a partir do que o app entrega; no máximo repetir o setup.

A receita é o arquivo que o app exporta e o lote consome. Se ela não for
fiel, tudo o que vem depois herda a infidelidade — por isso ela tem teste
próprio, e por isso o teste de ida e volta é o mais importante deles.
"""
import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import receita


def _exemplo():
    return receita.montar(
        banco="adhdata",
        sujeitos=["v10p", "v11p"],
        etapas={
            "preproc": {"l_freq": 0.5, "h_freq": 57.6, "freq_rede": 50},
            "epocas": {"duracao_s": 2.0, "passo_s": 1.0},
        },
        notas="teste",
    )


# ---------------------------------------------------------------------------
# ida e volta
# ---------------------------------------------------------------------------

def test_ida_e_volta_preserva_tudo(tmp_path):
    """O teste que sustenta o resto: o que sai do arquivo é o que entrou."""
    r = _exemplo()
    caminho = tmp_path / "receita.json"
    receita.salvar(r, caminho)
    de_volta = receita.carregar(caminho)
    assert de_volta["banco"] == r["banco"]
    assert de_volta["sujeitos"] == r["sujeitos"]
    assert de_volta["etapas"] == r["etapas"]


def test_o_arquivo_e_json_legivel_por_humano(tmp_path):
    """Alguém vai abrir isso num editor para conferir um parâmetro. Uma linha
    só de JSON compactado tornaria a conferência um exercício de paciência."""
    caminho = tmp_path / "receita.json"
    receita.salvar(_exemplo(), caminho)
    texto = caminho.read_text(encoding="utf-8")
    assert texto.count("\n") > 5
    json.loads(texto)


def test_acentos_sobrevivem(tmp_path):
    """As decisões do preproc trazem texto em português com acento. Escrever
    com escape unicode faria o arquivo ilegível justamente onde ele explica."""
    r = receita.montar(banco="adhdata", sujeitos=["v10p"],
                       etapas={"preproc": {"motivo": "rede não detectada"}})
    caminho = tmp_path / "r.json"
    receita.salvar(r, caminho)
    assert "não detectada" in caminho.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# o que a receita tem de carregar
# ---------------------------------------------------------------------------

def test_registra_as_versoes_das_bibliotecas():
    """Um resultado que depende de mne/numpy/scipy/sklearn sem dizer quais
    versões produziram não é reproduzível — é uma foto."""
    r = _exemplo()
    for lib in ("numpy", "scipy", "sklearn", "mne"):
        assert lib in r["versoes"], f"falta a versão de {lib}"


def test_registra_data_e_versao_do_formato():
    r = _exemplo()
    assert r["gerado_em"]
    assert r["formato"] >= 1


def test_carrega_as_etapas_na_ordem_em_que_foram_declaradas(tmp_path):
    """A ordem das etapas é informação: filtrar depois de epocar não é a
    mesma coisa que epocar depois de filtrar."""
    r = receita.montar(banco="x", sujeitos=[], etapas={
        "preproc": {}, "epocas": {}, "split": {}, "normalizacao": {},
    })
    caminho = tmp_path / "r.json"
    receita.salvar(r, caminho)
    assert list(receita.carregar(caminho)["etapas"]) == [
        "preproc", "epocas", "split", "normalizacao"]


# ---------------------------------------------------------------------------
# divergência de versão: aviso declarado, nunca silêncio
# ---------------------------------------------------------------------------

def test_versao_diferente_gera_aviso_e_nao_erro(tmp_path):
    """Rodar uma receita antiga numa máquina com biblioteca nova tem de ser
    POSSÍVEL — recusar travaria a reanálise de um resultado antigo, que é
    justamente o que a receita existe para permitir. Mas não pode ser
    silencioso: o número pode mudar."""
    r = _exemplo()
    r["versoes"]["numpy"] = "0.0.0-inventada"
    caminho = tmp_path / "r.json"
    receita.salvar(r, caminho)
    carregada = receita.carregar(caminho)
    avisos = receita.conferir_versoes(carregada)
    assert any("numpy" in a for a in avisos)


def test_versoes_iguais_nao_geram_aviso(tmp_path):
    caminho = tmp_path / "r.json"
    receita.salvar(_exemplo(), caminho)
    assert receita.conferir_versoes(receita.carregar(caminho)) == []


def test_formato_futuro_recusa_com_o_numero(tmp_path):
    """Formato mais novo que o código sabe ler é o único caso em que recusar
    é mais honesto que tentar: os campos podem ter mudado de significado."""
    r = _exemplo()
    r["formato"] = 999
    caminho = tmp_path / "r.json"
    caminho.write_text(json.dumps(r), encoding="utf-8")
    with pytest.raises(ValueError, match="999"):
        receita.carregar(caminho)


def test_receita_sem_campo_obrigatorio_recusa(tmp_path):
    caminho = tmp_path / "r.json"
    caminho.write_text(json.dumps({"formato": 1, "banco": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="etapas"):
        receita.carregar(caminho)
