import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from fenotipo_hbn import fenotipo, DIMENSOES, _histograma_texto

import pandas as pd


def _escrever_tsv(caminho, linhas, colunas=None):
    """participants.tsv sintético. As colunas seguem o header real do
    ds005505 (24 colunas), reduzido ao que os scripts leem."""
    colunas = colunas or ["participant_id", "sex", "age"] + DIMENSOES
    caminho.parent.mkdir(parents=True, exist_ok=True)
    conteudo = "\t".join(colunas) + "\n"
    for linha in linhas:
        conteudo += "\t".join(str(v) for v in linha) + "\n"
    caminho.write_text(conteudo, encoding="utf-8")
    return caminho


def _tsv_padrao(tmp_path):
    """5 sujeitos: 3 completos, 1 com as 4 dimensões n/a, 1 com age n/a."""
    return _escrever_tsv(
        tmp_path / "participants.tsv",
        [
            ["sub-001", "F", 11.3386, -0.603, -0.446, 1.248, 0.325],
            ["sub-002", "M", 7.6648, 0.512, 1.104, -0.221, 0.876],
            ["sub-003", "M", 21.674, 1.201, 0.334, 0.115, -0.442],
            ["sub-004", "F", 19.4946, "n/a", "n/a", "n/a", "n/a"],
            ["sub-005", "M", "n/a", 0.101, -0.998, 0.554, 0.203],
        ],
    )


def test_fenotipo_na_minusculo_conta_como_ausente(tmp_path):
    """A armadilha central: o pandas não reconhece 'n/a' minúsculo por
    padrão. Sem na_values explícito, age viraria object e a média
    levantaria TypeError — ou sairia errada em silêncio."""
    resultado = fenotipo(_tsv_padrao(tmp_path), tmp_path / "out" / "fen.txt")

    idade = resultado["resumos"]["age"]
    assert idade["n"] == 4
    assert idade["ausentes"] == 1
    # média dos 4 válidos, sem o "n/a" entrar como texto ou como zero
    assert idade["media"] == pytest.approx((11.3386 + 7.6648 + 21.674 + 19.4946) / 4)


def test_fenotipo_contagem_de_ausentes_por_dimensao(tmp_path):
    resultado = fenotipo(_tsv_padrao(tmp_path), tmp_path / "fen.txt")
    for dimensao in DIMENSOES:
        assert resultado["resumos"][dimensao]["ausentes"] == 1
        assert resultado["resumos"][dimensao]["n"] == 4


def test_fenotipo_completos_todas_dimensoes(tmp_path):
    """O n real utilizável: sub-004 tem as 4 ausentes, então sobram 4."""
    resultado = fenotipo(_tsv_padrao(tmp_path), tmp_path / "fen.txt")
    assert resultado["completos_todas_dimensoes"] == 4
    assert resultado["n_sujeitos"] == 5


def test_fenotipo_dimensao_ausente_no_release_nao_levanta(tmp_path):
    """Descobrir que a release não traz p_factor é o resultado, não erro."""
    caminho = _escrever_tsv(
        tmp_path / "participants.tsv",
        [["sub-001", "F", 11.3, -0.4, 1.2, 0.3], ["sub-002", "M", 9.1, 0.5, -0.2, 0.8]],
        colunas=["participant_id", "sex", "age", "attention", "internalizing", "externalizing"],
    )
    saida = tmp_path / "fen.txt"
    resultado = fenotipo(caminho, saida)

    assert resultado["dimensoes_ausentes"] == ["p_factor"]
    assert "p_factor" not in resultado["resumos"]
    assert "AUSENTE NO RELEASE" in saida.read_text(encoding="utf-8")


def test_fenotipo_coluna_toda_ausente_nao_levanta(tmp_path):
    """Coluna 100% n/a não é o mesmo que coluna ausente: n=0, sem TypeError."""
    caminho = _escrever_tsv(
        tmp_path / "participants.tsv",
        [
            ["sub-001", "F", 11.3, "n/a", -0.4, 1.2, 0.3],
            ["sub-002", "M", 9.1, "n/a", 0.5, -0.2, 0.8],
        ],
    )
    resultado = fenotipo(caminho, tmp_path / "fen.txt")
    assert resultado["resumos"]["p_factor"]["n"] == 0
    assert resultado["resumos"]["p_factor"]["media"] is None
    assert resultado["completos_todas_dimensoes"] == 0


def test_fenotipo_grava_arquivo_utf8_com_veredito(tmp_path):
    saida = tmp_path / "sub" / "pasta" / "fen.txt"
    fenotipo(_tsv_padrao(tmp_path), saida)

    texto = saida.read_text(encoding="utf-8")
    assert "--- veredito ---" in texto
    assert "completos_todas_dimensoes: 4" in texto
    # acentuados sobreviveram ao encoding explícito
    assert "distribuição" in texto
    assert "gênero" in texto
    for dimensao in DIMENSOES:
        assert dimensao in texto


def test_fenotipo_arquivo_inexistente(tmp_path):
    with pytest.raises(ValueError, match="participants.tsv não encontrado"):
        fenotipo(tmp_path / "nao_existe.tsv", tmp_path / "fen.txt")


def test_fenotipo_tsv_lido_como_csv(tmp_path):
    """Arquivo separado por vírgula vira uma coluna só — falha em voz alta."""
    caminho = tmp_path / "participants.tsv"
    caminho.write_text("id,sex,age\nsub-001,F,11.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sem coluna participant_id"):
        fenotipo(caminho, tmp_path / "fen.txt")


def test_histograma_serie_degenerada():
    assert "sem dados suficientes" in _histograma_texto(pd.Series([], dtype=float))[0]
    assert "sem dados suficientes" in _histograma_texto(pd.Series([5.0, 5.0, 5.0]))[0]


def test_histograma_escala_pelo_maior_bin():
    """Distribuição concentrada não pode achatar as barras a zero."""
    serie = pd.Series([1.0] * 50 + [10.0])
    linhas = _histograma_texto(serie)
    assert any("#" in l for l in linhas)
