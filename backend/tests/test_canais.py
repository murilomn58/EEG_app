import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from canais import resolver


def test_nome_direto_quando_o_banco_ja_nomeia_em_10_20():
    resolvido, faltantes = resolver(config.CANAIS_10_20, config.CANAIS_10_20)
    assert faltantes == []
    assert all(alvo == arquivo for alvo, arquivo in resolvido.items())


def test_traduz_pelo_mapa_quando_o_nome_direto_nao_existe():
    resolvido, faltantes = resolver(
        ["E11", "E62"], ["Fz", "Pz"], config.MAPA_EGI_1020
    )
    assert faltantes == []
    assert resolvido == {"Fz": "E11", "Pz": "E62"}


def test_nome_direto_vence_o_mapa():
    """Um banco que nomeia certo nao pode ser reescrito por uma tabela feita
    para outra malha. Aqui existem os DOIS nomes, e o direto tem de ganhar."""
    resolvido, _ = resolver(["Fz", "E11"], ["Fz"], config.MAPA_EGI_1020)
    assert resolvido["Fz"] == "Fz"


def test_faltante_e_reportado_e_nao_inventado():
    resolvido, faltantes = resolver(["E11"], ["Fz", "Pz"], config.MAPA_EGI_1020)
    assert faltantes == ["Pz"]
    assert "Pz" not in resolvido


def test_preserva_a_ordem_dos_canais_de_analise():
    """A ordem dos canais e contrato: mne_infer recebe um array (19, n) e
    assume a ordem de CANAIS_10_20. Embaralhar aqui trocaria eletrodo de
    lugar no modelo de fonte sem levantar erro nenhum."""
    resolvido, _ = resolver(config.CANAIS_10_20, config.CANAIS_10_20)
    assert list(resolvido.keys()) == list(config.CANAIS_10_20)


def test_mapa_egi_cobre_os_19_canais_de_analise():
    """Se um dia a lista de analise crescer sem o mapa crescer junto, o HBN
    volta a ser barrado — e este teste diz por que, antes da tela dizer."""
    _, faltantes = resolver(
        list(config.MAPA_EGI_1020.values()),
        config.CANAIS_10_20,
        config.MAPA_EGI_1020,
    )
    assert faltantes == []


def test_mapa_egi_nao_repete_eletrodo():
    """Dois canais 10-20 apontando para o mesmo E<n> seria erro de
    transcricao da tabela do fabricante, e duplicaria um sinal na analise."""
    valores = list(config.MAPA_EGI_1020.values())
    assert len(valores) == len(set(valores))


# ---------------------------------------------------------------------------
# A TERCEIRA VIA: grafia. Levantada por um banco de fora, nao por hipotese.
# ---------------------------------------------------------------------------
# O EEG Motor Movement/Imagery do PhysioNet (eegmmidb) nomeia os mesmos
# eletrodos 10-10 como `Cz..`, `Fc5.` e `C3..`: ponto de preenchimento e
# caixa propria, herdados do campo de tamanho fixo do cabecalho EDF.
#
# MEDIDO sobre a gravacao S001R03 antes desta via existir: `resolver` casava
# ZERO dos 19 canais de analise, e `eletrodos.posicoes` devolvia ZERO
# posicoes. Na tela isso nao vira mensagem de erro: vira a cabeca de
# conferencia desenhada VAZIA, sem um eletrodo sequer.

NOMES_EEGMMIDB = [
    "Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.",
    "C5..", "C3..", "C1..", "Cz..", "C2..", "C4..", "C6..",
    "Fp1.", "Fp2.", "F7..", "F3..", "Fz..", "F4..", "F8..",
    "T7..", "T8..", "P7..", "P3..", "Pz..", "P4..", "P8..",
    "O1..", "O2..",
]


def test_grafia_do_edf_casa_os_dezenove():
    """O caso que motivou a via 3, com os nomes reais do arquivo."""
    from canais import resolver
    resolvido, faltantes = resolver(NOMES_EEGMMIDB, config.CANAIS_10_20)
    assert faltantes == []
    assert len(resolvido) == 19
    # devolve o nome DO ARQUIVO, e nao o normalizado: e ele que o MNE aceita
    assert resolvido["Cz"] == "Cz.."
    assert resolvido["T7"] == "T7.."


def test_o_nome_devolvido_existe_mesmo_no_arquivo():
    """A garantia que impede a via 3 de virar armadilha. Devolver a forma
    normalizada (`CZ`) passaria por aqui como se tivesse resolvido, e o
    `pick_channels` falharia depois, longe daqui, com o nome ja reescrito."""
    from canais import resolver
    resolvido, _ = resolver(NOMES_EEGMMIDB, config.CANAIS_10_20)
    for alvo, no_arquivo in resolvido.items():
        assert no_arquivo in NOMES_EEGMMIDB, f"{alvo} -> {no_arquivo} nao existe no arquivo"


def test_o_mapa_publicado_vence_a_grafia():
    """Ordem de autoridade: mapa e declaracao de fonte publicada, grafia e
    palpite tipografico. Onde os dois discordam, manda quem tem procedencia.

    O caso e artificial de proposito: um arquivo que tem `FZ` (que casaria
    por grafia) e tambem `E11` (que o mapa EGI declara). O mapa vence."""
    from canais import resolver
    resolvido, _ = resolver(["FZ", "E11"], ["Fz"], {"Fz": "E11"})
    assert resolvido["Fz"] == "E11"


def test_o_nome_exato_vence_a_grafia():
    """`Cz` exato nao pode ser trocado por `CZ..` so porque ele veio antes
    na lista de canais do arquivo."""
    from canais import resolver
    resolvido, _ = resolver(["CZ..", "Cz"], ["Cz"])
    assert resolvido["Cz"] == "Cz"


def test_resolver_detalhado_diz_quem_veio_por_grafia():
    """A tela precisa poder distinguir canal que casou por identidade de
    canal que casou por aproximacao de escrita. Sem isso, aproximacao vira
    identidade na leitura de quem confere."""
    from canais import resolver_detalhado
    _, _, por_grafia = resolver_detalhado(NOMES_EEGMMIDB, config.CANAIS_10_20)
    assert len(por_grafia) == 19

    _, _, nenhum = resolver_detalhado(config.CANAIS_10_20, config.CANAIS_10_20)
    assert nenhum == [], "canal que casou por nome exato nao pode entrar como grafia"


def test_nao_inventa_equivalencia_entre_nomes_diferentes():
    """T3 e T7 sao o mesmo eletrodo em convencoes distintas do 10-20, e
    normalizar grafia NAO pode virar tradutor de convencao. Isso e traducao,
    tem de ser declarada em `mapa_canais` com procedencia, e inferir aqui
    faria o app afirmar equivalencia que ninguem conferiu."""
    from canais import resolver
    _, faltantes = resolver(["T3", "T4", "T5", "T6"], ["T7", "T8", "P7", "P8"])
    assert set(faltantes) == {"T7", "T8", "P7", "P8"}


def test_resolver_nao_guarda_estado_entre_chamadas():
    """O backend atende em varias threads. Se o detalhe de uma resolucao
    ficasse num atributo de modulo, a resposta de uma requisicao descreveria
    a resolucao de outra — erro que so aparece sob carga."""
    from canais import resolver_detalhado
    _, _, a = resolver_detalhado(NOMES_EEGMMIDB, config.CANAIS_10_20)
    _, _, b = resolver_detalhado(config.CANAIS_10_20, config.CANAIS_10_20)
    assert len(a) == 19 and b == [], "a segunda chamada contaminou a primeira"
