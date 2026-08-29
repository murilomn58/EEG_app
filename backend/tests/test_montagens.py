"""Trava o arquivo de montagens que a cena 3D lê para desenhar os eletrodos.

POR QUE ISTO PRECISA DE TESTE. Posição de eletrodo errada não levanta
exceção em lugar nenhum: o marcador simplesmente aparece alguns
centímetros fora do lugar, e um eletrodo fora do lugar continua parecendo
um eletrodo. Foi assim que a tabela escrita à mão que este arquivo
substituiu passou despercebida — ela errava 6,9 graus na mediana e 12,4 no
pior caso (T7 e T8), o que dá 1,1 cm e 2,0 cm no couro cabeludo.

O que se ancora aqui é a CONVENÇÃO, e não os números um a um: copiar os 129
valores para dentro do teste seria travar o arquivo contra ele mesmo, e o
teste passaria mesmo se a montagem inteira estivesse girada.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

ARQUIVO = RAIZ.parent / "assets" / "montagens.json"


@pytest.fixture(scope="module")
def montagens():
    if not ARQUIVO.is_file():
        pytest.fail(
            f"{ARQUIVO} não existe. Rode: python backend/scripts/gerar_montagens.py"
        )
    return json.loads(ARQUIVO.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# o arquivo tem o que a cena espera encontrar
# ---------------------------------------------------------------------------

def test_traz_as_montagens_que_o_app_desenha(montagens):
    """Cada uma cobre um banco real. Faltar uma não quebra a cena — ela cai
    na reserva de 19 —, mas faz um banco de 129 aparecer com 19 eletrodos,
    que é exatamente o defeito que este arquivo veio consertar."""
    assert set(montagens["montagens"]) >= {
        "standard_1020", "GSN-HydroCel-129", "standard_1005",
    }


@pytest.mark.parametrize("ident,esperado", [
    ("standard_1020", 19),
    ("GSN-HydroCel-129", 129),
    ("GSN-HydroCel-128", 128),
])
def test_contagem_de_eletrodos(montagens, ident, esperado):
    """A contagem é o que a tela mostra no selo. Errar aqui faz o app
    afirmar um número de eletrodos que a touca não tem."""
    assert montagens["montagens"][ident]["n"] == esperado


def test_declara_procedencia_e_natureza(montagens):
    """Template é média de população, não digitalização deste sujeito.
    Apagar essa distinção é o que faria a tela apresentar posição média como
    se fosse medida — e ninguém teria como perceber olhando."""
    for ident, m in montagens["montagens"].items():
        assert m["fonte"], f"{ident} sem fonte"
        assert "template" in m["natureza"], f"{ident} não declara que é template"
        assert "metros" in m["unidade_xyz"], f"{ident} sem unidade declarada"


def test_diz_que_foi_gerado(montagens):
    """Editar o JSON à mão o faz divergir da montagem de origem em silêncio.
    O aviso não impede, mas tira a desculpa."""
    assert "não editar" in montagens["aviso"].lower()
    assert montagens["mne_versao"]


# ---------------------------------------------------------------------------
# a convenção de coordenadas — a parte que erra sem avisar
# ---------------------------------------------------------------------------

def _por_nome(montagens, ident):
    return {e["nome"]: e for e in montagens["montagens"][ident]["eletrodos"]}


def test_cz_fica_no_alto_e_nao_exatamente_no_topo(montagens):
    """Cz é o eletrodo mais alto do 10-20, mas NÃO está no eixo vertical.

    No standard_1020 ele fica a 5,2 graus do topo, porque a montagem é
    ajustada ao fsaverage e o vértice geométrico da cabeça não coincide com
    o Cz do sistema 10-20. A tabela antiga trazia (0, 0), e esse
    arredondamento é o tipo de 'limpeza' que apaga geometria real.

    A faixa é folgada de propósito: o teste existe para pegar montagem
    deitada ou girada, não para congelar o segundo decimal do MNE."""
    cz = _por_nome(montagens, "standard_1020")["Cz"]
    assert 0.5 < cz["theta"] < 15, "Cz saiu do alto da cabeça"


def test_fz_na_frente_e_pz_atras(montagens):
    """phi 0 é a frente e ±180 são as costas. Trocar o sinal do phi espelha
    a cabeça no eixo antero-posterior, e uma cabeça espelhada continua
    parecendo uma cabeça — é o erro que não se vê."""
    p = _por_nome(montagens, "standard_1020")
    assert abs(p["Fz"]["phi"]) < 15, "Fz deixou de estar na frente"
    assert abs(p["Pz"]["phi"]) > 165, "Pz deixou de estar atrás"


def test_t7_a_esquerda_e_t8_a_direita(montagens):
    """phi negativo é a esquerda. Este é o par que mais se desloca quando a
    convenção quebra, e é também o que mais importa clinicamente: trocar os
    lados troca o hemisfério de um achado."""
    p = _por_nome(montagens, "standard_1020")
    assert p["T7"]["phi"] < -60, "T7 não está à esquerda"
    assert p["T8"]["phi"] > 60, "T8 não está à direita"
    assert p["T7"]["xyz"][0] < 0 < p["T8"]["xyz"][0], "o x dos temporais inverteu"


def test_fp_na_frente_e_o_atras(montagens):
    """O eixo antero-posterior inteiro, e não só a linha média: um giro que
    preserve Fz e Pz ainda pode ter torcido os laterais."""
    p = _por_nome(montagens, "standard_1020")
    assert p["Fp1"]["xyz"][1] > 0.05, "Fp1 deixou de estar à frente (y positivo)"
    assert p["O1"]["xyz"][1] < -0.05, "O1 deixou de estar atrás (y negativo)"


def test_xyz_e_esfericas_descrevem_o_mesmo_ponto(montagens):
    """As duas convenções saem da mesma fonte e não podem divergir.

    Este é o teste que impede o modo de falha mais traiçoeiro do arquivo:
    alguém corrigir o xyz e esquecer o theta/phi, deixando a cena desenhar
    por um par e o backend medir pelo outro."""
    import numpy as np
    from eletrodos import para_esfericas

    for ident, m in montagens["montagens"].items():
        for e in m["eletrodos"]:
            theta, phi = para_esfericas(e["xyz"])
            assert theta == pytest.approx(e["theta"], abs=0.02), f"{ident}/{e['nome']}"
            # phi dá volta em ±180: comparar o cosseno da diferença evita
            # que 179,99 e -179,99 sejam lidos como 360 graus de erro
            d = np.radians(phi - e["phi"])
            assert np.cos(d) == pytest.approx(1.0, abs=1e-4), f"{ident}/{e['nome']}"


def test_todo_eletrodo_esta_na_superficie_de_uma_cabeca(montagens):
    """Raio entre 4 e 15 cm. Não é precisão anatômica: é o portão que pega
    unidade trocada. Uma montagem em milímetros passaria por todos os testes
    de ângulo acima e desenharia a touca a 90 metros do cérebro."""
    import numpy as np
    for ident, m in montagens["montagens"].items():
        raios = [float(np.linalg.norm(e["xyz"])) for e in m["eletrodos"]]
        assert 0.04 < min(raios), f"{ident}: eletrodo perto demais do centro"
        assert max(raios) < 0.15, f"{ident}: eletrodo fora de qualquer cabeça"


# ---------------------------------------------------------------------------
# o arquivo é reprodutível
# ---------------------------------------------------------------------------

def test_o_gerador_reproduz_o_arquivo_em_disco(montagens):
    """O JSON em disco é o que o gerador produz HOJE.

    Sem isto, o arquivo vira um fóssil: alguém atualiza o MNE, as posições
    mudam, e a cena segue desenhando as de dois anos atrás sem nada avisar.
    Falhar aqui não é defeito — é o pedido para rodar o gerador de novo e
    olhar o que mudou antes de aceitar."""
    import gerar_montagens

    gerado = gerar_montagens.gerar()
    assert gerado["montagens"].keys() == montagens["montagens"].keys()
    for ident in gerado["montagens"]:
        assert gerado["montagens"][ident]["eletrodos"] == \
               montagens["montagens"][ident]["eletrodos"], (
            f"{ident} mudou. Rode: python backend/scripts/gerar_montagens.py"
        )


def test_o_frontend_le_o_mesmo_arquivo():
    """O caminho que o HTML busca tem de ser o caminho que este teste trava.

    Um teste verde sobre um arquivo que a página não lê é pior que teste
    nenhum: ele afirma cobertura que não existe."""
    html = (RAIZ.parent / "eeg-cerebro-3d.html").read_text(encoding="utf-8")
    assert 'fetch("assets/montagens.json")' in html


# ---------------------------------------------------------------------------
# o papel 10-20 de cada eletrodo
# ---------------------------------------------------------------------------

def test_a_malha_egi_diz_quais_eletrodos_sao_os_dezenove(montagens):
    """Sem o campo `papel`, a cena nao reconhece que o E11 desta touca e o
    mesmo Fz que ela ja desenhou como dipolo, e desenha os dois: 147
    marcadores para uma touca de 129."""
    egi = montagens["montagens"]["GSN-HydroCel-129"]["eletrodos"]
    papeis = {e["papel"] for e in egi if e.get("papel")}
    assert len(papeis) == 19, "a malha de 129 nao cobre os 19 canais de analise"


def test_o_vertice_e_reconhecido_pelos_dois_nomes(montagens):
    """O eletrodo do vertice tem DOIS nomes em uso, e os dois sao legitimos:
    a documentacao da EGI o chama de E129 (e o sensor de referencia), e
    tanto o GSN-HydroCel-129 do MNE quanto os arquivos do HBN o chamam de
    `Cz`. Sem conciliar as duas grafias a touca sai com 18 papeis."""
    egi = {e["nome"]: e for e in montagens["montagens"]["GSN-HydroCel-129"]["eletrodos"]}
    assert "Cz" in egi, "a montagem de 129 do MNE nomeia o vertice como Cz"
    assert egi["Cz"].get("papel") == "Cz"


def test_a_malha_de_128_nao_tem_o_vertice_e_isso_e_correto(montagens):
    """A GSN-HydroCel-128 e a de 129 SEM o vertice de referencia. Ter 18
    papeis ali nao e falha: e a resposta certa para uma touca que nao tem o
    Cz. Este teste existe para que 18 nao seja lido como defeito na proxima
    vez que alguem contar."""
    m128 = montagens["montagens"]["GSN-HydroCel-128"]["eletrodos"]
    nomes = {e["nome"] for e in m128}
    assert "Cz" not in nomes
    assert len({e["papel"] for e in m128 if e.get("papel")}) == 18


def test_o_papel_declarado_bate_com_o_mapa_publicado(montagens):
    """O arquivo nao pode inventar correspondencia: cada papel da malha EGI
    tem de sair de config.MAPA_EGI_1020, que cita o Technical Note da EGI e
    o mapa do fabricante. Inferir por proximidade geometrica daria outro
    resultado — a regra da linha media contraria a proximidade, e e por isso
    que o Fz e o E11 e nao o E6."""
    import config
    esperado = {egi: alvo for alvo, egi in config.MAPA_EGI_1020.items()}
    esperado["Cz"] = "Cz"   # o apelido do vertice, ver _papeis_egi
    for e in montagens["montagens"]["GSN-HydroCel-129"]["eletrodos"]:
        if e.get("papel"):
            assert esperado.get(e["nome"]) == e["papel"], f"{e['nome']} destoa do mapa"


def test_o_dez_zero_cinco_identifica_os_dezenove(montagens):
    """E o que faz um banco de 64 canais 10-10 aparecer com os 19 de analise
    ja identificados, sem depender do backend."""
    m = montagens["montagens"]["standard_1005"]["eletrodos"]
    papeis = {e["papel"] for e in m if e.get("papel")}
    assert len(papeis) == 19
    # aqui papel e identidade, nao traducao
    for e in m:
        if e.get("papel"):
            assert e["papel"] == e["nome"]
