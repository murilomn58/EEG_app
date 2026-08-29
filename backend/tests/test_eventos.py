import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from eventos import detectar_eventos, _descricoes_do_sidecar

from test_inventario_hbn import _escrever_set

TSV = (
    "onset\tduration\tsample\tvalue\tevent_code\n"
    "0\tn/a\t0\tbreak cnt\tbreak cnt\n"
    "1.288\tn/a\t644\tresting_start\t90\n"
    "47.84\tn/a\t23920\tinstructed_toOpenEyes\t20\n"
    "67.84\t2.5\t33920\tinstructed_toCloseEyes\t30\n"
)

# reproduz o erro real do HBN: dois níveis com a MESMA descrição
SIDECAR = {
    "value": {
        "Description": "The event marker value (type).",
        "Levels": {
            "resting_start": "Recording of Resting State task started",
            "instructed_toOpenEyes": "A voice prompt instructed subject to open their eyes",
            "instructed_toCloseEyes": "A voice prompt instructed subject to open their eyes",
        },
    }
}


def _arvore(tmp_path, com_sidecar=True, com_tsv=True):
    raiz = tmp_path / "bids"
    caminho_set = raiz / "sub-001" / "eeg" / "sub-001_task-RestingState_eeg.set"
    _escrever_set(caminho_set, n_canais=4, n_amostras=500, srate=100.0)
    if com_tsv:
        caminho_set.with_name("sub-001_task-RestingState_events.tsv").write_text(TSV, encoding="utf-8")
    if com_sidecar:
        (raiz / "task-RestingState_events.json").write_text(
            json.dumps(SIDECAR), encoding="utf-8"
        )
    return raiz, caminho_set


def test_le_os_eventos_em_ordem(tmp_path):
    raiz, caminho_set = _arvore(tmp_path)
    r = detectar_eventos(caminho_set, raiz=raiz)

    assert r["n"] == 4
    assert [e["onset"] for e in r["eventos"]] == [0.0, 1.288, 47.84, 67.84]
    assert r["eventos"][2]["valor"] == "instructed_toOpenEyes"
    assert r["eventos"][2]["codigo"] == "20"
    assert r["motivo"] is None


def test_duration_na_vira_none(tmp_path):
    """n/a em duration é o caso NORMAL no HBN: a duração é dada pelo
    próximo evento, não pela coluna."""
    raiz, caminho_set = _arvore(tmp_path)
    r = detectar_eventos(caminho_set, raiz=raiz)
    assert r["eventos"][0]["duracao"] is None
    assert r["eventos"][3]["duracao"] == 2.5


def test_enriquece_com_a_descricao_do_sidecar(tmp_path):
    raiz, caminho_set = _arvore(tmp_path)
    r = detectar_eventos(caminho_set, raiz=raiz)
    resting = next(e for e in r["eventos"] if e["valor"] == "resting_start")
    assert resting["descricao"] == "Recording of Resting State task started"


def test_detecta_descricao_repetida_no_dicionario(tmp_path):
    """O erro real do HBN: instructed_toCloseEyes recebeu a descrição do
    toOpenEyes. A regra é genérica, não uma lista de casos conhecidos."""
    raiz, caminho_set = _arvore(tmp_path)
    r = detectar_eventos(caminho_set, raiz=raiz)

    assert len(r["divergencias"]) == 1
    assert r["divergencias"][0]["valores"] == [
        "instructed_toCloseEyes", "instructed_toOpenEyes",
    ]
    # os dois eventos afetados vêm marcados, para a interface preferir o valor
    suspeitos = {e["valor"] for e in r["eventos"] if e["suspeita"]}
    assert suspeitos == {"instructed_toCloseEyes", "instructed_toOpenEyes"}
    assert not next(e for e in r["eventos"] if e["valor"] == "resting_start")["suspeita"]


def test_sem_sidecar_nao_derruba(tmp_path):
    """Sidecar ausente custa a descrição, não os eventos."""
    raiz, caminho_set = _arvore(tmp_path, com_sidecar=False)
    r = detectar_eventos(caminho_set, raiz=raiz)
    assert r["n"] == 4
    assert all(e["descricao"] is None for e in r["eventos"])
    assert r["divergencias"] == []


def test_sem_events_tsv_devolve_motivo(tmp_path):
    raiz, caminho_set = _arvore(tmp_path, com_tsv=False)
    r = detectar_eventos(caminho_set, raiz=raiz)
    assert r["n"] == 0
    assert "events.tsv" in r["motivo"]


def test_adhdata_devolve_vazio_com_motivo(tmp_path):
    """Lista vazia silenciosa seria indistinguível de falha. O dataset não
    tem marcador de estímulo, e isso é informação."""
    caminho = tmp_path / "adhdata.csv"
    caminho.write_text("Fp1,ID\n1.0,v1\n", encoding="utf-8")
    r = detectar_eventos(caminho)
    assert r["eventos"] == []
    assert "não traz coluna de evento" in r["motivo"]


def test_extensao_desconhecida(tmp_path):
    caminho = tmp_path / "sinal.edf"
    caminho.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="extensão não suportada"):
        detectar_eventos(caminho)


def test_arquivo_inexistente(tmp_path):
    with pytest.raises(ValueError, match="arquivo não encontrado"):
        detectar_eventos(tmp_path / "nao_existe.set")


def test_sidecar_sem_divergencia(tmp_path):
    """Descrições todas distintas: nada a sinalizar."""
    caminho = tmp_path / "task-X_events.json"
    caminho.write_text(json.dumps({"value": {"Levels": {"a": "um", "b": "dois"}}}), encoding="utf-8")
    descricoes, divergencias = _descricoes_do_sidecar(caminho)
    assert descricoes == {"a": "um", "b": "dois"}
    assert divergencias == []


# ---------------------------------------------------------------------------
# O MARCADOR TEM DE SER DO SUJEITO QUE ESTA TOCANDO
# ---------------------------------------------------------------------------
# Achado desta rodada, medido contra o backend de pe. O app pedia
# `/eventos?dataset_id=...` SEM `subject_id`, e o backend caia no primeiro
# sujeito do release. As etiquetas desenhadas sobre o tracado eram sempre as
# do sub-NDARAC904DMU: 36 eventos de uma tarefa de pontos (dot_no1_ON,
# seqLearning_start). Tocar o sub-NDARAG143ARJ, que e RestingState com 14
# eventos, desenhava mesmo assim as 36 etiquetas de um protocolo que aquela
# gravacao nunca rodou.
#
# E nao era so o conjunto de rotulos: os carimbos de tempo tambem eram de
# outra gravacao. O terceiro evento do primeiro sujeito cai em 47,84 s, o do
# segundo em 80,96 s. Mesmo entre duas gravacoes do MESMO protocolo a
# etiqueta pousava dezenas de segundos fora do lugar — e etiqueta no lugar
# errado sobre um tracado e pior que etiqueta nenhuma, porque quem le usa
# ela para interpretar o sinal.

HTML = Path(__file__).resolve().parent.parent.parent / "eeg-cerebro-3d.html"


def _corpo_da_funcao(html, assinatura):
    """O texto entre uma declaracao de funcao e a proxima do arquivo.

    Existe porque a primeira versao destes testes recortava um numero FIXO
    de caracteres (1200) depois da assinatura. Funcionou ate `tocarSujeito`
    ganhar o bloco de reinicio no comeco: a chamada procurada continuava
    la, so que umas centenas de caracteres adiante, e o teste passou a
    acusar uma regressao que nao existia.

    Um teste que quebra quando o codigo CERTO cresce nao protege nada:
    cobra apenas que ninguem mexa perto dele."""
    depois = html.split(assinatura, 1)[1]
    fim = depois.find("\nfunction ")
    return depois if fim == -1 else depois[:fim]


def test_o_app_pede_os_eventos_do_sujeito_que_vai_tocar():
    """O `subject_id` viaja na chamada de eventos.

    Teste de grep, e nao de comportamento, porque o que se protege esta no
    frontend. E frouxo de proposito: ele nao afirma que o app esta correto,
    afirma que a correcao nao foi desfeita sem ninguem notar."""
    html = HTML.read_text(encoding="utf-8")
    assert "function recarregarEventosDoSujeito" in html
    corpo = _corpo_da_funcao(html, "function recarregarEventosDoSujeito(sujeito) {")
    # o que importa e que a URL de eventos carregue o sujeito; como as duas
    # partes estao quebradas em linhas e o formato pode mudar, casa cada
    # pedaco por si em vez de exigir a formatacao exata
    assert "/eventos?" in corpo
    assert "&subject_id=" in corpo


def test_trocar_de_sujeito_dispara_a_releitura():
    """`tocarSujeito` chama a releitura. Sem esta chamada a funcao existe e
    nunca roda, que e o modo de falha mais silencioso possivel: o codigo do
    conserto esta no arquivo, e o defeito continua na tela."""
    html = HTML.read_text(encoding="utf-8")
    corpo = _corpo_da_funcao(html, "function tocarSujeito(sujeito) {")
    assert "recarregarEventosDoSujeito(sujeito)" in corpo


def test_falha_na_releitura_apaga_as_etiquetas_antigas():
    """Se a leitura do novo sujeito falhar, as etiquetas do anterior NAO
    podem ficar na tela: elas seriam lidas como sendo desta gravacao.
    Ausencia e honesta, etiqueta errada nao e."""
    html = HTML.read_text(encoding="utf-8")
    trecho = html.split("function recarregarEventosDoSujeito", 1)[1].split("\n}", 1)[0]
    catch = trecho.split(".catch(", 1)[1]
    assert "marcadoresDaTarefa = []" in catch, "o catch nao limpa os marcadores"
