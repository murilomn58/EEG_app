import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

# scripts/ não entra no sys.path pelos mecanismos que fazem os outros
# testes funcionarem (CWD=backend/ + basedir de tests/), então repetimos
# aqui o mesmo hack de verificar_referencia.py — com .parent.parent
# porque este arquivo está um nível mais fundo, em backend/tests/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from inventario_hbn import inventariar, _tarefa_do_nome, _caminho_events


def _escrever_set(caminho, n_canais=4, n_amostras=500, srate=100.0):
    """Um .set do EEGLAB mínimo que o mne.io.read_raw_eeglab consegue ler,
    com os dados embutidos (é assim que o ds005505 vem — sem .fdt).
    Escrito com scipy.io.savemat para não precisar do eeglabio só nos
    testes. chanlocs PRECISA ser array estruturado: uma lista de strings
    vira cell array que o reader do MNE não desempacota como espera."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    dados = np.random.RandomState(0).randn(n_canais, n_amostras).astype(np.float32)

    chanlocs = np.zeros(
        n_canais, dtype=[("labels", "O"), ("X", "O"), ("Y", "O"), ("Z", "O")]
    )
    for i in range(n_canais):
        chanlocs["labels"][i] = f"E{i + 1}"
        chanlocs["X"][i] = float(i)
        chanlocs["Y"][i] = 0.0
        chanlocs["Z"][i] = 0.0

    eeg = {
        "setname": caminho.stem,
        "filename": caminho.name,
        "filepath": str(caminho.parent),
        "nbchan": float(n_canais),
        "trials": 1.0,
        "pnts": float(n_amostras),
        "srate": float(srate),
        "xmin": 0.0,
        "xmax": (n_amostras - 1) / srate,
        "data": dados,
        "chanlocs": chanlocs,
        "event": [],
        "epoch": [],
        "icawinv": [],
        "icasphere": [],
        "icaweights": [],
        "ref": "common",
    }
    savemat(caminho, {"EEG": eeg}, appendmat=False)


def _arvore_bids(tmp_path):
    """Árvore BIDS sintética com um caso normal e um de cada borda."""
    raiz = tmp_path / "bids"

    # sub-001: normal, com events.tsv
    s1 = raiz / "sub-001" / "eeg" / "sub-001_task-RestingState_eeg.set"
    _escrever_set(s1)
    s1.with_name("sub-001_task-RestingState_events.tsv").write_text(
        "onset\tduration\n0.0\t1.0\n", encoding="utf-8"
    )

    # sub-002: duas tarefas, nenhuma com events.tsv
    _escrever_set(raiz / "sub-002" / "eeg" / "sub-002_task-RestingState_eeg.set")
    _escrever_set(raiz / "sub-002" / "eeg" / "sub-002_task-DespicableMe_eeg.set")

    # sub-003: sem pasta eeg/
    (raiz / "sub-003").mkdir(parents=True)

    # sub-004: .set corrompido
    corrompido = raiz / "sub-004" / "eeg" / "sub-004_task-RestingState_eeg.set"
    corrompido.parent.mkdir(parents=True)
    corrompido.write_bytes(b"isto nao e um arquivo matlab")

    # derivatives/ não pode virar linha
    _escrever_set(raiz / "derivatives" / "sub-999" / "eeg" / "sub-999_task-RestingState_eeg.set")

    return raiz


def test_inventariar_conta_linhas_e_ignora_derivatives(tmp_path):
    linhas = inventariar(_arvore_bids(tmp_path), tmp_path / "out" / "inv.csv")
    # 1 (sub-001) + 2 (sub-002) + 1 (sub-003 sem eeg) + 1 (sub-004) = 5
    assert len(linhas) == 5
    assert all(l["sujeito"] != "sub-999" for l in linhas)


def test_inventariar_metadados_do_set_valido(tmp_path):
    linhas = inventariar(_arvore_bids(tmp_path), tmp_path / "inv.csv")
    linha = next(l for l in linhas if l["sujeito"] == "sub-001")

    assert linha["tarefa"] == "RestingState"
    assert linha["n_canais"] == 4
    assert linha["sfreq_hz"] == "100.0"
    # 500 amostras a 100 Hz = 5.00s. raw.times[-1] daria 4.99 — o bug
    # que esta asserção existe para pegar.
    assert linha["duracao_s"] == "5.00"
    assert linha["tem_events_tsv"] == "sim"
    assert linha["erro"] == ""


def test_inventariar_sem_events_tsv(tmp_path):
    linhas = inventariar(_arvore_bids(tmp_path), tmp_path / "inv.csv")
    linha = next(l for l in linhas if l["sujeito"] == "sub-002" and l["tarefa"] == "DespicableMe")
    assert linha["tem_events_tsv"] == "nao"
    assert linha["erro"] == ""


def test_inventariar_sujeito_sem_pasta_eeg_emite_linha(tmp_path):
    linhas = inventariar(_arvore_bids(tmp_path), tmp_path / "inv.csv")
    linha = next(l for l in linhas if l["sujeito"] == "sub-003")
    assert "eeg" in linha["erro"]
    assert linha["n_canais"] == ""
    assert linha["sfreq_hz"] == ""


def test_inventariar_set_corrompido_nao_aborta(tmp_path):
    """A decisão central do script: um arquivo ilegível não pode derrubar
    a varredura nem contaminar as outras linhas."""
    linhas = inventariar(_arvore_bids(tmp_path), tmp_path / "inv.csv")

    linha_ruim = next(l for l in linhas if l["sujeito"] == "sub-004")
    assert linha_ruim["erro"] != ""
    assert linha_ruim["n_canais"] == ""

    intactas = [l for l in linhas if l["erro"] == ""]
    assert len(intactas) == 3
    assert all(l["n_canais"] == 4 for l in intactas)


def test_inventariar_csv_gravado_com_cabecalho_exato(tmp_path):
    caminho = tmp_path / "sub" / "pasta" / "inv.csv"
    inventariar(_arvore_bids(tmp_path), caminho)

    with open(caminho, newline="", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        assert leitor.fieldnames == [
            "sujeito", "tarefa", "arquivo", "n_canais",
            "sfreq_hz", "duracao_s", "tem_events_tsv", "erro",
        ]
        assert len(list(leitor)) == 5


def test_inventariar_ordem_deterministica(tmp_path):
    raiz = _arvore_bids(tmp_path)
    primeira = inventariar(raiz, tmp_path / "a.csv")
    segunda = inventariar(raiz, tmp_path / "b.csv")
    assert [l["arquivo"] for l in primeira] == [l["arquivo"] for l in segunda]


def test_inventariar_raiz_inexistente(tmp_path):
    with pytest.raises(ValueError, match="raiz bids não encontrada"):
        inventariar(tmp_path / "nao_existe", tmp_path / "inv.csv")


def test_inventariar_sem_sujeitos(tmp_path):
    vazia = tmp_path / "vazia"
    vazia.mkdir()
    with pytest.raises(ValueError, match="nenhum sujeito"):
        inventariar(vazia, tmp_path / "inv.csv")


def test_tarefa_do_nome_com_run():
    """run- desloca as posições: um parser por índice fixo quebraria."""
    assert _tarefa_do_nome("sub-X_task-surroundSupp_run-1_eeg.set") == "surroundSupp"
    assert _tarefa_do_nome("sub-X_task-seqLearning8target_eeg.set") == "seqLearning8target"
    assert _tarefa_do_nome("sub-X_eeg.set") == "desconhecida"


def test_caminho_events_usa_sufixo_bids():
    """with_suffix('.tsv') daria _eeg.tsv e nunca acharia o arquivo."""
    caminho = Path("sub-X_task-RestingState_eeg.set")
    assert _caminho_events(caminho).name == "sub-X_task-RestingState_events.tsv"
