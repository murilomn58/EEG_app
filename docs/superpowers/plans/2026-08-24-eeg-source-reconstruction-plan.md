# EEG 3D Source Reconstruction (MNE-Python) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adicionar uma visualização de reconstrução de fontes corticais (MNE-Python: forward model + inverse solution sobre o template fsaverage) ao dashboard EEG 3D existente, usando dados reais de `adhdata.csv`, numa nova aba "Fontes corticais" que convive com a aba de sensores já existente.

**Architecture:** Backend novo em `backend/` (FastAPI + MNE-Python) calcula forward/inverse operator UMA VEZ no boot (usando o template fsaverage e a montagem 10-20 dos 19 canais), e expõe `GET /subjects` + `POST /source-localization` para o frontend consumir por request (rápido, só álgebra linear sobre o operador já pronto). O frontend (`eeg-cerebro-3d.html`) ganha uma segunda malha 3D (superfície cortical fsaverage, exportada offline por um script auxiliar) colorida pelos valores devolvidos pelo backend, numa aba nova ao lado da aba de sensores já existente.

**Tech Stack:** Python 3 + FastAPI + uvicorn + MNE-Python + pandas/numpy (backend); Three.js r128 + OBJLoader (frontend, já em uso); pytest + FastAPI TestClient (testes do backend); Playwright MCP para verificação visual do frontend (mesmo padrão já usado na parte 1 deste projeto).

**Spec:** `docs/superpowers/specs/2026-08-24-eeg-source-reconstruction-design.md`

## Global Constraints

- Backend roda de DENTRO de `backend/`: `cd backend && uvicorn app:app --reload --port 8000`. Módulos usam import plano (`import csv_data`, não `from . import csv_data`) — sem `__init__.py` de pacote, uvicorn e pytest adicionam o cwd ao `sys.path` automaticamente.
- Testes do backend rodam com `cd backend && python -m pytest` (o `-m` garante que o cwd entra no `sys.path`, necessário pros imports planos acima).
- Testes que dependem de `build_source_model()` (baixam o fsaverage e levam minutos) ficam atrás da env var `RUN_SLOW_MNE_TESTS=1` — pulados por padrão.
- Frontend continua servido a partir da raiz do projeto, mas numa porta DIFERENTE da 8000 (que passa a ser do backend) — ex: `python -m http.server 5500`, abrir `http://localhost:5500/eeg-cerebro-3d.html`.
- CSV fixo em `adhdata.csv` na raiz do projeto. `FS = 128.0` Hz (taxa confirmada pelo tamanho do dataset — ~121 sujeitos, 11k–25k linhas cada).
- Montagem de 19 canais UNIFICADA em todo o projeto (backend e frontend): `Fp1,Fp2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T7,T8,P7,P8,Fz,Cz,Pz` (bate exatamente com as colunas do CSV).
- CORS do backend: `allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+"` — permissivo pra qualquer porta local, é um protótipo que não sai do localhost.
- `BACKEND_URL` no frontend = `"http://localhost:8000"`.
- Não existe repositório git neste diretório — não há passo de commit nas tarefas abaixo; "Commit" está substituído por "Marcar checkpoint" (não é uma ação de ferramenta, só o critério de "tarefa concluída, pode seguir pra próxima").

---

## Task 1: `csv_data.py` — leitura de sujeitos e janelas do CSV

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/csv_data.py`
- Test: `backend/tests/test_csv_data.py`

**Interfaces:**
- Produces: `CANAIS_19: list[str]` (19 nomes, ordem fixa), `FS: float` (128.0), `load_csv(caminho) -> pandas.DataFrame`, `list_subjects(df) -> list[dict]` (cada dict: `{"id": str, "classe": str, "duracao_s": float}`), `get_window(df, subject_id: str, t_start: float, t_end: float) -> numpy.ndarray` (shape `(19, n_amostras)`, ordem de canais = `CANAIS_19`; lança `ValueError` com mensagem descritiva se `t_end <= t_start`, sujeito não existe, ou janela cai fora do intervalo gravado).

- [ ] **Step 1: Criar `backend/requirements.txt`**

```
fastapi>=0.110
uvicorn[standard]>=0.27
mne>=1.6
numpy
pandas
pytest
httpx
```

- [ ] **Step 2: Escrever o teste (vai falhar — `csv_data.py` ainda não existe)**

Criar `backend/tests/test_csv_data.py`:

```python
import numpy as np
import pandas as pd
import pytest

from csv_data import CANAIS_19, list_subjects, get_window


def _df_teste():
    linhas = []
    for sid, classe, n in [("suj1", "ADHD", 5), ("suj2", "Control", 3)]:
        for i in range(n):
            linha = {c: float(i) for c in CANAIS_19}
            linha["ID"] = sid
            linha["Class"] = classe
            linhas.append(linha)
    return pd.DataFrame(linhas)


def test_list_subjects():
    df = _df_teste()
    assert list_subjects(df) == [
        {"id": "suj1", "classe": "ADHD", "duracao_s": 5 / 128},
        {"id": "suj2", "classe": "Control", "duracao_s": 3 / 128},
    ]


def test_get_window_janela_valida():
    df = _df_teste()
    janela = get_window(df, "suj1", 0.0, 3 / 128)
    assert janela.shape == (19, 3)
    assert list(janela[0]) == [0.0, 1.0, 2.0]


def test_get_window_sujeito_inexistente():
    df = _df_teste()
    with pytest.raises(ValueError, match="sujeito não encontrado"):
        get_window(df, "suj999", 0.0, 1.0)


def test_get_window_fora_do_intervalo():
    df = _df_teste()
    with pytest.raises(ValueError, match="fora do intervalo"):
        get_window(df, "suj2", 10.0, 11.0)


def test_get_window_t_end_menor_que_t_start():
    df = _df_teste()
    with pytest.raises(ValueError, match="janela inválida"):
        get_window(df, "suj1", 1.0, 0.5)
```

- [ ] **Step 3: Rodar e confirmar que falha por import**

Run: `cd backend && python -m pytest tests/test_csv_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'csv_data'`

- [ ] **Step 4: Implementar `backend/csv_data.py`**

```python
import numpy as np
import pandas as pd

FS = 128.0
CANAIS_19 = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]


def load_csv(caminho):
    """Lê adhdata.csv inteiro (colunas dos 19 canais + ID + Class) uma vez
    só — o resultado fica guardado em app.state pelo FastAPI, não é
    relido a cada request."""
    return pd.read_csv(caminho, usecols=CANAIS_19 + ["ID", "Class"])


def list_subjects(df):
    """Um item por sujeito único, na ordem de primeira aparição no CSV."""
    contagens = df.groupby("ID", sort=False).size()
    classes = df.groupby("ID", sort=False)["Class"].first()
    return [
        {"id": sid, "classe": classes[sid], "duracao_s": contagens[sid] / FS}
        for sid in contagens.index
    ]


def get_window(df, subject_id, t_start, t_end):
    """Linhas do subject_id no intervalo [t_start, t_end) segundos,
    assumindo FS=128Hz. Devolve array (19, n_amostras) na ordem de
    CANAIS_19. Lança ValueError com mensagem descritiva em qualquer
    entrada inválida."""
    if t_end <= t_start:
        raise ValueError(
            f"janela inválida: t_end ({t_end}) deve ser maior que t_start ({t_start})"
        )

    sujeito = df[df["ID"] == subject_id]
    if sujeito.empty:
        raise ValueError(f"sujeito não encontrado: {subject_id}")

    idx_start = int(round(t_start * FS))
    idx_end = int(round(t_end * FS))
    n_amostras = len(sujeito)

    if idx_start >= n_amostras:
        raise ValueError(
            f"janela fora do intervalo gravado: sujeito tem {n_amostras} amostras "
            f"({n_amostras / FS:.2f}s), pedido começa em {t_start:.2f}s"
        )

    janela = sujeito.iloc[idx_start:idx_end]
    if janela.empty:
        raise ValueError("janela pedida não contém nenhuma amostra")

    return np.array([janela[c].to_numpy(dtype=float) for c in CANAIS_19])
```

- [ ] **Step 5: Rodar de novo e confirmar que passa**

Run: `cd backend && python -m pytest tests/test_csv_data.py -v`
Expected: 5 passed

- [ ] **Step 6: Checkpoint**

Tarefa concluída quando os 5 testes passam. Sem commit (projeto não é repositório git).

---

## Task 2: `mne_setup.py` — montagem, forward model e inverse operator

**Files:**
- Create: `backend/mne_setup.py`
- Test: `backend/tests/test_mne_setup.py`

**Interfaces:**
- Consumes: `csv_data.CANAIS_19` (lista de 19 nomes de canal).
- Produces: `CANAIS_19` e `FS` (reexportados, mesmos valores de `csv_data`), `build_info() -> mne.Info` (rápido, sem download — 19 canais EEG com montagem `standard_1020` aplicada), `build_source_model() -> tuple[forward, inverse_operator, n_vertices]` (lento — baixa o fsaverage na primeira vez, usado por `app.py` e por `mne_infer.py`).

- [ ] **Step 1: Escrever o teste rápido (vai falhar — `mne_setup.py` ainda não existe)**

Criar `backend/tests/test_mne_setup.py`:

```python
import os

import numpy as np
import pytest

from mne_setup import build_info, build_source_model, CANAIS_19, FS

RODAR_TESTES_LENTOS = "RUN_SLOW_MNE_TESTS" in os.environ


def test_build_info_rapido():
    info = build_info()
    assert info["sfreq"] == FS
    assert info["ch_names"] == CANAIS_19
    assert len(info["chs"]) == 19
    # cada canal EEG deve ter posição 3D válida (não [0,0,0]) depois do
    # standard_1020 montage ser aplicado
    for ch in info["chs"]:
        assert not np.allclose(ch["loc"][:3], 0.0)


@pytest.mark.skipif(
    not RODAR_TESTES_LENTOS,
    reason="baixa o fsaverage (~centenas de MB) e leva minutos — rode com RUN_SLOW_MNE_TESTS=1",
)
def test_build_source_model_integracao():
    forward, inverse_operator, n_vertices = build_source_model()
    assert n_vertices > 1000
    assert forward is not None
    assert inverse_operator is not None
```

- [ ] **Step 2: Rodar e confirmar que falha por import**

Run: `cd backend && python -m pytest tests/test_mne_setup.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mne_setup'`

- [ ] **Step 3: Implementar `backend/mne_setup.py`**

```python
from pathlib import Path

import mne

from csv_data import CANAIS_19, FS

__all__ = ["CANAIS_19", "FS", "build_info", "build_source_model"]


def build_info():
    """mne.Info com os 19 canais EEG na montagem standard_1020. Rápido,
    sem download — seguro de chamar em qualquer teste."""
    info = mne.create_info(ch_names=CANAIS_19, sfreq=FS, ch_types="eeg")
    montagem = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montagem, on_missing="raise")
    return info


def build_source_model():
    """Baixa (se necessário) o fsaverage e monta forward + inverse
    operator usando a superfície cortical ico-5 já pronta que vem junto
    (sem precisar de FreeSurfer instalado). Lento na primeira vez
    (download + alguns minutos de cálculo) — chamado uma vez no startup
    do servidor, nunca por request.

    Devolve (forward, inverse_operator, n_vertices).
    """
    fs_dir = Path(mne.datasets.fetch_fsaverage(verbose=False))
    subjects_dir = fs_dir.parent
    trans = "fsaverage"
    src_path = fs_dir / "bem" / "fsaverage-ico-5-src.fif"
    bem_path = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"

    src = mne.read_source_spaces(src_path)
    info = build_info()

    forward = mne.make_forward_solution(
        info, trans=trans, src=src, bem=str(bem_path), eeg=True, meg=False
    )

    noise_cov = mne.make_ad_hoc_cov(info)

    inverse_operator = mne.minimum_norm.make_inverse_operator(
        info, forward, noise_cov, loose=0.2, depth=0.8
    )

    n_vertices = sum(len(s["vertno"]) for s in forward["src"])

    return forward, inverse_operator, n_vertices
```

- [ ] **Step 4: Rodar o teste rápido e confirmar que passa (o lento fica pulado)**

Run: `cd backend && python -m pytest tests/test_mne_setup.py -v`
Expected: `test_build_info_rapido PASSED`, `test_build_source_model_integracao SKIPPED`

- [ ] **Step 5: Rodar o teste lento uma vez, manualmente, pra validar a integração de verdade**

Run: `cd backend && RUN_SLOW_MNE_TESTS=1 python -m pytest tests/test_mne_setup.py -v -s`
Expected: PASSED (pode levar vários minutos na primeira vez — baixa o fsaverage). Se der erro de nome de arquivo (`fsaverage-ico-5-src.fif` ou `fsaverage-5120-5120-5120-bem-sol.fif` não encontrados), rodar `python -c "import mne; from pathlib import Path; print(list((Path(mne.datasets.fetch_fsaverage()) / 'bem').iterdir()))"` pra ver os nomes reais nessa versão do MNE instalada e ajustar `src_path`/`bem_path` de acordo.

- [ ] **Step 6: Checkpoint**

Tarefa concluída quando o teste rápido passa sempre, e o teste lento passou pelo menos uma vez manualmente.

---

## Task 3: `mne_infer.py` — aplicar a reconstrução de fontes numa janela

**Files:**
- Create: `backend/mne_infer.py`
- Test: `backend/tests/test_mne_infer.py`

**Interfaces:**
- Consumes: `mne_setup.build_info()`, `mne_setup.build_source_model()` (para o teste), `inverse_operator` (objeto MNE, vindo de `build_source_model()`).
- Produces: `apply_source_localization(dados_janela: numpy.ndarray, inverse_operator, method: str = "dSPM") -> numpy.ndarray` — `dados_janela` tem shape `(19, n_amostras)` na ordem de `CANAIS_19`; devolve array 1D de tamanho `n_vertices` (média temporal da janela, um valor por vértice do espaço de fontes).

- [ ] **Step 1: Escrever o teste (vai falhar — `mne_infer.py` ainda não existe)**

Criar `backend/tests/test_mne_infer.py`:

```python
import os

import numpy as np
import pytest

from mne_setup import build_source_model, CANAIS_19, FS
from mne_infer import apply_source_localization

RODAR_TESTES_LENTOS = "RUN_SLOW_MNE_TESTS" in os.environ


@pytest.mark.skipif(
    not RODAR_TESTES_LENTOS,
    reason="depende de build_source_model() — rode com RUN_SLOW_MNE_TESTS=1",
)
def test_apply_source_localization_formato_saida():
    _forward, inverse_operator, n_vertices = build_source_model()
    rng = np.random.default_rng(0)
    janela = rng.normal(scale=20.0, size=(len(CANAIS_19), int(2 * FS)))

    valores = apply_source_localization(janela, inverse_operator, method="dSPM")

    assert valores.shape == (n_vertices,)
    assert np.all(np.isfinite(valores))
```

- [ ] **Step 2: Rodar e confirmar que falha por import**

Run: `cd backend && RUN_SLOW_MNE_TESTS=1 python -m pytest tests/test_mne_infer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mne_infer'`

- [ ] **Step 3: Implementar `backend/mne_infer.py`**

```python
import mne

from mne_setup import build_info


def apply_source_localization(dados_janela, inverse_operator, method="dSPM"):
    """dados_janela: numpy array (19, n_amostras) na ordem de CANAIS_19,
    em microvolts. Aplica average reference (CAR) e a inverse solution
    já pronta, devolve um array 1D com a média temporal por vértice do
    espaço de fontes."""
    info = build_info()
    raw = mne.io.RawArray(dados_janela * 1e-6, info, verbose=False)  # MNE espera Volts
    raw.set_eeg_reference("average", projection=True, verbose=False)

    stc = mne.minimum_norm.apply_inverse_raw(
        raw, inverse_operator, lambda2=1.0 / 9.0, method=method, verbose=False
    )

    return stc.data.mean(axis=1)
```

- [ ] **Step 4: Rodar o teste lento manualmente e confirmar que passa**

Run: `cd backend && RUN_SLOW_MNE_TESTS=1 python -m pytest tests/test_mne_infer.py -v -s`
Expected: PASSED

- [ ] **Step 5: Checkpoint**

Tarefa concluída quando o teste lento passa pelo menos uma vez manualmente.

---

## Task 4: `app.py` — FastAPI com `/subjects` e `/source-localization`

**Files:**
- Create: `backend/app.py`
- Test: `backend/tests/test_app.py`

**Interfaces:**
- Consumes: `csv_data.load_csv/list_subjects/get_window/CANAIS_19`, `mne_setup.build_source_model`, `mne_infer.apply_source_localization`.
- Produces: `app` (instância `FastAPI`), `GET /subjects` → `list[{"id","classe","duracao_s"}]`, `POST /source-localization` (body `{"subject_id","t_start","t_end","method"}`, `method` default `"dSPM"`) → `{"values": list[float], "n_vertices": int, "time": float, "method": str}` em caso de sucesso, ou `400` com `{"detail": str}` em caso de sujeito inexistente, janela inválida, ou erro do MNE.

- [ ] **Step 1: Escrever os testes (vão falhar — `app.py` ainda não existe)**

Criar `backend/tests/test_app.py`:

```python
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# IMPORTANTE: TestClient(app) SEM "with" não dispara o lifespan (que
# baixaria o fsaverage e demoraria minutos) — os testes setam app.state
# manualmente e testam só a camada HTTP, sem rodar MNE de verdade.
from app import app
from csv_data import CANAIS_19


def _df_teste():
    linhas = []
    for i in range(300):  # 300/128 ≈ 2.3s, o bastante pra uma janela de 2s
        linha = {c: float(i) for c in CANAIS_19}
        linha["ID"] = "suj1"
        linha["Class"] = "ADHD"
        linhas.append(linha)
    return pd.DataFrame(linhas)


@pytest.fixture
def client():
    app.state.df = _df_teste()
    app.state.inverse_operator = object()  # não usado diretamente — apply_source_localization é mockado
    app.state.n_vertices = 4
    return TestClient(app)


def test_get_subjects(client):
    resp = client.get("/subjects")
    assert resp.status_code == 200
    assert resp.json() == [{"id": "suj1", "classe": "ADHD", "duracao_s": 300 / 128}]


def test_source_localization_sucesso(client, monkeypatch):
    def fake_apply(janela, inverse_operator, method="dSPM"):
        assert janela.shape == (19, 256)  # 2s * 128Hz
        return np.array([1.0, 2.0, 3.0, 4.0])

    monkeypatch.setattr("app.mne_infer.apply_source_localization", fake_apply)

    resp = client.post(
        "/source-localization",
        json={"subject_id": "suj1", "t_start": 0.0, "t_end": 2.0, "method": "dSPM"},
    )
    assert resp.status_code == 200
    corpo = resp.json()
    assert corpo["values"] == [1.0, 2.0, 3.0, 4.0]
    assert corpo["n_vertices"] == 4
    assert corpo["method"] == "dSPM"


def test_source_localization_sujeito_inexistente(client):
    resp = client.post(
        "/source-localization",
        json={"subject_id": "naoexiste", "t_start": 0.0, "t_end": 2.0},
    )
    assert resp.status_code == 400
    assert "não encontrado" in resp.json()["detail"]


def test_source_localization_erro_mne(client, monkeypatch):
    def fake_apply_com_erro(janela, inverse_operator, method="dSPM"):
        raise RuntimeError("falha simulada")

    monkeypatch.setattr("app.mne_infer.apply_source_localization", fake_apply_com_erro)

    resp = client.post(
        "/source-localization",
        json={"subject_id": "suj1", "t_start": 0.0, "t_end": 2.0},
    )
    assert resp.status_code == 400
    assert "falha simulada" in resp.json()["detail"]
```

- [ ] **Step 2: Rodar e confirmar que falha por import**

Run: `cd backend && python -m pytest tests/test_app.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 3: Implementar `backend/app.py`**

```python
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import csv_data
import mne_infer
import mne_setup

CSV_PATH = Path(__file__).resolve().parent.parent / "adhdata.csv"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.df = csv_data.load_csv(CSV_PATH)
    forward, inverse_operator, n_vertices = mne_setup.build_source_model()
    app.state.forward = forward
    app.state.inverse_operator = inverse_operator
    app.state.n_vertices = n_vertices
    print(
        f"[startup] forward/inverse prontos: {n_vertices} vértices, "
        f"{len(csv_data.CANAIS_19)} canais"
    )
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/subjects")
def get_subjects():
    return csv_data.list_subjects(app.state.df)


class SourceLocalizationRequest(BaseModel):
    subject_id: str
    t_start: float
    t_end: float
    method: str = "dSPM"


@app.post("/source-localization")
def post_source_localization(req: SourceLocalizationRequest):
    try:
        janela = csv_data.get_window(app.state.df, req.subject_id, req.t_start, req.t_end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        valores = mne_infer.apply_source_localization(
            janela, app.state.inverse_operator, method=req.method
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"falha no MNE: {e}")

    return {
        "values": valores.tolist(),
        "n_vertices": app.state.n_vertices,
        "time": req.t_start,
        "method": req.method,
    }
```

- [ ] **Step 4: Rodar de novo e confirmar que passa**

Run: `cd backend && python -m pytest tests/test_app.py -v`
Expected: 4 passed

- [ ] **Step 5: Verificação manual end-to-end (com MNE de verdade)**

Run: `cd backend && uvicorn app:app --reload --port 8000` (primeira vez demora — baixa o fsaverage e monta forward/inverse; espera log `[startup] forward/inverse prontos: N vértices, 19 canais`).

Em outro terminal:
```bash
curl http://localhost:8000/subjects | head -c 300
curl -X POST http://localhost:8000/source-localization \
  -H "Content-Type: application/json" \
  -d '{"subject_id": "v107", "t_start": 2.0, "t_end": 4.0, "method": "dSPM"}'
```
Expected: `/subjects` devolve uma lista JSON não vazia; `/source-localization` devolve `{"values": [...], "n_vertices": N, ...}` com `len(values) == N` e `N` igual ao logado no startup.

- [ ] **Step 6: Checkpoint**

Tarefa concluída quando os 4 testes rápidos passam e a verificação manual com `curl` funciona.

---

## Task 5: `export_fsaverage_mesh.py` — exportar a malha fsaverage pro frontend

**Files:**
- Create: `backend/export_fsaverage_mesh.py`

**Interfaces:**
- Consumes: `mne_setup.build_source_model()`.
- Produces (arquivos em disco, não símbolos Python): `assets/fsaverage-cortex.obj` (geometria — hemisfério esquerdo seguido do direito, centralizado na origem) e `assets/fsaverage-cortex-vertices.json` (array JSON `[[x,y,z], ...]`, MESMA ordem de vértices que `values[]` em `/source-localization` e que as linhas `v` do `.obj` — é o contrato de correspondência que o frontend usa na Task 8).

- [ ] **Step 1: Implementar `backend/export_fsaverage_mesh.py`**

```python
"""
Roda uma vez, offline: exporta a superfície cortical fsaverage (a MESMA
usada pelo forward model em mne_setup.build_source_model) para um .obj
combinando hemisfério esquerdo + direito, e um .json com a mesma lista
de vértices em array puro.

A ORDEM dos vértices em AMBOS os arquivos é: primeiro os vértices do
hemisfério esquerdo (forward['src'][0]['vertno']), depois os do direito
(forward['src'][1]['vertno']) — é exatamente a ordem que
mne_infer.apply_source_localization devolve em stc.data. Ou seja: o
vértice i do .obj / .json corresponde a values[i] na resposta de
POST /source-localization. O frontend (Task 8) usa o .json pra montar
essa correspondência, já que o Three.js OBJLoader pode duplicar
vértices por face e embaralhar essa ordem original.

Uso: cd backend && python export_fsaverage_mesh.py
Saída: ../assets/fsaverage-cortex.obj, ../assets/fsaverage-cortex-vertices.json
"""
import json
from pathlib import Path

import numpy as np

from mne_setup import build_source_model


def exportar(caminho_obj, caminho_vertices_json):
    forward, _inverse_operator, _n_vertices = build_source_model()
    src = forward["src"]

    pontos = []
    for hemi in src:
        rr = hemi["rr"][hemi["vertno"]]  # (n_vert_hemi, 3), metros, espaço MRI
        pontos.append(rr)
    pontos = np.concatenate(pontos, axis=0)  # (n_vertices, 3)

    centro = pontos.mean(axis=0)
    pontos = pontos - centro

    triangulos = []
    offset = 0
    for hemi in src:
        vertno = hemi["vertno"]
        mapa = {v: i + offset for i, v in enumerate(vertno)}
        for tri in hemi["use_tris"]:
            if all(v in mapa for v in tri):
                triangulos.append([mapa[v] for v in tri])
        offset += len(vertno)

    with open(caminho_obj, "w") as f:
        f.write(f"# fsaverage cortex — {len(pontos)} vértices, {len(triangulos)} faces\n")
        for x, y, z in pontos:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in triangulos:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")

    with open(caminho_vertices_json, "w") as f:
        json.dump(pontos.tolist(), f)

    print(
        f"[export_fsaverage_mesh] {len(pontos)} vértices, {len(triangulos)} faces -> "
        f"{caminho_obj}, {caminho_vertices_json}"
    )
    return len(pontos)


if __name__ == "__main__":
    pasta_assets = Path(__file__).resolve().parent.parent / "assets"
    exportar(
        pasta_assets / "fsaverage-cortex.obj",
        pasta_assets / "fsaverage-cortex-vertices.json",
    )
```

- [ ] **Step 2: Rodar e verificar a saída**

Run: `cd backend && python export_fsaverage_mesh.py`
Expected: imprime `[export_fsaverage_mesh] N vértices, M faces -> ...`, cria `assets/fsaverage-cortex.obj` e `assets/fsaverage-cortex-vertices.json`. Se `hemi["use_tris"]` não existir nessa versão do MNE (erro `KeyError`), rodar `python -c "import mne; from mne_setup import build_source_model; f,_,_ = build_source_model(); print(f['src'][0].keys())"` pra ver as chaves reais e trocar `"use_tris"` pela chave equivalente (`"tris"` é o fallback mais provável).

- [ ] **Step 3: Conferir consistência entre os dois arquivos**

Run:
```bash
python -c "
import json
linhas_v = sum(1 for l in open('../assets/fsaverage-cortex.obj') if l.startswith('v '))
vertices = json.load(open('../assets/fsaverage-cortex-vertices.json'))
print('linhas v no obj:', linhas_v, '| itens no json:', len(vertices))
assert linhas_v == len(vertices)
print('OK: contagens batem')
"
```
Expected: `OK: contagens batem`

- [ ] **Step 4: Checkpoint**

Tarefa concluída quando os dois arquivos existem em `assets/` e as contagens batem.

---

## Task 6: Frontend — unificar a montagem de 19 canais

**Files:**
- Modify: `eeg-cerebro-3d.html`

**Interfaces:**
- Consumes: nada de tarefas anteriores (é só frontend).
- Produces: `CANAIS_19` e `POS_10_20` atualizados — usados por todas as tarefas de frontend seguintes e por qualquer código já existente que dependia da lista antiga.

- [ ] **Step 1: Trocar `CANAIS_19` e `POS_10_20`**

Em `eeg-cerebro-3d.html`, substituir (por volta da linha 153-164):

```js
// =========================================================================
// 1. MONTAGEM — os 19 canais exatos de Dados_100sbj_CORRIGIDO
// =========================================================================
const CANAIS_19 = ["Fp2","Fp1","F7","T3","T5","O1","Oz","O2","T6","T4","F8","F3","Fz","F4","C3","P3","Pz","P4","C4"];

const POS_10_20 = {
  "Fp1": [90, -18], "Fp2": [90, 18],
  "F7":  [90, -54], "F3":  [63, -45], "Fz":  [45, 0], "F4":  [63, 45], "F8":  [90, 54],
  "T3":  [90, -90], "C3":  [45, -90], "C4":  [45, 90], "T4":  [90, 90],
  "T5":  [90, -126],"P3":  [63, -135],"Pz":  [45, 180],"P4":  [63, 135],"T6":  [90, 126],
  "O1":  [90, -162],"Oz":  [90, 180], "O2":  [90, 162],
};
```

por:

```js
// =========================================================================
// 1. MONTAGEM — os 19 canais exatos de adhdata.csv (Fp1,Fp2,F3,F4,C3,C4,
//    P3,P4,O1,O2,F7,F8,T7,T8,P7,P8,Fz,Cz,Pz — nomenclatura moderna,
//    inclui Cz como canal real, sem Oz)
// =========================================================================
const CANAIS_19 = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T7","T8","P7","P8","Fz","Cz","Pz"];

const POS_10_20 = {
  "Fp1": [90, -18], "Fp2": [90, 18],
  "F7":  [90, -54], "F3":  [63, -45], "Fz":  [45, 0], "F4":  [63, 45], "F8":  [90, 54],
  "T7":  [90, -90], "C3":  [45, -90], "C4":  [45, 90], "T8":  [90, 90],
  "P7":  [90, -126],"P3":  [63, -135],"Pz":  [45, 180],"P4":  [63, 135],"P8":  [90, 126],
  "O1":  [90, -162], "O2":  [90, 162],
  "Cz":  [0, 0],
};
```

- [ ] **Step 2: Remover `equivalenteModerno` (ficou obsoleta — os nomes já são os modernos)**

Substituir (por volta da linha 618):

```js
function equivalenteModerno(nome) { return { T3: "T7", T4: "T8", T5: "P7", T6: "P8" }[nome]; }

function atualizarTooltip(el) {
  eletrodoSobre = el;
  canalRealcado = el ? el.nome : null;
  if (el) {
    const eq = equivalenteModerno(el.nome);
    const nomesRegiao = { frontal: "Lobo frontal", central: "Região central", parietal: "Lobo parietal", temporal: "Lobo temporal", occipital: "Lobo occipital" };
    tooltip.innerHTML = `<b>${el.nome}</b>${eq ? ` <span class="sub">(= ${eq})</span>` : ""}<br>` +
      `<span class="sub">${nomesRegiao[el.regiao]} · relevância ${(relevancia[el.nome]*100).toFixed(0)}% em ${bandaAtiva}</span>`;
  }
}
```

por:

```js
function atualizarTooltip(el) {
  eletrodoSobre = el;
  canalRealcado = el ? el.nome : null;
  if (el) {
    const nomesRegiao = { frontal: "Lobo frontal", central: "Região central", parietal: "Lobo parietal", temporal: "Lobo temporal", occipital: "Lobo occipital" };
    tooltip.innerHTML = `<b>${el.nome}</b><br>` +
      `<span class="sub">${nomesRegiao[el.regiao]} · relevância ${(relevancia[el.nome]*100).toFixed(0)}% em ${bandaAtiva}</span>`;
  }
}
```

- [ ] **Step 3: Atualizar o comentário do hook de dados reais**

Substituir (por volta da linha 808-816):

```js
// =========================================================================
// 10. HOOK PARA DADOS REAIS
//
//    window.setEEGData((nomeCanal, tempoSegundos) => valorEmMicrovolts)
//
//    Nomes esperados: exatamente os 19 de CANAIS_19 (grafia antiga
//    T3/T4/T5/T6, igual às colunas do seu CSV).
//    window.clearEEGData() volta ao sinal sintético.
// =========================================================================
```

por:

```js
// =========================================================================
// 10. HOOK PARA DADOS REAIS
//
//    window.setEEGData((nomeCanal, tempoSegundos) => valorEmMicrovolts)
//
//    Nomes esperados: exatamente os 19 de CANAIS_19 (Fp1,Fp2,F3,F4,C3,
//    C4,P3,P4,O1,O2,F7,F8,T7,T8,P7,P8,Fz,Cz,Pz — mesma grafia das
//    colunas de adhdata.csv).
//    window.clearEEGData() volta ao sinal sintético.
// =========================================================================
```

- [ ] **Step 4: Checar sintaxe**

Run:
```bash
node -e "
const fs = require('fs');
const html = fs.readFileSync('eeg-cerebro-3d.html', 'utf8');
const match = html.match(/<script>([\s\S]*)<\/script>/);
new Function(match[1]);
console.log('OK: script parses without syntax errors');
"
```
Expected: `OK: script parses without syntax errors`

- [ ] **Step 5: Verificação visual (Playwright)**

Servir a pasta (`python -m http.server 5500` a partir da raiz do projeto) e abrir `http://localhost:5500/eeg-cerebro-3d.html` via Playwright MCP (`browser_navigate` + `browser_snapshot`/`browser_take_screenshot`). Conferir: painel de traçados mostra 19 rótulos incluindo `Cz`, `T7`, `T8`, `P7`, `P8`, e NÃO mostra `Oz`, `T3`, `T4`, `T5`, `T6`; o cérebro 3D carrega normalmente com 19 eletrodos (o eletrodo `Cz` deve aparecer no topo da cabeça). Sem erros novos no console.

- [ ] **Step 6: Checkpoint**

Tarefa concluída quando a verificação visual confirma os 19 rótulos corretos e nenhuma quebra visual.

---

## Task 7: Frontend — abas "Sensores" / "Fontes corticais" (scaffolding, sem backend ainda)

**Files:**
- Modify: `eeg-cerebro-3d.html`

**Interfaces:**
- Consumes: `malhaCerebro`, `grupoEletrodos`, `#banda` (já existentes).
- Produces: função `ativarAba(aba)` (aba = `"sensores"` | `"fontes"`), variável `abaAtiva`, elementos `#hud-abas`, `#controles-fontes` — usados pelas Tasks 8-10 pra saber quando carregar/mostrar a malha fsaverage.

- [ ] **Step 1: Adicionar CSS das abas e da barra de controles da aba de fontes**

Em `eeg-cerebro-3d.html`, depois da regra `#controles button.ativo { ... }` (por volta da linha 75), adicionar:

```css
  #hud-abas { top: 16px; right: 20px; display: flex; gap: 6px; pointer-events: auto; }
  #hud-abas button {
    font-family: inherit; font-size: 11px;
    background: rgba(154,166,191,0.07); border: 1px solid rgba(154,166,191,0.18);
    color: var(--ink-dim); border-radius: 5px; padding: 5px 10px; cursor: pointer; transition: all 0.15s;
  }
  #hud-abas button:hover { color: var(--ink); border-color: rgba(154,166,191,0.35); }
  #hud-abas button.ativo { color: #eef3fb; background: rgba(154,166,191,0.16); border-color: rgba(154,166,191,0.45); }

  #controles-fontes {
    position: absolute; bottom: 18px; right: 22px; display: none; gap: 8px;
    align-items: center; pointer-events: auto; font-size: 11px; color: var(--ink-dim);
  }
  #controles-fontes select, #controles-fontes button {
    font-family: inherit; font-size: 11px;
    background: rgba(154,166,191,0.07); border: 1px solid rgba(154,166,191,0.18);
    color: var(--ink-dim); border-radius: 5px; padding: 5px 9px; cursor: pointer;
  }
  #controles-fontes button:hover { color: var(--ink); border-color: rgba(154,166,191,0.35); }
  #controles-fontes input[type=range] { width: 140px; }
  #fontes-status { max-width: 260px; line-height: 1.5; }
```

- [ ] **Step 2: Adicionar o HTML das abas e da barra de controles**

Em `eeg-cerebro-3d.html`, dentro de `<div id="painel-3d">`, depois de `<div id="status"></div>` (por volta da linha 115), adicionar:

```html
    <div class="hud" id="hud-abas">
      <button id="aba-sensores" class="ativo">Sensores</button>
      <button id="aba-fontes">Fontes corticais (MNE)</button>
    </div>
```

E depois de `<div id="banda">...</div>` (por volta da linha 123), adicionar:

```html
    <div id="controles-fontes">
      <select id="fontes-sujeito"></select>
      <input type="range" id="fontes-tempo" min="0" max="0" step="0.5" value="0">
      <span id="fontes-tempo-label">0.0s</span>
      <button id="fontes-rodar">Rodar reconstrução</button>
      <span id="fontes-status"></span>
    </div>
```

- [ ] **Step 3: Adicionar a lógica de troca de aba**

Em `eeg-cerebro-3d.html`, logo depois do bloco `document.getElementById("banda").addEventListener(...)` (por volta da linha 672), adicionar:

```js
// =========================================================================
// 11. ABAS — "Sensores" (existente) / "Fontes corticais" (MNE). Reusa o
//    MESMO renderer/câmera/cena; só troca o que fica visível e qual
//    barra de controles aparece embaixo.
// =========================================================================
let abaAtiva = "sensores";
const btnAbaSensores = document.getElementById("aba-sensores");
const btnAbaFontes = document.getElementById("aba-fontes");
const controlesFontesEl = document.getElementById("controles-fontes");
const bandaEl = document.getElementById("banda");

function ativarAba(aba) {
  abaAtiva = aba;
  btnAbaSensores.classList.toggle("ativo", aba === "sensores");
  btnAbaFontes.classList.toggle("ativo", aba === "fontes");
  bandaEl.style.display = aba === "sensores" ? "flex" : "none";
  controlesFontesEl.style.display = aba === "fontes" ? "flex" : "none";
  if (malhaCerebro) malhaCerebro.visible = aba === "sensores";
  grupoEletrodos.visible = aba === "sensores";
  if (typeof malhaFontes !== "undefined" && malhaFontes) malhaFontes.visible = aba === "fontes";
  if (aba === "fontes" && typeof aoAtivarAbaFontes === "function") aoAtivarAbaFontes();
}

btnAbaSensores.addEventListener("click", () => ativarAba("sensores"));
btnAbaFontes.addEventListener("click", () => ativarAba("fontes"));
```

Nota: `aoAtivarAbaFontes` ainda não existe — vai ser definida na Task 8/9 (carregamento da malha e da lista de sujeitos, sob demanda). Por enquanto o `typeof ... === "function"` evita erro se essa tarefa rodar sozinha.

- [ ] **Step 4: Checar sintaxe**

Run (mesmo comando node da Task 6, Step 4).
Expected: `OK: script parses without syntax errors`

- [ ] **Step 5: Verificação visual (Playwright)**

Servir e abrir a página. Clicar em "Fontes corticais (MNE)": os eletrodos e o cérebro atual devem SUMIR, a barra `#banda` deve sumir, a barra `#controles-fontes` (vazia por enquanto, sem sujeitos) deve aparecer. Clicar de volta em "Sensores": tudo volta ao normal.

- [ ] **Step 6: Checkpoint**

Tarefa concluída quando a troca de aba esconde/mostra os elementos certos, sem quebrar a aba de sensores.

---

## Task 8: Frontend — carregar a malha fsaverage e a correspondência de vértices

**Files:**
- Modify: `eeg-cerebro-3d.html`

**Interfaces:**
- Consumes: `assets/fsaverage-cortex.obj` e `assets/fsaverage-cortex-vertices.json` (Task 5), `scene`, `THREE.OBJLoader` (já existentes), `abaAtiva` (Task 7).
- Produces: `malhaFontes` (THREE.Mesh ou `null`), `indiceValorPorVertice` (Int32Array ou `null` — por vértice CARREGADO na geometria, o índice correspondente em `values[]`), função `aoAtivarAbaFontes()` (chamada pela Task 7 ao entrar na aba — dispara o carregamento uma vez).

- [ ] **Step 1: Adicionar o carregamento da malha + correspondência de vértices**

Em `eeg-cerebro-3d.html`, logo depois do bloco de abas adicionado na Task 7, adicionar:

```js
// =========================================================================
// 12. MALHA FSAVERAGE — carregada sob demanda ao entrar na aba "Fontes
//    corticais". O Three.js OBJLoader pode duplicar vértices por face
//    (geometria não-indexada), então NÃO dá pra assumir que o vértice i
//    carregado é o vértice i do JSON/backend — por isso
//    construirCorrespondenciaVertices() casa cada posição carregada com
//    a lista original por coordenada (x,y,z arredondada).
// =========================================================================
let malhaFontes = null;
let carregandoFontes = false;
let indiceValorPorVertice = null;
const fontesStatusEl = document.getElementById("fontes-status");

function construirCorrespondenciaVertices(geo, verticesOriginais) {
  const chave = (x, y, z) => `${x.toFixed(4)},${y.toFixed(4)},${z.toFixed(4)}`;
  const mapa = new Map();
  for (let i = 0; i < verticesOriginais.length; i++) {
    const [x, y, z] = verticesOriginais[i];
    mapa.set(chave(x, y, z), i);
  }
  const pos = geo.attributes.position;
  const n = pos.count;
  const indices = new Int32Array(n);
  let semCorrespondencia = 0;
  for (let i = 0; i < n; i++) {
    const k = chave(pos.getX(i), pos.getY(i), pos.getZ(i));
    const idx = mapa.get(k);
    if (idx === undefined) { semCorrespondencia++; indices[i] = 0; }
    else indices[i] = idx;
  }
  if (semCorrespondencia > 0) {
    console.warn(`${semCorrespondencia}/${n} vértices carregados sem correspondência exata no JSON de origem.`);
  }
  return indices;
}

function carregarMalhaFontes() {
  if (malhaFontes || carregandoFontes) return;
  carregandoFontes = true;
  fontesStatusEl.textContent = "Carregando malha fsaverage…";

  Promise.all([
    fetch("assets/fsaverage-cortex-vertices.json").then((r) => {
      if (!r.ok) throw new Error("fsaverage-cortex-vertices.json não encontrado");
      return r.json();
    }),
    new Promise((resolve, reject) => {
      new THREE.OBJLoader().load("assets/fsaverage-cortex.obj", resolve, undefined, reject);
    }),
  ])
    .then(([verticesOriginais, obj]) => {
      let malha = null;
      obj.traverse((child) => { if (child.isMesh) malha = child; });
      if (!malha) throw new Error("Nenhum mesh encontrado em fsaverage-cortex.obj.");

      const geo = malha.geometry.index ? malha.geometry.toNonIndexed() : malha.geometry;
      const nVerts = geo.attributes.position.count;
      const cores = new Float32Array(nVerts * 3);
      for (let i = 0; i < nVerts * 3; i++) cores[i] = 0.35;
      geo.setAttribute("color", new THREE.BufferAttribute(cores, 3));
      geo.computeVertexNormals();

      indiceValorPorVertice = construirCorrespondenciaVertices(geo, verticesOriginais);

      const mat = new THREE.MeshStandardMaterial({
        vertexColors: true, roughness: 0.6, metalness: 0.05, side: THREE.DoubleSide,
      });
      malhaFontes = new THREE.Mesh(geo, mat);
      malhaFontes.userData.cores = cores;
      malhaFontes.visible = abaAtiva === "fontes";
      scene.add(malhaFontes);
      fontesStatusEl.textContent = "";
      carregandoFontes = false;
    })
    .catch((err) => {
      fontesStatusEl.textContent =
        "Não consegui carregar a malha fsaverage. Rode export_fsaverage_mesh.py dentro de backend/ primeiro.";
      carregandoFontes = false;
      console.error(err);
    });
}

function aoAtivarAbaFontes() {
  carregarMalhaFontes();
}
```

- [ ] **Step 2: Checar sintaxe**

Run (mesmo comando node da Task 6, Step 4).
Expected: `OK: script parses without syntax errors`

- [ ] **Step 3: Verificação visual (Playwright) — com os assets da Task 5 já gerados**

Servir e abrir a página, clicar em "Fontes corticais (MNE)". Esperar o texto de status sumir (carregou) e conferir visualmente que uma malha cinza-azulada de superfície cortical aparece na cena (formato de cérebro "aberto"/dobrado, bem diferente do `brain.obj` liso da aba Sensores). Abrir o console do navegador e conferir que não aparece o aviso `vértices carregados sem correspondência exata` (ou, se aparecer, que é uma fração pequena — indica arredondamento de ponto flutuante entre Python e o parser do OBJLoader, não um bug de lógica).

Se `assets/fsaverage-cortex.obj`/`.json` ainda não existirem (Task 5 não rodada nesta máquina), a mensagem de erro deve aparecer no lugar do texto de status — confirmar isso também.

- [ ] **Step 4: Checkpoint**

Tarefa concluída quando a malha aparece corretamente colorida em cinza neutro e a correspondência de vértices não perde uma fração relevante (>1%) dos pontos.

---

## Task 9: Frontend — seletor de sujeito e slider de tempo

**Files:**
- Modify: `eeg-cerebro-3d.html`

**Interfaces:**
- Consumes: `BACKEND_URL` (nova constante desta task), `GET /subjects` (Task 4), `#fontes-sujeito`/`#fontes-tempo`/`#fontes-tempo-label` (Task 7).
- Produces: `carregarSujeitos()`, `ajustarSliderParaSujeito(duracaoS)` — chamadas por `aoAtivarAbaFontes()` (Task 8) e usadas pela Task 10 pra saber sujeito/janela escolhidos.

- [ ] **Step 1: Adicionar a constante do backend e o carregamento de sujeitos**

Em `eeg-cerebro-3d.html`, logo no topo do `<script>`, depois de `const CAMINHO_BRAIN_OBJ = "assets/brain.obj";` (por volta da linha 151), adicionar:

```js
const BACKEND_URL = "http://localhost:8000";
```

Depois do bloco da Task 8 (função `aoAtivarAbaFontes`), adicionar:

```js
// =========================================================================
// 13. SUJEITO + JANELA DE TEMPO — populado a partir de GET /subjects.
// =========================================================================
let sujeitosCarregados = false;
const selectSujeito = document.getElementById("fontes-sujeito");
const sliderTempo = document.getElementById("fontes-tempo");
const labelTempo = document.getElementById("fontes-tempo-label");

function ajustarSliderParaSujeito(duracaoS) {
  const max = Math.max(0, duracaoS - 2); // janela fixa de 2s, não deixa passar do fim gravado
  sliderTempo.max = max.toFixed(1);
  sliderTempo.value = "0";
  labelTempo.textContent = "0.0s";
}

function carregarSujeitos() {
  if (sujeitosCarregados) return;
  sujeitosCarregados = true;
  fetch(`${BACKEND_URL}/subjects`)
    .then((r) => { if (!r.ok) throw new Error("status " + r.status); return r.json(); })
    .then((sujeitos) => {
      selectSujeito.innerHTML = "";
      for (const s of sujeitos) {
        const opt = document.createElement("option");
        opt.value = s.id;
        opt.textContent = `${s.id} (${s.classe}, ${s.duracao_s.toFixed(1)}s)`;
        opt.dataset.duracao = s.duracao_s;
        selectSujeito.appendChild(opt);
      }
      if (sujeitos.length) ajustarSliderParaSujeito(sujeitos[0].duracao_s);
    })
    .catch((err) => {
      sujeitosCarregados = false; // permite tentar de novo ao reentrar na aba
      fontesStatusEl.textContent =
        `Backend Python não respondeu em ${BACKEND_URL}. Rode "uvicorn app:app --reload" dentro de backend/.`;
      console.error(err);
    });
}

selectSujeito.addEventListener("change", () => {
  const opt = selectSujeito.selectedOptions[0];
  if (opt) ajustarSliderParaSujeito(parseFloat(opt.dataset.duracao));
});
sliderTempo.addEventListener("input", () => {
  labelTempo.textContent = parseFloat(sliderTempo.value).toFixed(1) + "s";
});
```

- [ ] **Step 2: Atualizar `aoAtivarAbaFontes` (definida na Task 8) pra também carregar sujeitos**

Trocar:

```js
function aoAtivarAbaFontes() {
  carregarMalhaFontes();
}
```

por:

```js
function aoAtivarAbaFontes() {
  carregarMalhaFontes();
  carregarSujeitos();
}
```

- [ ] **Step 3: Checar sintaxe**

Run (mesmo comando node da Task 6, Step 4).
Expected: `OK: script parses without syntax errors`

- [ ] **Step 4: Verificação visual (Playwright) — com o backend da Task 4 rodando**

Com `uvicorn app:app --reload --port 8000` rodando dentro de `backend/`, servir o frontend (`python -m http.server 5500` na raiz) e abrir a página. Clicar em "Fontes corticais (MNE)": o `<select>` deve preencher com os sujeitos reais do CSV (ex: `v107 (ADHD, 154.6s)`), e o slider deve ir de `0` até `duração - 2`. Trocar de sujeito no `<select>` e conferir que o slider ajusta o `max` de acordo.

- [ ] **Step 5: Checkpoint**

Tarefa concluída quando o dropdown popula com sujeitos reais e o slider respeita a duração de cada um.

---

## Task 10: Frontend — botão "Rodar reconstrução", coloração e tratamento de erro

**Files:**
- Modify: `eeg-cerebro-3d.html`

**Interfaces:**
- Consumes: `BACKEND_URL`, `POST /source-localization` (Task 4), `malhaFontes`/`indiceValorPorVertice` (Task 8), `selectSujeito`/`sliderTempo` (Task 9), `corPorAmplitude` (já existente, seção 9 do arquivo).
- Produces: handler do clique em `#fontes-rodar` — não expõe nada pra tarefas seguintes (é a última tarefa do plano).

- [ ] **Step 1: Adicionar a normalização de cor e o handler do botão**

Em `eeg-cerebro-3d.html`, logo depois do bloco da Task 9, adicionar:

```js
// =========================================================================
// 14. RODAR RECONSTRUÇÃO — POST /source-localization, colore malhaFontes
//    pelos valores devolvidos (normalizados pelo min/max da própria
//    resposta — escala dSPM não é 0..1 como a amplitude sintética).
// =========================================================================
function corPorValorFontes(v, minV, maxV) {
  const t = maxV > minV ? (v - minV) / (maxV - minV) : 0;
  return corPorAmplitude(Math.max(0, Math.min(1, t)));
}

document.getElementById("fontes-rodar").addEventListener("click", () => {
  if (!malhaFontes || !indiceValorPorVertice) {
    fontesStatusEl.textContent = "Malha ainda carregando, espera um instante.";
    return;
  }
  const sujeito = selectSujeito.value;
  if (!sujeito) {
    fontesStatusEl.textContent = "Nenhum sujeito selecionado.";
    return;
  }
  const tStart = parseFloat(sliderTempo.value);
  const tEnd = tStart + 2;

  fontesStatusEl.textContent = "Rodando reconstrução (MNE)…";
  fetch(`${BACKEND_URL}/source-localization`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ subject_id: sujeito, t_start: tStart, t_end: tEnd, method: "dSPM" }),
  })
    .then(async (r) => {
      if (!r.ok) {
        const corpo = await r.json().catch(() => ({}));
        throw new Error(corpo.detail || ("status " + r.status));
      }
      return r.json();
    })
    .then((resp) => {
      const valores = resp.values;
      let minV = Infinity, maxV = -Infinity;
      for (const v of valores) { if (v < minV) minV = v; if (v > maxV) maxV = v; }

      const cores = malhaFontes.userData.cores;
      for (let i = 0; i < indiceValorPorVertice.length; i++) {
        const v = valores[indiceValorPorVertice[i]];
        const [r2, g2, b2] = corPorValorFontes(v, minV, maxV);
        cores[i * 3] = r2; cores[i * 3 + 1] = g2; cores[i * 3 + 2] = b2;
      }
      malhaFontes.geometry.attributes.color.needsUpdate = true;

      fontesStatusEl.innerHTML =
        `<b>${sujeito}</b> · ${tStart.toFixed(1)}–${tEnd.toFixed(1)}s · ${resp.method}<br>` +
        `<span style="color:#7f8ba1">19 canais é pouco pra localizar fontes com precisão — isso é ` +
        `uma estimativa aproximada, não a origem exata do sinal. fsaverage é um cérebro template, ` +
        `não a anatomia real do sujeito.</span>`;
    })
    .catch((err) => {
      fontesStatusEl.textContent = "Erro na reconstrução: " + err.message;
      console.error(err);
    });
});
```

- [ ] **Step 2: Checar sintaxe**

Run (mesmo comando node da Task 6, Step 4).
Expected: `OK: script parses without syntax errors`

- [ ] **Step 3: Verificação visual — caminho de sucesso (Playwright, com backend rodando)**

Com o backend rodando (Task 4) e os assets gerados (Task 5), abrir a página, ir pra aba "Fontes corticais", escolher um sujeito, clicar "Rodar reconstrução". Esperar o texto de status mudar pra "Rodando reconstrução (MNE)…" e depois pro resumo final (sujeito, janela, método, aviso de limitação). Tirar screenshot confirmando que a malha mudou de cinza neutro pra um gradiente de cores (a mesma rampa azul→vermelho já usada na aba Sensores).

- [ ] **Step 4: Verificação visual — caminho de erro, backend derrubado**

Parar o backend (`Ctrl+C` no terminal do uvicorn) e recarregar a página. Clicar em "Fontes corticais": o status deve mostrar a mensagem `Backend Python não respondeu em http://localhost:8000. Rode "uvicorn app:app --reload" dentro de backend/.` em vez de travar carregando.

- [ ] **Step 5: Verificação visual — caminho de erro, sujeito inválido (opcional, só se quiser forçar via DevTools)**

Com o backend rodando de novo, abrir o console do navegador e rodar manualmente:
```js
fetch("http://localhost:8000/source-localization", {
  method: "POST", headers: {"Content-Type": "application/json"},
  body: JSON.stringify({subject_id: "naoexiste", t_start: 0, t_end: 2})
}).then(r => r.json()).then(console.log)
```
Expected: `{"detail": "sujeito não encontrado: naoexiste"}` — confirma que o backend valida antes de rodar o MNE.

- [ ] **Step 6: Checkpoint**

Tarefa concluída (e o plano inteiro concluído) quando: reconstrução real funciona fim-a-fim com um sujeito do CSV, a malha muda de cor de forma visível, e os dois caminhos de erro (backend fora do ar, sujeito inexistente) mostram mensagens claras em vez de travar.
