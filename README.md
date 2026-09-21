# EEG Brain 3D

A local workbench for inspecting EEG recordings alongside a three-dimensional brain view. It combines signal traces, frequency-band filters, electrode visualization and cortical source estimates in one interface.

![EEG traces and the three-dimensional sensor view](docs/images/shot-01-overview.png)

## What it does

- Displays EEG traces and lets you inspect a moment in the recording.
- Filters frequency bands and switches reference configurations.
- Maps measured sensor power onto a brain mesh.
- Estimates cortical sources through an MNE-Python forward/inverse pipeline.
- Reads supported CSV and HBN/BIDS recordings, with controls for montage and preprocessing.
- Provides descriptive signal measurements and scripts for subject-based evaluation.

The sensor view interpolates electrode measurements. Source reconstruction uses an inverse model. They are separate modes and should be interpreted separately.

![Cortical source reconstruction](docs/images/shot-06-source-reconstruction.png)

## Repository layout

```text
assets/                  Application assets
backend/                 API, EEG processing and tests
docs/
  images/                Documentation screenshots and image index
    hbn/                 HBN recording views
    med/                 Signal, filter and source exploration
    wizard/              Import wizard steps
  superpowers/           Design notes and implementation plans
  technical-reference.md Processing details and scientific references
eeg-cerebro-3d.html       Browser interface
iniciar.py               Local launcher
```

Browse the [screenshot index](docs/images/README.md) for the interface captures. Runtime assets are kept separate from documentation images.

## Run locally

Requirements: Python 3.11 or later and a browser with WebGL. The 3D viewer loads dependencies from public CDNs; the anatomical template may require a separate download on first use.

Create and activate a virtual environment, then install the backend dependencies from the repository root:

```bash
python -m pip install -r backend/requirements.txt
```

Obtain the recordings separately. For the supported `adhdata` format, place `adhdata.csv` in the repository root. For HBN releases, set `EEG_DADOS` to the directory containing the BIDS releases. The default is documented in `backend/config.py`.

```bash
python iniciar.py --sem-mapa
```

| Address | Service |
| --- | --- |
| <http://localhost:5500/eeg-cerebro-3d.html> | Viewer |
| <http://localhost:8001/docs> | Backend API documentation |

The optional project map is intended for the local research workspace. Start without `--sem-mapa` only after reviewing its local path configuration. Keep the services bound to the local machine.

## Verification

The backend includes tests with synthetic inputs as well as tests that require separately downloaded recordings. A data-independent subset can be run from the root:

```bash
python -m pytest backend/tests/test_split.py backend/tests/test_epocas.py backend/tests/test_caracteristicas.py backend/tests/test_estatistica.py backend/tests/test_experimento_svm.py backend/tests/test_exportar_features.py -q
```

The evaluation scripts distinguish splits by segment from splits by subject. They support research experiments; the application is not a diagnostic system. Fine-tuning of an EEG foundation model belongs to a separate research project and is not implemented here.

## Data, interpretation and license

Recordings are not bundled with the application. The research uses the [HBN EEG release ds005505](https://doi.org/10.18112/openneuro.ds005505.v1.0.1) and the [ADHD/control EEG dataset](https://doi.org/10.21227/rzfh-zn36), subject to their providers' terms. Screenshots illustrate the interface and retain dataset-supplied participant labels; they should not be described as anonymized examples.

The app is a research and educational tool. Source estimates depend on model assumptions and electrode coverage. Descriptive measurements do not establish a clinical diagnosis.

Code is covered by [MIT](LICENSE). Dataset terms are separate. Provider attribution, processing details, API routes and known limitations are retained in the [technical reference](docs/technical-reference.md).

## Stack

Python, FastAPI, MNE-Python, NumPy, pandas, scikit-learn and Three.js.
