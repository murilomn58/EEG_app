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


@app.get("/raw-data")
def get_raw_data(subject_id: str):
    try:
        canais = csv_data.get_subject_raw(app.state.df, subject_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"subject_id": subject_id, "fs": csv_data.FS, "channels": canais}


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
