from pathlib import Path

import mne
import numpy as np

from csv_data import CANAIS_19, FS

__all__ = ["CANAIS_19", "FS", "build_info", "build_source_model"]


def build_info():
    """mne.Info com os 19 canais EEG na montagem standard_1020, já com o
    projetor de referência average (CAR) declarado — necessário pra
    make_inverse_operator() calcular a covariância de forma consistente
    com a referência que mne_infer.apply_source_localization aplica nos
    dados de verdade. Rápido, sem download — seguro de chamar em
    qualquer teste."""
    info = mne.create_info(ch_names=CANAIS_19, sfreq=FS, ch_types="eeg")
    montagem = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montagem, on_missing="raise")
    raw_tmp = mne.io.RawArray(np.zeros((len(CANAIS_19), 1)), info, verbose=False)
    raw_tmp.set_eeg_reference("average", projection=True, verbose=False)
    return raw_tmp.info


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
