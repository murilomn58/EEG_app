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
