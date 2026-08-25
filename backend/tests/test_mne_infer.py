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
