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
