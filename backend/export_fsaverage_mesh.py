"""
Roda uma vez, offline: exporta a superfície cortical fsaverage (a MESMA
usada pelo forward model em mne_setup.build_source_model) para um .obj
combinando hemisfério esquerdo + direito, e um .json com o índice de
vértice original de cada canto de face, na MESMA ordem em que as faces
são escritas no .obj.

A ORDEM dos vértices é: primeiro os do hemisfério esquerdo
(forward['src'][0]['vertno']), depois os do direito
(forward['src'][1]['vertno']) — é exatamente a ordem que
mne_infer.apply_source_localization devolve em stc.data. Ou seja: o
índice i em fsaverage-cortex-face-indices.json aponta pra values[i] na
resposta de POST /source-localization.

O frontend (Task 8) usa esse .json de índices — e não as coordenadas —
pra colorir a malha: o Three.js OBJLoader carrega geometria NÃO
indexada (cada canto de face vira uma entrada própria em
geometry.attributes.position, na ordem em que as faces aparecem no
arquivo), então o canto k carregado corresponde exatamente ao k-ésimo
valor deste array. Casar por COORDENADA foi tentado primeiro e se
mostrou frágil: o Three.js guarda posições em Float32Array (32 bits),
enquanto o JSON usa float64 — perto de fronteiras de arredondamento
(ex: valor terminando em "...5" na 5ª casa decimal) as duas
representações podem arredondar pra lados diferentes em toFixed(4),
quebrando a correspondência pra ~1% dos vértices. Índice de face é
exato e não depende de ponto flutuante.

Uso: cd backend && python export_fsaverage_mesh.py
Saída: ../assets/fsaverage-cortex.obj, ../assets/fsaverage-cortex-face-indices.json
"""
import json
from pathlib import Path

import numpy as np

from mne_setup import build_source_model


def exportar(caminho_obj, caminho_indices_json):
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

    # flat: [t0a, t0b, t0c, t1a, t1b, t1c, ...] — mesma ordem das linhas "f"
    indices_por_canto = [idx for tri in triangulos for idx in tri]
    with open(caminho_indices_json, "w") as f:
        json.dump(indices_por_canto, f)

    print(
        f"[export_fsaverage_mesh] {len(pontos)} vértices, {len(triangulos)} faces -> "
        f"{caminho_obj}, {caminho_indices_json}"
    )
    return len(pontos)


if __name__ == "__main__":
    pasta_assets = Path(__file__).resolve().parent.parent / "assets"
    exportar(
        pasta_assets / "fsaverage-cortex.obj",
        pasta_assets / "fsaverage-cortex-face-indices.json",
    )
