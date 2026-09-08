# -*- coding: utf-8 -*-
"""Classificação com SVM sob validação cruzada aninhada por sujeito.

POR QUE A AGREGAÇÃO POR SUJEITO EXISTE

O classificador prediz por época, e a pergunta clínica é por criança. Entre a
época e a criança há uma escolha que muda o resultado e costuma ficar
implícita: como as N épocas de um sujeito viram um número.

Aqui é a MEDIANA. O adhdata tem sujeitos com 62 s e sujeitos com 338 s de
gravação — uma razão de 5,4 —, e a média deixaria uma única época contaminada
por artefato deslocar o escore de um sujeito inteiro. A mediana não.
"""
import numpy as np


def agregar_por_sujeito(escores, grupos, rotulos):
    """(escores_sujeito, rotulos_sujeito, ids_sujeito), na ordem de `np.unique`.

    Recusa sujeito com mais de um rótulo em vez de escolher um. Um sujeito com
    dois rótulos significa que a montagem do conjunto está errada, e devolver o
    primeiro esconderia o defeito atrás de um número plausível."""
    escores = np.asarray(escores, dtype=float)
    grupos = np.asarray(grupos)
    rotulos = np.asarray(rotulos)

    if not (len(escores) == len(grupos) == len(rotulos)):
        raise ValueError(
            f"escores ({len(escores)}), grupos ({len(grupos)}) e rotulos "
            f"({len(rotulos)}) têm de ter o mesmo comprimento"
        )

    ids = np.unique(grupos)
    esc_s = np.empty(len(ids), dtype=float)
    rot_s = np.empty(len(ids), dtype=rotulos.dtype)

    for i, sid in enumerate(ids):
        dentro = grupos == sid
        rot_unicos = np.unique(rotulos[dentro])
        if len(rot_unicos) > 1:
            raise ValueError(
                f"sujeito {sid!r} tem mais de um rótulo ({rot_unicos.tolist()}): "
                f"o conjunto foi montado errado"
            )
        esc_s[i] = float(np.median(escores[dentro]))
        rot_s[i] = rot_unicos[0]

    return esc_s, rot_s, ids
