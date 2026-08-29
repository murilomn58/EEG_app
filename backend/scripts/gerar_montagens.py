# -*- coding: utf-8 -*-
"""Congela em disco as posições de eletrodo que o app 3D desenha.

POR QUE ISTO EXISTE
-------------------
A cena 3D principal posicionava os 19 eletrodos por uma tabela de pares
(theta, phi) escrita à mão dentro do HTML. Medido contra o `standard_1020`
do MNE, o erro angular daquela tabela era de 6,9° na mediana e 12,4° no pior
caso (T7 e T8) — 1,1 cm e 2,0 cm de deslocamento no couro cabeludo de uma
cabeça de 9 cm de raio. Nada na tela denunciava: um eletrodo 2 cm fora do
lugar continua parecendo um eletrodo.

E os 129 canais da malha EGI não estavam salvos em lugar nenhum. Eles só
existiam enquanto o backend estivesse de pé COM o release do HBN baixado, o
que faz a tela de conferência depender de 3 GB de dado que talvez não esteja
ali. Pior: as posições vinham do arquivo de um sujeito, e o app pedia sem
dizer qual.

Este script resolve os dois de uma vez: lê as montagens do MNE, converte
para a convenção da cena, e grava em `assets/montagens.json`. O app passa a
ler dali. O JSON é gerado, não editado à mão — quem quiser mudar posição
muda a montagem de origem e roda isto de novo.

O QUE NÃO É
-----------
Não é digitalização do sujeito. É montagem template, ou seja, posição MÉDIA
de população, e o JSON diz isso em cada entrada. Quando a gravação traz a
digitalização própria, ela continua tendo precedência: quem decide é
`eletrodos.posicoes`, e este arquivo é o que sobra quando não há nada
melhor. Chamar template de medição é o tipo de erro que não levanta exceção.

CONVENÇÃO DE COORDENADAS
------------------------
Grava nas DUAS convenções de propósito, e isso não é redundância:

  `xyz` fica em coordenada de cabeça do MNE (x = direita, y = frente,
  z = cima), em METROS, porque é assim que o backend entrega em `/eletrodos`
  e o frontend já sabe converter com `xyzParaCena`.

  `theta`/`phi` fica na convenção esférica do app (theta a partir do topo,
  phi da frente para a direita), porque é a reserva para eletrodo sem xyz.

As duas saem da MESMA fonte pela mesma função (`eletrodos.para_esfericas`),
então elas não podem divergir por edição — que é exatamente como a tabela
antiga divergiu da realidade.

Uso:
    python scripts/gerar_montagens.py            # grava assets/montagens.json
    python scripts/gerar_montagens.py --conferir # só compara, não grava
"""
import json
import sys
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from eletrodos import para_esfericas  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent.parent
DESTINO = RAIZ / "assets" / "montagens.json"

# Os 19 canais de análise, na ordem que o resto do app assume.
CANAIS_10_20 = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]

def _papeis_egi():
    """nome-na-malha-EGI -> canal de análise 10-20.

    Sai de `config.MAPA_EGI_1020`, invertido, MAIS um apelido que precisa
    ficar escrito porque custou uma contagem errada para aparecer.

    O eletrodo do vértice tem DOIS nomes em uso, e os dois são legítimos: a
    documentação do fabricante o chama de E129 (é o sensor de referência,
    o 129º), e tanto o `GSN-HydroCel-129` do MNE quanto os arquivos do HBN
    o chamam de `Cz`. Conferido: os nomes da montagem terminam em
    [..., 'E127', 'E128', 'Cz'], e 'E129' não existe nela.

    Nada disso quebrava a resolução de canal, porque `canais.resolver`
    acerta pelo nome exato antes de consultar mapa nenhum. O que quebrava
    era ESTE arquivo: sem o apelido, o vértice saía sem papel e a touca de
    129 vinha com 18 correspondências em vez de 19.

    O `config` não muda: ele cita o documento do fabricante e está certo
    sobre o que aquele documento diz. Quem concilia as duas grafias é aqui,
    que é onde as duas se encontram.

    A malha de 128 usa o mesmo mapa de propósito: ela é a de 129 sem o
    vértice, então o apelido simplesmente não casa com nada e sobram 18
    papéis — que é a resposta correta para uma touca que não tem o Cz.
    """
    papeis = {egi: alvo for alvo, egi in config.MAPA_EGI_1020.items()}
    papeis["Cz"] = "Cz"
    return papeis


# Uma entrada por montagem que o app precisa desenhar sem backend.
#
# `standard_1005` entra porque é o superconjunto do 10-20 e do 10-10: é ele
# que atende bancos de 32, 64 ou 128 canais nomeados no padrão internacional
# (o EEG Motor Movement/Imagery do PhysioNet, por exemplo, tem 64 canais
# 10-10). Sem ele, um banco desses cai no `standard_1020`, casa nada, e a
# tela de conferência aparece VAZIA sem dizer por quê.
MONTAGENS = [
    {
        "id": "standard_1020",
        "mne": "standard_1020",
        "rotulo": "10-20 internacional (19 canais de análise)",
        "so_estes": CANAIS_10_20,
        "papeis": {n: n for n in CANAIS_10_20},
        "fonte": "mne.channels.make_standard_montage('standard_1020'), "
                 "posições médias sobre o fsaverage",
    },
    {
        "id": "GSN-HydroCel-129",
        "mne": "GSN-HydroCel-129",
        "rotulo": "EGI GSN-HydroCel 129 canais (HBN)",
        "so_estes": None,
        "papeis": _papeis_egi(),
        "fonte": "mne.channels.make_standard_montage('GSN-HydroCel-129'), "
                 "posições médias do fabricante; correspondência 10-20 de "
                 "config.MAPA_EGI_1020 (EGI doc. 8403486-52 e Luu & Ferree 2005)",
    },
    {
        "id": "GSN-HydroCel-128",
        "mne": "GSN-HydroCel-128",
        "rotulo": "EGI GSN-HydroCel 128 canais (sem o vértice de referência)",
        "so_estes": None,
        "papeis": _papeis_egi(),
        "fonte": "mne.channels.make_standard_montage('GSN-HydroCel-128')",
    },
    {
        "id": "standard_1005",
        "mne": "standard_1005",
        "rotulo": "10-05 internacional (superconjunto do 10-10 e do 10-20)",
        "so_estes": None,
        # O 10-05 traz os 19 pelos proprios nomes: aqui o papel e identidade,
        # nao traducao. E o que faz um banco de 64 canais 10-10 aparecer com
        # os 19 de analise ja identificados, sem depender do backend.
        "papeis": {n: n for n in CANAIS_10_20},
        "fonte": "mne.channels.make_standard_montage('standard_1005')",
    },
]



def _uma(spec):
    """Uma montagem pronta para o JSON, ou levanta dizendo qual falhou.

    Não engole exceção: uma montagem que sumiu de uma versão do MNE tem de
    parar a geração, e não sair do arquivo em silêncio deixando a tela sem
    eletrodo e sem explicação."""
    montagem = mne.channels.make_standard_montage(spec["mne"])
    ch_pos = montagem.get_positions().get("ch_pos") or {}

    nomes = spec["so_estes"] if spec["so_estes"] else list(ch_pos)
    eletrodos, sem_posicao = [], []
    for nome in nomes:
        xyz = ch_pos.get(nome)
        if xyz is None or not np.all(np.isfinite(xyz)):
            sem_posicao.append(nome)
            continue
        esf = para_esfericas(xyz)
        if esf is None:
            sem_posicao.append(nome)
            continue
        item = {
            "nome": nome,
            "xyz": [round(float(v), 6) for v in xyz],
            "theta": esf[0],
            "phi": esf[1],
        }
        # Para qual canal de análise 10-20 este eletrodo serve, quando isso
        # for sabido por FONTE PUBLICADA. Sem este campo a cena não tem como
        # saber, na reserva, que o E11 desta touca é o mesmo Fz que ela já
        # desenhou como dipolo — e acabava desenhando os dois, 147
        # marcadores para uma touca de 129.
        #
        # Sai de config.MAPA_EGI_1020, que cita o Technical Note da EGI e o
        # mapa do fabricante. Nada é inferido por proximidade geométrica
        # aqui: a regra de equivalência da linha média contraria a
        # proximidade, e é justamente por isso que o Fz é o E11 e não o E6.
        papel = spec.get("papeis", {}).get(nome)
        if papel:
            item["papel"] = papel
        eletrodos.append(item)

    if spec["so_estes"] and sem_posicao:
        raise SystemExit(
            f"[gerar_montagens] {spec['id']}: canais pedidos sem posição: {sem_posicao}"
        )

    return {
        "id": spec["id"],
        "rotulo": spec["rotulo"],
        "n": len(eletrodos),
        "fonte": spec["fonte"],
        "natureza": "montagem template (média de população), NÃO digitalização deste sujeito",
        "unidade_xyz": "metros, coordenada de cabeça do MNE (x=direita, y=frente, z=cima)",
        "convencao_esferica": "theta = graus a partir do topo (0 = vértice); "
                              "phi = graus da frente para a direita",
        "eletrodos": eletrodos,
        "sem_posicao": sem_posicao,
    }


def gerar():
    """O objeto inteiro, pronto para serializar."""
    return {
        "gerado_por": "backend/scripts/gerar_montagens.py",
        "mne_versao": mne.__version__,
        "aviso": "arquivo GERADO — não editar à mão; rode o script de novo",
        "montagens": {m["id"]: m for m in (_uma(s) for s in MONTAGENS)},
    }


def _conferir(dados):
    """Compara com a tabela que estava escrita à mão no HTML.

    Fica no script, e não num teste, porque é medição histórica: serve para
    dizer QUANTO a tabela antiga errava, e some quando ninguém mais
    perguntar. O teste que trava a montagem nova vive em
    backend/tests/test_montagens.py."""
    antiga = {
        "Fp1": (90, -18), "Fp2": (90, 18), "F7": (90, -54), "F3": (63, -45),
        "Fz": (45, 0), "F4": (63, 45), "F8": (90, 54), "T7": (90, -90),
        "C3": (45, -90), "C4": (45, 90), "T8": (90, 90), "P7": (90, -126),
        "P3": (63, -135), "Pz": (45, 180), "P4": (63, 135), "P8": (90, 126),
        "O1": (90, -162), "O2": (90, 162), "Cz": (0, 0),
    }

    def direcao(theta, phi):
        t, p = np.radians(theta), np.radians(phi)
        return np.array([np.sin(t) * np.sin(p), np.sin(t) * np.cos(p), np.cos(t)])

    erros = []
    for e in dados["montagens"]["standard_1020"]["eletrodos"]:
        th_a, ph_a = antiga[e["nome"]]
        cosseno = float(np.dot(direcao(th_a, ph_a), direcao(e["theta"], e["phi"])))
        erros.append((np.degrees(np.arccos(np.clip(cosseno, -1, 1))), e["nome"]))

    erros.sort(reverse=True)
    graus = np.array([g for g, _ in erros])
    print("erro da tabela antiga contra o standard_1020:")
    print("  mediana %.1f°  média %.1f°  máximo %.1f° (%s)"
          % (np.median(graus), graus.mean(), graus[0], erros[0][1]))
    print("  em cm no couro cabeludo (r = 9 cm): mediana %.1f  máximo %.1f"
          % (np.radians(np.median(graus)) * 9, np.radians(graus[0]) * 9))
    print("  piores: " + ", ".join("%s %.1f°" % (n, g) for g, n in erros[:5]))


def main():
    dados = gerar()
    for ident, m in dados["montagens"].items():
        aviso = f"  ({len(m['sem_posicao'])} sem posição)" if m["sem_posicao"] else ""
        print("%-20s %4d eletrodos%s" % (ident, m["n"], aviso))

    if "--conferir" in sys.argv:
        print()
        _conferir(dados)
        return

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    DESTINO.write_text(
        json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print()
    print("gravado: %s (%.0f KB)" % (DESTINO, DESTINO.stat().st_size / 1024))


if __name__ == "__main__":
    main()
