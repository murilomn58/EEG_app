# -*- coding: utf-8 -*-
"""Exporta as características do adhdata no contrato que o eeg_transformer lê.

O BURACO QUE ISTO FECHA

O README do `eeg_transformer` diz que o app "trata, particiona e prepara os
dados que este projeto consome". Nenhum formato tinha sido definido, e por isso
aquele projeto estava vazio: não havia o que consumir.

A COLUNA QUE IMPORTA

`sujeito_id` traz o ID ORIGINAL do banco (`v10p`), e não o índice interno da
montagem. O índice é local: rodar com `--max-sujeitos` muda o índice do mesmo
sujeito, e duas exportações ficam impossíveis de juntar. Mais grave, sem o ID
original o outro projeto não consegue refazer a partição por sujeito, e um CSV
que não permite partição honesta é um convite ao vazamento.

O rótulo sai como TEXTO (`ADHD` / `Control`) e não como 0/1, porque um código
numérico exige um dicionário externo para ser lido — e esse dicionário é
exatamente o que se perde na fronteira entre dois projetos.

Uso:
    python scripts/exportar_features.py
    python scripts/exportar_features.py --max-sujeitos 10 --saida dados/features.csv
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import caracteristicas
import config
import csv_data
import epocas as mod_epocas
import preproc_basico
import receita as mod_receita

COLUNAS_DE_IDENTIFICACAO = ["sujeito_id", "rotulo", "epoca_idx"]

RESSALVA_LICENCA = (
    "adhdata: a licença deste banco não pôde ser confirmada; ausência de "
    "licença não é licença permissiva"
)


def escrever_csv(caminho, X, y, ids_originais, indices_epoca, nomes):
    """Grava o CSV do contrato. Uma linha por época.

    Recusa comprimentos incompatíveis em vez de truncar pelo menor: truncar
    produziria um arquivo plausível com linhas faltando, e ninguém notaria."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    for rotulo, vetor in (("y", y), ("ids_originais", ids_originais),
                          ("indices_epoca", indices_epoca)):
        if len(vetor) != n:
            raise ValueError(
                f"X tem {n} linhas e {rotulo} tem {len(vetor)}: comprimento "
                f"incompatível"
            )
    if X.ndim != 2 or X.shape[1] != len(nomes):
        raise ValueError(
            f"X tem forma {X.shape} e há {len(nomes)} nomes de característica"
        )

    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(COLUNAS_DE_IDENTIFICACAO + list(nomes))
        for i in range(n):
            escritor.writerow(
                [ids_originais[i], y[i], int(indices_epoca[i])]
                + [f"{v:.10g}" for v in X[i]]
            )


def sha256_do_arquivo(caminho):
    """Hash sha256 hexadecimal do conteúdo de um arquivo.

    O CSV que este script gera não é versionado (`dados/*` está no
    `.gitignore` do eeg_transformer, porque dado de EEG é grande demais) —
    ele é regerado a partir do comando registrado na receita. Sem um hash do
    conteúdo, duas pessoas com "o mesmo" arquivo não têm como confirmar que é
    literalmente o mesmo: só há o timestamp `gerado_em`, que prova quando o
    arquivo foi escrito, não o que há dentro dele.

    Lê em blocos de 64 KB para não carregar o CSV inteiro (pode passar de
    267 MB) de uma vez na memória."""
    h = hashlib.sha256()
    with Path(caminho).open("rb") as f:
        for bloco in iter(lambda: f.read(65536), b""):
            h.update(bloco)
    return h.hexdigest()


def montar_conjunto_tbr(df, duracao_s=4.0, passo_s=4.0, max_sujeitos=None):
    """(X, y, grupos, ids_originais, indices_epoca, nomes, decisoes).

    `grupos` é o índice interno, que o GroupKFold consome; `ids_originais` é o
    ID do banco, que vai para o CSV. Os dois existem porque servem a coisas
    diferentes, e confundi-los é o defeito que o teste do contrato tranca.

    Sujeito que falha APARECE em `decisoes`, e não some da contagem: um
    denominador que encolhe em silêncio faz a média mentir."""
    sujeitos = csv_data.list_subjects(df)
    if max_sujeitos:
        # Amostra equilibrada, e nao os N primeiros: a lista do adhdata chega
        # ORDENADA POR CLASSE, e os N primeiros dariam uma classe so.
        por_classe = {}
        for s in sujeitos:
            por_classe.setdefault(s["classe"], []).append(s)
        metade = max(1, max_sujeitos // max(1, len(por_classe)))
        escolhidos = []
        for lista in por_classe.values():
            escolhidos.extend(lista[:metade])
        sujeitos = escolhidos[:max_sujeitos]

    blocos_X, blocos_y, blocos_g, blocos_id, blocos_ep = [], [], [], [], []
    nomes = None
    dec_epocas = dec_carac = dec_preproc = None
    falhas = []

    for idx, s in enumerate(sujeitos):
        sid = s["id"]
        try:
            raw = preproc_basico.raw_de_dataframe(df, sid)
            filtrado, dec_preproc = preproc_basico.preprocessar(
                raw, l_freq=0.5, h_freq=None
            )
            dados = filtrado.get_data() * 1e6

            janelas, dec_epocas = mod_epocas.epocar_janela_fixa(
                dados, csv_data.FS, duracao_s, passo_s
            )
            if len(janelas) == 0:
                falhas.append((sid, "nenhuma época coube na gravação"))
                continue

            X, nomes, dec_carac = caracteristicas.razao_theta_beta(
                janelas, csv_data.FS,
                nomes_canais=list(csv_data.CANAIS_19), com_decisoes=True,
            )
            blocos_X.append(X)
            blocos_y.append(np.full(len(X), s["classe"], dtype=object))
            blocos_g.append(np.full(len(X), idx))
            blocos_id.append(np.full(len(X), sid, dtype=object))
            blocos_ep.append(np.arange(len(X)))
        except Exception as e:            # noqa: BLE001 — sujeito que falha aparece
            falhas.append((sid, str(e)))

    if not blocos_X:
        raise RuntimeError("nenhum sujeito produziu época: nada a exportar")

    decisoes = {
        "preproc": dec_preproc,
        "epocas": dec_epocas,
        "caracteristicas": dec_carac,
        "n_sujeitos_pedidos": len(sujeitos),
        "n_sujeitos_usados": len(blocos_X),
        "sujeitos_que_falharam": [{"id": s, "motivo": m} for s, m in falhas],
        "ressalva_licenca": RESSALVA_LICENCA,
    }
    return (np.vstack(blocos_X), np.concatenate(blocos_y),
            np.concatenate(blocos_g), np.concatenate(blocos_id),
            np.concatenate(blocos_ep), nomes, decisoes)


def main():
    """CLI: monta o conjunto, grava o CSV e a receita lado a lado.

    Um comando que produz o CSV sem a receita ao lado é o que este módulo
    existe para evitar — a receita é o que permite a outra pessoa refazer a
    extração a partir do que o app entrega, não só consumir o número."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epoca", type=float, default=4.0)
    p.add_argument("--passo", type=float, default=4.0)
    p.add_argument("--max-sujeitos", type=int, default=None)
    p.add_argument("--saida", default="dados/features_adhdata_tbr.csv")
    args = p.parse_args()

    df = csv_data.load_csv(config.CAMINHO_ADHDATA)
    X, y, grupos, ids_originais, indices_epoca, nomes, dec = montar_conjunto_tbr(
        df, duracao_s=args.epoca, passo_s=args.passo,
        max_sujeitos=args.max_sujeitos,
    )

    escrever_csv(args.saida, X, y, ids_originais, indices_epoca, nomes)

    dec["sha256_do_csv"] = sha256_do_arquivo(args.saida)
    dec["arquivo_csv"] = Path(args.saida).name

    receita = mod_receita.montar(
        banco="adhdata",
        sujeitos=sorted(set(ids_originais.tolist())),
        etapas=dec,
        notas=RESSALVA_LICENCA,
    )
    mod_receita.salvar(receita, Path(args.saida).with_suffix(".receita.json"))

    print(f"linhas: {len(X)}")
    print(f"sujeitos usados: {dec['n_sujeitos_usados']}")
    print(f"características: {len(nomes)}")
    if dec["sujeitos_que_falharam"]:
        print("sujeitos que falharam:")
        for f in dec["sujeitos_que_falharam"]:
            print(f"  {f['id']}: {f['motivo']}")
    print(f"\n{RESSALVA_LICENCA}")


if __name__ == "__main__":
    main()
