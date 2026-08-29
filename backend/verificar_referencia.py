"""
Verifica empiricamente qual referência um arquivo de EEG parece usar,
olhando estatísticas simples por canal:

  - canal com desvio-padrão ~0 (flat) -> provavelmente É a referência
    física (ex: Cz numa gravação referenciada em Cz, como o HBN)
  - média instantânea (linha a linha) dos canais já perto de 0 -> sinal
    de que já veio com average reference (CAR) aplicado
  - nenhum dos dois -> referência EXTERNA ao conjunto de canais (ex:
    linked-ears/A1-A2, que nem aparecem como coluna)

Não decide sozinho — imprime os números pra cruzar com a documentação
do dataset. Foi assim que confirmamos o adhdata.csv: nenhum canal é
flat (Cz incluso, tem valores reais variando) e a média instantânea não
fica perto de 0, batendo com a ficha técnica do dataset ("EEG Dataset
for ADHD", Kaggle/IEEE DataPort), que documenta referência A1/A2
(linked-ears).

Uso: cd backend && python verificar_referencia.py [caminho_csv]
Padrão: ../adhdata.csv com os 19 canais de csv_data.CANAIS_19
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csv_data import CANAIS_19


def diagnosticar(dados, nomes):
    """(dict) com o que dá para dizer sobre a referência olhando só o sinal.

    Separada de verificar() para poder ser chamada de dentro de um endpoint:
    print não serve para HTTP, e esta é a única detecção de referência que
    existe no projeto. Recebe uma matriz (n_canais, n_amostras) e os nomes,
    para servir tanto ao CSV quanto a um Raw do MNE.

    O veredito é deliberadamente conservador. Canal com desvio ~0 é forte
    indício de ser a referência física, porque um eletrodo referenciado
    contra si mesmo dá zero. Média instantânea perto de zero sugere CAR já
    aplicado. Nenhum dos dois é prova: a resposta honesta é cruzar com a
    documentação do dataset, e por isso o dict devolve a evidência junto do
    palpite."""
    desvios = dados.std(axis=1)
    tipico = float(np.median(desvios[desvios > 0])) if (desvios > 0).any() else 0.0

    # "plano" e "perto de zero" precisam ser RELATIVOS à amplitude do próprio
    # dataset, não absolutos. Um limiar em microvolts erra por seis ordens de
    # grandeza quando o dado chega em volts (que é como o MNE entrega), e
    # erra de novo no HBN, cujos valores brutos chegam com ordem de grandeza
    # muito diferente da do adhdata — o offset DC domina, e a calibração do
    # release não está confirmada. Em nenhum dos dois casos o limiar absoluto
    # sobrevive, e é por isso que o critério aqui é uma razão.
    flat = [n for n, d in zip(nomes, desvios) if tipico > 0 and d < tipico * 1e-6]
    media_linha = dados.mean(axis=0)
    razao = float(media_linha.std() / tipico) if tipico > 0 else 0.0

    # SINAL SEM CONTEÚDO NÃO PRODUZ VEREDITO.
    #
    # Quando todo canal tem desvio zero, `tipico` é 0, `razao` é forçada a
    # 0.0 na linha acima, e o `elif razao < 0.01` acertava — devolvendo
    # "average" com a evidência afirmativa "CAR aparentemente já aplicado"
    # para uma gravação IDENTICAMENTE MORTA. Medido:
    # diagnosticar(np.zeros((4,100)), [...]) devolvia referencia='average'.
    #
    # Isso é o oposto do que a docstring acima promete. E é o mesmo caso que
    # preproc_basico.detectar_frequencia_rede já trata do jeito certo, com o
    # motivo `sinal_sem_conteudo_ac`: dizer que não dá para medir.
    if tipico <= 0:
        return {
            "referencia": None,
            "evidencia": "todos os canais têm desvio zero: não há sinal para diagnosticar "
                         "a referência (gravação vazia, canal errado ou leitura falhada)",
            "canais_flat": [],
            "razao_media_instantanea": 0.0,
            "desvio_tipico_canal": 0.0,
        }

    if len(flat) > 1:
        # MAIS DE UM CANAL PLANO NÃO É REFERÊNCIA — é canal morto.
        #
        # Uma gravação tem UMA referência física; se três canais estão em
        # zero, o que se descobriu foi eletrodo solto, canal desligado ou
        # leitura truncada. A versão anterior pegava `flat[0]` e o anunciava
        # como "a referência física", escolhendo pela ordem do arquivo.
        valor, evidencia = None, (
            f"{len(flat)} canais com desvio ~0 ({', '.join(flat[:4])}"
            f"{'…' if len(flat) > 4 else ''}): isso não é referência, é canal sem sinal"
        )
    elif flat:
        valor, evidencia = flat[0], f"canal {flat[0]} com desvio ~0 (é a referência física)"
    elif razao < 0.01:
        valor, evidencia = "average", (
            f"média instantânea dos canais é {razao:.1%} do desvio típico de um canal "
            f"(CAR aparentemente já aplicado)"
        )
    else:
        valor, evidencia = None, (
            f"nenhum canal plano, e a média instantânea vale {razao:.0%} do desvio típico "
            f"de um canal: a referência é externa aos canais gravados"
        )

    return {
        "referencia": valor,
        "evidencia": evidencia,
        "canais_flat": flat,
        "razao_media_instantanea": razao,
        "desvio_tipico_canal": tipico,
    }


def verificar(caminho_csv, canais=None):
    """Imprime média e desvio por canal e o diagnóstico de referência.

    É a versão de terminal do `diagnosticar`: mesma conclusão, com a tabela
    canal a canal que permite conferir o veredito em vez de acreditar nele.

    O sinal procurado é um canal com desvio praticamente zero. Quando a
    referência física está entre os canais gravados, ela é o zero contra o
    qual os outros são medidos, e portanto não varia. Foi assim que o Cz do
    HBN foi identificado como referência sem depender da documentação.

    Ausência de canal plano NÃO significa ausência de referência: significa
    que a referência não está entre os canais do arquivo, que é o caso do
    adhdata, referenciado nos mastoides (A1/A2), que não são gravados."""
    canais = canais or CANAIS_19
    df = pd.read_csv(caminho_csv, usecols=canais)

    print(f"[verificar_referencia] {caminho_csv} — {len(df)} amostras, {len(canais)} canais\n")

    print(f"{'canal':<6} {'média':>10} {'desvio':>10}")
    candidatos_flat = []
    for c in canais:
        media = df[c].mean()
        desvio = df[c].std()
        print(f"{c:<6} {media:>10.3f} {desvio:>10.3f}")
        if desvio < 1e-6:
            candidatos_flat.append(c)

    media_linha_a_linha = df[canais].mean(axis=1)
    media_da_media = media_linha_a_linha.mean()
    desvio_da_media = media_linha_a_linha.std()

    print(
        f"\nmédia instantânea dos {len(canais)} canais (linha a linha): "
        f"média={media_da_media:.3f}, desvio={desvio_da_media:.3f}"
    )

    print("\n--- veredito ---")
    if candidatos_flat:
        print(f"Canal(is) com desvio ~0 (candidato a referência física): {candidatos_flat}")
    else:
        print(
            "Nenhum canal é flat — a referência física NÃO está entre esses "
            "canais (provavelmente é um canal externo, tipo A1/A2 linked-ears, "
            "que nem aparece nas colunas)."
        )

    if abs(media_da_media) < 1.0 and desvio_da_media < 5.0:
        print(
            "A média instantânea dos canais fica perto de 0 — pode já vir com "
            "average reference (CAR) aplicado. Verifique contra a documentação "
            "do dataset antes de concluir — isso sozinho não é prova definitiva."
        )
    else:
        print(
            "A média instantânea dos canais NÃO fica perto de 0 — não parece "
            "vir com average reference já aplicado."
        )


if __name__ == "__main__":
    caminho = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parent.parent / "adhdata.csv"
    )
    verificar(caminho)
