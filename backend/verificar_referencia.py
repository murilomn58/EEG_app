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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csv_data import CANAIS_19


def verificar(caminho_csv, canais=None):
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
