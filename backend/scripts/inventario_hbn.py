"""
Percorre a árvore BIDS de um release do HBN-EEG e emite um .csv com uma
linha por (sujeito, tarefa): duração, taxa de amostragem, número de
canais e presença do events.tsv.

Existe porque a documentação de um dataset descreve o que ele deveria
ter, e o inventário conta o que ele tem de fato. No ds005505 nem todo
sujeito gravou todas as tarefas — o participants.tsv marca cada uma
como available/unavailable/caution — e escolher tarefa ou recorte
clínico antes de contar é como descobrir três meses depois que metade
dos sujeitos não serve.

DECISÃO DE PROJETO: o inventário NUNCA aborta. Um .set ilegível vira
uma linha com a coluna "erro" preenchida, e a varredura continua. Um
raise no sujeito 100 de 136 destruiria os 99 anteriores, e o valor
deste script é justamente o mapa completo — buracos inclusive. Sujeito
sem pasta eeg/ também emite linha: o buraco é o achado, e um sujeito
que sumisse silencioso seria exatamente o bug que isto deveria pegar.

Colunas que não puderam ser lidas ficam VAZIAS, nunca 0 nem NaN: 0 é
indistinguível de "arquivo com zero canais", e NaN vira a string "nan"
no csv. Para filtrar só o que leu: df[df.erro == ""].

Nota sobre o .fdt: o ds005505 não tem nenhum — os dados vêm embutidos
no próprio .set (~105 MB cada), então read_raw_eeglab lê o arquivo
sozinho. Ainda assim usamos preload=False: n_times, sfreq e ch_names
saem do header sem materializar 129 canais a 500 Hz na RAM.

Uso: cd backend && python scripts/inventario_hbn.py [raiz_bids] [csv_saida]
Saída: ../relatorios/inventario_hbn.csv (uma linha por sujeito+tarefa)
"""
import csv
import sys
from pathlib import Path

import mne

COLUNAS = [
    "sujeito",
    "tarefa",
    "arquivo",
    "n_canais",
    "sfreq_hz",
    "duracao_s",
    "tem_events_tsv",
    "erro",
]


def _listar_sujeitos(raiz):
    """Pastas sub-* diretamente sob a raiz, em ordem alfabética. O sorted
    é explícito de propósito: a ordem de iterdir() depende do filesystem,
    e um .csv que muda de ordem entre máquinas torna o git diff inútil."""
    return sorted(p for p in raiz.iterdir() if p.is_dir() and p.name.startswith("sub-"))


def _listar_sets_do_sujeito(pasta_sujeito):
    """Os .set sob eeg/ do sujeito, ou None se a pasta eeg/ não existe.
    Não usa rglob: uma varredura cega entraria em derivatives/ e
    duplicaria linhas."""
    pasta_eeg = pasta_sujeito / "eeg"
    if not pasta_eeg.is_dir():
        return None
    return sorted(pasta_eeg.glob("*.set"))


def _tarefa_do_nome(nome):
    """Valor da entidade task- do nome BIDS. Procura o par que começa com
    'task-' em vez de usar índice fixo, porque run- e ses- deslocam as
    posições (task-surroundSupp_run-1_eeg.set). Sem a entidade devolve
    'desconhecida' — isto é inventário, não validador de nome."""
    for parte in nome.split("_"):
        if parte.startswith("task-"):
            return parte[len("task-"):]
    return "desconhecida"


def _caminho_events(caminho_set):
    """O events.tsv irmão do .set. Troca o sufixo _eeg.set por
    _events.tsv — with_suffix('.tsv') daria sub-X_task-Y_eeg.tsv, que
    não é nome BIDS e faria a coluna responder 'nao' para sempre."""
    nome = caminho_set.name
    if nome.endswith("_eeg.set"):
        return caminho_set.with_name(nome[: -len("_eeg.set")] + "_events.tsv")
    return caminho_set.with_name(caminho_set.stem + "_events.tsv")


def _ler_metadados(caminho_set):
    """(n_canais, sfreq, duracao_s, erro) de um .set. Único ponto que
    toca o MNE. Captura Exception larga de propósito: os modos de falha
    reais são heterogêneos (ValueError em arquivo corrompido,
    FileNotFoundError em .fdt ausente, e o que um release novo trouxer),
    e nenhum deles justifica derrubar a varredura inteira."""
    try:
        raw = mne.io.read_raw_eeglab(str(caminho_set), preload=False, verbose=False)
    except Exception as e:
        # a mensagem do MNE costuma ser multi-linha e quebraria o csv
        return None, None, None, str(e).replace("\n", " ").strip()[:200]

    sfreq = float(raw.info["sfreq"])
    # n_times/sfreq, não raw.times[-1]: este último dá (n-1)/sfreq e
    # subestima a duração em exatamente uma amostra
    duracao = raw.n_times / sfreq
    return len(raw.ch_names), sfreq, duracao, ""


def _linha_vazia(sujeito, erro):
    """Linha de sujeito cujo arquivo não pôde ser lido, com o erro na
    coluna `erro`.

    As colunas de medida ficam com string VAZIA, nunca com 0 nem NaN. É a
    regra central deste inventário: um zero seria uma AFIRMAÇÃO sobre a
    gravação ("tem zero canais"), e o que houve foi ausência de leitura.
    Trocar um pelo outro faria a média de canais do release despencar sem
    que nada na tabela denunciasse a causa."""
    return {
        "sujeito": sujeito,
        "tarefa": "",
        "arquivo": "",
        "n_canais": "",
        "sfreq_hz": "",
        "duracao_s": "",
        "tem_events_tsv": "nao",
        "erro": erro,
    }


def inventariar(raiz_bids, caminho_csv):
    """Varre a árvore BIDS em raiz_bids, grava o inventário em
    caminho_csv e devolve as linhas. Só levanta ValueError em erro do
    operador (caminho errado); tudo que é falha de dado vira coluna."""
    raiz = Path(raiz_bids)
    if not raiz.is_dir():
        raise ValueError(f"raiz bids não encontrada: {raiz}")

    sujeitos = _listar_sujeitos(raiz)
    if not sujeitos:
        raise ValueError(f"nenhum sujeito sub-* encontrado em: {raiz}")

    linhas = []
    for pasta_sujeito in sujeitos:
        sets = _listar_sets_do_sujeito(pasta_sujeito)

        if sets is None:
            linhas.append(_linha_vazia(pasta_sujeito.name, "pasta eeg/ ausente"))
            continue
        if not sets:
            linhas.append(_linha_vazia(pasta_sujeito.name, "nenhum .set em eeg/"))
            continue

        for caminho_set in sets:
            n_canais, sfreq, duracao, erro = _ler_metadados(caminho_set)
            linhas.append(
                {
                    "sujeito": pasta_sujeito.name,
                    "tarefa": _tarefa_do_nome(caminho_set.name),
                    "arquivo": caminho_set.relative_to(raiz).as_posix(),
                    "n_canais": "" if n_canais is None else n_canais,
                    "sfreq_hz": "" if sfreq is None else f"{sfreq:.1f}",
                    "duracao_s": "" if duracao is None else f"{duracao:.2f}",
                    "tem_events_tsv": "sim" if _caminho_events(caminho_set).is_file() else "nao",
                    "erro": erro,
                }
            )

    caminho_csv = Path(caminho_csv)
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUNAS)
        escritor.writeheader()
        escritor.writerows(linhas)

    _imprimir_resumo(linhas, raiz, caminho_csv)
    return linhas


def _imprimir_resumo(linhas, raiz, caminho_csv):
    """Resumo agregado POR TAREFA, impresso no terminal.

    Com 136 sujeitos e as tarefas do release, uma linha por arquivo daria
    mais de mil linhas de terminal — que ninguém lê, e portanto não é
    verificação de nada.

    Os erros são contados e listados em separado, e não somados ao resto:
    um release com 20 arquivos ilegíveis e um release completo têm de
    parecer diferentes no resumo, senão o inventário afirma cobertura que
    não tem. O CSV completo continua em disco para quem precisar do
    detalhe."""
    print(f"[inventario_hbn] {raiz} — {len(linhas)} linhas -> {caminho_csv}\n")

    ok = [l for l in linhas if l["erro"] == ""]
    com_erro = [l for l in linhas if l["erro"] != ""]

    tarefas = {}
    for l in ok:
        d = tarefas.setdefault(l["tarefa"], {"n": 0, "eventos": 0, "duracoes": []})
        d["n"] += 1
        d["eventos"] += 1 if l["tem_events_tsv"] == "sim" else 0
        d["duracoes"].append(float(l["duracao_s"]))

    print(f"{'tarefa':<32} {'arquivos':>9} {'c/events':>9} {'dur.média(s)':>13}")
    for tarefa in sorted(tarefas):
        d = tarefas[tarefa]
        media = sum(d["duracoes"]) / len(d["duracoes"])
        print(f"{tarefa:<32} {d['n']:>9} {d['eventos']:>9} {media:>13.1f}")

    taxas = sorted({l["sfreq_hz"] for l in ok})
    canais = sorted({l["n_canais"] for l in ok})

    print("\n--- veredito ---")
    print(f"sujeitos: {len({l['sujeito'] for l in linhas})} | tarefas distintas: {len(tarefas)}")

    if len(taxas) > 1:
        print(f"ATENÇÃO: {len(taxas)} taxas de amostragem distintas {taxas} — exige reamostragem")
    elif taxas:
        print(f"taxa de amostragem uniforme: {taxas[0]} Hz")

    if len(canais) > 1:
        print(f"ATENÇÃO: {len(canais)} contagens de canais distintas {canais} — exige interpolação")
    elif canais:
        print(f"contagem de canais uniforme: {canais[0]}")

    if com_erro:
        print(f"\n{len(com_erro)} linha(s) com erro (5 primeiras):")
        for l in com_erro[:5]:
            print(f"  {l['sujeito']:<20} {l['tarefa']:<24} {l['erro']}")
    else:
        print("nenhum erro de leitura")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config

    raiz = sys.argv[1] if len(sys.argv) > 1 else str(config.caminho_release())
    saida = sys.argv[2] if len(sys.argv) > 2 else str(
        config.caminho_relatorios() / "inventario_hbn.csv"
    )
    inventariar(raiz, saida)
