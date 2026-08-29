"""
Lê os eventos da tarefa cognitiva de uma gravação: quando cada marcador
aconteceu e o que ele significa.

Serve para o app desenhar etiquetas sobre o eixo de tempo, de modo que se
possa relacionar um evento do protocolo com uma alteração no traçado sem
sair da tela. É diferente do detector de artefato que o frontend já tem
(aquele acha instantes em que muitos canais desviam juntos, o que é ruído);
aqui os carimbos vêm do protocolo experimental.

DE ONDE VÊM OS DADOS: do próprio release, não de raspagem de site. O BIDS
grava um `<sujeito>_task-<tarefa>_events.tsv` ao lado de cada `.set`, com
`onset` em segundos relativo ao início da gravação, e um `value` legível.
A raiz do release traz o dicionário oficial `task-<tarefa>_events.json`,
com a descrição de cada nível em texto corrido e a anotação HED.

Isso é melhor que raspar um portal por duas razões práticas: o dicionário
é versionado junto do dado (então descreve exatamente esta versão) e é
citável, enquanto HTML de portal muda sem aviso.

O QUE ESTE MÓDULO DESCONFIA: o dicionário do HBN tem erro de copiar e
colar. No `task-RestingState_events.json`, o nível
`instructed_toCloseEyes` recebeu a MESMA descrição do
`instructed_toOpenEyes` ("A voice prompt instructed subject to open their
eyes"). Por isso a regra aqui é: quando dois níveis diferentes
compartilham a mesma descrição, os dois são marcados como divergentes, e
quem exibe deve preferir o `value` à descrição. A detecção é genérica,
não uma lista de casos conhecidos.

O adhdata.csv não tem coluna de evento nenhuma. Não é limitação deste
código: é do dataset, e é a razão de o app não ter componentes de ERP.
Nesse caso a resposta é lista vazia COM motivo declarado, nunca lista
vazia silenciosa.

Uso: cd backend && python scripts/eventos.py <arquivo .set ou .csv>
Saída: imprime os eventos encontrados e o veredito
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inventario_hbn import _tarefa_do_nome

# o BIDS marca ausente assim; sem isto o pandas deixaria a string "n/a"
AUSENTES = ["n/a", "N/A", "NA", "na", ""]


def _caminho_events_tsv(caminho_set):
    """O events.tsv irmão do .set. Troca o sufixo _eeg.set por
    _events.tsv, que é a convenção BIDS; with_suffix daria _eeg.tsv, que
    não existe."""
    nome = caminho_set.name
    if nome.endswith("_eeg.set"):
        return caminho_set.with_name(nome[: -len("_eeg.set")] + "_events.tsv")
    return caminho_set.with_name(caminho_set.stem + "_events.tsv")


def _caminho_sidecar(caminho_set, raiz=None):
    """O dicionário da tarefa, que mora na RAIZ do release e não junto do
    sujeito: um arquivo por tarefa, compartilhado por todos. Sobe a árvore
    procurando, para não depender de o chamador saber onde é a raiz."""
    tarefa = _tarefa_do_nome(caminho_set.name)
    alvo = f"task-{tarefa}_events.json"
    if raiz:
        candidato = Path(raiz) / alvo
        if candidato.is_file():
            return candidato
    pasta = caminho_set.parent
    for _ in range(4):  # eeg/ -> sub-X/ -> raiz/
        pasta = pasta.parent
        candidato = pasta / alvo
        if candidato.is_file():
            return candidato
    return None


def _descricoes_do_sidecar(caminho_json):
    """(descricoes_por_valor, divergencias). Duas coisas de uma vez porque
    a segunda só se descobre olhando a primeira inteira.

    A regra de divergência é genérica: se dois níveis DIFERENTES têm a
    mesma descrição, alguém copiou e colou, e a descrição não distingue os
    dois. Vale para o erro conhecido do HBN e para qualquer outro igual."""
    if not caminho_json or not caminho_json.is_file():
        return {}, []

    try:
        sidecar = json.loads(caminho_json.read_text(encoding="utf-8"))
    except Exception:
        return {}, []

    niveis = (sidecar.get("value") or {}).get("Levels") or {}
    descricoes = {k: str(v) for k, v in niveis.items()}

    por_descricao = {}
    for valor, texto in descricoes.items():
        por_descricao.setdefault(texto, []).append(valor)

    divergencias = []
    for texto, valores in por_descricao.items():
        if len(valores) > 1:
            divergencias.append({
                "valores": sorted(valores),
                "descricao_repetida": texto,
                "nota": "níveis distintos com a mesma descrição no dicionário do release; "
                        "prefira o valor à descrição",
            })
    return descricoes, divergencias


def _eventos_de_bids(caminho_set, raiz=None):
    """Os eventos de uma gravação BIDS, enriquecidos com o dicionário.

    Junta duas fontes: o `events.tsv` irmão do `.set`, que traz os carimbos
    de tempo, e o `task-<tarefa>_events.json` da raiz do release, que traz
    a descrição legível de cada nível.

    Sem o sidecar NÃO falha: devolve os eventos com `descricao` em None. O
    carimbo de tempo é o dado essencial e a descrição é conveniência; deixar
    de desenhar a etiqueta porque falta o dicionário seria perder o que
    importa por falta do acessório.

    Quando o dicionário se contradiz — dois níveis diferentes com a mesma
    descrição — o evento sai com `suspeita=True` e a divergência entra em
    `divergencias`. O app então mostra o VALOR do evento, não a descrição.
    Isso não é hipótese: o `task-RestingState_events.json` do ds005505
    descreve `instructed_toCloseEyes` com o texto de `toOpenEyes`."""
    caminho_tsv = _caminho_events_tsv(caminho_set)
    if not caminho_tsv.is_file():
        return {
            "eventos": [], "n": 0, "arquivo": None, "divergencias": [],
            "motivo": f"não há events.tsv ao lado de {caminho_set.name}",
        }

    tabela = pd.read_csv(caminho_tsv, sep="\t", na_values=AUSENTES, keep_default_na=True)
    descricoes, divergencias = _descricoes_do_sidecar(_caminho_sidecar(caminho_set, raiz))
    valores_divergentes = {v for d in divergencias for v in d["valores"]}

    eventos = []
    for _, linha in tabela.iterrows():
        valor = linha.get("value")
        valor = None if pd.isna(valor) else str(valor)
        codigo = linha.get("event_code")
        codigo = None if pd.isna(codigo) else str(codigo)
        duracao = linha.get("duration")

        eventos.append({
            "onset": float(linha["onset"]),
            # n/a em duration é o caso NORMAL aqui: o BIDS do HBN diz que a
            # duração é dada pelo próximo evento, não pela coluna
            "duracao": None if pd.isna(duracao) else float(duracao),
            "valor": valor,
            "codigo": codigo,
            "descricao": descricoes.get(valor),
            "suspeita": valor in valores_divergentes,
        })

    eventos.sort(key=lambda e: e["onset"])
    return {
        "eventos": eventos, "n": len(eventos),
        "arquivo": caminho_tsv.name,
        "divergencias": divergencias,
        "motivo": None if eventos else "o events.tsv existe mas está vazio",
    }


def detectar_eventos(caminho, subject_id=None, raiz=None):
    """Os eventos da tarefa de uma gravação, com o motivo quando não há.

    Despacha por formato, como carregar_raw faz: .set (BIDS/HBN) tem
    events.tsv ao lado; .csv (adhdata) não tem evento nenhum, e dizer isso
    com todas as letras é mais útil que devolver lista vazia."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise ValueError(f"arquivo não encontrado: {caminho}")

    if caminho.suffix == ".set":
        return _eventos_de_bids(caminho, raiz)

    if caminho.suffix == ".csv":
        return {
            "eventos": [], "n": 0, "arquivo": None, "divergencias": [],
            "motivo": "o adhdata.csv não traz coluna de evento; o dataset foi publicado "
                      "sem marcador de estímulo, e é por isso que o app não calcula "
                      "componentes de ERP",
        }

    raise ValueError(
        f"extensão não suportada: {caminho.suffix} (esperado .set do HBN ou .csv do adhdata)"
    )


def _imprimir(caminho, resultado):
    print(f"[eventos] {caminho}")
    if not resultado["eventos"]:
        print(f"  nenhum evento: {resultado['motivo']}")
        return

    print(f"  {resultado['n']} eventos em {resultado['arquivo']}\n")
    print(f"  {'onset (s)':>10}  {'valor':<26} {'código':<10} descrição")
    for e in resultado["eventos"][:20]:
        marca = " (!)" if e["suspeita"] else ""
        desc = (e["descricao"] or "")[:46]
        print(f"  {e['onset']:>10.3f}  {(e['valor'] or ''):<26} {(e['codigo'] or ''):<10} {desc}{marca}")
    if resultado["n"] > 20:
        print(f"  ... mais {resultado['n'] - 20}")

    print("\n--- veredito ---")
    distintos = len({e["valor"] for e in resultado["eventos"]})
    print(f"{resultado['n']} eventos, {distintos} tipos distintos, "
          f"de {resultado['eventos'][0]['onset']:.1f}s a {resultado['eventos'][-1]['onset']:.1f}s")
    if resultado["divergencias"]:
        print(f"\n{len(resultado['divergencias'])} divergência(s) no dicionário do release:")
        for d in resultado["divergencias"]:
            print(f"  {d['valores']} compartilham a descrição \"{d['descricao_repetida'][:60]}\"")
        print("  -> exiba o valor, não a descrição")
    else:
        print("nenhuma divergência no dicionário")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("uso: python scripts/eventos.py <arquivo .set ou .csv>")
        sys.exit(1)
    caminho = sys.argv[1]
    _imprimir(caminho, detectar_eventos(caminho))
