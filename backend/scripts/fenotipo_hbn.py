"""
Consolida o participants.tsv de um release do HBN-EEG e reporta a
distribuição de idade, gênero e das quatro dimensões de psicopatologia,
mais a contagem de quantos sujeitos têm cada campo ausente.

Existe porque o HBN não rotula "TDAH / controle" como o adhdata.csv faz
na coluna Class. Ele traz QUATRO DIMENSÕES CONTÍNUAS (p_factor,
attention, internalizing, externalizing), derivadas de um modelo
bifator sobre questionário. Um rótulo binário só sai de uma
limiarização de attention — que é decisão de método, a ser declarada e
justificada, não escondida no código. Antes de decidir o corte é
preciso saber se a coluna existe, como está distribuída, e quantos
sujeitos se perdem por ausência.

DECISÃO DE PROJETO: coluna ausente NÃO é erro fatal. Descobrir que uma
release não traz p_factor é o resultado principal, não uma falha —
o relatório escreve "AUSENTE NO RELEASE" e segue. Só levanta
ValueError em erro do operador (arquivo errado).

A linha que mais importa do relatório é completos_todas_dimensoes: o
número de sujeitos com as QUATRO dimensões preenchidas. É sempre menor
que o menor n individual, e é ele que limita a modelagem — não o total
de sujeitos do release.

ARMADILHA que este script trava: o BIDS marca ausente como "n/a"
minúsculo, e o pandas NÃO reconhece essa grafia por padrão (reconhece
"NA" e "N/A"). Sem na_values explícito, a coluna age vira dtype=object
com strings misturadas, e .mean() levanta TypeError — ou pior,
describe() devolve estatística de string sem erro nenhum.

Uso: cd backend && python scripts/fenotipo_hbn.py [participants.tsv] [txt_saida]
Saída: ../relatorios/fenotipo_hbn.txt
"""
import sys
from pathlib import Path

import pandas as pd

DIMENSOES = ["p_factor", "attention", "internalizing", "externalizing"]

# o BIDS usa "n/a"; os outros entram por segurança contra variação de release
AUSENTES = ["n/a", "N/A", "NA", "na", ""]


def _carregar_participantes(caminho_tsv):
    """Lê o participants.tsv com os ausentes virando NaN de verdade.

    O BIDS marca ausente como `n/a`, e o pandas não reconhece essa grafia
    sozinho: sem a lista AUSENTES, a coluna inteira viraria texto e a média
    de idade sairia como erro ou, pior, como contagem de strings.

    Levanta ValueError com o caminho quando o arquivo não existe, em vez de
    devolver um DataFrame vazio. Um relatório de fenótipo sobre zero
    sujeitos imprimiria tabelas vazias e pareceria dizer "não há ninguém com
    rótulo", que é uma conclusão, não uma falha de leitura."""
    caminho = Path(caminho_tsv)
    if not caminho.is_file():
        raise ValueError(f"participants.tsv não encontrado: {caminho}")

    df = pd.read_csv(caminho, sep="\t", na_values=AUSENTES, keep_default_na=True)

    # um .tsv lido como .csv vira uma coluna só — falha em voz alta
    if "participant_id" not in df.columns:
        raise ValueError(
            f"participants.tsv sem coluna participant_id: {caminho} "
            f"(colunas lidas: {list(df.columns)[:5]})"
        )

    for coluna in ["age"] + DIMENSOES:
        if coluna in df.columns:
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    return df


def _resumo_numerico(serie):
    """Estatísticas de uma coluna contínua, num dicionário.

    Traz `n` (quantos valores válidos) e `ausentes` SEMPRE, e os
    estatísticos (media, desvio, minimo, p25, mediana, p75, maximo) só
    quando há dado. Com n == 0 eles vêm None, e o formatador imprime '-'.

    Publicar `ausentes` ao lado de cada estatística é o ponto do relatório:
    uma média de idade calculada sobre 40 dos 136 sujeitos é um número
    diferente de uma calculada sobre 136, e sem a contagem ao lado os dois
    são indistinguíveis na tabela."""
    validos = serie.dropna()
    resumo = {"n": len(validos), "ausentes": int(serie.isna().sum())}
    if len(validos) == 0:
        for chave in ["media", "desvio", "minimo", "p25", "mediana", "p75", "maximo"]:
            resumo[chave] = None
        return resumo

    resumo.update(
        {
            "media": float(validos.mean()),
            "desvio": float(validos.std()),
            "minimo": float(validos.min()),
            "p25": float(validos.quantile(0.25)),
            "mediana": float(validos.median()),
            "p75": float(validos.quantile(0.75)),
            "maximo": float(validos.max()),
        }
    )
    return resumo


def _resumo_categorico(serie):
    """Contagem por valor, com os ausentes contados como categoria.

    `dropna=False` é deliberado. O padrão do pandas descarta os NaN em
    silêncio, e o total da coluna passaria a não bater com o número de
    sujeitos — sem nada na tabela explicando a diferença. Aqui os ausentes
    aparecem como uma linha, e a soma fecha."""
    return serie.value_counts(dropna=False).to_dict()


def _histograma_texto(serie, n_faixas=10, largura=40):
    """Distribuição em barras de '#', sem matplotlib — isto é relatório
    de terminal. As barras escalam pelo MAIOR bin, não pelo total: com
    distribuição concentrada, escalar pelo total achataria tudo a zero."""
    validos = serie.dropna()
    if len(validos) < 2 or validos.min() == validos.max():
        return ["  (sem dados suficientes para histograma)"]

    faixas = pd.cut(validos, bins=n_faixas)
    contagens = faixas.value_counts().sort_index()
    maior = contagens.max()

    linhas = []
    for intervalo, n in contagens.items():
        barra = "#" * int(round(largura * n / maior)) if maior else ""
        rotulo = f"[{intervalo.left:>7.2f}, {intervalo.right:>7.2f}]"
        linhas.append(f"  {rotulo} {n:>4}  {barra}")
    return linhas


def _formatar_numerico(nome, resumo):
    """Uma linha de tabela; None vira '-'."""
    def fmt(chave):
        valor = resumo[chave]
        return "-" if valor is None else f"{valor:.3f}"

    return (
        f"{nome:<16} {resumo['n']:>5} {resumo['ausentes']:>9} "
        f"{fmt('media'):>9} {fmt('desvio'):>9} {fmt('minimo'):>9} "
        f"{fmt('mediana'):>9} {fmt('maximo'):>9}"
    )


def fenotipo(caminho_tsv, caminho_saida):
    """O relatório de fenótipo completo: grava, imprime e devolve.

    Escreve o texto em `caminho_saida` (UTF-8), imprime o mesmo texto no
    terminal e devolve os resumos num dicionário, para quem quiser usar os
    números em vez de lê-los.

    O relatório responde a uma pergunta só, e ela é de viabilidade: QUANTOS
    sujeitos têm as quatro dimensões de psicopatologia preenchidas, e
    portanto podem receber um alvo. Idade e gênero entram porque são as
    covariáveis que qualquer split terá de equilibrar depois.

    O QUE ELE NÃO FAZ: não limiariza nada. Transformar `attention` contínua
    em rótulo binário de TDAH é uma decisão de método, com corte a
    justificar, e ela pertence ao capítulo de metodologia — não a um script
    de inventário que a aplicaria em silêncio."""
    df = _carregar_participantes(caminho_tsv)

    linhas = []
    linhas.append(f"[fenotipo_hbn] {caminho_tsv} — {len(df)} sujeitos")
    linhas.append("")

    presentes = [d for d in DIMENSOES if d in df.columns]
    ausentes_no_release = [d for d in DIMENSOES if d not in df.columns]

    # --- idade e dimensões, em tabela única ---
    linhas.append("distribuição das variáveis contínuas")
    linhas.append(
        f"{'campo':<16} {'n':>5} {'ausentes':>9} {'média':>9} {'desvio':>9} "
        f"{'mín':>9} {'mediana':>9} {'máx':>9}"
    )

    resumos = {}
    for nome in ["age"] + presentes:
        resumos[nome] = _resumo_numerico(df[nome])
        linhas.append(_formatar_numerico(nome, resumos[nome]))

    for nome in ausentes_no_release:
        linhas.append(f"{nome:<16} {'AUSENTE NO RELEASE':>50}")

    # --- gênero ---
    linhas.append("")
    linhas.append("distribuição de gênero (coluna sex)")
    if "sex" in df.columns:
        for valor, n in _resumo_categorico(df["sex"]).items():
            rotulo = "ausente" if pd.isna(valor) else str(valor)
            linhas.append(f"  {rotulo:<12} {n:>5}  ({100 * n / len(df):.1f}%)")
    else:
        linhas.append("  AUSENTE NO RELEASE")

    # --- histogramas ---
    linhas.append("")
    linhas.append("histograma de idade (anos)")
    linhas.extend(_histograma_texto(df["age"]) if "age" in df.columns else ["  AUSENTE NO RELEASE"])

    if "attention" in df.columns:
        linhas.append("")
        linhas.append("histograma de attention (dimensão que decide o alvo TDAH)")
        linhas.extend(_histograma_texto(df["attention"]))

    # --- contagem de ausentes por campo ---
    linhas.append("")
    linhas.append("contagem de ausentes por campo")
    linhas.append(f"{'campo':<16} {'ausentes':>9} {'%':>7} {'presentes':>10}")
    for nome in ["age", "sex"] + presentes:
        if nome not in df.columns:
            continue
        n_ausentes = int(df[nome].isna().sum())
        linhas.append(
            f"{nome:<16} {n_ausentes:>9} {100 * n_ausentes / len(df):>6.1f}% "
            f"{len(df) - n_ausentes:>10}"
        )
    for nome in ausentes_no_release:
        linhas.append(f"{nome:<16} {'coluna não existe no release':>35}")

    # --- veredito ---
    completos = int(df[presentes].notna().all(axis=1).sum()) if presentes else 0

    linhas.append("")
    linhas.append("--- veredito ---")
    linhas.append(f"sujeitos no release: {len(df)}")
    linhas.append(f"dimensões presentes: {len(presentes)}/4 {presentes}")
    if ausentes_no_release:
        linhas.append(f"dimensões AUSENTES: {ausentes_no_release}")
    linhas.append(
        f"completos_todas_dimensoes: {completos} — este é o n real utilizável, "
        f"sempre menor que o total; é ele que limita a modelagem"
    )
    if "attention" in df.columns:
        n_attention = int(df["attention"].notna().sum())
        linhas.append(
            f"attention preenchida em {n_attention}/{len(df)} sujeitos — "
            f"é a dimensão que vira o rótulo de TDAH por limiarização "
            f"(o corte é decisão de método, precisa ser declarado)"
        )

    texto = "\n".join(linhas)

    caminho_saida = Path(caminho_saida)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    # encoding explícito: sem ele o cp1252 default do Windows levanta
    # UnicodeEncodeError em "distribuição" e "gênero"
    caminho_saida.write_text(texto + "\n", encoding="utf-8")

    print(texto)
    print(f"\n[fenotipo_hbn] relatório gravado em {caminho_saida}")

    return {
        "n_sujeitos": len(df),
        "resumos": resumos,
        "dimensoes_presentes": presentes,
        "dimensoes_ausentes": ausentes_no_release,
        "completos_todas_dimensoes": completos,
    }


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config

    caminho = sys.argv[1] if len(sys.argv) > 1 else str(
        config.caminho_release() / "participants.tsv"
    )
    saida = sys.argv[2] if len(sys.argv) > 2 else str(
        config.caminho_relatorios() / "fenotipo_hbn.txt"
    )
    fenotipo(caminho, saida)
