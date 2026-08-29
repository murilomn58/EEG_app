"""
O único lugar do projeto que sabe ONDE o dado mora.

Existe por uma regra explícita: nenhum módulo deve carregar caminho de
dado fixo no meio do código. Antes desta centralização, cada script tinha
seu próprio default e eles já haviam divergido — dois apontavam para uma
pasta dentro do repositório, e o dado real estava em outra.

O dado fica FORA de qualquer repositório de propósito. O app EEG e o
projeto do transformer são separados e consomem o MESMO dado; apontar os
dois para um caminho neutro evita cópia duplicada e evita versionar dado
no git.

CORREÇÃO DE UM COMENTÁRIO QUE ESTAVA AQUI. Este texto dizia que "o erro a
não repetir está no próprio app: o adhdata.csv tem 267 MB versionados
aqui dentro". Não tem, e o comentário ensinava como erro justamente o que
o repositório faz certo. Conferido: `git ls-files adhdata.csv` devolve
vazio (o arquivo não está sob controle de versão) e
`git check-ignore -v adhdata.csv` aponta .gitignore:2 (está ignorado pelo
nome, com a justificativa do limite de 100 MB por arquivo do GitHub
escrita ali mesmo). O arquivo existe no diretório de trabalho, com
267.279.775 bytes, e nunca entrou no git.

O que sobra de real, e é o motivo de CAMINHO_ADHDATA continuar existindo:
o caminho do adhdata está amarrado à RAIZ DO REPOSITÓRIO, enquanto o do
HBN sai de EEG_DADOS. São duas políticas diferentes para o mesmo tipo de
coisa, e quem clona o repo recebe a pasta sem o CSV e tem de colocá-lo ali
à mão. Uniformizar é possível; enquanto não for feito, o custo fica
declarado aqui em vez de virar surpresa no primeiro clone.

Para apontar para outro lugar, defina a variável de ambiente EEG_DADOS.
Nada mais precisa mudar.

Uso: from config import caminho_release, CAMINHO_ADHDATA
"""
import os
from pathlib import Path

# Raiz dos releases de EEG baixados. Um subdiretório por accession.
RAIZ_DADOS = Path(os.environ.get("EEG_DADOS", r"C:\dados\hbn"))

# O CSV do adhdata é a exceção histórica: nasceu dentro da PASTA do
# repositório — ignorado pelo git, ver o cabeçalho — e continua lá porque
# o app inteiro aponta para ele. Fica aqui para que o dia da mudança seja
# uma linha, não uma caçada.
CAMINHO_ADHDATA = Path(__file__).resolve().parent.parent / "adhdata.csv"

# Accession do release em uso hoje. Trocar de release é trocar isto.
RELEASE_PADRAO = "ds005505"


def caminho_release(accession=None):
    """A raiz BIDS de um release. Não confere se existe: quem chama é que
    sabe se a ausência é erro (rodar o inventário) ou informação (o wizard
    dizendo que o dataset não está disponível)."""
    return RAIZ_DADOS / (accession or RELEASE_PADRAO)


def caminho_relatorios():
    """A pasta de relatórios, na raiz do repositório.

    Relatório é saída derivada, não dado: pode ser regerado a qualquer
    momento rodando os scripts de novo. Por isso ele fica DENTRO do
    repositório, ao contrário do dado bruto, e por isso a pasta pode entrar
    no .gitignore sem perda.

    Devolve o caminho mesmo que a pasta ainda não exista — quem grava cria,
    e criar aqui, num getter, seria efeito colateral escondido numa função
    cujo nome promete só responder uma pergunta."""
    return Path(__file__).resolve().parent.parent / "relatorios"


# POR QUE O WIZARD VALIDA UM BANCO ANTES DE DEIXAR ENTRAR
# --------------------------------------------------------
# Taxa de amostragem e conjunto de canais atravessam o app inteiro (buffers,
# montagem 3D, biquads, análise), e violá-los em silêncio não dá erro: dá
# tela plausível e errada. Uma gravação de 500 Hz tocada num relógio de
# 128 Hz anda a 3,9x a velocidade real, com beta e gama acima do Nyquist
# efetivo, e nada na tela denuncia isso.
#
# O que mudou: antes o app tinha UMA premissa global (128 Hz, 19 canais) e
# barrava todo banco que não batesse com ela. Agora a pergunta é outra, e são
# duas. Primeira: o sinal bate com o que a documentação DO BANCO declara?
# Divergir aí é atenção, não bloqueio — significa que a doc mente sobre o
# dado, o que é informação de QC e não motivo para barrar. Segunda: o
# pipeline consegue processar este sinal? Aí sim bloqueia, mas só diante de
# limite real — FS_MINIMA abaixo, e canal de análise sem tradução possível.

# O conjunto 10-20 sobre o qual a análise roda, na ordem que o resto do
# backend assume. Vive aqui, e não em csv_data, porque deixou de ser
# propriedade de um banco: é o alvo para o qual QUALQUER banco é traduzido.
CANAIS_10_20 = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]

# O topo da banda mais alta que o app calcula (gama, 30-45 Hz). É daqui que
# sai o único limite de taxa de amostragem que ainda bloqueia um banco.
BANDA_MAIS_ALTA_HZ = 45.0

# A taxa mínima que o pipeline consegue processar, com folga sobre o Nyquist
# exato: em fs = 2 x 45 = 90 Hz a banda gama cairia EM CIMA do Nyquist e o
# biquad correspondente sairia degenerado, não apenas apertado.
MARGEM_NYQUIST = 1.2
FS_MINIMA = 2.0 * BANDA_MAIS_ALTA_HZ * MARGEM_NYQUIST

# Mapa 10-20 da malha EGI GSN-HydroCel-129, do MAPA OFICIAL DO FABRICANTE.
# Duas fontes primárias independentes e concordantes:
#
#   1. EGI Technical Note, Luu & Ferree (2005), "Determination of the
#      HydroCel Geodesic Sensor Nets' Average Electrode Positions and Their
#      10-10 International Equivalents", Tabela 1.
#   2. Electrical Geodesics, Inc., "HydroCel Geodesic Sensor Net:
#      128-Channel Map", documento 8403486-52.
#
# NÃO inferir por proximidade geométrica. O Technical Note declara três
# regras de equivalência em ordem de autoridade, e a primeira é que posições
# de linha média do 10-10 têm de cair na linha média do array GSN mesmo
# quando o sensor geometricamente mais próximo não está nela. É isso que faz
# o Fz ser o E11 e não o E6 (y = +86,2 mm contra +41,2 mm) — a proximidade só
# vale como regra fora da linha média.
#
# A correspondência é aproximada por natureza: o mesmo documento informa que
# 42% das posições ficam a menos de 1 cm da posição 10-20 real, 42% entre 1 e
# 2 cm, e 16% entre 2,1 e 2,5 cm. É por isso que a tela pede confirmação
# visual em vez de aplicar o mapa em silêncio.
MAPA_EGI_1020 = {
    "Fp1": "E22", "Fp2": "E9",
    "F7": "E33", "F3": "E24", "Fz": "E11", "F4": "E124", "F8": "E122",
    "T7": "E45", "C3": "E36", "Cz": "E129", "C4": "E104", "T8": "E108",
    "P7": "E58", "P3": "E52", "Pz": "E62", "P4": "E92", "P8": "E96",
    "O1": "E70", "O2": "E83",
}

# Os bancos que o wizard oferece. Acrescentar um release novo é acrescentar
# uma entrada aqui, e não caçar caminho espalhado pelos módulos.
#
# `mapa_canais` traduz nome-de-análise para nome-no-arquivo. Vem vazio quando
# o banco já nomeia seus canais no padrão 10-20: a resolução tenta o nome
# direto antes de consultar mapa nenhum, então identidade não precisa ser
# escrita. `procedencia_mapa` existe para a tela poder distinguir mapa de
# fonte publicada de mapa inferido — inferido exige confirmação explícita.
DATASETS = {
    "adhdata": {
        "nome": "ADHD/Control children (adhdata.csv)",
        "tipo": "csv",
        "referencia_declarada": "linked-ears (A1/A2)",
        "fs_declarada": 128.0,
        "n_canais_declarado": 19,
        "canais_analise": CANAIS_10_20,
        "mapa_canais": {},
        "procedencia_mapa": "nomes 10-20 no próprio arquivo",
    },
    "hbn_ds005505": {
        "nome": "HBN-EEG Release 1 (ds005505)",
        "tipo": "bids",
        "accession": "ds005505",
        "referencia_declarada": "Cz",
        "fs_declarada": 500.0,
        "n_canais_declarado": 129,
        "canais_analise": CANAIS_10_20,
        "mapa_canais": MAPA_EGI_1020,
        "procedencia_mapa": "oficial: EGI doc. 8403486-52 e Luu & Ferree (2005)",
    },
}


def descrever_datasets():
    """Um item por banco, dizendo se está disponível no disco e, quando não
    estiver, por quê. O wizard mostra os indisponíveis com o motivo em vez
    de escondê-los: saber que o HBN existe e não está baixado é informação."""
    itens = []
    for ident, meta in DATASETS.items():
        item = dict(meta, id=ident)
        if meta["tipo"] == "csv":
            existe = CAMINHO_ADHDATA.is_file()
            item["caminho"] = str(CAMINHO_ADHDATA)
            item["motivo_indisponivel"] = None if existe else f"arquivo não encontrado: {CAMINHO_ADHDATA}"
        else:
            raiz = caminho_release(meta["accession"])
            sujeitos = sorted(p.name for p in raiz.glob("sub-*") if p.is_dir()) if raiz.is_dir() else []
            existe = bool(sujeitos)
            item["caminho"] = str(raiz)
            item["n_sujeitos"] = len(sujeitos)
            item["motivo_indisponivel"] = None if existe else (
                f"nenhum sujeito em {raiz} (defina EEG_DADOS ou baixe o release)"
            )
        item["disponivel"] = existe
        itens.append(item)
    return itens
