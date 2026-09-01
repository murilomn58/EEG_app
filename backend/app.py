import sys
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import mne
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import csv_data
import eletrodos
import mne_infer
import mne_setup
import verificar_referencia
# a função, e não o módulo: `canais` é nome de variável local em duas rotas
# deste arquivo, e importar o módulo inteiro criaria sombreamento silencioso
from canais import resolver as resolver_canais

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import eventos
import preproc_basico

CSV_PATH = config.CAMINHO_ADHDATA

# Quantas gravações BIDS INTEIRAS o backend guarda em memória ao mesmo tempo.
# Duas, e não mais, porque cada uma é a gravação completa do HBN (129 canais a
# 500 Hz por ~403 s = ~208 MB) e o padrão de uso real do app é alternar bruto /
# filtrado do MESMO sujeito, com no máximo uma comparação com o anterior. O
# número medido que motivou o limite está na docstring do lifespan.
LIMITE_CACHE_RAW = 2

# O corte de passa-baixa do conjunto clínico, em Hz. 70 Hz é o valor que o
# eletroencefalografista espera de um traçado de rotina: acima disso o que
# domina é EMG (contração de músculo temporal, principalmente) e ruído de alta
# frequência do amplificador, não ritmo cerebral. Note que ele fica ACIMA do
# topo da banda gama que o app calcula (45 Hz) de propósito — o filtro serve
# para o traçado que o médico LÊ, não para a análise de bandas.
H_FREQ_CLINICO_HZ = 70.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega, UMA vez na subida, o que é caro e não muda por requisição.

    São três coisas: o CSV inteiro (267 MB, alguns segundos), o modelo de
    fonte do MNE (forward mais operador inverso, dezenas de segundos) e o
    dicionário vazio que guarda as versões filtradas por sujeito.

    Fazer isso por requisição deixaria cada troca de sujeito no frontend
    esperando o mesmo trabalho de novo. O preço é o tempo de subida do
    servidor, pago uma vez, e é o preço certo para um app de uma pessoa só.

    OS DOIS CACHES TÊM TETO, e o raciocínio de "alguns MB" que já esteve
    escrito aqui valia só para o adhdata — foi escrito antes de existir caminho
    BIDS, e sobreviveu tempo demais.

    O `cache_filtrado` guarda a versão filtrada de um sujeito. No adhdata isso
    é da ordem de alguns MB; com o HBN passou a custar ~79 MB por entrada,
    MEDIDO: dez sujeitos filtrados levaram o RSS do uvicorn de 2150 MB para
    2937 MB, sem despejar nada. Hoje ele nasce OrderedDict e passa por
    _podar_cache antes de cada inserção, nos dois ramos do /raw-data.

    O `cache_raw` é o mais pesado dos dois. Cada entrada dele é a gravação
    INTEIRA de um sujeito do HBN, com os 129 canais: 129 x 500 Hz x ~403 s x
    8 bytes = ~208 MB por sujeito. MEDIDO nesta máquina, com o servidor de pé
    e o RSS lido do Win32_Process: 1003 MB com um sujeito em cache, 1549 MB
    depois de mais dois — ~273 MB por sujeito, contando as cópias reduzidas que
    passam por ali. Numa medição anterior o mesmo processo foi de 445 MB para
    1249 MB depois de quatro sujeitos. Sem despejo, trocar de sujeito no
    seletor é vazamento de memória com outro nome.

    Por isso ele nasce OrderedDict e guarda no máximo LIMITE_CACHE_RAW
    entradas, descartando a mais antiga. Não é política de expiração
    sofisticada, e não precisa ser: são duas linhas em _raw_do_banco e nenhuma
    dependência nova. Com o teto de pé, o mesmo servidor foi de 1561 MB com
    cinco sujeitos pedidos para 1623 MB com nove — 15 MB por sujeito novo
    contra os 273 MB de antes, e o que ainda sobe é working set que o Windows
    não devolve na hora, não cache crescendo."""
    app.state.df = csv_data.load_csv(CSV_PATH)
    # gravação filtrada por sujeito: filtrar com MNE leva segundos, e sem
    # cache alternar bruto/filtrado duas vezes no frontend refiltraria tudo
    app.state.cache_filtrado = OrderedDict()
    # gravação BIDS já lida por sujeito: ler um .set do HBN leva segundos, e
    # o frontend repede a mesma gravação a cada troca de bruto/filtrado.
    # OrderedDict e não dict comum: aqui a ORDEM é o mecanismo de despejo, e
    # dizer isso no tipo evita que alguém "simplifique" para dict sem perceber
    # que a política de descarte depende dela
    app.state.cache_raw = OrderedDict()
    forward, inverse_operator, n_vertices = mne_setup.build_source_model()
    app.state.forward = forward
    app.state.inverse_operator = inverse_operator
    app.state.n_vertices = n_vertices
    print(
        f"[startup] forward/inverse prontos: {n_vertices} vértices, "
        f"{len(csv_data.CANAIS_19)} canais"
    )
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _set_do_sujeito(meta, subject_id):
    """(arquivo .set, subject_id escolhido, raiz do release) de um banco BIDS.

    EXISTE PARA QUE A REGRA DE ESCOLHA DE SUJEITO SEJA UMA SÓ. Ela estava
    escrita três vezes — aqui, em /dataset-config e em /eventos — e duas
    dessas cópias tinham o mesmo defeito: o `next(..., sets[0])` caía no
    PRIMEIRO sujeito do release quando o id pedido não existia, e a rota
    devolvia 200 rotulado com o id PEDIDO. MEDIDO contra o backend de pé:
    /eventos?subject_id=sub-INEXISTENTE voltou "subject_id":
    "sub-INEXISTENTE" com os 36 eventos e os onsets (0.0, 1.288, 47.84) de
    sub-NDARAC904DMU, byte por byte iguais aos da chamada sem subject_id.
    Isso é servir o dado de um sujeito sob o nome de outro, que num app
    clínico é a pior classe de erro que existe: não parece erro nenhum.

    A distinção que o default None preserva: subject_id AUSENTE continua
    significando "escolha o primeiro", que é o comportamento legítimo de quem
    abre a tela sem ter escolhido ninguém ainda. Só o id PRESENTE e
    inexistente vira ValueError — quem chama traduz para 400."""
    raiz = config.caminho_release(meta["accession"])
    sets = sorted(raiz.glob("sub-*/eeg/*task-RestingState*.set"))
    if not sets:
        raise ValueError(f"nenhuma gravação encontrada em {raiz}")

    escolhido = subject_id or sets[0].parent.parent.name
    # default None e NÃO sets[0]: ver a docstring
    arquivo = next((s for s in sets if s.parent.parent.name == escolhido), None)
    if arquivo is None:
        raise ValueError(f"sujeito não encontrado: {escolhido}")

    return arquivo, escolhido, raiz


@app.get("/subjects")
def get_subjects(dataset_id: str = "adhdata"):
    """Os sujeitos disponíveis num banco, para o seletor do app.

    No adhdata sai do DataFrame já em memória, não do disco: é o que permite
    ao frontend montar o seletor sem esperar leitura nenhuma. Num banco BIDS
    sai da árvore de diretórios, que é barato — não abre nenhum .set."""
    meta = config.DATASETS.get(dataset_id)
    if meta is None:
        raise HTTPException(status_code=400, detail=f"dataset desconhecido: {dataset_id}")

    if meta["tipo"] == "csv":
        return csv_data.list_subjects(app.state.df)

    raiz = config.caminho_release(meta["accession"])
    sets = sorted(raiz.glob("sub-*/eeg/*task-RestingState*.set"))
    # a duração não vem daqui de propósito: sabê-la exigiria abrir cada .set,
    # e o seletor precisa aparecer antes disso. Quem quer a duração medida
    # chama /dataset-config, que já abre um arquivo e diz.
    return [
        {"id": s.parent.parent.name, "classe": None, "duracao_s": None}
        for s in sets
    ]


@app.get("/datasets")
def get_datasets():
    """Os bancos que o app conhece, com disponibilidade e motivo. O wizard
    lista os indisponíveis também: saber que o HBN existe e não está baixado
    é informação, esconder não é."""
    return {"datasets": config.descrever_datasets()}


@app.get("/dataset-config")
def get_dataset_config(dataset_id: str, subject_id: str = None):
    """O que o app MEDE de um banco, e como ele vai se CONFIGURAR para ele.

    É o passo 2 do wizard: o app mede e mostra, e o usuário confirma antes de
    entrar. Cada campo declara a origem, e a referência sai sempre como
    declarada e nunca como medida, porque ela não é detectável com segurança
    a partir do sinal.

    Bloqueia só diante de limite real do pipeline (taxa abaixo do mínimo,
    canal de análise sem tradução). Sinal que apenas diverge da documentação
    do banco passa com aviso: quem errou foi a doc, e o app segue o sinal."""
    meta = config.DATASETS.get(dataset_id)
    if meta is None:
        raise HTTPException(status_code=400, detail=f"dataset desconhecido: {dataset_id}")

    try:
        if meta["tipo"] == "csv":
            sujeitos = csv_data.list_subjects(app.state.df)
            escolhido = subject_id or sujeitos[0]["id"]
            # raw_de_dataframe e não carregar_raw: este último releria os
            # 267 MB do CSV a cada request
            raw = preproc_basico.raw_de_dataframe(app.state.df, escolhido)
            n_sujeitos = len(sujeitos)
        else:
            arquivo, escolhido, raiz = _set_do_sujeito(meta, subject_id)
            raw = preproc_basico.carregar_raw(arquivo)
            n_sujeitos = len(sorted(p for p in raiz.glob("sub-*") if p.is_dir()))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"falha ao ler o dataset: {e}")

    freq_rede, diagnostico = preproc_basico.detectar_frequencia_rede(raw)
    ref = verificar_referencia.diagnosticar(raw.get_data(picks="eeg"), raw.ch_names)

    fs = float(raw.info["sfreq"])
    n_canais = len(raw.ch_names)

    canais_analise = meta["canais_analise"]
    resolvido, faltantes = resolver_canais(
        raw.ch_names, canais_analise, meta.get("mapa_canais")
    )

    avisos = []

    # Bloqueio 1: limite físico do pipeline. Abaixo de FS_MINIMA a banda mais
    # alta não cabe sob o Nyquist e o filtro correspondente sai degenerado.
    # Note que fs ALTA não bloqueia mais: o app toca no relógio do sinal.
    if fs < config.FS_MINIMA:
        avisos.append({
            "codigo": "fs_incompativel", "severidade": "bloqueio",
            "texto": f"a {fs:.0f} Hz o Nyquist fica em {fs / 2:.0f} Hz e a banda gama "
                     f"(até {config.BANDA_MAIS_ALTA_HZ:.0f} Hz) não cabe; "
                     f"o mínimo processável é {config.FS_MINIMA:.0f} Hz",
        })

    # Bloqueio 2: canal de análise sem tradução. Ter MAIS canais que os 19 não
    # é problema — é o caso normal de malha densa, e a redução é justamente o
    # que o mapa faz. Problema é faltar um canal que a análise precisa.
    if faltantes:
        avisos.append({
            "codigo": "canais_incompativel", "severidade": "bloqueio",
            "texto": f"a análise usa {len(canais_analise)} canais 10-20 e "
                     f"{len(faltantes)} não têm correspondente neste banco "
                     f"({', '.join(faltantes)})",
        })

    # Divergência entre o que a doc declara e o que o sinal mostra é atenção,
    # não bloqueio: quem está errada é a documentação, e saber disso vale mais
    # do que barrar a entrada.
    if fs != meta["fs_declarada"]:
        avisos.append({
            "codigo": "fs_divergente_da_declarada", "severidade": "atencao",
            "texto": f"a documentação do banco declara {meta['fs_declarada']:.0f} Hz, "
                     f"o sinal mede {fs:.0f} Hz. O app segue o sinal",
        })
    if n_canais != meta["n_canais_declarado"]:
        avisos.append({
            "codigo": "canais_divergentes_da_declarada", "severidade": "atencao",
            "texto": f"a documentação do banco declara {meta['n_canais_declarado']} canais, "
                     f"o arquivo traz {n_canais}",
        })

    # Um canal de análise plano é quase sempre a referência física: um
    # eletrodo referenciado contra si mesmo dá zero. No HBN é o Cz, e ele
    # entra no conjunto de 19. Não bloqueia porque tem conserto conhecido
    # (re-referenciar para média comum devolve o sinal), mas não pode passar
    # calado: um Cz achatado no traçado é plausível e errado, que é
    # exatamente o que esta tela existe para impedir.
    flat_na_analise = [a for a, arquivo in resolvido.items() if arquivo in ref["canais_flat"]]
    if flat_na_analise:
        avisos.append({
            "codigo": "canal_analise_plano", "severidade": "atencao",
            "texto": f"{', '.join(flat_na_analise)} vem com desvio ~0 por ser a "
                     f"referência física da gravação. Sem re-referenciar para média "
                     f"comum, esse canal entra morto na análise",
        })

    # Mapa sem fonte publicada nunca entra em silêncio: quem confirma é o
    # usuário, na tela, e não uma heurística aqui.
    traduzidos = [a for a, arquivo in resolvido.items() if a != arquivo]
    if traduzidos and not meta.get("procedencia_mapa", "").startswith("oficial"):
        avisos.append({
            "codigo": "mapa_nao_oficial", "severidade": "atencao",
            "texto": f"{len(traduzidos)} canais vêm de mapa sem fonte publicada "
                     f"({meta.get('procedencia_mapa') or 'procedência não declarada'}); "
                     f"confira a correspondência antes de entrar",
        })

    if diagnostico.get("motivo") == "candidata_unica_por_nyquist":
        avisos.append({
            "codigo": "rede_por_eliminacao", "severidade": "atencao",
            "texto": f"{freq_rede:.0f} Hz venceu por Nyquist, não por medição. "
                     f"Confira a rede elétrica do local de gravação",
        })
    if freq_rede is None:
        avisos.append({
            "codigo": "rede_nao_detectada", "severidade": "atencao",
            "texto": f"nenhuma frequência de rede detectada ({diagnostico.get('motivo')}); "
                     f"o notch não será aplicado",
        })

    return {
        "dataset_id": dataset_id,
        "nome": meta["nome"],
        "amostra": {"subject_id": escolhido, "n_sujeitos": n_sujeitos},
        "detectado": {
            "fs_hz": fs,
            "n_canais": n_canais,
            "duracao_s": raw.n_times / fs,
            "freq_rede_hz": freq_rede,
            "freq_rede_diagnostico": diagnostico,
            "referencia": {
                "valor": meta["referencia_declarada"],
                "origem": "declarada",
                # o que o wizard pode OFERECER como troca de base neste banco.
                # Sai dos canais de análise, que são os que a base referencia:
                # 'orelha' exige A1/A2, e a malha GSN-HydroCel do HBN não tem
                "bases_disponiveis": preproc_basico.bases_disponiveis(
                    meta["canais_analise"]
                ),
                "evidencia_no_sinal": ref["evidencia"],
                "canais_flat": ref["canais_flat"],
            },
        },
        # O que o app vai USAR, agora que ele se adapta ao banco em vez de
        # exigir que o banco se adapte a ele. O frontend se configura a partir
        # daqui: é o que tira as constantes 128/19 de dentro do HTML.
        "esperado": {
            "fs_hz": fs,
            "n_canais": len(resolvido),
            "fs_minima_hz": config.FS_MINIMA,
        },
        "canais": {
            "analise": canais_analise,
            "resolvido": resolvido,
            "faltantes": faltantes,
            "traduzidos": traduzidos,
            "procedencia_mapa": meta.get("procedencia_mapa"),
        },
        "compativel": {
            "ok": not any(a["severidade"] == "bloqueio" for a in avisos),
            "avisos": avisos,
        },
    }


@app.get("/eletrodos")
def get_eletrodos(dataset_id: str, subject_id: str = None):
    """Onde cada eletrodo do banco está na cabeça, para a tela desenhar.

    É o que permite a conferência do mapa ser visual em vez de textual. A
    correspondência EGI→10-20 é aproximada por natureza — o fabricante
    informa desvios de até 2,5 cm —, então confirmar sobre uma lista de
    nomes é confirmar no escuro; sobre a cabeça desenhada, é conferência.

    Devolve TODOS os canais da gravação, não só os 19 escolhidos: o ponto da
    tela é justamente poder ver os outros 110 e trocar."""
    meta = config.DATASETS.get(dataset_id)
    if meta is None:
        raise HTTPException(status_code=400, detail=f"dataset desconhecido: {dataset_id}")

    try:
        if meta["tipo"] == "csv":
            sujeitos = csv_data.list_subjects(app.state.df)
            escolhido = subject_id or sujeitos[0]["id"]
            raw = preproc_basico.raw_de_dataframe(app.state.df, escolhido)
        else:
            raw, escolhido = _raw_do_banco(meta, subject_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"falha ao ler o dataset: {e}")

    resolvido, faltantes = resolver_canais(
        raw.ch_names, meta["canais_analise"], meta.get("mapa_canais")
    )
    descricao = eletrodos.descrever(raw, resolvido)

    return {
        "dataset_id": dataset_id,
        "subject_id": escolhido,
        "canais_analise": meta["canais_analise"],
        "sugestao": resolvido,
        "faltantes": faltantes,
        "procedencia_mapa": meta.get("procedencia_mapa"),
        **descricao,
    }


@app.get("/eventos")
def get_eventos(dataset_id: str, subject_id: str = None):
    """Os eventos do protocolo experimental de uma gravação, para o app
    desenhar etiquetas sobre o eixo de tempo.

    Quando não há evento, devolve lista vazia COM motivo. É a diferença
    entre "este dataset não tem marcador de estímulo" e "algo falhou", e
    quem lê a tela precisa saber qual dos dois."""
    meta = config.DATASETS.get(dataset_id)
    if meta is None:
        raise HTTPException(status_code=400, detail=f"dataset desconhecido: {dataset_id}")

    try:
        if meta["tipo"] == "csv":
            resultado = eventos.detectar_eventos(config.CAMINHO_ADHDATA)
        else:
            arquivo, escolhido, raiz = _set_do_sujeito(meta, subject_id)
            resultado = eventos.detectar_eventos(arquivo, raiz=raiz)
            resultado["subject_id"] = escolhido
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    resultado["dataset_id"] = dataset_id
    return resultado


def _h_freq_pedido(preproc):
    """O corte de passa-baixa a PEDIR para este preproc, ou None.

    Note que é o pedido, e não o aplicado. Quem rebaixa o pedido ao que cabe
    sob a Nyquist do arquivo é preproc_basico.limitar_h_freq, chamado por
    preprocessar antes de filtrar — 70 Hz num sinal de 128 Hz é impossível
    (Nyquist = 64 Hz) e viraria ValueError do MNE. A conta não é refeita aqui
    de propósito: duas cópias da mesma regra de Nyquist é como elas divergem.

    O que o app tem de fazer é NÃO ESCONDER o rebaixamento — por isso a
    resposta HTTP carrega decisoes["h_freq"], que é sempre o valor
    efetivamente aplicado (57,6 Hz no adhdata, 70 Hz no HBN). Escrever
    "70 Hz" sobre um traçado cortado em 57,6 Hz seria mentir para quem lê o
    traçado."""
    return H_FREQ_CLINICO_HZ if preproc == "clinico" else None


def _validar_base(base, ch_names):
    """Recusa uma base que este banco não comporta, com a razão.

    A lista do que dá para oferecer sai de preproc_basico.bases_disponiveis,
    e não de uma segunda cópia da regra aqui: duas cópias é como elas
    divergem, e a divergência apareceria como um botão no wizard que devolve
    400 ao ser clicado."""
    if base not in preproc_basico.BASES:
        raise HTTPException(
            status_code=400,
            detail=f"base inválida: {base} (use {', '.join(preproc_basico.BASES)})",
        )

    disponiveis = preproc_basico.bases_disponiveis(ch_names)
    if base not in disponiveis:
        raise HTTPException(
            status_code=400,
            detail=(
                f"base '{base}' não é aplicável a este banco "
                f"(disponíveis: {', '.join(disponiveis)})"
            ),
        )


def _preprocessar_sujeito(subject_id, canais, preproc="basico", base="nativa"):
    """Filtra a gravação de um sujeito do adhdata e devolve
    (canais_filtrados, decisoes).

    Em `basico` aplica passa-alta + notch (com a rede detectada do espectro) e
    NENHUM passa-baixa. Isso é deliberado e não se mexe: é exatamente o que o
    qc_relatorio.py mede, e se o frontend exibisse um filtro diferente do que o
    relatório mediu, o relatório deixaria de ser evidência sobre o que o
    usuário vê.

    Em `clinico` entra o passa-baixa, porque quem lê traçado espera vê-lo: sem
    passa-baixa, contração de músculo temporal entra inteira no registro e
    parece atividade rápida. O corte pedido é de 70 Hz, mas o adhdata está a
    128 Hz e 70 não cabe sob o Nyquist — quem resolve isso é
    preproc_basico.limitar_h_freq, dentro de preprocessar, e o valor
    efetivamente aplicado (57,6 Hz) volta em decisoes["h_freq"] para a tela
    poder escrevê-lo."""
    dados = np.array([canais[c] for c in csv_data.CANAIS_19], dtype=float) * 1e-6
    info = mne.create_info(list(csv_data.CANAIS_19), csv_data.FS, "eeg", verbose=False)
    raw = mne.io.RawArray(dados, info, verbose=False)

    filtrado, decisoes = preproc_basico.preprocessar(
        raw, l_freq=0.5, h_freq=_h_freq_pedido(preproc), base=base
    )
    saida = filtrado.get_data() * 1e6  # de volta pra µV, a escala do frontend
    return {c: saida[i].tolist() for i, c in enumerate(csv_data.CANAIS_19)}, decisoes


def _podar_cache(cache):
    """Deixa espaço para mais uma entrada, despejando a mais antiga.

    Existe porque `cache_filtrado` tinha o MESMO defeito que `cache_raw` já
    teve, e ficou para trás quando aquele foi consertado: dicionário sem teto,
    justificado por uma docstring que falava em "alguns MB" do adhdata. Com o
    caminho BIDS servindo o HBN, cada entrada passou a custar ~79 MB, medidos:
    dez sujeitos filtrados levaram o RSS do uvicorn de 2150 MB para 2937 MB,
    sem despejar nada.

    O teto é o mesmo LIMITE_CACHE_RAW, e pelo mesmo motivo: o uso real é
    alternar entre versões do MESMO sujeito, e no máximo comparar com o
    anterior."""
    while len(cache) >= LIMITE_CACHE_RAW:
        cache.pop(next(iter(cache)))


def _raw_do_banco(meta, subject_id):
    """(raw, subject_id) de um banco BIDS, com cache.

    Ler um .set do HBN é da ordem de segundos, e o frontend pede a mesma
    gravação de novo a cada troca de bruto/filtrado. Sem cache, cada clique
    releria o arquivo inteiro — o mesmo motivo que já justifica o
    cache_filtrado do adhdata.

    O cache tem TETO, ao contrário do cache_filtrado: cada entrada aqui é a
    gravação inteira, ~208 MB por sujeito do HBN, e o RSS do uvicorn medido
    subiu ~273 MB a cada sujeito novo (números e método na docstring do
    lifespan). Guardar dois e descartar o mais antigo cobre o uso real —
    alternar bruto/filtrado do mesmo sujeito, comparar com o anterior — sem
    deixar o processo crescer com o número de cliques.

    O despejo é escrito com pop/next(iter(...)) em vez de popitem(last=False)
    de propósito: assim ele funciona igual se alguém (um teste, tipicamente)
    trocar o OrderedDict por um dict comum, que em Python 3.7+ também preserva
    a ordem de inserção. É a diferença entre um teste que falha por AttributeError
    e um teste que mede o que quer medir.

    A escolha do arquivo saiu daqui para `_set_do_sujeito`: esta era a única
    das três cópias da regra que recusava um id inexistente, e mantê-la
    separada era garantir que as outras duas voltassem a divergir."""
    arquivo, escolhido, _ = _set_do_sujeito(meta, subject_id)

    if escolhido in app.state.cache_raw:
        # tira e recoloca: "mais antigo" tem de significar usado há mais tempo,
        # e não "entrou primeiro e nunca mais saiu". Sem isto, o sujeito que o
        # usuário está olhando agora seria despejado antes do que ele abandonou
        app.state.cache_raw[escolhido] = app.state.cache_raw.pop(escolhido)
    else:
        app.state.cache_raw[escolhido] = preproc_basico.carregar_raw(arquivo)
        while len(app.state.cache_raw) > LIMITE_CACHE_RAW:
            app.state.cache_raw.pop(next(iter(app.state.cache_raw)))

    return app.state.cache_raw[escolhido], escolhido


def _amplitude_referencia(valores):
    """O percentil 99 do módulo, para o frontend auto-escalar o traçado.

    Existe porque a AMPLITUDE varia muito entre bancos, mesmo os dois estando
    em µV. Medido nesta máquina, sujeito sub-NDARAC904DMU do HBN contra v10p
    do adhdata: no sinal BRUTO o p99 do HBN dá ~89.000 µV, número que não é
    fisiológico e é dominado pelo offset DC — o próprio passa-alta o derruba
    para 235,8 µV, contra 455,3 µV do adhdata sob o MESMO filtro (o bruto do
    adhdata dá 568,0). Ou seja: depois do tratamento o HBN é MENOR que o
    adhdata e cabe na faixa fisiológica.

    O que a evidência sustenta sobre o HBN é que a calibração NÃO ESTÁ
    CONFIRMADA — não que a escala seja arbitrária. Por isso o app converte os
    dois bancos para µV e escala o traçado por este percentil em vez de fixar
    um ganho: fixá-lo faria o bruto do HBN virar uma linha reta na tela por
    causa do offset, e inventar um fator de conversão seria pior ainda, porque
    afirmaria uma calibração que ninguém verificou."""
    return float(np.percentile(np.abs(valores), 99))



@app.get("/raw-data")
def get_raw_data(
    subject_id: str,
    preproc: str = "nenhum",
    dataset_id: str = "adhdata",
    base: str = "nativa",
):
    """A gravação de um sujeito, crua ou pré-processada. Os metadados da
    decisão (qual rede foi detectada, quais harmônicos saíram) viajam
    junto com o dado, para a interface poder dizer ao usuário o que foi
    removido em vez de só mostrar um traçado diferente.

    `fs` e `unidade` saem daqui em vez de serem premissa do frontend: é o
    que permite o app tocar no relógio do próprio sinal e dizer, no mesmo
    campo, o que se sabe sobre a calibração de cada banco.

    UNIDADE: os dois ramos (CSV e BIDS) devolvem µV. Já não foi assim — o ramo
    BIDS devolvia volts crus, 1e6 de diferença dentro do MESMO endpoint,
    invisível porque o frontend auto-escala pelo amplitude_p99. Uma tela que
    escreve "µV" ao lado de um número em volts é uma tela errada esperando um
    leitor distraído.

    Os três valores de `preproc`:

      nenhum  — o sinal como está no arquivo, sem filtro.
      basico  — passa-alta 0,5 Hz + notch na rede detectada. É o que o
                qc_relatorio.py mede, e por isso NÃO muda.
      clinico — o anterior mais passa-baixa (70 Hz, ou o que couber sob o
                Nyquist). É o conjunto que um eletroencefalografista espera
                ver num traçado de rotina.

    `base` é a referência contra a qual o sinal é lido: nativa (a do arquivo),
    car, cz ou orelha. Trocá-la REFAZ o pré-processamento a partir do bruto,
    em vez de re-referenciar o sinal já filtrado — e é por isso que ela vive
    aqui e não no frontend. Antes desta rota o CAR que o usuário recebia era
    um laço em JavaScript, e o CAR testado (o do MNE, que as figuras do
    caderno usam) não era o entregue.

    Base só se aplica sobre sinal pré-processado: com preproc=nenhum o
    endpoint devolve o sinal do arquivo, e re-referenciar ali seria entregar
    sinal tratado sob o rótulo de bruto."""
    if preproc not in ("nenhum", "basico", "clinico"):
        raise HTTPException(
            status_code=400,
            detail=f"preproc inválido: {preproc} (use nenhum, basico ou clinico)",
        )

    if preproc == "nenhum" and base != "nativa":
        raise HTTPException(
            status_code=400,
            detail=(
                f"base '{base}' pedida com preproc=nenhum: a troca de base exige "
                "pré-processamento (use preproc=basico ou clinico)"
            ),
        )

    meta = config.DATASETS.get(dataset_id)
    if meta is None:
        raise HTTPException(status_code=400, detail=f"dataset desconhecido: {dataset_id}")

    if meta["tipo"] != "csv":
        return _raw_data_bids(meta, dataset_id, subject_id, preproc, base)

    _validar_base(base, csv_data.CANAIS_19)

    try:
        canais = csv_data.get_subject_raw(app.state.df, subject_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    resposta = {
        "subject_id": subject_id,
        "dataset_id": dataset_id,
        "fs": csv_data.FS,
        "channels": canais,
        "preproc": preproc,
        "base": base,
        "unidade": "µV",
        "amplitude_p99": _amplitude_referencia(
            np.array([canais[c] for c in csv_data.CANAIS_19], dtype=float)
        ),
    }
    if preproc == "nenhum":
        return resposta

    # o preproc e a base entram na chave: são sinais DIFERENTES sobre o mesmo
    # sujeito, e servir um pelo outro seria mostrar um traçado com legenda
    # errada
    chave = (subject_id, preproc, base)
    if chave not in app.state.cache_filtrado:
        try:
            filtrados, decisoes = _preprocessar_sujeito(subject_id, canais, preproc, base)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"falha no pré-processamento: {e}")
        diagnostico = decisoes["diagnostico_rede"] or {}
        _podar_cache(app.state.cache_filtrado)
        app.state.cache_filtrado[chave] = {
            "channels": filtrados,
            "freq_rede_hz": decisoes["freq_rede"],
            "freq_rede_motivo": diagnostico.get("motivo"),
            "harmonicos_notchados": decisoes["harmonicos_notchados"],
            "l_freq": decisoes["l_freq"],
            "canais_referencia": decisoes["canais_referencia"],
            # o Cz do HBN vem plano por ser a referência física, e a tela
            # precisa poder dizer se a base escolhida o devolveu ao traçado
            "referencia_fisica": decisoes["referencia_fisica"],
            # o corte que FOI aplicado, não o que foi pedido: em basico é None
            # (não há passa-baixa) e em clinico no adhdata é 57,6 e não 70,
            # porque a 128 Hz o Nyquist não deixa. O pedido original fica em
            # decisoes["h_freq_pedido"], para quem quiser auditar a diferença
            "h_freq": decisoes["h_freq"],
        }

    resposta.update(app.state.cache_filtrado[chave])
    # O amplitude_p99 calculado lá em cima é o do sinal BRUTO. Depois do
    # update ele continuaria descrevendo um sinal que não é mais o que está em
    # `channels` — no v10p ficava 568,0 tanto sem filtro quanto com, quando o
    # filtrado mede 455,3. Recalcular aqui é o que mantém o campo descrevendo
    # o que a resposta de fato carrega.
    resposta["amplitude_p99"] = _amplitude_referencia(
        np.array([resposta["channels"][c] for c in csv_data.CANAIS_19], dtype=float)
    )
    return resposta


def _raw_data_bids(meta, dataset_id, subject_id, preproc, base="nativa"):
    """A gravação de um banco BIDS, já reduzida ao conjunto de análise.

    A redução acontece AQUI, no backend, e não no frontend: é o que a tela
    de conferência dizia que faltava. Os canais saem nomeados em 10-20
    (Fz, e não E11), porque o resto do app raciocina em 10-20 — o nome do
    arquivo viaja junto, em `canais_origem`, para a tela poder mostrar de
    qual eletrodo cada traçado veio.

    A BASE é aplicada sobre o sinal JÁ REDUZIDO aos canais de análise, e não
    sobre os 129 do arquivo. É uma decisão, não um detalhe: a média comum de
    19 canais 10-20 não é a mesma que a de 129 sensores da malha inteira, e é
    a dos 19 que corresponde ao que o app mostra e mede. A divergência conexa
    já está registrada — o qc_relatorio mede os 129 do arquivo enquanto o app
    serve 19, e no sub-NDARAC904DMU isso é a diferença entre 1,52 dB (passa) e
    6,20 dB (reprova)."""
    try:
        raw, escolhido = _raw_do_banco(meta, subject_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"falha ao ler o dataset: {e}")

    canais_analise = meta["canais_analise"]
    resolvido, faltantes = resolver_canais(
        raw.ch_names, canais_analise, meta.get("mapa_canais")
    )
    if faltantes:
        raise HTTPException(
            status_code=400,
            detail=f"canais sem correspondente neste banco: {', '.join(faltantes)}",
        )

    nomes_arquivo = [resolvido[c] for c in canais_analise]
    reduzido = raw.copy().pick(nomes_arquivo).reorder_channels(nomes_arquivo)

    fs = float(raw.info["sfreq"])

    # sobre os canais de ANÁLISE, que são os que a base vai referenciar
    _validar_base(base, canais_analise)

    decisoes = None
    if preproc in ("basico", "clinico"):
        # o preproc e a base entram na chave junto com o banco e o sujeito:
        # são sinais diferentes e não podem compartilhar entrada
        chave = (dataset_id, escolhido, preproc, base)
        if chave not in app.state.cache_filtrado:
            try:
                filtrado, decisoes = preproc_basico.preprocessar(
                    reduzido, l_freq=0.5, h_freq=_h_freq_pedido(preproc), base=base
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"falha no pré-processamento: {e}")
            _podar_cache(app.state.cache_filtrado)
            app.state.cache_filtrado[chave] = (filtrado.get_data(), decisoes)
        dados, decisoes = app.state.cache_filtrado[chave]
    else:
        dados = reduzido.get_data()

    # volts -> µV. O MNE trabalha em volts por dentro, e este ramo devolvia o
    # volt cru enquanto o ramo do CSV já devolvia µV: 1e6 de diferença no mesmo
    # endpoint, escondida pelo auto-escalonamento do frontend. Converter aqui é
    # o que torna `unidade` uma afirmação verdadeira para os dois bancos.
    dados = dados * 1e6

    resposta = {
        "subject_id": escolhido,
        "dataset_id": dataset_id,
        "fs": fs,
        "channels": {c: dados[i].tolist() for i, c in enumerate(canais_analise)},
        "canais_origem": resolvido,
        "preproc": preproc,
        "base": base,
        # µV, como o ramo do CSV — a ressalva é sobre a CALIBRAÇÃO, não sobre a
        # unidade. O que a evidência sustenta: ninguém confirmou o ganho deste
        # banco contra um sinal de referência conhecido. O que ela NÃO sustenta,
        # e já foi escrito aqui: que a escala seja arbitrária. O p99 do bruto
        # (~89.000) é dominado pelo offset DC; depois do passa-alta ele cai para
        # 235,8 contra 455,3 do adhdata sob o mesmo filtro — MENOR, e dentro da
        # faixa fisiológica. Comparar amplitude entre bancos ainda pede cautela;
        # chamar a escala de "arbitrária" era ir além do medido.
        "unidade": "µV (calibração não confirmada neste banco)",
        "amplitude_p99": _amplitude_referencia(dados),
    }
    if decisoes is not None:
        diagnostico = decisoes["diagnostico_rede"] or {}
        resposta.update({
            "freq_rede_hz": decisoes["freq_rede"],
            "freq_rede_motivo": diagnostico.get("motivo"),
            "harmonicos_notchados": decisoes["harmonicos_notchados"],
            "l_freq": decisoes["l_freq"],
            # o corte aplicado de verdade; None quando não houve passa-baixa
            "h_freq": decisoes["h_freq"],
        })
    return resposta


class SourceLocalizationRequest(BaseModel):
    subject_id: str
    t_start: float
    t_end: float
    method: str = "dSPM"
    dataset_id: str = "adhdata"


@app.post("/source-localization")
def post_source_localization(req: SourceLocalizationRequest):
    """Estima, para uma janela de tempo, a atividade nos vértices do córtex.

    Recebe sujeito e o intervalo [t_start, t_end] em segundos, e devolve um
    valor por vértice do modelo de fonte, para o app colorir o cérebro 3D.

    Os dois try/except são separados de propósito e devolvem 400 nos dois
    casos, com mensagens distintas: o primeiro é erro de PEDIDO (sujeito
    inexistente, janela fora da gravação) e o segundo é falha do MNE ao
    resolver o problema inverso. Um 500 no segundo caso mandaria o usuário
    procurar bug no servidor quando o que houve foi um pedido que o método
    não consegue atender."""
    # O modelo de fonte é construído uma vez, no startup, a partir de um info
    # de 128 Hz com os 19 canais do adhdata. Rodar o HBN por ele daria número:
    # 500 Hz tocados num info de 128, e um ganho que ninguém confirmou entrando
    # numa solução inversa que depende de amplitude absoluta. O dSPM é
    # normalizado, então o mapa até sairia plausível — e errado, sem nada na
    # tela denunciando. Recusar é a resposta honesta até o modelo passar a ser
    # construído por banco.
    meta = config.DATASETS.get(req.dataset_id)
    if meta is None:
        raise HTTPException(status_code=400, detail=f"dataset desconhecido: {req.dataset_id}")
    if meta["tipo"] != "csv":
        raise HTTPException(
            status_code=400,
            detail=f"reconstrução de fonte ainda não cobre {req.dataset_id}: o modelo "
                   f"é construído a {csv_data.FS:.0f} Hz com os "
                   f"{len(csv_data.CANAIS_19)} canais do adhdata, e este banco chega a "
                   f"{meta['fs_declarada']:.0f} Hz com calibração não confirmada",
        )

    try:
        janela = csv_data.get_window(app.state.df, req.subject_id, req.t_start, req.t_end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        valores = mne_infer.apply_source_localization(
            janela, app.state.inverse_operator, method=req.method
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"falha no MNE: {e}")

    return {
        "values": valores.tolist(),
        "n_vertices": app.state.n_vertices,
        "time": req.t_start,
        "method": req.method,
    }
