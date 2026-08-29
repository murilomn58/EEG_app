"""Posições 3D dos eletrodos de um banco, na convenção que o app 3D usa.

Existe para a tela de conferência poder MOSTRAR onde cada eletrodo está,
em vez de só listar nomes. A correspondência EGI→10-20 é aproximada por
natureza (o próprio fabricante informa desvios de até 2,5 cm), então pedir
confirmação sobre uma lista de texto seria pedir fé; sobre a cabeça
desenhada, é conferência.

CONVENÇÃO DE COORDENADAS — a parte fácil de errar em silêncio.

O MNE entrega posições em coordenadas de cabeça, em metros:
    x = direita, y = frente (anterior), z = cima (superior)

O app desenha com o eixo Y para cima e o Z para a frente, e posiciona cada
eletrodo por um par (theta, phi) em graus que `dirEsfera` no frontend
converte em direção:
    theta = ângulo a partir do topo da cabeça (0° = Cz)
    phi   = ângulo em torno do eixo vertical, medido da frente para a direita

Traduzir errado entre os dois não levanta erro nenhum: os eletrodos apenas
aparecem espelhados ou girados, e a tela fica plausível e errada — que é o
que esta camada existe para impedir. Por isso a tradução mora aqui, num
lugar só, com teste que ancora Cz no topo e Fz na frente.
"""
import mne
import numpy as np

from canais import normalizar

# Montagem template por número de canais, para bancos que não trazem a
# digitalização no arquivo. É o segundo melhor: posições médias de população,
# não deste sujeito. Quem usa declara isso na tela.
TEMPLATES = {
    129: "GSN-HydroCel-129",
    128: "GSN-HydroCel-128",
}
TEMPLATE_10_20 = "standard_1020"
# Superconjunto do 10-20 e do 10-10, com 343 posicoes. E a montagem que
# atende banco de 32, 64 ou 128 canais nomeados no padrao internacional.
TEMPLATE_10_05 = "standard_1005"


def para_esfericas(xyz):
    """(theta, phi) em graus a partir de um xyz em coordenadas de cabeça.

    Ver a convenção no topo do módulo. Devolve None quando o ponto está na
    origem, onde a direção é indefinida — e chutar 0,0 ali colocaria o
    eletrodo no topo da cabeça como se fosse medição."""
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    r = np.sqrt(x * x + y * y + z * z)
    if not np.isfinite(r) or r < 1e-9:
        return None
    theta = np.degrees(np.arccos(np.clip(z / r, -1.0, 1.0)))
    phi = np.degrees(np.arctan2(x, y))
    return round(float(theta), 2), round(float(phi), 2)


def _limpar(ch_pos):
    """Só os canais com coordenada utilizável. O MNE devolve NaN para
    eletrodo sem digitalização, e NaN vira `null` no JSON, que no frontend
    vira eletrodo desenhado no meio do cérebro."""
    return {
        nome: p for nome, p in (ch_pos or {}).items()
        if p is not None and len(p) == 3 and np.all(np.isfinite(p))
    }


def posicoes(raw):
    """(dict nome->xyz, origem) para os canais de uma gravação.

    Duas autoridades, nesta ordem: a digitalização que veio no próprio
    arquivo (posições DESTE sujeito) e, na falta dela, a melhor montagem
    template — ver `_do_template`, que escolhe por cobertura medida e não
    por contagem de canais.

    A origem volta junto porque a diferença importa: template é média de
    população, e quem olha a tela precisa saber qual dos dois está vendo."""
    try:
        montagem = raw.get_montage()
    except Exception:
        montagem = None

    if montagem is not None:
        do_arquivo = _limpar(montagem.get_positions().get("ch_pos"))
        if do_arquivo:
            return do_arquivo, "digitalização do próprio arquivo"

    return _do_template(raw)


def _do_template(raw):
    """A melhor montagem template para esta gravação, e quantos ela cobre.

    Escolher pela CONTAGEM de canais era o critério antigo, e ele falha em
    silêncio no caso mais comum de banco de fora. Uma gravação de 64 canais
    não está em TEMPLATES, caía no `standard_1020` de 19 nomes, e a
    interseção dava zero — a tela de conferência aparecia sem eletrodo
    nenhum e sem motivo. O `standard_1005` cobre 10-20, 10-10 e 10-05 com
    343 posições, e é ele que atende esse caso.

    Por isso a escolha agora é por COBERTURA MEDIDA: tenta cada candidata e
    fica com a que nomeia mais canais desta gravação. Contagem continua
    valendo como desempate à frente da lista, porque para a malha EGI ela é
    a informação certa: `E11` não existe em montagem 10-05 nenhuma.

    A comparação de nome usa a forma normalizada de `canais.normalizar`
    (sem caixa, sem ponto de preenchimento do EDF), senão `Cz..` não acha o
    `Cz` da montagem — que é exatamente o que acontecia com o eegmmidb."""
    candidatas = []
    por_contagem = TEMPLATES.get(len(raw.ch_names))
    if por_contagem:
        candidatas.append(por_contagem)
    candidatas += [TEMPLATE_10_05, TEMPLATE_10_20]

    # nome-normalizado -> nome-no-arquivo, para devolver a chave que o MNE aceita
    do_arquivo = {}
    for nome in raw.ch_names:
        do_arquivo.setdefault(normalizar(nome), nome)

    melhor, melhor_nome, melhor_n = {}, None, 0
    for nome_template in candidatas:
        try:
            template = mne.channels.make_standard_montage(nome_template)
        except Exception:
            continue
        posicoes_template = _limpar(template.get_positions().get("ch_pos"))

        presentes = {}
        for nome_na_montagem, xyz in posicoes_template.items():
            alvo = do_arquivo.get(normalizar(nome_na_montagem))
            if alvo is not None:
                presentes[alvo] = xyz

        if len(presentes) > melhor_n:
            melhor, melhor_nome, melhor_n = presentes, nome_template, len(presentes)

    if not melhor:
        return {}, None
    return melhor, f"montagem template {melhor_nome} (média de população)"


def descrever(raw, resolvido=None):
    """Lista pronta para a tela: um item por canal da gravação, com posição
    esférica e para qual canal de análise ele foi escolhido (se foi).

    Canal sem posição utilizável entra na lista mesmo assim, com `pos: None`:
    escondê-lo faria a tela mostrar 127 eletrodos e dizer 129."""
    ch_pos, origem = posicoes(raw)
    # invertido: nome-no-arquivo -> nome-de-análise, que é o que a tela rotula
    papel = {arquivo: alvo for alvo, arquivo in (resolvido or {}).items()}

    itens = []
    for nome in raw.ch_names:
        xyz = ch_pos.get(nome)
        esf = para_esfericas(xyz) if xyz is not None else None
        itens.append({
            "nome": nome,
            "pos": {"theta": esf[0], "phi": esf[1]} if esf else None,
            "xyz": [round(float(v), 5) for v in xyz] if xyz is not None else None,
            "sugerido_para": papel.get(nome),
        })
    return {"origem_posicoes": origem, "eletrodos": itens}
