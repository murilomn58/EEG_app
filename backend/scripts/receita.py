# -*- coding: utf-8 -*-
"""A receita de uma análise: tudo o que é preciso para refazê-la igual.

O BURACO QUE ELA FECHA

O `decisoes` do pré-processamento — filtro aplicado, rede detectada,
rebaixamento por Nyquist — já existe e é bem feito. Só que ele viaja na
resposta HTTP e morre no selo de texto da tela. Nenhuma das cinco medidas
quantitativas do app (TBR, perfil espectral, correlação, Higuchi,
localização de fonte) sai em arquivo nenhum.

O efeito prático: outra pessoa não consegue repetir uma análise a partir do
que o app entrega. No máximo repetir o SETUP — banco e tratamento — e mesmo
assim sem saber sobre qual janela de tempo cada número foi produzido.

A receita é o arquivo que o app exporta e o lote consome. É a peça que faz o
app deixar de ser uma tela e virar instrumento: o mesmo caminho de código,
com os mesmos parâmetros, sobre um sujeito ou sobre os 121.

POR QUE VERSÃO DE BIBLIOTECA ENTRA

Um resultado que depende de mne, numpy, scipy e sklearn sem dizer quais
versões o produziram não é reproduzível — é uma foto. E a divergência é
AVISO, nunca erro: recusar rodar uma receita antiga travaria a reanálise de
um resultado antigo, que é justamente o que este arquivo existe para
permitir. Mas também não pode ser silêncio, porque o número pode mudar.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

# Versão do FORMATO deste arquivo, não do app.
#
# Sobe quando um campo muda de significado — nunca quando um campo é
# acrescentado. Ler uma receita de formato MAIOR que este é o único caso em
# que recusar é mais honesto que tentar: os campos que este código conhece
# podem ter passado a significar outra coisa, e "quase certo" num parâmetro
# de análise é pior que parar.
FORMATO = 1

_OBRIGATORIOS = ("formato", "banco", "sujeitos", "etapas")


def _versoes():
    """As versões que importam para o número sair igual.

    Import local: quem só quer LER uma receita não deveria precisar ter
    sklearn instalado."""
    saida = {}
    for nome, modulo in (("numpy", "numpy"), ("scipy", "scipy"),
                         ("sklearn", "sklearn"), ("mne", "mne"),
                         ("pandas", "pandas")):
        try:
            saida[nome] = __import__(modulo).__version__
        except Exception:
            saida[nome] = None   # ausente é informação; não é o mesmo que "não olhei"
    return saida


def montar(banco, sujeitos, etapas, notas=None):
    """A receita como dicionário serializável.

    `etapas` é um dicionário ORDENADO de nome-da-etapa para o `decisoes`
    daquela etapa. A ordem é informação: filtrar depois de epocar não é a
    mesma coisa que epocar depois de filtrar, e o dicionário do Python
    preserva a ordem de inserção desde a 3.7."""
    return {
        "formato": FORMATO,
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "banco": banco,
        "sujeitos": list(sujeitos),
        "etapas": dict(etapas),
        "versoes": _versoes(),
        "notas": notas,
    }


def salvar(receita, caminho):
    """Grava em JSON legível por humano.

    `indent` e `ensure_ascii=False` não são estética: alguém vai abrir isto
    num editor para conferir um parâmetro, e as decisões do preproc trazem
    texto em português. JSON compactado com escape unicode seria ilegível
    justamente na parte que explica o que foi feito."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(receita, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return caminho


def carregar(caminho):
    """Lê e confere o mínimo. Recusa formato futuro e campo faltando."""
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))

    faltando = [c for c in _OBRIGATORIOS if c not in dados]
    if faltando:
        raise ValueError(
            f"receita incompleta em {caminho}: falta(m) {', '.join(faltando)}"
        )

    if int(dados["formato"]) > FORMATO:
        raise ValueError(
            f"receita em formato {dados['formato']}, e este código lê até o "
            f"{FORMATO}. Um campo conhecido pode ter mudado de significado; "
            f"atualize o código em vez de adivinhar."
        )
    return dados


def conferir_versoes(receita):
    """Lista de avisos, uma linha por biblioteca que mudou de versão.

    Devolve lista vazia quando tudo bate — e nunca levanta. A decisão de
    seguir ou parar é de quem chama; o dever daqui é não deixar a divergência
    passar em silêncio."""
    agora = _versoes()
    avisos = []
    for lib, gravada in (receita.get("versoes") or {}).items():
        atual = agora.get(lib)
        if gravada is None or atual is None:
            continue
        if gravada != atual:
            avisos.append(
                f"{lib}: a receita foi gerada com {gravada}, esta máquina tem "
                f"{atual} — o número pode não sair idêntico"
            )
    return avisos
