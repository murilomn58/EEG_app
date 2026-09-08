# -*- coding: utf-8 -*-
"""Trava o classificador SVM e o protocolo de avaliação.

O QUE ESTE MÓDULO ESTÁ DEFENDENDO

Um classificador de EEG com 121 sujeitos tem três formas de mentir, e as três
produzem número publicável:

1. Vazamento de sujeito — épocas da mesma criança nos dois lados da partição.
2. Vazamento de normalização — média e desvio calculados com o teste dentro.
3. Ausência de nulo — uma AUC de 0,62 que ninguém comparou com o acaso.

Os testes aqui existem para que as três falhem ruidosamente em vez de virarem
resultado.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import classificador


def test_agrega_pela_mediana_das_epocas():
    """Um sujeito com épocas [1, 2, 9] vale 2, não 4.

    A média daria 4, arrastada pelo 9. Num sinal real o 9 é a época em que a
    criança se mexeu, e ela não deve decidir o diagnóstico dela."""
    escores = np.array([1.0, 2.0, 9.0, 10.0, 20.0])
    grupos = np.array([0, 0, 0, 1, 1])
    rotulos = np.array([0, 0, 0, 1, 1])

    esc, rot, ids = classificador.agregar_por_sujeito(escores, grupos, rotulos)

    assert list(ids) == [0, 1]
    assert esc[0] == pytest.approx(2.0)
    assert esc[1] == pytest.approx(15.0)
    assert list(rot) == [0, 1]


def test_rotulo_inconsistente_dentro_do_sujeito_levanta():
    """Um sujeito com dois rótulos é dado corrompido, não caso de borda.

    Devolver o primeiro rótulo, ou o mais frequente, esconderia um defeito de
    montagem do conjunto atrás de um número plausível."""
    escores = np.array([1.0, 2.0])
    grupos = np.array([0, 0])
    rotulos = np.array([0, 1])

    with pytest.raises(ValueError, match="rótulo"):
        classificador.agregar_por_sujeito(escores, grupos, rotulos)


def conjunto_separavel(n_sujeitos=8, n_epocas=6, semente=0):
    """Sujeitos de classe 1 com média deslocada: separável, mas com ruído.

    Deslocamento pequeno de propósito. Um conjunto perfeitamente separável
    daria AUC 1,0 em qualquer implementação, inclusive numa que vaza — e o
    teste deixaria de distinguir o certo do errado."""
    rng = np.random.default_rng(semente)
    X, y, g = [], [], []
    for s in range(n_sujeitos):
        classe = s % 2
        X.append(rng.normal(loc=classe * 1.2, scale=1.0, size=(n_epocas, 4)))
        y.extend([classe] * n_epocas)
        g.extend([s] * n_epocas)
    return np.vstack(X), np.array(y), np.array(g)


def test_loso_produz_um_escore_por_sujeito():
    """121 sujeitos dariam 121 escores; aqui são 8.

    A AUC sai de UMA conta sobre todos os escores, e não da média de contas por
    dobra: no LOSO cada dobra tem um sujeito só, e AUC de uma classe só é
    indefinida."""
    X, y, g = conjunto_separavel()

    r = classificador.avaliar_loso(X, y, g, n_dobras_internas=2)

    assert r["n_sujeitos"] == 8
    assert len(r["escores_por_sujeito"]) == 8
    assert len(r["rotulos_por_sujeito"]) == 8
    assert 0.0 <= r["auc"] <= 1.0


def test_loso_nao_vaza_sujeito():
    """O sujeito de teste nunca aparece no treino da sua dobra.

    Redundante com o LeaveOneGroupOut, e deliberado: a garantia fica afirmada no
    ponto de uso, e não depende de a biblioteca continuar se comportando assim
    numa versão futura."""
    X, y, g = conjunto_separavel()

    r = classificador.avaliar_loso(X, y, g, n_dobras_internas=2)

    for dobra in r["decisoes"]["dobras"]:
        assert dobra["sujeito_de_teste"] not in dobra["sujeitos_de_treino"]


def test_paralelizar_nao_muda_o_resultado():
    """Os mesmos escores com n_jobs=1 e n_jobs=-1.

    As 121 dobras externas do LOSO são independentes por construção: cada uma
    treina sobre um subconjunto diferente e não compartilha estado com as
    outras. Distribuí-las entre núcleos é, portanto, uma mudança de escalonamento
    e não de método — e este teste é o que sustenta essa afirmação. Se algum dia
    alguém introduzir estado compartilhado entre dobras, é aqui que aparece."""
    X, y, g = conjunto_separavel(n_sujeitos=8)

    serial = classificador.avaliar_loso(X, y, g, n_dobras_internas=2, n_jobs=1)
    paralelo = classificador.avaliar_loso(X, y, g, n_dobras_internas=2, n_jobs=-1)

    np.testing.assert_allclose(
        serial["escores_por_sujeito"], paralelo["escores_por_sujeito"], rtol=1e-10,
        err_msg="os escores mudaram ao paralelizar: alguma dobra compartilha estado",
    )
    assert serial["auc"] == paralelo["auc"]
    assert [d["sujeito_de_teste"] for d in serial["decisoes"]["dobras"]] == \
           [d["sujeito_de_teste"] for d in paralelo["decisoes"]["dobras"]], \
        "a ordem das dobras mudou ao paralelizar"


def test_normalizacao_ajustada_fora_da_dobra_mudaria_o_escore():
    """O escore honesto difere do escore com normalização vazada.

    A COMPARAÇÃO É ENTRE DOIS AJUSTES DOS MESMOS DADOS, e não entre dois
    conjuntos de dados. Um teste que alterasse o dado de um sujeito e exigisse
    que os escores dos outros não mudassem seria inválido: nas dobras dos
    outros, aquele sujeito é dado de TREINO legítimo, e mudar o treino muda o
    modelo. Medido em 08/09/2026: o escore de um sujeito ia de -0,82 para
    +1,00 quando outro sujeito era multiplicado por 1000, com o pipeline
    correto.

    O que este teste faz é diferente: os dados são os MESMOS nos dois lados, e
    só muda DE ONDE saem a média e o desvio. Se `avaliar_loso` ajustasse a
    escala no conjunto inteiro, o resultado dele bateria com o braço vazado."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X, y, g = conjunto_separavel(n_sujeitos=8)
    X[g == 0] *= 50.0     # o sujeito de teste tem escala própria

    grade_fixa = {"svm__C": [1.0], "svm__gamma": ["scale"]}
    r = classificador.avaliar_loso(
        X, y, g, grade=grade_fixa, n_dobras_internas=2
    )

    # O braço VAZADO, construído de propósito: escala ajustada com o teste dentro.
    treino, teste = g != 0, g == 0
    escalador = StandardScaler().fit(X)
    Xn = escalador.transform(X)
    modelo = SVC(kernel="rbf", C=1.0, gamma="scale")
    modelo.fit(Xn[treino], y[treino])
    escore_vazado = float(np.median(modelo.decision_function(Xn[teste])))

    escore_honesto = float(r["escores_por_sujeito"][0])
    assert not np.isclose(escore_honesto, escore_vazado, rtol=1e-6), (
        "o escore do LOSO bateu com o de uma normalização ajustada no conjunto "
        "inteiro: a escala está sendo ajustada fora da dobra"
    )


def test_permutacao_mantem_o_rotulo_constante_dentro_do_sujeito():
    """Permutar por época destruiria a estrutura de sujeito.

    Esse é o defeito silencioso do nulo: com rótulo embaralhado por época, cada
    sujeito passa a ter as duas classes, o classificador não consegue aprender
    nada e a AUC nula despenca — fazendo QUALQUER AUC observada parecer
    significativa."""
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])
    g = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    yp = classificador.permutar_rotulos_por_sujeito(y, g, semente=3)

    for sid in np.unique(g):
        assert len(np.unique(yp[g == sid])) == 1, "sujeito ficou com dois rótulos"
    assert sorted(yp.tolist()) == sorted(y.tolist()), "a contagem de classes mudou"


def test_nulo_por_permutacao_fica_bem_abaixo_do_sinal():
    """A AUC nula fica bem abaixo da AUC do sinal separável.

    NÃO se afirma que a nula ronda 0,5, porque no LOSO ela NÃO é centrada em
    0,5. Medido em 08/09/2026: média 0,086 com 8 sujeitos e 0,348 com 16.

    A razão é estrutural e não é defeito. Ao tirar o sujeito de teste do
    treino, o treino fica desbalanceado CONTRA a classe dele — sempre, por
    construção. O modelo aprende a maioria e erra justamente nele.

    Correção de 08/09/2026: uma redação anterior deste docstring media a
    correlação entre o desbalanceio do treino e o rótulo do sujeito de teste
    (−0,071 com 8 sujeitos, −0,033 com 16, −0,013 com 40, −0,004 com 121, uma
    quantidade no espaço dos rótulos) e concluía dali que o viés da AUC era
    desprezível em n=121. Estava errado: a correlação por dobra escala com
    1/n; o viés na AUC não, porque atua no mesmo sentido em todas as n dobras
    e a agregação o soma em vez de cancelá-lo. Medido sob H0 puro: AUC nula de
    0,000 (n=8), 0,010 (n=16), 0,179 (n=40), 0,384 (n=121) e 0,451 (n=300).
    Confirmado com o pipeline real sob ruído puro em n=121: AUC nula entre
    0,311 e 0,364 conforme o cenário. O viés é grande, negativo, e não
    desaparece com n.

    Isto é a razão de o nulo existir, e não um problema dele: o p-valor compara
    a AUC observada com a distribuição nula MEDIDA. Comparar com 0,5 teórico é
    que seria errado.

    O teto de 0,75 continua valendo e é o que pega vazamento: se o pipeline
    vazasse, a AUC nula subiria, porque o modelo reconheceria o sujeito em vez
    do rótulo. Não há piso de propósito: a nula do LOSO pode legitimamente
    chegar perto de 0,0 — foi medido 0,0 exato em n=121 com 30 épocas por
    sujeito."""
    X, y, g = conjunto_separavel(n_sujeitos=8)

    r = classificador.nulo_por_permutacao(
        X, y, g, n_permutacoes=8, n_dobras_internas=2, semente=0
    )

    assert len(r["aucs_nulas"]) == 8
    assert float(np.mean(r["aucs_nulas"])) < 0.75, (
        "a AUC nula subiu: com rótulos permutados o modelo não deveria "
        "distinguir nada, e uma nula alta indica vazamento"
    )
    assert r["media_nula"] < r["auc_observada"], (
        "a AUC observada não superou a nula: o sinal separável do conjunto "
        "sintético deveria bater a permutação"
    )
    assert 0.0 <= r["p_empirico"] <= 1.0


def test_progresso_nao_vaza_para_avaliar_loso():
    """`progresso` e `checkpoint` são parâmetros nomeados de `nulo_por_permutacao`,
    extraídos ANTES de repassar `**kw` para `avaliar_loso`.

    `avaliar_loso` não conhece esses dois nomes. Se vazarem dentro do `**kw`
    repassado nas chamadas internas, o resultado é um `TypeError` em produção,
    que só aparece no dia em que alguém finalmente passa `progresso=` — ou
    seja, no pior momento possível: no meio de uma rodada de horas."""
    X, y, g = conjunto_separavel(n_sujeitos=8)

    r = classificador.nulo_por_permutacao(
        X, y, g, n_permutacoes=2, n_dobras_internas=2, semente=0,
        progresso=lambda t: None,
    )

    assert 0.0 <= r["p_empirico"] <= 1.0


def test_retomada_reproduz_a_rodada_continua(tmp_path):
    """Retomar do índice k e continuar até n produz as MESMAS n permutações
    que uma execução contínua, porque cada permutação usa
    `semente=semente + i + 1` — função pura do índice, sem estado
    compartilhado entre iterações. Se essa propriedade for quebrada (por
    exemplo trocando por um `np.random.default_rng` único reaproveitado ao
    longo do laço), a retomada deixa de bater com a rodada contínua, e
    ninguém percebe até comparar as duas."""
    X, y, g = conjunto_separavel(n_sujeitos=8)

    continua = classificador.nulo_por_permutacao(
        X, y, g, n_permutacoes=4, n_dobras_internas=2, semente=0,
    )

    caminho = tmp_path / "x.json"
    classificador.nulo_por_permutacao(
        X, y, g, n_permutacoes=2, n_dobras_internas=2, semente=0,
        checkpoint=caminho,
    )
    retomada = classificador.nulo_por_permutacao(
        X, y, g, n_permutacoes=4, n_dobras_internas=2, semente=0,
        checkpoint=caminho, retomar=True,
    )

    np.testing.assert_allclose(
        retomada["aucs_nulas"], continua["aucs_nulas"], atol=1e-12
    )


def test_retomada_recusa_semente_diferente(tmp_path):
    """Misturar dois nulos de sementes diferentes produz uma distribuição sem
    procedência. Recusar é a única opção correta: não há como "aproveitar
    parcialmente" um checkpoint cuja semente diverge da chamada atual."""
    X, y, g = conjunto_separavel(n_sujeitos=8)
    caminho = tmp_path / "x.json"

    classificador.nulo_por_permutacao(
        X, y, g, n_permutacoes=2, n_dobras_internas=2, semente=0,
        checkpoint=caminho,
    )

    with pytest.raises(ValueError):
        classificador.nulo_por_permutacao(
            X, y, g, n_permutacoes=4, n_dobras_internas=2, semente=7,
            checkpoint=caminho, retomar=True,
        )


def test_progresso_do_nulo_conta_as_permutacoes():
    """Cada permutação concluída emite uma linha numerada, para que uma rodada
    de horas mostre progresso em vez de silêncio total."""
    X, y, g = conjunto_separavel(n_sujeitos=8)
    linhas = []

    classificador.nulo_por_permutacao(
        X, y, g, n_permutacoes=3, n_dobras_internas=2, semente=0,
        progresso=linhas.append,
    )

    for i in range(1, 4):
        assert any(f"{i}/3" in linha for linha in linhas), (
            f"nenhuma linha de progresso menciona a permutação {i}/3"
        )


def test_nulo_recusa_zero_permutacoes():
    """n_permutacoes=0 produziria uma média sobre lista vazia — nan — em vez de
    um erro claro. `--sem-nulo` é o caminho certo para pular a condição D."""
    X, y, g = conjunto_separavel(n_sujeitos=8)

    with pytest.raises(ValueError):
        classificador.nulo_por_permutacao(
            X, y, g, n_permutacoes=0, n_dobras_internas=2, semente=0,
        )
