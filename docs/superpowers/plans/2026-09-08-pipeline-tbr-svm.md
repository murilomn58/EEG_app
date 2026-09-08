# Pipeline TBR + SVM — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classificar TDAH × controle no adhdata usando razão theta/beta com `sklearn.svm.SVC`, sob validação cruzada aninhada com LOSO externo, e exportar as features num CSV que o projeto `eeg_transformer` consome.

**Architecture:** O backend do app já tem extração de features, split por sujeito e normalização não-vazante, todos testados. Este plano acrescenta dois módulos (`classificador.py` e `exportar_features.py`) que compõem o que existe, mais um notebook que narra sem implementar. A lógica vive em módulo testado; o notebook importa.

**Tech Stack:** Python 3, scikit-learn 1.9.0, numpy 2.4.6, scipy 1.17.1, MNE 1.12.1, pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-pipeline-tbr-svm-design.md`

## Global Constraints

- **Diretório de trabalho:** `C:\3D brain\VERSAO GRATUITA MINHA`. O interpretador é `backend/.venv/Scripts/python.exe`.
- **Idioma do código:** nomes de função, variáveis, docstrings e mensagens de erro em **português**, como todo o resto de `backend/scripts/`.
- **Docstring explica o porquê, não o quê.** O padrão do projeto é documentar a decisão e o defeito que ela evita. Ver `caracteristicas.py` e `normalizacao.py` como referência de tom.
- **Nenhum arquivo existente é alterado.** Os módulos novos compõem os antigos. Se parecer necessário alterar um existente, pare e pergunte.
- **Zero regressão:** os 421 testes atuais do backend continuam passando ao fim de cada tarefa.
- **Sem `fit_transform` fora de `Pipeline`.** O projeto tem teste que falha se alguém acrescentar `fit_transform` a `normalizacao.py`. Dentro de um `Pipeline` do sklearn é permitido, porque o `Pipeline` garante o ajuste por dobra.
- **`SVC(probability=False)`**, sempre. A AUC sai de `decision_function()`.
- **A ressalva de licença do adhdata viaja com todo artefato exportado:** `"adhdata: a licença deste banco não pôde ser confirmada; ausência de licença não é licença permissiva"`.
- **Import path dos testes:** o padrão do projeto é o preâmbulo abaixo, copiado de `backend/tests/test_split.py`:
  ```python
  import sys
  from pathlib import Path
  RAIZ = Path(__file__).resolve().parent.parent
  sys.path.insert(0, str(RAIZ))
  sys.path.insert(0, str(RAIZ / "scripts"))
  ```
- **Comando de teste:** `cd backend && .venv/Scripts/python.exe -m pytest tests/<arquivo> -v`

---

## Estrutura de arquivos

| Arquivo | Responsabilidade |
|---|---|
| `backend/scripts/classificador.py` (criar) | Agregação por sujeito, nested CV com LOSO, nulo por permutação. Funções puras sobre arrays; não lê arquivo, não escreve arquivo. |
| `backend/tests/test_classificador.py` (criar) | Testes do acima, com dados sintéticos. |
| `backend/scripts/exportar_features.py` (criar) | Monta o conjunto a partir do adhdata e grava CSV + receita JSON. É a única parte que toca disco. |
| `backend/tests/test_exportar_features.py` (criar) | Testes do contrato do CSV. |
| `backend/scripts/experimento_svm.py` (criar) | CLI que roda as 4 condições e grava o CSV de resultados. |
| `backend/tests/test_experimento_svm.py` (criar) | Testes do experimento com `--max-sujeitos` pequeno. |
| `C:\dev\eeg_transformer\notebooks\01_tbr_svm_adhdata.ipynb` (criar) | Narra e importa. Nenhuma lógica. |

A separação entre `classificador.py` (puro) e `exportar_features.py` (toca disco) é deliberada: o primeiro é testável com arrays de 6 sujeitos sintéticos em milissegundos; o segundo precisa do CSV de 267 MB.

---

## Task 1: Agregação de escores por sujeito

**Files:**
- Create: `backend/scripts/classificador.py`
- Test: `backend/tests/test_classificador.py`

**Interfaces:**
- Consumes: nada de tarefas anteriores.
- Produces: `agregar_por_sujeito(escores, grupos, rotulos) -> (escores_sujeito, rotulos_sujeito, ids_sujeito)`, todos `np.ndarray` 1-D de mesmo comprimento, ordenados por `np.unique(grupos)`.

**Contexto para quem implementa:** o LOSO testa um sujeito por dobra, mas esse sujeito tem várias épocas. O modelo produz um escore por época. Para virar um ponto na curva ROC, as épocas de um sujeito precisam virar um número. Usamos a **mediana**, porque o número de épocas varia 5,4× entre sujeitos e a mediana resiste a uma época contaminada por artefato.

- [ ] **Step 1: Escrever o teste que falha**

Criar `backend/tests/test_classificador.py`:

```python
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
```

- [ ] **Step 2: Rodar o teste e ver falhar**

```
cd backend && .venv/Scripts/python.exe -m pytest tests/test_classificador.py -v
```
Esperado: `ModuleNotFoundError: No module named 'classificador'`

- [ ] **Step 3: Implementação mínima**

Criar `backend/scripts/classificador.py`:

```python
# -*- coding: utf-8 -*-
"""Classificação com SVM sob validação cruzada aninhada por sujeito.

POR QUE A AGREGAÇÃO POR SUJEITO EXISTE

O classificador prediz por época, e a pergunta clínica é por criança. Entre a
época e a criança há uma escolha que muda o resultado e costuma ficar
implícita: como as N épocas de um sujeito viram um número.

Aqui é a MEDIANA. O adhdata tem sujeitos com 62 s e sujeitos com 338 s de
gravação — uma razão de 5,4 —, e a média deixaria uma única época contaminada
por artefato deslocar o escore de um sujeito inteiro. A mediana não.
"""
import numpy as np


def agregar_por_sujeito(escores, grupos, rotulos):
    """(escores_sujeito, rotulos_sujeito, ids_sujeito), na ordem de `np.unique`.

    Recusa sujeito com mais de um rótulo em vez de escolher um. Um sujeito com
    dois rótulos significa que a montagem do conjunto está errada, e devolver o
    primeiro esconderia o defeito atrás de um número plausível."""
    escores = np.asarray(escores, dtype=float)
    grupos = np.asarray(grupos)
    rotulos = np.asarray(rotulos)

    if not (len(escores) == len(grupos) == len(rotulos)):
        raise ValueError(
            f"escores ({len(escores)}), grupos ({len(grupos)}) e rotulos "
            f"({len(rotulos)}) têm de ter o mesmo comprimento"
        )

    ids = np.unique(grupos)
    esc_s = np.empty(len(ids), dtype=float)
    rot_s = np.empty(len(ids), dtype=rotulos.dtype)

    for i, sid in enumerate(ids):
        dentro = grupos == sid
        rot_unicos = np.unique(rotulos[dentro])
        if len(rot_unicos) > 1:
            raise ValueError(
                f"sujeito {sid!r} tem mais de um rótulo ({rot_unicos.tolist()}): "
                f"o conjunto foi montado errado"
            )
        esc_s[i] = float(np.median(escores[dentro]))
        rot_s[i] = rot_unicos[0]

    return esc_s, rot_s, ids
```

- [ ] **Step 4: Rodar e ver passar**

```
cd backend && .venv/Scripts/python.exe -m pytest tests/test_classificador.py -v
```
Esperado: 2 passed

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/classificador.py backend/tests/test_classificador.py
git commit -m "Agrega escores de epoca em escore de sujeito pela mediana"
```

---

## Task 2: Nested CV com LOSO externo

**Files:**
- Modify: `backend/scripts/classificador.py`
- Test: `backend/tests/test_classificador.py`

**Interfaces:**
- Consumes: `agregar_por_sujeito` da Task 1.
- Produces: `avaliar_loso(X, y, grupos, estimador=None, grade=None, n_dobras_internas=5, semente=0) -> dict` com as chaves `auc`, `acuracia`, `n_sujeitos`, `escores_por_sujeito`, `rotulos_por_sujeito`, `ids_sujeito`, `hiperparametros_por_dobra`, `decisoes`.

**Contexto crítico, verificado no ambiente em 08/09/2026:** cada dobra do LOSO testa **um único sujeito**, então a dobra tem uma classe só e `roc_auc_score` é indefinida nela. **Não existe "AUC média das dobras" aqui.** O desenho correto acumula um escore por sujeito ao longo das 121 dobras e calcula a AUC **uma vez** no fim.

- [ ] **Step 1: Escrever os testes que falham**

Acrescentar a `backend/tests/test_classificador.py`:

```python
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
```

- [ ] **Step 2: Rodar e ver falhar**

```
cd backend && .venv/Scripts/python.exe -m pytest tests/test_classificador.py -v
```
Esperado: `AttributeError: module 'classificador' has no attribute 'avaliar_loso'`

- [ ] **Step 3: Implementar**

Acrescentar a `backend/scripts/classificador.py`, logo após os imports:

```python
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import (GridSearchCV, LeaveOneGroupOut,
                                     StratifiedGroupKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# A grade do interno. Pequena de propósito: cada valor a mais multiplica 121
# dobras externas por 5 internas, e uma grade generosa aqui compra decimais de
# AUC ao custo de horas.
GRADE_SVM = {
    "svm__C": [0.1, 1.0, 10.0, 100.0],
    "svm__gamma": ["scale", 0.01, 0.1],
}


def estimador_svm():
    """SVM RBF com a escala DENTRO do pipeline.

    A escala tem de ser um passo do Pipeline, e não uma transformação aplicada
    antes: é isso que faz o sklearn reajustá-la a cada dobra, em vez de usar uma
    média calculada com o sujeito de teste dentro.

    `probability=False` porque a AUC sai de `decision_function`. Ligar
    `probability` acrescentaria um Platt scaling com CV próprio, multiplicando o
    custo por cerca de cinco para produzir a mesma ordenação."""
    return Pipeline([
        ("escala", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=False)),
    ])
```

E a função principal:

```python
def avaliar_loso(X, y, grupos, estimador=None, grade=None,
                 n_dobras_internas=5, semente=0):
    """Nested CV: LOSO externo, StratifiedGroupKFold interno para os hiperparâmetros.

    POR QUE A AUC SAI DE UMA CONTA SÓ, NO FIM

    Cada dobra externa testa UM sujeito. Uma dobra com um sujeito tem uma classe
    só, e a AUC de uma classe só é indefinida — o sklearn avisa e devolve NaN.
    Uma 'AUC média das dobras' aqui seria a média de 121 NaN.

    O que se faz: acumular o escore de cada sujeito ao longo das dobras e
    calcular a AUC uma vez, sobre o vetor de escores e o vetor de rótulos.

    O escore de cada época sai de `decision_function`, que é a distância
    assinada ao hiperplano. Não é probabilidade e não precisa ser: a AUC depende
    só da ordenação."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    grupos = np.asarray(grupos)

    if len(np.unique(grupos)) < 2:
        raise ValueError(
            f"LOSO precisa de pelo menos 2 sujeitos, há {len(np.unique(grupos))}"
        )

    estimador = estimador if estimador is not None else estimador_svm()
    grade = grade if grade is not None else GRADE_SVM

    escores = np.empty(len(y), dtype=float)
    dobras = []

    for treino, teste in LeaveOneGroupOut().split(X, y, groups=grupos):
        g_treino = grupos[treino]
        sujeito_teste = np.unique(grupos[teste])[0]

        # O interno nunca vê o sujeito de teste: ele parte de `treino`.
        #
        # StratifiedGroupKFold, e NÃO GroupKFold. Medido em 08/09/2026: o
        # GroupKFold respeita a fronteira de sujeito mas ignora a classe, e com
        # poucos sujeitos ele produz dobra interna com UMA CLASSE SÓ — o SVC
        # levanta "The number of classes has to be greater than one; got 1
        # class" e o experimento inteiro morre no meio. O estratificado respeita
        # as duas restrições ao mesmo tempo.
        n_int = min(n_dobras_internas, len(np.unique(g_treino)))
        busca = GridSearchCV(
            estimador, grade,
            cv=StratifiedGroupKFold(n_splits=n_int, shuffle=True,
                                    random_state=semente),
            scoring="roc_auc",
            n_jobs=1,
        )
        busca.fit(X[treino], y[treino], groups=g_treino)
        escores[teste] = busca.best_estimator_.decision_function(X[teste])

        dobras.append({
            "sujeito_de_teste": sujeito_teste.item()
            if hasattr(sujeito_teste, "item") else sujeito_teste,
            "sujeitos_de_treino": [
                s.item() if hasattr(s, "item") else s
                for s in np.unique(g_treino)
            ],
            "melhores_parametros": busca.best_params_,
            "n_dobras_internas": int(n_int),
        })

    esc_s, rot_s, ids = agregar_por_sujeito(escores, grupos, y)

    if len(np.unique(rot_s)) < 2:
        raise ValueError(
            "todos os sujeitos têm o mesmo rótulo: a AUC não é definida"
        )

    auc = float(roc_auc_score(rot_s, esc_s))
    predito = (esc_s > 0).astype(int)

    return {
        "auc": auc,
        "acuracia": float(accuracy_score(rot_s, predito)),
        "n_sujeitos": int(len(ids)),
        "escores_por_sujeito": esc_s,
        "rotulos_por_sujeito": rot_s,
        "ids_sujeito": ids,
        "hiperparametros_por_dobra": [d["melhores_parametros"] for d in dobras],
        "decisoes": {
            "externo": "LeaveOneGroupOut (LOSO)",
            "interno": f"StratifiedGroupKFold({n_dobras_internas}) com scoring roc_auc",
            "agregacao": "mediana dos escores das épocas do sujeito",
            "escore": "decision_function (distância assinada ao hiperplano)",
            "auc": "calculada UMA vez sobre os escores de sujeito; a AUC por "
                   "dobra é indefinida no LOSO porque a dobra tem uma classe só",
            "grade": {k: list(v) for k, v in grade.items()},
            "semente": int(semente),
            "n_epocas": int(len(y)),
            "dobras": dobras,
        },
    }
```

- [ ] **Step 4: Rodar e ver passar**

```
cd backend && .venv/Scripts/python.exe -m pytest tests/test_classificador.py -v
```
Esperado: 5 passed

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/classificador.py backend/tests/test_classificador.py
git commit -m "Avaliacao LOSO aninhada com SVM e AUC sobre escores de sujeito"
```

---

## Task 3: Nulo por permutação

**Files:**
- Modify: `backend/scripts/classificador.py`
- Test: `backend/tests/test_classificador.py`

**Interfaces:**
- Consumes: `avaliar_loso` da Task 2.
- Produces: `permutar_rotulos_por_sujeito(y, grupos, semente) -> np.ndarray` e `nulo_por_permutacao(X, y, grupos, n_permutacoes=100, semente=0, **kw) -> dict` com `auc_observada`, `aucs_nulas`, `media_nula`, `desvio_nulo`, `p_empirico`, `n_permutacoes`, `decisoes`.

**Contexto:** com 121 sujeitos, uma AUC de 0,62 pode ser sinal fraco ou ruído. O nulo dá a distribuição sob a hipótese nula. **A permutação é do rótulo de cada sujeito, não de cada época** — permutar por época quebraria a estrutura de sujeito e produziria um nulo otimista demais, fazendo qualquer AUC parecer significativa.

- [ ] **Step 1: Escrever os testes que falham**

Acrescentar a `backend/tests/test_classificador.py`:

```python
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
    construção. O modelo aprende a maioria e erra justamente nele. A
    correlação entre o desbalanceio do treino e o rótulo do sujeito de teste
    foi medida em 2000 sorteios por tamanho: −0,071 com 8 sujeitos, −0,033 com
    16, −0,013 com 40 e −0,004 com 121. Escala com 1/n, que é a assinatura da
    origem combinatória.

    Isto é a razão de o nulo existir, e não um problema dele: o p-valor compara
    a AUC observada com a distribuição nula MEDIDA. Comparar com 0,5 teórico é
    que seria errado.

    O teto de 0,75 continua valendo e é o que pega vazamento: se o pipeline
    vazasse, a AUC nula subiria, porque o modelo reconheceria o sujeito em vez
    do rótulo."""
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
```

- [ ] **Step 2: Rodar e ver falhar**

```
cd backend && .venv/Scripts/python.exe -m pytest tests/test_classificador.py -v
```
Esperado: `AttributeError: ... has no attribute 'permutar_rotulos_por_sujeito'`

- [ ] **Step 3: Implementar**

Acrescentar a `backend/scripts/classificador.py`:

```python
def permutar_rotulos_por_sujeito(y, grupos, semente=0):
    """Embaralha os rótulos ENTRE sujeitos, mantendo-os constantes DENTRO.

    A distinção é o nulo inteiro. Permutar por época daria a cada sujeito as
    duas classes; nenhum classificador aprenderia nada, a AUC nula cairia bem
    abaixo de 0,5 e qualquer resultado observado passaria por significativo.

    Permutando por sujeito, o nulo preserva tudo o que não é o rótulo — o
    desequilíbrio de épocas, a estrutura de correlação dentro do sujeito — e
    mede exatamente o que se quer medir."""
    y = np.asarray(y)
    grupos = np.asarray(grupos)

    ids = np.unique(grupos)
    rot_por_sujeito = np.array([np.unique(y[grupos == s])[0] for s in ids])

    embaralhados = rot_por_sujeito.copy()
    np.random.default_rng(semente).shuffle(embaralhados)

    saida = np.empty_like(y)
    for sid, rot in zip(ids, embaralhados):
        saida[grupos == sid] = rot
    return saida


def nulo_por_permutacao(X, y, grupos, n_permutacoes=100, semente=0, **kw):
    """AUC observada, distribuição sob o nulo e p-valor empírico.

    O p-valor usa a correção de Phipson e Smyth (2010), `(b + 1) / (m + 1)`:
    com m permutações, um p-valor de zero é impossível de sustentar, e o
    estimador ingênuo `b / m` produz exatamente isso quando nenhuma permutação
    supera a observação."""
    r_obs = avaliar_loso(X, y, grupos, semente=semente, **kw)
    auc_obs = r_obs["auc"]

    aucs = []
    for i in range(n_permutacoes):
        yp = permutar_rotulos_por_sujeito(y, grupos, semente=semente + i + 1)
        aucs.append(avaliar_loso(X, yp, grupos, semente=semente, **kw)["auc"])
    aucs = np.array(aucs, dtype=float)

    b = int(np.sum(aucs >= auc_obs))
    return {
        "auc_observada": auc_obs,
        "aucs_nulas": aucs,
        "media_nula": float(aucs.mean()),
        "desvio_nulo": float(aucs.std()),
        "p_empirico": float((b + 1) / (n_permutacoes + 1)),
        "n_permutacoes": int(n_permutacoes),
        "decisoes": {
            "permutacao": "rótulo por SUJEITO, nunca por época",
            "p_valor": "(b+1)/(m+1), Phipson e Smyth 2010",
            "avaliacao": r_obs["decisoes"],
        },
    }
```

- [ ] **Step 4: Rodar e ver passar**

```
cd backend && .venv/Scripts/python.exe -m pytest tests/test_classificador.py -v
```
Esperado: 7 passed

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/classificador.py backend/tests/test_classificador.py
git commit -m "Nulo por permutacao de rotulo por sujeito, com p-valor empirico"
```

---

## Task 4: Exportador de features

**Files:**
- Create: `backend/scripts/exportar_features.py`
- Test: `backend/tests/test_exportar_features.py`

**Interfaces:**
- Consumes: `caracteristicas.razao_theta_beta`, `caracteristicas.potencia_de_banda`, `receita.montar`, `receita.salvar`.
- Produces: `montar_conjunto_tbr(df, duracao_s=4.0, passo_s=4.0, max_sujeitos=None) -> (X, y, grupos, ids_originais, indices_epoca, nomes, decisoes)` — **sete** valores e `escrever_csv(caminho, X, y, ids_originais, indices_epoca, nomes) -> None`.

**Contexto:** este é o contrato entre o app e o `eeg_transformer`, hoje inexistente. O campo que importa é `sujeito_id`: tem que ser o ID **original** do adhdata (`v10p`), não o índice interno (`0`, `1`, `2`). Sem ele o transformer não consegue refazer a partição por sujeito e o CSV é inútil para o propósito que tem.

- [ ] **Step 1: Escrever os testes que falham**

Criar `backend/tests/test_exportar_features.py`:

```python
# -*- coding: utf-8 -*-
"""Trava o contrato do CSV de features entre o app e o eeg_transformer.

O QUE ESTE CONTRATO EXISTE PARA GARANTIR

O README do eeg_transformer diz que o app "prepara os dados que este projeto
consome", e nenhum formato tinha sido definido. Um CSV que pareça certo e não
traga a fronteira de sujeito é pior que nenhum: o outro projeto treina, publica
um número, e o vazamento só aparece quando alguém pergunta.

A coluna que carrega essa garantia é `sujeito_id`, e ela tem de trazer o ID
ORIGINAL do banco, não o índice interno da montagem.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import exportar_features


def test_csv_traz_o_id_original_do_sujeito(tmp_path):
    """`v10p`, e não `0`.

    O índice interno é o que o GroupKFold usa, e é local à montagem: rodar com
    --max-sujeitos muda o índice do mesmo sujeito. Um CSV com índice não permite
    juntar duas exportações nem rastrear um sujeito até o banco."""
    destino = tmp_path / "features.csv"
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    exportar_features.escrever_csv(
        destino, X,
        y=np.array(["ADHD", "Control"]),
        ids_originais=np.array(["v10p", "v20c"]),
        indices_epoca=np.array([0, 0]),
        nomes=["Cz_tbr", "Pz_tbr"],
    )

    linhas = list(csv.DictReader(destino.open(encoding="utf-8")))
    assert linhas[0]["sujeito_id"] == "v10p"
    assert linhas[1]["sujeito_id"] == "v20c"


def test_csv_tem_o_cabecalho_do_contrato(tmp_path):
    """As três colunas de identificação vêm primeiro, e nesta ordem."""
    destino = tmp_path / "features.csv"

    exportar_features.escrever_csv(
        destino, np.array([[1.0]]),
        y=np.array(["ADHD"]),
        ids_originais=np.array(["v10p"]),
        indices_epoca=np.array([0]),
        nomes=["Cz_tbr"],
    )

    cabecalho = destino.open(encoding="utf-8").readline().strip().split(",")
    assert cabecalho[:3] == ["sujeito_id", "rotulo", "epoca_idx"]
    assert cabecalho[3:] == ["Cz_tbr"]


def test_rotulo_sai_como_texto_e_nao_como_codigo(tmp_path):
    """`ADHD`, não `1`.

    Um CSV com 0 e 1 exige um dicionário externo para ser lido, e esse
    dicionário é justamente o que se perde entre dois projetos."""
    destino = tmp_path / "features.csv"

    exportar_features.escrever_csv(
        destino, np.array([[1.0], [2.0]]),
        y=np.array(["ADHD", "Control"]),
        ids_originais=np.array(["a", "b"]),
        indices_epoca=np.array([0, 0]),
        nomes=["Cz_tbr"],
    )

    linhas = list(csv.DictReader(destino.open(encoding="utf-8")))
    assert {l["rotulo"] for l in linhas} == {"ADHD", "Control"}


def test_recusa_comprimentos_incompativeis(tmp_path):
    """X com 2 linhas e 3 ids é defeito de montagem, não caso de borda."""
    with pytest.raises(ValueError, match="comprimento"):
        exportar_features.escrever_csv(
            tmp_path / "f.csv", np.array([[1.0], [2.0]]),
            y=np.array(["ADHD", "Control"]),
            ids_originais=np.array(["a", "b", "c"]),
            indices_epoca=np.array([0, 0]),
            nomes=["Cz_tbr"],
        )
```

- [ ] **Step 2: Rodar e ver falhar**

```
cd backend && .venv/Scripts/python.exe -m pytest tests/test_exportar_features.py -v
```
Esperado: `ModuleNotFoundError: No module named 'exportar_features'`

- [ ] **Step 3: Implementar**

Criar `backend/scripts/exportar_features.py`:

```python
# -*- coding: utf-8 -*-
"""Exporta as características do adhdata no contrato que o eeg_transformer lê.

O BURACO QUE ISTO FECHA

O README do `eeg_transformer` diz que o app "trata, particiona e prepara os
dados que este projeto consome". Nenhum formato tinha sido definido, e por isso
aquele projeto estava vazio: não havia o que consumir.

A COLUNA QUE IMPORTA

`sujeito_id` traz o ID ORIGINAL do banco (`v10p`), e não o índice interno da
montagem. O índice é local: rodar com `--max-sujeitos` muda o índice do mesmo
sujeito, e duas exportações ficam impossíveis de juntar. Mais grave, sem o ID
original o outro projeto não consegue refazer a partição por sujeito, e um CSV
que não permite partição honesta é um convite ao vazamento.

O rótulo sai como TEXTO (`ADHD` / `Control`) e não como 0/1, porque um código
numérico exige um dicionário externo para ser lido — e esse dicionário é
exatamente o que se perde na fronteira entre dois projetos.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import caracteristicas
import csv_data
import epocas as mod_epocas
import preproc_basico

COLUNAS_DE_IDENTIFICACAO = ["sujeito_id", "rotulo", "epoca_idx"]

RESSALVA_LICENCA = (
    "adhdata: a licença deste banco não pôde ser confirmada; ausência de "
    "licença não é licença permissiva"
)


def escrever_csv(caminho, X, y, ids_originais, indices_epoca, nomes):
    """Grava o CSV do contrato. Uma linha por época.

    Recusa comprimentos incompatíveis em vez de truncar pelo menor: truncar
    produziria um arquivo plausível com linhas faltando, e ninguém notaria."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    for rotulo, vetor in (("y", y), ("ids_originais", ids_originais),
                          ("indices_epoca", indices_epoca)):
        if len(vetor) != n:
            raise ValueError(
                f"X tem {n} linhas e {rotulo} tem {len(vetor)}: comprimento "
                f"incompatível"
            )
    if X.ndim != 2 or X.shape[1] != len(nomes):
        raise ValueError(
            f"X tem forma {X.shape} e há {len(nomes)} nomes de característica"
        )

    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(COLUNAS_DE_IDENTIFICACAO + list(nomes))
        for i in range(n):
            escritor.writerow(
                [ids_originais[i], y[i], int(indices_epoca[i])]
                + [f"{v:.10g}" for v in X[i]]
            )


def montar_conjunto_tbr(df, duracao_s=4.0, passo_s=4.0, max_sujeitos=None):
    """(X, y, grupos, ids_originais, indices_epoca, nomes, decisoes).

    `grupos` é o índice interno, que o GroupKFold consome; `ids_originais` é o
    ID do banco, que vai para o CSV. Os dois existem porque servem a coisas
    diferentes, e confundi-los é o defeito que o teste do contrato tranca.

    Sujeito que falha APARECE em `decisoes`, e não some da contagem: um
    denominador que encolhe em silêncio faz a média mentir."""
    sujeitos = csv_data.list_subjects(df)
    if max_sujeitos:
        # Amostra equilibrada, e nao os N primeiros: a lista do adhdata chega
        # ORDENADA POR CLASSE, e os N primeiros dariam uma classe so.
        por_classe = {}
        for s in sujeitos:
            por_classe.setdefault(s["classe"], []).append(s)
        metade = max(1, max_sujeitos // max(1, len(por_classe)))
        escolhidos = []
        for lista in por_classe.values():
            escolhidos.extend(lista[:metade])
        sujeitos = escolhidos[:max_sujeitos]

    blocos_X, blocos_y, blocos_g, blocos_id, blocos_ep = [], [], [], [], []
    nomes = None
    dec_epocas = dec_carac = dec_preproc = None
    falhas = []

    for idx, s in enumerate(sujeitos):
        sid = s["id"]
        try:
            raw = preproc_basico.raw_de_dataframe(df, sid)
            filtrado, dec_preproc = preproc_basico.preprocessar(
                raw, l_freq=0.5, h_freq=None
            )
            dados = filtrado.get_data() * 1e6

            janelas, dec_epocas = mod_epocas.epocar_janela_fixa(
                dados, csv_data.FS, duracao_s, passo_s
            )
            if len(janelas) == 0:
                falhas.append((sid, "nenhuma época coube na gravação"))
                continue

            X, nomes, dec_carac = caracteristicas.razao_theta_beta(
                janelas, csv_data.FS,
                nomes_canais=list(csv_data.CANAIS_19), com_decisoes=True,
            )
            blocos_X.append(X)
            blocos_y.append(np.full(len(X), s["classe"], dtype=object))
            blocos_g.append(np.full(len(X), idx))
            blocos_id.append(np.full(len(X), sid, dtype=object))
            blocos_ep.append(np.arange(len(X)))
        except Exception as e:            # noqa: BLE001 — sujeito que falha aparece
            falhas.append((sid, str(e)))

    if not blocos_X:
        raise RuntimeError("nenhum sujeito produziu época: nada a exportar")

    decisoes = {
        "preproc": dec_preproc,
        "epocas": dec_epocas,
        "caracteristicas": dec_carac,
        "n_sujeitos_pedidos": len(sujeitos),
        "n_sujeitos_usados": len(blocos_X),
        "sujeitos_que_falharam": [{"id": s, "motivo": m} for s, m in falhas],
        "ressalva_licenca": RESSALVA_LICENCA,
    }
    return (np.vstack(blocos_X), np.concatenate(blocos_y),
            np.concatenate(blocos_g), np.concatenate(blocos_id),
            np.concatenate(blocos_ep), nomes, decisoes)
```

- [ ] **Step 4: Rodar e ver passar**

```
cd backend && .venv/Scripts/python.exe -m pytest tests/test_exportar_features.py -v
```
Esperado: 4 passed

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/exportar_features.py backend/tests/test_exportar_features.py
git commit -m "Exporta TBR no contrato de CSV que o eeg_transformer consome"
```

---

## Task 5: CLI do experimento, com as quatro condições

**Files:**
- Create: `backend/scripts/experimento_svm.py`
- Test: `backend/tests/test_experimento_svm.py`

**Interfaces:**
- Consumes: `classificador.avaliar_loso`, `classificador.nulo_por_permutacao`, `exportar_features.montar_conjunto_tbr`, `caracteristicas.potencia_de_banda`, `receita.montar`.
- Produces: `rodar(duracao_s, passo_s, max_sujeitos, n_permutacoes, semente) -> (resultados, receita)`; `resultados` é lista de dicts com `condicao`, `features`, `modelo`, `auc`, `acuracia`, `n_sujeitos`.

**Contexto:** o spec define quatro condições. A **A** é o que a orientadora pediu; as outras três existem para que o número da A signifique alguma coisa. Sem a D (nulo), uma AUC de 0,62 é indistinguível de ruído.

- [ ] **Step 1: Escrever o teste que falha**

Criar `backend/tests/test_experimento_svm.py`:

```python
# -*- coding: utf-8 -*-
"""Trava as quatro condições do experimento de SVM.

POR QUE QUATRO E NÃO UMA

A condição A é a que foi pedida: SVM sobre TBR. Sozinha ela produz um número
que não se pode interpretar. B diz se a TBR está jogando informação fora; C diz
se o SVM ganha algo sobre a regressão logística que já existia; D diz se o
número bate o acaso.

O teste roda com poucos sujeitos e poucas permutações, porque o que ele trava é
a ESTRUTURA — que as quatro condições existem, que cada uma reporta AUC e n —
e não os valores, que dependem do banco inteiro.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import csv_data
import experimento_svm


@pytest.fixture
def df_sintetico():
    """Quatro sujeitos, 40 s cada, com theta deslocado num dos grupos.

    Sintético e não amostra do banco real: o CSV real tem 267 MB, e um teste que
    o lê deixa de ser teste e vira experimento."""
    rng = np.random.default_rng(0)
    fs = int(csv_data.FS)
    n = fs * 40
    t = np.arange(n) / fs
    linhas = []
    for i, (sid, classe) in enumerate(
        [("s1", "ADHD"), ("s2", "ADHD"), ("s3", "Control"), ("s4", "Control")]
    ):
        amp_theta = 3.0 if classe == "ADHD" else 1.0
        bloco = {}
        for c in csv_data.CANAIS_19:
            sinal = (amp_theta * np.sin(2 * np.pi * 6 * t)
                     + 1.0 * np.sin(2 * np.pi * 20 * t)
                     + rng.normal(0, 0.5, n))
            bloco[c] = sinal
        bloco["ID"] = sid
        bloco["Class"] = classe
        linhas.append(pd.DataFrame(bloco))
    return pd.concat(linhas, ignore_index=True)


def test_rodar_produz_as_quatro_condicoes(df_sintetico):
    resultados, receita = experimento_svm.rodar(
        df=df_sintetico, duracao_s=4.0, passo_s=4.0,
        n_permutacoes=3, n_dobras_internas=2, semente=0,
    )

    condicoes = {r["condicao"] for r in resultados}
    assert condicoes == {"A", "B", "C", "D"}
    for r in resultados:
        assert 0.0 <= r["auc"] <= 1.0
        assert r["n_sujeitos"] == 4
    assert receita["banco"] == "adhdata"
    assert "licença" in receita["notas"]


def test_a_condicao_d_reporta_p_empirico(df_sintetico):
    """O nulo sem p-valor é um histograma, não um teste."""
    resultados, _ = experimento_svm.rodar(
        df=df_sintetico, duracao_s=4.0, passo_s=4.0,
        n_permutacoes=3, n_dobras_internas=2, semente=0,
    )
    d = next(r for r in resultados if r["condicao"] == "D")
    assert 0.0 <= d["p_empirico"] <= 1.0
```

- [ ] **Step 2: Rodar e ver falhar**

```
cd backend && .venv/Scripts/python.exe -m pytest tests/test_experimento_svm.py -v
```
Esperado: `ModuleNotFoundError: No module named 'experimento_svm'`

- [ ] **Step 3: Implementar**

Criar `backend/scripts/experimento_svm.py`:

```python
# -*- coding: utf-8 -*-
"""SVM sobre a razão theta/beta do adhdata, sob validação aninhada por sujeito.

A MISSÃO

Folha manuscrita da reunião de 08/09/2026: "Executar pipeline de classificador e
extração de feature para θ/β e com SKLEARN. SVM".

POR QUE QUATRO CONDIÇÕES E NÃO UMA

A condição A é a pedida. Sozinha, ela produz um número que ninguém consegue
interpretar: 0,62 é bom? É melhor que o que já havia? Bate o acaso?

    A   TBR (19)              SVM-RBF     o que foi pedido
    B   potência de banda(95) SVM-RBF     a TBR joga informação fora?
    C   TBR (19)              LogReg      o SVM ganha sobre o que existia?
    D   TBR (19)              SVM, nulo   o número bate o acaso?

O QUE ESTE EXPERIMENTO NÃO AFIRMA

Nada sobre o estado da arte. A TBR é literatura contestada — efeito declinante
com o ano de publicação, parecer negativo de sociedade médica para uso
diagnóstico, e reanálise multiverso atribuindo boa parte do efeito ao componente
aperiódico e à frequência individual de alfa. Ela entra como linha de base a
bater, não como marcador em que o projeto aposta.

EXPECTATIVA REGISTRADA ANTES DE MEDIR

A medição de 07/09/2026 encontrou a TBR do grupo TDAH MENOR que a do controle no
adhdata (direção contrária à literatura), p = 0,132 em Cz. Uma AUC entre 0,55 e
0,70 é o resultado coerente com isso. AUC acima de 0,90 deve ser tratada como
suspeita de defeito, e investigada antes de reportada.

Uso:
    python scripts/experimento_svm.py
    python scripts/experimento_svm.py --max-sujeitos 10 --permutacoes 10
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import caracteristicas
import classificador
import config
import csv_data
import epocas as mod_epocas
import exportar_features
import preproc_basico
import receita as mod_receita


def _potencia_de_banda_do_conjunto(df, duracao_s, passo_s, max_sujeitos):
    """As 95 características de potência, para a condição B.

    Repete a montagem em vez de generalizar `montar_conjunto_tbr` com um
    parâmetro de tipo de característica: a montagem é curta, e um parâmetro que
    troca o que é extraído esconderia, numa flag, a diferença entre duas
    condições do experimento."""
    sujeitos = csv_data.list_subjects(df)
    if max_sujeitos:
        por_classe = {}
        for s in sujeitos:
            por_classe.setdefault(s["classe"], []).append(s)
        metade = max(1, max_sujeitos // max(1, len(por_classe)))
        escolhidos = []
        for lista in por_classe.values():
            escolhidos.extend(lista[:metade])
        sujeitos = escolhidos[:max_sujeitos]

    bX, by, bg = [], [], []
    for idx, s in enumerate(sujeitos):
        try:
            raw = preproc_basico.raw_de_dataframe(df, s["id"])
            filtrado, _ = preproc_basico.preprocessar(raw, l_freq=0.5, h_freq=None)
            janelas, _ = mod_epocas.epocar_janela_fixa(
                filtrado.get_data() * 1e6, csv_data.FS, duracao_s, passo_s
            )
            if len(janelas) == 0:
                continue
            X, _ = caracteristicas.potencia_de_banda(
                janelas, csv_data.FS, nomes_canais=list(csv_data.CANAIS_19)
            )
            bX.append(X)
            by.append(np.full(len(X), 1 if s["classe"] == "ADHD" else 0))
            bg.append(np.full(len(X), idx))
        except Exception:                 # noqa: BLE001
            continue
    return np.vstack(bX), np.concatenate(by), np.concatenate(bg)


def rodar(df=None, duracao_s=4.0, passo_s=4.0, max_sujeitos=None,
          n_permutacoes=100, n_dobras_internas=5, semente=0):
    """As quatro condições, e a receita para refazê-las."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if df is None:
        df = csv_data.load_csv(config.CAMINHO_ADHDATA)

    Xt, yt_txt, gt, _, _, _, dec = exportar_features.montar_conjunto_tbr(
        df, duracao_s, passo_s, max_sujeitos
    )
    yt = (yt_txt == "ADHD").astype(int)

    resultados = []

    a = classificador.avaliar_loso(
        Xt, yt, gt, n_dobras_internas=n_dobras_internas, semente=semente
    )
    resultados.append({"condicao": "A", "features": "TBR (19)", "modelo": "SVM-RBF",
                       "auc": a["auc"], "acuracia": a["acuracia"],
                       "n_sujeitos": a["n_sujeitos"], "p_empirico": None})

    Xb, yb, gb = _potencia_de_banda_do_conjunto(df, duracao_s, passo_s, max_sujeitos)
    b = classificador.avaliar_loso(
        Xb, yb, gb, n_dobras_internas=n_dobras_internas, semente=semente
    )
    resultados.append({"condicao": "B", "features": "potência de banda (95)",
                       "modelo": "SVM-RBF", "auc": b["auc"],
                       "acuracia": b["acuracia"], "n_sujeitos": b["n_sujeitos"],
                       "p_empirico": None})

    logreg = Pipeline([("escala", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=2000,
                                                  random_state=semente))])
    c = classificador.avaliar_loso(
        Xt, yt, gt, estimador=logreg, grade={"clf__C": [0.1, 1.0, 10.0]},
        n_dobras_internas=n_dobras_internas, semente=semente,
    )
    resultados.append({"condicao": "C", "features": "TBR (19)",
                       "modelo": "LogisticRegression", "auc": c["auc"],
                       "acuracia": c["acuracia"], "n_sujeitos": c["n_sujeitos"],
                       "p_empirico": None})

    d = classificador.nulo_por_permutacao(
        Xt, yt, gt, n_permutacoes=n_permutacoes,
        n_dobras_internas=n_dobras_internas, semente=semente,
    )
    resultados.append({"condicao": "D", "features": "TBR (19)",
                       "modelo": "SVM-RBF, rótulos permutados",
                       "auc": d["media_nula"], "acuracia": None,
                       "n_sujeitos": a["n_sujeitos"],
                       "p_empirico": d["p_empirico"]})

    dec["avaliacao"] = a["decisoes"]
    dec["nulo"] = d["decisoes"]
    receita = mod_receita.montar(
        banco="adhdata",
        sujeitos=[f"{dec['n_sujeitos_usados']} sujeitos"],
        etapas=dec,
        notas=exportar_features.RESSALVA_LICENCA,
    )
    return resultados, receita


def salvar_csv(resultados, caminho):
    """Uma linha por condição."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    campos = ["condicao", "features", "modelo", "auc", "acuracia",
              "n_sujeitos", "p_empirico"]
    with caminho.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(resultados)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epoca", type=float, default=4.0)
    p.add_argument("--passo", type=float, default=4.0)
    p.add_argument("--max-sujeitos", type=int, default=None)
    p.add_argument("--permutacoes", type=int, default=100)
    p.add_argument("--dobras-internas", type=int, default=5)
    p.add_argument("--saida", default="relatorios/experimento_svm.csv")
    args = p.parse_args()

    resultados, receita = rodar(
        duracao_s=args.epoca, passo_s=args.passo,
        max_sujeitos=args.max_sujeitos, n_permutacoes=args.permutacoes,
        n_dobras_internas=args.dobras_internas,
    )
    salvar_csv(resultados, args.saida)
    mod_receita.salvar(receita, Path(args.saida).with_suffix(".receita.json"))

    print(f"{'cond':<5}{'features':<24}{'modelo':<30}{'AUC':>7}{'p':>8}")
    for r in resultados:
        pe = "" if r["p_empirico"] is None else f"{r['p_empirico']:.3f}"
        print(f"{r['condicao']:<5}{r['features']:<24}{r['modelo']:<30}"
              f"{r['auc']:>7.3f}{pe:>8}")
    print(f"\n{exportar_features.RESSALVA_LICENCA}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Rodar e ver passar**

```
cd backend && .venv/Scripts/python.exe -m pytest tests/test_experimento_svm.py -v
```
Esperado: 2 passed

- [ ] **Step 5: Rodar a suíte inteira, conferir zero regressão**

```
cd backend && .venv/Scripts/python.exe -m pytest tests/ -q
```
Esperado: 421 + 13 novos = 434 passed, 2 skipped, 0 failed

- [ ] **Step 6: Commit**

```bash
git add backend/scripts/experimento_svm.py backend/tests/test_experimento_svm.py
git commit -m "Experimento das quatro condicoes de SVM sobre TBR"
```

---

## Task 6: Rodar sobre os 121 sujeitos e exportar os artefatos

**Files:**
- Create: `relatorios/experimento_svm.csv` (gerado)
- Create: `relatorios/experimento_svm.receita.json` (gerado)
- Create: `C:\dev\eeg_transformer\dados\features_adhdata_tbr.csv` (gerado)
- Create: `C:\dev\eeg_transformer\dados\features_adhdata_tbr.receita.json` (gerado)

**Interfaces:**
- Consumes: tudo das tarefas 1 a 5.
- Produces: os quatro artefatos acima.

**Contexto:** esta tarefa não escreve código. Ela roda o que foi construído e produz os arquivos. É onde o resultado real aparece pela primeira vez.

- [ ] **Step 1: Ensaio rápido, para confirmar que a montagem completa funciona**

```
cd backend && .venv/Scripts/python.exe scripts/experimento_svm.py --max-sujeitos 10 --permutacoes 5 --dobras-internas 2 --saida relatorios/ensaio_svm.csv
```
Esperado: tabela de quatro linhas impressa, sem exceção. Os valores não importam aqui — 10 sujeitos não sustentam conclusão.

- [ ] **Step 2: Rodada completa**

```
cd backend && .venv/Scripts/python.exe scripts/experimento_svm.py --permutacoes 100 --saida relatorios/experimento_svm.csv
```
Esperado: quatro linhas com `n_sujeitos = 121`.

⚠️ **Ao ler o resultado:** AUC entre 0,55 e 0,70 na condição A é o esperado. **AUC acima de 0,90 é suspeita de defeito, não sucesso** — se aparecer, pare e investigue vazamento antes de reportar qualquer coisa.

- [ ] **Step 3: Exportar as features para o eeg_transformer**

```
cd backend && .venv/Scripts/python.exe -c "
import sys; from pathlib import Path
sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
import config, csv_data, exportar_features, receita as mod_receita
df = csv_data.load_csv(config.CAMINHO_ADHDATA)
X, y, g, ids, eps, nomes, dec = exportar_features.montar_conjunto_tbr(df, 4.0, 4.0)
destino = Path(r'C:\dev\eeg_transformer\dados\features_adhdata_tbr.csv')
exportar_features.escrever_csv(destino, X, y, ids, eps, nomes)
r = mod_receita.montar(banco='adhdata', sujeitos=[f\"{dec['n_sujeitos_usados']} sujeitos\"], etapas=dec, notas=exportar_features.RESSALVA_LICENCA)
mod_receita.salvar(r, destino.with_suffix('.receita.json'))
print('linhas:', len(X), 'sujeitos:', len(set(ids.tolist())), 'colunas:', len(nomes))
"
```
Esperado: `sujeitos: 121`, e cerca de 4 000 linhas com épocas de 4 s.

- [ ] **Step 4: Conferir o contrato no arquivo gerado**

```
cd backend && .venv/Scripts/python.exe -c "
import csv
p = r'C:\dev\eeg_transformer\dados\features_adhdata_tbr.csv'
linhas = list(csv.DictReader(open(p, encoding='utf-8')))
print('cabecalho ok:', list(linhas[0])[:3] == ['sujeito_id','rotulo','epoca_idx'])
print('id original:', linhas[0]['sujeito_id'])
print('rotulos:', sorted({l['rotulo'] for l in linhas}))
print('sujeitos:', len({l['sujeito_id'] for l in linhas}))
"
```
Esperado: `cabecalho ok: True`, um ID como `v10p`, rótulos `['ADHD', 'Control']`, 121 sujeitos.

- [ ] **Step 5: Commit**

```bash
git add relatorios/experimento_svm.csv relatorios/experimento_svm.receita.json
git commit -m "Resultado das quatro condicoes sobre os 121 sujeitos do adhdata"
```

---

## Task 7: Notebook narrativo no eeg_transformer

**Files:**
- Create: `C:\dev\eeg_transformer\notebooks\01_tbr_svm_adhdata.ipynb`

**Interfaces:**
- Consumes: os módulos do backend do app, por `sys.path`; os artefatos da Task 6.
- Produces: nada que outro código consuma. É o artefato de leitura.

**Contexto e a regra que não pode ser quebrada:** o notebook **importa e narra, nunca implementa**. O motivo está escrito no `caracteristicas.py` do projeto: já houve uma segunda implementação que divergiu em silêncio da primeira, e a TBR do frontend passou a seguir `TBR_app = 1,396 × TBR_real^0,341` sem que nada denunciasse. Um notebook com lógica dentro seria essa segunda implementação, agora sem testes.

Se uma célula precisar de mais que uma chamada e um print, a lógica pertence a um módulo do backend, com teste.

- [ ] **Step 1: Criar o notebook**

Estrutura de células, alternando markdown e código:

**MD 1 — Título e contexto**
```markdown
# Razão theta/beta e SVM no adhdata

Missão 2 da reunião com a orientadora de 08/09/2026: *"Executar pipeline de
classificador e extração de feature para θ/β e com SKLEARN. SVM"*.

Este notebook **não implementa nada**. Ele importa o backend do app EEG, que é
onde a lógica vive com testes, e narra o que cada peça faz. Se você quiser
mudar como algo é calculado, mude lá — e o teste correspondente.

⚠️ `adhdata`: a licença deste banco não pôde ser confirmada. Ausência de
licença não é licença permissiva.
```

**CODE 1 — Import**
```python
import sys
from pathlib import Path

APP = Path(r"C:\3D brain\VERSAO GRATUITA MINHA\backend")
sys.path.insert(0, str(APP))
sys.path.insert(0, str(APP / "scripts"))

import caracteristicas, classificador, config, csv_data, exportar_features

print("bandas:", caracteristicas.BANDAS)
print("canais:", len(csv_data.CANAIS_19))
```

**MD 2 — O dataset**
```markdown
## O tamanho do dataset, e por que ele decide a partição

A orientadora pediu para separar treino, validação e teste, e para isso
verificar o tamanho do banco. A conta abaixo é o que decidiu o desenho.
```

**CODE 2 — Medir**
```python
df = csv_data.load_csv(config.CAMINHO_ADHDATA)
sujeitos = csv_data.list_subjects(df)
duracoes = [s["duracao_s"] for s in sujeitos]

print(f"sujeitos: {len(sujeitos)}")
print(f"ADHD: {sum(s['classe'] == 'ADHD' for s in sujeitos)}")
print(f"duração: {min(duracoes):.1f}s a {max(duracoes):.1f}s")
print(f"razão entre o mais longo e o mais curto: {max(duracoes)/min(duracoes):.1f}x")
```

**MD 3 — A decisão da partição**
```markdown
A unidade estatística é o **sujeito**, não a época: são 121, não as ~4 000
janelas. Um teste de 15% deixaria ~18 sujeitos, e com 18 sujeitos o
erro-padrão de uma AUC próxima de 0,75 fica em torno de ±0,12 — um intervalo
que não distingue um classificador que funciona de um que não funciona.

Por isso o desenho é **nested cross-validation**: LOSO por fora (121 dobras,
cada uma testa um sujeito), StratifiedGroupKFold-5 por dentro para escolher `C` e
`gamma` sem olhar o sujeito de teste. Isso também é o que a Proposta v2 do
mestrado já compromete como método.
```

**MD 4 — A feature**
```markdown
## A razão theta/beta

Sai das mesmas potências de banda que o resto do projeto usa, e não de uma PSD
própria: duas estimativas espectrais no mesmo projeto é como elas divergem.

A orientação é θ/β, a TBR clássica, com limiar de 4,0 na literatura. A folha
manuscrita de 01/09/2026 escreve β/θ; a de 08/09/2026 escreve θ/β. O recíproco
de uma razão de 9 é 0,11, um número apresentável que não é comparável com nada
publicado — por isso a decisão fica escrita.
```

**CODE 3 — Uma amostra da feature**
```python
X, y, g, ids, eps, nomes, dec = exportar_features.montar_conjunto_tbr(
    df, duracao_s=4.0, passo_s=4.0, max_sujeitos=6
)
print("nomes:", nomes[:3], "...", nomes[-1])
print("forma:", X.shape)
print("sujeitos:", sorted(set(ids.tolist())))
```

**MD 5 — O que a TBR não é**
```markdown
⚠️ A TBR é literatura contestada: o tamanho de efeito cai com o ano de
publicação, há parecer negativo de sociedade médica contra o uso diagnóstico, e
reanálise multiverso atribui boa parte do efeito ao componente aperiódico e à
frequência individual de alfa.

Ela entra aqui como **linha de base a bater**, não como marcador em que o
projeto aposta.

Além disso: a medição de 07/09/2026 encontrou a TBR **menor** no grupo TDAH
neste banco, direção contrária à literatura. AUC entre 0,55 e 0,70 é o
esperado. **AUC acima de 0,90 é suspeita de defeito, não sucesso.**
```

**MD 6 — A avaliação**
```markdown
## A avaliação, e a armadilha do LOSO

Cada dobra do LOSO testa um sujeito só. Uma dobra com um sujeito tem uma classe
só, e a AUC de uma classe só é indefinida — não existe "AUC média das dobras"
aqui. O que se faz: acumular um escore por sujeito e calcular a AUC uma vez.

O escore de um sujeito é a **mediana** dos escores das suas épocas. Mediana e
não média, porque o número de épocas varia 5,4× entre sujeitos.
```

**CODE 4 — Rodar num subconjunto**
```python
import numpy as np

y_bin = (y == "ADHD").astype(int)
r = classificador.avaliar_loso(X, y_bin, g, n_dobras_internas=2)

print(f"AUC: {r['auc']:.3f}")
print(f"acurácia: {r['acuracia']:.3f}")
print(f"sujeitos: {r['n_sujeitos']}")
print("hiperparâmetros escolhidos, por dobra:")
for h in r["hiperparametros_por_dobra"]:
    print("  ", h)
```

**MD 7 — O resultado completo**
```markdown
## O resultado sobre os 121 sujeitos

A célula acima roda com 6 sujeitos, para o notebook abrir rápido. O resultado
que vale está em `relatorios/experimento_svm.csv`, produzido por
`scripts/experimento_svm.py` sobre o banco inteiro, com as quatro condições.
```

**CODE 5 — Ler o resultado**
```python
import csv

p = APP.parent / "relatorios" / "experimento_svm.csv"
for linha in csv.DictReader(p.open(encoding="utf-8")):
    print(f"{linha['condicao']}  {linha['features']:<24} "
          f"{linha['modelo']:<30} AUC={float(linha['auc']):.3f}  "
          f"p={linha['p_empirico'] or '-'}")
```

**MD 8 — O contrato com este projeto**
```markdown
## O que este projeto recebe

`dados/features_adhdata_tbr.csv`, uma linha por época:

| coluna | o que é |
|---|---|
| `sujeito_id` | o ID original do banco (`v10p`), não o índice interno |
| `rotulo` | `ADHD` ou `Control`, texto |
| `epoca_idx` | índice da época dentro do sujeito |
| `<canal>_tbr` | 19 colunas, a razão theta/beta por canal |

`sujeito_id` é a coluna que importa: é ela que permite refazer a partição por
sujeito aqui sem reimplementar o split. Ao lado, `features_adhdata_tbr.receita.json`
traz o pré-processamento, o janelamento e os parâmetros da PSD — sem ele o CSV
é um número sem procedência.
```

**CODE 6 — Ler o artefato**
```python
import csv

f = Path(r"C:\dev\eeg_transformer\dados\features_adhdata_tbr.csv")
linhas = list(csv.DictReader(f.open(encoding="utf-8")))
print("linhas:", len(linhas))
print("sujeitos:", len({l["sujeito_id"] for l in linhas}))
print("colunas:", list(linhas[0])[:5], "...")
```

- [ ] **Step 2: Rodar o notebook de ponta a ponta**

Abrir no VS Code, escolher o interpretador `C:\3D brain\VERSAO GRATUITA MINHA\backend\.venv\Scripts\python.exe`, e executar todas as células.
Esperado: nenhuma exceção; todas as células com saída.

- [ ] **Step 3: Commit no repositório do eeg_transformer**

```bash
cd /c/dev/eeg_transformer
git add notebooks/01_tbr_svm_adhdata.ipynb dados/features_adhdata_tbr.csv dados/features_adhdata_tbr.receita.json
git commit -m "Notebook da TBR com SVM e o CSV de features vindo do app EEG"
```

⚠️ Conferir antes: se `dados/` estiver no `.gitignore` do `eeg_transformer`, **não force o add**. Nesse caso versione só o notebook e registre no README de onde o CSV vem.

---

## Task 8: Verificação independente

**Files:**
- Nenhum arquivo alterado. Esta tarefa produz um parecer.

**Contexto:** três ângulos, cada um por um agente que **não implementou o código**. Verificação pelo mesmo agente que escreveu é auto-aprovação.

- [ ] **Step 1: Ângulo de vazamento**

Despachar um subagente com este encargo:

> Leia `backend/scripts/classificador.py` e `backend/scripts/experimento_svm.py` de forma adversarial. Sua tarefa é encontrar QUALQUER caminho pelo qual informação do sujeito de teste chegue ao treino da sua dobra: normalização ajustada fora da dobra, seleção de features feita antes do split, hiperparâmetro escolhido com o teste dentro, ou o sujeito de teste aparecendo no `groups` do GridSearchCV interno. Não avalie estilo. Reprove se achar um caminho não declarado; cite arquivo e linha.

- [ ] **Step 2: Ângulo estatístico**

Despachar outro subagente:

> Leia `backend/scripts/classificador.py`, foco em `nulo_por_permutacao` e `agregar_por_sujeito`, e o CSV `relatorios/experimento_svm.csv`. Verifique: (a) a permutação é do rótulo por SUJEITO e não por época; (b) o p-valor usa `(b+1)/(m+1)` e não `b/m`; (c) a AUC é calculada uma vez sobre escores de sujeito e não como média por dobra; (d) as conclusões que os números permitem, com n = 121. Reprove qualquer afirmação que o n não sustente.

- [ ] **Step 3: Ângulo de contrato**

Despachar um terceiro:

> Sem ler o código que gerou o arquivo, abra `C:\dev\eeg_transformer\dados\features_adhdata_tbr.csv` e o `.receita.json` ao lado. Responda: com apenas estes dois arquivos, é possível refazer uma partição por sujeito sem vazamento? Existe alguma informação necessária que falta? Tente montar um GroupKFold a partir do CSV e relate se conseguiu.

- [ ] **Step 4: Consolidar**

Se algum ângulo reprovar, corrigir e repetir aquele ângulo. Só então seguir.

- [ ] **Step 5: Commit do parecer**

```bash
git add docs/superpowers/verificacao-2026-09-08.md
git commit -m "Parecer das tres verificacoes independentes do pipeline SVM"
```

---

## Task 9: Fechar o registro no vault

**Files:**
- Modify: `C:\Obsidian\MESTRADO_ITA\03 - Pesquisa\Reuniao - Orientadora 2026-09-08.md`
- Modify: `C:\Obsidian\MESTRADO_ITA\00 - Painel\Pendencias e Acoes.md`

**Contexto:** o vault roda sob protocolo RAG estrito. Um resultado medido entra como 🔢 **DERIVADO**, com a conta visível e a data. Não entra como ✅ DOCUMENTADO, que é reservado ao que está literal numa fonte.

- [ ] **Step 1: Registrar o resultado na nota da reunião**

Na seção "Missão dada", sob a linha da missão 2, acrescentar o resultado medido com marcador 🔢 **DERIVADO**, a data, o número de sujeitos, a AUC de cada condição e o p-valor empírico. Incluir a ressalva de licença e a ressalva sobre a TBR ser literatura contestada.

- [ ] **Step 2: Marcar a missão 2 como concluída nas pendências**

Em `Pendencias e Acoes.md`, seção "MISSÕES DA REUNIÃO DE 08/09/2026", trocar os `- [ ]` da missão 2 por `- [x]`, com a data. **Manter a linha da segunda parte** (componente periódico/aperiódico) aberta: ela continua sendo lacuna.

- [ ] **Step 3: Atualizar `updated:` no frontmatter das duas notas**

- [ ] **Step 4: Conferir que não há byte de controle nas notas**

```bash
cd "/c/Obsidian/MESTRADO_ITA" && python -c "
import io
for p in ['00 - Painel/Pendencias e Acoes.md','03 - Pesquisa/Reuniao - Orientadora 2026-09-08.md']:
    s = io.open(p, encoding='utf-8').read()
    ruins = [hex(ord(c)) for c in s if ord(c) < 9 or (13 < ord(c) < 32)]
    print(('LIMPO ' if not ruins else 'SUJO  ') + p, ruins[:5])
"
```
Esperado: LIMPO nas duas.

---

## Critério de conclusão

1. `classificador.py`, `exportar_features.py` e `experimento_svm.py` existem, com testes verdes.
2. A suíte inteira do backend passa sem regressão (421 antes, ~434 depois).
3. As quatro condições rodaram sobre os 121 sujeitos; CSV e receita gravados.
4. `features_adhdata_tbr.csv` e sua receita estão em `C:\dev\eeg_transformer\dados\`.
5. O notebook roda de ponta a ponta.
6. Os três ângulos de verificação passaram.
7. O vault registra o resultado com marcador de confiança, e a missão 2 está marcada como concluída.
