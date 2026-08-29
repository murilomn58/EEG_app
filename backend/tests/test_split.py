# -*- coding: utf-8 -*-
"""Trava o split por sujeito e a recusa de vazamento.

O QUE ESTE MÓDULO ESTÁ DEFENDENDO

Épocas do mesmo sujeito são parecidas entre si por razões que não têm nada a
ver com o rótulo: impedância dos eletrodos, formato do crânio, quanto a
criança se mexeu, a hora do dia. Um classificador treinado com algumas
épocas de uma criança e testado com outras épocas da MESMA criança aprende a
reconhecer a criança, e a acurácia sobe.

Esse número sobe de verdade, aparece na tabela e é publicável. É por isso que
o experimento inteiro existe: medir quanto dele some quando a fronteira de
sujeito é respeitada.

Os dois splits existem de propósito. `split_por_segmento` NÃO é um erro que
ficou no código — é a condição de controle, o número inflado que a figura
precisa mostrar ao lado do honesto.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import split


def grupos_de(*contagens):
    """Rótulo de sujeito por época: grupos_de(3, 2) -> [0,0,0,1,1]."""
    saida = []
    for s, n in enumerate(contagens):
        saida.extend([s] * n)
    return np.array(saida)


# ---------------------------------------------------------------------------
# a garantia central
# ---------------------------------------------------------------------------

def test_nenhum_sujeito_aparece_dos_dois_lados():
    """A razão de ser do módulo, sobre estrutura conhecida: 6 sujeitos com
    números DIFERENTES de épocas, porque gravação de duração diferente é o
    caso real e o desequilíbrio é onde o descuido aparece."""
    g = grupos_de(10, 3, 7, 5, 12, 4)
    for treino, teste in split.split_por_sujeito(g, n_dobras=3):
        s_treino = set(g[treino].tolist())
        s_teste = set(g[teste].tolist())
        assert not (s_treino & s_teste), (
            f"sujeito(s) {sorted(s_treino & s_teste)} em treino E teste"
        )


def test_toda_epoca_e_usada_como_teste_exatamente_uma_vez():
    """Cobertura: sem isso, um split poderia 'evitar vazamento' simplesmente
    descartando épocas, e a acurágica média sairia sobre um subconjunto
    escolhido."""
    g = grupos_de(10, 3, 7, 5, 12, 4)
    vistos = []
    for _, teste in split.split_por_sujeito(g, n_dobras=3):
        vistos.extend(teste.tolist())
    assert sorted(vistos) == list(range(len(g)))


def test_verificar_sem_vazamento_levanta_quando_ha_cruzamento():
    """A verificação é chamada de dentro do split, mas também é pública: quem
    montar um split à mão pode (e deve) passar por ela."""
    g = grupos_de(4, 4)
    treino = np.array([0, 1, 2, 4])   # sujeito 0 e 1
    teste = np.array([3, 5, 6, 7])    # sujeito 0 (índice 3) e 1 -> cruza
    with pytest.raises(ValueError, match="vazamento"):
        split.verificar_sem_vazamento(treino, teste, g)


def test_verificar_sem_vazamento_aceita_split_limpo():
    g = grupos_de(4, 4)
    split.verificar_sem_vazamento(np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7]), g)


def test_a_mensagem_de_vazamento_nomeia_o_sujeito():
    """Erro que diz 'houve vazamento' e não diz de quem obriga a caçar. O
    projeto inteiro trata mensagem de erro como parte do produto."""
    g = grupos_de(2, 2)
    with pytest.raises(ValueError) as exc:
        split.verificar_sem_vazamento(np.array([0, 1]), np.array([1, 2]), g)
    assert "0" in str(exc.value)


# ---------------------------------------------------------------------------
# a condição de controle
# ---------------------------------------------------------------------------

def test_split_por_segmento_ignora_sujeito_de_proposito():
    """É a condição que produz o número inflado. Ela TEM de vazar — se não
    vazasse, a figura não teria o que comparar.

    Com 6 sujeitos embaralhados em 3 dobras, é praticamente certo que algum
    sujeito caia dos dois lados; o teste afirma que isso acontece."""
    g = grupos_de(10, 3, 7, 5, 12, 4)
    cruzou = False
    for treino, teste in split.split_por_segmento(len(g), n_dobras=3, semente=0):
        if set(g[treino].tolist()) & set(g[teste].tolist()):
            cruzou = True
    assert cruzou, "o split por segmento deveria vazar; é a condição de controle"


def test_split_por_segmento_tambem_cobre_tudo():
    vistos = []
    for _, teste in split.split_por_segmento(20, n_dobras=4, semente=0):
        vistos.extend(teste.tolist())
    assert sorted(vistos) == list(range(20))


def test_split_por_segmento_e_reprodutivel_pela_semente():
    """Sem semente fixa, dois runs do experimento dão barras diferentes e
    ninguém sabe se a diferença é o método ou o sorteio."""
    a = [t.tolist() for _, t in split.split_por_segmento(20, 4, semente=7)]
    b = [t.tolist() for _, t in split.split_por_segmento(20, 4, semente=7)]
    c = [t.tolist() for _, t in split.split_por_segmento(20, 4, semente=8)]
    assert a == b
    assert a != c


# ---------------------------------------------------------------------------
# recusa em vez de silêncio
# ---------------------------------------------------------------------------

def test_mais_dobras_que_sujeitos_recusa_com_o_numero():
    """GroupKFold não consegue 5 dobras com 3 sujeitos. A mensagem diz os
    dois números, para o chamador saber o que ajustar."""
    g = grupos_de(5, 5, 5)
    with pytest.raises(ValueError) as exc:
        list(split.split_por_sujeito(g, n_dobras=5))
    assert "3" in str(exc.value) and "5" in str(exc.value)


def test_um_sujeito_so_recusa():
    """Com um sujeito não existe split por sujeito — e devolver uma dobra
    degenerada seria pior que recusar."""
    with pytest.raises(ValueError):
        list(split.split_por_sujeito(grupos_de(10), n_dobras=2))


def test_decisoes_do_split_registram_o_que_foi_feito():
    """Vai para a receita: outra pessoa precisa saber quantas dobras, qual
    semente e quantos sujeitos produziram aquelas barras."""
    g = grupos_de(4, 4, 4)
    dec = split.decisoes_do_split("sujeito", g, n_dobras=3, semente=42)
    assert dec["tipo"] == "sujeito"
    assert dec["n_dobras"] == 3
    assert dec["n_sujeitos"] == 3
    assert dec["n_epocas"] == 12
    assert dec["semente"] == 42
    assert dec["epocas_por_sujeito"] == {"0": 4, "1": 4, "2": 4}
