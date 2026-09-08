# Pipeline de classificação TBR + SVM sobre o adhdata

**Data:** 08/09/2026
**Origem:** missão 2 da folha manuscrita da reunião com a orientadora de 08/09/2026
**Prazo:** 15/09/2026 (próxima reunião; cadência semanal às terças)
**Estado:** aprovado para implementação

---

## 1. O que a orientadora pediu, literalmente

> *"Executar pipeline de classificador e extração de feature para θ/β e com SKLEARN. SVM"*
>
> Anotação na margem, ligada por seta: *"Periódico e Aperiódico ↓ que muda por idade."*

Em relato verbal na mesma data (sem documento), acrescentou: separar o dataset em treino,
validação e teste, e para isso verificar o tamanho do dataset.

**O que este spec cobre:** a primeira metade — θ/β com SVM, com partição honesta.

**O que este spec NÃO cobre:** o componente periódico/aperiódico. A folha não nomeia o método;
a folha de 01/09/2026 associa a decomposição à sigla FOOOF, mas esta não. Fica como lacuna a
confirmar com a orientadora antes de qualquer implementação. Inventar o método seria inventar a
missão.

---

## 2. O que já existe (e por que este spec é menor do que parece)

Levantamento do repositório em 08/09/2026. Estes módulos estão implementados e cobertos por testes:

| Módulo | O que faz | Estado |
|---|---|---|
| `scripts/caracteristicas.py` | `potencia_de_banda()` (95 features), `razao_theta_beta()` (19 features) | pronto |
| `scripts/split.py` | `split_por_sujeito()` com GroupKFold e verificação de vazamento | pronto |
| `scripts/normalizacao.py` | `ajustar()` / `aplicar()` separados, contra vazamento de normalização | pronto |
| `scripts/epocas.py` | `epocar_janela_fixa()` | pronto |
| `scripts/receita.py` | serialização do que é preciso para refazer uma análise | pronto |
| `scripts/experimento_vazamento.py` | pipeline completo de ponta a ponta | pronto, mas com `LogisticRegression` |

**O buraco real é estreito:** o classificador é regressão logística, e a missão pede SVM. Mais o
artefato de features, que não existe em lugar nenhum.

Isto muda a natureza do trabalho: não é construir um pipeline, é **acrescentar um estimador e um
protocolo de avaliação a um pipeline que já roda**, e exportar o resultado.

---

## 3. O dataset, medido

Medições feitas em 08/09/2026 sobre `adhdata.csv`, não citadas de documentação:

| Medida | Valor |
|---|---|
| Sujeitos | 121 (61 TDAH, 60 controle) |
| Amostras totais | 2 166 383 |
| Canais | 19 (padrão 10-20) |
| Taxa de amostragem | 128 Hz |
| Duração por sujeito | 62,4 s a 337,9 s (mediana 130,4 s) |
| Épocas de 2 s, sem sobreposição | ~8 400 |
| Épocas de 4 s, sem sobreposição | ~4 173 |

**A observação que decide a partição:** a razão entre o sujeito mais longo e o mais curto é 5,4.
A unidade estatística é o **sujeito**, não a época — são 121, não 8 400.

---

## 4. Decisão: nested cross-validation, não holdout de três vias

A orientadora pediu "treino, validação e teste". A implementação escolhida é nested CV.
Não é desobediência: é a forma de honrar a intenção — não escolher hiperparâmetro olhando o dado
de teste — num dataset onde o holdout literal não funciona.

**Razão 1, documental.** A Proposta v2 do mestrado já compromete o trabalho com
*leave-one-subject-out*: em `Metricas - Avaliacao Clinica e Computacional` do vault, LOSO aparece
como "um sujeito por vez fora do treino — evita vazamento entre amostras do mesmo participante".
Adotar holdout agora contrariaria o método proposto, e a banca perguntaria por quê.

**Razão 2, matemática.** Um teste de 15% de 121 sujeitos deixa ~18 sujeitos. Com 18 sujeitos, o
erro-padrão de uma AUC próxima de 0,75 fica na casa de ±0,12. O resultado reportável seria
"AUC entre 0,63 e 0,87" — um intervalo que não distingue um classificador que funciona de um que
não funciona. Gastar 18 dos 121 sujeitos num teste que se olha uma vez é caro demais aqui.

**O desenho:**

```
Externo:  LOSO — 121 dobras, cada uma testa 1 sujeito
Interno:  StratifiedGroupKFold-5 sobre o treino da dobra, para C e gamma
```

Cada dobra externa produz uma predição para um sujeito que o modelo nunca viu, e cujos
hiperparâmetros foram escolhidos sem olhá-lo. As 121 predições formam a métrica.

**Efeito colateral desejável:** como cada dobra testa exatamente um sujeito, a métrica é
naturalmente por sujeito, e o sujeito de gravação longa não pesa mais que o de gravação curta.

**Consequência que muda a implementação (verificada no ambiente em 08/09/2026).** Como cada dobra
externa testa **um único sujeito**, a dobra tem uma classe só, e `roc_auc_score` é indefinida nela
(`UndefinedMetricWarning: Only one class is present in y_true`). **Não existe "AUC média entre as
dobras" no LOSO.** O desenho correto é acumular um escore por sujeito ao longo das 121 dobras e
calcular a AUC **uma vez**, sobre o vetor de 121 escores e 121 rótulos.

Isso também define como cada sujeito vira um escore: o modelo prediz um escore por época, e o
escore do sujeito é a **mediana** dos escores das suas épocas. Mediana, e não média, porque o
número de épocas varia de 5,4× entre sujeitos e a mediana é menos sensível a uma época
contaminada por artefato.

**Correção de 08/09/2026, medida no ambiente.** O interno tem de ser
`StratifiedGroupKFold`, e não `GroupKFold`: este respeita a fronteira de sujeito mas ignora a
classe, e produz dobras internas com uma classe só, em que o `SVC` levanta *"The number of
classes has to be greater than one"*. O estratificado satisfaz as duas restrições.

**Degrau de recuo, se o custo apertar:** StratifiedGroupKFold externo de 10 dobras, mesma
estrutura aninhada. Registrar a troca na receita se acontecer.

---

## 5. O classificador

```python
Pipeline([
    ('escala', StandardScaler()),
    ('svm', SVC(kernel='rbf')),
])
```

**`probability=False`, e a AUC sai de `decision_function()`.** `probability=True` roda Platt
scaling internamente, com um CV próprio, e multiplica o custo por cerca de 5. A AUC depende só da
ordenação dos escores, e `decision_function` fornece a ordenação. São matematicamente equivalentes
para AUC, e um é cinco vezes mais barato.

**A normalização entra como passo do `Pipeline`, não antes dele.** É o que garante que o
`StandardScaler` seja ajustado dentro de cada dobra, e não sobre o conjunto todo. O projeto já tem
`scripts/normalizacao.py` justamente para tornar esse vazamento visível quando ele acontece de
propósito; aqui ele não deve acontecer, e o `Pipeline` é a forma de o sklearn garantir isso.

**Grade de busca do interno:**

| Parâmetro | Valores |
|---|---|
| `C` | 0,1 · 1 · 10 · 100 |
| `gamma` | `'scale'` · 0,01 · 0,1 |

12 combinações × 5 dobras internas × 121 dobras externas = 7 260 ajustes de SVM. Com ~4 173 épocas
de 4 s e 19 features, cada ajuste é de milissegundos a poucos segundos. Aceitável.

---

## 6. As quatro condições

Um número sozinho não sustenta afirmação. O que este experimento produz é uma comparação:

| Condição | Features | Modelo | Pergunta que responde |
|---|---|---|---|
| **A** | TBR (19) | SVM-RBF | o que a orientadora pediu |
| **B** | potência de banda (95) | SVM-RBF | a TBR está jogando informação fora? |
| **C** | TBR (19) | LogisticRegression | o SVM ganha algo sobre o que já existe? |
| **D** | TBR (19) | SVM-RBF, rótulos permutados | a AUC medida bate acaso? |

**A condição D é inegociável.** Com 121 sujeitos, uma AUC de 0,62 pode ser sinal fraco ou ruído.
O nulo por permutação (rótulos embaralhados **respeitando a fronteira de sujeito**, **50**
repetições) dá a distribuição sob a hipótese nula, e daí o p-valor empírico da AUC observada.

⏱️ **50 e não 100, por custo medido em 08/09/2026.** O nulo roda o LOSO inteiro uma vez por
permutação: a 13,4 min por avaliação sobre 121 sujeitos, 100 permutações custam 22,3 h, contra
~12 h com 50. O p-valor mínimo passa de 0,010 para 0,020, sem consequência dada a expectativa de
AUC entre 0,55 e 0,70. Reduzir a grade do nulo em vez disso foi descartado: acelera 8×, mas
triplica o desvio da distribuição nula (0,073 → 0,210).

⚠️ **Atenção de implementação:** permutar rótulo por época destruiria a estrutura de sujeito e
produziria um nulo otimista demais. A permutação é do rótulo **de cada sujeito**, e todas as épocas
daquele sujeito carregam o rótulo permutado.

**Expectativa registrada antes de medir**, para que o resultado não seja lido a posteriori:
a medição de 07/09/2026 encontrou a TBR do grupo TDAH **menor** que a do controle no adhdata
(direção contrária à literatura), com p = 0,132 em Cz e efeito maior em T7 e P7. Uma AUC modesta
— 0,55 a 0,70 — é o resultado coerente com isso. **AUC acima de 0,90 deve ser tratada como suspeita
de defeito, não como sucesso**, e investigada antes de ser reportada.

---

## 7. O contrato entre o app e o eeg_transformer

Hoje inexistente. O `README.md` do `eeg_transformer` diz que o app "prepara os dados que este
projeto consome", mas nenhum formato foi definido, e o projeto está vazio por isso.

**Formato: CSV, uma linha por época.**

```
sujeito_id, rotulo, epoca_idx, Fp1_tbr, Fp2_tbr, F3_tbr, ..., Pz_tbr
```

- `sujeito_id` — o `ID` original do adhdata (ex.: `v10p`), **não** o índice interno. É a chave que
  permite ao transformer refazer a partição por sujeito sem reimplementar o split.
- `rotulo` — `ADHD` ou `Control`, texto, não código numérico. Legível sem consultar um dicionário.
- `epoca_idx` — índice da época dentro do sujeito.
- 19 colunas de TBR, uma por canal.

Ao lado do CSV, um `features_adhdata_tbr.receita.json` gerado por `scripts/receita.py`, com o
pré-processamento, o janelamento e os parâmetros da PSD. Sem ele o CSV é um número sem procedência.

**Por que CSV e não `.npz`:** o arquivo vai ser aberto no notebook, mostrado à orientadora e ter
trechos colados na dissertação. São ~8 400 linhas × 22 colunas, cerca de 2 MB. Legibilidade vence
compactação nesta escala.

---

## 8. Onde cada coisa mora

```
VERSAO GRATUITA MINHA/backend/
├── scripts/classificador.py          NOVO — SVM, nested CV, nulo por permutação
├── scripts/exportar_features.py      NOVO — gera o CSV + receita
├── tests/test_classificador.py       NOVO
├── tests/test_exportar_features.py   NOVO
├── scripts/caracteristicas.py        (usado, não alterado)
├── scripts/split.py                  (usado, não alterado)
└── scripts/normalizacao.py           (usado, não alterado)

eeg_transformer/
├── dados/features_adhdata_tbr.csv           ARTEFATO gerado
├── dados/features_adhdata_tbr.receita.json  ARTEFATO gerado
└── notebooks/01_tbr_svm_adhdata.ipynb       NOVO — narra e importa
```

**A regra que separa as duas coisas:** a lógica vive em módulo testado; o notebook **importa e
narra, nunca implementa**.

O motivo está escrito no próprio `caracteristicas.py`: o projeto já sofreu de uma segunda
implementação divergir em silêncio da primeira — a TBR do frontend passou a seguir
`TBR_app = 1,396 × TBR_real^0,341` sem que nada denunciasse. Um notebook com lógica dentro seria
exatamente essa segunda implementação, agora sem testes.

O notebook fica com markdown explicando cada bloco e chamadas de uma linha. É o artefato que a
orientadora abre e lê; o módulo é o que garante que o que ela lê é o que roda.

---

## 9. Testes

Além dos habituais de contorno, três que existem para impedir defeitos específicos:

1. **`test_loso_nao_vaza_sujeito`** — nenhum `sujeito_id` aparece em treino e teste da mesma dobra.
   Redundante com o `GroupKFold`, e deliberado: a garantia fica afirmada no ponto de uso.
2. **`test_normalizacao_ajustada_fora_da_dobra_mudaria_o_escore`** — os MESMOS dados dos dois
   lados, mudando só de onde saem média e desvio: dentro da dobra contra o conjunto inteiro. Se o
   `StandardScaler` fosse ajustado fora da dobra, o escore do LOSO bateria com o do braço vazado.

   ⚠️ **Correção de 08/09/2026.** A primeira redação deste teste alterava o dado de um sujeito e
   exigia que os escores dos outros não mudassem. Era inválida: nas dobras dos outros sujeitos,
   aquele sujeito é dado de **treino legítimo**, e mudar o treino muda o modelo. Nenhum pipeline
   correto passaria. Medido: o escore de um sujeito ia de −0,82 para +1,00, com o pipeline certo.
3. **`test_nulo_por_permutacao_fica_bem_abaixo_do_sinal`** — com rótulos permutados por sujeito, a
   AUC nula fica bem abaixo da observada. Se subir, há vazamento em algum lugar do pipeline.

   ⚠️ **Correção de 08/09/2026, medida no ambiente.** A primeira redação exigia AUC nula próxima de
   0,5. **É falso no LOSO**, e o teste falhava com o código correto. A AUC nula do LOSO tem **viés
   negativo estrutural**: ao remover o sujeito de teste do treino, o treino fica sistematicamente
   desbalanceado contra a classe dele, e o modelo aprende a maioria e erra justamente nele. A
   correlação entre o desbalanceio do treino e o rótulo do sujeito de teste, em 2000 sorteios por
   tamanho, é −0,071 (n=8), −0,033 (n=16), −0,013 (n=40) e −0,004 (n=121) — escala com 1/n, a
   assinatura da origem combinatória. Medido nas AUC nulas reais: 0,086 com 8 sujeitos, 0,348 com 16.

   ⚠️ **Segunda correção, de 08/09/2026, sobre a frase anterior deste mesmo parágrafo.** A redação
   que fechava este item dizia "com 121 sujeitos o viés é de −0,004, desprezível". Estava errada, e
   a verificação independente do dia 08/09/2026 confirmou o erro.

   O engano: mediu-se a correlação entre o desbalanceio do treino e o rótulo do sujeito de teste,
   uma quantidade no **espaço dos rótulos**, e concluiu-se algo sobre o **viés da AUC**, que é
   estatística de ordenação sobre os escores. São grandezas diferentes. A correlação por dobra
   escala com 1/n; o viés na AUC não, porque atua no mesmo sentido em todas as n dobras e a
   agregação o soma em vez de cancelá-lo.

   Números medidos, sob H0 puro (rótulos sem relação com o sinal), com um modelo que só enxerga a
   prevalência do treino:

   | n | AUC nula |
   |---|---|
   | 8 | 0,000 |
   | 16 | 0,010 |
   | 40 | 0,179 |
   | 121 | 0,384 |
   | 300 | 0,451 |

   Confirmado de forma independente com o pipeline SVM real sob ruído puro em n=121: AUC nula entre
   0,311 e 0,364 conforme o cenário, chegando a 0,0 exato numa repetição com 30 épocas por sujeito.

   A conclusão correta é o oposto da anterior: o viés é grande, é negativo, e **não** desaparece com
   n.

   Isto não invalida o método do nulo por permutação, é a razão dele existir. O p-valor empírico
   compara a AUC observada com a distribuição nula **medida**, nunca com 0,5 teórico. Comparar com
   0,5 é que seria o erro, e um erro que muda a leitura do resultado numa direção específica: a
   correção torna o resultado deste trabalho **mais forte**, não mais fraco. Uma AUC observada de
   0,55 parece medíocre contra 0,5; é substancial contra uma nula medida de 0,35 a 0,38.

Mais um teste de contrato: **`test_csv_tem_sujeito_id_original`** — a coluna `sujeito_id` traz o ID
do adhdata, não o índice interno. É o que o transformer precisa para refazer o split.

---

## 10. Verificação independente

Decisão de 08/09/2026: usar agentes de verificação, com uma regra — **quem verifica não é quem
implementou**. Verificação pelo mesmo agente que escreveu o código é auto-aprovação.

| Ângulo | O que examina | Critério de reprovação |
|---|---|---|
| **Vazamento** | leitura adversarial de `classificador.py` procurando qualquer caminho em que informação do teste chegue ao treino | achou um caminho não declarado |
| **Estatística** | se o nulo por permutação está correto, se a agregação é por sujeito, se a AUC é interpretável no n disponível | conclusão que o n não sustenta |
| **Contrato** | se o CSV gerado é de fato consumível pelo `eeg_transformer` sem reimplementar o split | falta informação para refazer a partição |

O terceiro é o que costuma falhar em silêncio: o CSV "parece certo" e só na hora de usar se
descobre que falta a chave.

---

## 11. O que este trabalho NÃO vai afirmar

- **Nada sobre acurácia absoluta do estado da arte.** SVM sobre TBR é linha de base, e a própria
  TBR é literatura contestada — efeito declinante com o ano de publicação, parecer negativo de
  sociedade médica para uso diagnóstico, e reanálise multiverso atribuindo boa parte do efeito ao
  componente aperiódico e à frequência individual de alfa. Isto já está escrito no docstring de
  `razao_theta_beta` e deve sair também na saída do experimento.
- **Nada sobre o HBN.** Este trabalho é sobre o adhdata. O Drive do HBN é somente leitura e não
  entra nesta fase.
- **Nada sobre o efeito do ICA.** É a missão 3, e vem depois justamente para ter este baseline
  contra o que comparar.
- **Nada sobre idade.** A anotação "muda por idade" pertence ao componente periódico/aperiódico,
  fora do escopo deste spec. O adhdata tem faixa de 7 a 12 anos e a idade individual não está no
  CSV.

---

## 12. Ressalva de licença

A licença do `adhdata` não pôde ser confirmada. O `README.md` do app registra que ausência de
licença não é licença permissiva. A ressalva já viaja no `experimento_vazamento.py` e deve viajar
também no CSV exportado e no notebook, não ficar só no repositório.

---

## 13. Critério de conclusão

A missão 2 está entregue quando:

1. `classificador.py` e `exportar_features.py` existem, com testes verdes e **sem regressão** nos
   421 testes atuais do backend.
2. As quatro condições rodaram sobre os 121 sujeitos, com resultado em CSV e receita JSON.
3. O CSV de features e sua receita estão em `eeg_transformer/dados/`.
4. O notebook roda de ponta a ponta e narra o que cada bloco faz.
5. Os três ângulos de verificação independente passaram.
6. A nota `Reuniao - Orientadora 2026-09-08` do vault registra o resultado medido, com marcador de
   confiança, e `Pendencias e Acoes` marca a missão 2 como concluída.
