# O app vira instrumento: epocagem, split por sujeito, e a figura do vazamento

> **Para quem executa:** use `superpowers:subagent-driven-development` ou `superpowers:executing-plans`. Os passos usam checkbox (`- [ ]`).

**Objetivo:** fazer o app produzir o primeiro artefato que sai da tela e se sustenta numa banca — quatro barras medindo quanto da acurácia publicada em classificação de EEG é vazamento, e não patologia.

**Nota de local:** esta spec deveria morar em `docs/superpowers/specs/`. O modo de plano só permite editar este arquivo; mover para lá é o primeiro passo da execução.

---

## Contexto

O app hoje implementa, com rigor incomum, **as duas primeiras etapas** de um pipeline de EEG — passa-alta 0,5 Hz e notch na rede detectada do espectro, mais um passa-baixa clínico opcional. E **nenhuma das que vêm depois**: sem epocagem, sem rejeição, sem ICA, sem interpolação, sem normalização entre sujeitos.

Mais grave para o objetivo declarado: **nada quantitativo sai dele**. O `decisoes` — que registra o filtro aplicado, a rede detectada e o rebaixamento por Nyquist — viaja na resposta HTTP e morre no selo de texto da tela. TBR, PSD, correlação, Higuchi e localização de fonte: nenhum exporta. A única saída da interface é um TSV de anotações manuais. **Ninguém consegue repetir uma análise a partir do que o app entrega.**

O que muda isso já está desenhado e esperando. O próximo bloco do currículo é *5. Epocagem e normalização*, com caderno `status: pronto-para-executar`. As tarefas formam uma corrente travada em ordem — `epocas.py` → split por sujeito → `normalizacao.py` → medir o vazamento — e essa última é a única tarefa do app classificada como `experimento`, descrita no `Tarefas.md` como *"o único item da fila que produz número próprio, e a figura-manchete do próximo caderno"*.

Ela também responde diretamente à RQ1 recomendada na pauta com a orientadora: *"Quanto do desempenho publicado em classificação de TDAH/TEA por EEG sobrevive à avaliação rigorosa — e quanto do que sobra é identidade de sujeito, idade, sexo e site, em vez de patologia?"* — justificada ali como *"a única das três em que o resultado negativo também é contribuição"*.

### Duas incertezas que ficam declaradas, não resolvidas

**A linha de pesquisa está em disputa.** A `Proposta v2` registrada é sobre LLMs leves em linguagem espontânea; o rascunho da dissertação é sobre EEG e modelos fundacionais, e o README dele diz textualmente *"Divergência de linha de pesquisa — registrada, não resolvida"*. A reunião com a Prof.ª Sarah está `⬜ a marcar`. Este plano aposta na linha EEG. Se ela cair, o trabalho continua valendo como pipeline de preparação de dados — que é a proposta do CTML 2026 — mas deixa de ser caminho de tese.

**A licença do adhdata não pôde ser confirmada.** O README já registra: *"An absent license is not a permissive license."* O experimento roda sobre ele por ser o único dado onde fecha hoje (rótulo binário, sem DUA, fronteira de sujeito explícita). A ressalva entra na figura e no caderno, não fica só no repositório.

---

## Decisões já tomadas

| Decisão | Escolha |
|---|---|
| O que o app precisa produzir | Resultado de dissertação **e**, mais adiante, a entrada do transformer de EEG |
| Como o lote entra | **Backend em lote, app como bancada**: o app decide e confere um sujeito, o lote roda os 121 com os mesmos parâmetros |
| Primeiro artefato | **A figura do vazamento** |
| Dado do experimento | **adhdata**, com a ressalva de licença escrita na figura |

---

# Parte 1 — A espinha: um pipeline, duas portas

O princípio já está no repositório e é o que dá valor ao QC: `qc_relatorio.py` **importa** `preproc_basico` em vez de reimplementar, porque *"um QC que mede outra implementação não é evidência sobre nada"*. O lote segue a mesma regra. **Não existe um segundo pipeline.**

Cada módulo devolve `(dado, decisoes)`, como `preprocessar` já faz — é o padrão de registro que o projeto adotou e que faz a receita da Parte 2 ser possível.

## Tarefa 1: `backend/scripts/epocas.py`

- [ ] **Passo 1:** `segmentos_continuos(raw_ou_array, fronteiras)` — devolve as faixas de amostra que **não** atravessam descontinuidade. Duas fontes de fronteira, e as duas são reais e medidas:
  - HBN: evento `boundary` do EEGLAB, presente em **9 dos 10** sujeitos inventariados. Já vem lido por `eventos.py`.
  - adhdata: a coluna `ID` é a única fronteira que existe. Janela que a cruza **emenda duas crianças numa época só**.
- [ ] **Passo 2:** `epocar_janela_fixa(dado, fs, duracao_s, passo_s, segmentos)` — janelas de comprimento fixo com sobreposição configurável, descartando qualquer janela que não caiba inteira dentro de um segmento. Devolve `(épocas, decisoes)` com quantas foram descartadas e por quê.
- [ ] **Passo 3:** teste que **prova** a fronteira: dado sintético com dois "sujeitos" de amplitude conhecida concatenados; nenhuma época pode conter amostra dos dois. Este é o teste que dá sentido ao módulo.
- [ ] **Passo 4:** o comprimento e o passo escolhidos aqui são os parâmetros que o tokenizador do transformer vai receber — o caderno já diz isso. Registrá-los na receita com esse nome.

## Tarefa 2: `backend/scripts/split.py`

- [ ] **Passo 1:** `split_por_sujeito(grupos, n_dobras)` sobre `GroupKFold`, e `split_por_segmento(n, n_dobras)` sobre `KFold` — **os dois**, porque o segundo é o que produz o número inflado que a figura precisa mostrar.
- [ ] **Passo 2:** `verificar_sem_vazamento(treino, teste, grupos)` que **levanta** se um sujeito aparece dos dois lados. Chamada dentro do split por sujeito, sempre.
- [ ] **Passo 3:** teste sobre dado sintético com estrutura de sujeitos conhecida, provando ausência de vazamento — é uma tarefa já escrita no vault (60 XP).

## Tarefa 3: `backend/scripts/normalizacao.py`

- [ ] **Passo 1:** **duas funções separadas**, e essa separação é o ponto estrutural do módulo: `ajustar(X) -> params` e `aplicar(X, params) -> X'`. Nunca um `fit_transform`.
- [ ] **Passo 2:** a API torna o vazamento **um erro de tipo**: quem aplica precisa de um `params`, e `params` só nasce de um `ajustar`. Ajustar no conjunto inteiro passa a ser uma linha visível e deliberada, não um descuido.
- [ ] **Passo 3:** z-score por canal como primeira implementação; a interface aceita outras sem mudar quem chama.

## Tarefa 4: `backend/scripts/receita.py`

- [ ] **Passo 1:** a receita é um dicionário serializável com: banco, sujeitos, todo parâmetro de cada etapa, o `decisoes` do preproc, versões de `mne`/`numpy`/`scipy`/`sklearn`, e timestamp.
- [ ] **Passo 2:** `salvar(receita, caminho)` e `carregar(caminho)`. O lote **consome** uma receita e roda idêntico.
- [ ] **Passo 3:** conferir na carga que as versões batem; divergência é **aviso declarado**, não erro — e nunca silêncio.

---

# Parte 2 — O app exporta o que já sabe

## Tarefa 5: a sessão vira arquivo

- [ ] **Passo 1:** rota `GET /receita` que devolve a receita do estado corrente (banco, sujeito, preproc, referência, janela). Reaproveita o `decisoes` que `_preprocessar_sujeito` já produz.
- [ ] **Passo 2:** botão na gaveta: **baixar receita + medidas**. Junto da receita saem as medidas que o painel já calcula — TBR, perfil espectral, correlação, Higuchi — com a janela de tempo que as produziu. Hoje nenhuma sai.
- [ ] **Passo 3:** o arquivo carrega a mesma ressalva de procedência que o TSV de rótulos já carrega, e o identificador do sujeito — que o README avisa viajar em todo relatório gerado.

---

# Parte 3 — A figura

## Tarefa 6: `backend/scripts/experimento_vazamento.py`

- [ ] **Passo 1:** características por época: potência de banda por canal (19 × 5 = 95), integrada da PSD de Welch **do backend**, que já tem teste numérico. Nada de reimplementar.
- [ ] **Passo 2:** classificador transparente — regressão logística. O que a figura afirma é a **diferença entre condições**, não a acurácia absoluta, e isso tem de estar escrito na legenda.
- [ ] **Passo 3:** as quatro barras, que separam as **duas** fontes de vazamento em vez de misturá-las:

  | | normalização global | normalização por dobra |
  |---|---|---|
  | **split por segmento** | A — mais otimista | B |
  | **split por sujeito** | C | **D — o número honesto** |

- [ ] **Passo 4:** saída em CSV (uma linha por dobra e condição, com acurácia, AUC e n) mais figura vetorial em PDF, no padrão de `figuras_limpeza.py`. O CSV é o que permite alguém refazer a figura sem refazer o experimento.
- [ ] **Passo 5:** a legenda declara: banco, n de sujeitos, n de épocas, comprimento de janela, classificador, e a ressalva de licença do adhdata.
- [ ] **Passo 6:** **época por sujeito tem de ser equilibrada, ou declarada.** Gravações de duração diferente geram números diferentes de épocas, e sem cuidado o sujeito com mais épocas pesa mais no treino e no teste — o que confunde "aprendeu patologia" com "aprendeu o sujeito mais comprido". Ou se amostra o mesmo número por sujeito, ou o desequilíbrio entra na legenda com o número.
- [ ] **Passo 7:** **o resultado esperado é a acurácia cair de A para D.** Se não cair, isso é achado e vai escrito assim — não é motivo para procurar outra configuração até a barra ficar bonita.

## Tarefa 7: `backend/scripts/lote.py`

- [ ] **Passo 1:** CLI que recebe uma receita e roda os 121 sujeitos pelo mesmo caminho de código do app.
- [ ] **Passo 2:** memória com teto: processar sujeito a sujeito, escrevendo características em disco. O `LIMITE_CACHE_RAW = 2` existe porque dez sujeitos em cache levaram o RSS de 2150 a 2937 MB — o lote não pode reintroduzir o problema por outra porta.
- [ ] **Passo 3:** progresso e falha por sujeito no relatório final. Sujeito que falhou **aparece**; não some da contagem.

---

# Parte 4 — A dívida que impede confiar na tela

> **Trilha separada.** As Partes 1 a 3 formam um entregável só: a figura. A Parte 4 é outro, e **não bloqueia** a figura — o experimento roda no backend, cuja matemática já tem teste numérico. Executar como plano próprio, antes ou depois, sem misturar as duas revisões.

## Tarefa 8: extrair e testar a matemática de sinal

Hoje o frontend é verificado por **60 asserções de grep** em 4 arquivos de teste, e nenhuma compara um número. Elas afirmam coisas como `assert "melhorDist > 2 / FS" in corpo` — presença de uma linha literal. Renomear uma variável quebra o teste; inverter a lógica sem mudar o texto, não.

- [ ] **Passo 1:** extrair para `assets/sinal.js` as ~250 linhas que **não leem nenhuma global**: `fftRadix2`, `janelaHann`, `tamanhoBlocoPSD`, `densidadeEspectral`, `potenciaDeBanda`, `correlacao`, `dimensaoFractalHiguchi`, `criarBiquad`, `aplicarBiquad`, `_percentil`. Verificado: todas recebem série e `fs` por parâmetro.
- [ ] **Passo 2:** testar com números, contra fatos conhecidos: senoide de 10 Hz concentra a potência em alpha; a integral da PSD bate com a variância do sinal (Parseval); Higuchi dá ~1,0 para senoide, ~1,5 para browniano, ~2,0 para ruído branco; o biquad tem o mesmo corte **em Hz** a 128 e a 500 Hz.
- [ ] **Passo 3:** **o backend como oráculo** — mesma série, `densidadeEspectral` em JS contra a PSD do scipy, com tolerância declarada. É isto que impede o dashboard mostrar um TBR e o relatório de QC mostrar outro sem nada denunciar.
- [ ] **Passo 4:** unificar o CAR. Ele existe em duas cópias **já divergentes**: só uma tem a guarda para `Cz` indefinido. Uma função, chamada pelos dois caminhos.

## Tarefa 9: declarar as dependências

- [ ] **Passo 1:** `scipy`, `scikit-learn` e `matplotlib` estão instalados e **não estão no `requirements.txt`**, que lista só fastapi, uvicorn, mne, numpy, pandas, pytest, httpx. Quem clonar não roda o experimento. Acrescentar com piso de versão.

---

## Verificação

1. **Fronteira de época:** dado sintético com dois sujeitos concatenados de amplitude conhecida — nenhuma época contém amostra dos dois. O teste falha se a guarda for removida (rodar mutação).
2. **Ausência de vazamento:** dado sintético com estrutura de sujeitos conhecida; `verificar_sem_vazamento` levanta quando um sujeito cruza a fronteira.
3. **Receita ida e volta:** exportar do app, rodar o lote com ela, e conferir que os parâmetros aplicados são idênticos aos exportados.
4. **A figura:** as quatro barras com n declarado. Conferir que D ≤ C ≤ A e D ≤ B ≤ A — se a ordem quebrar, investigar antes de escrever.
5. **Oráculo:** `densidadeEspectral` do JS contra scipy sobre a mesma série, dentro da tolerância declarada.
6. **Sem regressão:** `cd backend && python -m pytest tests/` — hoje 295 passed, 2 skipped.
7. **Mutação em todo teste novo:** desfazer o conserto e confirmar que o teste cai. Nesta sessão, dois de três testes escritos passaram sobre código quebrado por casarem com o comentário.

---

## Fora de escopo, e por quê

- **ICA.** Está travada por dependência declarada no grafo, e epocagem vem antes. Entra depois da figura.
- **Rótulo que exija DUA.** Sem o Data Use Agreement não há diagnóstico clínico, só as quatro dimensões contínuas do fenótipo aberto.
- **Reescrever `desenharTracosClinico`.** São 330 linhas lendo 25 globais, a maior função do arquivo. É dívida real, mas mexer nela agora não aproxima a figura.
- **HBN no experimento.** Exige limiarizar `attention`, e essa decisão está travada pelo portão da linha de pesquisa. Rodar nos dois bancos é o passo seguinte natural, e fortalece muito o achado — depois que a linha for decidida.
- **Rejeição automática de época.** O vault já registra o viés: limiar fixo descarta mais trecho de quem se mexe mais, e em TDAH pediátrico isso correlaciona com o rótulo. Fazer isso mal contamina o experimento que este plano existe para produzir.
