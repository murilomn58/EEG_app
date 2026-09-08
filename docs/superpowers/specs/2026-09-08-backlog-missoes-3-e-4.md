# Backlog das missões 3 e 4 — reunião de 08/09/2026

**Data:** 08/09/2026
**Origem:** missões 3 e 4 da folha manuscrita da reunião com a orientadora de 08/09/2026
**Estado:** não iniciadas, aguardando a conclusão da missão 2
**Spec da missão em curso:** `2026-09-08-pipeline-tbr-svm-design.md`

Este documento existe para que a próxima sessão retome sem re-derivar o levantamento. Não é um
spec de implementação: é o estado apurado, as lacunas conhecidas e as decisões que faltam.

---

## Por que estas duas estão em espera

Decisão de 08/09/2026: **foco na missão 2**, com prazo de 15/09/2026 (próxima reunião semanal).
A missão 2 é a que a orientadora mais espera, é a que destrava o `eeg_transformer`, e uma
entregue bem vale mais que quatro pela metade.

**A missão 3 tem uma dependência real, não é só ordem de prioridade.** ICA é pré-processamento.
Sem o baseline da missão 2 medido, a pergunta "a AUC sobe com ICA?" não tem contra o que ser
respondida. Fazer o ICA antes obrigaria a refazer a medição depois.

**A missão 4 é revisão, não desenvolvimento**, e pode correr em paralelo assim que houver folga.

---

## Missão 3 — ICA no pré-processamento

> Transcrição da folha: *"Colocar opção de pré-processamento com ICA, juntamente ou sem o CAR."*

### Estado apurado

Não existe nenhum código de ICA no repositório. Verificado em 08/09/2026 por varredura de
`backend/**/*.py`: as únicas ocorrências da sigla são em `scripts/grafo.py`, onde "Implementar ICA"
aparece como **tarefa bloqueada**, e em `tests/test_painel_progresso.py`, num texto de fixture.

O `grafo.py` registra a dependência assim:

> `filtro passa-alta e notch no pipeline` → `Implementar ICA`
> *"As tarefas de ICA, referência configurável e rejeição de época ficam bloqueadas por este entregável"*

O bloqueio já foi levantado: o passa-alta e o notch estão implementados em
`scripts/preproc_basico.py` (`aplicar_passa_alta`, `aplicar_notch`), e a referência configurável
também (`aplicar_base`, `bases_disponiveis`). **A missão 3 está destravada do lado do código.**

### Onde entra

`scripts/preproc_basico.py` já tem a estrutura pronta para receber mais uma etapa: a função
`preprocessar(raw, l_freq, h_freq, freq_rede, car, base)` orquestra as etapas e devolve um
dicionário `decisoes` com o que foi aplicado. Uma etapa de ICA entra nesse mesmo desenho, com as
suas decisões registradas do mesmo jeito.

### Lacunas que impedem começar

A folha é curta e não decide nada disto. **Confirmar com a orientadora antes de implementar:**

1. **Qual variante de ICA.** Infomax (o padrão histórico em EEG), FastICA (o mais rápido) ou
   Picard (o recomendado pelo MNE hoje). A folha diz só "ICA".
2. **Quantos componentes.** Com 19 canais, o teto é 19. O uso comum é reter uma fração da
   variância, mas o número não está na folha.
3. **Como identificar o que remover.** ICA separa componentes; alguém precisa decidir quais são
   artefato (piscada, movimento ocular, ECG). Há métodos automáticos (`find_bads_eog`,
   ICLabel) e há inspeção manual. A folha não diz.
   ⚠️ **Este é o ponto de maior risco de vazamento metodológico:** escolher componentes olhando o
   resultado da classificação seria vazamento grosseiro. A decisão precisa ser automática e
   cega ao rótulo, ou manual e feita antes de qualquer classificação.
4. **O que "juntamente ou sem o CAR" decide.** Três leituras possíveis, e a folha não distingue:
   - oferecer as duas opções ao usuário do app (ICA com CAR, ICA sem CAR);
   - medir qual das duas dá melhor resultado;
   - ambas.

### Desenho provável, quando as lacunas fecharem

O experimento natural é 2×2, medido pela AUC do pipeline da missão 2:

|  | sem CAR | com CAR |
|---|---|---|
| **sem ICA** | linha de base da missão 2 | condição já usada hoje |
| **com ICA** | | |

Isso reaproveita `classificador.py` inteiro e só troca o pré-processamento. É o argumento mais
forte para a missão 2 vir primeiro: ela produz o instrumento de medida da missão 3.

⚠️ **Uma ressalva a registrar quando isto rodar:** o adhdata tem 19 canais. ICA com 19 canais é
pouco: a separação de fontes melhora com a contagem de eletrodos, e a literatura de ICA em EEG
costuma trabalhar com 32, 64 ou 128. O resultado no adhdata pode ser fraco por limitação do banco,
não por limitação do método, e essa distinção precisa sair escrita.

---

## Missão 4 — revisar escrita e gráficos da revisão de literatura

> Transcrição da folha (margem inferior, parcialmente cortada): *"Verificar exato do Revisão
> Literatura"*. Esclarecido por Murilo em 08/09/2026: verificar **a escrita e os gráficos** da
> revisão em andamento.

### Onde vive

Projeto separado: `C:\dev\literature_review_master_ITA`, branch `coleta-openalex`.
Estado apurado em `docs/ESTADO_2026-09-07.md`.

### Estado apurado (07/09/2026, do próprio projeto)

| Item | Estado |
|---|---|
| Corpus | 2 165 artigos, 9 combos completos, sem teto; busca de 07/09/2026 |
| Cobertura | DOI 97,6% · tópicos 97,8% · afiliação 85,0% · resumo 78,7% · quartil SJR 49,7% |
| Triagem E1-E6 | 328 incluídos · 589 excluídos · **1 248 aguardando confirmação manual** |
| Figuras | 6, a 300 dpi, em `relatorio/figuras/` |
| Subconjunto de síntese | 61 artigos (TDAH, incluídos, Q1+Q2) |
| Testes | 201, todos verdes |

### Escopo decidido em 08/09/2026: fazer as duas coisas

**(i) Revisar a escrita e os gráficos do que existe** e entregar a lista do que falta.

**(ii) Preparar o caminho dos 61 do subconjunto de síntese**, para reduzir o trabalho manual.

### O que trava a conclusão, e é trabalho manual de Murilo

Nenhum agente fecha estes quatro itens. Estão listados no próprio `ESTADO_2026-09-07.md` como
"o que depende do Murilo":

1. **`data/processed/triagem_pendente.csv` — 1 248 linhas** a revisar. Volume alto porque a
   carência de 24 meses mandou 886 artigos recentes para revisão manual.
2. **`data/processed/taxonomia_amostra.csv` — 30 linhas**, coluna `correto` a validar. Enquanto
   não voltar preenchida, a figura de método × condição sai estampada como pendente de validação.
3. **`relatorio/subconjunto_sintese.csv` — 61 artigos** a extrair: tabela de síntese do PRISMA
   item 17 e checklist de risco de viés.
4. **Decidir o merge** de `triagem-artigos` + `coleta-openalex` em `master` (o tronco aqui chama
   `master`, não `main`).

### A saída que reduz 1 248 para ~61

O próprio `ESTADO_2026-09-07.md` propõe: revisar só o que entra no subconjunto de síntese e
declarar o resto como pendente. O funil PRISMA passa a mostrar a etapa como **não realizada**, e
não como "zero incluídos", que é uma afirmação diferente e defensável.

Decisão de 08/09/2026: **preparar este caminho** como parte da missão 4. A decisão final de adotá-lo
é de Murilo, porque vai para o método da dissertação.

### O que dizer à orientadora, se ela perguntar se está pronto

As figuras e a escrita podem estar prontas. **A triagem manual não está**, e ela é etapa do PRISMA.
Dizer "a revisão está pronta" com 1 248 linhas pendentes seria afirmar mais do que o material
sustenta.

---

## Registrado, fora de escopo por ora

**Drive do HBN.** `murilomn58@gmail.com` tem acesso **somente leitura**. Decisão de 08/09/2026:
não usar nesta fase. Começar pelo adhdata, que já está em disco e não depende de DUA. Se em algum
momento for preciso trabalhar com esse material, **baixar e trabalhar offline** — nunca escrever
no Drive.

**A coluna direita cortada da folha.** A foto de 08/09/2026 corta uma segunda coluna com o que
parecem ser itens numerados. Pode conter missões. Refotografar a folha inteira se ainda existir.

**Componente periódico e aperiódico.** É a segunda metade da missão 2, não da 3 nem da 4. Está
fora do spec da missão 2 porque a folha não nomeia o método. Ver seção 1 daquele documento.
