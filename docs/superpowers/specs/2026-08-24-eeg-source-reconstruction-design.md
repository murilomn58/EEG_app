# EEG 3D Source Reconstruction (MNE-Python) — Design

Data: 2026-08-24
Status: aprovado pelo usuário em brainstorming, pronto para plano de implementação.

## Contexto

O dashboard atual (`eeg-cerebro-3d.html`) mostra 19 eletrodos sobre um mesh
genérico (`assets/brain.obj`), coloridos por potência de banda medida no
próprio eletrodo (atividade de **sensor**, não de fonte neural). O objetivo
deste trabalho é adicionar uma segunda visualização que reconstrói a
atividade **cortical** (fonte) a partir do sinal EEG, usando MNE-Python
(forward model + inverse solution sobre o template fsaverage), e mapeia o
resultado para uma malha 3D renderizada com Three.js — igual ao que já
existe para os sensores, mas geometricamente correta em vez de
interpolação IDW.

Dado de entrada real disponível: `adhdata.csv` (já no projeto), dataset
público de EEG ADHD/Controle — 121 sujeitos, 19 canais nomeados
`Fp1,Fp2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T7,T8,P7,P8,Fz,Cz,Pz`, amostrado a
128 Hz (taxa padrão conhecida desse dataset).

## Trabalho relacionado já entregue (fora do escopo desta spec)

Antes desta spec, foi implementado um toggle "Cz / Average" na aba de
sensores existente, que re-referencia o sinal (CAR — Common Average
Reference) antes de plotar/colorir. Isso é reaproveitado conceitualmente
aqui: o backend de source reconstruction também aplica average reference
antes de qualquer cálculo do MNE — é um requisito do método, não só
didático.

## Decisões de escopo

- **v1 = snapshot estático.** Usuário escolhe um sujeito do CSV e um
  instante de tempo, clica em "Rodar reconstrução", vê o resultado uma
  vez. Streaming em tempo real e reconstrução por banda de frequência
  ficam fora desta spec (trabalho futuro).
- **Malha do source reconstruction = fsaverage do MNE**, não o
  `brain.obj` atual. Garante correspondência exata vértice-a-vértice
  entre a saída do MNE e a malha renderizada, sem inventar um registro
  anatômico aproximado. O `brain.obj` continua sendo usado, sem mudança,
  na aba de sensores.
- **Montagem de 19 canais é unificada** em todo o app para bater com o
  CSV real: `Fp1,Fp2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T7,T8,P7,P8,Fz,Cz,Pz`.
  Isso substitui a lista antiga (`T3/T4/T5/T6`, `Oz`, sem `Cz`) usada pela
  aba de sensores/sinal sintético.

## Arquitetura

```
adhdata.csv (subject_id, t_start, t_end)
        │
        ▼
FastAPI (backend/) ── na inicialização:
        │              fetch_fsaverage() [download único, cacheado]
        │              standard_1020 montage → 19 canais
        │              src (superfície decimada ico4/oct5)
        │              forward = make_forward_solution(src, bem fsaverage, montage)
        │              noise_cov = make_ad_hoc_cov(info)
        │              inverse_operator = make_inverse_operator(...)   [calculado 1x, fica em memória]
        │
        ▼  (por request, rápido: só álgebra linear sobre o operador pronto)
   RawArray(janela do CSV) → set_eeg_reference('average') → apply_inverse_raw(..., method="dSPM")
        │
        ▼
   JSON { values[], n_vertices, time, method }
        │
        ▼
Three.js (eeg-cerebro-3d.html, aba "Fontes corticais")
        │
        ▼
malha fsaverage (assets/fsaverage-cortex.obj, exportada 1x offline) colorida por vértice
```

## Componentes

### 1. `backend/` — FastAPI + MNE-Python

- **Setup no boot do servidor** (não por request):
  - `mne.datasets.fetch_fsaverage()` — modelo template: superfícies
    corticais, BEM 3-camadas já resolvido, transformação MRI↔cabeça.
  - Monta `standard_1020` montage do MNE, seleciona os 19 canais da lista
    unificada.
  - Cria espaço de fontes (`src`) decimado (ico4 ou oct5 — meta:
    5.000–8.000 vértices totais, leve pro browser).
  - `mne.make_forward_solution(info, trans='fsaverage', src, bem)` →
    forward operator.
  - `mne.make_ad_hoc_cov(info)` — sem gravação de ruído real disponível,
    usa a covariância padrão recomendada pelo MNE nesse caso.
  - `mne.minimum_norm.make_inverse_operator(info, forward, noise_cov,
    loose=0.2, depth=0.8)` → inverse operator, mantido em memória para
    todos os requests seguintes.
  - Loga no console: número de vértices do `src`, número de canais
    usados — smoke test manual de que o setup funcionou.

- **`GET /subjects`** — lê `adhdata.csv` uma vez no boot (ou lazy no
  primeiro request), devolve lista `[{id, classe, duracao_s}]` a partir
  dos IDs únicos e contagem de linhas / 128 Hz.

- **`POST /source-localization`** — body
  `{subject_id, t_start, t_end, method}` (method default `"dSPM"`):
  1. Filtra as linhas daquele `subject_id` no intervalo `[t_start,
     t_end)` (índices de linha = tempo × 128 Hz).
  2. Monta `mne.io.RawArray` com esses dados + `info` fixo (19 canais,
     `sfreq=128`, `standard_1020` montage).
  3. `raw.set_eeg_reference('average', projection=True)`.
  4. `mne.minimum_norm.apply_inverse_raw(raw, inverse_operator,
     lambda2=1/9, method=method)`.
  5. Extrai os valores do `SourceEstimate` (média ou último instante da
     janela — a decidir no plano de implementação, não crítico pro
     design).
  6. Responde `{values: [...], n_vertices: N, time, method}` — `values`
     na MESMA ordem de vértices da malha exportada (seção 2).
  7. Erros (sujeito inexistente, janela fora do intervalo gravado) →
     HTTP 4xx com mensagem clara, não 500 genérico.

- **CORS**: `CORSMiddleware` liberando a origem do servidor estático do
  frontend (ex: `http://localhost:8000` do `python -m http.server`).

### 2. `export_fsaverage_mesh.py` — script offline, roda uma vez

Exporta a MESMA superfície `src` usada no forward model (hemisfério
esquerdo + direito combinados) para `assets/fsaverage-cortex.obj`,
centralizada na origem do mesmo jeito que o `brain.obj` atual (facilita
reuso de câmera/controles). A ordem dos vértices nesse arquivo é a
mesma ordem em que o backend devolve `values[]` — é o contrato entre
backend e frontend, documentado como comentário no topo do script.

### 3. Frontend (`eeg-cerebro-3d.html`)

**Unificação de montagem** (afeta a aba de sensores já existente):
- `CANAIS_19` → `["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T7","T8","P7","P8","Fz","Cz","Pz"]`.
- `POS_10_20`: renomeia T3→T7, T4→T8, T5→P7, T6→P8 (mesmas posições
  angulares); remove `Oz`; adiciona `Cz: [0, 0]`.
- `loboDe()` não muda — a classificação por prefixo de nome já cobre os
  nomes novos corretamente (T7/T8 → temporal, P7/P8 → parietal).
- Resto da lógica da aba de sensores (toggle Cz/Average, filtros de
  banda, sinal sintético, `window.setEEGData`) fica intacto.

**Nova aba "Fontes corticais (MNE)"** — dois botões no HUD ao lado do
título 3D, alternando entre "Sensores" (existente) e a nova aba. Reusa
o MESMO renderer/câmera/cena Three.js (não duplica), trocando apenas o
que fica visível:
- Esconde `malhaCerebro` + `grupoEletrodos`; mostra `malhaFontes`
  (carrega `assets/fsaverage-cortex.obj` sob demanda no primeiro clique
  na aba, com a mesma UI de erro de carregamento já usada pro
  `brain.obj`).
- Troca a barra de controles inferior: sai `#banda` + `#referencia`,
  entra **seletor de sujeito** (populado por `GET /subjects`), **slider
  de tempo** (0 até a duração do sujeito escolhido) e botão **"Rodar
  reconstrução"**.
- Ao clicar "Rodar": `POST /source-localization`, spinner enquanto
  espera (pode levar alguns segundos), depois colore os vértices de
  `malhaFontes` usando a mesma rampa `corPorAmplitude` já existente,
  normalizando os valores dSPM retornados pelo min/max da própria
  resposta (escala diferente da amplitude sintética 0–1 usada na aba de
  sensores).
- HUD fixo nessa aba com aviso: *"19 canais é pouco para localizar fontes
  com precisão — isso é uma estimativa aproximada, não a origem exata do
  sinal"* + *"fsaverage é um cérebro template, não a anatomia real do
  sujeito"* + método/sujeito/janela usados.

## Tratamento de erro

- Backend fora do ar → mensagem na aba: *"Backend Python não respondeu
  em `http://localhost:8000`. Rode `uvicorn app:app --reload` dentro de
  `backend/`."*
- Falha ao carregar `fsaverage-cortex.obj` → mesmo padrão de erro do
  `brain.obj`.
- Erro do MNE no request → backend responde 4xx com mensagem; frontend
  mostra no HUD em vez de deixar o spinner girando pra sempre.

## Testes (manuais — protótipo local, sem CI)

- Smoke test no boot do backend: log com número de vértices/canais do
  forward operator.
- `curl` manual em `/subjects` e `/source-localization`, conferindo que
  `n_vertices` bate com a contagem de vértices de
  `assets/fsaverage-cortex.obj`.
- Frontend: abrir no navegador, trocar de aba, escolher sujeito, rodar
  reconstrução, checar visualmente que a malha muda de cor; derrubar o
  backend de propósito e checar que a mensagem de erro aparece em vez de
  travar.

## Fora do escopo desta spec (trabalho futuro)

- Streaming em tempo real (janela deslizante + POST periódico).
- Reconstrução por banda de frequência (delta/theta/alpha/beta/gamma
  aplicada à fonte, não só ao sensor).
- Comparação multi-sujeito / multi-instante lado a lado.
- Registro anatômico exato entre fsaverage e o `brain.obj` atual (ficam
  como duas visualizações separadas, não uma sobreposição).

## Nota sobre versionamento

Este diretório de projeto não é um repositório git (`git init` não foi
rodado). Esta spec fica salva em disco normalmente; não há commit
associado.
