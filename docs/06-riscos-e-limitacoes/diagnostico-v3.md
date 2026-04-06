# Evolução do Classificador — V1 → V2 → V3

## Objetivo deste documento

Documentar a **narrativa de evolução** do classificador de lances de xadrez ao longo de três iterações, explicando a motivação de cada upgrade, o que foi aprendido com cada diagnóstico, e por que a V3 é o passo natural para atingir a meta de F1 ≥ 0.50.

Este documento serve como material de apresentação do processo científico do projecto.

---

## A jornada em três actos

```
V1 (33 features)        V2 (52 features)         V3 (67 features)
"O cenário"              "As ameaças"              "O impacto"
    │                        │                         │
    ▼                        ▼                         ▼
Descreve a posição →    Detecta perigos →         Avalia o que o lance
material, mobilidade,   peças indefesas,          FAZ à posição:
estrutura de peões,     cravadas, tensão,         deltas, respostas do
controle do centro      ataques ao rei            adversário, trocas
    │                        │                         │
    ▼                        ▼                         ▼
F1 = 0.35                F1 = 0.37                 Meta: F1 ≥ 0.50
AUC = 0.68               AUC = 0.71                Meta: AUC ≥ 0.75
```

---

## Acto I — V1: descrevendo o cenário

### O que fizemos

Construímos o pipeline completo de classificação supervisionada:
- 3.000 partidas do Lichess (rating 1400–1700, equivalente a 1200–1500 Chess.com)
- 109.290 lances rotulados via Stockfish depth 15 (bom/ruim, zona cinzenta descartada)
- 33 features em 7 grupos: material, mobilidade, segurança do rei, estrutura de peões, controle do centro, características do lance, contexto do jogo
- Decision Tree + Random Forest com `class_weight="balanced"` e GridSearchCV

### Resultados

| Modelo | F1 (ruim) | AUC | Recall (ruim) |
|--------|-----------|-----|---------------|
| Decision Tree | 0.328 | 0.649 | 59.7% |
| Random Forest | 0.353 | 0.684 | 55.2% |

### O que aprendemos

O modelo funciona — é claramente melhor que aleatório (AUC 0.68 vs 0.50, recall 55% vs 15%). Mas o F1 é baixo porque a **precision** é muito baixa (0.26): para cada lance ruim corretamente detectado, há 3 falsos alarmes.

### Diagnóstico (por que não basta)

A análise quantitativa revelou a causa raiz:

1. **Correlações negligíveis** — nenhuma feature atinge r = 0.10 com o label
2. **Cohen's d negligível** — a melhor feature (`is_capture`) tem d = 0.24 (efeito "pequeno")
3. **Learning curves estáveis** — mais dados não ajudariam
4. **Threshold ótimo = 0.50** — não há margem para ajuste

**Conclusão:** as features descrevem o *cenário* da posição (quantas peças, quem controla o centro), mas não capturam as *ameaças táticas* que causam erros reais em jogadores de 1400–1700.

---

## Acto II — V2: detectando ameaças

### Motivação

O diagnóstico V1 apontou uma direção clara: adicionar features que detectem **perigos concretos** no tabuleiro — peças indefesas, cravadas, ameaças de captura.

### O que fizemos

19 features táticas em 5 novos grupos:
- **G8** — Peças indefesas (`hanging_pieces`, `hanging_value`)
- **G9** — Capturas com ganho material (`threats_against`, `max_threat_value`)
- **G10** — Cravadas absolutas (`pinned_pieces`)
- **G11** — Segurança do rei avançada (`king_attackers`, `king_escape_squares`)
- **G12** — Tensão e complexidade (`contested_squares`, `total_attacks`)

### Resultados

| Modelo | F1 V1 → V2 | AUC V1 → V2 | Delta F1 | Delta AUC |
|--------|-----------|------------|----------|-----------|
| Decision Tree | 0.328 → 0.343 | 0.649 → 0.674 | +1.49 pp | +2.57 pp |
| Random Forest | 0.353 → 0.366 | 0.684 → 0.708 | +1.39 pp | +2.42 pp |

### O que aprendemos

A melhoria é **consistente mas modesta**. Três insights importantes:

1. **`contested_squares` é a feature #1** em ambos os modelos (importância 0.114 no RF) — medir tensão no tabuleiro tem valor real
2. **Cohen's d = 0.398** para `contested_squares` — a primeira feature a ultrapassar d > 0.30 em todo o projecto
3. **Features de hanging/pins ficaram fracas** (d < 0.15) — saber que existem peças indefesas no tabuleiro não distingue lances bons de maus

### Diagnóstico (por que +1.4pp não basta)

O insight crucial da V2:

> **Dois lances diferentes na mesma posição teriam exatamente as mesmas 52 features.**

Se a posição tem 2 peças penduradas e 1 cravada, essas features são iguais quer o jogador jogue `Nf3` (salvando a peça) ou `a3` (ignorando tudo). O modelo não consegue distinguir porque ambos compartilham o mesmo snapshot **pré-lance**.

As features V2 melhoraram a descrição do cenário (agora sabemos que há perigos), mas não dizem se o lance jogado **lida com esses perigos ou os ignora**.

**Conclusão:** o teto de performance está no tipo de informação — precisamos de features que avaliem o **impacto do lance**, não apenas o estado antes dele.

---

## Acto III — V3: avaliando o impacto

### Motivação

A evolução do pensamento ao longo do projecto segue uma lógica natural:

| Versão | Pergunta que responde | Analogia médica |
|--------|----------------------|-----------------|
| V1 | "Como é o paciente?" | Exame físico geral |
| V2 | "Que doenças tem?" | Exames de diagnóstico |
| **V3** | **"O tratamento melhorou ou piorou?"** | **Comparar antes e depois** |

A V3 responde à pergunta fundamental: **o que acontece ao tabuleiro depois deste lance?**

### O que vamos fazer

15 features em 3 novos grupos:

**G13 — Deltas (antes vs depois):** Para cada feature relevante, calcular o valor **depois** do lance e a diferença. Exemplo: se o jogador tinha 0 peças penduradas antes e 2 depois do lance, `delta_hanging_player = +2` — sinal forte de lance ruim.

**G14 — Resposta do adversário:** O que o adversário pode fazer imediatamente após o lance? Se pode capturar uma peça de graça ou dar xeque, o lance é suspeito.

**G15 — Static Exchange Evaluation (SEE):** Simulação de sequências de captura num quadrado — "se todos trocarem peças ali, quem ganha material?". Técnica usada por **todas** as engines de xadrez desde os anos 80.

### Por que estas features devem funcionar

Três fontes de evidência:

**1. Lógica do domínio**

Um lance ruim tipicamente:
- Deixa peças indefesas que antes estavam protegidas (`delta_hanging_player > 0`)
- Permite ao adversário capturas favoráveis (`opponent_best_capture_value > 0`)
- Move uma peça para uma casa sem defensores (`created_hanging_self = 1`)

Estas são exatamente as features do G13–G15. Pela primeira vez, o modelo pode distinguir o lance `Nf3` (delta = 0, nenhuma peça fica exposta) do lance `a3` (delta = +2, duas peças ficam penduradas).

**2. Referências académicas**

| Referência | Abordagem | Resultado |
|---|---|---|
| ResearchSquare 2025 | Features de delta de avaliação + complexidade | F1 = 0.75, AUC = 0.82 |
| Stockfish 18 (SFNNv10) | "Threat Inputs" — pares atacante-alvo dinâmicos | +46 Elo sobre v17 |
| Maia Chess (Microsoft) | Predição de lances humanos com posição antes e depois | 53% acerto na jogada exata |

**3. Evidência interna do projecto**

A feature mais forte da V2 (`contested_squares`, d = 0.398) mede tensão *estática*. A versão *delta* (`delta_contested_squares`) captura se o lance *aumenta ou diminui* essa tensão — informação mais rica. Esperamos Cohen's d > 0.50 para as melhores features V3.

### Custo computacional

| Aspecto | Impacto |
|---|---|
| Re-rotulagem Stockfish (etapa 03, ~174 min) | **NÃO necessária** |
| Re-coleta de dados | **NÃO necessária** |
| Extração de features | ~20-25s (vs ~9s na V2) |
| Treino dos modelos | ~35s |
| Pipeline completo V3 | **~60 segundos** |

A V3 é computacionalmente barata porque usa apenas operações do `python-chess` (`board.push(move)` e recalculação de features) — não invoca engines externas.

---

## Evolução das métricas — visão consolidada

| Métrica | Baseline (aleatório) | V1 | V2 | V3 (meta) |
|---------|---------------------|----|----|-----------|
| **F1 (ruim)** | ~0.16 | 0.35 | 0.37 | **≥ 0.50** |
| **AUC** | 0.50 | 0.68 | 0.71 | **≥ 0.75** |
| **Recall (ruim)** | ~15.6% | 55.2% | 57.1% | **≥ 60%** |
| **Precision (ruim)** | ~15.6% | 25.9% | 27.0% | **≥ 40%** |
| Nº features | — | 33 | 52 | 67 |
| Tipo de informação | — | Cenário | + Ameaças | + Impacto |

### Sobre a meta de F1 ≥ 0.50

Um F1 de 0.50 na classe minoritária ("ruim", 15.6% do dataset) é um resultado **sólido** no contexto académico:

- Está **3× acima** do baseline aleatório proporcional (F1 ~0.16)
- Demonstra que o modelo aprendeu padrões reais, não ruído
- Pesquisas publicadas na área com features semelhantes atingem F1 0.50–0.75, posicionando o nosso resultado na faixa inferior-média da literatura
- Para a disciplina, o **processo** (diagnóstico → melhoria iterativa) é tão valioso quanto o número final

---

## Evolução do poder preditivo das features

| Versão | Melhor correlação (r) | Melhor Cohen's d | Interpretação |
|--------|----------------------|-----------------|---------------|
| V1 | 0.084 (`is_capture`) | 0.24 | Negligível — features não separam classes |
| V2 | 0.143 (`contested_squares`) | 0.398 | Pequeno — primeira feature com sinal real |
| V3 (esperado) | > 0.20 | > 0.50 | **Médio** — features capturam causa dos erros |

A progressão mostra que cada iteração não é aleatória — é guiada por diagnóstico quantitativo que identifica a barreira e propõe a solução.

---

## O ciclo científico do projecto

```
    ┌──────────────┐
    │   Treinar    │
    │   modelos    │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   Avaliar    │ ← F1, AUC, Recall, Precision
    │   métricas   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Diagnosticar │ ← Correlações, Cohen's d, learning curves
    │  barreiras   │   análise de erros (FP/FN)
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  Projetar    │ ← Novas features baseadas na causa raiz
    │  solução     │
    └──────┬───────┘
           │
           └──────────── volta ao início
```

Este ciclo foi executado **três vezes**:

| Iteração | Diagnóstico | Solução |
|----------|------------|---------|
| V1 → V2 | Features descrevem cenário, não ameaças | Adicionar features táticas (G8–G12) |
| V2 → V3 | Features descrevem ameaças estáticas, não impacto do lance | Adicionar features de look-ahead (G13–G15) |

A capacidade de **identificar problemas com dados** e **projetar soluções informadas** é o objectivo central da disciplina de Paradigmas de Aprendizagem de Máquina.

---

## Limitações conhecidas e trabalhos futuros

### O que a V3 provavelmente NÃO resolve

- **Combinações táticas de 2+ lances** (ex.: sacrifício seguido de mate em 3) — o look-ahead de 1 ply não vê isso
- **Avaliação posicional profunda** (ex.: peão fraco que só será explorado 20 lances depois) — requer conhecimento de engine
- **Variabilidade entre jogadores** — o mesmo lance pode ser "bom para um 1400" e "ruim para um 1700"

### Possíveis extensões (V4+)

| Extensão | Impacto esperado | Viabilidade |
|----------|-----------------|-------------|
| Stockfish depth 1-3 como feature | F1 ~0.65-0.75 | Requer validação de circularidade com o professor |
| Look-ahead de 2 plies | F1 ~0.55-0.65 | Custo computacional cresce quadraticamente |
| Gradient Boosting (XGBoost/LightGBM) | +2-5pp sobre RF | Fácil de implementar, menor interpretabilidade |
| CNN em representação bitboard 8×8 | F1 ~0.60-0.70 | Mudança de paradigma; fora do escopo de DT/RF |

---

## Resumo para apresentação

> O projecto demonstra o ciclo completo de ML aplicado: **dados → features → modelo → avaliação → diagnóstico → melhoria**. Em três iterações, evoluímos de features que descrevem o *cenário* (V1), para features que detectam *ameaças* (V2), para features que medem o *impacto do lance* (V3). Cada iteração foi motivada por diagnóstico quantitativo — correlações, Cohen's d, análise de erros — e resultou em melhoria mensurável. A meta de F1 ≥ 0.50 é sustentada por referências académicas e pela progressão observada de d = 0.24 → 0.40 → 0.50+ nas features.
