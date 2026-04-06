# Diagnóstico da V2 — Por que o F1-ruim parou em 0.37

## Contexto

A V2 adicionou 19 features táticas (grupos G8–G12) ao pipeline original de 33 features posicionais, totalizando 52 features. A melhoria foi consistente mas modesta:

| Modelo | F1 (ruim) V1 | F1 (ruim) V2 | Delta |
|--------|-------------|-------------|-------|
| Decision Tree | 0.328 | 0.343 | +1.49 pp |
| Random Forest | 0.353 | 0.366 | +1.39 pp |

**Meta acadêmica: F1-ruim ≥ 0.50.** A V2 ficou longe deste objetivo.

---

## O que a V2 melhorou (e por quê)

A feature `contested_squares` (casas atacadas por ambos os lados) tornou-se a #1 em importância em ambos os modelos, com Cohen's d = 0.398 — a única feature a ultrapassar d > 0.30 em todo o projeto. Isso prova que **medir a tensão da posição tem valor preditivo**.

Porém, das 19 features táticas, apenas 6 entraram no top 15 do RF. As features de peças penduradas e cravadas ficaram com d < 0.15, abaixo do esperado.

---

## Causa raiz: as 52 features descrevem o "antes", não o "depois"

### A limitação fundamental

Todas as 52 features (V1 + V2) descrevem o **estado do tabuleiro antes do lance ser jogado**. Nenhuma feature captura **o que o lance faz ao tabuleiro**.

| O que temos (52 features) | O que falta |
|---|---|
| Material, mobilidade, tensão da posição **antes** | Como a posição **muda depois** do lance |
| Peças indefesas que **existem** no tabuleiro | Se o lance **cria ou resolve** peças indefesas |
| Ameaças que **já estão** presentes | Se o lance **ignora** ameaças existentes |

### Exemplo concreto

Dois lances diferentes na mesma posição teriam **exatamente as mesmas 52 features**, mas resultados opostos:

- Lance A: `Nf3` (defende e5, melhora a posição) → **bom**
- Lance B: `a3` (ignora ameaça em e5, perde material) → **ruim**

O modelo não consegue distingui-los porque ambos compartilham o mesmo snapshot pré-lance.

### Evidência quantitativa

As correlações point-biserial confirmam a limitação:

| Tipo de feature | Melhor correlação | Cohen's d |
|---|---|---|
| Posicionais V1 (33) | r = 0.084 (`is_capture`) | d = 0.24 |
| Táticas V2 (19) | r = 0.143 (`contested_squares`) | d = 0.398 |
| **Necessário para F1 ≥ 0.50** | **r > 0.20** | **d > 0.50** |

Mesmo a melhor feature V2 fica abaixo dos limiares necessários.

---

## O que NÃO é o problema

| Hipótese | Veredicto | Evidência |
|---|---|---|
| "Precisamos de mais dados" | **Descartada** | Learning curves V2 estáveis — mais dados não melhoram |
| "O modelo é fraco" | **Descartada** | DT e RF concordam; o sinal não existe nas features |
| "O desbalanceamento não foi tratado" | **Descartada** | `class_weight="balanced"` ativo; ratio 5.4:1 está tratado |
| "XGBoost/MLP resolveriam" | **Descartada** | Com features fracas (d < 0.40), ±2pp no máximo |
| "A rotulagem é má" | **Descartada** | Stockfish depth 15 é gold standard |
| "É problema de amostragem" | **Descartada** | 109K lances é volume adequado para 52 features |
| "É problema computacional" | **Descartada** | Extração leva ~9s; treino ~30s; não há limitação de recursos |

---

## Solução: features de look-ahead (posição depois do lance)

### Insight principal

A qualidade de um lance não depende apenas da posição — depende do **impacto do lance na posição**. Para capturar isso, precisamos:

1. **Avaliar o tabuleiro DEPOIS do lance** (fazer `board.push(move)`)
2. **Calcular deltas** (diferença entre features antes e depois)
3. **Avaliar a resposta do adversário** (o que ele pode fazer agora?)

### Por que isso funciona

Pesquisa publicada (ResearchSquare 2025) usando features de delta + complexidade atingiu **F1 = 0.75 e AUC = 0.82** num problema semelhante. O Stockfish 18 adicionou "Threat Inputs" (pares atacante-alvo) como feature da NNUE, ganhando +46 Elo — confirmando que ameaças dinâmicas são fundamentais.

### Impacto estimado

| Cenário | F1-ruim estimado | Justificativa |
|---|---|---|
| V2 (atual) | 0.37 | Features pré-lance apenas |
| V3 com deltas | 0.45–0.55 | Features antes + depois + delta |
| V3 + SEE | 0.50–0.60 | Adiciona avaliação de sequências de captura |

### Impacto no pipeline

**Não é necessário re-rodar a rotulagem Stockfish (etapa 03).** Os dados rotulados (`moves_labeled.csv`) e os FENs já contêm tudo o que é preciso. A V3 apenas adiciona lógica de extração de features — `board.push(move)` no `python-chess` é O(1).

Tempo estimado de extração V3: ~15-20s com 6 workers (vs ~9s da V2). Incremento trivial.

---

## Conclusão

A barreira da V2 é **estrutural e de informação**: as features descrevem o cenário pré-lance mas não o impacto do lance. A solução — features de look-ahead com deltas — é conceptualmente simples, computacionalmente barata, e suportada por pesquisa académica recente. É o passo natural para ultrapassar F1 ≥ 0.50.

Ver: [features-lookahead.md](../03-features/features-lookahead.md) para a especificação da V3.
