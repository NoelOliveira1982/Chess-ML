# Diagnóstico da V1 — Por que o classificador tem F1-ruim de 0.35

## Contexto

A primeira versão do classificador (V1) foi concluída com o pipeline completo:
dados → rotulagem → 33 features posicionais → Decision Tree + Random Forest → avaliação.

Os resultados ficaram abaixo do esperado. Este documento analisa **por que**, com dados concretos, e serve de base para a iteração V2.

---

## Resultados da V1

| Modelo | Accuracy | F1 (ruim) | Recall (ruim) | Precision (ruim) | ROC-AUC |
|--------|----------|-----------|---------------|-------------------|---------|
| Decision Tree | 0.6176 | 0.3280 | 0.5967 | 0.2262 | 0.6487 |
| Random Forest | 0.6827 | 0.3525 | 0.5523 | 0.2589 | 0.6837 |

### O que esses números significam na prática

- **Recall 0.55 (RF):** o modelo detecta 55% dos lances ruins — pouco mais da metade.
- **Precision 0.26 (RF):** quando o modelo diz "ruim", acerta apenas 1 em 4 vezes.
- **F1 0.35:** reflexo do desequilíbrio entre precision e recall.
- **AUC 0.68:** melhor que aleatório (0.50), mas longe de útil (> 0.80).

---

## É comparável a aleatório?

**Não.** Mas a margem é pequena.

| Baseline | Accuracy | F1 (ruim) | Recall (ruim) | AUC |
|----------|----------|-----------|---------------|-----|
| Sempre "bom" | 84.4% | 0.00 | 0.00% | 0.50 |
| Aleatório proporcional | ~74% | ~0.16 | ~15.6% | 0.50 |
| **Nosso RF** | **68.3%** | **0.35** | **55.2%** | **0.68** |

O modelo **é melhor que aleatório** — detecta 55% dos erros vs. 15% do aleatório, e tem AUC 0.68 vs. 0.50. Mas a precision é tão baixa que o ganho prático é limitado: para cada erro real detectado, há 3 falsos alarmes.

---

## Causa raiz: as features não separam as classes

### Evidência 1 — Correlações quase nulas

A correlação (point-biserial) entre cada feature e o label é negligível. As mais fortes:

| Feature | Correlação com label | Interpretação |
|---------|---------------------|---------------|
| `is_capture` | r = −0.084 | Capturas são ligeiramente mais "boas" |
| `legal_moves_opponent` | r = +0.080 | Mais opções do adversário → ligeiramente mais "ruim" |
| `legal_moves_player` | r = +0.070 | Idem para o jogador |
| `black_queens` | r = +0.067 | Presença de damas → ligeiramente mais "ruim" |
| `move_number` | r = +0.054 | Lances mais tardios → ligeiramente mais "ruim" |

**Nenhuma feature atinge r = 0.10.** Para um preditor ser realmente útil, esperaríamos |r| > 0.20 para as melhores features.

### Evidência 2 — Cohen's d (separabilidade) negligível

O Cohen's d mede quão separadas as distribuições de cada feature são entre as duas classes:
- d < 0.2: negligível
- d 0.2–0.5: pequeno
- d 0.5–0.8: médio
- d > 0.8: grande

| Feature | Cohen's d | Média (bom) | Média (ruim) |
|---------|-----------|-------------|--------------|
| `is_capture` | 0.24 | 0.32 | 0.21 |
| `legal_moves_opponent` | 0.23 | 30.8 | 33.6 |
| `legal_moves_player` | 0.20 | 30.7 | 32.9 |
| `black_queens` | 0.19 | 0.69 | 0.77 |
| `white_queens` | 0.19 | 0.69 | 0.77 |
| Todas as outras 28 features | < 0.19 | — | — |

**A melhor feature (is_capture) tem efeito "pequeno".** Todas as outras 32 features têm efeito negligível. Nenhuma feature sequer se aproxima de efeito "médio" (d ≥ 0.5).

### Evidência 3 — Probabilidades completamente sobrepostas

O Random Forest atribui probabilidades de "ruim" quase iguais para as duas classes:

| Percentil | P(ruim) — lances BOM | P(ruim) — lances RUIM |
|-----------|----------------------|-----------------------|
| P10 | 0.286 | 0.372 |
| P25 | 0.358 | 0.436 |
| P50 (mediana) | 0.434 | 0.514 |
| P75 | 0.518 | 0.606 |
| P90 | 0.601 | 0.664 |

A separação entre medianas é de **apenas 0.08**. Os P25–P75 sobrepõem-se quase totalmente. O modelo praticamente não consegue distinguir as classes.

### Evidência 4 — Otimização de threshold não ajuda

O threshold ótimo para F1 no validation set é... **0.50** (exatamente o default). Não há margem para melhorar por ajuste de threshold.

### Evidência 5 — Learning curves estáveis

As curvas de aprendizado (F1 treino vs. validação em função do tamanho do dataset) mostram que:
- O gap é estável — não há overfitting significativo.
- A curva de validação está plana — **mais dados com as mesmas features não melhorariam o resultado**.

---

## Por que as features falham

### O que as features capturam (o "cenário")

As 33 features V1 descrevem o **ambiente** da posição:
- Quantas peças cada lado tem (material)
- Quanta liberdade de movimento cada lado tem (mobilidade)
- Se os reis estão seguros (roque, escudo de peões)
- Como estão os peões (dobrados, isolados, passados)
- Quem controla o centro
- Que tipo de lance foi jogado (captura? xeque?)
- Em que fase do jogo estamos

### O que as features NÃO capturam (as "ameaças")

Um lance ruim na faixa 1400–1700 Lichess tipicamente acontece por uma **razão tática concreta**:

| Causa do erro | Exemplo | Feature que capturaria |
|---------------|---------|------------------------|
| Peça **pendurada** (indefesa) | Cavalo em e5 não tem nenhuma peça a defendê-lo | `hanging_pieces_player` |
| **Cravada** (pin) | Bispo crava o cavalo contra a dama | `pinned_pieces_player` |
| **Garfo** (fork) | Adversário pode jogar Nc7 atacando rei + torre | `opponent_fork_threats` |
| **Captura grátis** | Torre adversária pode comer peão sem consequência | `opponent_capture_value` |
| **Raio-x** (skewer) | Torre adversária alinha rei + dama | `skewer_threats` |
| **Retaguarda** exposta | Rei na 1ª fila sem peças a defender | `back_rank_vulnerability` |

**Nenhuma destas causas é capturada pelas 33 features da V1.**

### Exemplo concreto

Falso negativo real (DT): lance `g3` na posição `r1b1k1nr/1p1p1ppp/pq2p3/8/3nP3/PPb5/R2NNPPP/2BQKB1R`

- **Delta:** −10,443 cp (desastre completo — perde a dama)
- **Features da V1:** mobilidade 20 (normal), lance 10 (abertura), sem captura
- **Causa real:** o bispo em c3 está a fazer raio-x no cavalo e2 contra a dama d1, e g3 ignora completamente esta ameaça
- **O modelo não tem como saber.** As features dizem "posição normal no lance 10" — e o lance é catastrófico.

---

## O que NÃO é o problema

| Hipótese descartada | Evidência |
|---------------------|-----------|
| "Precisamos de mais dados" | Learning curves estáveis — mais dados não ajudariam |
| "O modelo é fraco" | RF e DT concordam nos resultados; o sinal não existe nas features |
| "O threshold está errado" | Threshold ótimo = 0.50 (default), sem margem |
| "O desbalanceamento não foi tratado" | `class_weight="balanced"` já está ativo |
| "A base de dados é ruim" | 109K lances é volume adequado; rótulos Stockfish depth 15 são confiáveis |
| "XGBoost resolveria" | Com features fracas (d < 0.25), nenhum modelo faria milagre |

---

## Conclusão

O problema é **estrutural**: as features descrevem o cenário da posição mas não as ameaças táticas que causam erros. Para obter melhoria significativa (F1-ruim > 0.50), é necessário adicionar **features táticas** que capturem peças indefesas, cravadas, garfos e outras ameaças concretas.

Ver: [features-taticas.md](../03-features/features-taticas.md) para a especificação da V2.
