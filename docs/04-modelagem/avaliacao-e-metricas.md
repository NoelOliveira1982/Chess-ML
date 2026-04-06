# Avaliação e Métricas

## Métricas quantitativas

### Métricas obrigatórias

| Métrica | O que mede | Por que usar |
|---------|------------|--------------|
| **Accuracy** | Proporção de acertos totais | Visão geral simples |
| **Precision (por classe)** | Dos que o modelo disse "ruim", quantos realmente são | Evitar alarmes falsos |
| **Recall (por classe)** | Dos lances realmente "ruins", quantos o modelo detectou | Não deixar erros escaparem |
| **F1-score (por classe)** | Média harmônica de precision e recall | Métrica balanceada |
| **Matriz de confusão** | Distribuição de TP, TN, FP, FN | Visualização completa dos erros |

### Métricas opcionais

| Métrica | Quando usar |
|---------|-------------|
| **ROC-AUC** | Se o modelo produz probabilidades (DT, RF, MLP sim; KNN sim com `predict_proba`) |
| **Precision-Recall AUC** | Mais informativo que ROC quando há desbalanceamento |

### Qual classe é mais importante?

A classe **"ruim"** é a mais interessante do ponto de vista do domínio — o objetivo do classificador é **detectar erros**. Portanto, o **recall da classe "ruim"** merece atenção especial: um modelo que nunca detecta erros é inútil, mesmo que tenha accuracy alta.

---

## Implementação

```python
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["bom", "ruim"]))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["bom", "ruim"], cmap="Blues"
)
```

---

## Comparação entre modelos

Apresentar uma tabela comparativa no notebook:

| Modelo | Accuracy | F1 (bom) | F1 (ruim) | Recall (ruim) | AUC |
|--------|----------|----------|-----------|---------------|-----|
| Decision Tree | ... | ... | ... | ... | ... |
| Modelo 2 | ... | ... | ... | ... | ... |

Complementar com **gráfico de barras** comparando F1 por modelo.

---

## Interpretabilidade

### Árvore de Decisão — regras

Extrair as regras mais relevantes e traduzir para linguagem de xadrez:

```python
from sklearn.tree import export_text

rules = export_text(best_dt, feature_names=feature_names, max_depth=5)
```

Exemplo de tradução:
- "Se `material_diff <= -3.0` e `is_capture == 0` → **ruim**"
- Em xadrez: "O jogador está com 3+ pontos de material a menos e não está recuperando material neste lance — provavelmente perdeu peça sem compensação."

### Feature importance

```python
import matplotlib.pyplot as plt
import numpy as np

importances = best_dt.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.barh(
    [feature_names[i] for i in indices[:15]],
    importances[indices[:15]],
)
plt.xlabel("Importância")
plt.title("Top 15 features — Árvore de Decisão")
plt.gca().invert_yaxis()
plt.tight_layout()
```

Fazer o mesmo para Random Forest (se for o modelo 2) — comparar se as features mais importantes coincidem.

---

## Análise qualitativa de erros

Examinar manualmente **ao menos 5 exemplos** de cada tipo:

### Falsos positivos (modelo disse "ruim", mas o lance é bom)

Possíveis causas:
- Features não capturam compensação posicional (ex.: sacrifício correto).
- Posição complexa onde a avaliação do Stockfish é incerta.

### Falsos negativos (modelo disse "bom", mas o lance é ruim)

Possíveis causas:
- Erro tático sutil que as features de material/mobilidade não capturam.
- O lance parece "normal" pelas features, mas perde por razão específica (cravada, raio-x, etc.).

### Formato sugerido

Para cada exemplo comentado:

| Campo | Valor |
|-------|-------|
| FEN | `r1bqkb1r/pppppppp/2n5/...` |
| Lance jogado | `e4e5` |
| Delta (cp) | -210 |
| Label real | ruim |
| Predição | bom |
| Features dominantes | `material_diff=0`, `is_capture=False`, `king_pawn_shield=2` |
| Explicação | O lance parece inofensivo pelas features, mas perde cavalo por cravada na diagonal. O modelo não tem feature para detectar cravadas. |

---

## Curva de aprendizado (opcional)

Plotar accuracy/F1 no treino e validação em função do tamanho do dataset:

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train, cv=5,
    train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    scoring="f1",
)
```

Útil para responder: "precisamos de mais dados?" ou "o modelo está com overfitting?".
