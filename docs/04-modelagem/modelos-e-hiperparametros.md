# Modelos e Hiperparâmetros

## Estratégia geral

Treinar **ao menos dois modelos** e compará-los com as mesmas métricas sobre os mesmos splits de dados. O foco não é atingir a maior accuracy possível, mas demonstrar domínio do processo de ML: escolha de modelo, tuning, avaliação e interpretação.

---

## Modelo 1 — Árvore de Decisão (obrigatório)

### Por que

- Exigida pela disciplina (interpretabilidade).
- Permite extrair regras "se ... então ..." legíveis.
- Fácil de visualizar e relacionar com conceitos de xadrez.

### Implementação

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(
    max_depth=None,       # será tunado
    min_samples_leaf=5,   # evitar folhas com 1 exemplo
    class_weight="balanced",
    random_state=42,
)
```

### Hiperparâmetros a tunar

| Parâmetro | Faixa de busca | Método |
|-----------|----------------|--------|
| `max_depth` | 3, 5, 7, 10, 15, None | Grid search + validação |
| `min_samples_leaf` | 1, 5, 10, 20 | Grid search + validação |
| `criterion` | "gini", "entropy" | Grid search |

### Tuning

Usar `GridSearchCV` com `cv=5` e `scoring="f1"` (ou `f1_weighted`):

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth": [3, 5, 7, 10, 15, None],
    "min_samples_leaf": [1, 5, 10, 20],
    "criterion": ["gini", "entropy"],
}

grid = GridSearchCV(
    DecisionTreeClassifier(class_weight="balanced", random_state=42),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
grid.fit(X_train, y_train)
best_dt = grid.best_estimator_
```

### Interpretabilidade

Após treinar, extrair e comentar regras:

```python
from sklearn.tree import export_text, plot_tree

rules = export_text(best_dt, feature_names=feature_names, max_depth=5)
print(rules)
```

Traduzir as regras mais importantes para linguagem de xadrez:
- "Se `material_diff <= -3` e `is_capture == False` → **ruim**" → "Perder material sem compensação é erro."
- "Se `king_pawn_shield <= 1` e `opponent_center_control >= 3` → **ruim**" → "Rei exposto + adversário controlando o centro."

---

## Modelo 2 — Opções (escolher um)

### Opção A: Random Forest

**Vantagem:** ganho natural de performance sobre árvore única via ensemble.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
```

Hiperparâmetros a tunar:

| Parâmetro | Faixa |
|-----------|-------|
| `n_estimators` | 50, 100, 200 |
| `max_depth` | 5, 10, 15, None |
| `min_samples_leaf` | 1, 5, 10 |

### Opção B: KNN

**Vantagem:** conceitualmente simples; mostra outro paradigma (baseado em instância).

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
```

Hiperparâmetros a tunar:

| Parâmetro | Faixa |
|-----------|-------|
| `n_neighbors` | 3, 5, 7, 11, 15 |
| `weights` | "uniform", "distance" |
| `metric` | "euclidean", "manhattan" |

**Atenção:** KNN exige normalização das features. Usar `StandardScaler` antes:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier()),
])
```

### Opção C: MLP (Rede Neural simples)

**Vantagem:** conecta com aulas de redes neurais; pode capturar relações não-lineares.

```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=500,
    random_state=42,
)
```

Hiperparâmetros a tunar:

| Parâmetro | Faixa |
|-----------|-------|
| `hidden_layer_sizes` | (32,), (64, 32), (128, 64) |
| `learning_rate_init` | 0.001, 0.01 |
| `alpha` | 0.0001, 0.001, 0.01 |

**Atenção:** também exige `StandardScaler`.

---

## Split de dados

```
Dataset rotulado
  │
  ├─ 70% Treino ──────── usado para treinar e tunar (com cross-validation interna)
  ├─ 15% Validação ───── escolha final de modelo / threshold (opcional, pode usar CV)
  └─ 15% Teste ────────── avaliação final, tocado UMA VEZ
```

```python
from sklearn.model_selection import train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
)
# 0.176 de 85% ≈ 15% do total
```

**Estratificação** (`stratify=y`) garante que treino, validação e teste mantêm a mesma proporção de bom/ruim.

---

## Tratamento de desbalanceamento

Se a classe "ruim" for minoria significativa (< 30% do dataset):

1. **`class_weight="balanced"`** nos modelos que suportam (DT, RF) — ajusta o peso da loss automaticamente.
2. **Undersampling da maioria** — remover exemplos "bom" aleatoriamente até igualar.
3. **SMOTE** (Synthetic Minority Oversampling) — gerar exemplos sintéticos da classe "ruim".

Recomendação: começar com `class_weight="balanced"` (mais simples) e comparar com/sem. Documentar a escolha.
