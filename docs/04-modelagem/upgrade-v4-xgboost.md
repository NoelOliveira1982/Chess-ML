# Upgrade V4 — XGBoost + Threshold Tuning

## Motivação

### Diagnóstico V3: o teto agora está no modelo, não nas features

A V3 foi o maior salto do projeto (+6.48pp F1-ruim, +6.0pp AUC). Mas o diagnóstico revela que o gap restante (0.43 → 0.50) **não se resolve com mais features**:

| Evidência | Implicação |
|---|---|
| Learning curves V3 estáveis (treino ~0.55, val ~0.43) | Sem overfitting — mais dados não ajudam |
| Top features V3 com r < 0.15, d < 0.40 | Features informativas mas não decisivas individualmente |
| FN típicos: combinações de 2+ lances | Look-ahead de 1 ply é limitação fundamental |
| RF V3 depth ótimo subiu de 10→15 | Modelo mais profundo explora melhor as features |
| Precision-ruim 0.37, recall-ruim 0.52 | Trade-off desequilibrado — threshold default (0.50) não é ótimo |

**Conclusão:** as 67 features contêm mais informação do que DT/RF conseguem extrair. O gap é de **capacidade do modelo**, não de qualidade das features.

### Evolução do projeto: features → modelo

| Versão | Mudança | Tipo |
|---|---|---|
| V1 → V2 | +19 features táticas | Engenharia de features |
| V2 → V3 | +15 features look-ahead | Engenharia de features |
| **V3 → V4** | **XGBoost + threshold tuning** | **Upgrade de modelo** |

Esta evolução é natural: primeiro melhoramos os dados (V1→V2→V3), agora melhoramos o algoritmo que aprende deles.

---

## O que é XGBoost

**eXtreme Gradient Boosting** é um algoritmo de ensemble que constrói árvores de decisão **sequencialmente** — cada nova árvore corrige os erros das anteriores.

### Diferença conceptual face ao Random Forest

| Aspecto | Random Forest | XGBoost |
|---|---|---|
| Estratégia | Árvores independentes, média dos votos | Árvores sequenciais, cada uma corrige a anterior |
| Tratamento de erros | Todos os exemplos têm peso igual | Exemplos mal classificados recebem mais peso |
| Regularização | Via profundidade e min_samples | L1/L2 nos pesos das folhas + shrinkage |
| Interações entre features | Captura combinações limitadas | Captura interações mais complexas |
| Desbalanceamento | `class_weight="balanced"` | `scale_pos_weight` + controlo fino |

### Por que XGBoost deve melhorar

1. **Boosting corrige erros iterativamente** — os FN da V3 (lances ruins não detectados) receberão mais atenção nas árvores seguintes
2. **Melhor gestão do desbalanceamento 5.4:1** — `scale_pos_weight` ajusta o peso da classe "ruim" directamente na loss function
3. **Regularização nativa** — evita overfitting mesmo com árvores mais complexas
4. **Referências da literatura** — XGBoost/LightGBM tipicamente supera RF em +2–5pp para problemas tabulares com desbalanceamento

---

## Threshold Tuning

### O problema

O RF V3 usa threshold = 0.50 por padrão. Mas a curva Precision-Recall mostra que existem thresholds melhores:

- Precision-ruim = 0.37, Recall-ruim = 0.52 → F1 = 0.43
- Baixar o threshold para ~0.40 pode aumentar o recall sem destruir a precision

### Implementação

Usar o conjunto de **validação** (15% dos dados) para encontrar o threshold que maximiza F1-ruim:

```python
from sklearn.metrics import f1_score
import numpy as np

y_proba_val = model.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.20, 0.65, 0.01)
best_threshold = 0.50
best_f1 = 0.0

for t in thresholds:
    y_pred_t = (y_proba_val >= t).astype(int)
    f1 = f1_score(y_val, y_pred_t, pos_label=1)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t
```

O threshold ótimo é aplicado apenas na predição final — o treino do modelo não muda.

### Aplicação

- Otimizar threshold para **todos os modelos** (DT, RF, XGB)
- Reportar métricas com threshold default (0.50) **e** com threshold ótimo
- O threshold ótimo é encontrado no validation set e avaliado no test set

---

## Hiperparâmetros XGBoost

### Grid de busca

```python
from xgboost import XGBClassifier

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_child_weight": [1, 5, 10],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}

xgb = XGBClassifier(
    scale_pos_weight=5.39,   # ratio bom:ruim
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
```

### Notas sobre o grid

- `scale_pos_weight=5.39` substitui `class_weight="balanced"` — equivalente funcional
- `subsample=0.8` e `colsample_bytree=0.8` adicionam aleatoriedade (como RF)
- `learning_rate` controla o "shrinkage" — valores menores = mais árvores, menos overfitting
- Grid menor que o ideal para manter o tempo de treino razoável (~2-3 min)

---

## Pipeline V4

### O que NÃO muda

- **Dados** (PGN, filtragem, rotulagem) — reutilizamos
- **Features** — mesmas 67 da V3 (`data/features/features_v3.csv`)
- **Split** — mesmo 70/15/15 estratificado (seed=42)
- **Decision Tree e Random Forest** — mantidos como baselines

### O que muda

1. **`src/train_models.py`** — adicionar flag `--v4`, treinar XGBClassifier, threshold tuning
2. **`data/models_v4/`** — salvar DT, RF, XGB e thresholds
3. **`src/evaluate_v4.py`** — comparação V1→V2→V3→V4 com 8 modelos
4. **`requirements.txt`** — adicionar `xgboost`
5. **`src/version_config.py`** — adicionar V4

### Custo computacional

| Operação | Tempo estimado |
|---|---|
| Extração de features (V3) | Já feita (~23s) |
| GridSearchCV do XGBoost | ~2-3 min |
| Threshold tuning (3 modelos) | ~5s |
| Avaliação e gráficos | ~30s |
| **Total** | **~3-4 min** |

---

## Impacto esperado

| Cenário | F1-ruim estimado | Justificativa |
|---|---|---|
| V3 RF (actual) | 0.43 | Baseline actual |
| V4 XGB (conservador) | 0.46–0.48 | Boosting típico +3-5pp sobre RF |
| V4 XGB + threshold (optimista) | 0.48–0.52 | Threshold tuning pode dar +2-3pp adicionais |
| V4 RF + threshold | 0.44–0.46 | Threshold tuning melhora RF mesmo sem XGB |

### Meta

| Métrica | V3 actual | Meta V4 |
|---|---|---|
| **F1 (ruim)** | 0.4312 | **≥ 0.50** |
| **AUC** | 0.7678 | **≥ 0.78** (manter/melhorar) |
| **Recall (ruim)** | 52.1% | **≥ 55%** |
| **Precision (ruim)** | 36.8% | **≥ 40%** |

---

## Etapas do dev-loop V4

| Etapa | Descrição | Artefato |
|---|---|---|
| 16 — Documentação V4 | Diagnóstico V3 + spec deste documento | `docs/04-modelagem/upgrade-v4-xgboost.md` |
| 17 — Treino V4 | XGBoost + threshold tuning com 67 features V3 | `data/models_v4/` |
| 18 — Avaliação V4 | Comparação V1→V2→V3→V4, gráficos | `data/evaluation_v4/` |
| 19 — Notebook V4 | Atualizar notebook com seção V4 | `notebooks/v4_xgboost.ipynb` |

---

## Referências

- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) (Chen & Guestrin, 2016)
- [diagnostico-v3.md](../06-riscos-e-limitacoes/diagnostico-v3.md) — evolução V1→V2→V3
- [modelos-e-hiperparametros.md](modelos-e-hiperparametros.md) — DT + RF original
