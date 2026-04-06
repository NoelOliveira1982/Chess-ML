---
description: "Etapa 07: Consolidação do notebook final"
---

## Contexto

Ler `PROGRESS.md`. Pré-requisito: etapa 06 concluída (métricas e gráficos gerados).

## Documentação de referência

- `docs/05-pipeline/fluxo-do-notebook.md`

## O que fazer

1. **Consolidar em dois notebooks:**
   - `notebooks/01_data_prep.ipynb`: seções 2–4 (coleta, rotulagem, features) usando código de `src/`.
   - `notebooks/02_modeling.ipynb`: seções 5–8 (treino, avaliação, exemplos, conclusão) carregando CSV.
2. **Para cada notebook:**
   - Célula Markdown de introdução com contexto.
   - Cada bloco de código precedido por Markdown explicativo em português.
   - Gráficos com título e labels.
   - Justificativas inline para decisões metodológicas.
3. **Seção de conclusão** no notebook de modelagem:
   - Resumo dos resultados.
   - Limitações (referenciar `docs/06-riscos-e-limitacoes/riscos.md`).
   - Trabalhos futuros.
4. **Função demo** (opcional):
   - Célula que recebe FEN + lance, extrai features, roda modelo, exibe resultado + explicação.
5. **Verificar:** rodar ambos os notebooks do zero (Kernel → Restart & Run All).

## Ao finalizar

- Atualizar `PROGRESS.md`: etapa 07 → concluída.
- Reportar: "Notebooks finalizados. Projeto pronto para entrega."
