---
description: "Loop de desenvolvimento: identifica a próxima etapa e executa"
---

Você é o orquestrador do projeto Chess Move Classifier. Siga este protocolo rigorosamente:

## Protocolo do loop

1. **Ler estado atual:** abrir `PROGRESS.md` e identificar qual é a próxima etapa com status "pendente" ou "em andamento".
2. **Ler documentação da etapa:** abrir o doc correspondente em `docs/` (mapeamento abaixo).
3. **Verificar artefatos:** checar se os arquivos de entrada necessários já existem (ex.: CSV, notebook anterior).
4. **Executar a etapa:** implementar SOMENTE a próxima etapa pendente — não pular etapas nem fazer várias de uma vez.
5. **Validar:** rodar o código, confirmar que funciona, verificar outputs.
6. **Atualizar PROGRESS.md:** marcar a etapa como "concluída" ou "em andamento", registrar artefatos gerados e observações.
7. **Reportar:** resumir o que foi feito, o que foi gerado e qual é a próxima etapa.

## Mapeamento etapa → docs → command

### V1 — Pipeline original

| Etapa | Documentação | Command específico |
|-------|--------------|--------------------|
| 00 | `docs/README.md` | `etapa-00-setup` |
| 01 | `docs/02-dados/fonte-e-coleta.md` | `etapa-01-coleta` |
| 02 | `docs/02-dados/filtragem-e-amostragem.md` | `etapa-02-filtragem` |
| 03 | `docs/02-dados/rotulagem.md` | `etapa-03-rotulagem` |
| 04 | `docs/03-features/engenharia-de-features.md` | `etapa-04-features` |
| 05 | `docs/04-modelagem/modelos-e-hiperparametros.md` | `etapa-05-treino` |
| 06 | `docs/04-modelagem/avaliacao-e-metricas.md` | `etapa-06-avaliacao` |
| 07 | `docs/05-pipeline/fluxo-do-notebook.md` | `etapa-07-notebook` |

### V2 — Upgrade com features táticas

| Etapa | Documentação | Command específico |
|-------|--------------|--------------------|
| 08 | `docs/03-features/features-taticas.md` | `etapa-08-features-taticas` |
| 09 | `docs/04-modelagem/modelos-e-hiperparametros.md` | `etapa-09-retreino-v2` |
| 10 | `docs/04-modelagem/avaliacao-e-metricas.md` + `docs/06-riscos-e-limitacoes/diagnostico-v1.md` | `etapa-10-avaliacao-v2` |
| 11 | `docs/05-pipeline/fluxo-do-notebook.md` | `etapa-11-notebook-v2` |

### V3 — Upgrade com features look-ahead (delta + SEE)

| Etapa | Documentação | Command específico |
|-------|--------------|--------------------|
| 12 | `docs/03-features/features-lookahead.md` | `etapa-12-features-lookahead` |
| 13 | `docs/04-modelagem/modelos-e-hiperparametros.md` | `etapa-13-retreino-v3` |
| 14 | `docs/04-modelagem/avaliacao-e-metricas.md` + `docs/06-riscos-e-limitacoes/diagnostico-v2.md` | `etapa-14-avaliacao-v3` |
| 15 | `docs/05-pipeline/fluxo-do-notebook.md` | `etapa-15-notebook-v3` |

### V4 — Upgrade de modelo (XGBoost + threshold tuning)

| Etapa | Documentação | Command específico |
|-------|--------------|--------------------|
| 16 | `docs/04-modelagem/upgrade-v4-xgboost.md` + `docs/06-riscos-e-limitacoes/diagnostico-v3.md` | `etapa-16-doc-v4` |
| 17 | `docs/04-modelagem/upgrade-v4-xgboost.md` | `etapa-17-treino-v4` |
| 18 | `docs/04-modelagem/avaliacao-e-metricas.md` + `docs/06-riscos-e-limitacoes/diagnostico-v3.md` | `etapa-18-avaliacao-v4` |
| 19 | `docs/05-pipeline/fluxo-do-notebook.md` | `etapa-19-notebook-v4` |

## Regras

- Nunca pular uma etapa. Se a etapa anterior não está concluída, concluí-la primeiro.
- Se a etapa anterior está "em andamento", retomá-la de onde parou.
- Cada run do loop faz UMA etapa. Isso economiza contexto e tokens.
- Sempre atualizar PROGRESS.md ao final.
