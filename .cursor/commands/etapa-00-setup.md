---
description: "Etapa 00: Setup do ambiente (dependências, Stockfish, estrutura)"
---

## Contexto

Ler `PROGRESS.md` para verificar se esta etapa já foi feita. Se já estiver concluída, informar e parar.

## Documentação de referência

- `docs/README.md` (stack técnica)
- `docs/02-dados/fonte-e-coleta.md` (dependências de dados)

## O que fazer

1. **Criar `requirements.txt`** com as dependências do projeto:
   - python-chess, zstandard, pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter
2. **Criar estrutura de pastas:**
   - `data/` (para CSVs e PGNs — adicionar ao .gitignore)
   - `notebooks/` (para os Jupyter notebooks)
   - `src/` (para módulos Python reutilizáveis)
3. **Criar `.gitignore`** adequado (dados, checkpoints, __pycache__, .ipynb_checkpoints, *.pgn, *.zst, *.csv grandes).
4. **Verificar Stockfish** instalado: `which stockfish`. Se não, orientar `brew install stockfish`.
5. **Criar ambiente virtual:** `python -m venv .venv` e instalar dependências.
6. **Testar imports** criando um script mínimo que importa todas as libs.

## Ao finalizar

- Atualizar `PROGRESS.md`: etapa 00 → concluída, registrar artefatos (requirements.txt, .gitignore, pastas).
- Reportar: "Setup concluído. Próxima etapa: 01 — Coleta de dados."
