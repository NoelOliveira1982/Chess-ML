# Objetivo e Escopo

## Problema

Jogadores de nível intermediário-baixo (rating 1200–1500) frequentemente jogam lances que prejudicam sua posição sem perceber o impacto. Falta a esses jogadores um feedback rápido e interpretável sobre a qualidade de cada lance.

## Objetivo principal

Treinar um modelo de Machine Learning supervisionado que, dada uma posição de xadrez e um lance candidato, classifique o lance como **bom** ou **ruim** para a faixa de rating 1200–1500, usando rótulos derivados de avaliação de engine (Stockfish).

## Objetivos secundários

- Demonstrar domínio de **pré-processamento de dados tabulares complexos** (posições de xadrez convertidas em features numéricas).
- Formular corretamente um **problema de classificação binária supervisionada**.
- **Comparar ao menos dois algoritmos** (Árvore de Decisão e um segundo modelo) usando métricas adequadas.
- Produzir **análise de erros** e **interpretabilidade**: extrair regras da árvore, feature importances e exemplos comentados.
- Justificar todas as decisões de amostragem, balanceamento e escolha de features.

## Escopo incluído

- Dataset de posições de xadrez com lances, recortado para rating ~1200–1500 (ajustado por plataforma).
- Rótulos "bom" / "ruim" gerados via delta de avaliação do Stockfish (centipawn loss).
- Engenharia de features tabulares (material, estrutura de peões, segurança do rei, características do lance).
- Treino e avaliação de pelo menos dois classificadores.
- Notebook com descrição do problema, pipeline, resultados, gráficos e discussão.

## Fora de escopo

O que segue **não será implementado**, mas pode ser mencionado como trabalho futuro:

- Engine completa de xadrez jogando partidas fim a fim.
- Treino de redes neurais profundas (estilo AlphaZero).
- Suporte a regras avançadas (repetição tripla, regra dos 50 lances) num agente jogador.
- Interface web ou aplicação para o usuário final.

## Usuário-alvo

| Perfil | Descrição |
|--------|-----------|
| Conceitual | Jogador 1200–1500 querendo feedback sobre qualidade de lances |
| Real | Professora, monitores e banca da disciplina, avaliando formulação de ML, qualidade experimental e relação com o domínio |

## Uso esperado do notebook final

No notebook, será possível:

1. Carregar o modelo treinado.
2. Fornecer uma posição (FEN) + lance.
3. Ver a classificação (bom/ruim) e, quando possível, uma explicação baseada em features ou regras da árvore de decisão.
