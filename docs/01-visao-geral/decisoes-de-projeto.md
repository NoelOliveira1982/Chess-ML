# Decisões de Projeto

Log de decisões arquiteturais e metodológicas, com justificativas. Cada entrada marca **o quê** foi decidido, **por quê**, e **alternativas descartadas**.

---

## DP-01 — Classificação binária (bom / ruim)

**Decisão:** usar duas classes apenas, sem granularidade intermediária (ex.: "ok", "imprecisão", "erro grave").

**Justificativa:**
- Simplifica a formulação para um problema clássico de classificação binária.
- Adequado ao nível da disciplina — mais classes exigiriam mais dados e métricas multiclasse mais complexas.
- Lances na zona cinzenta entre "bom" e "ruim" serão **descartados** do dataset (ver DP-05).

**Alternativa descartada:** multiclasse (brilhante / bom / imprecisão / erro / blunder). Rejeitada por complexidade desnecessária para o escopo.

---

## DP-02 — Fonte de dados: Lichess open database

**Decisão:** usar exclusivamente os PGNs públicos do Lichess (database.lichess.org), licença CC0.

**Justificativa:**
- Download massivo e reprodutível (ficheiro mensal completo ou streaming via API).
- Licença CC0 — sem restrições para uso acadêmico.
- Cabeçalhos PGN incluem `WhiteElo`, `BlackElo`, `TimeControl`, `ECO` — todos os campos necessários para filtragem.
- A API do Chess.com é viável para consultas pontuais, mas não oferece dump massivo equivalente e tem rate limits.

**Alternativa descartada:** PGNs do Chess.com via API. Rejeitada pela dificuldade de obter volume e reprodutibilidade comparáveis.

**Alternativa descartada:** bases de torneios oficiais (FIDE). Rejeitada porque contêm quase exclusivamente jogadores de rating alto (>2000), inúteis para a faixa-alvo.

---

## DP-03 — Ajuste de faixa de rating (Lichess vs. Chess.com)

**Decisão:** a faixa conceitual do projeto é "1200–1500 Chess.com". No Lichess, isso corresponde aproximadamente a **1400–1700** devido à inflação de rating da plataforma.

**Justificativa:**
- Ratings do Lichess usam Glicko-2 com pool de jogadores diferente, resultando em ~200–300 pontos acima do Chess.com para o mesmo nível de jogo.
- Filtrar por 1200–1500 *no Lichess* capturaria jogadores mais fracos do que o público-alvo.
- Referências: comparações empíricas publicadas em fóruns e artigos da comunidade.

**Ação:** filtrar partidas onde `WhiteElo` **e** `BlackElo` estejam entre 1400 e 1700 nos PGNs do Lichess.

---

## DP-04 — Rotulagem por engine (delta-score), não por estatística de resultados

**Decisão:** gerar rótulos usando a diferença de avaliação do Stockfish antes e depois de cada lance (centipawn loss).

**Justificativa:**
- A avaliação do Stockfish é determinística (dado depth fixo) e independente de amostra.
- A alternativa (taxa de vitória condicionada ao lance) exigiria milhões de partidas na mesma posição para ter significância estatística — impraticável.
- Centipawn loss é a métrica padrão da comunidade de xadrez para classificar qualidade de lances.

**Alternativa descartada:** rotulagem por resultado da partida (vitória/derrota) associado ao lance. Rejeitada porque o resultado depende de muitos lances posteriores, tornando o rótulo ruidoso.

---

## DP-05 — Zona cinzenta: descartar lances ambíguos

**Decisão:** lances com delta-score entre -50 e -150 centipawns (entre -0.5 e -1.5 peão) não recebem rótulo e são excluídos do dataset.

**Justificativa:**
- Esses lances são "imprecisões leves" — não são claramente bons nem claramente ruins.
- Incluí-los como "bom" ou "ruim" introduziria ruído nos rótulos e prejudicaria a performance do classificador.
- Descartar a zona cinzenta cria fronteiras de classe mais nítidas, beneficiando modelos simples como árvore de decisão.

**Trade-off:** reduz o tamanho do dataset. Mitigação: a maioria dos lances cai fora da zona cinzenta, então a perda é moderada.

---

## DP-06 — Features semânticas em vez de one-hot de casas

**Decisão:** não codificar casa de origem/destino como one-hot (64+64 = 128 colunas). Em vez disso, usar features semânticas: coluna central vs. flanco, rank relativo, distância do centro.

**Justificativa:**
- 128 colunas binárias para posição de casa explodem a dimensionalidade sem ganho interpretável para modelos simples.
- Features semânticas (ex.: "o lance vai para o centro?", "avança em direção ao rei adversário?") são mais interpretáveis e alinhadas com conceitos de xadrez.
- Reduz risco de overfitting com datasets pequenos (~30k exemplos).

---

## DP-07 — Controle de tempo: foco em rápidas

**Decisão:** filtrar partidas com controle de tempo de 3 a 10 minutos (blitz e rapid curto).

**Justificativa:**
- É o formato mais jogado na faixa 1200–1500 no Lichess — garante volume de dados.
- Jogos muito longos (clássico) têm poucos dados online; jogos muito curtos (bullet, 1+0) são dominados por tempo, não por qualidade posicional.
- Valores de `TimeControl` aceitos: "180+0", "180+2", "300+0", "300+3", "600+0", "600+5".
