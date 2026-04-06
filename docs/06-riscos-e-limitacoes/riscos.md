# Riscos e Limitações

## Riscos do projeto

### R1 — Tempo de rotulagem com Stockfish

**Risco:** avaliar dezenas de milhares de posições com Stockfish pode demorar horas se a profundidade (depth) for alta demais ou o dataset for grande.

**Probabilidade:** média.

**Impacto:** atraso na entrega; impossibilidade de iterar rapidamente sobre os limiares de rotulagem.

**Mitigação:**
- Usar depth 15 (balança qualidade vs. velocidade).
- Limitar o dataset a ~2000 partidas inicialmente.
- Salvar os resultados da rotulagem em CSV intermediário para não re-rodar.
- Se necessário, reduzir para depth 10 (ainda detecta a maioria dos erros táticos de 2–3 lances).

---

### R2 — Desbalanceamento severo

**Risco:** a maioria dos lances na faixa 1200–1500 é "razoável" (bom), e poucos são erros graves (ruim). Após descartar a zona cinzenta, a classe "ruim" pode ficar com < 20% do dataset.

**Probabilidade:** alta.

**Impacto:** modelo enviesado para a classe majoritária; accuracy enganosamente alta (prediz sempre "bom").

**Mitigação:**
- Usar `class_weight="balanced"` no treinamento.
- Monitorar F1 e recall da classe "ruim", não apenas accuracy.
- Se necessário, aplicar undersampling da classe "bom" ou ajustar limiares de rotulagem (ex.: relaxar "ruim" para delta <= -100 cp).

---

### R3 — Features insuficientes para padrões táticos

**Risco:** as features propostas (material, mobilidade, estrutura de peões) capturam aspectos **posicionais**, mas não capturam **táticas** específicas (cravadas, enfiadas, raio-x, ataques duplos, mates).

**Probabilidade:** alta.

**Impacto:** o modelo pode falhar em classificar lances que perdem material por tática — exatamente os erros mais comuns na faixa 1200–1500.

**Mitigação:**
- Reconhecer essa limitação explicitamente no notebook.
- Na análise de erros, mostrar exemplos onde a tática não foi capturada.
- Sugerir como trabalho futuro: features de ameaças (peças atacadas sem defesa), detecção de cravadas via raio de bispos/torres.

---

### R4 — Overfitting com dataset pequeno

**Risco:** com ~30 features e ~5k–30k exemplos, modelos de árvore profunda ou MLP podem memorizar o treino.

**Probabilidade:** média.

**Impacto:** performance boa no treino, ruim no teste.

**Mitigação:**
- Limitar profundidade da árvore (`max_depth`).
- Usar `min_samples_leaf` para evitar folhas com poucos exemplos.
- Monitorar gap treino-teste nas métricas.
- Curva de aprendizado para diagnosticar.

---

### R5 — Ruído nos rótulos

**Risco:** o Stockfish em profundidade limitada pode avaliar incorretamente posições complexas. Posições com múltiplas linhas de igual força podem ter avaliações instáveis.

**Probabilidade:** baixa a média.

**Impacto:** rótulos incorretos para uma fração dos exemplos, limitando a performance máxima do classificador.

**Mitigação:**
- Usar depth suficiente (15+) para reduzir avaliações instáveis.
- Descartar posições com avaliação de mate (onde centipawns perdem sentido).
- Na análise de erros, verificar se exemplos mal classificados são de fato rótulos ruidosos.

---

### R6 — Diferença de rating entre plataformas

**Risco:** ao comunicar os resultados, dizer "1200–1500" pode ser confuso se os dados vêm do Lichess (onde esses números correspondem a um nível mais baixo).

**Probabilidade:** baixa (mais comunicacional que técnico).

**Impacto:** mal-entendido sobre o público-alvo.

**Mitigação:**
- Documentar explicitamente: "1400–1700 Lichess ≈ 1200–1500 Chess.com" (ver [DP-03](../01-visao-geral/decisoes-de-projeto.md)).
- Nos slides e no notebook, usar a faixa Lichess com nota de rodapé explicando a conversão.

---

## Limitações conhecidas

| Limitação | Consequência |
|-----------|-------------|
| Features puramente posicionais, sem detecção de táticas | Modelo não captura erros por cravada, enfiada, etc. |
| Dataset de uma única plataforma (Lichess) | Pode não generalizar para jogadores de Chess.com ou OTB |
| Classificação binária simplificada | Perde nuance (imprecisão leve vs. blunder catastrófico) |
| Depth fixo do Stockfish | Rótulos imprecisos em posições muito complexas |
| Foco em meio-jogo (lance 8–40) | Não cobre aberturas nem finais |
| Dataset de um único mês | Pode ter sazonalidade ou viés temporal |

---

## Trabalhos futuros

Sugestões para extensão do projeto (a mencionar na conclusão do notebook):

1. **Features táticas:** adicionar detecção de peças indefesas, ameaças de mate, cravadas.
2. **Multiclasse:** expandir para brilhante / bom / imprecisão / erro / blunder (5 classes).
3. **Fases separadas:** treinar modelos distintos para meio-jogo e finais.
4. **Representação bitboard:** usar os 12 bitboards como features (64 bits cada, total 768 features) e testar com modelos mais complexos.
5. **Agente recomendador:** dado o classificador treinado, gerar os N lances legais de uma posição, classificar cada um, e recomendar os "bons".
6. **Validação cruzada temporal:** treinar em meses mais antigos e testar em meses mais recentes.
7. **Deep learning:** CNN sobre representação matricial do tabuleiro (8x8x12).
