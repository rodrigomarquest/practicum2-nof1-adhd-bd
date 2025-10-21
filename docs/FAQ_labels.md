# FAQ — Rótulos heurísticos (State of Mind Synthetic)

Este documento fornece respostas resumidas e justificadas para perguntas que podem surgir durante a defesa/discussão acadêmica sobre a implementação de rótulos heurísticos no repositório (arquivo `etl_tools/generate_heuristic_labels.py`). As respostas estão em português e foram redigidas para uma banca de professores que avaliará escolhas metodológicas, validade e implicações éticas.

---

Q: Por que foi feita a alteração para agregar por data antes de rotular?

A: Em estudos longitudinales e para rótulos de "estado mental" é desejável ter uma observação por dia (um rótulo diário), porque:

- Muitas fontes (Apple/Zepp/joins) podem produzir múltiplas linhas para o mesmo dia (segmentos, múltiplos devices, várias leituras). Trabalhar com múltiplas linhas por dia pode inflacionar amostras e enviesar métricas.
- Agregar (ex.: média do escore heurístico por `date`) reduz ruído e produz um rótulo único por dia, que é compatível com os notebooks de EDA e com a maioria dos métodos de modelagem temporal (diários).
- A decisão de agregar é conservadora: preserva sinal central (média) e evita contar repetidos como observações independentes.

Q: Como a heurística foi construída (fórmula) e por que essas variáveis?

A: A heurística definida foi: score_raw = z(HRV) + z(Sleep) - z(ScreenTime) - z(RestHR), normalizada com tanh(score_raw). Racional:

- HRV (p.ex. RMSSD/SDNN) está associado a regulação autonômica e tende a aumentar em estados de maior relaxamento/recuperação. Contribui positivamente para um melhor estado de humor (hipótese baseada em literatura).
- Qualidade/quantidade de sono (p.ex. sono eficiente ou minutos) também melhora estado mental; soma-se ao score.
- Tempo de uso de tela foi usado como proxy para sobrecarga cognitiva/isolamento do sono e tende a correlacionar negativamente com bem-estar em muitos estudos. Subtrai-se do score.
- Resting HR (maior HR em repouso) pode indicar stress fisiológico; subtrai-se do score.

Os sinais foram padronizados (z-score) para torná-los comparáveis; em seguida usamos tanh para limitar o score a (-1, 1) e aplicar limiares fixos (+0.5/−0.5) para categorizar em positive/neutral/negative.

Q: Por que normalizar com tanh em vez de min-max ou z direto?

A: tanh oferece um mapeamento contínuo e robusto que preserva a ordem (monótono) e limita valores extremos. Vantagens:

- Reduz influência de outliers sem truncamento abrupto.
- Mantém interpretação relativa (valores >0.5 refletem contribuições relativamente fortes).

Min-max também é válido, mas é sensível a valores extremos e à janela de observação; tanh com z-scores é simples, rápido e reproduzível.

Q: Como o script lida com colunas faltantes (p.ex., não existe screen_time ou sleep)?

A: O código aplica busca por nomes de coluna com correspondência frouxa (fuzzy substring) entre candidatos comuns (incluindo variações `zepp_*`). Se uma variável não for encontrada, ela é tratada como série zero, ou seja, não contribui ao score. Essa escolha evita falhas em pipelines reais onde nem todas as fontes estão presentes e torna o rótulo aplicável de forma conservadora.

Q: Por que usar limiares fixos (+0.5 / -0.5) para categorizar em positive/neutral/negative?

A: Limiar fixo sobre valores normalizados (tanh) é interpretável e fácil de justificar: 0.5 corresponde a um efeito moderado (aprox. tanh^-1(0.5) ≈ 0.55 em z-score bruto), o que evita classificar pequenas variações como mudança de estado. Para fins acadêmicos, é transparente e permite comparação — limiares alternativos ou calibrados por validação podem ser demonstrados em análises suplementares.

Q: Há risco de circularidade (labels derivadas das mesmas features usadas em modelos)?

A: Sim — estes rótulos são heurísticos e devem ser tratados como rótulos sintéticos. Eles não substituem rótulos clinicamente validados (EMA/entrevistas). Importante:

- Temos clareza no repositório que estes rótulos são sintéticos e heurísticos (`state_of_mind_synthetic.csv`).
- Em análise de modelos, é crucial separar claramente testes usando rótulos sintéticos e rótulos originais/EMA; usar rótulos heurísticos apenas para exploração, geração de hipóteses e para treinar modelos de pré-treinamento, não como substituto final.

Q: Como a banca pode testar/validar esta heurística?

A: Sugestões de validação:

1. Comparar distribuição dos rótulos sintéticos com labels EMA (quando disponíveis) por período e calcular métricas de concordância (kappa, F1 macro).
2. Realizar análise de sensibilidade: variar limiar (ex.: 0.3 → 0.7) e observar estabilidade das contagens.
3. Rodar ablações: remover uma variável (p.ex., screen_time) e verificar impacto no score/labels.
4. Testar agregação alternativa (mediana por dia, ou último valor do dia) e reportar diferenças.

Q: E a ética / privacidade — gerar rótulos pode representar risco?

A: Geração de rótulos por si só não aumenta identificação, mas deve-se considerar:

- Divulgação: rótulos sintéticos devem ser compartilhados apenas em conjunto com as políticas de anonimização já aplicadas ao repositório e conforme aprovação ética.
- Uso responsável: não conjecturar diagnósticos clínicos com base em rótulos heurísticos.
- Consentimento: os processos que levaram aos dados originais e ao uso de rótulos devem estar alinhados com o consentimento informado e com o plano de governança no `/docs`.

Q: Se a banca pedir para melhorar, por onde começar?

A: Recomendações práticas:

- Calibrar pesos e limiares usando um conjunto rotulado (EMA) e reportar melhoria de concordância.
- Incluir features adicionais ou transformações (p.ex., variabilidade intra-diária).
- Construir um classificador simples (ex.: regressão logística) treinado sobre as features e compará-lo com a heurística.
- Adotar uma estratégia de validação temporal e reportar robustez (drift ao longo do tempo).

---

Se a banca quiser, posso gerar um apêndice com: (a) código para comparar rótulos sintéticos com labels EMA (quando existirem); (b) scripts de sensibilidade que varrem limiares e produção de tabelas com métricas de concordância.
