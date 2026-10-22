# Modelagem de tópicos: comparação simplificada

O experimento principal para discutir a relação entre tópicos descobertos e classes originais está em
[modelagem_topicos_dificuldades_k10_k54.ipynb](modelagem_topicos_dificuldades_k10_k54.ipynb).

Foram refeitos quatro ajustes: LLDA + LDA e LDA direto, com K=10 e K=54.
O híbrido reutiliza os títulos gerados; o direto usa até 1.500 caracteres do
resumo recuperado, o mesmo limite da geração original. Semente, iterações e
configuração do vetorizador são iguais. O ajuste usa os 2.000 artigos; a
comparação descritiva usa os mesmos 1.986 artigos com termos em ambos os pipelines.

A leitura se limita a palavras-chave, exemplos, distribuição tópico × classe
e uma contagem: quantos tópicos dominantes reúnem 80% dos artigos de cada classe.
Não há acurácia, alinhamento Húngaro, sonda supervisionada nem ranking de qualidade.
Uma classe espalhada pode conter subtemas legítimos e não implica erro.

Em K=54, todas as 54 classes precisam de mais de um tópico para atingir 80%.
A mediana é 18,5 tópicos no híbrido e 19 no direto. Isso ilustra a dificuldade
adicional de relacionar descoberta à taxonomia; não prova superioridade da
classificação. A mudança de tarefa é motivada por um alvo predefinido e uma
correspondência direta entre scores e classes conhecidas.

Os resultados estão em `resultados_modelagem_simples/`. Para refazer os ajustes:

```bash
.venv/bin/python experimento_aapd_comparativo/modelagem_simples.py
```

No notebook, `REEXECUTAR = False` carrega os resultados salvos; `True` refaz
os quatro ajustes em CPU. Os notebooks comparativos antigos de K=10/K=54
permanecem como histórico; a comparação simplificada substitui sua função
principal no relatório consolidado.
