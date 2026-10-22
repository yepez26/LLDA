# LLDA: da geração de temas à distribuição de evidências sobre tópicos

Revisão do repositório em **13 de outubro de 2026**. Este texto reúne os experimentos de modelagem e classificação de tópicos, distinguindo o que foi implementado, o que possui resultados persistidos e o que ainda depende de validação. Os números vêm dos CSVs, configurações e saídas dos notebooks; esta revisão não repetiu inferência nem treinamento.

[Abrir a versão com sete gráficos interativos](sintese_experimentos.html). Os gráficos estão incorporados nesse HTML e funcionam sem conexão à internet. Tabelas, especificações dos gráficos e inventário dos arquivos estão em [dados/](dados/).

## A pergunta de pesquisa e o significado de “distribuição de tópicos”

A pergunta que organiza o projeto é: **uma LLM pode produzir uma representação temática útil do texto a partir das alternativas disponíveis durante a geração, especialmente das probabilidades extraídas dos logits?** A trajetória do repositório explora duas tarefas diferentes.

Na **modelagem de tópicos**, os temas são descobertos: a LLM gera expressões livres, e outro mecanismo pode agrupá-las em tópicos compartilhados pelo corpus. Na **classificação de tópicos**, as classes já existem: o sistema precisa atribuir e ordenar um catálogo conhecido. O AAPD é multi-rótulo, de modo que o mesmo artigo pode pertencer a várias classes.

“LLDA”, neste repositório, designa a abordagem por LLM do projeto; não deve ser confundido com o modelo estatístico Labeled LDA. Há também quatro objetos probabilísticos diferentes ao longo dos experimentos:

| Objeto | O que representa | Limite de interpretação |
|---|---|---|
| Frequência de sugestões de uma LLM | Frequência empírica de strings geradas sob um prompt e uma configuração | Não é automaticamente probabilidade de um tópico semântico; paráfrases dividem a contagem |
| Mistura de tópicos do LDA | Pesos dos tópicos latentes para o documento | Não equivale diretamente à probabilidade das classes originais |
| Score de beam ou de sequência | Preferência do modelo por uma sequência de tokens | Comprimento, palavras escolhidas, terminação e estratégia de busca influenciam o valor |
| Softmax entre scores de classes | Evidência relativa dentro de um catálogo | Soma 1; não é, por si só, probabilidade independente de pertinência multi-rótulo |

**Conclusão geral:** o repositório demonstra viabilidade técnica e resultados exploratórios promissores para classificação baseada em logits. Ainda não demonstra uma distribuição de pertinência temática calibrada, nem superioridade sobre baselines clássicos sob um protocolo único.

## Dados e comparabilidade dos experimentos

A primeira linha de trabalho usa **20 Newsgroups**, com sugestões temáticas livres. O CSV principal de Ollama tem 18.331 documentos e dez gerações contabilizadas por documento. A grade de diversidade tem 1.836 documentos efetivamente registrados em cada uma de cinco condições.

A segunda linha usa **2.000 artigos do AAPD**, correspondentes aos arquivos locais `test.txt` e `val.txt`, com 54 classes, média de 2,4105 rótulos por artigo e suporte por classe variando de 9 a 683. A designação “teste” neste relatório se refere ao split de cada experimento, não necessariamente ao nome do arquivo original do AAPD.

Um achado decisivo da revisão é a diferença entre **texto original e resumo abreviado**. O script comparativo inicial envia à LLM no máximo 1.500 caracteres, mas armazena no CSV apenas `doc_text[:140] + "..."`. O LDA inicial é ajustado no corpus carregado. Já vários notebooks posteriores e os pipelines Seeded LDA/Guided BERTopic leem diretamente o campo abreviado. Os ensaios recentes de trie e scores exatos recuperam o resumo completo, conferindo prefixo e classes.

| Família | Entrada efetiva | Protocolo principal |
|---|---|---|
| Comparativo AAPD inicial | LLDA: até 1.500 caracteres; LDA: corpus carregado | Análise descritiva de 2.000 artigos |
| Agregação dos títulos LLDA | Título temático gerado | Ajuste no corpus inteiro |
| Modelagem simplificada refeita, K=10/K=54 | Títulos existentes versus até 1.500 caracteres dos resumos recuperados | Ajuste descritivo nos 2.000 artigos; diagnóstico comum em 1.986 |
| Comparativos K=10/K=54 anteriores, históricos | Títulos versus fragmentos | Protocolos antigos de sonda/alinhamento; não são a análise principal |
| Seeded LDA e Guided BERTopic | Fragmentos do CSV | 1.200 treino / 300 validação / 500 teste |
| Trie de 54 classes | Resumos completos | Inferência em todos os 2.000 artigos |
| Scores exatos e verbalizadores | Resumos completos | 400 validação / 1.600 avaliação exploratória |

Não há base para ordenar todos esses métodos em um único gráfico de “melhor desempenho”: mudam as entradas, a supervisão, o catálogo, o prompt, o número de hipóteses e os splits. Nas classes do AAPD, “classe primária” é apenas o primeiro código do CSV, não uma hierarquia temática validada. Pares como `cs.IT`/`math.IT` também exigem atenção à coocorrência e à equivalência semântica.

## Experimentos de modelagem de tópicos

### 1. Sugestões livres com Ollama e distribuição por frequência

**Proposta e tarefa.** Obter temas curtos e interpretáveis sem fixar previamente o vocabulário de tópicos. Repetir a geração fornece uma primeira aproximação de distribuição por documento.

**Metodologia.** Os scripts pedem expressões de duas a cinco palavras. O script principal configura `llama3.1`, temperatura 0,7 e dez sugestões, agrupando strings iguais e calculando frequência dividida pelo número de chamadas. O módulo alternativo também aplica uma penalidade posterior de similaridade lexical. Essas chamadas repetidas ao Ollama são amostragem, não beam search, apesar do nome de algumas variáveis.

**Resultados persistidos.** O CSV principal reúne 144.532 linhas de sugestões agregadas, 18.331 documentos e 137.361 strings distintas no corpus. Todos os documentos desse arquivo somam dez gerações. Um segundo CSV com rótulos, associado à variante JSON, contém 183.086 linhas e 18.327 documentos; 15 documentos não somam dez frequências, indicando necessidade de auditoria desse artefato antes de uso quantitativo.

**Conclusão.** O pipeline produz temas legíveis em escala, mas o enorme número de strings distintas revela fragmentação da representação. “Pittsburgh Penguins Dominate” e “Pittsburgh Penguins Domination”, por exemplo, recebem contagens separadas. A frequência não resolve a equivalência entre expressões, e os artefatos antigos não registram todos os parâmetros necessários para atribuir cada resultado a uma versão exata do modelo.

Fontes: [LLDA.py](../../experimento_ollama_llda/LLDA.py), [módulo Ollama](../../experimento_ollama_llda/Ollama_LLDA_module.py), [CSV principal](../../experimento_ollama_llda/topic_modeling_results_ollama.csv).

### 2. Diverse Group Beam Search e penalidade de diversidade

**Proposta e tarefa.** Controlar explicitamente a variedade dos temas sugeridos para um mesmo documento, em vez de depender apenas da temperatura.

**Metodologia.** O código configura Llama-3-8B-Instruct, cinco beams, cinco grupos e penalidades de 0,01 a 1,00. Avalia taxa de nomes únicos, Distinct-2 e distância semântica entre sugestões, usando embeddings. O arquivo de métricas registra 9.180 combinações documento–penalidade: 1.836 documentos por condição, com cinco sugestões contabilizadas em cada combinação. Esses valores descrevem os registros; a configuração atual do script não substitui um manifesto histórico da execução.

| diversity_penalty | documentos | taxa_unicos | distinct_2 | diversidade_semantica | diversidade_global |
| --- | --- | --- | --- | --- | --- |
| 0.0100 | 1836 | 0.2228 | 0.2160 | 0.0252 | 0.8880 |
| 0.2500 | 1836 | 0.5792 | 0.4802 | 0.2217 | 0.8869 |
| 0.5000 | 1836 | 0.7399 | 0.6042 | 0.2552 | 0.8870 |
| 0.7500 | 1836 | 0.8195 | 0.6700 | 0.2756 | 0.8867 |
| 1.0000 | 1836 | 0.8625 | 0.7135 | 0.2895 | 0.8866 |

<!-- FIG:diversidade -->
[Ver gráfico de diversidade](sintese_experimentos.html#diversidade).

**Descoberta.** Aumentar a penalidade elevou a taxa média de sugestões únicas de 22,28% para 86,25%, o Distinct-2 de 0,2160 para 0,7135 e a diversidade semântica intra-documento de 0,0252 para 0,2895. A diversidade global ficou praticamente estável, perto de 0,887.

**Limite e conclusão.** A coerência tópico–documento está ausente em todas as linhas: o script configura `DOCUMENTS_PATH=None`. Portanto, não foi demonstrado que mais diversidade traz temas mais corretos. Além disso, a diversidade semântica pondera pares de strings distintas, mas não inclui os pares idênticos decorrentes das repetições; não é exatamente a distância média entre todas as cinco gerações expandidas. O experimento demonstra controle de variedade, não qualidade temática ou calibração.

Fontes: [grade](../../experimento_diversidade_transformers/diversity_influency.py), [análise das métricas](../../experimento_diversidade_transformers/analise_diversidade.py), [dados](../../experimento_diversidade_transformers/metricas_diversidade.csv).

### 3. BERTopic como agregador de temas gerados

**Proposta e tarefa.** Reduzir a fragmentação dos títulos livres, formando um vocabulário compartilhado de macrotemas.

**Metodologia.** O agregador limpa as sugestões, calcula embeddings e aplica UMAP, HDBSCAN e representação c-TF-IDF. Cada linha agregada de sugestão entra uma vez no ajuste; a coluna de frequência não é passada como peso ao clustering.

**Resultados.** O arquivo anotado tem exatamente as mesmas 144.532 entradas básicas do CSV principal de Ollama, correspondentes a 18.331 documentos. Há **455 clusters temáticos**, além do cluster `-1`. Os outliers representam **40,73% das linhas** e **41,39% da massa de frequências**.

**Conclusão.** A agregação transforma expressões livres em grupos compartilhados, mas deixa uma fração expressiva como ruído. Não há nesse artefato uma avaliação persistida de alinhamento com as 20 classes ou de qualidade humana dos clusters. Ele prova a execução do pipeline, não a superioridade da modelagem híbrida. Uma distribuição por documento exigiria agregar explicitamente suas frequências por cluster e definir como tratar o ruído.

Fontes: [agregador](../../experimento_bertopic/bertopic_aggregator.py), [CSV anotado](../../experimento_bertopic/topic_modeling_bertopic_annotated.csv).

### 4. LLDA versus LDA clássico no AAPD e agregação por LDA

**Proposta e tarefa.** Comparar a expressividade dos temas da LLM com os 54 tópicos de um LDA clássico e investigar se os títulos gerados podem ser condensados em macrotemas.

**Metodologia.** O comparativo gera sugestões via Ollama, ajusta LDA no corpus carregado e armazena o título predominante de cada método. No consolidado há 2.000 artigos. O detalhamento LLDA registra duas gerações em 1.985 artigos e três em 15 artigos; os resultados combinam diferentes quantidades de sugestões. O modelo Ollama é selecionado entre os instalados, sem manifesto histórico suficiente para confirmar uma única versão usada em todas as linhas.

**Resultados descritivos.** O notebook registra 1.989 títulos LLDA únicos contra 54 representações de tópicos LDA. Distinct-2: 84,01% versus 2,63%; Distinct-1: 27,20% versus 1,42%. Os títulos LLDA têm média de 4,27 palavras; o LDA exporta seis palavras-chave por tópico. A média de “confiança” LLDA é 0,545, enquanto o maior peso LDA tem média 0,639: esses números representam objetos distintos e não medem vantagem de um método.

**Agregação.** Um segundo notebook vetoriza os títulos LLDA em 1.327 unigramas/bigramas e compara K em {5, 8, 10, 12, 15, 20}. A menor perplexidade registrada é **1.258,11 em K=5**; K=10 tem **1.514,74**, embora seja posteriormente adotado como `K_OPTIMAL=10`. O notebook justifica K=10 por parcimônia e granularidade; essa escolha interpretativa não é o mínimo da perplexidade registrada nem vem acompanhada de validação humana. Seus dez grupos recebem de 159 a 232 documentos, mas exemplos e palavras-chave indicam mistura de assuntos.

**Conclusão.** A LLM oferece descrições mais variadas; isso não prova melhor recuperação de tópicos, pois a quantidade de saídas possíveis é radicalmente diferente. O híbrido fornece pesos de macrotemas, mas a evidência não sustenta que sua partição seja semanticamente melhor. Outro limite de proveniência: o CSV detalhado do LDA cobre apenas 15 documentos, enquanto o consolidado tem 2.000.

Fontes: [comparativo inicial](../../experimento_aapd_comparativo/aadp_lda_llda.ipynb), [agregação](../../experimento_aapd_comparativo/lda_agregacao_llda.ipynb), [script de execução](../../experimento_aapd_comparativo/experimento_comparativo_aapd.py).

### 5. Modelagem simplificada: os grupos descobertos não são as classes

**Proposta e tarefa.** Esta seção substitui a comparação anterior, que combinava alinhamento, sondas e várias métricas. A pergunta agora é descritiva: **os artigos de uma classe original ficam reunidos em um único tópico descoberto ou se distribuem por vários?** Não buscamos vencedor entre pipelines, nem tratamos correspondência imperfeita como erro de modelagem.

**Experimento refeito.** Ajustamos novamente quatro LDAs: LLDA + LDA e LDA direto, cada um com K=10 e K=54. Usamos os mesmos 2.000 artigos, semente 42, 35 iterações e a mesma configuração de vetorização com unigramas/bigramas; cada pipeline aprende seu vocabulário. No híbrido, reutilizamos os títulos LLDA já gerados. No direto, recuperamos os resumos completos e usamos até **1.500 caracteres**, o mesmo limite de entrada da geração original dos títulos. Assim, o baseline não fica limitado aos antigos fragmentos de 140 caracteres. Não chamamos a LLM novamente nem ajustamos os modelos com rótulos.

Quatorze títulos ficam sem termos após a vetorização. Para não interpretar suas misturas uniformes como descoberta temática, os diagnósticos usam os **mesmos 1.986 artigos com termos nos dois pipelines**. O ajuste continua usando o corpus de 2.000 artigos. Cada artigo participa de todas as suas classes originais; não escolhemos uma classe primária.

**Leitura simples.** Apresentamos palavras-chave dos tópicos, exemplos e uma visualização tópico × classe. Para descrever a dispersão, perguntamos **quantos tópicos dominantes são necessários para reunir 80% dos artigos de uma classe**. Se dez artigos estão distribuídos em 5, 3 e 2 por tópico, dois tópicos reúnem oito artigos. O limiar de 80% é uma convenção descritiva fixada antes do ensaio, não um critério de qualidade.

| Metodo | K | Topicos usados | Classes espalhadas em 2+ topicos (80%) | Mediana de topicos para 80% da classe |
| --- | --- | --- | --- | --- |
| LLDA + LDA | 10 | 10 | 54 | 6.0000 |
| LLDA + LDA | 54 | 54 | 54 | 18.5000 |
| LDA direto | 10 | 10 | 53 | 4.0000 |
| LDA direto | 54 | 54 | 54 | 19.0000 |

<!-- FIG:k54 -->
[Ver gráfico descritivo de modelagem](sintese_experimentos.html#k54).

**O que isso mostra.** Em K=10, 54 das 54 classes no híbrido e 53 das 54 no direto precisam de mais de um tópico para reunir 80% de seus artigos. Em K=54, isso ocorre nas 54 classes de ambos; a mediana é de 18,5 tópicos no híbrido e 19 no direto. Mesmo quando K=54, a igualdade do número de grupos e classes não atribui significado aos IDs dos tópicos. O notebook permite observar a distribuição de cada classe e ler os termos dos grupos. A leitura usa apenas o tópico dominante para simplificar; as misturas completas foram preservadas e não são substituídas por uma hipótese de classe única.

**O que não mostra.** Uma classe pode conter vários subtemas legítimos; um tópico pode reunir categorias próximas ou coocorrentes. Por isso, uma classe distribuída entre grupos não significa que o modelo errou, e uma mediana menor não torna automaticamente um pipeline melhor. Avaliar a qualidade da descoberta ainda exige interpretação, coerência/relevância e, idealmente, avaliação humana.

**Por que mudar para classificação.** A motivação é a clareza da tarefa e da avaliação, não uma suposta derrota da modelagem. Na modelagem, precisamos descobrir os grupos e decidir depois como relacioná-los à taxonomia. Na classificação, o catálogo é definido antes da inferência: cada score aponta para uma classe conhecida, permitindo avaliar diretamente ranking e atribuições multi-rótulo. A classificação não garante acerto nem calibração, mas retira a ambiguidade adicional de interpretar tópicos latentes como classes.

Fontes: [novo notebook executado](../../experimento_aapd_comparativo/modelagem_topicos_dificuldades_k10_k54.ipynb), [implementação simples](../../experimento_aapd_comparativo/modelagem_simples.py), [resultados refeitos](../../experimento_aapd_comparativo/resultados_modelagem_simples/).

Os notebooks antigos de K=10/K=54 permanecem como histórico, não como protocolo principal desta seção. Seu alinhamento e suas sondas respondem perguntas diferentes; a coluna manual de vencedores de K=10 é inconsistente e não deve orientar conclusões. Os resultados antigos e seus intervalos continuam nos arquivos originais.

## Experimentos de classificação de tópicos

### 6. Seeded LDA e Guided BERTopic

**Proposta e tarefa.** Atribuir os artigos a classes conhecidas usando descrições e sementes, em vez de descobrir tópicos sem orientação.

**Metodologia.** O Seeded LDA é uma implementação variacional própria com 54 tópicos e prior assimétrico tópico–palavra; as sementes recebem reforço em todas as atualizações, não apenas na inicialização. O Guided BERTopic usa `seed_topic_list` e acrescenta uma camada de projeção: compara embeddings das palavras de cada cluster com descritores das classes e transfere a pertinência para as classes mais semelhantes. Isso não garante 54 clusters nem probabilidade calibrada das classes. Ambos recebem os fragmentos do CSV e usam splits de 1.200/300/500; a validação escolhe k entre 1 e 5 para as atribuições.

| Método | LRAP | F1 micro | F1 macro (54) | k validado |
| --- | --- | --- | --- | --- |
| Seeded LDA | 0.2635 | 0.1944 | 0.1418 | 3 |
| Guided BERTopic | 0.4532 | 0.3982 | 0.1342 | 2 |

<!-- FIG:baselines -->
[Ver gráfico dos baselines orientados](sintese_experimentos.html#baselines).

**Resultados e conclusão.** O Guided BERTopic melhora LRAP e F1 micro, mas tem F1 macro menor e ranking loss pior que o Seeded LDA: 0,3967 contra 0,3598. São encontradas 20 clusters e 20 classes recebem algum peso. O HDBSCAN marca 31% do teste como outlier, embora documentos assim ainda possam ter pertinência parcial e receber rótulos. A comparação mostra diferenças entre pipelines orientados, mas não constitui baseline equivalente aos ensaios recentes de LLM: estes usam textos completos e outros protocolos. Os CSVs provam a execução, mesmo que o notebook didático não tenha saídas salvas.

Fontes: [notebook](../../experimento_classificacao_topicos/classificacao_topicos_aapd.ipynb), [Seeded LDA](../../experimento_classificacao_topicos/seeded_lda.py), [Guided BERTopic](../../experimento_classificacao_topicos/guided_bertopic.py), [métricas](../../experimento_classificacao_topicos/resultados/ambos/metricas.csv).

### 7. Ensaio controlado em cinco artigos: prompt, trie e ranking de candidatos

**Proposta e tarefa.** Recriar a geração de temas por múltiplos beams dentro de um catálogo fechado e separar conformidade de formato de acerto semântico.

**Metodologia.** Usam-se os cinco primeiros registros do CSV, com textos completos recuperados. Nove códigos são traduzidos em oito nomes legíveis, unificando `cs.IT` e `math.IT`. O modelo é Qwen2.5-0.5B-Instruct, com quatro beams padrão. Comparam-se restrição apenas no prompt, restrição do próximo token por trie e score de todos os nomes por teacher forcing. A trie aceita apenas prefixos capazes de terminar em um nome permitido, liberando EOS ao final.

| Estratégia | Validade exata | Acerto top-1 | Recall top-3 |
|---|---:|---:|---:|
| Prompt + beams | 75% | 60% | 60% |
| Trie + beams | 100% | 60% | 40% |
| Ranking dos candidatos | 100% | 60% | 50% |

**Conclusão.** A restrição elimina nomes fora do catálogo, mas não melhora o top-1. O catálogo é construído com a união dos rótulos desses cinco artigos: é um exercício de mecanismo, não um teste de generalização. Beams são hipóteses determinísticas, não amostras independentes; sua frequência não deve ser usada como confiança calibrada.

Fonte: [notebook e respostas executadas](../../experimento_classificacao_topicos/llda_5_artigos_decoding_restrito.ipynb).

### 8. Trie_beams nos 2.000 artigos e nas 54 classes

**Proposta e tarefa.** Escalar o controle de geração para todo o catálogo e observar sua qualidade de classificação e custo computacional.

**Metodologia.** Qwen2.5-0.5B-Instruct em float16, RTX 5080, lotes de oito artigos completos e quatro beams por artigo. As 54 classes têm nomes distintos; categorias próximas recebem qualificadores. A trie controla os tokens e a avaliação usa as três primeiras hipóteses. Não há ajuste dos pesos da LLM. Há checkpoint, identificação do modelo e validação exata das respostas.

**Resultados.** Foram produzidas 8.000 hipóteses, todas dentro do catálogo. Acerto top-1: **15,95%**; precisão top-3: **16,08%**; recall top-3: **17,92%**; F1 micro: **0,1694**; F1 macro: **0,1177**. Nove classes não aparecem no top-3. Statistical Mechanics, com 33 artigos positivos, é atribuída 1.049 vezes; Machine Learning, com 294 positivos, aparece 177 vezes e atinge precisão de 67,25%.

**Conclusão.** O mecanismo é operacional e escalável, mas não transforma automaticamente preferência por strings em pertinência de classes. A preferência muito desigual por nomes é observada; atribuí-la exclusivamente ao comprimento, ao tamanho do modelo ou à poda seria uma hipótese, não uma conclusão isolada por esse experimento. A execução leva 68,33 segundos de inferência/gravação, com pico de 2,31 GiB de memória CUDA alocada.

Fonte: [notebook completo](../../experimento_classificacao_topicos/llda_trie_beams_2000_artigos_54_classes_gpu.ipynb), [dados e configuração](../../experimento_classificacao_topicos/resultados/llda_trie_54_gpu/).

### 9. Scores exatos, múltiplas expressões e correção lexical

**Proposta e tarefa.** Construir evidência relativa sobre todas as classes diretamente das probabilidades de sequência, sem eliminar candidatos pelo beam search, e reduzir dependência de uma única forma de nomear o tópico.

**Metodologia.** São usadas três expressões manuais por classe, 162 no total. Todas as ablações compartilham o mesmo prompt com esse catálogo. O modelo continua sendo Qwen2.5-0.5B-Instruct. O score bruto é a soma de log-probabilidades de todos os tokens da expressão e do EOS, calculadas sobre o vocabulário completo. “Exato” significa avaliar cada sequência tokenizada escolhida sem poda; não significa somar todas as tokenizações ou paráfrases possíveis do mesmo nome. O cache do prompt é compartilhado, mas cada candidato tem uma cópia independente. Testes verificam equivalência com teacher forcing completo e invariância à ordem dos candidatos.

A variante com múltiplas expressões usa a média uniforme de suas probabilidades, computada por logsumexp. A correção desconta lambda vezes o log da probabilidade média em três entradas neutras: texto vazio, `N/A` e `Scientific abstract.`. Lambda é escolhido em {0; 0,25; 0,5; 0,75; 1} pelo LRAP de 400 artigos, separadamente para nome único e múltiplas expressões. Os outros 1.600 compõem a avaliação exploratória; os rótulos não entram no prompt. A etapa de seleção de lambda usa supervisão, ainda que os pesos da LLM permaneçam congelados.

| Metodo | lambda | LRAP | Acerto top1 | F1 micro top3 | F1 macro top3 |
| --- | --- | --- | --- | --- | --- |
| nome_unico | 0.0000 | 0.2148 | 0.1219 | 0.1307 | 0.1011 |
| nome_unico_corrigido | 0.7500 | 0.2684 | 0.2119 | 0.1908 | 0.1772 |
| multiplas_expressoes | 0.0000 | 0.2341 | 0.1363 | 0.1527 | 0.1251 |
| multiplas_expressoes_corrigidas | 1.0000 | 0.3556 | 0.3738 | 0.2736 | 0.2064 |

<!-- FIG:ablacoes -->
[Ver gráfico das ablações](sintese_experimentos.html#ablacoes).

**Descoberta principal.** A combinação eleva LRAP de **0,2148 para 0,3556**, um ganho absoluto de 0,1408, e acerto top-1 de **12,19% para 37,38%**. O F1 macro aumenta de 0,1011 para 0,2064. A correção isolada melhora mais que a simples adição de expressões nesta configuração; a combinação apresenta o maior LRAP observado. Isso apoia a investigação do viés lexical, sem provar que o efeito se repete em outros modelos ou dados.

**Heterogeneidade.** Comparando a combinação com nome único nos mesmos 1.600 artigos, **34 classes melhoram, 13 pioram e 7 empatam no F1**. Entre os maiores ganhos estão Physics and Society e Quantum Physics; Machine Learning e Artificial Intelligence estão entre as perdas. Há melhora agregada, não uniforme.

<!-- FIG:classes -->
[Ver ganhos e perdas por classe](sintese_experimentos.html#classes).

**O que a distribuição significa.** Softmax sobre os 54 scores produz uma distribuição relativa de evidências. Após correção, os scores são baseados em razões de probabilidades, não na probabilidade bruta de gerar o nome. Nenhuma variante foi demonstrada como probabilidade independente de pertinência multi-rótulo. Não há avaliação de calibração, cobertura de intervalos ou taxa de erro sob abstenção.

**Custo.** Foram calculados 324.000 scores de expressões, mais as referências neutras, em 259,86 segundos e pico de 1,71 GiB. É um resultado de viabilidade no hardware registrado. Não se deve comparar esse tempo ao beam como se ambos calculassem a mesma saída: prompts, número de candidatos e cargas de trabalho são diferentes.

| Método | Segundos | VRAM alocada GiB | Quantidade de hipóteses/scores |
| --- | --- | --- | --- |
| Trie: 4 beams, 54 nomes | 68.3330 | 2.3060 | 8000 |
| Score exato: 162 expressões | 259.8629 | 1.7060 | 324000 |

<!-- FIG:custos -->
[Ver custo computacional](sintese_experimentos.html#custos).

Fontes: [notebook executado](../../experimento_classificacao_topicos/llda_scores_exatos_verbalizadores_calibracao.ipynb), [implementação](../../experimento_classificacao_topicos/llda_scores_exatos.py), [catálogo](../../experimento_classificacao_topicos/verbalizadores_54.py), [scores e métricas](../../experimento_classificacao_topicos/resultados/llda_scores_exatos_54/).

## Um diagnóstico adicional: quanto da probabilidade está nos nomes exatos?

A revisão calculou, a partir dos log-scores já salvos, a soma das probabilidades das **162 expressões exatas terminadas em EOS, nas tokenizações canônicas escolhidas pelo tokenizer**. A massa mediana é **28,96%**, variando de aproximadamente 0,079% a 69,33%. Todas as somas observadas ficaram abaixo de 1. Esse diagnóstico cobre os 2.000 artigos e não usa seus rótulos para escolher um limiar.

<!-- FIG:massa -->
[Ver distribuição da massa dos nomes](sintese_experimentos.html#massa).

Os 71,04% restantes no artigo mediano pertencem a outras sequências. Elas podem incluir paráfrases corretas, outras tokenizações dos mesmos nomes, aspas, explicações, terminações diferentes e também respostas inadequadas. **Massa fora do conjunto de strings não é probabilidade de um tema desconhecido.** Entretanto, o diagnóstico explicita o que se perde ao normalizar apenas os candidatos: uma distribuição relativa pode parecer concentrada mesmo quando o modelo atribui pouca massa aos nomes exatos. Investigar essa informação como sinal de incerteza, juntamente com estabilidade semântica, é um caminho de pesquisa testável.

## Descobertas sustentadas e limites da evidência

1. **Decoding controla o espaço de saídas.** A penalidade aumenta variedade e a trie garante nomes válidos nos ensaios observados. Nenhum desses controles, sozinho, demonstra pertinência temática.
2. **Expressões livres precisam de uma ponte semântica.** Contar strings fragmenta paráfrases; agrupá-las reduz o espaço, mas introduz ruído e decisões adicionais sobre alinhamento e outliers.
3. **Descoberta e classificação são tarefas diferentes.** O ensaio simplificado descreve a dispersão das classes entre grupos descobertos. Isso não mede erro: a taxonomia pode ser diferente da estrutura temática do corpus. Fixar as classes antes da inferência torna a avaliação direta mais clara.
4. **Nomes de classe fazem parte do modelo efetivo.** As ablações mostram que mudar agregação e corrigir preferência lexical altera substancialmente a recuperação dos rótulos, mesmo sem ajustar os pesos da LLM.
5. **A proposta recente é computacionalmente viável no recorte testado.** A GPU registrada processa o corpus em minutos. Isso não estabelece desempenho em corpora maiores nem com LLMs maiores.
6. **Ainda falta uma comparação clássica equivalente.** Baselines orientados usam fragmentos, outras divisões e outra política de atribuição. O Guided BERTopic tem LRAP numericamente maior em seu protocolo, mas não podemos declarar que supera a proposta recente — nem o contrário.
7. **Calibração e novidade continuam abertas.** Há uma candidata a metodologia: agregar evidência de expressões, corrigir preferência lexical e normalizar scores. Verbalizadores e correções de viés já têm antecedentes; é necessário demonstrar o diferencial frente a eles. Não há prova de calibração ou revisão sistemática de novidade neste repositório.

## Próximos passos, em ordem de prioridade

| Etapa | Ação | Evidência que deve produzir |
|---|---|---|
| 1. Consolidar dados e protocolo | Manifesto único com textos completos, IDs, hashes, classes, versões e splits; corrigir a leitura de fragmentos nos baselines | Comparações reproduzíveis e entradas equivalentes |
| 2. Comparar em condições comuns | Mesmo conjunto e splits para LDA, Seeded LDA, Guided BERTopic, TF-IDF + regressão logística e embeddings + classificador; separar métodos com e sem supervisão | Estimar a vantagem real e o custo da proposta |
| 3. Isolar a contribuição do decoding | Mesmo prompt/catálogo/modelo para score exato e beams de 4, 8, 16 e 54; manter top-k de avaliação fixo | Medir perda por poda e custo, sem confundir com mudança de prompt |
| 4. Testar robustez semântica | Variar expressões, ordem das classes, templates e modelo; revisar categorias equivalentes sem decidir com o gabarito final | Distinguir evidência do assunto de preferência pela formulação |
| 5. Construir pertinência multi-rótulo | Comparar a distribuição relativa com logits de sim/não por classe e transformações calibradas em dados separados | Probabilidades com interpretação apropriada ao AAPD |
| 6. Avaliar incerteza | Brier/log-loss por classe, curvas de confiabilidade, precisão sob abstenção e risco–cobertura; estudar massa de strings e estabilidade | Testar se a confiança prevê acerto e quando rejeitar |
| 7. Validar em dados novos | Congelar catálogo, correção e limiares; testar em documentos ainda não inspecionados, com bootstrap pareado e análise de suporte | Evidência de generalização e incerteza das diferenças |
| 8. Retomar descoberta de tópicos | Gerar candidatos livres, agrupar paráfrases e pontuar o catálogo descoberto em outros documentos; tratar massa residual | Unificar descoberta e distribuição sem depender das classes do benchmark |

Para a linha de modelagem, também falta avaliação humana ou automática de coerência, relevância, redundância e cobertura. Para a linha de classificação, reduzir top-k a uma decisão fixa de três rótulos limita a recuperação de artigos com cardinalidade diferente; o número de rótulos ou seus limiares devem ser aprendidos somente em dados de desenvolvimento.

Uma contribuição futura convincente seria demonstrar que uma **distribuição temática derivada dos logits, agregada semanticamente e com correção lexical** é mais estável, útil e confiável do que alternativas simples, com um custo aceitável. O resultado atual mais forte é a melhora dentro das ablações recentes; ainda é cedo para chamá-la de técnica universalmente superior ou calibrada.

## Proveniência e uso deste relatório

O [inventário com hashes](dados/inventario_e_proveniencia.csv) registra arquivos de experimentos, quantidade de linhas/documentos nos CSVs e presença de saídas nos notebooks. A ausência de saídas no notebook não significa ausência de execução quando existem CSVs correspondentes. Configurações atuais de scripts antigos são descritas como configurações, não como prova da configuração histórica de cada linha.

As tabelas do relatório são geradas dos artefatos atuais, exceto os números históricos de diversidade lexical do comparativo inicial e grade de perplexidade, transcritos das saídas persistidas dos notebooks citados. O HTML incorpora Plotly; os dados de cada gráfico estão exportados para CSV/JSON. Para reconstruir tabelas e visualizações, execute na raiz:

```bash
.venv/bin/python relatorios/revisao_experimentos_llda_2026-10-13/gerar_relatorio.py
```

O script usa o template textual preservado e não treina modelos nem acessa serviços externos. Uma nova execução dos experimentos requer revisar também a interpretação textual, não apenas atualizar tabelas.

Bibliografia orientativa já relacionada ao projeto: [verbalizadores em classificação de tópicos](https://aclanthology.org/2024.lrec-main.527/) e [vieses de rótulos em modelos de linguagem](https://aclanthology.org/2023.acl-long.783/). Esses trabalhos situam os componentes da proposta; não substituem uma revisão sistemática para estabelecer novidade.
