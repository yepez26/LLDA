# Atribuição de tópicos predefinidos — AAPD

Aqui a pergunta é: **quais das 54 classes conhecidas descrevem cada artigo?**
Usamos os mesmos 2.000 resumos de `experimento_aapd_comparativo.csv`, sem chamar
Ollama e sem usar os títulos LLDA. Os rótulos reais servem para validação e teste.

## Como os métodos funcionam

| Método | Orientação | Saída para classificação |
|---|---|---|
| Seeded LDA | Um tópico por classe; palavras-semente recebem maior prior | 54 pesos por artigo, já na ordem das classes |
| Guided BERTopic | `seed_topic_list` orienta embeddings/UMAP e representação dos clusters | Pertinência dos clusters projetada nas classes predefinidas |

**Seeded LDA:** implementamos inferência variacional com uma matriz de prior
`tópico × palavra`. A cada atualização, somamos esse prior às contagens esperadas.
Assim, as sementes continuam influenciando o modelo durante todo o treino.
`seed_strength` controla a intensidade. Este código é uma implementação experimental
própria, não o pacote `guidedlda`. O LDA do scikit-learn aceita prior escalar de
palavras, por isso não usamos uma simples inicialização para simular esse efeito.

**Guided BERTopic:** as 54 listas de sementes guiam o BERTopic real. Isso **não
obriga o HDBSCAN a criar 54 clusters**. Para atribuir classes fixas, comparamos os
embeddings das palavras dos clusters aprendidos com os descritores das classes.
Cada cluster transfere sua pertinência para a classe mais semelhante. Empates
compartilham o peso; vários clusters podem apontar para a mesma classe. Classes
sem cluster associado recebem score zero. O CSV de mapeamento torna isso auditável.
Essa projeção é nossa camada de classificação, não uma funcionalidade nativa que
transforma automaticamente seeds em classes.

Os scores do LDA são misturas de tópicos; os do BERTopic são pertinências projetadas.
Nenhum deles é uma probabilidade calibrada de uma classe multi-rótulo. Outliers do
BERTopic são reportados separadamente; documentos com massa nula não recebem classes.
Documentos sem termos no vocabulário LDA recebem a mistura uniforme do prior.

## Executar

Na raiz do projeto, usando o ambiente Python com as dependências:

```bash
.venv/bin/python experimento_classificacao_topicos/experimento.py --metodo seeded
.venv/bin/python experimento_classificacao_topicos/experimento.py --metodo guided
.venv/bin/python experimento_classificacao_topicos/experimento.py --metodo ambos
```

Ou abra `classificacao_topicos_aapd.ipynb` e execute as células em ordem.
O modelo de embeddings padrão é `sentence-transformers/all-MiniLM-L6-v2`, em CPU.
O primeiro uso pode baixar o modelo; com cache completo, use `--local-files-only`.
Instalação, se necessária: `pip install -r experimento_classificacao_topicos/requirements.txt`.

## O que comparar

- **Treino: 1.200 artigos.** Aprende vocabulário, tópicos e clusters, sem os rótulos reais.
- **Validação: 300 artigos.** Escolhe quantos rótulos atribuir (`k` entre 1 e 5), pelo F1 micro.
- **Teste: 500 artigos.** Avalia LRAP (principal), ranking loss, coverage error, F1 micro/macro e Jaccard.

O mesmo split e catálogo são usados pelos dois métodos. Não fornecemos ao modelo
a quantidade verdadeira de classes de cada artigo. Não usamos alinhamento
Húngaro nem classificadores treinados com os rótulos reais. A avaliação mantém
as 54 classes, inclusive raras, e exporta o suporte em cada split.

Os resultados ficam em `resultados/<metodo>/`: métricas, scores, atribuições,
suporte, split, configurações, cobertura de sementes e mapeamento do BERTopic.
Scores empatados usam a ordem do catálogo para top-k; LRAP trata empates como
empates. Considere isso ao interpretar classes com descritores idênticos.

## Próximas explorações

Edite `topicos_predefinidos.py` para melhorar as sementes com conhecimento de domínio.
O catálogo inicial é manual e deve ser revisado. Pares como `cs.IT`/`math.IT`,
`cs.NA`/`math.NA` e `math.ST`/`stat.TH` compartilham descritores por serem próximos;
apenas sementes não distinguem esses códigos de forma confiável.
Experimente força das sementes e tamanho mínimo dos clusters na validação,
preservando o teste para a avaliação final. Uma única semente/split não comprova
superioridade geral. Os modelos continuam sendo orientados por conhecimento prévio;
não são classificadores supervisionados convencionais.

Referências: [Guided BERTopic](https://maartengr.github.io/BERTopic/getting_started/guided/guided.html),
[distribuições BERTopic](https://maartengr.github.io/BERTopic/getting_started/distribution/distribution.html),
[prior no LDA do scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html).

## Primeira execução validada

Os dois métodos foram executados com os 2.000 artigos; os resultados estão em
`resultados/ambos/` (20 iterações LDA, força 20, tamanho mínimo de cluster 15).

| Métrica de teste | Seeded LDA | Guided BERTopic |
|---|---:|---:|
| LRAP ↑ | 0,2635 | 0,4532 |
| F1 micro ↑ | 0,1944 | 0,3982 |
| F1 macro (54 classes) ↑ | 0,1418 | 0,1342 |
| Ranking loss ↓ | 0,3598 | 0,3967 |

O Guided BERTopic obteve maior LRAP e F1 micro nesta execução, enquanto o Seeded
LDA obteve maior F1 macro e menor ranking loss. Foram encontrados 20 clusters,
associados a 20 classes do catálogo, e 31% dos artigos de teste foram marcados como
outliers pelo HDBSCAN. Outliers ainda podem ter pertinência parcial e receber
atribuições; são diferentes de documentos com score totalmente nulo.
A cobertura limitada ajuda a interpretar o resultado das classes raras.
Todas as classes do LDA tiveram alguma palavra-semente presente no vocabulário.

Para variar parâmetros: `--seed-strength 40 --iterations 30` ou
`--min-cluster-size 10`. Testes: `.venv/bin/python -m unittest discover -s experimento_classificacao_topicos -p 'test_*.py'`.

## LLDA com decoding restrito: ensaio em cinco artigos

Abra `llda_5_artigos_decoding_restrito.ipynb` para o primeiro exercício de adaptação
do LLDA por LLM do projeto. Ele usa os cinco primeiros registros do CSV, renomeia
nove códigos em oito temas legíveis e compara:

1. Solicitar o catálogo no prompt, com múltiplos beams.
2. Restringir próximos tokens com uma árvore de prefixos dos nomes, também com beams.
3. Pontuar cada nome completo por log-probabilidade e ranquear os candidatos.

O catálogo é a união dos temas desses cinco artigos, não o gabarito individual de
cada artigo. `cs.IT` e `math.IT` são agrupados em **Information Theory**.
Esse catálogo informado pelos rótulos serve ao exercício, não a um benchmark de
classificação geral. Usamos beam search padrão; os grupos com penalidade de
diversidade do módulo anterior ficam como uma próxima ablação.

O notebook usa `Qwen/Qwen2.5-0.5B-Instruct` para caber em CPU, com modelo configurável.
Transformers controla o decoding; chamadas repetidas ao Ollama não equivalem a
beam search. `llda_decoding_restrito.py` contém as funções reutilizáveis.
Resultados: `resultados/llda_5_artigos/`. LLDA aqui significa o método por LLM do
projeto, não o modelo estatístico Labeled LDA.

Neste ensaio LLDA, os textos são **recuperados por inteiro** em
`Castor-data/datasets/AAPD/test.txt` e `val.txt`: o CSV comparativo contém apenas
140 caracteres por resumo. A recuperação exige correspondência única do prefixo
e das classes e preserva o `doc_id` do CSV. As métricas anteriores dos pipelines
que leram diretamente `resumo_texto` do CSV foram calculadas sobre esses fragmentos.

O notebook LLDA foi executado com os resumos completos e Qwen2.5-0.5B-Instruct.
A validade exata foi 75% no prompt e 100% na trie/ranking de candidatos; as três
estratégias acertaram top-1 em 3/5 artigos. A restrição resolveu o formato, mas
não melhorou o acerto top-1 neste ensaio. O notebook mantém as saídas executadas,
e os CSVs preservam inclusive os nomes inválidos gerados pelo prompt.

## LLDA trie_beams completo em GPU

Notebook: `llda_trie_beams_2000_artigos_54_classes_gpu.ipynb` (executado).
Implementação: `llda_trie_54.py`. Os 2.000 resumos completos são classificados
com 54 nomes distintos, mantendo inclusive códigos de categorias próximas.
A inferência exige CUDA, usa lotes, valida todas as saídas e salva checkpoint
atômico após cada lote. Uma nova execução retoma os documentos pendentes.

```bash
.venv/bin/python experimento_classificacao_topicos/llda_trie_54.py --batch-size 8 --num-beams 4 --top-k 3
```

A execução utilizou Qwen2.5-0.5B-Instruct na NVIDIA RTX 5080 de 16 GB: 2.000
artigos e 8.000 hipóteses concluídos, em 68,33 segundos de inferência e gravação
dos lotes, excluindo carregamento do modelo e preparação dos prompts.
Pico de memória CUDA alocada: 2,31 GiB. Validade exata: 100%; acerto top-1:
15,95%; F1 micro top-3: 0,1694; F1 macro (54): 0,1177.
A restrição garante conformidade; a qualidade semântica permanece limitada
nesta configuração. O notebook inclui gráficos por classe e os resultados
estão em `resultados/llda_trie_54_gpu/`.

## Scores exatos, múltiplas expressões e correção lexical

Notebook: `llda_scores_exatos_verbalizadores_calibracao.ipynb`.
Implementação: `llda_scores_exatos.py`; catálogo manual: `verbalizadores_54.py`.

O experimento calcula **log P(expressão + EOS | artigo, prompt)** para as
162 expressões (três por classe), com logits do vocabulário completo e sem beam
search. O cache do prompt é reutilizado, mas cada candidato recebe uma cópia
independente. Os testes comparam essa implementação com teacher forcing completo,
inclusive após trocar a ordem dos candidatos e o tamanho dos lotes.

Quatro ablações usam o mesmo prompt: nome único, nome único corrigido,
múltiplas expressões e múltiplas expressões corrigidas. A agregação é a média
uniforme das probabilidades das três expressões, calculada em log-space.
A correção subtrai lambda vezes o log da probabilidade média em três entradas
neutras. Lambda é escolhido pelo LRAP em 400 artigos; as métricas reportadas
usam os outros 1.600. Como o conjunto já foi inspecionado antes, a avaliação
continua exploratória e não equivale a um teste final em dados novos.

```bash
.venv/bin/python experimento_classificacao_topicos/llda_scores_exatos.py --candidate-batch 24
```

A inferência exige GPU CUDA e tem checkpoint a cada dez artigos. Resultados
em `resultados/llda_scores_exatos_54/`: log-probabilidades brutas, referências,
scores agregados, distribuições relativas, busca de lambda, splits, métricas,
catálogo, configuração e registro da GPU. O softmax produz uma **distribuição
relativa que soma 1**, não probabilidades independentes de pertinência
multi-rótulo nem uma garantia de calibração.

Primeira execução concluída na RTX 5080 com Qwen2.5-0.5B-Instruct: 324.000
scores dos candidatos, mais três referências neutras, em 259,86 segundos;
pico de memória CUDA alocada de 1,71 GiB. O notebook mantém as células executadas.

| Variante | Lambda | LRAP (1.600 artigos) | Acerto top-1 |
|---|---:|---:|---:|
| Nome único | 0 | 0,2148 | 12,19% |
| Nome único corrigido | 0,75 | 0,2684 | 21,19% |
| Múltiplas expressões | 0 | 0,2341 | 13,62% |
| Múltiplas expressões corrigidas | 1 | 0,3556 | 37,38% |

Os lambdas foram escolhidos nos 400 artigos de validação. A melhora é
exploratória; não demonstra probabilidades calibradas nem generalização para
um novo conjunto. O prompt com 162 expressões difere do prompt do trie anterior.
