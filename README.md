# LLDA: Topic Modeling com LLMs e Avaliação de Diversidade

> Revisão consolidada dos experimentos: [relatório com gráficos interativos](relatorios/revisao_experimentos_llda_2026-10-13/sintese_experimentos.html) · [versão em texto](relatorios/revisao_experimentos_llda_2026-10-13/sintese_experimentos.md).
>
> A revisão distingue modelagem e classificação, registra as limitações dos resumos abreviados e reúne resultados, comparações e próximos passos.


Este repositório contém pipelines e experimentos para **Modelagem de Tópicos (Topic Modeling)** assistida por Modelos de Linguagem de Grande Porte (LLMs - LLaMA 3 / LLaMA 3.1 / LLaMA 3.2), baselines probabilísticos clássicos (**Latent Dirichlet Allocation - LDA**), decodificação controlada (*Diverse Group Beam Search*), métricas de diversidade/coerência e refinamento com **BERTopic**.

---

## 📁 Estrutura de Pastas e Experimentos

O projeto está organizado em diretórios modulares, cada um contendo seus scripts (`.py`) e respectivos resultados (`.csv`):

```
LLDA/
├── dataset_loader.py                     # Carregador unificado de dados (20 Newsgroups e AAPD)
├── requirements.txt                      # Dependências do projeto
├── README.md                             # Documentação geral
│
├── Castor-data/                          # Datasets brutos (ex: AAPD do arXiv)
│   └── datasets/AAPD/
│
├── experimento_ollama_llda/           # Experimentos de amostragem estocástica com Ollama
│   ├── LLDA.py                           # Script de amostragem repetida e contagem de tópicos
│   ├── Ollama_LLDA_module.py             # Classe TopicModeler com penalidade de similaridade
│   ├── ollama_test.py                    # Teste básico de conexão
│   └── *.csv                             # CSVs de resultados gerados pelo Ollama
│
├── experimento_diversidade_transformers/ # Decodificação com Group Beam Search (Hugging Face)
│   ├── transformer_LLDA_module.py        # Modelagem usando PyTorch, Transformers e diversity_penalty
│   ├── diversity_influency.py            # Experimento em grade variando diversity_penalty
│   ├── analise_diversidade.py            # Métricas: Distinct-2, Semantic Diversity, Coerência
│   └── *.csv                             # Resultados das grades e métricas
│
├── experimento_bertopic/              # Agregação e Clusterização de Macro-tópicos
│   ├── bertopic_aggregator.py            # Pipeline BERTopic (UMAP + HDBSCAN + Embeddings)
│   └── *.csv                             # CSVs anotados com macro-clusters
│
└── experimento_aapd_comparativo/      # Comparativo no Dataset AAPD (LLDA vs. LDA Clássico)
    ├── experimento_comparativo_aapd.py   # Script de execução comparativa na mesma amostra
    ├── lda_module.py                     # Implementação do LDA Clássico (scikit-learn, K=54)
    ├── experimento_aapd_comparativo.csv  # Tabela comparativa consolidada
    ├── experimento_aapd_llda_detalhado.csv # Detalhamento dos tópicos gerados pelo LLDA
    └── experimento_aapd_lda_detalhado.csv  # Distribuição de probabilidades e palavras do LDA
```

---

## 🚀 Instalação

Recomenda-se o uso de um ambiente virtual Python:

```bash
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📊 Como Executar os Experimentos

### 1. Comparativo no AAPD (2.000 Documentos: LLDA vs. LDA Clássico)
```bash
python experimento_aapd_comparativo/experimento_comparativo_aapd.py --sample-size 2000 --num-suggestions 2
```

### 2. Modelagem Clássica via LDA
```bash
python experimento_aapd_comparativo/lda_module.py --dataset aapd --n-topics 54
```

### 3. Decodificação Diversificada com Transformers
```bash
python experimento_diversidade_transformers/diversity_influency.py
python experimento_diversidade_transformers/analise_diversidade.py
```

### 4. Agregação Semântica com BERTopic
```bash
python experimento_bertopic/bertopic_aggregator.py \
  --input experimento_aapd_comparativo/experimento_aapd_comparativo.csv \
  --output experimento_bertopic/aapd_bertopic_clusters.csv
```

Os experimentos de classificação estão em [`experimento_classificacao_topicos/`](experimento_classificacao_topicos/README.md): Seeded LDA, Guided BERTopic, trie_beams e scores exatos com verbalizadores e correção lexical. A síntese dos resultados encontra-se no relatório consolidado acima.

A comparação principal de modelagem foi simplificada: [novo ensaio K=10/K=54](experimento_aapd_comparativo/modelagem_topicos_dificuldades_k10_k54.ipynb), com descrição dos grupos e dispersão por classe, sem ranking de qualidade por alinhamento ou sondas.
