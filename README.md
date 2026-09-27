# LLDA: Topic Modeling com LLMs e Avaliação de Diversidade

Este repositório contém pipelines e experimentos para **Modelagem de Tópicos (Topic Modeling)** assistida por Modelos de Linguagem de Grande Porte (LLMs - LLaMA 3 / LLaMA 3.1), explorando geração estocástica via Ollama, decodificação controlada (*Diverse Group Beam Search*) com Hugging Face Transformers, baselines probabilísticos clássicos (**Latent Dirichlet Allocation - LDA**), métricas de diversidade/coerência e refinamento de tópicos com **BERTopic**.

---

## 📁 Estrutura do Projeto

* `dataset_loader.py`: Carregador unificado de datasets para `20newsgroups` (scikit-learn) e `aapd` (ArXiv Academic Paper Dataset do Castor-data).
* `Ollama_LLDA_module.py`: Módulo orientado a objetos (`TopicModeler`) para amostragem estocástica com Ollama e penalidade de diversidade por pós-processamento (`difflib`).
* `transformer_LLDA_module.py`: Implementação com PyTorch e Hugging Face Transformers utilizando *Diverse Group Beam Search* (`diversity_penalty`).
* `lda_module.py`: Implementação de modelagem clássica via **Latent Dirichlet Allocation (LDA)** sobre os mesmos datasets, servindo como baseline probabilístico direto.
* `diversity_influency.py`: Experimento de variação de `diversity_penalty` sobre os datasets.
* `analise_diversidade.py`: Pipeline de cálculo de métricas de diversidade lexical (*Distinct-2*), variabilidade (*Unique Topic Rate*), diversidade semântica (cosseno sobre embeddings MPNet) e coerência tópica.
* `bertopic_aggregator.py`: Módulo e CLI para clusterização e agregação de macro-tópicos via BERTopic (UMAP + HDBSCAN) sobre os tópicos gerados.
* `LLDA.py`: Script procedural inicial para Ollama.
* `ollama_test.py`: Script minimalista de teste de conexão com o Ollama.
* `requirements.txt`: Lista de dependências do projeto.

---

## 🚀 Instalação

Recomenda-se o uso de um ambiente virtual Python:

```bash
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📊 Como Usar

### 1. Modelagem Clássica via LDA (Baseline)
```bash
# Execução no 20 Newsgroups (K=20 tópicos)
python lda_module.py --dataset 20newsgroups --n-topics 20 --output resultados_lda_20ng.csv --evaluate-agreement

# Execução no AAPD (K=54 tópicos)
python lda_module.py --dataset aapd --dataset-path Castor-data/datasets/AAPD --n-topics 54 --output resultados_lda_aapd.csv
```

### 2. Geração de Tópicos com Transformers & Diverse Beam Search
```bash
python diversity_influency.py
```

### 3. Análise de Diversidade e Métricas
```bash
python analise_diversidade.py
```

### 4. Agregação com BERTopic
```bash
python bertopic_aggregator.py \
  --input topic_modeling_results_ollama.csv \
  --output topic_modeling_bertopic_annotated.csv \
  --min-cluster-size 50 \
  --attach-dataset 20newsgroups \
  --evaluate-agreement
```
