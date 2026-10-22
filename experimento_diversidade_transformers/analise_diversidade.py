import os
import sys
import pandas as pd
import numpy as np
import re
from itertools import combinations
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for p in [BASE_DIR, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

INPUT_PATH = os.path.join(CURRENT_DIR, "experimento_diversidade.csv")
if not os.path.exists(INPUT_PATH) and os.path.exists("experimento_diversidade.csv"):
    INPUT_PATH = "experimento_diversidade.csv"

# Modelo para gerar embeddings dos tópicos
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# Se você tiver os textos originais dos documentos,
# coloque o caminho aqui.
DOCUMENTS_PATH = None

OUTPUT_PATH = os.path.join(CURRENT_DIR, "metricas_diversidade.csv")


# ============================================================
# 1. CARREGAR DADOS
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("\nDados:")
print(df.head())

print("\nNúmero de linhas:", len(df))
print("Documentos:", df["doc_id"].nunique())
print("Valores de diversity_penalty:",
      sorted(df["diversity_penalty"].unique()))


# ============================================================
# 2. LIMPEZA DOS TÓPICOS
# ============================================================

def clean_topic(topic):
    """
    Remove aspas e espaços extras.
    """
    topic = str(topic).strip()

    # Remove aspas externas
    topic = topic.strip('"').strip("'")

    # Remove espaços duplicados
    topic = re.sub(r"\s+", " ", topic)

    return topic


df["topico_limpo"] = df["topico_sugerido"].apply(clean_topic)


# ============================================================
# 3. CARREGAR MODELO DE EMBEDDINGS
# ============================================================

print("\nCarregando modelo de embeddings...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# 4. EMBEDDINGS DOS TÓPICOS
# ============================================================

# Evita calcular o mesmo embedding várias vezes
unique_topics = unique_topics = df["topico_limpo"].unique().tolist()

print(f"Calculando embeddings para {len(unique_topics)} tópicos únicos...")

topic_embeddings = embedding_model.encode(
    unique_topics,
    normalize_embeddings=True,
    show_progress_bar=True
)

topic_embedding_dict = {
    topic: embedding
    for topic, embedding in zip(unique_topics, topic_embeddings)
}


# ============================================================
# 5. DISTINCT-2
# ============================================================

def distinct_n(topics, n=2):
    """
    Distinct-N:

        número de n-grams únicos
        ------------------------
        número total de n-grams

    Mede diversidade lexical.
    """

    ngrams = []

    for topic in topics:

        words = topic.lower().split()

        if len(words) < n:
            continue

        topic_ngrams = [
            tuple(words[i:i+n])
            for i in range(len(words) - n + 1)
        ]

        ngrams.extend(topic_ngrams)

    if len(ngrams) == 0:
        return 0.0

    return len(set(ngrams)) / len(ngrams)


# ============================================================
# 6. UNIQUE TOPIC RATE
# ============================================================

def unique_topic_rate(group):
    """
    Proporção de tópicos diferentes em relação
    ao número total de sugestões.

    Como 'frequencia' representa quantas vezes
    cada tópico foi gerado, usamos a soma das
    frequências como número total de sugestões.
    """

    total_suggestions = group["frequencia"].sum()

    unique_topics = group["topico_limpo"].nunique()

    if total_suggestions == 0:
        return 0.0

    return unique_topics / total_suggestions


# ============================================================
# 7. DIVERSIDADE SEMÂNTICA INTRA-DOCUMENTO
# ============================================================

def semantic_diversity(group):
    
    """
    Calcula:

        1 - similaridade semântica média

    entre as sugestões de um documento.

    A frequência é utilizada como peso.

    Portanto, se:

        tópico A -> frequência 5

    significa que ele apareceu nas 5 gerações,
    aumentando sua contribuição para a similaridade.

    Resultado:

        0 = nenhuma diversidade
        1 = máxima diversidade
    """

    topics = group["topico_limpo"].tolist()
    frequencies = group["frequencia"].astype(float).tolist()

    # Se existe apenas um tópico
    if len(topics) <= 1:
        return 0.0

    embeddings = np.array([
        topic_embedding_dict[t]
        for t in topics
    ])

    similarities = cosine_similarity(embeddings)

    weighted_similarities = []
    weights = []

    for i, j in combinations(range(len(topics)), 2):

        similarity = similarities[i, j]

        # Peso do par
        weight = frequencies[i] * frequencies[j]

        weighted_similarities.append(similarity)
        weights.append(weight)

    weighted_mean_similarity = np.average(
        weighted_similarities,
        weights=weights
    )

    diversity = 1 - weighted_mean_similarity

    return diversity


# ============================================================
# 8. DIVERSIDADE SEMÂNTICA GLOBAL
# ============================================================

def global_semantic_diversity(group):
    """
    Mede a diversidade semântica entre todos os tópicos
    produzidos para determinado diversity_penalty.

    Aqui também utilizamos a frequência como peso.
    """

    topics = group["topico_limpo"].tolist()
    frequencies = group["frequencia"].astype(float).tolist()

    if len(topics) <= 1:
        return 0.0

    embeddings = np.array([
        topic_embedding_dict[t]
        for t in topics
    ])

    similarities = cosine_similarity(embeddings)

    weighted_similarities = []
    weights = []

    for i, j in combinations(range(len(topics)), 2):

        similarity = similarities[i, j]

        weight = frequencies[i] * frequencies[j]

        weighted_similarities.append(similarity)
        weights.append(weight)

    mean_similarity = np.average(
        weighted_similarities,
        weights=weights
    )

    return 1 - mean_similarity


# ============================================================
# 9. COERÊNCIA TÓPICO-DOCUMENTO
# ============================================================

documents = None

if DOCUMENTS_PATH is not None:

    documents = pd.read_csv(DOCUMENTS_PATH)

    # Espera:
    # doc_id | texto

    documents["doc_id"] = documents["doc_id"].astype(df["doc_id"].dtype)

    print("\nCalculando embeddings dos documentos...")

    document_embeddings = embedding_model.encode(
        documents["texto"].astype(str).tolist(),
        normalize_embeddings=True,
        show_progress_bar=True
    )

    document_embedding_dict = {
        doc_id: embedding
        for doc_id, embedding
        in zip(documents["doc_id"], document_embeddings)
    }


def topic_document_coherence(group):

    # Caso não tenhamos os documentos
    if documents is None:
        return np.nan

    doc_id = group["doc_id"].iloc[0]

    if doc_id not in document_embedding_dict:
        return np.nan

    document_embedding = document_embedding_dict[doc_id]

    similarities = []

    for _, row in group.iterrows():

        topic = row["topico_limpo"]

        frequency = row["frequencia"]

        topic_embedding = topic_embedding_dict[topic]

        similarity = cosine_similarity(
            document_embedding.reshape(1, -1),
            topic_embedding.reshape(1, -1)
        )[0, 0]

        # Repetições têm maior peso
        similarities.extend(
            [similarity] * int(frequency)
        )

    if len(similarities) == 0:
        return np.nan

    return np.mean(similarities)


# ============================================================
# 10. MÉTRICAS POR DOCUMENTO + PENALTY
# ============================================================

print("\nCalculando métricas...")

results = []

for (doc_id, diversity_penalty), group in df.groupby(
    ["doc_id", "diversity_penalty"]
):

    # --------------------------------------------------------
    # Número total de sugestões
    # --------------------------------------------------------

    total_suggestions = group["frequencia"].sum()

    # --------------------------------------------------------
    # Número de tópicos diferentes
    # --------------------------------------------------------

    n_unique_topics = group["topico_limpo"].nunique()

    # --------------------------------------------------------
    # Unique Topic Rate
    # --------------------------------------------------------

    unique_rate = (
        n_unique_topics / total_suggestions
        if total_suggestions > 0
        else 0
    )

    # --------------------------------------------------------
    # Distinct-2
    # --------------------------------------------------------

    # Reconstrói as sugestões considerando frequência
    expanded_topics = []

    for _, row in group.iterrows():

        expanded_topics.extend(
            [row["topico_limpo"]] *
            int(row["frequencia"])
        )

    distinct2 = distinct_n(
        expanded_topics,
        n=2
    )

    # --------------------------------------------------------
    # Diversidade semântica
    # --------------------------------------------------------

    semantic_div = semantic_diversity(group)

    # --------------------------------------------------------
    # Coerência
    # --------------------------------------------------------

    coherence = topic_document_coherence(group)

    results.append({

        "doc_id": doc_id,

        "diversity_penalty":
            diversity_penalty,

        "total_suggestions":
            total_suggestions,

        "unique_topics":
            n_unique_topics,

        "unique_topic_rate":
            unique_rate,

        "distinct_2":
            distinct2,

        "semantic_diversity":
            semantic_div,

        "topic_document_coherence":
            coherence
    })


metrics_df = pd.DataFrame(results)


# ============================================================
# 11. DIVERSIDADE GLOBAL POR PENALTY
# ============================================================

global_results = []

for diversity_penalty, group in df.groupby(
    "diversity_penalty"
):

    global_div = global_semantic_diversity(group)

    global_results.append({

        "diversity_penalty":
            diversity_penalty,

        "global_semantic_diversity":
            global_div
    })


global_df = pd.DataFrame(global_results)


# Junta a métrica global
metrics_df = metrics_df.merge(
    global_df,
    on="diversity_penalty",
    how="left"
)


# ============================================================
# 12. SALVAR RESULTADOS POR DOCUMENTO
# ============================================================

metrics_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 13. RESUMO POR DIVERSITY PENALTY
# ============================================================

summary = (
    metrics_df
    .groupby("diversity_penalty")
    .agg({

        "unique_topic_rate":
            ["mean", "std"],

        "distinct_2":
            ["mean", "std"],

        "semantic_diversity":
            ["mean", "std"],

        "topic_document_coherence":
            ["mean", "std"],

        "global_semantic_diversity":
            "first"
    })
    .reset_index()
)


# Remove MultiIndex das colunas
summary.columns = [
    "_".join(col).strip("_")
    for col in summary.columns
]


# ============================================================
# 14. EXIBIR RESULTADOS
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    200
)

print("\n")
print("=" * 100)
print("MÉTRICAS POR DOCUMENTO")
print("=" * 100)

print(metrics_df.head(20))


print("\n")
print("=" * 100)
print("RESUMO POR DIVERSITY PENALTY")
print("=" * 100)

print(summary)


print("\n")
print("=" * 100)
print(f"Resultados salvos em: {OUTPUT_PATH}")
print("=" * 100)