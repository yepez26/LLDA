import sys
import os
import time
import ollama
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for p in [BASE_DIR, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset_loader import load_dataset


# ============================================================
# 1. CARREGAR DATASET
# ============================================================

DATASET_NAME = "20newsgroups"  # Opções: "20newsgroups" ou "aapd"
DATASET_PATH = None  # Caminho para Castor-data/datasets/AAPD se DATASET_NAME="aapd"

print(f"Carregando dataset '{DATASET_NAME}'...")

data = load_dataset(
    dataset_name=DATASET_NAME,
    subset="all",
    dataset_path=DATASET_PATH,
    dataset_remove=("headers", "footers", "quotes")
)

docs = data.docs
y = data.targets

print(f"Documentos: {len(docs)}")
print(f"\nTotal de classificações ({len(data.target_names)}):")
for i, name in enumerate(data.target_names[:20]):
    print(f"{i+1}. {name}")
if len(data.target_names) > 20:
    print(f"... e mais {len(data.target_names) - 20} categorias.")


# ============================================================
# 2. CONFIGURAÇÕES DO OLLAMA
# ============================================================

MODEL_NAME = "llama3.1"

NUM_SUGGESTIONS = 10

TEMPERATURE = 0.7

MAX_TOKENS = 20

OUTPUT_PATH = "topic_modeling_results_ollama.csv"


# ============================================================
# 3. TESTAR CONEXÃO COM OLLAMA
# ============================================================

print("\nTestando conexão com Ollama...")

try:

    models = ollama.list()

    available_models = [
        model.model
        for model in models.models
    ]

    print("Modelos disponíveis:")
    for model in available_models:
        print(f" - {model}")

    if not any(MODEL_NAME in model for model in available_models):
        raise Exception(
            f"Modelo '{MODEL_NAME}' não encontrado."
        )

    print("\nOllama funcionando!")

except Exception as e:

    print("\nERRO ao conectar ao Ollama:")
    print(e)

    print("\nVerifique se o Ollama está rodando:")
    print("ollama list")

    exit()


# ============================================================
# 4. PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a topic analyst.

Your task is to identify the main theme of the provided text
and summarize it as a concise topic consisting of 2 to 5 words.

Be specific, use varied vocabulary, and avoid overly generic terms.

Respond with ONLY the topic.
Do not provide explanations.
Do not use punctuation.
"""


# ============================================================
# 5. FUNÇÃO PARA GERAR UM TÓPICO
# ============================================================

def generate_topic(doc, temperature=0.7):

    response = ollama.chat(

        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": doc
            }
        ],

        options={
            "temperature": temperature,
            "num_predict": MAX_TOKENS
        }
    )

    topic = response["message"]["content"].strip()

    return topic


# ============================================================
# 6. TESTE COM PRIMEIRO DOCUMENTO
# ============================================================

print("\n" + "=" * 70)
print("TESTE COM PRIMEIRO DOCUMENTO")
print("=" * 70)

doc = docs[0]

print("\nDOCUMENTO:")
print(doc[:1000])

print("\nGERANDO TÓPICOS...\n")


topics = []


for i in range(NUM_SUGGESTIONS):

    print(f"Gerando sugestão {i+1}/{NUM_SUGGESTIONS}...")

    topic = generate_topic(
        doc,
        temperature=TEMPERATURE
    )

    topics.append(topic)

    print(f"  → {topic}")


# ============================================================
# 7. CONTAR TÓPICOS REPETIDOS
# ============================================================

topic_counts = pd.Series(topics).value_counts()

df_outputs = pd.DataFrame({

    "topico_sugerido": topic_counts.index,

    "frequencia": topic_counts.values

})

df_outputs["confianca"] = (
    df_outputs["frequencia"] / NUM_SUGGESTIONS
)

df_outputs = df_outputs.sort_values(
    "confianca",
    ascending=False
)


print("\n" + "=" * 70)
print("RESULTADOS")
print("=" * 70)

print(df_outputs.to_string(index=False))


# ============================================================
# 8. PROCESSAMENTO DO CORPUS COMPLETO
# ============================================================

print("\n\nIniciando processamento completo...")


# ------------------------------------------------------------
# Recuperar progresso anterior
# ------------------------------------------------------------

start_idx = 0

if os.path.exists(OUTPUT_PATH):

    try:

        existing_df = pd.read_csv(
            OUTPUT_PATH
        )

        if not existing_df.empty:

            start_idx = (
                int(existing_df["doc_id"].max()) + 1
            )

            print(
                f"Arquivo encontrado."
                f" Retomando do documento {start_idx + 1}..."
            )

    except Exception as e:

        print(
            f"Erro ao ler arquivo existente: {e}"
        )

        start_idx = 0


# ============================================================
# LOOP PRINCIPAL
# ============================================================

for idx in range(start_idx, len(docs)):

    doc = docs[idx]

    if not doc.strip():
        continue

    print(
        f"\nProcessando documento "
        f"{idx + 1}/{len(docs)}..."
    )

    topics = []

    try:

        # ----------------------------------------------------
        # Gerar múltiplas sugestões
        # ----------------------------------------------------

        for beam in range(NUM_SUGGESTIONS):

            topic = generate_topic(
                doc,
                temperature=TEMPERATURE
            )

            topics.append(topic)

            print(
                f"  Sugestão {beam + 1}: {topic}"
            )


        # ----------------------------------------------------
        # Contabilizar frequência
        # ----------------------------------------------------

        topic_counts = pd.Series(
            topics
        ).value_counts()


        # ----------------------------------------------------
        # Criar DataFrame
        # ----------------------------------------------------

        df_temp = pd.DataFrame({

            "doc_id": idx,

            "topico_sugerido":
                topic_counts.index,

            "frequencia":
                topic_counts.values

        })


        df_temp["confianca"] = (
            df_temp["frequencia"]
            / NUM_SUGGESTIONS
        )


        # ----------------------------------------------------
        # Salvar
        # ----------------------------------------------------

        header = not os.path.exists(
            OUTPUT_PATH
        )

        df_temp.to_csv(
            OUTPUT_PATH,
            mode="a",
            index=False,
            header=header
        )


        print(
            f"Documento {idx + 1} salvo."
        )


    except Exception as e:

        print(
            f"Erro no documento {idx}: {e}"
        )

        print(
            "Pulando documento..."
        )


print("\n" + "=" * 70)
print("PROCESSAMENTO CONCLUÍDO")
print("=" * 70)

print(
    f"Resultados salvos em: {OUTPUT_PATH}"
)