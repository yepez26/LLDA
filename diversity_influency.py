# experimento_diversidade.py

import os
import random
import pandas as pd

from transformer_LLDA_module import TopicModeler

# --- Configurações do experimento ---
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_NAME = "20newsgroups"  # Opções: "20newsgroups" ou "aapd"
DATASET_PATH = None  # Ex: "Castor-data/datasets/AAPD" (necessário se DATASET_NAME='aapd')
DATASET_SUBSET = "all"

SYSTEM_PROMPT = (
    " You are a topic analyst. Your task is to identify the main theme of the provided text "
    "and summarize it as a concise topic consisting of 2 to 5 words. "
    "Be specific, use varied vocabulary, and avoid overly generic terms. "
    "Respond with only the topic and no additional text."
)
MAX_TOKENS = 16
NUM_SUGGESTIONS = 5
SAMPLE_FRACTION = 0.10  # 10% do dataset
DIVERSITY_VALUES = [0.01, 0.25, 0.5, 0.75, 1.0]
OUTPUT_PATH = "experimento_diversidade.csv"
RANDOM_SEED = 42


def append_to_csv(df, path):
    """Salva o DataFrame no CSV, escrevendo o header apenas se o arquivo ainda não existir."""
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=header)


def main():
    # Instancia o modeler uma única vez; diversity_penalty inicial é só um placeholder,
    # vamos sobrescrever a cada rodada do loop.
    modeler = TopicModeler(
        model_name=MODEL_NAME,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=MAX_TOKENS,
        num_suggestions=NUM_SUGGESTIONS,
        dataset_name=DATASET_NAME,
        dataset_path=DATASET_PATH,
        dataset_subset=DATASET_SUBSET,
        diversity_penalty=DIVERSITY_VALUES[0],
        output_path=OUTPUT_PATH,
    )

    print("\nCarregando modelo (uma única vez para todo o experimento)...")
    modeler.load_model()

    # Amostra fixa de 10% do dataset, mesma amostra usada em todos os valores de diversidade
    random.seed(RANDOM_SEED)
    total_docs = len(modeler.docs)
    sample_size = int(total_docs * SAMPLE_FRACTION)
    sample_indices = random.sample(range(total_docs), sample_size)

    print(f"\nTotal de documentos: {total_docs}")
    print(f"Amostra ({SAMPLE_FRACTION * 100:.0f}%): {sample_size} documentos")
    print(f"Valores de diversity_penalty a testar: {DIVERSITY_VALUES}")

    # Se já existir um CSV de execução anterior, avisa (mas não sobrescreve — sempre em modo append)
    if os.path.exists(OUTPUT_PATH):
        print(f"\nAVISO: '{OUTPUT_PATH}' já existe. Novos resultados serão adicionados ao final.")

    total_iteracoes = len(DIVERSITY_VALUES) * sample_size
    contador = 0

    for diversity_penalty in DIVERSITY_VALUES:
        print("\n" + "=" * 70)
        print(f"DIVERSITY_PENALTY = {diversity_penalty}")
        print("=" * 70)

        modeler.diversity_penalty = diversity_penalty

        for idx in sample_indices:
            contador += 1
            print(f"\n[{contador}/{total_iteracoes}] diversity={diversity_penalty} | doc {idx}")

            try:
                df_doc = modeler.process_document(idx)
            except Exception as e:
                print(f"  Erro ao processar documento {idx} (diversity={diversity_penalty}): {e}")
                continue

            if df_doc is None:
                continue

            df_doc["diversity_penalty"] = diversity_penalty
            append_to_csv(df_doc, OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("EXPERIMENTO CONCLUÍDO")
    print("=" * 70)
    print(f"Resultados salvos incrementalmente em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()