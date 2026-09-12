# -*- coding: utf-8 -*-

import ollama
import pandas as pd
import numpy as np
import os
import time
from difflib import SequenceMatcher # Import for string similarity

from sklearn.datasets import fetch_20newsgroups

class TopicModeler:
    def __init__(self, model_name, system_prompt, temperature, max_tokens, num_suggestions, output_path, dataset_subset="all", dataset_remove=("headers", "footers", "quotes"), diversity_penalty_threshold=0.7, diversity_penalty_strength=0.2):
        """
        Initializes the TopicModeler with LLM configurations, dataset, and output path.

        Args:
            model_name (str): The name of the Ollama model to use (e.g., "llama3.1").
            system_prompt (str): The system prompt for the LLM.
            temperature (float): The sampling temperature for topic generation.
            max_tokens (int): The maximum number of tokens for topic generation.
            num_suggestions (int): The number of topic suggestions to generate per document.
            output_path (str): The file path to save the topic modeling results.
            dataset_subset (str): Subset of the 20 Newsgroups dataset to load ("all", "train", "test").
            dataset_remove (tuple): Parts of the dataset to remove ("headers", "footers", "quotes").
            diversity_penalty_threshold (float): Similarity threshold (0-1). Topics above this are considered similar for penalty.
            diversity_penalty_strength (float): Factor (0-1) to reduce confidence for similar topics. Higher means stronger penalty.
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_suggestions = num_suggestions
        self.output_path = output_path
        self.dataset_subset = dataset_subset
        self.dataset_remove = dataset_remove
        self.diversity_penalty_threshold = diversity_penalty_threshold
        self.diversity_penalty_strength = diversity_penalty_strength

        self._load_dataset()
        #self._test_ollama_connection()


    def _load_dataset(self):
        """Loads the 20 Newsgroups dataset."""
        print("Carregando 20 Newsgroups...")
        data = fetch_20newsgroups(
            subset=self.dataset_subset,
            remove=self.dataset_remove
        )
        self.docs = data["data"]
        self.target_names = data.target_names
        print(f"Documentos carregados: {len(self.docs)}")
        print("\nAs 20 classificações possíveis são:")
        for i, name in enumerate(self.target_names):
            print(f"{i+1}. {name}")

    def _test_ollama_connection(self):
        """Tests connection to Ollama and checks for model availability."""
        print(f"\nTestando conexão com Ollama para o modelo '{self.model_name}'...")
        try:
            models = ollama.list()
            # ollama.list() returns a dict with 'models' key, which is a list of dicts. Each model dict has a 'name' key.
            available_models = [model['name'] for model in models['models']]
            print("Modelos disponíveis:")
            for model_name_full in available_models:
                print(f" - {model_name_full}")

            if not any(self.model_name in model_name_full for model_name_full in available_models):
                raise Exception(f"Modelo '{self.model_name}' não encontrado. Por favor, faça download com 'ollama pull {self.model_name}'")

            print("\nOllama funcionando e modelo disponível!")
        except Exception as e:
            print("\nERRO ao conectar ou verificar modelo Ollama:")
            print(e)
            print("\nVerifique se o Ollama está rodando e o modelo baixado:")
            print("ollama list")
            print(f"ollama pull {self.model_name}")
            raise # Re-raise the exception to stop execution if connection fails

    def _generate_topic(self, doc):
        """Generates a single topic suggestion for a given document."""
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": doc
                }
            ],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        )
        topic = response["message"]["content"].strip()
        return topic

    def _load_previous_progress(self):
        """Loads existing results from the output path to resume processing."""
        start_idx = 0
        if os.path.exists(self.output_path):
            try:
                existing_df = pd.read_csv(self.output_path)
                if not existing_df.empty:
                    # Ensure 'doc_id' exists and is numeric before finding max
                    if 'doc_id' in existing_df.columns:
                        existing_df['doc_id'] = pd.to_numeric(existing_df['doc_id'], errors='coerce')
                        start_idx = int(existing_df["doc_id"].max()) + 1
                        print(f"Arquivo de resultados encontrado. Retomando do documento {start_idx + 1}...")
                    else:
                        print(f"Arquivo {self.output_path} existe, mas não contém a coluna 'doc_id'. Iniciando do zero.")
                        # Consider removing corrupted or incompatible file
                        os.remove(self.output_path)
            except Exception as e:
                print(f"Erro ao ler arquivo existente '{self.output_path}': {e}")
                print("Iniciando processamento do zero.")
                if os.path.exists(self.output_path):
                    os.remove(self.output_path) # Remove potentially corrupted file
                start_idx = 0
        return start_idx

    def _calculate_topic_similarity(self, topic1, topic2):
        """Calculates similarity between two topic strings using SequenceMatcher."""
        return SequenceMatcher(None, topic1.lower(), topic2.lower()).ratio()

    def _apply_diversity_penalty(self, df_topics):
        """
        Applies a diversity penalty to the confidence scores.
        Reduces confidence for topics that are highly similar to more confident topics.
        """
        if self.diversity_penalty_threshold <= 0 or self.diversity_penalty_strength == 0 or len(df_topics) <= 1:
            return df_topics

        # Sort by confidence descending to ensure we penalize less confident similar topics
        df_sorted = df_topics.sort_values(by='confianca', ascending=False).reset_index(drop=True)
        
        # Create a copy to store penalized confidences to avoid modifying while iterating
        penalized_confidences = df_sorted['confianca'].copy()

        for i in range(len(df_sorted)):
            topic_i = df_sorted.loc[i, 'topico_sugerido']

            # Compare with all *less confident* topics to avoid self-penalization or symmetric issues
            for j in range(i + 1, len(df_sorted)): 
                topic_j = df_sorted.loc[j, 'topico_sugerido']
                similarity = self._calculate_topic_similarity(topic_i, topic_j)

                if similarity >= self.diversity_penalty_threshold:
                    # Apply penalty to the less confident topic (topic_j)
                    # The penalty is proportional to similarity, strength, and topic_j's current confidence
                    reduction = similarity * self.diversity_penalty_strength * penalized_confidences.loc[j]
                    penalized_confidences.loc[j] = max(0, penalized_confidences.loc[j] - reduction)
        
        df_sorted['confianca'] = penalized_confidences
        return df_sorted.sort_values(by='confianca', ascending=False).reset_index(drop=True)

    def process_document(self, idx):
        """Processes a single document, generates multiple topic suggestions, and returns a DataFrame."""
        doc = self.docs[idx]

        if not doc.strip():
            print(f"Documento {idx + 1} vazio. Pulando...")
            return None

        print(f"\nProcessando documento {idx + 1}/{len(self.docs)}...")
        topics = []
        for beam in range(self.num_suggestions):
            topic = self._generate_topic(doc)
            topics.append(topic)
            print(f"  Sugestão {beam + 1}: {topic}")

        topic_counts = pd.Series(topics).value_counts()
        df_temp = pd.DataFrame({
            "doc_id": idx,
            "topico_sugerido": topic_counts.index,
            "frequencia": topic_counts.values
        })
        df_temp["confianca"] = df_temp["frequencia"] / self.num_suggestions

        # Apply diversity penalty
        df_temp = self._apply_diversity_penalty(df_temp)
        
        return df_temp

    def run_experiment(self):
        """Runs the topic modeling experiment across the entire corpus."""
        print("\n\nIniciando processamento completo do corpus...")
        start_idx = self._load_previous_progress()

        for idx in range(start_idx, len(self.docs)):
            try:
                df_doc_topics = self.process_document(idx)
                if df_doc_topics is not None:
                    # Write header only if file doesn't exist or if it's a brand new run (start_idx == 0)
                    header = not os.path.exists(self.output_path) or start_idx == 0
                    df_doc_topics.to_csv(self.output_path, mode="a", index=False, header=header)
                    print(f"Documento {idx + 1} salvo.")
            except Exception as e:
                print(f"Erro no documento {idx + 1}: {e}")
                print("Pulando documento devido ao erro...")
                # Optionally, log the error to a separate file or track problematic doc_ids

        print("\n" + "=" * 70)
        print("PROCESSAMENTO CONCLUÍDO")
        print("=" * 70)
        print(f"Resultados salvos em: {self.output_path}")

def main():
    print("\nConfigurando o TopicModeler...")
    topic_modeler = TopicModeler(
        model_name='llama3.1',
        system_prompt="Suggest a single, short topic that summarizes the following text:",
        temperature=1,
        max_tokens=5,
        num_suggestions=5,
        output_path="teste_modulo.csv",
        diversity_penalty_threshold=0.75,
        diversity_penalty_strength=0.3
    )

    print("\n" + "=" * 70)
    print("TESTE COM PRIMEIRO DOCUMENTO")
    print("=" * 70)
    first_doc_idx = 0
    if topic_modeler.docs:
        print("\nDOCUMENTO:")
        print(topic_modeler.docs[first_doc_idx][:1000])
        print("\nGERANDO TÓPICOS PARA O PRIMEIRO DOCUMENTO...\n")
        try:
            df_first_doc_output = topic_modeler.process_document(first_doc_idx)
            if df_first_doc_output is not None:
                print("\n" + "=" * 70)
                print("RESULTADOS DO PRIMEIRO DOCUMENTO (COM PENALIDADE DE DIVERSIDADE)")
                print("=" * 70)
                print(df_first_doc_output)
        except Exception as e:
            print(f"Erro ao processar o primeiro documento: {e}")
    else:
        print("Nenhum documento carregado para teste.")

    # topic_modeler.run_experiment()


if __name__ == "__main__":
    main()