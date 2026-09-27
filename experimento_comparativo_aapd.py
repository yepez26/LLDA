# -*- coding: utf-8 -*-
"""
experimento_comparativo_aapd.py

Experimento comparativo de modelagem de tópicos no dataset AAPD
(ArXiv Academic Paper Dataset), comparando na mesma amostra de documentos:
1. LLDA (Topic Modeling baseado em LLM via Ollama - LLDA.py)
2. LDA Clássico (Latent Dirichlet Allocation probabilístico via lda_module.py)

Gera resultados detalhados por documento, métricas de diversidade (Distinct-2)
e arquivos CSV para análise posterior.
"""

import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

import ollama
from dataset_loader import load_dataset
from lda_module import LDATopicModeler


SYSTEM_PROMPT_LLDA = """You are a topic analyst.
Your task is to identify the main theme of the provided text and summarize it as a concise topic consisting of 2 to 5 words.
Be specific, use varied vocabulary, and avoid overly generic terms.
Respond with ONLY the topic. Do not provide explanations. Do not use punctuation."""


def get_available_ollama_model(preferred_models: Optional[List[str]] = None) -> str:
    """Verifica e retorna o melhor modelo disponível no Ollama."""
    if preferred_models is None:
        preferred_models = ["llama3.2:1b", "llama3.1", "llama3", "llama3.2", "mistral", "qwen2.5:0.5b"]

    try:
        models_info = ollama.list()
        available = [m.model for m in models_info.models]
        print(f"[Ollama] Modelos disponíveis localmente: {available}")

        for pref in preferred_models:
            for avail in available:
                if pref in avail:
                    return avail

        if available:
            return available[0]
        else:
            raise RuntimeError("Nenhum modelo baixado no Ollama. Execute 'ollama pull llama3.2:1b'")
    except Exception as e:
        raise RuntimeError(f"Erro ao conectar ao Ollama: {e}")


def generate_llda_topic(doc_text: str, model_name: str, temperature: float = 0.7, max_tokens: int = 20) -> str:
    """Gera um tópico usando o prompt e configurações do LLDA.py."""
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_LLDA},
            {"role": "user", "content": doc_text[:1500]}  # Trunca para evitar estouro de contexto
        ],
        options={
            "temperature": temperature,
            "num_predict": max_tokens
        }
    )
    topic = response["message"]["content"].strip().strip('"\'')
    return topic


def calculate_distinct_2(topics: List[str]) -> float:
    """Calcula a métrica Distinct-2 (diversidade lexical de bigramas)."""
    ngrams = []
    for topic in topics:
        words = topic.lower().split()
        if len(words) < 2:
            continue
        topic_ngrams = [tuple(words[i:i+2]) for i in range(len(words) - 1)]
        ngrams.extend(topic_ngrams)

    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def run_comparative_experiment(
    sample_size: int = 20,
    random_seed: int = 42,
    num_suggestions: int = 3,
    dataset_path: str = "Castor-data/datasets/AAPD",
    output_path: str = "experimento_aapd_comparativo.csv",
    ollama_model_name: Optional[str] = None
):
    print("=" * 80)
    print("INICIANDO EXPERIMENTO COMPARATIVO NO AAPD: LLDA vs LDA CLÁSSICO")
    print("=" * 80)

    # 1. Carregar dataset AAPD
    ds = load_dataset("aapd", dataset_path=dataset_path, subset="all")
    total_docs = len(ds.docs)
    print(f"\nTotal de documentos no dataset: {total_docs}")
    print(f"Total de categorias do arXiv: {len(ds.target_names)}")

    # 2. Selecionar a mesma amostra com semente fixa
    random.seed(random_seed)
    effective_sample_size = min(sample_size, total_docs)
    sample_indices = sorted(random.sample(range(total_docs), effective_sample_size))
    print(f"\nAmostra selecionada ({effective_sample_size} documentos): {sample_indices[:10]}...")

    # 3. Detectar modelo do Ollama
    if ollama_model_name is None:
        ollama_model_name = get_available_ollama_model()
    print(f"\n[LLDA] Utilizando modelo Ollama: '{ollama_model_name}'")

    # 4. Ajustar modelo LDA Clássico sobre o corpus
    print("\n" + "-" * 60)
    print("[LDA Clássico] Treinando modelo Latent Dirichlet Allocation (K=54)...")
    print("-" * 60)
    start_time_lda = time.time()
    lda_modeler = LDATopicModeler(
        n_topics=54,
        num_suggestions=num_suggestions,
        dataset_name="aapd",
        dataset_path=dataset_path,
        max_iter=20,
        random_state=random_seed
    )
    lda_modeler.fit()
    time_lda_total = time.time() - start_time_lda
    print(f"[LDA Clássico] Treinamento concluído em {time_lda_total:.2f} segundos.")

    # 5. Executar inferência na amostra comum
    print("\n" + "-" * 60)
    print(f"[Inferência] Processando {effective_sample_size} documentos da amostra com ambos os métodos...")
    print("-" * 60)

    comparison_records = []
    llda_detailed_records = []
    lda_detailed_records = []

    start_time_llda = time.time()

    for i, idx in enumerate(sample_indices, start=1):
        doc_text = ds.docs[idx]
        active_cats = [ds.target_names[c] for c in ds.targets_multilabel[idx]]
        cat_str = ", ".join(active_cats)

        print(f"\n[{i}/{effective_sample_size}] Documento ID {idx} | Classes Reais: [{cat_str}]")

        # --- A. Execução LLDA (Ollama) ---
        llda_suggestions = []
        for beam in range(num_suggestions):
            topic = generate_llda_topic(doc_text, model_name=ollama_model_name, temperature=0.7)
            llda_suggestions.append(topic)

        # Frequência e confiança do LLDA
        topic_counts = pd.Series(llda_suggestions).value_counts()
        best_llda_topic = topic_counts.index[0]
        best_llda_conf = topic_counts.values[0] / num_suggestions

        for top, freq in topic_counts.items():
            llda_detailed_records.append({
                "doc_id": idx,
                "metodo": "LLDA_Ollama",
                "topico": top,
                "frequencia": freq,
                "confianca": freq / num_suggestions,
                "classes_reais": cat_str
            })

        print(f"  → [LLDA] Tópico Predominante: '{best_llda_topic}' (Confiança: {best_llda_conf:.2f})")

        # --- B. Execução LDA Clássico ---
        df_doc_lda = lda_modeler.process_document(idx)
        best_lda_topic = df_doc_lda.iloc[0]["topico_sugerido"]
        best_lda_prob = df_doc_lda.iloc[0]["confianca"]
        best_lda_id = df_doc_lda.iloc[0]["topico_id"]

        for _, row in df_doc_lda.iterrows():
            lda_detailed_records.append({
                "doc_id": idx,
                "metodo": "LDA_Classico",
                "topico_id": row["topico_id"],
                "topico": row["topico_sugerido"],
                "ranking": row["ranking"],
                "confianca": row["confianca"],
                "classes_reais": cat_str
            })

        print(f"  → [LDA Clássico] Tópico {best_lda_id}: '{best_lda_topic}' (Probabilidade: {best_lda_prob:.2%})")

        # Registro comparativo
        comparison_records.append({
            "doc_id": idx,
            "classes_reais_aapd": cat_str,
            "topico_llda": best_llda_topic,
            "confianca_llda": best_llda_conf,
            "topico_lda_classico": best_lda_topic,
            "probabilidade_lda_classico": best_lda_prob,
            "resumo_texto": doc_text[:140] + "..."
        })

    time_llda_total = time.time() - start_time_llda

    df_comp = pd.DataFrame(comparison_records)
    df_comp.to_csv(output_path, index=False)

    df_llda_all = pd.DataFrame(llda_detailed_records)
    df_llda_all.to_csv("experimento_aapd_llda_detalhado.csv", index=False)

    df_lda_all = pd.DataFrame(lda_detailed_records)
    df_lda_all.to_csv("experimento_aapd_lda_detalhado.csv", index=False)

    # 6. Métricas de Avaliação
    distinct2_llda = calculate_distinct_2(df_comp["topico_llda"].tolist())
    distinct2_lda = calculate_distinct_2(df_comp["topico_lda_classico"].tolist())

    print("\n" + "=" * 80)
    print("RESULTADOS E COMPARAÇÃO DO EXPERIMENTO")
    print("=" * 80)
    print(f"Tamanho da amostra: {effective_sample_size} documentos")
    print(f"Tempo de inferência LLDA (Ollama - {ollama_model_name}): {time_llda_total:.2f} s ({time_llda_total/effective_sample_size:.2f} s/doc)")
    print(f"Tempo de treinamento e inferência LDA Clássico: {time_lda_total:.2f} s")
    print(f"Distinct-2 (Diversidade Lexical) LLDA: {distinct2_llda:.2%}")
    print(f"Distinct-2 (Diversidade Lexical) LDA Clássico: {distinct2_lda:.2%}")

    print("\n--- AMOSTRA COMPARATIVA (PRIMEIROS 8 DOCUMENTOS) ---")
    cols_display = ["doc_id", "classes_reais_aapd", "topico_llda", "topico_lda_classico"]
    print(df_comp[cols_display].head(8).to_string(index=False))

    print(f"\nResultados salvos em:")
    print(f" - {output_path}")
    print(f" - experimento_aapd_llda_detalhado.csv")
    print(f" - experimento_aapd_lda_detalhado.csv")

    return df_comp


def main():
    parser = argparse.ArgumentParser(description="Executa experimento comparativo no dataset AAPD: LLDA vs LDA.")
    parser.add_argument("--sample-size", "-s", type=int, default=15, help="Quantidade de documentos da amostra.")
    parser.add_argument("--num-suggestions", "-n", type=int, default=3, help="Número de sugestões de tópicos por documento.")
    parser.add_argument("--seed", type=int, default=42, help="Semente para amostragem.")
    parser.add_argument("--model", "-m", type=str, default=None, help="Modelo Ollama a utilizar.")
    parser.add_argument("--output", "-o", type=str, default="experimento_aapd_comparativo.csv", help="Caminho do CSV de saída.")
    parser.add_argument("--dataset-path", type=str, default="Castor-data/datasets/AAPD", help="Caminho dos dados AAPD.")

    args = parser.parse_args()

    run_comparative_experiment(
        sample_size=args.sample_size,
        random_seed=args.seed,
        num_suggestions=args.num_suggestions,
        dataset_path=args.dataset_path,
        output_path=args.output,
        ollama_model_name=args.model
    )


if __name__ == "__main__":
    main()
