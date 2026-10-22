# -*- coding: utf-8 -*-
"""
experimento_comparativo_aapd.py

Experimento comparativo de modelagem de tópicos no dataset AAPD
(ArXiv Academic Paper Dataset), comparando nos 2.000 documentos:
1. LLDA (Topic Modeling baseado em LLM via Ollama - LLDA.py)
2. LDA Clássico (Latent Dirichlet Allocation probabilístico via lda_module.py)

Inclui:
- Salvamento incremental em CSV a cada documento (checkpointing)
- Retomada automática de progresso caso interrompido
- Cálculo de métricas de diversidade (Distinct-2)
- Saída organizada na pasta do experimento
"""

import sys
import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Set

# Garante que o diretório raiz e o diretório atual estejam no sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for p in [BASE_DIR, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

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
            {"role": "user", "content": doc_text[:1500]}  # Trunca para otimizar prompt processing
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


def load_processed_doc_ids(csv_path: str) -> Set[int]:
    """Lê os doc_ids já salvos para permitir retomada rápida."""
    if not os.path.exists(csv_path):
        return set()
    try:
        df_existing = pd.read_csv(csv_path)
        if "doc_id" in df_existing.columns:
            return set(df_existing["doc_id"].dropna().astype(int).tolist())
    except Exception as e:
        print(f"Aviso ao ler CSV existente '{csv_path}': {e}")
    return set()


def run_comparative_experiment(
    sample_size: int = 2000,
    random_seed: int = 42,
    num_suggestions: int = 2,
    dataset_path: str = "Castor-data/datasets/AAPD",
    output_path: Optional[str] = None,
    ollama_model_name: Optional[str] = None
):
    if output_path is None:
        output_path = os.path.join(CURRENT_DIR, "experimento_aapd_comparativo.csv")
    llda_detailed_path = os.path.join(CURRENT_DIR, "experimento_aapd_llda_detalhado.csv")
    lda_detailed_path = os.path.join(CURRENT_DIR, "experimento_aapd_lda_detalhado.csv")

    print("=" * 80)
    print("INICIANDO EXPERIMENTO COMPARATIVO NO AAPD: LLDA vs LDA CLÁSSICO")
    print("=" * 80)

    # 1. Carregar dataset AAPD
    ds = load_dataset("aapd", dataset_path=dataset_path, subset="all")
    total_docs = len(ds.docs)
    print(f"\nTotal de documentos no dataset: {total_docs}")
    print(f"Total de categorias do arXiv: {len(ds.target_names)}")

    # 2. Selecionar a amostra (se sample_size >= total_docs, usa todos os documentos)
    if sample_size >= total_docs:
        sample_indices = list(range(total_docs))
        print(f"\n[Amostragem] Executando em TODOS os {total_docs} documentos do dataset.")
    else:
        random.seed(random_seed)
        sample_indices = sorted(random.sample(range(total_docs), sample_size))
        print(f"\n[Amostragem] Amostra de {len(sample_indices)} documentos sorteados.")

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
        num_suggestions=max(3, num_suggestions),
        dataset_name="aapd",
        dataset_path=dataset_path,
        max_iter=20,
        random_state=random_seed
    )
    lda_modeler.fit()
    time_lda_total = time.time() - start_time_lda
    print(f"[LDA Clássico] Treinamento concluído em {time_lda_total:.2f} segundos.")

    # Gera e salva a inferência de todo o LDA de uma só vez
    if not os.path.exists(lda_detailed_path):
        print(f"[LDA Clássico] Salvando detalhamento de inferência em '{lda_detailed_path}'...")
        df_lda_all = lda_modeler.run_experiment(output_path=lda_detailed_path)
    else:
        print(f"[LDA Clássico] Arquivo '{lda_detailed_path}' já existe.")

    # 5. Checkpointing e Retomada para LLDA
    processed_ids = load_processed_doc_ids(output_path)
    if processed_ids:
        print(f"\n[Checkpointing] Encontrados {len(processed_ids)} documentos já processados em '{output_path}'.")
        print("Retomando processamento dos documentos restantes...")

    remaining_indices = [idx for idx in sample_indices if idx not in processed_ids]
    print(f"[Execução] Documentos pendentes para inferência com LLDA: {len(remaining_indices)}")

    start_time_llda = time.time()
    count_processed = len(processed_ids)
    total_target = len(sample_indices)

    for i, idx in enumerate(remaining_indices, start=1):
        doc_text = ds.docs[idx]
        active_cats = [ds.target_names[c] for c in ds.targets_multilabel[idx]]
        cat_str = ", ".join(active_cats)
        count_processed += 1

        pct = (count_processed / total_target) * 100
        print(f"\n[{count_processed}/{total_target}] ({pct:.1f}%) Doc ID {idx} | Classes: [{cat_str}]")

        # --- A. Execução LLDA (Ollama) ---
        llda_suggestions = []
        for beam in range(num_suggestions):
            try:
                topic = generate_llda_topic(doc_text, model_name=ollama_model_name, temperature=0.7)
                llda_suggestions.append(topic)
            except Exception as e:
                print(f"  Aviso no doc {idx} feixe {beam+1}: {e}")
                llda_suggestions.append("Scientific Research")

        # Contagem e confiança do LLDA
        topic_counts = pd.Series(llda_suggestions).value_counts()
        best_llda_topic = topic_counts.index[0]
        best_llda_conf = topic_counts.values[0] / num_suggestions

        print(f"  → [LLDA] Tópico Predominante: '{best_llda_topic}' (Confiança: {best_llda_conf:.2f})")

        # Salva registros detalhados do LLDA incrementalmente
        doc_llda_records = []
        for top, freq in topic_counts.items():
            doc_llda_records.append({
                "doc_id": idx,
                "metodo": "LLDA_Ollama",
                "topico": top,
                "frequencia": freq,
                "confianca": freq / num_suggestions,
                "classes_reais": cat_str
            })
        df_llda_doc = pd.DataFrame(doc_llda_records)
        df_llda_doc.to_csv(
            llda_detailed_path,
            mode="a",
            header=not os.path.exists(llda_detailed_path),
            index=False
        )

        # --- B. Execução LDA Clássico ---
        df_doc_lda = lda_modeler.process_document(idx)
        best_lda_topic = df_doc_lda.iloc[0]["topico_sugerido"]
        best_lda_prob = df_doc_lda.iloc[0]["confianca"]
        best_lda_id = df_doc_lda.iloc[0]["topico_id"]

        print(f"  → [LDA Clássico] Tópico {best_lda_id}: '{best_lda_topic}' (Probabilidade: {best_lda_prob:.2%})")

        # Registro comparativo consolidado (salvo incrementalmente)
        comp_record = [{
            "doc_id": idx,
            "classes_reais_aapd": cat_str,
            "topico_llda": best_llda_topic,
            "confianca_llda": best_llda_conf,
            "topico_lda_classico": best_lda_topic,
            "probabilidade_lda_classico": best_lda_prob,
            "resumo_texto": doc_text[:140] + "..."
        }]
        df_single_comp = pd.DataFrame(comp_record)
        df_single_comp.to_csv(
            output_path,
            mode="a",
            header=not os.path.exists(output_path),
            index=False
        )

    time_llda_total = time.time() - start_time_llda
    print(f"\nInferência LLDA concluída em {time_llda_total:.2f} s.")

    # 6. Métricas de Avaliação Finais
    df_final_comp = pd.read_csv(output_path)
    distinct2_llda = calculate_distinct_2(df_final_comp["topico_llda"].astype(str).tolist())
    distinct2_lda = calculate_distinct_2(df_final_comp["topico_lda_classico"].astype(str).tolist())

    print("\n" + "=" * 80)
    print("RESUMO CONSOLIDADO DO EXPERIMENTO")
    print("=" * 80)
    print(f"Total de documentos processados: {len(df_final_comp)}")
    print(f"Distinct-2 (Diversidade Lexical) LLDA: {distinct2_llda:.2%}")
    print(f"Distinct-2 (Diversidade Lexical) LDA Clássico: {distinct2_lda:.2%}")

    print(f"\nResultados salvos em:")
    print(f" - {output_path}")
    print(f" - {llda_detailed_path}")
    print(f" - {lda_detailed_path}")

    return df_final_comp


def main():
    parser = argparse.ArgumentParser(description="Executa experimento comparativo no dataset AAPD: LLDA vs LDA.")
    parser.add_argument("--sample-size", "-s", type=int, default=2000, help="Quantidade de documentos da amostra (default: 2000).")
    parser.add_argument("--num-suggestions", "-n", type=int, default=2, help="Número de sugestões de tópicos por documento (default: 2).")
    parser.add_argument("--seed", type=int, default=42, help="Semente para amostragem.")
    parser.add_argument("--model", "-m", type=str, default=None, help="Modelo Ollama a utilizar.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Caminho do CSV de saída.")
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
