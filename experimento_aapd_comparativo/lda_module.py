# -*- coding: utf-8 -*-
"""
lda_module.py

Módulo para modelagem clássica de tópicos via Latent Dirichlet Allocation (LDA)
utilizando scikit-learn (CountVectorizer + LatentDirichletAllocation).

Compatível com o ecossistema LLDA, suportando os mesmos datasets (20 Newsgroups e AAPD)
através do `dataset_loader.py` e gerando saídas estruturadas com distribuições
documento-tópico, palavras-chave e cálculo de concordância.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for p in [BASE_DIR, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from dataset_loader import load_dataset, DatasetContainer


class LDATopicModeler:
    """
    Modelador de tópicos probabilístico baseado em Latent Dirichlet Allocation (LDA).
    """

    def __init__(
        self,
        n_topics: int = 20,
        num_suggestions: int = 5,
        output_path: str = "topic_modeling_results_lda.csv",
        dataset_name: str = "20newsgroups",
        dataset_subset: str = "all",
        dataset_path: Optional[str] = None,
        dataset_remove: Tuple[str, ...] = ("headers", "footers", "quotes"),
        max_features: int = 10000,
        min_df: int = 3,
        max_df: float = 0.85,
        max_iter: int = 25,
        learning_method: str = "batch",
        random_state: int = 42,
    ):
        """
        Inicializa o modelador LDA.

        Args:
            n_topics: Quantidade de tópicos latentes a inferir (K).
            num_suggestions: Quantos tópicos mais prováveis retornar por documento.
            output_path: Caminho do arquivo CSV de saída.
            dataset_name: Nome do dataset ('20newsgroups' ou 'aapd').
            dataset_subset: Split ('all', 'train', 'test', 'dev').
            dataset_path: Caminho para diretório de dados customizado (para AAPD).
            dataset_remove: Partes a remover no 20 Newsgroups.
            max_features: Tamanho máximo do vocabulário.
            min_df: Frequência mínima de documento para manter uma palavra.
            max_df: Frequência máxima relativa de documento para ignorar termos ubíquos.
            max_iter: Número máximo de iterações do algoritmo de inferência variacional.
            learning_method: 'batch' (mais estável) ou 'online' (mais rápido para bases gigantes).
            random_state: Semente de reproducibilidade.
        """
        self.n_topics = n_topics
        self.num_suggestions = num_suggestions
        self.output_path = output_path
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.dataset_path = dataset_path
        self.dataset_remove = dataset_remove

        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.max_iter = max_iter
        self.learning_method = learning_method
        self.random_state = random_state

        # Componentes do modelo
        self.vectorizer: Optional[CountVectorizer] = None
        self.lda_model: Optional[LatentDirichletAllocation] = None
        self.doc_topic_dist: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.topic_descriptions: Dict[int, str] = {}
        self.topic_top_words: Dict[int, List[str]] = {}

        # Carregar dataset
        self._load_dataset()

    def _load_dataset(self):
        """Carrega o dataset escolhido via dataset_loader."""
        print(f"\n[LDATopicModeler] Carregando dataset '{self.dataset_name}' (subset='{self.dataset_subset}')...")
        data: DatasetContainer = load_dataset(
            dataset_name=self.dataset_name,
            subset=self.dataset_subset,
            dataset_path=self.dataset_path,
            dataset_remove=self.dataset_remove
        )
        self.docs = data.docs
        self.target_names = data.target_names
        self.targets = data.targets
        self.targets_multilabel = data.targets_multilabel
        self.target_labels_text = data.target_labels_text

        print(f"[LDATopicModeler] Total de documentos carregados: {len(self.docs)}")
        print(f"[LDATopicModeler] Total de classes originais: {len(self.target_names)}")

    def fit(self, docs: Optional[List[str]] = None, n_top_words: int = 6):
        """
        Ajusta o vetorizador e o modelo LDA sobre os documentos.

        Args:
            docs: Lista opcional de textos. Se None, utiliza self.docs.
            n_top_words: Número de palavras para formar o rótulo descritivo de cada tópico.
        """
        if docs is None:
            docs = self.docs

        print(f"\n[LDATopicModeler] Vetorizando {len(docs)} documentos (CountVectorizer)...")
        # Token pattern para aceitar apenas termos alfabéticos com 3+ caracteres
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
        )
        dtm = self.vectorizer.fit_transform(docs)
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()
        print(f"[LDATopicModeler] Matriz Documento-Termo criada: {dtm.shape[0]} docs x {dtm.shape[1]} termos únicos.")

        print(f"[LDATopicModeler] Treinando modelo LDA com K={self.n_topics} tópicos (max_iter={self.max_iter})...")
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            learning_method=self.learning_method,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        self.doc_topic_dist = self.lda_model.fit_transform(dtm)

        # Extração das palavras mais representativas por tópico
        self._extract_topic_words(n_top_words=n_top_words)

        perplexity = self.lda_model.perplexity(dtm)
        print(f"[LDATopicModeler] Treinamento concluído. Perplexidade no corpus: {perplexity:.2f}")

    def _extract_topic_words(self, n_top_words: int = 6):
        """Extrai as palavras mais frequentes em cada tópico e constrói seus rótulos descritivos."""
        self.topic_descriptions.clear()
        self.topic_top_words.clear()

        feature_array = np.array(self.feature_names)
        for topic_idx, topic_dist in enumerate(self.lda_model.components_):
            top_word_indices = topic_dist.argsort()[: -n_top_words - 1 : -1]
            top_words = [feature_array[i] for i in top_word_indices]
            self.topic_top_words[topic_idx] = top_words
            self.topic_descriptions[topic_idx] = ", ".join(top_words)

    def get_topic_overview_dataframe(self) -> pd.DataFrame:
        """Retorna uma tabela com os tópicos descobertos e suas palavras mais representativas."""
        records = []
        for topic_idx in range(self.n_topics):
            records.append({
                "topico_id": topic_idx,
                "topico_rotulo": f"Topico_{topic_idx:02d}",
                "palavras_chave": self.topic_descriptions.get(topic_idx, ""),
                "top_palavras_lista": self.topic_top_words.get(topic_idx, [])
            })
        return pd.DataFrame(records)

    def process_document(self, idx: int) -> Optional[pd.DataFrame]:
        """
        Retorna as principais sugestões de tópicos para um documento específico com suas confianças.

        Args:
            idx: Índice do documento.

        Returns:
            pd.DataFrame contendo doc_id, topico_id, topico_sugerido, confianca e ranking.
        """
        if self.doc_topic_dist is None:
            raise RuntimeError("O modelo LDA ainda não foi treinado. Execute fit() primeiro.")

        doc_probs = self.doc_topic_dist[idx]
        k_suggestions = min(self.num_suggestions, len(doc_probs))

        # Índices ordenados por probabilidade decrescente
        top_indices = np.argsort(doc_probs)[::-1][:k_suggestions]

        records = []
        for rank, topic_id in enumerate(top_indices, start=1):
            records.append({
                "doc_id": idx,
                "ranking": rank,
                "topico_id": int(topic_id),
                "topico_sugerido": self.topic_descriptions.get(topic_id, f"Topico_{topic_id}"),
                "frequencia": 1,
                "confianca": float(doc_probs[topic_id]),
                "alvo_real": self.target_labels_text[idx] if self.target_labels_text else None
            })

        return pd.DataFrame(records)

    def run_experiment(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Executa a inferência para todos os documentos, formata no padrão esperado
        pelos scripts de análise de diversidade e salva em CSV.

        Args:
            output_path: Caminho opcional para salvar o arquivo CSV.

        Returns:
            pd.DataFrame com todos os tópicos e confianças inferidos.
        """
        out_file = output_path or self.output_path

        if self.doc_topic_dist is None:
            self.fit()

        print(f"\n[LDATopicModeler] Gerando DataFrame de resultados para {len(self.docs)} documentos...")
        all_records = []
        for idx in range(len(self.docs)):
            doc_df = self.process_document(idx)
            if doc_df is not None:
                all_records.append(doc_df)

        df_final = pd.concat(all_records, ignore_index=True)

        if out_file:
            print(f"[LDATopicModeler] Salvando resultados em: '{out_file}'...")
            df_final.to_csv(out_file, index=False)
            print("[LDATopicModeler] Arquivo salvo com sucesso!")

        return df_final

    def evaluate_agreement(self) -> Tuple[pd.DataFrame, float]:
        """
        Calcula a concordância (Purity / Agreement Ratio) entre o tópico predominante do LDA
        e as classes reais do benchmark.

        Returns:
            Tuple contendo:
              1. DataFrame com agreement_ratio e classe predominante por tópico.
              2. Concordância média do modelo.
        """
        if self.doc_topic_dist is None:
            raise RuntimeError("O modelo precisa ser ajustado com fit() antes de avaliar.")

        if not self.targets:
            raise ValueError("O dataset atual não possui classes de referência (targets).")

        pred_topics = np.argmax(self.doc_topic_dist, axis=1)
        true_labels = self.targets

        df_comparison = pd.DataFrame({
            "pred_topic": pred_topics,
            "true_label": true_labels
        })

        ctab = pd.crosstab(df_comparison["pred_topic"], df_comparison["true_label"])
        agreement_records = []

        for topic_id in ctab.index:
            row = ctab.loc[topic_id]
            total_in_topic = row.sum()
            max_intersection = row.max()
            agreement = max_intersection / total_in_topic if total_in_topic > 0 else 0.0
            predominant_class = row.idxmax()
            predominant_name = (
                self.target_names[predominant_class]
                if predominant_class < len(self.target_names)
                else str(predominant_class)
            )

            agreement_records.append({
                "topico_id": topic_id,
                "palavras_chave": self.topic_descriptions.get(topic_id, ""),
                "total_documentos": total_in_topic,
                "intersecao_maxima": max_intersection,
                "agreement_ratio": agreement,
                "classe_predominante": predominant_name
            })

        df_agreement = pd.DataFrame(agreement_records)
        mean_agreement = df_agreement["agreement_ratio"].mean() if not df_agreement.empty else 0.0

        return df_agreement, mean_agreement


def main():
    parser = argparse.ArgumentParser(
        description="Modelagem de Tópicos via Latent Dirichlet Allocation (LDA) para 20 Newsgroups e AAPD."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="20newsgroups",
        choices=["20newsgroups", "aapd"],
        help="Dataset a ser utilizado ('20newsgroups' ou 'aapd'). Default: '20newsgroups'."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Caminho para o diretório do dataset AAPD (se aplicável)."
    )
    parser.add_argument(
        "--n-topics",
        "-k",
        type=int,
        default=20,
        help="Número de tópicos a descobrir (K). Default: 20."
    )
    parser.add_argument(
        "--num-suggestions",
        "-n",
        type=int,
        default=5,
        help="Quantos tópicos mais prováveis salvar por documento. Default: 5."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="topic_modeling_results_lda.csv",
        help="Arquivo CSV para salvar os resultados."
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20,
        help="Número de iterações do LDA. Default: 20."
    )
    parser.add_argument(
        "--evaluate-agreement",
        action="store_true",
        help="Calcula a concordância (agreement ratio) com os rótulos reais."
    )

    args = parser.parse_args()

    # Ajusta n_topics padrão para AAPD se não especificado explicitamente
    n_topics = args.n_topics
    if args.dataset == "aapd" and args.n_topics == 20:
        n_topics = 54  # AAPD possui 54 categorias padrão

    modeler = LDATopicModeler(
        n_topics=n_topics,
        num_suggestions=args.num_suggestions,
        output_path=args.output,
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        max_iter=args.max_iter,
    )

    print("\n" + "=" * 70)
    print("TREINANDO MODELO LDA")
    print("=" * 70)
    modeler.fit()

    print("\n" + "=" * 70)
    print(f"TOP 10 TÓPICOS DESCOBERTOS (TOTAL: {modeler.n_topics})")
    print("=" * 70)
    overview_df = modeler.get_topic_overview_dataframe()
    print(overview_df.head(10)[["topico_id", "palavras_chave"]].to_string(index=False))

    # Executa e salva o processamento completo
    df_results = modeler.run_experiment()

    print("\n" + "=" * 70)
    print("PRIMEIRAS LINHAS DOS RESULTADOS GERADOS")
    print("=" * 70)
    print(df_results.head(10)[["doc_id", "ranking", "topico_id", "topico_sugerido", "confianca"]].to_string(index=False))

    if args.evaluate_agreement or True:
        try:
            df_agree, mean_agree = modeler.evaluate_agreement()
            print("\n" + "=" * 70)
            print(f"CONCORDÂNCIA COM AS CLASSES REAIS (Média: {mean_agree:.2%})")
            print("=" * 70)
            print(df_agree.sort_values(by="agreement_ratio", ascending=False).head(10)[["topico_id", "classe_predominante", "agreement_ratio", "total_documentos"]].to_string(index=False))
        except Exception as e:
            print(f"\nNão foi possível calcular concordância: {e}")


if __name__ == "__main__":
    main()

