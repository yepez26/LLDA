# -*- coding: utf-8 -*-
"""
bertopic_aggregator.py

Módulo e script para agregação e refinamento de tópicos gerados por LLMs
utilizando BERTopic, UMAP, HDBSCAN e SentenceTransformers.

Recebe um conjunto de documentos com tópicos sugeridos (ex: saídas de Ollama ou Transformers),
realiza a modelagem/clusterização de tópicos e retorna os dados com as novas marcações.
"""

import sys
import os
import re
import argparse
import numpy as np
import pandas as pd
from typing import Optional, List, Union, Tuple, Dict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for p in [BASE_DIR, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Verificação de dependências com mensagens amigáveis
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from bertopic import BERTopic
except ImportError:
    BERTopic = None

try:
    from umap import UMAP
except ImportError:
    UMAP = None

try:
    from hdbscan import HDBSCAN
except ImportError:
    HDBSCAN = None

from dataset_loader import load_dataset


def clean_topic_text(topic: Union[str, float]) -> str:
    """
    Limpa o texto do tópico removendo aspas externas, espaços duplicados e pontuações estranhas.
    """
    if pd.isna(topic):
        return ""
    text = str(topic).strip()
    # Remove aspas externas simples, duplas ou triplas
    text = text.strip('"\'`')
    # Normaliza múltiplos espaços
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class BERTopicAggregator:
    """
    Classe responsável por aplicar BERTopic sobre tópicos sugeridos por LLMs
    para descobrir macro-tópicos/clusters e enriquecer os dados.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        n_neighbors: int = 5,
        n_components: int = 2,
        min_dist: float = 0.0,
        metric: str = "cosine",
        min_cluster_size: int = 50,
        nr_topics: Union[str, int] = "auto",
        language: str = "multilingual",
        random_state: int = 42,
    ):
        """
        Inicializa o agregador BERTopic com os parâmetros de UMAP e HDBSCAN.

        Args:
            embedding_model_name: Nome do modelo de SentenceTransformer (ex: 'all-MiniLM-L6-v2' ou 'paraphrase-multilingual-mpnet-base-v2').
            n_neighbors: Número de vizinhos no UMAP.
            n_components: Dimensões reduzidas no UMAP.
            min_dist: Distância mínima no UMAP.
            metric: Métrica de distância no UMAP (default: 'cosine').
            min_cluster_size: Tamanho mínimo do cluster no HDBSCAN.
            nr_topics: Redução de tópicos ('auto' ou int fixo).
            language: Idioma para representações do BERTopic ('multilingual' ou 'english').
            random_state: Semente de reprodutibilidade.
        """
        self._check_dependencies()
        self.embedding_model_name = embedding_model_name
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.metric = metric
        self.min_cluster_size = min_cluster_size
        self.nr_topics = nr_topics
        self.language = language
        self.random_state = random_state

        self.embedding_model: Optional[SentenceTransformer] = None
        self.topic_model: Optional[BERTopic] = None
        self.umap_model: Optional[UMAP] = None
        self.hdbscan_model: Optional[HDBSCAN] = None

    def _check_dependencies(self):
        """Valida se as bibliotecas necessárias estão instaladas."""
        missing = []
        if BERTopic is None:
            missing.append("bertopic")
        if SentenceTransformer is None:
            missing.append("sentence-transformers")
        if UMAP is None:
            missing.append("umap-learn")
        if HDBSCAN is None:
            missing.append("hdbscan")

        if missing:
            raise ImportError(
                f"As seguintes bibliotecas são necessárias para o BERTopicAggregator: {', '.join(missing)}.\n"
                f"Instale-as executando: pip install {' '.join(missing)}"
            )

    def _init_models(self, dataset_size: int):
        """Instancia os componentes do pipeline ajustando parâmetros dinamicamente se o dataset for pequeno."""
        print(f"\n[BERTopicAggregator] Carregando modelo de embeddings: '{self.embedding_model_name}'...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Ajusta parâmetros caso o número de itens seja menor que as configurações padrão
        effective_n_neighbors = min(self.n_neighbors, max(2, dataset_size - 1))
        effective_min_cluster_size = min(self.min_cluster_size, max(2, dataset_size // 5))

        if effective_min_cluster_size != self.min_cluster_size:
            print(
                f"[BERTopicAggregator] Ajustando min_cluster_size de {self.min_cluster_size} para "
                f"{effective_min_cluster_size} devido ao tamanho do dataset ({dataset_size} amostras)."
            )

        self.umap_model = UMAP(
            n_neighbors=effective_n_neighbors,
            n_components=self.n_components,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        )

        self.hdbscan_model = HDBSCAN(
            min_cluster_size=effective_min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        self.topic_model = BERTopic(
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            embedding_model=self.embedding_model,
            nr_topics=self.nr_topics,
            language=self.language,
            calculate_probabilities=False,
        )

    def fit_transform_dataframe(
        self,
        df: pd.DataFrame,
        topic_column: str = "topico_sugerido",
        doc_id_column: Optional[str] = "doc_id",
        attach_dataset_labels: Optional[str] = None,  # '20newsgroups' ou 'aapd'
        attach_20newsgroups_labels: bool = False,  # Retrocompatibilidade
        dataset_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Executa a modelagem de tópicos sobre uma coluna do DataFrame e adiciona as novas marcações.

        Args:
            df: DataFrame contendo os documentos/tópicos sugeridos.
            topic_column: Nome da coluna contendo os tópicos sugeridos pelo LLM.
            doc_id_column: Nome da coluna do ID do documento (opcional).
            attach_dataset_labels: Nome do dataset para anexar labels reais ('20newsgroups' ou 'aapd').
            attach_20newsgroups_labels: Se True, anexa labels do 20 Newsgroups (retrocompatibilidade).
            dataset_path: Caminho da pasta do dataset (para AAPD).

        Returns:
            pd.DataFrame enriquecido com:
              - 'topico_limpo': Texto do tópico higienizado.
              - 'bertopic_cluster': ID do tópico/cluster (inteiro, -1 indica ruído/outlier).
              - 'bertopic_topic_name': Nome gerado para o macro-tópico (ex: "0_space_nasa_orbit").
              - 'bertopic_top_words': Principais palavras-chave do cluster.
        """
        df_result = df.copy()

        if topic_column not in df_result.columns:
            raise ValueError(f"A coluna '{topic_column}' não foi encontrada no DataFrame. Colunas: {list(df_result.columns)}")

        # 1. Limpeza do texto dos tópicos
        df_result["topico_limpo"] = df_result[topic_column].apply(clean_topic_text)
        topic_texts = df_result["topico_limpo"].tolist()

        if len(topic_texts) == 0:
            raise ValueError("O DataFrame fornecido está vazio.")

        # 2. Inicializar modelos
        self._init_models(dataset_size=len(topic_texts))

        # 3. Gerar embeddings
        print(f"[BERTopicAggregator] Gerando embeddings para {len(topic_texts)} entradas...")
        embeddings = self.embedding_model.encode(
            topic_texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        # 4. Ajustar modelo BERTopic
        print("[BERTopicAggregator] Ajustando modelo BERTopic...")
        clusters, _ = self.topic_model.fit_transform(topic_texts, embeddings=embeddings)

        # 5. Adicionar IDs dos clusters
        df_result["bertopic_cluster"] = clusters

        # 6. Mapear informações textuais dos tópicos
        topic_info = self.topic_model.get_topic_info().set_index("Topic")
        
        # Mapeamento do nome do tópico
        topic_names_dict = topic_info["Name"].to_dict()
        df_result["bertopic_topic_name"] = df_result["bertopic_cluster"].map(topic_names_dict).fillna("Outlier")

        # Mapeamento das palavras-chave
        if "Representation" in topic_info.columns:
            topic_rep_dict = topic_info["Representation"].apply(
                lambda rep: ", ".join(rep[:5]) if isinstance(rep, (list, tuple)) else str(rep)
            ).to_dict()
            df_result["bertopic_top_words"] = df_result["bertopic_cluster"].map(topic_rep_dict).fillna("")
        else:
            df_result["bertopic_top_words"] = ""

        # 7. Opcional: Associar labels originais (20 Newsgroups ou AAPD)
        attach_ds = attach_dataset_labels or ("20newsgroups" if attach_20newsgroups_labels else None)
        if attach_ds and doc_id_column and doc_id_column in df_result.columns:
            try:
                print(f"[BERTopicAggregator] Buscando labels originais do dataset '{attach_ds}'...")
                ds_container = load_dataset(dataset_name=attach_ds, dataset_path=dataset_path)
                
                target_df = pd.DataFrame({
                    doc_id_column: range(len(ds_container.targets)),
                    "target_id": ds_container.targets,
                    "target_name": ds_container.target_labels_text
                })
                # Assegura que o tipo de dado de doc_id coincida
                target_df[doc_id_column] = target_df[doc_id_column].astype(df_result[doc_id_column].dtype)
                df_result = pd.merge(df_result, target_df, on=doc_id_column, how="left")
                print(f"[BERTopicAggregator] Labels do dataset '{attach_ds}' adicionadas com sucesso.")
            except Exception as e:
                print(f"[BERTopicAggregator] Aviso ao carregar labels do dataset '{attach_ds}': {e}")

        n_clusters = len(set(clusters) - {-1})
        n_outliers = (df_result["bertopic_cluster"] == -1).sum()
        print(f"\n[BERTopicAggregator] Concluído: {n_clusters} tópicos/clusters descobertos ({n_outliers} itens como ruído/outlier).")

        return df_result

    def get_document_level_summary(
        self,
        df_annotated: pd.DataFrame,
        doc_id_column: str = "doc_id",
        confidence_column: str = "confianca",
    ) -> pd.DataFrame:
        """
        Gera uma visão condensada onde cada documento aparece apenas uma vez,
        selecionando a marcação de tópico de maior confiança/frequência.

        Args:
            df_annotated: DataFrame anotado com as colunas do BERTopic.
            doc_id_column: Coluna identificadora do documento.
            confidence_column: Coluna de confiança utilizada para desempate/ordenação.

        Returns:
            pd.DataFrame com 1 linha por documento e os metadados do tópico predominante.
        """
        if doc_id_column not in df_annotated.columns:
            return df_annotated

        df_sorted = df_annotated.copy()
        if confidence_column in df_sorted.columns:
            df_sorted = df_sorted.sort_values(by=[doc_id_column, confidence_column], ascending=[True, False])

        # Mantém a sugestão de maior confiança para cada doc_id
        df_doc_level = df_sorted.drop_duplicates(subset=[doc_id_column], keep="first").copy()
        return df_doc_level

    def evaluate_agreement(
        self,
        df_annotated: pd.DataFrame,
        doc_id_column: str = "doc_id",
        target_column: str = "target_id",
    ) -> Tuple[pd.DataFrame, float]:
        """
        Calcula a métrica de concordância (Agreement Ratio / Purity) entre os clusters do BERTopic
        e os rótulos reais de classificação.

        Args:
            df_annotated: DataFrame anotado.
            doc_id_column: Coluna de ID do documento.
            target_column: Coluna de classe real (ex: 'target_id' ou 'target_name').

        Returns:
            Tuple contendo:
              1. DataFrame com agreement_ratio por cluster
              2. Concordância média (excluindo ruído -1)
        """
        if target_column not in df_annotated.columns:
            raise ValueError(f"A coluna de rótulos '{target_column}' não está presente no DataFrame.")

        # Utiliza o nível de documento único para cálculo justo
        df_unique = self.get_document_level_summary(df_annotated, doc_id_column=doc_id_column)
        
        ctab = pd.crosstab(df_unique["bertopic_cluster"], df_unique[target_column])
        agreement_records = []

        for topic_id in ctab.index:
            row = ctab.loc[topic_id]
            total_in_topic = row.sum()
            max_intersection = row.max()
            agreement = max_intersection / total_in_topic if total_in_topic > 0 else 0.0
            predominant_target = row.idxmax()

            agreement_records.append({
                "bertopic_cluster": topic_id,
                "total_docs": total_in_topic,
                "max_intersection": max_intersection,
                "agreement_ratio": agreement,
                "predominant_target": predominant_target,
            })

        df_agreement = pd.DataFrame(agreement_records)
        valid_topics = df_agreement[df_agreement["bertopic_cluster"] != -1]
        mean_agreement = valid_topics["agreement_ratio"].mean() if not valid_topics.empty else 0.0

        return df_agreement, mean_agreement


def process_topics_csv(
    input_csv: str,
    output_csv: str,
    topic_column: str = "topico_sugerido",
    doc_id_column: str = "doc_id",
    embedding_model_name: str = "all-MiniLM-L6-v2",
    min_cluster_size: int = 50,
    nr_topics: Union[str, int] = "auto",
    attach_dataset_labels: Optional[str] = None,
    attach_20newsgroups_labels: bool = False,
    dataset_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Função utilitária direta para ler um arquivo CSV, aplicar o BERTopicAggregator e salvar o resultado.

    Args:
        input_csv: Caminho para o CSV de entrada.
        output_csv: Caminho para salvar o CSV anotado com os tópicos do BERTopic.
        topic_column: Nome da coluna que contém os tópicos sugeridos pelo LLM.
        doc_id_column: Nome da coluna do doc_id.
        embedding_model_name: Nome do modelo de embeddings.
        min_cluster_size: Tamanho mínimo do cluster.
        nr_topics: 'auto' ou número fixo de tópicos.
        attach_dataset_labels: Nome do dataset ('20newsgroups' ou 'aapd').
        attach_20newsgroups_labels: Se True, anexa labels do 20 Newsgroups (retrocompatibilidade).
        dataset_path: Caminho da pasta do dataset (para AAPD).

    Returns:
        pd.DataFrame com as anotações do BERTopic.
    """
    print(f"\n[process_topics_csv] Lendo '{input_csv}'...")
    df = pd.read_csv(input_csv)

    aggregator = BERTopicAggregator(
        embedding_model_name=embedding_model_name,
        min_cluster_size=min_cluster_size,
        nr_topics=nr_topics,
    )

    df_annotated = aggregator.fit_transform_dataframe(
        df=df,
        topic_column=topic_column,
        doc_id_column=doc_id_column,
        attach_dataset_labels=attach_dataset_labels,
        attach_20newsgroups_labels=attach_20newsgroups_labels,
        dataset_path=dataset_path,
    )

    print(f"\n[process_topics_csv] Salvando resultados em '{output_csv}'...")
    df_annotated.to_csv(output_csv, index=False)
    print(f"[process_topics_csv] Concluído com sucesso!")

    return df_annotated


def main():
    parser = argparse.ArgumentParser(
        description="Aplica modelagem de tópicos BERTopic sobre tópicos gerados por LLMs e anota o dataset."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="topic_modeling_results_ollama.csv",
        help="Caminho do arquivo CSV de entrada com os tópicos gerados.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="topic_modeling_bertopic_annotated.csv",
        help="Caminho do arquivo CSV de saída para salvar os dados enriquecidos.",
    )
    parser.add_argument(
        "--topic-col",
        type=str,
        default="topico_sugerido",
        help="Nome da coluna que contém o texto dos tópicos (default: 'topico_sugerido').",
    )
    parser.add_argument(
        "--doc-id-col",
        type=str,
        default="doc_id",
        help="Nome da coluna com o ID do documento (default: 'doc_id').",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Modelo do SentenceTransformer a ser utilizado (default: 'all-MiniLM-L6-v2').",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=50,
        help="Tamanho mínimo de cluster para o HDBSCAN (default: 50).",
    )
    parser.add_argument(
        "--nr-topics",
        type=str,
        default="auto",
        help="Redução de tópicos no BERTopic ('auto' ou número inteiro, default: 'auto').",
    )
    parser.add_argument(
        "--attach-dataset",
        type=str,
        default=None,
        help="Nome do dataset para buscar classes reais ('20newsgroups' ou 'aapd').",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Caminho para os arquivos do dataset (para AAPD, ex: 'Castor-data/datasets/AAPD').",
    )
    parser.add_argument(
        "--attach-20newsgroups",
        action="store_true",
        help="Se informado, busca e mescla as classes reais do 20 Newsgroups via scikit-learn.",
    )
    parser.add_argument(
        "--evaluate-agreement",
        action="store_true",
        help="Se informado e as classes reais estiverem presentes, calcula e exibe o agreement ratio.",
    )

    args = parser.parse_args()

    nr_topics_val: Union[str, int] = int(args.nr_topics) if args.nr_topics.isdigit() else args.nr_topics

    if not os.path.exists(args.input):
        print(f"Erro: Arquivo de entrada '{args.input}' não encontrado.")
        return

    attach_ds = args.attach_dataset or ("20newsgroups" if args.attach_20newsgroups else None)

    df_annotated = process_topics_csv(
        input_csv=args.input,
        output_csv=args.output,
        topic_column=args.topic_col,
        doc_id_column=args.doc_id_col,
        embedding_model_name=args.embedding_model,
        min_cluster_size=args.min_cluster_size,
        nr_topics=nr_topics_val,
        attach_dataset_labels=attach_ds,
        dataset_path=args.dataset_path,
    )

    print("\n--- Primeiras 10 linhas anotadas ---")
    cols_to_show = [col for col in [args.doc_id_col, args.topic_col, "bertopic_cluster", "bertopic_topic_name", "target_name"] if col in df_annotated.columns]
    print(df_annotated[cols_to_show].head(10).to_string(index=False))

    if args.evaluate_agreement and "target_id" in df_annotated.columns:
        aggregator = BERTopicAggregator()
        df_agree, mean_agree = aggregator.evaluate_agreement(
            df_annotated, doc_id_column=args.doc_id_col, target_column="target_id"
        )
        print("\n" + "=" * 70)
        print(f"CONCORDÂNCIA COM AS CLASSES REAIS (Média: {mean_agree:.2%}):")
        print("=" * 70)
        print(df_agree.sort_values(by="agreement_ratio", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()

