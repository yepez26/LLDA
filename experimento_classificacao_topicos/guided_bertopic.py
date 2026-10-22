"""Guided BERTopic real + projeção explícita dos clusters nas classes fixas."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class GuidedBERTopicClassifier:
    def __init__(self, classes, descriptions, seed_topics,
                 embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                 min_cluster_size=15, random_state=42, local_files_only=False):
        if not (len(classes) == len(descriptions) == len(seed_topics)):
            raise ValueError('Classes, descrições e sementes devem ter o mesmo tamanho.')
        self.classes, self.descriptions, self.seed_topics = list(classes), descriptions, seed_topics
        self.embedding_model = embedding_model
        self.min_cluster_size, self.random_state = min_cluster_size, random_state
        self.local_files_only = local_files_only

    def fit(self, documents):
        from sentence_transformers import SentenceTransformer
        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN
        self.encoder_ = SentenceTransformer(self.embedding_model, device='cpu',
                                            local_files_only=self.local_files_only)
        self.model_ = BERTopic(
            embedding_model=self.encoder_, seed_topic_list=self.seed_topics,
            umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                            metric='cosine', random_state=self.random_state),
            hdbscan_model=HDBSCAN(min_cluster_size=self.min_cluster_size, metric='euclidean',
                                 cluster_selection_method='eom', prediction_data=True),
            vectorizer_model=CountVectorizer(stop_words='english', ngram_range=(1, 2)),
            calculate_probabilities=True, verbose=True,
        )
        self.model_.fit_transform(list(documents))  # nenhum y real é passado
        self.topic_ids_ = sorted(t for t in self.model_.get_topics() if t != -1)
        if not self.topic_ids_:
            raise RuntimeError('Guided BERTopic encontrou somente outliers; ajuste min_cluster_size no treino.')
        # Representações aprendidas no treino, comparadas aos descritores predefinidos.
        topic_texts = [' '.join(w for w, _ in self.model_.get_topic(t)[:10]) for t in self.topic_ids_]
        descriptor_texts = [desc + ': ' + ', '.join(seeds)
                            for desc, seeds in zip(self.descriptions, self.seed_topics)]
        topic_emb = self.encoder_.encode(topic_texts, normalize_embeddings=True)
        class_emb = self.encoder_.encode(descriptor_texts, normalize_embeddings=True)
        similarities = topic_emb @ class_emb.T
        # Empates entre descrições idênticas compartilham a massa, sem favorecer o primeiro ID.
        maxima = similarities.max(axis=1, keepdims=True)
        tied = np.isclose(similarities, maxima, atol=1e-6, rtol=0)
        self.projection_ = tied / tied.sum(axis=1, keepdims=True)
        self.mapping_ = pd.DataFrame({
            'Cluster': self.topic_ids_, 'Palavras': topic_texts,
            'Classes atribuídas': [', '.join(np.array(self.classes)[row]) for row in tied],
            'Similaridade': maxima[:, 0],
        })
        return self

    def transform(self, documents):
        topics, membership = self.model_.transform(list(documents))
        membership = np.asarray(membership)
        # BERTopic ordena as colunas pelos IDs dos tópicos, excluindo -1.
        if membership.ndim != 2 or membership.shape[1] != len(self.topic_ids_):
            raise RuntimeError('Matriz de pertinência inesperada; verifique a versão do BERTopic/HDBSCAN.')
        self.outliers_ = np.asarray(topics) == -1
        scores = membership @ self.projection_
        # Massa restante é rejeição/outlier; não renormalizamos nem a tratamos como classe.
        self.zero_scores_ = scores.sum(axis=1) <= 0
        return scores
