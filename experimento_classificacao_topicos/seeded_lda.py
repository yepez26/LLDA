"""LDA variacional com prior tópico-palavra assimétrico e persistente.

Cada linha de eta corresponde a uma classe predefinida. Não usa APIs privadas
nem apenas inicializa componentes: as sementes entram em todo passo M.
"""
import numpy as np
import pandas as pd
from scipy.special import digamma
from sklearn.feature_extraction.text import CountVectorizer


class SeededLDA:
    def __init__(self, classes, seed_topics, seed_strength=20.0, alpha=0.1,
                 eta=0.01, max_iter=20, doc_iter=30, max_features=10000,
                 random_state=42):
        if len(classes) != len(seed_topics) or len(set(classes)) != len(classes):
            raise ValueError('Informe uma lista de sementes para cada classe única.')
        if min(alpha, eta) <= 0 or seed_strength < 0 or min(max_iter, doc_iter) < 1:
            raise ValueError('Priors e iterações devem ser positivos; seed_strength >= 0.')
        self.classes = list(classes)
        self.seed_topics = seed_topics
        self.seed_strength, self.alpha, self.eta = seed_strength, alpha, eta
        self.max_iter, self.doc_iter = max_iter, doc_iter
        self.max_features, self.random_state = max_features, random_state

    def _infer(self, X, collect=False):
        k = len(self.classes)
        beta = np.exp(digamma(self.components_) - digamma(self.components_.sum(1))[:, None])
        theta = np.empty((X.shape[0], k))
        stats = np.zeros_like(beta) if collect else None
        for d in range(X.shape[0]):
            start, end = X.indptr[d:d+2]
            ids, counts = X.indices[start:end], X.data[start:end]
            gamma = np.full(k, self.alpha + counts.sum() / k)
            if len(ids):
                local_beta = beta[:, ids]
                for _ in range(self.doc_iter):
                    weights = np.exp(digamma(gamma) - digamma(gamma.sum()))[:, None] * local_beta
                    phi = weights / np.maximum(weights.sum(0), 1e-300)
                    new_gamma = self.alpha + phi @ counts
                    if np.mean(np.abs(new_gamma - gamma)) < 1e-4:
                        gamma = new_gamma
                        break
                    gamma = new_gamma
                if collect:
                    weights = np.exp(digamma(gamma) - digamma(gamma.sum()))[:, None] * local_beta
                    phi = weights / np.maximum(weights.sum(0), 1e-300)
                    stats[:, ids] += phi * counts[None, :]
            theta[d] = gamma / gamma.sum()
        return theta, stats

    def fit(self, documents):
        self.vectorizer_ = CountVectorizer(stop_words='english', min_df=2,
                                          max_features=self.max_features, ngram_range=(1, 2))
        X = self.vectorizer_.fit_transform(documents).astype(float).tocsr()
        self.prior_ = np.full((len(self.classes), X.shape[1]), self.eta)
        analyzer = self.vectorizer_.build_analyzer()
        coverage = []
        for c, seeds in enumerate(self.seed_topics):
            terms = set(term for seed in seeds for term in analyzer(seed))
            present = sorted(terms & self.vectorizer_.vocabulary_.keys())
            for term in present:
                self.prior_[c, self.vectorizer_.vocabulary_[term]] += self.seed_strength
            coverage.append({'Classe': self.classes[c], 'Sementes presentes': ', '.join(present),
                             'N presentes': len(present), 'N candidatas': len(terms)})
        self.seed_coverage_ = pd.DataFrame(coverage)
        rng = np.random.default_rng(self.random_state)
        self.components_ = self.prior_ + rng.gamma(100, 0.01, size=self.prior_.shape)
        for iteration in range(self.max_iter):
            _, stats = self._infer(X, collect=True)
            self.components_ = self.prior_ + stats
            if iteration == 0 or (iteration + 1) % 5 == 0:
                print(f'Seeded LDA: iteração {iteration + 1}/{self.max_iter}', flush=True)
        return self

    def transform(self, documents):
        X = self.vectorizer_.transform(documents).astype(float).tocsr()
        self.empty_documents_ = np.asarray(X.getnnz(axis=1) == 0)
        return self._infer(X)[0]

    def topic_words(self, n=8):
        terms = self.vectorizer_.get_feature_names_out()
        return pd.DataFrame({'Classe': self.classes,
                             'Palavras': [', '.join(terms[row.argsort()[-n:][::-1]])
                                          for row in self.components_]})
