"""Verificações pequenas dos contratos que afetam a atribuição de classes."""
import unittest
from types import SimpleNamespace
import numpy as np
from seeded_lda import SeededLDA
from guided_bertopic import GuidedBERTopicClassifier
from experimento import load_data, select_k, evaluate, predict_top_k
from topicos_predefinidos import CLASSES


class ClassificationTests(unittest.TestCase):
    def test_seeded_prior_and_assignment(self):
        documents = ['apple orchard fruit apple'] * 6 + ['rocket orbit space rocket'] * 6
        model = SeededLDA(['fruit', 'space'], [['apple', 'fruit'], ['rocket', 'space']],
                          max_iter=5, seed_strength=50, random_state=42).fit(documents)
        scores = model.transform(['apple fruit orchard', 'rocket space orbit', 'unseenword'])
        np.testing.assert_array_equal(scores[:2].argmax(1), [0, 1])
        np.testing.assert_allclose(scores.sum(1), 1)
        np.testing.assert_allclose(scores[2], [0.5, 0.5])
        self.assertTrue(np.all(model.components_ >= model.prior_))
        self.assertGreater(model.prior_[0, model.vectorizer_.vocabulary_['apple']], model.eta)
        repeated = SeededLDA(['fruit', 'space'], [['apple', 'fruit'], ['rocket', 'space']],
                             max_iter=5, seed_strength=50, random_state=42).fit(documents)
        np.testing.assert_allclose(scores, repeated.transform(['apple fruit orchard', 'rocket space orbit', 'unseenword']))

    def test_guided_projection_and_outlier(self):
        model = GuidedBERTopicClassifier(['a', 'b', 'c'], ['a', 'b', 'c'], [['a'], ['b'], ['c']])
        model.topic_ids_ = [0, 1]
        model.projection_ = np.array([[0.5, 0.5, 0], [0, 0, 1]])
        model.model_ = SimpleNamespace(transform=lambda docs: ([0, -1], np.array([[0.8, 0.1], [0, 0]])))
        scores = model.transform(['first', 'outlier'])
        np.testing.assert_allclose(scores, [[0.4, 0.4, 0.1], [0, 0, 0]])
        self.assertTrue(model.outliers_[1])
        self.assertEqual(predict_top_k(scores, 2)[1].sum(), 0)

    def test_split_and_validation(self):
        df, y, split = load_data()
        self.assertEqual(y.shape, (2000, 54))
        self.assertEqual(len(CLASSES), 54)
        sets = [set(indices) for indices in split.values()]
        self.assertEqual([len(s) for s in sets], [1200, 300, 500])
        self.assertEqual(len(set.union(*sets)), len(df))
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        truth = np.eye(len(CLASSES), dtype=int)
        k, _ = select_k(truth, truth.astype(float))
        self.assertEqual(k, 1)
        metrics, _, _ = evaluate(truth, truth.astype(float), k)
        self.assertEqual(metrics['LRAP'], 1)
        self.assertEqual(metrics['F1 micro'], 1)


if __name__ == '__main__':
    unittest.main()
