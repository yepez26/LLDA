"""Verifica corpus completo, identidade das classes e avaliação multi-rótulo."""
import unittest
import pandas as pd
from llda_trie_54 import CLASSES, LABELS, full_dataset, evaluate_results


class FullTrieTests(unittest.TestCase):
    def test_full_corpus_and_54_unique_outputs(self):
        df = full_dataset()
        self.assertEqual(len(df), 2000)
        self.assertEqual(df.doc_id.nunique(), 2000)
        self.assertEqual(len(set(LABELS)), 54)
        self.assertNotEqual(LABELS[CLASSES.index('cs.IT')], LABELS[CLASSES.index('math.IT')])
        self.assertTrue(all(text.startswith(prefix.removesuffix('...'))
                            for text, prefix in zip(df.resumo_texto, df.resumo_csv_abreviado)))

    def test_evaluation_and_missing_documents(self):
        df = full_dataset().head(2)
        rows = []
        for document in df.itertuples():
            truth = [c.strip() for c in document.classes_reais_aapd.split(',')]
            for rank, code in enumerate(truth, 1):
                rows.append({'doc_id': document.doc_id, 'rank': rank,
                             'topico': LABELS[CLASSES.index(code)]})
        outputs = pd.DataFrame(rows)
        metrics, per_class = evaluate_results(df, outputs, top_k=3)
        self.assertEqual(metrics.iloc[0]['acerto_top1'], 1)
        self.assertEqual(metrics.iloc[0]['F1 micro'], 1)
        self.assertEqual(len(per_class), 54)
        with self.assertRaises(ValueError):
            evaluate_results(df, outputs[outputs.doc_id == df.iloc[0].doc_id])


if __name__ == '__main__':
    unittest.main()
