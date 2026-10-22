"""Contratos da restrição por prefixos e da avaliação com respostas inválidas."""
import unittest
import pandas as pd
from llda_decoding_restrito import LabelTrie, initial_subset, compare_methods, generate_beams


class ConstrainedDecodingTests(unittest.TestCase):
    def test_prefixes_shared_and_terminal(self):
        trie = LabelTrie([[1, 2], [1, 3], [1, 2, 4]], eos_token_id=9)
        self.assertEqual(trie.allowed([]), [1])
        self.assertEqual(set(trie.allowed([1])), {2, 3})
        self.assertEqual(set(trie.allowed([1, 2])), {4, 9})
        self.assertEqual(trie.allowed([1, 3]), [9])
        self.assertEqual(trie.allowed([1, 2, 4]), [9])
        with self.assertRaises(ValueError):
            trie.allowed([1, 4])
        # EOS não existe antes de um nome completo.
        with self.assertRaises(ValueError):
            trie.allowed([1, 9])

    def test_callback_ignores_prompt(self):
        import torch
        callback = LabelTrie([[1, 2]], 9).callback(3)
        self.assertEqual(callback(0, torch.tensor([7, 8, 6, 1])), [2])

    def test_constrained_beams_with_tiny_random_model(self):
        # Integração com generate real; testa formato, não qualidade semântica.
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        labels = ['Alpha Topic', 'Alpha Theory', 'Beta Science', 'Gamma Networks']
        prompt = 'Choose: '
        alphabet = sorted(set(prompt + ''.join(labels)))
        encoding = {char: i for i, char in enumerate(alphabet)}
        decoding = {value: key for key, value in encoding.items()}
        eos = len(alphabet)

        class Tokenizer:
            eos_token_id = eos
            pad_token_id = eos

            def encode(self, text, add_special_tokens=False):
                return [encoding[char] for char in text]

            def decode(self, ids, skip_special_tokens=True):
                return ''.join(decoding[int(i)] for i in ids if int(i) != eos)

            def batch_decode(self, sequences, skip_special_tokens=True):
                return [self.decode(seq, skip_special_tokens) for seq in sequences]

        torch.manual_seed(42)
        torch.set_num_threads(1)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=eos + 1, n_positions=128,
                                         n_embd=16, n_layer=1, n_head=2,
                                         bos_token_id=eos, eos_token_id=eos,
                                         pad_token_id=eos)).eval()
        outputs = generate_beams(model, Tokenizer(), prompt, labels, constrained=True, num_beams=4)
        self.assertTrue(outputs.valido.all())
        self.assertEqual(len(outputs), 4)
        self.assertTrue(set(outputs.topico) <= set(labels))

    def test_catalog_and_evaluation(self):
        df, catalog, mapping = initial_subset()
        self.assertEqual(df.doc_id.tolist(), [51, 178, 209, 228, 285])
        self.assertEqual(len(mapping), 9)
        self.assertEqual(len(catalog), 8)
        self.assertTrue((df.resumo_texto.str.len() > df.resumo_csv_abreviado.str.len()).all())
        self.assertEqual(df.loc[3, 'topicos_reais_legiveis'], ['Information Theory'])
        document = df.head(1)
        truth = document.iloc[0].topicos_reais_legiveis[0]
        outputs = pd.DataFrame([
            {'metodo': 'prompt', 'doc_id': 51, 'rank': 1, 'topico': None, 'valido': False},
            {'metodo': 'prompt', 'doc_id': 51, 'rank': 2, 'topico': truth, 'valido': True},
            {'metodo': 'prompt', 'doc_id': 51, 'rank': 3, 'topico': truth, 'valido': True},
        ])
        metrics = compare_methods(outputs, document, catalog, top_k=2).iloc[0]
        self.assertEqual(metrics.acerto_top1, 0)
        self.assertEqual(metrics.rotulos_unicos_top2, 1)
        self.assertEqual(metrics.recall_top2, 0.5)


if __name__ == '__main__':
    unittest.main()
