import unittest
import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from llda_scores_exatos import exact_scores, aggregate
from verbalizadores_54 import EXPRESSIONS, VERBALIZERS


class ExactScoreTests(unittest.TestCase):
    def test_cached_scores_equal_full_teacher_forcing(self):
        prompt = 'Choose: '
        labels = ['Alpha Topic', 'Beta Theory', 'Gamma']
        alphabet = sorted(set(prompt + ''.join(labels)))
        encode = {c: i for i, c in enumerate(alphabet)}
        decode = {i: c for c, i in encode.items()}
        eos = len(alphabet)
        class Tokenizer:
            eos_token_id = eos
            pad_token_id = eos
            def encode(self, text, add_special_tokens=False):
                return [encode[c] for c in text]
            def decode(self, ids, skip_special_tokens=True):
                return ''.join(decode[int(i)] for i in ids if int(i) != eos)
        tokenizer = Tokenizer()
        torch.manual_seed(42)
        torch.set_num_threads(1)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=eos+1, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, max_position_embeddings=128,
            eos_token_id=eos, pad_token_id=eos)).eval()
        scores, lengths = exact_scores(model, tokenizer, prompt, labels, candidate_batch=2)
        expected = []
        for label in labels:
            prefix = tokenizer.encode(prompt)
            target = tokenizer.encode(label) + [eos]
            full = torch.tensor([prefix + target])
            with torch.inference_mode():
                logits = model(full).logits[0, len(prefix)-1:-1].log_softmax(-1)
                expected.append(float(logits.gather(1, torch.tensor(target)[:,None]).sum()))
        np.testing.assert_allclose(scores, expected, atol=1e-5)
        np.testing.assert_array_equal(lengths, [len(label)+1 for label in labels])
        other, _ = exact_scores(model, tokenizer, prompt, list(reversed(labels)), candidate_batch=1)
        np.testing.assert_allclose(scores, other[::-1], atol=1e-5)

    def test_equal_weights_and_bias(self):
        zeros = np.zeros((2,162))
        np.testing.assert_allclose(aggregate(zeros, np.zeros(162), multiple=True), 0)
        prior = np.full(162, -3.0)
        np.testing.assert_allclose(aggregate(zeros, prior, corrected=True, multiple=True, strength=0.5), 1.5)
        self.assertEqual(len(set(EXPRESSIONS)),162)
        self.assertTrue(all(len(row)==3 for row in VERBALIZERS))


if __name__ == '__main__':
    unittest.main()
