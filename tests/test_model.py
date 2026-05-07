import unittest

import torch

from finllm.config import ModelConfig
from finllm.model import FinanceGPT


class FinanceGPTTest(unittest.TestCase):
    def test_forward_loss_and_generation_shape_modern(self):
        config = ModelConfig(
            architecture="modern",
            vocab_size=64,
            block_size=8,
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            bias=False,
            num_kv_heads=1,
            mlp_type="swiglu",
            norm_type="rmsnorm",
        )
        model = FinanceGPT(config)
        idx = torch.randint(0, config.vocab_size, (4, config.block_size))
        logits, loss = model(idx, idx)

        self.assertEqual(logits.shape, (4, config.block_size, config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))

        generated = model.generate(idx[:, :2], max_new_tokens=3, top_k=10)
        self.assertEqual(generated.shape, (4, 5))

    def test_forward_legacy_architecture(self):
        config = ModelConfig(
            architecture="legacy",
            vocab_size=32,
            block_size=6,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            bias=False,
        )
        model = FinanceGPT(config)
        idx = torch.randint(0, config.vocab_size, (2, config.block_size))
        logits, loss = model(idx, idx)
        self.assertEqual(logits.shape, (2, config.block_size, config.vocab_size))
        self.assertTrue(torch.isfinite(loss))

    def test_rejects_too_long_context(self):
        config = ModelConfig(vocab_size=32, block_size=4, n_layer=1, n_head=1, n_embd=8)
        model = FinanceGPT(config)
        with self.assertRaises(ValueError):
            model(torch.zeros((1, 5), dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
