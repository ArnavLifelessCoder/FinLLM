import tempfile
import unittest
from pathlib import Path

import numpy as np

from finllm.data import BinaryTokenDataset, token_dtype


class DataPipelineTest(unittest.TestCase):
    def test_dtype_selection(self):
        self.assertEqual(token_dtype(32000), np.dtype(np.uint16))
        self.assertEqual(token_dtype(70000), np.dtype(np.uint32))

    def test_memmap_batch_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.bin"
            np.arange(100, dtype=np.uint16).tofile(path)
            dataset = BinaryTokenDataset(path, block_size=8, dtype="uint16")
            x, y = dataset.get_batch(batch_size=4, device="cpu")
            dataset.close()

        self.assertEqual(x.shape, (4, 8))
        self.assertEqual(y.shape, (4, 8))
        self.assertTrue(((y - x) == 1).all())


if __name__ == "__main__":
    unittest.main()
