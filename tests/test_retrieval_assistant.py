import tempfile
import unittest
from pathlib import Path

from finllm.assistant import GroundedFinanceAssistant
from finllm.retrieval import build_index, search


class RetrievalAssistantTest(unittest.TestCase):
    def test_search_and_grounded_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.txt"
            index = Path(tmp) / "index.sqlite"
            corpus.write_text(
                "Microsoft revenue increased because cloud demand improved. "
                "Operating cash flow also rose during the period.\n"
                "Apple reported lower services growth but stronger margins.\n",
                encoding="utf8",
            )
            metadata = build_index(corpus_path=corpus, index_path=index)
            results = search(index, "Why did Microsoft revenue increase?", limit=3)
            answer = GroundedFinanceAssistant(index).answer(
                "Why did Microsoft revenue increase?", top_k=3
            )

        self.assertGreater(metadata["chunks"], 0)
        self.assertGreater(len(results), 0)
        self.assertIn("cloud demand", answer["answer"])
        self.assertGreater(len(answer["evidence"]), 0)


if __name__ == "__main__":
    unittest.main()

