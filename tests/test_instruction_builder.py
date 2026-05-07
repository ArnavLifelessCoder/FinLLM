import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_instruction_corpus.py"
SPEC = importlib.util.spec_from_file_location("build_instruction_corpus", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class InstructionBuilderTest(unittest.TestCase):
    def test_records_for_chunk_emits_finance_prompts(self):
        text = (
            "Apple revenue increased 12% year over year while operating cash flow improved "
            "to $5.2 billion and gross margin expanded."
        )
        records = MODULE.records_for_chunk(text, MODULE.random.Random(1337))
        self.assertGreater(len(records), 0)
        self.assertTrue(any("revenue" in record["instruction"].lower() for record in records))
        self.assertTrue(any("cash flow" in record["instruction"].lower() for record in records))

    def test_clean_chunk_filters_markupy_text(self):
        bad = 'b"<td style=\\"font-size:10pt\\">noise</td>"'
        self.assertIsNone(MODULE.clean_chunk(bad))


if __name__ == "__main__":
    unittest.main()
