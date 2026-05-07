import unittest

from webapp.server import clean_generated_text, project_snapshot, safe_static_path


class WebAppTest(unittest.TestCase):
    def test_static_paths_stay_inside_static_root(self):
        self.assertIsNotNone(safe_static_path("/"))
        self.assertIsNotNone(safe_static_path("/app.js"))
        self.assertIsNone(safe_static_path("/../finance_tokenizer.model"))

    def test_project_snapshot_has_demo_fields(self):
        snapshot = project_snapshot()
        self.assertEqual(snapshot["tokenizer"]["vocab_size"], 32000)
        self.assertFalse(snapshot["cost"]["paid_apis"])
        self.assertFalse(snapshot["cost"]["external_model_apis"])
        self.assertIn("model", snapshot)
        self.assertIn("commands", snapshot)
        self.assertIn("checkpoints", snapshot)
        self.assertIn("sft_train", snapshot["commands"])

    def test_generated_text_cleaner_filters_markup(self):
        cleaned, was_cleaned = clean_generated_text(
            'Apple revenue improved</div><td style="font-size:10pt;">noise</td>'
        )
        self.assertTrue(was_cleaned)
        self.assertIn("Apple revenue improved", cleaned)
        self.assertNotIn("style=", cleaned)


if __name__ == "__main__":
    unittest.main()
