import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.protocol_revision import load_evaluation_revision


ROOT = Path(__file__).resolve().parents[1]


class EvaluationRevisionTests(unittest.TestCase):
    def setUp(self):
        self.protocol_path = ROOT / "configs/hda_thesis_protocol.json"
        self.revision_path = ROOT / "configs/hda_evaluation_v2.json"

    def test_preserves_original_checkpoint_protocol_hash(self):
        protocol, digest, metadata = load_evaluation_revision(
            self.protocol_path, self.revision_path
        )
        self.assertEqual(digest, hashlib.sha256(self.protocol_path.read_bytes()).hexdigest())
        self.assertEqual(protocol, json.loads(self.protocol_path.read_bytes()))
        self.assertEqual(metadata["evaluation_id"], "hda_eval_v2_purity")

    def test_rejects_changed_training_code(self):
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.py"
            changed.write_text("changed training code")
            def resolve(value):
                if value == "src/training/hda_v4.py":
                    return changed
                return ROOT / value
            with patch("evaluation.protocol_revision.resolve_path", side_effect=resolve):
                with self.assertRaisesRegex(ValueError, "src/training/hda_v4.py"):
                    load_evaluation_revision(self.protocol_path, self.revision_path)

    def test_rejects_unapproved_revision_changes(self):
        original = json.loads(self.revision_path.read_bytes())
        for change in ("protocol", "training_override", "evaluation_code"):
            revision = json.loads(json.dumps(original))
            if change == "protocol":
                revision["training_protocol_sha256"] = "wrong"
            elif change == "training_override":
                revision["code_sha256"]["src/training/hda_v4.py"] = "wrong"
            else:
                revision["code_sha256"]["src/evaluation/hda.py"] = "wrong"
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "revision.json"
                path.write_text(json.dumps(revision))
                with self.assertRaises(ValueError):
                    load_evaluation_revision(self.protocol_path, path)


if __name__ == "__main__":
    unittest.main()
