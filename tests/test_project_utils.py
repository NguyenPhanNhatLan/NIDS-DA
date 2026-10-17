import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from project_utils import domains, sha256
from training import data_revision


class ProjectUtilsTests(unittest.TestCase):
    def test_hash_matches_standard_sha256_for_bytes_and_path(self):
        payload = b'abc\x00\xff' * 250000
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'artifact.bin'
            path.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            self.assertEqual(sha256(path), expected)
            self.assertEqual(sha256(str(path)), expected)
            self.assertEqual(data_revision.file_sha256(path), expected)

    def test_direction_mapping_rejects_unknown_direction(self):
        self.assertEqual(domains('unsw_to_cicids'), ('unsw', 'cicids'))
        self.assertEqual(domains('cicids_to_unsw'), ('cicids', 'unsw'))
        with self.assertRaises(ValueError):
            domains('unknown')

    def test_fresh_checkout_resolves_defaults_but_cannot_skip_verification(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(data_revision, 'ROOT', Path(folder)), \
             patch.dict(os.environ, {}, clear=True):
            self.assertEqual(data_revision.revision_path('result_root', None), Path(folder)/'results/canonical')
            with self.assertRaisesRegex(FileNotFoundError, 'prepare'):
                data_revision.verify_revision()

    def test_explicit_missing_manifest_is_not_treated_as_fresh_checkout(self):
        with patch.dict(os.environ, {'KLTN_DATA_MANIFEST':'/nonexistent/explicit-manifest.json'}):
            with self.assertRaises(FileNotFoundError):
                data_revision.revision_path('result_root', None)

    def test_empty_manifest_keeps_synthetic_fixture_behavior(self):
        with patch.dict(os.environ, {'KLTN_DATA_MANIFEST':''}):
            self.assertEqual(data_revision.revision_path('result_root', '/tmp/fixture'), Path('/tmp/fixture'))
            self.assertEqual(data_revision.verify_revision(), {})


if __name__ == '__main__':
    unittest.main()
