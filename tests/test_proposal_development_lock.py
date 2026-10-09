import json
import tempfile
import unittest
from pathlib import Path

from evaluation.proposal_development_lock import require_development_lock, ROOT
from training.data_revision import file_sha256


class DevelopmentLockTests(unittest.TestCase):
    def test_final_gate_requires_complete_matrix_and_unchanged_artifacts(self):
        suite = json.loads((ROOT / 'configs/proposal_suite_v2.json').read_text())
        revision = {'data_revision': 'fixture', 'data_manifest_sha256': 'fixture-hash'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'development_lock.json'
            artifact = Path(directory) / 'checkpoint.pt'
            artifact.write_bytes(b'frozen checkpoint fixture')
            with self.assertRaises(FileNotFoundError):
                require_development_lock(revision, path)
            matrix = [{'direction': d, 'seed': s, 'method': m} for d in suite['directions']
                      for s in suite['seeds'] for m in suite['methods']]
            lock = {**revision, 'status': 'development_locked', 'matrix': matrix,
                    'expected_run_count': len(matrix),
                    'artifacts': [{'path': str(artifact), 'sha256': file_sha256(artifact)}]}
            def write():
                path.write_text(json.dumps(lock))
                path.with_suffix('.sha256').write_text(file_sha256(path) + '  development_lock.json\n')
            write()
            require_development_lock(revision, path)
            lock['matrix'] = matrix[:-1]
            write()
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                require_development_lock(revision, path)
            lock['matrix'] = matrix
            write()
            artifact.write_bytes(b'changed checkpoint fixture')
            with self.assertRaisesRegex(ValueError, 'changed'):
                require_development_lock(revision, path)


if __name__ == '__main__':
    unittest.main()
