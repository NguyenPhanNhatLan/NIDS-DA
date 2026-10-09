"""Final-test calls must be restricted to the frozen development matrix."""
import unittest
from unittest.mock import patch

from evaluation import proposal_final_test as final_test


class FinalTestMatrixLockTests(unittest.TestCase):
    def test_unlisted_seed_rejected_before_checkpoint_or_test_access(self):
        locked = {
            'matrix': [
                {'direction': 'unsw_to_cicids', 'method': 'source_only', 'seed': 42}
            ]
        }
        with patch.object(final_test, 'verify_revision', return_value={
            'data_revision': 'canonical', 'data_manifest_sha256': 'fixture',
        }), patch(
            'evaluation.proposal_development_lock.require_development_lock',
            return_value=locked,
        ), patch.object(final_test, 'validate_development') as validation:
            with self.assertRaisesRegex(ValueError, 'not in locked development matrix'):
                final_test.run('unsw_to_cicids', 'source_only', 43)
            validation.assert_not_called()


if __name__ == '__main__':
    unittest.main()
