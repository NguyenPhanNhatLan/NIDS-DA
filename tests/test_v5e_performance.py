import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

from training.v5e_performance import cached_pools


class PoolCacheTests(unittest.TestCase):
    def test_hit_is_read_only_and_preserves_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            before = torch.get_rng_state().clone()
            builder = Mock(side_effect=lambda: {"x": torch.randn(5, 2)})
            first = cached_pools(directory, {"data": "a", "code": "b"}, builder)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            payload = next(Path(directory).glob('*/pools.pt'))
            stat = payload.stat()
            second = cached_pools(directory, {"data": "a", "code": "b"}, builder)
            self.assertEqual(builder.call_count, 1)
            self.assertEqual(stat.st_mtime_ns, payload.stat().st_mtime_ns)
            self.assertTrue(torch.equal(first['x'], second['x']))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            cached_pools(directory, {"data": "changed", "code": "b"}, builder)
            self.assertEqual(builder.call_count, 2)

    def test_corruption_fails_without_rebuilding(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = Mock(return_value={"x": torch.ones(2)})
            cached_pools(directory, {"data": "a"}, builder)
            payload = next(Path(directory).glob('*/pools.pt'))
            payload.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, "mismatch"):
                cached_pools(directory, {"data": "a"}, builder)
            self.assertEqual(builder.call_count, 1)


if __name__ == '__main__':
    unittest.main()
