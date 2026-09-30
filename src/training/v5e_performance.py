"""Immutable dependency-keyed pools and explicit synchronized profiling."""
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from evaluation.hda_v5b_calibration import payload_hash
from training.adaptation import estimate_bandwidth_squared


def stream_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cached_pools(directory, dependencies, builder):
    """Existing entries are read-only. Partial/corrupt entries fail closed."""
    key = payload_hash(dependencies)
    entry = Path(directory) / key
    payload, manifest = entry / 'pools.pt', entry / 'manifest.json'
    if entry.exists():
        if not manifest.exists() or not payload.exists():
            raise ValueError(f'Incomplete pool cache: {entry}; use a new cache_dir')
        metadata = json.loads(manifest.read_text())
        if metadata['dependencies'] != dependencies or stream_hash(payload) != metadata['sha256']:
            raise ValueError(f'Pool cache dependency/content mismatch: {entry}')
        print(f'Pool cache HIT: {entry}', flush=True)
        return torch.load(payload, map_location='cpu', weights_only=True)
    print(f'Pool cache MISS: {entry}', flush=True)
    # Eval-mode builders consume CPU RNG through data loaders. A cache hit must
    # have the same RNG effect as a miss, so preserve CPU RNG on both paths.
    with torch.random.fork_rng(devices=[]):
        pools = builder()
    entry.mkdir(parents=True, exist_ok=False)
    with payload.open('xb') as stream:
        torch.save(pools, stream)
    with manifest.open('x') as stream:
        json.dump({'dependencies': dependencies, 'sha256': stream_hash(payload)}, stream, indent=2)
    return pools


def synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


class StepProfiler:
    """Synchronization is deliberately enabled ONLY for profile runs."""
    def __init__(self, device, enabled=False, warmup=10):
        self.device, self.enabled, self.warmup = device, enabled, warmup
        self.step = 0
        self.records = {}

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        synchronize(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            synchronize(self.device)
            elapsed = time.perf_counter() - start
            if self.step >= self.warmup:
                self.records.setdefault(name, []).append(elapsed)

    def iterate(self, loader, name):
        iterator = iter(loader)
        while True:
            with self.measure(name):
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
            yield batch

    def summary(self):
        return {name: {'calls': len(values), 'total_seconds': sum(values),
                       'mean_ms': 1000 * sum(values) / len(values)}
                for name, values in self.records.items() if values}


def legacy_mk_mmd_loss(s, t, scales, bandwidth_squared=None):
    """Original four-cdist algorithm retained ONLY for profiling/reference tests."""
    n = min(len(s), len(t))
    s, t = s[:n], t[:n]
    base = (estimate_bandwidth_squared(s, t) if bandwidth_squared is None else
            torch.as_tensor(bandwidth_squared, dtype=s.dtype, device=s.device).detach())
    distances = (torch.cdist(s, s).square(), torch.cdist(t, t).square(), torch.cdist(s, t).square())
    terms = []
    for scale in scales:
        ss, tt, st = [torch.exp(-d / (2 * base * scale)).mean() for d in distances]
        terms.append(ss + tt - 2 * st)
    return torch.stack(terms).mean(), base.sqrt()
