"""Synthetic, synchronized legacy/optimized MK-MMD forward+backward benchmark."""
import argparse
import json
import time

import torch

from training.mkmmd import MKMMDLoss
from training.v5e_performance import legacy_mk_mmd_loss, synchronize


def benchmark(device, steps=200, warmup=10):
    generator = torch.Generator().manual_seed(42)
    batches = [(torch.randn(n, d, generator=generator).to(device),
                torch.randn(n, d, generator=generator).to(device))
               for n, d in ((256, 256), (64, 168), (64, 168))]
    scales = (.25, .5, 1., 2., 4.)
    module = MKMMDLoss(scales).to(device=device, dtype=torch.float32)
    results = {}
    # Same shapes/data; three terms correspond to Hidden/Normal/Attack.
    for name in ('legacy', 'optimized'):
        forward, backward = 0., 0.
        for step in range(warmup + steps):
            pairs = [(s, t.detach().requires_grad_()) for s, t in batches]
            synchronize(device)
            start = time.perf_counter()
            losses = [(legacy_mk_mmd_loss(s, t, scales) if name == 'legacy' else module(s, t))[0]
                      for s, t in pairs]
            loss = sum(losses)
            synchronize(device)
            split = time.perf_counter()
            loss.backward()
            synchronize(device)
            end = time.perf_counter()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite benchmark loss')
            if step >= warmup:
                forward += split - start
                backward += end - split
        results[name] = {'forward_ms_per_three_terms': forward * 1000 / steps,
                         'backward_ms_per_three_terms': backward * 1000 / steps,
                         'three_term_steps_per_second': steps / (forward + backward)}
    return {'device': str(device), 'torch': str(torch.__version__), 'steps': steps,
            'warmup': warmup, 'results': results,
            'note': 'Synthetic MK-MMD only; not full training throughput. Legacy math excludes old validation checks.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='mps')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error('steps must be positive and warmup nonnegative')
    print(json.dumps(benchmark(torch.device(args.device), args.steps, args.warmup), indent=2))


if __name__ == '__main__':
    main()
