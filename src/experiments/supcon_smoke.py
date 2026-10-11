"""One-epoch, one-batch synthetic Parquet smoke test; no real artifacts touched."""
from copy import deepcopy
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from models.baseline import BaselineMLP
from training.proposal_data import ParquetBatchStream
from training.supcon_adaptation import initialize, source_evaluation, train


def smoke(device):
    from experiments.proposal_supcon_v2 import CONFIGS, load_config
    rng = np.random.default_rng(42)
    features = rng.normal(size=(8, 5)).astype(np.float32)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    torch.manual_seed(42)
    source_state = deepcopy(BaselineMLP(5).state_dict())
    results = {}
    with tempfile.TemporaryDirectory(prefix='supcon-smoke-') as directory:
        root = Path(directory)
        for name in ('source_train', 'source_val', 'target_train'):
            folder = root / name
            folder.mkdir()
            values = features if name != 'target_train' else features + .25
            columns = {'features': pa.array(values.tolist(), type=pa.list_(pa.float32(), 5))}
            if name != 'target_train':
                columns['label'] = pa.array(labels)
            pq.write_table(pa.table(columns), folder / 'part.parquet')
        projection_initial = None
        for experiment in ('E1', 'E2'):
            config = load_config(CONFIGS[experiment], experiment)
            config['training'].update(batch_size=8, epochs=1, patience=1)
            model, projection = initialize(source_state, 5, 42, device)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(), source_state[key])
            if projection_initial is None:
                projection_initial = {key: value.detach().cpu().clone() for key, value in projection.state_dict().items()}
            else:
                for key, value in projection.state_dict().items():
                    torch.testing.assert_close(value.cpu(), projection_initial[key])
            source = ParquetBatchStream(root / 'source_train', 8, True, 42, True, drop_last=True)
            target = (ParquetBatchStream(root / 'target_train', 8, True, 1042, False, drop_last=True)
                      if experiment == 'E2' else None)
            val = ParquetBatchStream(root / 'source_val', 8, False, 42, True)
            initial_ap = source_evaluation(model, val)['pr_auc']
            _, _, history = train(model, projection, source, target, val, [4, 4], config, initial_ap)
            assert len(history) == 2 and len(history[1]['batch_diagnostics']) == 1
            assert history[1]['valid_contrastive_anchors'] == 8
            for group in (model.parameters(), projection.parameters()):
                grads = [p.grad for p in group if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
                assert sum(float(g.abs().sum()) for g in grads) > 0
            results[experiment] = {'status': 'passed', 'device': str(device), 'training_batches': 1,
                                   'target_labels_available': False, 'history': history}
    print(f'Synthetic Parquet smoke PASSED on {device}: E1/E2, one batch each; no real outputs written')
    return results
