"""One frozen canonical revision shared by training and evaluation entry points."""
import json
import os
from pathlib import Path
from project_utils import sha256 as file_sha256

ROOT = Path(__file__).resolve().parents[2]


def revision_manifest():
    path = os.environ.get('KLTN_DATA_MANIFEST', str(ROOT / 'data/revisions/canonical/manifest.json'))
    if not path:
        # Explicitly empty is reserved for isolated synthetic test fixtures.
        return None
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Data manifest not found: {path}. Run the canonical prepare stage first."
        )
    manifest = json.loads(Path(path).read_text())
    if manifest.get('status') != 'frozen' or not manifest.get('quality_gate_passed'):
        raise ValueError('Selected data revision is not frozen/audited')
    return manifest


def revision_path(kind, fallback):
    # Resolve CLI defaults on a fresh checkout; verification remains mandatory at runtime.
    default_manifest = ROOT / "data/revisions/canonical/manifest.json"
    if "KLTN_DATA_MANIFEST" not in os.environ and not default_manifest.exists():
        roots = {
            "common_root": ROOT / "data/revisions/canonical/common",
            "feature_root": ROOT / "data/revisions/canonical/prepared",
            "model_root": ROOT / "models/canonical",
            "result_root": ROOT / "results/canonical",
        }
        return roots[kind]
    manifest = revision_manifest()
    return Path(manifest[kind]) if manifest else Path(fallback)


def verify_revision():
    manifest = revision_manifest()
    if not manifest:
        return {}
    manifest_path = Path(os.environ.get('KLTN_DATA_MANIFEST') or ROOT / 'data/revisions/canonical/manifest.json')
    checksum = manifest_path.with_name('manifest.sha256')
    if not checksum.is_file() or checksum.read_text().split()[0] != file_sha256(manifest_path):
        raise ValueError('Frozen manifest checksum is missing or changed')
    for artifact in manifest['frozen_files']:
        if file_sha256(artifact['path']) != artifact['sha256']:
            raise ValueError(f"Frozen data/preprocessing changed: {artifact['path']}")
    # Reject added/removed Parquets as well as byte mutations.
    expected = {a['path'] for a in manifest['frozen_files'] if a['path'].endswith('.parquet')}
    actual = {str(p.resolve()) for kind in ('common_root', 'feature_root')
              for p in Path(manifest[kind]).rglob('*.parquet')}
    if expected != actual:
        raise ValueError('Frozen revision Parquet inventory changed')
    return {'data_revision': manifest['data_revision'],
            'data_manifest_sha256': file_sha256(manifest_path)}


def require_development_open():
    manifest = revision_manifest()
    if manifest and (Path(manifest['result_root']) / 'development_lock.json').exists():
        raise RuntimeError('Development is locked; do not retrain or overwrite protocol artifacts after locking')


if __name__ == '__main__':
    print(json.dumps(verify_revision(), indent=2))
