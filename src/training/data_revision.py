"""Select and verify the frozen revision used by training and evaluation."""

import json
import os
from pathlib import Path

from project_utils import sha256 as file_sha256

ROOT = Path(__file__).resolve().parents[2]


def _selected_manifest_path():
    setting = os.environ.get(
        "KLTN_DATA_MANIFEST", str(ROOT / "data/revisions/canonical/manifest.json")
    )
    return Path(setting) if setting else None


def revision_manifest():
    path = _selected_manifest_path()
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(
            f"Data manifest not found: {path}. Run the canonical prepare stage first."
        )
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "frozen" or not manifest.get("quality_gate_passed"):
        raise ValueError("Selected data revision is not frozen/audited")
    return manifest


def revision_path(kind, fallback):
    default_manifest = ROOT / "data/revisions/canonical/manifest.json"
    if "KLTN_DATA_MANIFEST" not in os.environ and not default_manifest.exists():
        canonical_roots = {
            "common_root": ROOT / "data/revisions/canonical/common",
            "feature_root": ROOT / "data/revisions/canonical/prepared",
            "model_root": ROOT / "models/canonical",
            "result_root": ROOT / "results/canonical",
        }
        return canonical_roots[kind]
    manifest = revision_manifest()
    return Path(manifest[kind]) if manifest else Path(fallback)


def verify_revision():
    manifest = revision_manifest()
    if not manifest:
        return {}

    manifest_path = _selected_manifest_path()
    checksum_path = manifest_path.with_name("manifest.sha256")
    manifest_sha256 = file_sha256(manifest_path)
    if (
        not checksum_path.is_file()
        or checksum_path.read_text().split()[0] != manifest_sha256
    ):
        raise ValueError("Frozen manifest checksum is missing or changed")
    for artifact in manifest["frozen_files"]:
        if file_sha256(artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"Frozen data/preprocessing changed: {artifact['path']}")

    expected_parquet = {
        artifact["path"]
        for artifact in manifest["frozen_files"]
        if artifact["path"].endswith(".parquet")
    }
    actual_parquet = {
        str(path.resolve())
        for kind in ("common_root", "feature_root")
        for path in Path(manifest[kind]).rglob("*.parquet")
    }
    if expected_parquet != actual_parquet:
        raise ValueError("Frozen revision Parquet inventory changed")
    return {
        "data_revision": manifest["data_revision"],
        "data_manifest_sha256": manifest_sha256,
    }


def require_development_open():
    manifest = revision_manifest()
    if manifest and (Path(manifest["result_root"]) / "development_lock.json").exists():
        raise RuntimeError(
            "Development is locked; do not retrain or overwrite protocol artifacts after locking"
        )


if __name__ == "__main__":
    print(json.dumps(verify_revision(), indent=2))
