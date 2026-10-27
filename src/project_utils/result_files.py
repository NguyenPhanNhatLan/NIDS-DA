from functools import lru_cache
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FOLDERS = {".json": "json", ".png": "images", ".csv": "tables", ".md": "tables", ".npz": "arrays"}


@lru_cache(maxsize=1)
def frozen_results():
    paths = {RESULTS / "canonical/development_lock.json", RESULTS / "canonical/development_lock.sha256"}
    for manifest, key in ((ROOT / "data/revisions/canonical/manifest.json", "frozen_files"),
                          (RESULTS / "canonical/development_lock.json", "artifacts")):
        if manifest.is_file():
            records = json.loads(manifest.read_text())[key]
            for record in records:
                paths.add(Path(record["path"]).absolute())
    return paths


def result_file(path):
    path = Path(path).absolute()
    if path in frozen_results():
        return path
    try:
        relative = path.relative_to(RESULTS)
    except ValueError:
        return path
    folder = FOLDERS.get(path.suffix)
    if folder is None or relative.parts[0] in FOLDERS.values():
        return path
    parts = list(relative.parts)
    if parts[:2] == ["analysis", "section_4_5_canonical"]:
        parts = ["mechanisms", *parts[2:]]
    elif parts[:2] == ["analysis", "latent_knn_balanced_v1"]:
        parts = ["knn", *parts[2:]]
    elif parts[:2] == ["analysis", "figures"]:
        parts = parts[2:]
    return RESULTS / folder / "__".join(parts)


def publish_results(folder):
    folder = Path(folder).absolute()
    if not folder.is_relative_to(RESULTS) or not folder.is_dir():
        return
    moves = []
    for source in sorted(folder.rglob("*")):
        if source.is_file():
            destination = result_file(source)
            if destination != source:
                if destination.exists():
                    raise FileExistsError(f"Output already exists: {destination}")
                moves.append((source, destination))
    for source, destination in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
    for directory in sorted(folder.rglob("*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    if folder.exists() and not any(folder.iterdir()):
        folder.rmdir()
