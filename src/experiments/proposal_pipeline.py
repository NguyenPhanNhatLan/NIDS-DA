import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVISION = "canonical"
MANIFEST = ROOT / "data/revisions" / REVISION / "manifest.json"
WORK_ROOT = ROOT / "data/bigdata/thesis_20261005"
SUITE = json.loads((ROOT / "configs/proposal_suite_v2.json").read_text())
DIRECTIONS = tuple(SUITE["directions"])
SEEDS = tuple(SUITE["seeds"])
CONFIGS = ("proposal_mmd_v2", "proposal_mkmmd_v2", "proposal_class_aware_v2")


def prepare():
    from evaluation.proposal_raw_identity_audit import run as raw_audit
    from evaluation.proposal_split_replay import run as split_audit
    from evaluation.proposal_data_audit import run as common_audit
    from features.freeze_proposal_revision import freeze

    if (
        MANIFEST.parent.exists()
        or (ROOT / "models" / REVISION).exists()
        or (ROOT / "results" / REVISION).exists()
    ):
        raise FileExistsError(
            "Canonical output already exists; use verify and later stages. Never mix an incomplete build with an existing revision."
        )
    reports = ROOT / "results/data_audit" / REVISION
    if reports.exists():
        raise FileExistsError(
            f"Audit reports already exist: {reports}; preserve them and clean an incomplete build explicitly before retrying."
        )
    reports.mkdir(parents=True)
    raw = reports / "raw_identity.json"
    replay = reports / "split_replay.json"
    audit = reports / "data_quality.json"
    split_audit(replay, WORK_ROOT)
    raw_audit(raw, WORK_ROOT)
    common_audit(audit, WORK_ROOT, replay, raw)
    policy = json.loads((ROOT / 'configs/proposal_data_policy.json').read_text())
    freeze(audit, REVISION, **policy)


def select_revision():
    if not MANIFEST.is_file():
        raise FileNotFoundError(
            "Run --stage prepare first; canonical manifest is missing"
        )
    os.environ["KLTN_DATA_MANIFEST"] = str(MANIFEST)
    from training.data_revision import verify_revision

    revision = verify_revision()
    if revision["data_revision"] != REVISION:
        raise ValueError("The pipeline only accepts the canonical revision")
    return revision


def diagnostics():
    from evaluation.proposal_domain_shift import run_intrinsic, run
    from training.data_revision import revision_path

    run_intrinsic(revision_path("common_root", None), seed=42)
    for direction in DIRECTIONS:
        run(direction, seed=42)


def train():
    from experiments.proposal_source_only import run as source_only
    from training.proposal_mmd import run as adapted

    for direction in DIRECTIONS:
        for seed in SEEDS:
            source_only(direction, seed)
            for config in CONFIGS:
                adapted(direction, seed, ROOT / "configs" / f"{config}.json")


def aggregate():
    from evaluation.proposal_aggregate import aggregate as collect
    from training.data_revision import revision_path, require_development_open

    require_development_open()

    result = collect(DIRECTIONS, SEEDS)
    output = revision_path("result_root", None) / "aggregate/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved: {output}")


def evaluation():
    from evaluation.proposal_negative_transfer import run

    for direction in DIRECTIONS:
        for seed in SEEDS:
            for config in CONFIGS:
                run(direction, seed, ROOT / "configs" / f"{config}.json")


def final_test():
    from evaluation.proposal_development_lock import require_development_lock
    require_development_lock(select_revision())
    from evaluation.proposal_final_test import run

    for direction in DIRECTIONS:
        for seed in SEEDS:
            for method in SUITE["methods"]:
                run(direction, method, seed)


def lock_development():
    from evaluation.proposal_development_lock import freeze_development
    freeze_development()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "prepare",
            "verify",
            "diagnostics",
            "train",
            "evaluation",
            "aggregate",
            "lock-development",
            "final-test",
        ),
    )
    stage = parser.parse_args().stage
    os.environ["KLTN_DATA_MANIFEST"] = str(MANIFEST)
    if stage == "prepare":
        prepare()
        return
    revision = select_revision()
    if stage == "verify":
        print(json.dumps(revision, indent=2))
    else:
        {
            "diagnostics": diagnostics,
            "train": train,
            "evaluation": evaluation,
            "aggregate": aggregate,
            "lock-development": lock_development,
            "final-test": final_test,
        }[stage]()


if __name__ == "__main__":
    main()
