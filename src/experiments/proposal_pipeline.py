"""Run the canonical data, development and final-test stages."""

from project_utils.result_files import result_file
import argparse
import json
import os
from experiments.proposal_protocol import (
    ROOT,
    REVISION,
    MANIFEST,
    DIRECTIONS,
    SEEDS,
    METHOD_CONFIGS,
    experiment_matrix,
    validate_method_configs,
)

WORK_ROOT = ROOT / "data/bigdata/thesis_20261005"


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
    policy = json.loads((ROOT / "configs/proposal_data_policy.json").read_text())
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


def verify_input_row_counts():
    import pyarrow.parquet as pq
    from training.data_revision import revision_path

    common_root = revision_path("common_root", None)
    prepared_root = revision_path("feature_root", None)

    def rows(directory):
        files = sorted(directory.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"Missing Parquet input: {directory}")
        return sum(pq.ParquetFile(path).metadata.num_rows for path in files)

    for domain in ("unsw", "cicids"):
        for split in ("train", "val", "test"):
            expected = rows(common_root / f"{domain}_{split}")
            if expected == 0:
                raise ValueError(f"Empty canonical split: {domain}_{split}")
            for direction in DIRECTIONS:
                actual = rows(prepared_root / direction / f"{domain}_{split}")
                if actual != expected:
                    raise ValueError(
                        f"Inconsistent inputs: {direction}/{domain}_{split} "
                        f"has {actual:,} rows; common split has {expected:,}"
                    )
    print("Canonical/common and both directional prepared row counts match.")


def diagnostics():
    from evaluation.proposal_domain_shift import run_intrinsic, run
    from training.data_revision import revision_path

    verify_input_row_counts()
    result_root = revision_path("result_root", None)
    common_root = revision_path("common_root", None)

    for seed in SEEDS:
        intrinsic_dir = result_root / "domain_shift" / "intrinsic" / f"seed{seed}"
        if result_file(intrinsic_dir / "result.json").is_file() and (
            result_file(intrinsic_dir / "samples.npz")
        ).is_file():
            print(f"Reuse intrinsic diagnostics: seed{seed}", flush=True)
        else:
            run_intrinsic(
                common_root,
                seed=seed,
                overwrite=intrinsic_dir.exists(),
            )

        for direction in DIRECTIONS:
            output = result_file(result_root / "domain_shift" / f"{direction}_seed{seed}.json")
            if output.is_file():
                print(
                    f"Reuse directional diagnostics: {direction}/seed{seed}", flush=True
                )
            else:
                run(direction, seed=seed)


def train():
    from experiments.proposal_source_only import run as source_only
    from training.proposal_mmd import run as adapted

    verify_input_row_counts()
    for direction in DIRECTIONS:
        for seed in SEEDS:
            source_only(direction, seed)
            for config_path in METHOD_CONFIGS.values():
                adapted(direction, seed, config_path)


def resume_train():
    """Resume missing runs after replay-validating existing checkpoint/result pairs."""
    from evaluation.proposal_final_test import paths, validate_development
    from experiments.proposal_source_only import run as source_only
    from training.proposal_mmd import run as adapted
    from training.data_revision import require_development_open

    require_development_open()
    verify_input_row_counts()
    validate_method_configs()
    methods = [("source_only", None), *METHOD_CONFIGS.items()]

    for direction in DIRECTIONS:
        for seed in SEEDS:
            for method, config_path in methods:
                checkpoint, result = paths(method, direction, seed)
                if checkpoint.exists() != result.exists():
                    raise RuntimeError(
                        f"Incomplete {method} artifacts for {direction}/seed{seed}: "
                        f"{checkpoint}, {result}. Inspect the failed run; never auto-overwrite."
                    )
                if checkpoint.exists():
                    validate_development(direction, method, seed)
                    print(f"Validated and reused: {method} {direction} seed{seed}")
                elif method == "source_only":
                    source_only(direction, seed)
                else:
                    adapted(direction, seed, config_path)


def aggregate():
    from evaluation.proposal_aggregate import aggregate as collect
    from training.data_revision import revision_path, require_development_open

    require_development_open()

    result = collect(DIRECTIONS, SEEDS)
    output = revision_path("result_root", None) / "aggregate/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved: {output}")


def evaluation():
    from evaluation.proposal_negative_transfer import run

    for direction in DIRECTIONS:
        for seed in SEEDS:
            for config_path in METHOD_CONFIGS.values():
                run(direction, seed, config_path)


def final_test():
    from evaluation.proposal_development_lock import require_development_lock

    require_development_lock(select_revision())
    from evaluation.proposal_final_test import run

    for direction, seed, method in experiment_matrix():
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
            "resume-train",
            "evaluation",
            "aggregate",
            "lock-development",
            "final-test",
        ),
    )
    stage = parser.parse_args().stage
    validate_method_configs()
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
            "resume-train": resume_train,
            "evaluation": evaluation,
            "aggregate": aggregate,
            "lock-development": lock_development,
            "final-test": final_test,
        }[stage]()


if __name__ == "__main__":
    main()
