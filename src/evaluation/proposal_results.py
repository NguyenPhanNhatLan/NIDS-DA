import json
import math
import os
from pathlib import Path

from project_utils.result_files import result_file

from evaluation.proposal_development_lock import require_development_lock
from experiments.proposal_protocol import DIRECTIONS, METHOD_CONFIGS, ROOT, SEEDS
from project_utils import sha256
from training.data_revision import revision_manifest

RUN_COLUMNS = (
    "direction",
    "method",
    "seed",
    "mmd2_before",
    "mmd2_after",
    "delta_mmd2",
    "mmd_reduction_percent",
    "ap_before",
    "ap_after",
    "delta_ap",
    "selected_stage",
    "best_epoch",
)


def diagnostic_files(folder):
    paths = []
    for direction in DIRECTIONS:
        for method, config in METHOD_CONFIGS.items():
            for seed in SEEDS:
                name = f"seed{seed}.json" if method == "marginal_mmd" else f"seed{seed}_{config.stem}.json"
                paths.append(result_file(Path(folder) / direction / name))
    return paths


def _check_equal(actual, expected, context):
    if actual != expected:
        raise ValueError(f"{context}: {actual!r} != {expected!r}")


def load_alignment_runs(diagnostics_root):
    manifest = revision_manifest()
    if not manifest:
        raise ValueError("This export requires a frozen data revision")
    manifest_path = Path(
        os.environ.get("KLTN_DATA_MANIFEST")
        or (ROOT / "data/revisions/canonical/manifest.json")
    )
    revision = {
        "data_revision": manifest["data_revision"],
        "data_manifest_sha256": sha256(manifest_path),
    }
    lock = require_development_lock(revision)
    locked_hashes = {
        Path(row["path"]).resolve(): row["sha256"] for row in lock["artifacts"]
    }
    config_methods = {path.stem: method for method, path in METHOD_CONFIGS.items()}
    expected = {
        (direction, method, seed)
        for direction in DIRECTIONS
        for method in METHOD_CONFIGS
        for seed in SEEDS
    }
    runs = {}
    input_paths = set()
    for path in diagnostic_files(diagnostics_root):
        report = json.loads(path.read_text())
        config_name = Path(report["config"]).stem
        if config_name not in config_methods:
            raise ValueError(f"Unexpected diagnostic config: {path}: {config_name}")
        method = config_methods[config_name]
        identity = (report["direction"], method, report["seed"])
        if identity not in expected or identity in runs:
            raise ValueError(f"Unexpected or duplicate run: {identity}: {path}")
        for key, value in revision.items():
            _check_equal(report[key], value, f"{path}: {key}")
        config_path = METHOD_CONFIGS[method].resolve()
        _check_equal(
            report["config_sha256"], locked_hashes[config_path], f"{path}: config hash"
        )

        artifacts = report["artifacts"]
        artifact_paths = {}
        for role in (
            "source_result",
            "adapted_result",
            "source_checkpoint",
            "adapted_checkpoint",
        ):
            artifact_path = Path(artifacts[role]).resolve()
            if artifact_path not in locked_hashes:
                raise ValueError(
                    f"Diagnostic references an unlocked artifact: {artifact_path}"
                )
            if role.endswith("checkpoint"):
                _check_equal(
                    artifacts[role + "_sha256"],
                    locked_hashes[artifact_path],
                    f"{path}: {role}",
                )
            artifact_paths[role] = artifact_path
        source = json.loads(artifact_paths["source_result"].read_text())
        adapted = json.loads(artifact_paths["adapted_result"].read_text())
        for result in (source, adapted):
            for key in ("direction", "seed", *revision):
                _check_equal(result[key], report[key], f"{path}: {key}")
        _check_equal(adapted["method"], method, f"{path}: method")
        _check_equal(
            adapted["config_sha256"], report["config_sha256"], f"{path}: adapted config"
        )
        for result, role in (
            (source, "source_checkpoint"),
            (adapted, "adapted_checkpoint"),
        ):
            _check_equal(
                Path(result["checkpoint"]).resolve(),
                artifact_paths[role],
                f"{path}: checkpoint",
            )

        epoch = adapted["best_epoch"]
        if type(epoch) is not int or epoch < 0:
            raise ValueError(f"Invalid adaptation epoch: {path}: {epoch}")
        stage = adapted["selected_stage"]
        _check_equal(
            stage, "source_pretrained" if epoch == 0 else "adaptation", f"{path}: stage"
        )
        before = float(report["marginal_alignment"]["before"]["mmd2_mean"])
        after = float(report["marginal_alignment"]["after"]["mmd2_mean"])
        ap_before = float(source["target_development"]["pr_auc"])
        ap_after = float(adapted["cross_domain"]["pr_auc"])
        if (
            not all(math.isfinite(x) for x in (before, after, ap_before, ap_after))
            or before <= 0
        ):
            raise ValueError(f"Invalid MMD/AP values: {path}")
        transfer = report["negative_transfer"]
        for actual, saved in (
            (transfer["source_only_target"]["pr_auc"], ap_before),
            (transfer["adapted_target"]["pr_auc"], ap_after),
            (transfer["delta_target_pr_auc"], ap_after - ap_before),
        ):
            if not math.isclose(actual, saved, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Diagnostic AP differs from locked results: {path}")
        runs[identity] = dict(
            zip(
                RUN_COLUMNS,
                (
                    *identity,
                    before,
                    after,
                    after - before,
                    100 * (before - after) / before,
                    ap_before,
                    ap_after,
                    ap_after - ap_before,
                    stage,
                    epoch,
                ),
            )
        )
        input_paths.update((path.resolve(), *artifact_paths.values()))
    if set(runs) != expected:
        raise ValueError(f"Missing diagnostic runs: {sorted(expected - set(runs))}")
    ordered = [
        runs[direction, method, seed]
        for direction in DIRECTIONS
        for method in METHOD_CONFIGS
        for seed in SEEDS
    ]
    return ordered, revision, input_paths


def load_score_reports(diagnostics_root, seed):
    runs, revision, _ = load_alignment_runs(diagnostics_root)
    selections = {(row["direction"], row["method"]): row for row in runs if row["seed"] == seed}
    reports, sources = {}, []
    methods_by_config = {path.stem: method for method, path in METHOD_CONFIGS.items()}
    for path in diagnostic_files(diagnostics_root):
        if f"seed{seed}." not in path.name and f"seed{seed}_" not in path.name:
            continue
        report = json.loads(path.read_text())
        key = (report["direction"], methods_by_config[Path(report["config"]).stem])
        scores = report.get("target_validation_score_analysis")
        if not scores:
            raise ValueError(f"Score diagnostics missing; rerun diagnostics first: {path}")
        reports[key] = report
        sources.append({"path": str(path.resolve()), "sha256": sha256(path)})
    if set(reports) != set(selections) or not reports:
        raise ValueError(f"Incomplete diagnostic matrix for seed {seed}")
    ap_checks = []
    for direction in DIRECTIONS:
        baseline = reports[direction, "marginal_mmd"]["target_validation_score_analysis"]["before"]
        for method in METHOD_CONFIGS:
            before = reports[direction, method]["target_validation_score_analysis"]["before"]
            if before != baseline:
                raise ValueError(f"Source-only diagnostics differ between methods: {direction}")
            report = reports[direction, method]
            adapted = json.loads(Path(report["artifacts"]["adapted_result"]).read_text())
            threshold = report["target_validation_score_analysis"]["after"]["source_threshold_metrics"]["threshold"]
            if (
                not math.isfinite(threshold)
                or not 0 <= threshold <= 1
                or not math.isclose(threshold, adapted["threshold"], rel_tol=0, abs_tol=1e-12)
            ):
                raise ValueError(f"After decision threshold differs from locked result: {direction}/{method}")
            for phase, expected_ap in (("before", "ap_before"), ("after", "ap_after")):
                observed = reports[direction, method]["target_validation_score_analysis"][phase]
                if not math.isclose(observed["average_precision"], selections[direction, method][expected_ap], rel_tol=0, abs_tol=1e-3):
                    raise ValueError(f"PR AP differs from locked result: {direction}/{method}/{phase}")
                ap_checks.append({
                    "direction": direction, "method": method, "phase": phase,
                    "diagnostic_ap": observed["average_precision"],
                    "locked_training_ap": selections[direction, method][expected_ap],
                    "difference": observed["average_precision"] - selections[direction, method][expected_ap],
                })
                for label in ("benign", "attack"):
                    counts = observed["score_distributions"][label]["count"]
                    if counts != baseline["score_distributions"][label]["count"]:
                        raise ValueError("Before/after diagnostics use different class counts")
    return reports, selections, revision, sources, ap_checks
