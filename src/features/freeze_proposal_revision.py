import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import duckdb
import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from evaluation.proposal_data_audit import DOMAINS, SPLITS, audit_common, inventory
from features.common_features import COMMON_FEATURES
from models.proposal_pipeline import proposal_processor
from training.data_revision import file_sha256
from training.resource_usage import resource_observation

ROOT = Path(__file__).resolve().parents[2]


def relocation_audit(
    initial, final, labels, max_relocated_fraction, max_class_prevalence_change
):
    reports = {}
    for domain in DOMAINS:
        before, after, transitions = {}, {}, {}
        for i, split in enumerate(SPLITS):
            for membership, result in (
                (initial[domain], before),
                (final[domain], after),
            ):
                selected = membership == i
                classes = np.bincount(labels[domain][selected], minlength=2).tolist()
                result[split] = {
                    "rows": int(selected.sum()),
                    "class_counts": classes,
                    "attack_prevalence": (
                        classes[1] / sum(classes) if sum(classes) else None
                    ),
                }
            for j, destination in enumerate(SPLITS):
                selected = (initial[domain] == i) & (final[domain] == j)
                if i != j:
                    transitions[f"{split}_to_{destination}"] = {
                        "rows": int(selected.sum()),
                        "class_counts": np.bincount(
                            labels[domain][selected], minlength=2
                        ).tolist(),
                    }
        moved = int(np.count_nonzero(initial[domain] != final[domain]))
        fraction = moved / len(initial[domain])
        deltas = {
            split: (
                after[split]["attack_prevalence"] - before[split]["attack_prevalence"]
                if after[split]["attack_prevalence"] is not None
                and before[split]["attack_prevalence"] is not None
                else None
            )
            for split in SPLITS
        }
        reports[domain] = {
            "initial": before,
            "final": after,
            "relocated_unique_rows": moved,
            "relocated_fraction": fraction,
            "transitions": transitions,
            "attack_prevalence_change": deltas,
            "within_policy": fraction <= max_relocated_fraction
            and all(
                delta is not None and abs(delta) <= max_class_prevalence_change
                for delta in deltas.values()
            ),
        }
    return {
        "class_order": ["Normal (0)", "Attack (1)"],
        "domains": reports,
        "policy": {
            "max_relocated_fraction": max_relocated_fraction,
            "max_class_prevalence_change": max_class_prevalence_change,
            "interpretation": "predeclared design review limits, not statistical significance thresholds",
        },
        "quality_gate_passed": all(r["within_policy"] for r in reports.values()),
    }


def write_relocation_report(snapshot, report):
    (snapshot / "relocation_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = []
    for domain, audit in report["domains"].items():
        for split in SPLITS:
            before, after = audit["initial"][split], audit["final"][split]
            outgoing = [
                v
                for k, v in audit["transitions"].items()
                if k.startswith(split + "_to_")
            ]
            incoming = [
                v for k, v in audit["transitions"].items() if k.endswith("_to_" + split)
            ]
            rows.append(
                {
                    "domain": domain,
                    "split": split,
                    "initial_rows": before["rows"],
                    "final_rows": after["rows"],
                    "initial_normal": before["class_counts"][0],
                    "initial_attack": before["class_counts"][1],
                    "final_normal": after["class_counts"][0],
                    "final_attack": after["class_counts"][1],
                    "relocated_out_rows": sum(v["rows"] for v in outgoing),
                    "relocated_out_normal": sum(v["class_counts"][0] for v in outgoing),
                    "relocated_out_attack": sum(v["class_counts"][1] for v in outgoing),
                    "relocated_in_rows": sum(v["rows"] for v in incoming),
                    "initial_attack_prevalence": before["attack_prevalence"],
                    "final_attack_prevalence": after["attack_prevalence"],
                    "attack_prevalence_change": audit["attack_prevalence_change"][
                        split
                    ],
                }
            )
    with (snapshot / "relocation_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _copy_audited_splits(audit, common_root):
    for domain in DOMAINS:
        for split in SPLITS:
            entry = audit["common_inventory"][domain][split]
            source = Path(entry["path"])
            current = inventory(source)
            if current["split_sha256"] != entry["split_sha256"]:
                raise ValueError(f"Data changed since audit: {source}")
            destination = common_root / f"{domain}_{split}"
            destination.mkdir()
            for file in sorted(source.glob("*.parquet")):
                target = destination / file.name
                shutil.copyfile(file, target)


def _load_common_splits(common_root):
    arrays, labels, memberships = {}, {}, {}
    for domain in DOMAINS:
        feature_chunks, label_chunks, split_chunks = [], [], []
        for index, split in enumerate(SPLITS):
            table = pq.read_table(
                common_root / f"{domain}_{split}", columns=[*COMMON_FEATURES, "label"]
            )
            feature_chunks.append(
                np.column_stack([table.column(n).to_numpy() for n in COMMON_FEATURES])
            )
            label_chunks.append(table.column("label").to_numpy())
            split_chunks.append(np.full(len(table), index, dtype=np.int8))
        arrays[domain] = np.concatenate(feature_chunks)
        labels[domain] = np.concatenate(label_chunks)
        memberships[domain] = np.concatenate(split_chunks)
    return arrays, labels, memberships


def _move_collision_groups(connection, values, memberships, new_memberships):
    table = pa.table(
        {
            **{n: values[:, i] for i, n in enumerate(COMMON_FEATURES)},
            "split": memberships,
        }
    )
    connection.register("precision_rows", table)
    try:
        feature_columns = ", ".join(COMMON_FEATURES)
        collisions = connection.execute(
            f"SELECT {feature_columns}, max(split) AS destination "
            "FROM precision_rows "
            f"GROUP BY {feature_columns} HAVING count(distinct split)>1"
        ).fetchnumpy()
        if len(collisions[COMMON_FEATURES[0]]):
            collision_values = np.column_stack(
                [collisions[n] for n in COMMON_FEATURES]
            ).astype(np.float32)

            key_type = np.dtype(f"V{values.shape[1] * values.dtype.itemsize}")
            row_keys = np.ascontiguousarray(values).view(key_type).ravel()
            collision_keys = (
                np.ascontiguousarray(collision_values).view(key_type).ravel()
            )
            order = np.argsort(collision_keys)
            sorted_keys = collision_keys[order]
            positions = np.searchsorted(sorted_keys, row_keys)
            safe_positions = np.minimum(positions, len(sorted_keys) - 1)
            matched = (positions < len(sorted_keys)) & (
                sorted_keys[safe_positions] == row_keys
            )
            target_splits = collisions["destination"][order][safe_positions[matched]]
            new_memberships[matched] = np.maximum(
                new_memberships[matched], target_splits
            )
    finally:
        connection.unregister("precision_rows")


def _fit_without_split_leakage(
    connection,
    arrays,
    labels,
    memberships,
    snapshot,
    started,
    max_relocated_fraction,
    max_class_prevalence_change,
):
    initial_memberships = {
        domain: membership.copy() for domain, membership in memberships.items()
    }
    connection.execute("SET threads=2")
    connection.execute("SET memory_limit='1GB'")
    precision_iterations = []
    processors = {}
    for iteration in range(100):
        new_memberships = {domain: memberships[domain].copy() for domain in DOMAINS}
        for direction, source in (
            ("unsw_to_cicids", "unsw"),
            ("cicids_to_unsw", "cicids"),
        ):
            processor = proposal_processor().fit(
                arrays[source][memberships[source] == 0]
            )
            processors[direction] = processor
            for domain in DOMAINS:
                values = processor.transform(arrays[domain]).astype(np.float32)
                values[values == 0] = 0
                _move_collision_groups(
                    connection, values, memberships[domain], new_memberships[domain]
                )
        moved = {
            domain: {
                split: int(
                    np.count_nonzero(
                        (new_memberships[domain] != memberships[domain])
                        & (memberships[domain] == i)
                    )
                )
                for i, split in enumerate(SPLITS)
            }
            for domain in DOMAINS
        }
        precision_iterations.append(moved)
        print(
            f"Precision leakage audit iteration {iteration}: moved from {moved}",
            flush=True,
        )
        if not any(
            np.any(new_memberships[domain] != memberships[domain]) for domain in DOMAINS
        ):
            break
        for domain in DOMAINS:
            memberships[domain] = new_memberships[domain]
        candidate = relocation_audit(
            initial_memberships,
            memberships,
            labels,
            max_relocated_fraction,
            max_class_prevalence_change,
        )
        if not candidate["quality_gate_passed"]:
            candidate["resources"] = resource_observation(started)
            candidate["precision_cross_split_groups_after_final_fit"] = None
            candidate["status"] = "design_limits_exceeded_before_final_refit"
            write_relocation_report(snapshot, candidate)
            raise ValueError(
                "Relocation exceeds predeclared design limits; inspect relocation_audit.json and review split strategy. Revision not frozen."
            )
    else:
        raise ValueError(
            "Precision collision audit did not converge; snapshot not frozen"
        )
    relocation = relocation_audit(
        initial_memberships,
        memberships,
        labels,
        max_relocated_fraction,
        max_class_prevalence_change,
    )
    relocation["precision_cross_split_groups_after_final_fit"] = 0
    relocation["status"] = "passed"
    write_relocation_report(snapshot, relocation)
    if not relocation["quality_gate_passed"]:
        raise ValueError(
            "Relocation exceeds predeclared design limits; revision not frozen"
        )
    return processors, memberships, precision_iterations


def _write_common_splits(common_root, arrays, labels, memberships, record_frozen_file):
    for domain in DOMAINS:
        for i, split in enumerate(SPLITS):
            destination = common_root / f"{domain}_{split}"

            for copied in destination.glob("*.parquet"):
                copied.unlink()
            selected = memberships[domain] == i
            table = pa.table(
                {
                    **{
                        n: arrays[domain][selected, j]
                        for j, n in enumerate(COMMON_FEATURES)
                    },
                    "label": labels[domain][selected],
                }
            )
            output = destination / "data.parquet"
            pq.write_table(table, output, compression="snappy", row_group_size=65536)
            record_frozen_file(output)


def _write_prepared_split(common_dir, output_dir, processor):
    output_dir.mkdir(parents=True)
    output = output_dir / "data.parquet"
    schema = pa.schema(
        [
            ("features", pa.list_(pa.float32(), len(COMMON_FEATURES))),
            ("label", pa.int64()),
        ]
    )
    rows = 0
    with pq.ParquetWriter(output, schema, compression="snappy") as writer:
        for file in sorted(common_dir.glob("*.parquet")):
            for batch in pq.ParquetFile(file).iter_batches(
                batch_size=65536, columns=[*COMMON_FEATURES, "label"]
            ):
                raw = np.column_stack(
                    [
                        batch.column(name).to_numpy(zero_copy_only=False)
                        for name in COMMON_FEATURES
                    ]
                )
                values = processor.transform(raw).astype(np.float32)
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"Nonfinite prepared data: {common_dir.name}"
                    )
                labels = batch.column("label").to_numpy(
                    zero_copy_only=False
                )
                if not np.isin(labels, [0, 1]).all():
                    raise ValueError("Invalid label")
                vectors = pa.FixedSizeListArray.from_arrays(
                    pa.array(values.ravel()), len(COMMON_FEATURES)
                )
                writer.write_table(
                    pa.table(
                        {
                            "features": vectors,
                            "label": pa.array(labels, type=pa.int64()),
                        },
                        schema=schema,
                    )
                )
                rows += len(values)
    expected = sum(
        pq.ParquetFile(p).metadata.num_rows
        for p in common_dir.glob("*.parquet")
    )
    if rows != expected:
        raise ValueError("Prepared/common row count mismatch")
    return output, rows


def _audit_prepared_splits(feature_root, directions):
    prepared_connection = duckdb.connect()
    try:
        prepared_connection.execute("SET threads=2")
        prepared_connection.execute("SET memory_limit='1GB'")
        prepared_quality = {}
        for direction in directions:
            prepared_quality[direction] = {}
            for domain in DOMAINS:
                union = " UNION ALL ".join(
                    f"SELECT features,label,'{split}' split FROM read_parquet('{feature_root / direction / f'{domain}_{split}' / '*.parquet'}')"
                    for split in SPLITS
                )
                prepared_connection.execute(
                    "CREATE OR REPLACE TEMP VIEW prepared_audit AS " + union
                )
                overlap = prepared_connection.execute(
                    "SELECT count(*) FROM ("
                    "SELECT features FROM prepared_audit GROUP BY features "
                    "HAVING count(distinct split)>1)"
                ).fetchone()[0]
                invalid = prepared_connection.execute(
                    "SELECT count(*) FROM prepared_audit "
                    "WHERE label IS NULL OR label NOT IN (0,1) "
                    f"OR len(features)<>{len(COMMON_FEATURES)} "
                    "OR list_contains(list_transform(features, "
                    "x -> x IS NULL OR NOT isfinite(x)), true)"
                ).fetchone()[0]
                prepared_quality[direction][domain] = {
                    "cross_split_vector_groups": overlap,
                    "invalid_rows": invalid,
                }
                if overlap or invalid:
                    raise ValueError(
                        f"Prepared file quality gate failed: {direction}/{domain}"
                    )
    finally:
        prepared_connection.close()
    return prepared_quality


def _freeze_provenance(snapshot, audit_path, audit, record_frozen_file):
    source_manifest_path = Path(audit["source_manifest"]["path"])
    if file_sha256(source_manifest_path) != audit["source_manifest"]["sha256"]:
        raise ValueError("Source manifest changed since audit")
    source_manifest = json.loads(source_manifest_path.read_text())
    for paths in source_manifest["raw_inputs"].values():
        for name in paths:
            path = Path(name)
            record_frozen_file(path if path.is_absolute() else ROOT / path)
    for path in (
        audit_path,
        source_manifest_path,
        ROOT / "configs/common_features_v2.json",
    ):
        record_frozen_file(path)
    for name in (
        "proposal_suite_v2",
        "proposal_mmd_v2",
        "proposal_mkmmd_v2",
        "proposal_class_aware_v2",
    ):
        config = ROOT / "configs" / f"{name}.json"
        if config.exists():
            record_frozen_file(config)
    policy_path = ROOT / "configs/proposal_data_policy.json"
    if policy_path.exists():
        record_frozen_file(policy_path)
    if "split_replay" in audit:
        replay_path = Path(audit["split_replay"]["path"])
        if file_sha256(replay_path) != audit["split_replay"]["sha256"]:
            raise ValueError("Split replay changed since audit")
        record_frozen_file(replay_path)
    if "raw_identity_audit" in audit:
        raw_audit = audit["raw_identity_audit"]
        raw_path = Path(raw_audit["path"])
        if (
            file_sha256(raw_path) != raw_audit["sha256"]
            or not raw_audit["result"]["quality_gate_passed"]
        ):
            raise ValueError("Raw identity audit changed or failed")
        record_frozen_file(raw_path)
    semantic_review = ROOT / "docs/semantic_feature_review_20261009.md"
    if semantic_review.exists():
        record_frozen_file(semantic_review)
    (snapshot / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n"
    )
    (snapshot / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    record_frozen_file(snapshot / "source_manifest.json")
    record_frozen_file(snapshot / "audit.json")
    return source_manifest


def freeze(
    audit_path,
    revision="canonical",
    conflict_policy="preserve",
    max_relocated_fraction=0.01,
    max_class_prevalence_change=0.005,
):
    started = time.perf_counter()
    if (
        not 0 <= max_relocated_fraction <= 1
        or not 0 <= max_class_prevalence_change <= 1
    ):
        raise ValueError("Relocation design review limits must be in [0,1]")
    audit_path = Path(audit_path)
    audit = json.loads(audit_path.read_text())
    if not audit["quality_gate_passed"]:
        raise ValueError("Quality/leakage/mapping audit failed; cannot freeze")
    if conflict_policy != "preserve":
        raise ValueError(
            "Only explicit preservation of reduced-feature ambiguity is supported"
        )
    if not revision or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in revision
    ):
        raise ValueError("Revision must be a simple unique identifier")
    snapshot = ROOT / "data/revisions" / revision
    model_root = ROOT / "models" / revision
    result_root = ROOT / "results" / revision
    if any(p.exists() for p in (snapshot, model_root, result_root)):
        raise FileExistsError(
            "Use a new revision name; canonical roots are never overwritten"
        )
    common_root, feature_root = snapshot / "common", snapshot / "prepared"
    common_root.mkdir(parents=True)
    frozen_files = []

    def record_frozen_file(path):
        frozen_files.append(
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )

    _copy_audited_splits(audit, common_root)
    arrays, labels, memberships = _load_common_splits(common_root)
    connection = duckdb.connect()
    try:
        processors, memberships, precision_iterations = _fit_without_split_leakage(
            connection,
            arrays,
            labels,
            memberships,
            snapshot,
            started,
            max_relocated_fraction,
            max_class_prevalence_change,
        )
        record_frozen_file(snapshot / "relocation_audit.json")
        record_frozen_file(snapshot / "relocation_audit.csv")
        _write_common_splits(common_root, arrays, labels, memberships, record_frozen_file)
        canonical_quality = {
            domain: audit_common(connection, common_root, domain) for domain in DOMAINS
        }
        for domain in DOMAINS:
            actual_rows = sum(r["rows"] for r in canonical_quality[domain]["class_counts"])
            original_rows = sum(
                r["rows"] for r in audit["common_inventory"][domain].values()
            )
            if (
                actual_rows != original_rows
                or canonical_quality[domain]["cross_split_feature_groups"]
            ):
                raise ValueError(
                    "Canonical revision must preserve every flow and keep raw feature groups in one split"
                )
    finally:
        connection.close()
    (snapshot / "canonical_quality.json").write_text(
        json.dumps(
            {
                "quality": canonical_quality,
                "precision_relocation_iterations": precision_iterations,
                "precision_cross_split_groups_after_final_fit": 0,
                "policy": "Move complete cross-split prepared float32 collision groups in either direction to max split index (test > val > train); refit source-only preprocessing to convergence; retain all flows/labels; no test metric use.",
            },
            indent=2,
        )
        + "\n"
    )
    record_frozen_file(snapshot / "canonical_quality.json")
    del arrays, labels, memberships

    source_manifest = _freeze_provenance(snapshot, audit_path, audit, record_frozen_file)
    counts, preprocessing = {}, {}
    for direction, source in (("unsw_to_cicids", "unsw"), ("cicids_to_unsw", "cicids")):
        print(f"Write exact source-train fitted preprocessing: {direction}", flush=True)
        processor = processors[direction]
        processor.data_revision = revision
        path = model_root / direction / "preprocessor.joblib"
        path.parent.mkdir(parents=True)
        joblib.dump(processor, path)
        record_frozen_file(path)
        preprocessing[direction] = {
            "fit_role": f"{source}_train_only",
            "quantiles": "exact sklearn median/IQR",
            "medians": processor.named_steps["imputer"].statistics_.tolist(),
            "centers": processor.named_steps["scaler"].center_.tolist(),
            "scales": processor.named_steps["scaler"].scale_.tolist(),
            "sha256": file_sha256(path),
        }
        counts[direction] = {}
        for domain in DOMAINS:
            for split in SPLITS:
                output, rows = _write_prepared_split(
                    common_root / f"{domain}_{split}",
                    feature_root / direction / f"{domain}_{split}",
                    processor,
                )
                counts[direction][f"{domain}_{split}"] = rows
                record_frozen_file(output)

    prepared_quality = _audit_prepared_splits(feature_root, processors)
    quality_path = snapshot / "prepared_quality.json"
    quality_path.write_text(json.dumps(prepared_quality, indent=2) + "\n")
    record_frozen_file(quality_path)
    manifest = {
        "resources": resource_observation(started),
        "relocation_audit": str((snapshot / "relocation_audit.json").resolve()),
        "status": "frozen",
        "quality_gate_passed": True,
        "data_revision": revision,
        "protocol": "proposal_v2",
        "flow_counts_preserved": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "common_root": str(common_root.resolve()),
        "feature_root": str(feature_root.resolve()),
        "model_root": str(model_root.resolve()),
        "result_root": str(result_root.resolve()),
        "feature_order": list(COMMON_FEATURES),
        "split_seed": 42,
        "split_rule": "Initial thesis_20261005 Spark xxhash64 raw five-feature keys + lit(42), buckets 70/15/15; complete precision-collision groups moved to max split index until both directions have zero prepared overlap",
        "conflict_policy": "preserve raw flow multiplicity and original labels; reduced-feature ambiguity reported; no majority relabeling",
        "missing_policy": "invalid raw CICIDS durations already null; source-train median imputation downstream of split",
        "precision_collision_policy": "Move complete groups to furthest holdout; refit source-only preprocessing until zero overlap; identical rows and memberships for both directions; retain all labels/flows",
        "precision_relocation_iterations": precision_iterations,
        "canonical_quality": str((snapshot / "canonical_quality.json").resolve()),
        "prepared_quality": str(quality_path.resolve()),
        "semantic_byte_mapping_status": "conditional extractor equivalence; diagnostic sensitivity without bytes is required",
        "preprocessing": preprocessing,
        "prepared_counts": counts,
        "profiles": source_manifest.get("profiles", {}),
        "counts": {
            "protocol": "proposal_v2",
            "data_revision": revision,
            "class_counts": [
                {
                    "domain": domain,
                    "split": row["split"],
                    "label": row["label"],
                    "count": row["rows"],
                }
                for domain, quality in canonical_quality.items()
                for row in quality["class_counts"]
            ],
        },
        "dataset_statistics_stage": source_manifest.get("dataset_statistics_stage"),
        "audit": str((snapshot / "audit.json").resolve()),
        "frozen_files": frozen_files,
    }
    path = snapshot / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (snapshot / "manifest.sha256").write_text(file_sha256(path) + "  manifest.json\n")
    print(f"Frozen manifest: {path}", flush=True)
    return path
