from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

UNSW_DICTIONARY = ROOT / "docs" / "data_dictionary" / "feature_dictionary_unsw.csv"

CICIDS_DICTIONARY = ROOT / "docs" / "data_dictionary" / "feature_dictionary_cicids.csv"

CANDIDATES = ROOT / "docs" / "feature_candidates_v2.csv"

OUTPUT = ROOT / "docs" / "feature_mapping_audit.csv"


def decide_mapping(row):
    """
    Decide semantic compatibility.

    KEEP:
        Strong semantic equivalence.

    CONDITIONAL:
        Same network concept and direction,
        but computation/statistic requires verification.

    DROP:
        Semantic incompatibility.
    """

    if not row["concept_match"]:
        return "DROP"

    if not row["direction_match"]:
        return "DROP"

    if not row["unit_compatible"]:
        return "DROP"

    if not row["scope_compatible"]:
        return "DROP"

    if row["statistic_relation"] == "exact":
        return "KEEP"

    if row["statistic_relation"] == "conditional":
        return "CONDITIONAL"

    return "DROP"


def build_reason(row):
    problems = []

    if not row["concept_match"]:
        problems.append("different network concept")

    if not row["direction_match"]:
        problems.append("different traffic direction")

    if not row["unit_compatible"]:
        problems.append("units are not directly compatible")

    if not row["scope_compatible"]:
        problems.append("different protocol or measurement scope")

    statistic = row["statistic_relation"]

    if problems:
        return "; ".join(problems)

    if statistic == "exact":
        return (
            "Same network concept, direction, "
            "aggregation/statistic, compatible unit, "
            "and measurement scope."
        )

    if statistic == "conditional":
        return (
            "Same network concept and direction, "
            "but extractor computation or aggregation "
            "requires further verification."
        )

    return "Aggregation/statistic definitions are " "not sufficiently equivalent."


def requires_manual_review(decision):
    return decision == "CONDITIONAL"


def main():
    candidates = pd.read_csv(CANDIDATES)

    unsw = pd.read_csv(UNSW_DICTIONARY)

    cicids = pd.read_csv(CICIDS_DICTIONARY)
    
    unsw_meta = unsw[
        [
            "feature",
            "definition",
            "reference",
            "extractor_or_origin",
            "audit_notes",
        ]
    ].rename(
        columns={
            "feature": "unsw_feature",
            "definition": "unsw_definition_dict",
            "reference": "unsw_reference",
            "extractor_or_origin": "unsw_extractor",
            "audit_notes": "unsw_audit_notes",
        }
    )

    cicids_meta = cicids[
        [
            "feature",
            "definition",
            "reference",
            "extractor_or_origin",
            "audit_notes",
        ]
    ].rename(
        columns={
            "feature": "cicids_feature",
            "definition": "cicids_definition_dict",
            "reference": "cicids_reference",
            "extractor_or_origin": "cicids_extractor",
            "audit_notes": "cicids_audit_notes",
        }
    )

    audit = candidates.merge(
        unsw_meta,
        on="unsw_feature",
        how="left",
        validate="many_to_one",
    )

    audit = audit.merge(
        cicids_meta,
        on="cicids_feature",
        how="left",
        validate="many_to_one",
    )
    audit["decision"] = audit.apply(
        decide_mapping,
        axis=1,
    )

    audit["reason"] = audit.apply(
        build_reason,
        axis=1,
    )

    audit["manual_review_required"] = audit["decision"].apply(requires_manual_review)


    audit["unsw_evidence"] = (
        audit["unsw_definition_dict"].fillna("")
        + " | "
        + audit["unsw_reference"].fillna("")
    )

    audit["cicids_evidence"] = (
        audit["cicids_definition_dict"].fillna("")
        + " | "
        + audit["cicids_reference"].fillna("")
    )

    # -------------------------
    # Final thesis-friendly table
    # -------------------------

    columns = [
        "unsw_feature",
        "cicids_feature",
        "concept",
        "unsw_direction",
        "cicids_direction",
        "unsw_statistic",
        "cicids_statistic",
        "unsw_unit",
        "cicids_unit",
        "concept_match",
        "direction_match",
        "statistic_relation",
        "unit_compatible",
        "scope_compatible",
        "decision",
        "manual_review_required",
        "reason",
        "unsw_extractor",
        "cicids_extractor",
        "unsw_evidence",
        "cicids_evidence",
        "unsw_audit_notes",
        "cicids_audit_notes",
    ]

    audit = audit[columns]

    priority = {
        "KEEP": 0,
        "CONDITIONAL": 1,
        "DROP": 2,
    }

    audit["_priority"] = audit["decision"].map(priority)

    audit = (
        audit.sort_values(
            [
                "_priority",
                "concept",
                "unsw_feature",
            ]
        )
        .drop(columns="_priority")
        .reset_index(drop=True)
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        OUTPUT,
        index=False,
    )

    print(f"Saved: {OUTPUT}")

    print(f"Mappings: {len(audit)}")

    print("\nDecision summary:")

    print(audit["decision"].value_counts())

    print("\nManual review required:")

    print(
        audit[audit["manual_review_required"]][
            [
                "unsw_feature",
                "cicids_feature",
                "reason",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
