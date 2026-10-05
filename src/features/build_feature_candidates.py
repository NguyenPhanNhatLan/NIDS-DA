from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

UNSW_PATH = ROOT / "docs" / "data_dictionary" / "feature_dictionary_unsw.csv"
CICIDS_PATH = ROOT / "docs" / "data_dictionary" / "feature_dictionary_cicids.csv"

OUTPUT_PATH = ROOT / "docs" / "feature_candidates_v2.csv"


TIME_UNITS = {
    "seconds",
    "milliseconds",
    "microseconds",
}


def unit_compatible(unit_unsw: str, unit_cicids: str) -> bool:
    u1 = str(unit_unsw).strip().lower()
    u2 = str(unit_cicids).strip().lower()

    if u1 == u2:
        return True

    if u1 in TIME_UNITS and u2 in TIME_UNITS:
        return True

    return False


def scope_compatible(scope_unsw: str, scope_cicids: str) -> bool:
    s1 = str(scope_unsw).strip().lower()
    s2 = str(scope_cicids).strip().lower()

    if s1 == s2:
        return True

    if s1 == "all" or s2 == "all":
        return True

    return False


def statistic_relation(stat_unsw: str, stat_cicids: str) -> str:
    s1 = str(stat_unsw).strip().lower()
    s2 = str(stat_cicids).strip().lower()

    if s1 == s2:
        return "exact"

    conditional_pairs = {
        ("jitter", "std"),
        ("mean_or_dataset_interpacket_measure", "mean"),
        ("advertised_value", "initial_value"),
    }

    if (s1, s2) in conditional_pairs:
        return "conditional"

    return "mismatch"


def classify_candidate(
    concept_match: bool,
    direction_match: bool,
    statistic_relation_value: str,
    unit_match: bool,
    scope_match: bool,
) -> str:
    if not concept_match or not direction_match:
        return "REJECT"

    if statistic_relation_value == "exact" and unit_match and scope_match:
        return "STRONG"

    if statistic_relation_value == "conditional" and unit_match and scope_match:
        return "CONDITIONAL"

    return "REJECT"


def build_candidates():
    unsw = pd.read_csv(UNSW_PATH)
    cicids = pd.read_csv(CICIDS_PATH)

    # Labels must never be feature candidates.
    unsw = unsw[unsw["category"] != "label"].copy()
    cicids = cicids[cicids["category"] != "label"].copy()

    candidates = []

    for _, u in unsw.iterrows():

        for _, c in cicids.iterrows():

            concept_match = (
                str(u["concept"]).strip().lower() == str(c["concept"]).strip().lower()
            )

            if not concept_match:
                continue

            direction_match = (
                str(u["direction"]).strip().lower()
                == str(c["direction"]).strip().lower()
            )

            stat_relation = statistic_relation(
                u["statistic"],
                c["statistic"],
            )

            unit_match = unit_compatible(
                u["unit"],
                c["unit"],
            )

            scope_match = scope_compatible(
                u["protocol_scope"],
                c["protocol_scope"],
            )

            decision = classify_candidate(
                concept_match=concept_match,
                direction_match=direction_match,
                statistic_relation_value=stat_relation,
                unit_match=unit_match,
                scope_match=scope_match,
            )

            candidates.append(
                {
                    "unsw_feature": u["feature"],
                    "cicids_feature": c["feature"],
                    "concept": u["concept"],
                    "unsw_direction": u["direction"],
                    "cicids_direction": c["direction"],
                    "unsw_statistic": u["statistic"],
                    "cicids_statistic": c["statistic"],
                    "unsw_unit": u["unit"],
                    "cicids_unit": c["unit"],
                    "unsw_scope": u["protocol_scope"],
                    "cicids_scope": c["protocol_scope"],
                    "concept_match": concept_match,
                    "direction_match": direction_match,
                    "statistic_relation": stat_relation,
                    "unit_compatible": unit_match,
                    "scope_compatible": scope_match,
                    "preliminary_decision": decision,
                    "unsw_definition": u["definition"],
                    "cicids_definition": c["definition"],
                    "unsw_notes": u["audit_notes"],
                    "cicids_notes": c["audit_notes"],
                }
            )

    result = pd.DataFrame(candidates)

    priority = {
        "STRONG": 0,
        "CONDITIONAL": 1,
        "REJECT": 2,
    }

    result["_priority"] = result["preliminary_decision"].map(priority)

    result = (
        result.sort_values(
            [
                "_priority",
                "concept",
                "unsw_feature",
                "cicids_feature",
            ]
        )
        .drop(columns="_priority")
        .reset_index(drop=True)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Total candidates: {len(result)}")

    print("\nDecision counts:")

    print(result["preliminary_decision"].value_counts())

    print("\nStrong candidates:")

    strong = result[result["preliminary_decision"] == "STRONG"]

    print(
        strong[
            [
                "unsw_feature",
                "cicids_feature",
                "concept",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    build_candidates()
