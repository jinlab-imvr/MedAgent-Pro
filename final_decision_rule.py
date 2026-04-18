import os
import sys
import json
import re
import argparse

import pandas as pd
from tqdm import tqdm

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
sys.path.insert(1, os.path.join(_this_dir, ".."))
from key import SRC_ROOT, RESULTS_DIR

PLANS_DIR = "plans"

SCORE_LABELS = {
    1: "Definitely No",
    2: "Probably No",
    3: "Uncertain",
    4: "Probably Yes",
    5: "Definitely Yes",
}


def action_to_key(action: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", " ", (action or "").strip()).strip().lower()
    return "_".join(s.split()) or "unknown"


def get_definite_cases(finding):
    df = pd.read_csv(os.path.join(SRC_ROOT, "subset_chexpert.csv"))
    mask = df[finding].isin([0.0, 1.0])
    return df[mask].reset_index(drop=True)


def load_weights(finding):
    path = os.path.join(PLANS_DIR, f"{finding}_weights.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_plan_indicators(finding):
    path = os.path.join(PLANS_DIR, f"{finding}_plan.json")
    with open(path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    return [s for s in plan if s.get("output_type", "").strip().lower() == "final indicator"]


FORMULA_CHOICES = ["baseline", "A", "B", "C", "D"]


def compute_reliability(tq, es, td, cr, formula="baseline"):
    """Compute reliability factor for a given formula variant.

    baseline : (tq + es) / 10
    A        : (tq + es) / 10  ×  (6 - td) / 5          (difficulty penalty)
    B        : (tq + es) / 10  ×  cr / 5                (relevance boost)
    C        : (tq + es) / 10  ×  (6 - td) / 5  × cr/5 (both)
    D        : (tq + es + (6 - td) + cr) / 20            (additive)
    """
    base = (tq + es) / 10.0
    if formula == "A":
        return base * (6 - td) / 5.0
    elif formula == "B":
        return base * cr / 5.0
    elif formula == "C":
        return base * (6 - td) / 5.0 * cr / 5.0
    elif formula == "D":
        return (tq + es + (6 - td) + cr) / 20.0
    return base  # baseline


def rule_decision(case_dir, finding, indicators, weights_cfg, formula="baseline"):
    """Pure rule-based Phase 2+3 for one case."""
    # Load per-case data
    vlm_file = os.path.join(case_dir, "vlm_diagnosis.json")
    ev_file = os.path.join(case_dir, "evidence_evaluation.json")

    if not os.path.exists(vlm_file):
        return None

    with open(vlm_file, "r", encoding="utf-8") as f:
        vlm_data = json.load(f)

    evidence_eval = {}
    if os.path.exists(ev_file):
        with open(ev_file, "r", encoding="utf-8") as f:
            ev_all = json.load(f)
        evidence_eval = ev_all.get(finding, {})

    threshold = weights_cfg.get("threshold", 0.40)
    ind_cfg = weights_cfg.get("indicators", {})

    # Build per-indicator entries
    entries = []
    for step in indicators:
        key = action_to_key(step.get("action", ""))
        vlm_entry = vlm_data.get(key, {})
        ev = evidence_eval.get(key, {})

        cfg = ind_cfg.get(key, {})
        base_weight = cfg.get("weight", 0.0)
        role = cfg.get("role", "supportive")

        score = vlm_entry.get("score", 3)
        tool_quality = ev.get("tool_quality", 3)
        evidence_sufficiency = ev.get("evidence_sufficiency", 3)
        task_difficulty = ev.get("task_difficulty", 3)
        clinical_relevance = ev.get("clinical_relevance", 3)

        entries.append({
            "key": key,
            "action": step.get("action", ""),
            "vlm_score": score,
            "tool_quality": tool_quality,
            "evidence_sufficiency": evidence_sufficiency,
            "task_difficulty": task_difficulty,
            "clinical_relevance": clinical_relevance,
            "base_weight": base_weight,
            "role": role,
        })

    adjusted = {}
    discard_reasons = {}
    for e in entries:
        key = e["key"]
        if e["tool_quality"] <= 1:
            # Hard discard: tool output is garbage
            adjusted[key] = 0.0
            discard_reasons[key] = f"tool_quality={e['tool_quality']} (hard discard)"
            continue
        reliability = compute_reliability(
            e["tool_quality"], e["evidence_sufficiency"],
            e["task_difficulty"], e["clinical_relevance"], formula
        )
        adjusted[key] = e["base_weight"] * reliability

    total_w = sum(adjusted.values())
    if total_w <= 0:
        norm_weights = {e["key"]: 0.0 for e in entries}
    else:
        norm_weights = {k: v / total_w for k, v in adjusted.items()}

    contributions = []
    total_score = 0.0
    for e in entries:
        key = e["key"]
        value = (e["vlm_score"] - 1) / 4.0
        w = norm_weights.get(key, 0.0)
        weighted = value * w
        total_score += weighted
        used = key not in discard_reasons
        entry_out = {
            "key": key,
            "action": e["action"],
            "vlm_score": e["vlm_score"],
            "vlm_label": SCORE_LABELS.get(e["vlm_score"], "?"),
            "value": round(value, 4),
            "weight": round(w, 4),
            "weighted_contribution": round(weighted, 4),
            "used": used,
            "tool_quality": e["tool_quality"],
            "evidence_sufficiency": e["evidence_sufficiency"],
            "task_difficulty": e["task_difficulty"],
            "clinical_relevance": e["clinical_relevance"],
            "reliability": round(compute_reliability(
                e["tool_quality"], e["evidence_sufficiency"],
                e["task_difficulty"], e["clinical_relevance"], formula
            ), 4),
        }
        if not used:
            entry_out["discard_reason"] = discard_reasons[key]
        contributions.append(entry_out)

    # Primary evidence gate (disabled — fixed thresholds work better)
    # has_definitive = False
    # for e in entries:
    #     key = e["key"]
    #     if (e["role"] == "definitive"
    #             and key not in discard_reasons
    #             and e["vlm_score"] >= 4):
    #         has_definitive = True
    #         break
    gate_applied = False
    # if not has_definitive:
    #     threshold *= 1.10
    #     gate_applied = True

    diagnosis = "Positive" if total_score >= threshold else "Negative"

    result = {
        "diagnosis": diagnosis,
        "score": round(total_score, 4),
        "threshold": round(threshold, 4),
        "gate_applied": gate_applied,
        "reasoning": f"rule-based formula={formula}",
        "indicators": contributions,
    }
    return result


def process_finding(finding, limit=0, formula="baseline"):
    # Load config
    weights_cfg = load_weights(finding)
    indicators = load_plan_indicators(finding)
    if not indicators:
        print(f"[skip] No final indicators in {finding} plan.")
        return

    ind_keys = [action_to_key(s.get("action", "")) for s in indicators]
    print(f"[plan] {finding}: {len(indicators)} indicators, threshold={weights_cfg.get('threshold', '?')}")
    for k in ind_keys:
        cfg = weights_cfg.get("indicators", {}).get(k, {})
        print(f"  {k[:60]}: w={cfg.get('weight', 0):.3f} role={cfg.get('role', '?')}")

    # Load cases
    cases = get_definite_cases(finding)
    if limit > 0:
        cases = cases.head(limit)
    print(f"[data] {len(cases)} cases, formula={formula}")

    success, skipped = 0, 0

    for _, row in tqdm(cases.iterrows(), total=len(cases), desc=finding):
        subject_id = int(row["subject_id"])
        study_id = int(row["study_id"])
        case_id = f"{subject_id}_{study_id}"
        case_dir = os.path.join(RESULTS_DIR, case_id)

        if not os.path.isdir(case_dir):
            skipped += 1
            continue

        result = rule_decision(case_dir, finding, indicators, weights_cfg, formula)
        if result is None:
            skipped += 1
            continue

        # Save
        fd_path = os.path.join(case_dir, "final_decision.json")
        existing = {}
        if os.path.exists(fd_path):
            with open(fd_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing[finding] = result
        with open(fd_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=4, ensure_ascii=False)

        success += 1

    print(f"[done] {finding}: {success} decided, {skipped} skipped")


def main():
    parser = argparse.ArgumentParser(description="Rule-based final decision")
    parser.add_argument("--finding", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--formula", choices=FORMULA_CHOICES, default="baseline",
                        help="Reliability formula variant (default: baseline)")
    args = parser.parse_args()

    for finding in args.finding:
        process_finding(finding, limit=args.limit, formula=args.formula)


if __name__ == "__main__":
    main()
