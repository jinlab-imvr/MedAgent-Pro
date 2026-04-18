import os
import sys
import json
import re
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
import openai

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
sys.path.insert(1, os.path.join(_this_dir, ".."))
from key import OPENAI_API_KEY, MODEL, SRC_ROOT, RESULTS_DIR

openai.api_key = OPENAI_API_KEY

SCORE_LABELS = {
    1: "Definitely No",
    2: "Probably No",
    3: "Uncertain",
    4: "Probably Yes",
    5: "Definitely Yes",
}


def get_definite_cases(finding):
    df = pd.read_csv(os.path.join(SRC_ROOT, "subset_chexpert.csv"))
    mask = df[finding].isin([0.0, 1.0])
    return df[mask].reset_index(drop=True)


def action_to_key(action: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", " ", (action or "").strip()).strip().lower()
    return "_".join(s.split()) or "unknown"


def get_final_indicators(plan):
    return [s for s in plan if s.get("output_type", "").strip().lower() == "final indicator"]


def call_vlm_text_only(system_prompt, user_text):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    completion = openai.ChatCompletion.create(model=MODEL, messages=messages)
    return _parse_json_response(completion.choices[0].message.content)


def _parse_json_response(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        return {"_raw": raw, "_parse_error": True}


def run_weight_proposal_and_decision(case_dir, plan, finding, task, evidence_eval):
    """Phase 2: VLM proposes weights + threshold. Phase 3: mechanical computation."""
    indicators = get_final_indicators(plan)
    if not indicators:
        return None

    vlm_file = os.path.join(case_dir, "vlm_diagnosis.json")
    if not os.path.exists(vlm_file):
        return None
    with open(vlm_file, "r", encoding="utf-8") as f:
        vlm_data = json.load(f)

    entries = []
    for step in indicators:
        key = action_to_key(step.get("action", ""))
        vlm_entry = vlm_data.get(key, {})
        ev = evidence_eval.get(key, {})

        score = vlm_entry.get("score", 3)
        entries.append({
            "key": key,
            "action": step.get("action", ""),
            "vlm_score": score,
            "vlm_label": SCORE_LABELS.get(score, "?"),
            "vlm_reasoning": vlm_entry.get("reasoning", "")[:200],
            "tool_quality": ev.get("tool_quality", 3),
            "evidence_sufficiency": ev.get("evidence_sufficiency", 3),
            "task_difficulty": ev.get("task_difficulty", 3),
            "clinical_relevance": ev.get("clinical_relevance", 3),
            "evidence_reasoning": ev.get("reasoning", ""),
        })

    table_parts = []
    for i, e in enumerate(entries, 1):
        table_parts.append(
            f'{i}. "{e["key"]}"\n'
            f'   Action: {e["action"]}\n'
            f'   VLM score: {e["vlm_score"]}/5 ({e["vlm_label"]})\n'
            f'   VLM reasoning: {e["vlm_reasoning"]}\n'
            f'   Evidence: tool_quality={e["tool_quality"]}/5, '
            f'sufficiency={e["evidence_sufficiency"]}/5, '
            f'difficulty={e["task_difficulty"]}/5, '
            f'clinical_relevance={e["clinical_relevance"]}/5\n'
            f'   Evidence note: {e["evidence_reasoning"]}'
        )
    table_text = "\n\n".join(table_parts)
    keys_list = json.dumps([e["key"] for e in entries], ensure_ascii=False)

    # Grounding signal
    grounding_signal = ""
    gd_path = os.path.join(case_dir, "grounding_data.json")
    if os.path.exists(gd_path):
        try:
            with open(gd_path, "r", encoding="utf-8") as f:
                gd = json.load(f)
            if gd.get(finding.lower()) is not None:
                grounding_signal = (
                    f"\nGROUNDING STATUS: Grounding model produced a bbox for "
                    f"'{finding}'. NOTE: grounding has a high false-positive rate — "
                    "judge the Tier-1 indicator based on its VLM score (high score = "
                    "trust it; low score = grounding likely a false alarm).\n"
                )
            else:
                grounding_signal = (
                    f"\nGROUNDING STATUS: Grounding did not detect '{finding}'. "
                    "The Tier-1 indicator was assessed directly from the original "
                    "image (similar to baseline radiologist workflow). Trust its "
                    "VLM score as you would any direct visual assessment.\n"
                )
        except Exception:
            pass

    system = (
        "You are a clinical decision assistant performing evidence-based "
        "diagnosis on chest X-rays. For each diagnostic indicator you must "
        "decide whether the tool output is RELIABLE enough to use, "
        "then assign weights only to the reliable indicators."
    )

    user_text = (
        f"Disease: {finding}\n"
        f"Goal: {task.get('disease', '')}\n\n"
        f"{grounding_signal}\n"
        f"Diagnostic indicators and their assessments:\n{table_text}\n\n"
        "YOUR TASK (two steps):\n\n"
        "STEP 1 — RELIABILITY GATE (binary keep/discard):\n"
        "For each indicator, decide whether to KEEP or DISCARD it.\n"
        "DISCARD an indicator if ANY of the following is true:\n"
        "  - The upstream tool output is clearly wrong (e.g., bounding box "
        "misses the target structure, segmentation mask is inaccurate)\n"
        "  - The evidence is insufficient to support ANY conclusion "
        "(e.g., key structure is obscured or out of frame)\n"
        "  - The computed measurement is obviously unreliable "
        "(e.g., CTR computed from a wrong bounding box)\n"
        "Be decisive: if the tool output is questionable, DISCARD it. "
        "It is better to make a decision with fewer reliable indicators "
        "than to include noisy ones.\n\n"
        "STEP 2 — WEIGHTS & THRESHOLD (only for KEPT indicators):\n"
        "Assign weights ONLY to kept indicators. Weights MUST sum to 1.0.\n"
        "Discarded indicators get weight = 0.\n"
        "Consider:\n"
        f"  a) Clinical importance for diagnosing {finding}\n"
        "  b) Evidence quality among the kept indicators\n"
        "  c) GROUNDING STATUS above — adjust weights accordingly\n"
        "THRESHOLD: Propose a decision threshold in [0, 1].\n"
        "  score = Σ(weight_i × value_i), where value_i = (vlm_score - 1)/4.\n"
        "  Diagnosis = Positive if score ≥ threshold, else Negative.\n\n"
        "  Consider the disease's diagnostic logic:\n"
        "  - OR-logic diseases: LOW threshold (0.20-0.35).\n"
        "  - Diseases with convergent signs: LOW-MODERATE (0.25-0.40).\n"
        "  - Diseases with quantitative indicators: MODERATE (0.35-0.50).\n"
        "  Most findings benefit from thresholds in 0.30-0.50.\n"
        "  Aim for BALANCED accuracy (equal weight on sensitivity and specificity).\n\n"
        f"Indicator keys (use EXACTLY these): {keys_list}\n\n"
        "Respond with ONLY a JSON object:\n"
        "{\n"
        '  "indicators": {\n'
        '    "<indicator_key>": {"use": true/false, "weight": <float>, '
        '"reason": "<why keep or discard>"},\n'
        "    ...\n"
        "  },\n"
        '  "threshold": <float>,\n'
        '  "reasoning": "<2-3 sentences>"\n'
        "}"
    )

    proposal = call_vlm_text_only(system, user_text)

    raw_indicators = proposal.get("indicators", {})
    if not raw_indicators and "weights" in proposal:
        raw_indicators = {k: {"use": True, "weight": v, "reason": ""}
                          for k, v in proposal["weights"].items()}
    threshold = max(0.35, min(0.95, float(proposal.get("threshold", 0.50))))
    reasoning = str(proposal.get("reasoning", ""))

    indicator_roles = {}
    for e in entries:
        key = e["key"]
        indicator_roles[key] = "definitive" if e.get("clinical_relevance", 0) >= 5 else "supportive"

    kept_weights = {}
    discard_reasons = {}
    for e in entries:
        key = e["key"]
        ind_info = raw_indicators.get(key, {})
        use = ind_info.get("use", False)
        if isinstance(use, str):
            use = use.lower() in ("true", "yes", "keep")
        raw_w = float(ind_info.get("weight", 0))
        reason = str(ind_info.get("reason", ""))
        if use and raw_w > 0:
            kept_weights[key] = raw_w
        else:
            discard_reasons[key] = reason or "discarded by reliability gate"

    total_w = sum(kept_weights.values())
    if total_w <= 0:
        norm_weights = {e["key"]: 0.0 for e in entries}
    else:
        norm_weights = {k: v / total_w for k, v in kept_weights.items()}

    weights = {k: round(v, 4) for k, v in norm_weights.items()}
    contributions = []
    total_score = 0.0
    for e in entries:
        key = e["key"]
        value = (e["vlm_score"] - 1) / 4.0
        w = weights.get(key, 0.0)
        weighted = value * w
        total_score += weighted
        used = key not in discard_reasons
        entry_out = {
            "key": key,
            "action": e["action"],
            "vlm_score": e["vlm_score"],
            "vlm_label": e["vlm_label"],
            "value": round(value, 4),
            "weight": round(w, 4),
            "weighted_contribution": round(weighted, 4),
            "used": used,
        }
        if not used:
            entry_out["discard_reason"] = discard_reasons[key]
        contributions.append(entry_out)

    has_definitive = False
    for e in entries:
        key = e["key"]
        if (indicator_roles.get(key) == "definitive"
                and key not in discard_reasons
                and e["vlm_score"] >= 4):
            has_definitive = True
            break
    if not has_definitive:
        threshold *= 1.25

    diagnosis = "Positive" if total_score >= threshold else "Negative"

    result = {
        "diagnosis": diagnosis,
        "score": round(total_score, 4),
        "threshold": round(threshold, 4),
        "reasoning": reasoning,
        "indicators": contributions,
    }

    out_path = os.path.join(case_dir, "final_decision.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing[finding] = result
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4, ensure_ascii=False)

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 2+3: VLM-based final decision")
    parser.add_argument("--finding", required=True, help="Finding name")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N cases (0=all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip cases that already have final_decision for this finding")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of concurrent workers (default: 8)")
    args = parser.parse_args()

    finding = args.finding

    plan_path = os.path.join("plans", f"{finding}_plan.json")
    if not os.path.exists(plan_path):
        print(f"[skip] No plan found for {finding}")
        return
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    with open("tasks.json", "r", encoding="utf-8") as f:
        tasks = json.load(f)
    task = tasks.get(finding, {})

    fi = get_final_indicators(plan)
    if not fi:
        print(f"[info] No final indicators in {finding} plan.")
        return
    print(f"[plan] {finding}: {len(fi)} final indicators")
    for s in fi:
        print(f"  step {s['id']}: {action_to_key(s.get('action', ''))}")

    cases = get_definite_cases(finding)
    if args.limit > 0:
        cases = cases.head(args.limit)
    print(f"[data] {len(cases)} cases for {finding}")

    success, skipped, failed = 0, 0, 0

    def _process_one_case(row):
        subject_id = int(row["subject_id"])
        study_id = int(row["study_id"])
        case_id = f"{subject_id}_{study_id}"

        save_dir = os.path.join(RESULTS_DIR, case_id)
        if not os.path.isdir(save_dir):
            return "skipped"

        vlm_file = os.path.join(save_dir, "vlm_diagnosis.json")
        if not os.path.exists(vlm_file):
            tqdm.write(f"  [skip] {case_id}: no vlm_diagnosis.json")
            return "skipped"

        if args.skip_existing:
            fd_file = os.path.join(save_dir, "final_decision.json")
            if os.path.exists(fd_file):
                with open(fd_file, "r", encoding="utf-8") as f:
                    fd_data = json.load(f)
                if finding in fd_data:
                    return "skipped"

        # Load Phase 1 results
        ev_file = os.path.join(save_dir, "evidence_evaluation.json")
        if os.path.exists(ev_file):
            with open(ev_file, "r", encoding="utf-8") as f:
                ev_data = json.load(f)
            evidence_eval = ev_data.get(finding, {})
        else:
            evidence_eval = {}

        result = run_weight_proposal_and_decision(
            save_dir, plan, finding, task, evidence_eval
        )

        if result:
            return "success"
        return "skipped"

    rows = [cases.iloc[i] for i in range(len(cases))]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_one_case, row): row for row in rows}
        for future in tqdm(as_completed(futures), total=len(futures), desc=finding):
            try:
                status = future.result()
                if status == "success":
                    success += 1
                else:
                    skipped += 1
            except Exception as e:
                row = futures[future]
                tqdm.write(f"  [error] {int(row['subject_id'])}_{int(row['study_id'])}: {e}")
                failed += 1

    print(f"\n[done] {finding}: {success} decided, {skipped} skipped, {failed} errors")


if __name__ == "__main__":
    main()
