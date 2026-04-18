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

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp")

SCORE_LABELS = {
    1: "Definitely No",
    2: "Probably No",
    3: "Uncertain",
    4: "Probably Yes",
    5: "Definitely Yes",
}


_meta_df = None


def _load_metadata():
    global _meta_df
    if _meta_df is None:
        _meta_df = pd.read_csv(os.path.join(SRC_ROOT, "subset_metadata.csv"))
    return _meta_df


def get_definite_cases(finding):
    df = pd.read_csv(os.path.join(SRC_ROOT, "subset_chexpert.csv"))
    mask = df[finding].isin([0.0, 1.0])
    return df[mask].reset_index(drop=True)


def get_frontal_image(subject_id, study_id):
    meta = _load_metadata()
    rows = meta[(meta["subject_id"] == subject_id) & (meta["study_id"] == study_id)]
    frontal = rows[rows["ViewPosition"].isin(["PA", "AP"])]
    if frontal.empty:
        return None
    dicom_id = frontal.iloc[0]["dicom_id"]
    img_path = os.path.join(SRC_ROOT, "imgs", f"p{subject_id}", f"s{study_id}", f"{dicom_id}.jpg")
    if os.path.exists(img_path):
        return img_path
    img_dir = os.path.dirname(img_path)
    if os.path.isdir(img_dir):
        imgs = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))
        return os.path.join(img_dir, imgs[0]) if imgs else None
    return None


def action_to_key(action: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", " ", (action or "").strip()).strip().lower()
    return "_".join(s.split()) or "unknown"


def get_final_indicators(plan):
    return [s for s in plan if s.get("output_type", "").strip().lower() == "final indicator"]

def trace_evidence_chain(step, plan_by_id, save_dir, image_path):
    """Walk dependency tree for a final indicator step."""
    images = []
    texts = []
    descs = []
    visited = set()

    def _trace(step_id):
        if step_id in visited:
            return
        visited.add(step_id)
        if step_id == 0:
            return
        parent = plan_by_id.get(step_id)
        if parent is None:
            return
        for dep in (parent.get("input_type", []) or []):
            _trace(int(dep))
        out = parent.get("output_path", "")
        if not out:
            return
        full = os.path.join(save_dir, out)
        if out.lower().endswith(IMAGE_EXTS):
            if os.path.exists(full) and full not in images:
                images.append(full)
                tool_type = "bbox" if "bbox" in out else ("mask" if "mask" in out else "output")
                phrase = parent.get("phrase", parent.get("action", ""))
                descs.append(f"{tool_type} for '{phrase}' (step {step_id})")
        elif out.lower().endswith(".json"):
            if os.path.exists(full):
                try:
                    with open(full, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    key = f"step_{step_id}"
                    val = data.get(key, data)
                    if isinstance(val, (dict, list)):
                        texts.append(f"Step {step_id} ({parent.get('action', '')}): "
                                     f"{json.dumps(val, ensure_ascii=False)}")
                    else:
                        texts.append(f"Step {step_id} ({parent.get('action', '')}): {val}")
                except Exception:
                    pass

    for dep in (step.get("input_type", []) or []):
        _trace(int(dep))

    return images, texts, descs


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vlm_with_images(system_prompt, user_text, image_paths):
    """Call VLM with images, return parsed JSON dict."""
    image_messages = []
    for p in image_paths:
        b64 = encode_image(p)
        image_messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": image_messages + [{"type": "text", "text": user_text}]},
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


def run_evidence_evaluation(case_dir, plan, plan_by_id, image_path, finding, task):
    """Phase 1: VLM evaluates evidence quality for each final indicator."""
    indicators = get_final_indicators(plan)
    if not indicators:
        return {}

    all_images = [image_path]
    all_descs = ["Original chest X-ray"]
    indicator_info = []

    for step in indicators:
        key = action_to_key(step.get("action", ""))
        imgs, txts, img_descs = trace_evidence_chain(step, plan_by_id, case_dir, image_path)

        img_refs = []
        for img, desc in zip(imgs, img_descs):
            if img not in all_images:
                all_images.append(img)
                all_descs.append(desc)
            idx = all_images.index(img) + 1
            img_refs.append(f"Image {idx} ({desc})")

        vlm_file = os.path.join(case_dir, "vlm_diagnosis.json")
        vlm_score = None
        vlm_reasoning = ""
        if os.path.exists(vlm_file):
            with open(vlm_file, "r", encoding="utf-8") as f:
                vlm_data = json.load(f)
            if key in vlm_data:
                vlm_score = vlm_data[key].get("score")
                vlm_reasoning = vlm_data[key].get("reasoning", "")

        indicator_info.append({
            "key": key,
            "action": step.get("action", ""),
            "image_refs": img_refs,
            "text_refs": txts,
            "vlm_score": vlm_score,
            "vlm_reasoning": vlm_reasoning[:200],
        })

    image_list_text = "\n".join(f"  Image {i+1}: {d}" for i, d in enumerate(all_descs))

    indicator_parts = []
    for i, info in enumerate(indicator_info, 1):
        refs = ", ".join(info["image_refs"]) if info["image_refs"] else "no tool output images"
        score_text = (f"VLM judgment: {info['vlm_score']}/5 ({SCORE_LABELS.get(info['vlm_score'], '?')})"
                      if info["vlm_score"] else "VLM judgment: not available")
        part = (f'{i}. "{info["key"]}"\n'
                f'   Action: {info["action"]}\n'
                f'   Evidence images: {refs}')
        if info["text_refs"]:
            ctx = "\n    ".join(info["text_refs"])
            part += f"\n   Computed results:\n    {ctx}"
        part += f"\n   {score_text}"
        if info["vlm_reasoning"]:
            part += f"\n   VLM reasoning: {info['vlm_reasoning']}"
        indicator_parts.append(part)

    indicator_text = "\n\n".join(indicator_parts)

    system = (
        "You are a radiologist evaluating the quality of automated diagnostic "
        "tool outputs for chest X-ray analysis. You will assess whether the "
        "evidence chain supporting each diagnostic indicator is reliable."
    )

    user_text = (
        f"Disease: {finding}\n"
        f"Task: {task.get('disease', '')}\n\n"
        f"Images provided (in order):\n{image_list_text}\n\n"
        "For each indicator below, evaluate four dimensions:\n"
        "- tool_quality (1-5): How accurate are the upstream tool outputs "
        "(bounding boxes, segmentation masks)? Are they correctly localising "
        "the intended structure/region? "
        "(1=completely wrong/missing target, 3=partially correct, 5=accurate)\n"
        "- evidence_sufficiency (1-5): Is there enough visual evidence to "
        "support a confident clinical judgment? "
        "(1=no useful evidence, 3=partial, 5=strong unambiguous evidence)\n"
        "- task_difficulty (1-5): How inherently difficult is this specific "
        "indicator to assess on a single frontal CXR? "
        "(1=straightforward, 3=moderate, 5=very challenging)\n"
        f"- clinical_relevance (1-5): How directly relevant is this indicator "
        f"for diagnosing {finding} specifically? "
        "(1=completely unrelated to the target disease"
        "placement for diagnosing pneumonia; 2=tangentially related; "
        "3=moderately related as secondary sign; 4=closely related; "
        "5=primary/pathognomonic sign of the disease)\n\n"
        f"Indicators:\n{indicator_text}\n\n"
        "Respond with ONLY a JSON object mapping each indicator key to its "
        "evaluation. Keys must match exactly:\n"
        "{\n"
        '  "indicator_key": {\n'
        '    "tool_quality": <int 1-5>,\n'
        '    "evidence_sufficiency": <int 1-5>,\n'
        '    "task_difficulty": <int 1-5>,\n'
        '    "clinical_relevance": <int 1-5>,\n'
        '    "reasoning": "<1-2 sentences>"\n'
        "  }\n"
        "}"
    )

    result = call_vlm_with_images(system, user_text, all_images)

    # Validate
    validated = {}
    for info in indicator_info:
        key = info["key"]
        if key in result and isinstance(result[key], dict):
            entry = result[key]
            validated[key] = {
                "tool_quality": max(1, min(5, int(entry.get("tool_quality", 3)))),
                "evidence_sufficiency": max(1, min(5, int(entry.get("evidence_sufficiency", 3)))),
                "task_difficulty": max(1, min(5, int(entry.get("task_difficulty", 3)))),
                "clinical_relevance": max(1, min(5, int(entry.get("clinical_relevance", 3)))),
                "reasoning": str(entry.get("reasoning", "")),
            }
        else:
            validated[key] = {
                "tool_quality": 3,
                "evidence_sufficiency": 3,
                "task_difficulty": 3,
                "clinical_relevance": 3,
                "reasoning": "Not evaluated (missing from VLM response)",
            }

    out_path = os.path.join(case_dir, "evidence_evaluation.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing[finding] = validated
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4, ensure_ascii=False)

    return validated


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Evidence evaluation")
    parser.add_argument("--finding", required=True, help="Finding name")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N cases (0=all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip cases that already have evidence_evaluation for this finding")
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
    plan_by_id = {int(s["id"]): s for s in plan}

    # Load task
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

        image_path = get_frontal_image(subject_id, study_id)
        if image_path is None:
            return "skipped"

        save_dir = os.path.join(RESULTS_DIR, case_id)
        if not os.path.isdir(save_dir):
            return "skipped"

        vlm_file = os.path.join(save_dir, "vlm_diagnosis.json")
        if not os.path.exists(vlm_file):
            tqdm.write(f"  [skip] {case_id}: no vlm_diagnosis.json")
            return "skipped"

        # Skip if already done
        if args.skip_existing:
            ev_file = os.path.join(save_dir, "evidence_evaluation.json")
            if os.path.exists(ev_file):
                with open(ev_file, "r", encoding="utf-8") as f:
                    ev_data = json.load(f)
                if finding in ev_data:
                    return "skipped"

        run_evidence_evaluation(save_dir, plan, plan_by_id, image_path, finding, task)
        return "success"

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

    print(f"\n[done] {finding}: {success} evaluated, {skipped} skipped, {failed} errors")


if __name__ == "__main__":
    main()
