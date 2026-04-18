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

VLM_OUTPUT_FILE = "vlm_diagnosis.json"

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp")

SCORE_LABELS = {
    1: "Definitely No",
    2: "Probably No",
    3: "Uncertain",
    4: "Probably Yes",
    5: "Definitely Yes",
}

_rubrics_cache = {} 


def _load_rubrics(finding: str) -> dict:
    """Load rubrics for a finding, with caching."""
    if finding in _rubrics_cache:
        return _rubrics_cache[finding]
    rubrics_path = os.path.join("plans", f"{finding}_rubrics.json")
    if os.path.exists(rubrics_path):
        with open(rubrics_path, "r", encoding="utf-8") as f:
            rubrics = json.load(f)
        _rubrics_cache[finding] = rubrics
        return rubrics
    _rubrics_cache[finding] = {}
    return {}

SYSTEM_PROMPT = (
    "You are a radiologist assistant analysing chest X-ray images. "
    "You will be given one or more images and a diagnostic question.\n\n"
    "IMPORTANT GUIDELINES:\n"
    "- Grounding models may produce bounding boxes even when the target is "
    "absent. Verify whether the highlighted region genuinely shows the finding.\n"
    "- Before scoring, consider BOTH evidence supporting AND evidence against "
    "the finding (normal variants, artifacts, alternative explanations).\n\n"
    "- Use the FULL 1-5 range. If the finding is clearly present, score 4-5. "
    "If clearly absent, score 1-2. If genuinely ambiguous, score 3.\n"
    "- If the bounding box does NOT highlight the finding or shows an irrelevant region, "
    "score 1-2 even if the bounding box itself exists.\n\n"
    "Respond with ONLY a valid JSON object (no markdown, no extra text):\n"
    '  "evidence_for": brief note on evidence supporting the finding,\n'
    '  "evidence_against": brief note on evidence against the finding or normal variants,\n'
    '  "score": integer from 1 to 5 (1=Definitely No, 2=Probably No, '
    "3=Uncertain, 4=Probably Yes, 5=Definitely Yes),\n"
    '  "reasoning": a brief (2-4 sentences) clinical reasoning integrating both sides.\n'
    "Do NOT include any text outside the JSON object."
)

SYSTEM_PROMPT_TYPE_A = (
    "You are a radiologist assistant analysing chest X-ray images. "
    "You will be given:\n"
    "  1. A computed quantitative metric from upstream tools.\n"
    "  2. Bounding-box visualisation images showing what the grounding model detected.\n"
    "  3. A diagnostic question.\n\n"
    "IMPORTANT GUIDELINES:\n"
    "- Grounding models may produce bounding boxes even when the target structure "
    "is absent. Verify whether the bbox actually captures the intended structure.\n"
    "- Before scoring, consider evidence FOR and AGAINST the finding.\n\n"
    "- Use the FULL 1-5 range. If the finding is clearly present, score 4-5. "
    "If clearly absent, score 1-2. If genuinely ambiguous, score 3.\n"
    "- If the bounding box does NOT highlight the target structure or is clearly wrong, "
    "the computed metric is UNRELIABLE — score 1-2 regardless of the metric value.\n\n"
    "Respond with ONLY a valid JSON object (no markdown, no extra text):\n"
    '  "evidence_for": brief note on evidence supporting the finding,\n'
    '  "evidence_against": brief note on evidence against or normal variants,\n'
    '  "score": integer 1-5 (1=Definitely No … 5=Definitely Yes),\n'
    '  "reasoning": brief clinical reasoning (2-4 sentences),\n'
    '  "bbox_reliability": one of "accurate", "slightly off", "unreliable".\n'
    "Do NOT include any text outside the JSON object."
)


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


def step_tool_ids(step):
    tids = step.get("tool", []) or []
    if not isinstance(tids, list):
        tids = [tids]
    return [int(t) for t in tids]


def action_to_key(action: str) -> str:
    """Convert action text to snake_case key for JSON."""
    s = re.sub(r"[^0-9a-zA-Z]+", " ", (action or "").strip()).strip().lower()
    return "_".join(s.split()) or "unknown"


def is_vlm_step(step):
    return 1 in step_tool_ids(step)


def _has_coding_ancestor(step, plan_by_id):
    """Check if any dependency path leads to a coding step (tool 4)."""
    visited = set()
    queue = list(step.get("input_type", []) or [])
    while queue:
        dep = int(queue.pop(0))
        if dep == 0 or dep in visited:
            continue
        visited.add(dep)
        parent = plan_by_id.get(dep)
        if parent is None:
            continue
        if 4 in step_tool_ids(parent):
            return True
        queue.extend(parent.get("input_type", []) or [])
    return False


def _collect_ancestor_images(step, plan_by_id, save_dir, image_path):
    """Walk the dependency tree and collect all bbox/mask/original images."""
    images = []
    visited = set()
    queue = list(step.get("input_type", []) or [])
    while queue:
        dep = int(queue.pop(0))
        if dep in visited:
            continue
        visited.add(dep)
        if dep == 0:
            images.append(image_path)
            continue
        parent = plan_by_id.get(dep)
        if parent is None:
            continue
        out_path = parent.get("output_path", "")
        if out_path and out_path.lower().endswith(IMAGE_EXTS):
            full = os.path.join(save_dir, out_path)
            if os.path.exists(full):
                images.append(full)
        # Continue up the tree
        queue.extend(parent.get("input_type", []) or [])
    return images


def _collect_ancestor_text(step, plan_by_id, save_dir):
    """Read text from upstream JSON outputs (coding step results)."""
    texts = []
    deps = step.get("input_type", []) or []
    for dep in deps:
        dep = int(dep)
        if dep == 0:
            continue
        parent = plan_by_id.get(dep)
        if parent is None:
            continue
        out_path = parent.get("output_path", "")
        if not out_path or not out_path.lower().endswith(".json"):
            continue
        full = os.path.join(save_dir, out_path)
        if not os.path.exists(full):
            continue
        try:
            with open(full, "r", encoding="utf-8") as f:
                data = json.load(f)
            key = f"step_{dep}"
            val = data.get(key, data)
            if isinstance(val, (dict, list)):
                texts.append(json.dumps(val, ensure_ascii=False))
            else:
                texts.append(str(val))
        except Exception:
            pass
    return texts


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vlm(system_prompt, user_text, image_paths):
    """Call GPT-4o with images + text, return parsed JSON dict."""
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
    raw = completion.choices[0].message.content

    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON object in output
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        # Fallback: return raw text as uncertain
        return {"score": 3, "reasoning": raw, "_parse_error": True}


def normalize_vlm_result(result):
    """Ensure the result has required fields with valid values."""
    score = result.get("score", 3)
    if not isinstance(score, int) or score < 1 or score > 5:
        try:
            score = max(1, min(5, int(score)))
        except (ValueError, TypeError):
            score = 3
    out = {
        "score": score,
        "label": SCORE_LABELS.get(score, "Uncertain"),
        "reasoning": str(result.get("reasoning", "")),
    }
    for optional in ("evidence_for", "evidence_against", "bbox_reliability"):
        if optional in result:
            out[optional] = str(result[optional])
    return out


def run_vlm_step(step, plan_by_id, save_dir, image_path, finding=None):
    """Execute a single VLM step. Returns normalized result dict."""
    is_type_a = _has_coding_ancestor(step, plan_by_id)

    dep_images = []
    dep_texts = []

    deps = step.get("input_type", []) or []
    for dep in deps:
        dep = int(dep)
        if dep == 0:
            dep_images.append(image_path)
            continue
        parent = plan_by_id.get(dep)
        if parent is None:
            continue
        out = parent.get("output_path", "")
        if out.lower().endswith(IMAGE_EXTS):
            full = os.path.join(save_dir, out)
            if os.path.exists(full):
                dep_images.append(full)
            else:
                print(f"    [warn] step {step['id']}: expected image {out} not found, using original only")
        elif out.lower().endswith(".json"):
            full = os.path.join(save_dir, out)
            if os.path.exists(full):
                try:
                    with open(full, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    key = f"step_{dep}"
                    val = data.get(key, data)
                    if isinstance(val, (dict, list)):
                        dep_texts.append(json.dumps(val, ensure_ascii=False))
                    else:
                        dep_texts.append(str(val))
                except Exception:
                    pass

    if is_type_a:
        # Type A: also gather ancestor bbox images for reliability check
        ancestor_imgs = _collect_ancestor_images(step, plan_by_id, save_dir, image_path)
        # Deduplicate while preserving order
        seen = set()
        all_images = []
        for img in ancestor_imgs + dep_images:
            if img not in seen:
                seen.add(img)
                all_images.append(img)
        dep_images = all_images

    if image_path not in dep_images:
        dep_images.insert(0, image_path)

    grounding_failures = []
    for dep in deps:
        dep = int(dep)
        if dep == 0:
            continue
        parent = plan_by_id.get(dep)
        if parent is None:
            continue
        parent_tools = parent.get("tool", []) or []
        if not isinstance(parent_tools, list):
            parent_tools = [parent_tools]
        if 2 in [int(t) for t in parent_tools]:
            gd_path = os.path.join(save_dir, "grounding_data.json")
            if os.path.exists(gd_path):
                try:
                    with open(gd_path, "r", encoding="utf-8") as f:
                        gd = json.load(f)
                    phrase = parent.get("phrase", parent.get("action", "unknown"))
                    if gd.get(phrase) is None:
                        grounding_failures.append(phrase)
                except Exception:
                    pass

    prompt_text = step.get("prompt", "") or step.get("action", "")

    action_key = action_to_key(step.get("action", ""))
    rubrics = _load_rubrics(finding) if finding else {}
    rubric_text = rubrics.get(action_key, "")
    if rubric_text:
        prompt_text += f"\n\n{rubric_text}"

    if dep_texts:
        context = "\n".join(f"- {t}" for t in dep_texts)
        prompt_text = f"{prompt_text}\n\nUpstream quantitative results:\n{context}"

    if is_type_a:
        prompt_text += (
            "\n\nIMPORTANT: The bounding-box images show what the grounding model detected. "
            "Please assess whether the detections look correct and how that affects "
            "your confidence in the computed metric."
        )

    if grounding_failures:
        prompt_text += (
            "\n\nThe image shown is the original frontal CXR (the automated grounding "
            "model did not produce a bounding box for: "
            + ", ".join(grounding_failures) + "). "
            "Assess the finding directly from the visual evidence in the image — make "
            "a fresh, unbiased judgment as a radiologist would. Many true positive "
            "cases may not be detected by automated grounding tools but are still "
            "clearly diagnosable. Do NOT default to 'absent' just because grounding "
            "produced no bbox; rely on your own visual analysis."
        )
    else:
        prompt_text += (
            "\n\nYou are shown BOTH the original chest X-ray AND a grounding overlay "
            "with a bounding box highlighting where the automated model detected a "
            "potential finding.\n\n"
            "IMPORTANT DE-BIAS NOTE: The bounding box is produced by an automated "
            "grounding model that has a HIGH false-positive rate. The PRESENCE of a "
            "bbox does NOT mean the finding is present. Treat the bbox as a region "
            "of interest to examine, then make your OWN independent clinical judgment.\n\n"
            "Examine the highlighted region carefully:\n"
            "- Is the visual evidence inside/around the bbox consistent with the "
            "specific finding being asked about?\n"
            "- Could the bbox be highlighting a NORMAL anatomical structure or an "
            "unrelated artifact?\n"
            "- Consider the overall CXR context (e.g., heart size relative to "
            "thorax, bilateral comparison, density patterns).\n\n"
            "Score based on the actual visual evidence, NOT bbox presence:\n"
            "  - 4-5: Finding clearly present — unambiguous radiographic evidence\n"
            "  - 1-2: Finding absent or bbox highlights normal anatomy\n"
            "  - 3: Genuinely ambiguous — some but inconclusive evidence"
        )

    system = SYSTEM_PROMPT_TYPE_A if is_type_a else SYSTEM_PROMPT
    result = call_vlm(system, prompt_text, dep_images)
    return normalize_vlm_result(result)


def main():
    parser = argparse.ArgumentParser(description="Step 4a: qualitative VLM analysis")
    parser.add_argument("--finding", required=True, help="Finding name, e.g. Atelectasis")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES (not used but for consistency)")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N cases (0=all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip cases that already have vlm_diagnosis.json for this finding's steps")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of concurrent workers for API calls (default: 8)")
    args = parser.parse_args()

    finding = args.finding

    # Load plan
    plan_path = os.path.join("plans", f"{finding}_plan.json")
    if not os.path.exists(plan_path):
        print(f"[skip] No plan found for {finding}")
        return
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    plan_by_id = {int(s["id"]): s for s in plan}

    # Filter VLM steps
    vlm_steps = [s for s in plan if is_vlm_step(s)]
    if not vlm_steps:
        print(f"[info] No VLM steps in {finding} plan. Nothing to do.")
        return

    print(f"[plan] {finding}: {len(vlm_steps)} VLM steps to execute")
    for s in vlm_steps:
        key = action_to_key(s.get("action", ""))
        type_label = "Type A (quant+bbox)" if _has_coding_ancestor(s, plan_by_id) else "Type B (visual)"
        print(f"  step {s['id']}: {key} [{type_label}]")

    # Load cases
    cases = get_definite_cases(finding)
    if args.limit > 0:
        cases = cases.head(args.limit)
    print(f"[data] {len(cases)} cases for {finding}")

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

        vlm_file = os.path.join(save_dir, VLM_OUTPUT_FILE)

        # Load existing results
        if os.path.exists(vlm_file):
            with open(vlm_file, "r", encoding="utf-8") as f:
                vlm_data = json.load(f)
        else:
            vlm_data = {}

        # Execute each VLM step
        case_updated = False
        local_failed = 0
        for step in vlm_steps:
            key = action_to_key(step.get("action", ""))

            if args.skip_existing and key in vlm_data:
                continue

            # Check that required dependency outputs actually exist
            deps = step.get("input_type", []) or []
            deps_ok = True
            for dep in deps:
                dep = int(dep)
                if dep == 0:
                    continue
                parent = plan_by_id.get(dep)
                if parent is None:
                    deps_ok = False
                    break
                out = parent.get("output_path", "")
                if not out:
                    continue
                full = os.path.join(save_dir, out)
                if out.lower().endswith(IMAGE_EXTS) and not os.path.exists(full):
                    deps_ok = False
                    break

            if not deps_ok:
                tqdm.write(f"  [skip] {case_id} step {step['id']} ({key}): "
                           f"dependency output missing")
                continue

            try:
                result = run_vlm_step(step, plan_by_id, save_dir, image_path, finding=finding)
                vlm_data[key] = result
                case_updated = True
            except Exception as e:
                tqdm.write(f"  [error] {case_id} step {step['id']}: {e}")
                local_failed += 1

        if case_updated:
            with open(vlm_file, "w", encoding="utf-8") as f:
                json.dump(vlm_data, f, indent=4, ensure_ascii=False)
            return "success"
        return f"failed:{local_failed}" if local_failed else "skipped"

    success, skipped, failed = 0, 0, 0
    rows = [cases.iloc[i] for i in range(len(cases))]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_one_case, row): row for row in rows}
        for future in tqdm(as_completed(futures), total=len(futures), desc=finding):
            try:
                status = future.result()
                if status == "success":
                    success += 1
                elif status.startswith("failed:"):
                    failed += int(status.split(":")[1])
                else:
                    skipped += 1
            except Exception as e:
                row = futures[future]
                tqdm.write(f"  [error] {int(row['subject_id'])}_{int(row['study_id'])}: {e}")
                failed += 1

    print(f"[done] {finding}: {success} updated, {skipped} skipped, {failed} errors")


if __name__ == "__main__":
    main()
