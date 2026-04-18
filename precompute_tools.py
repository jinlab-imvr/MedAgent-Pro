import os
import sys
import json
import argparse

import pandas as pd
from tqdm import tqdm
from importlib import import_module, reload

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
sys.path.insert(1, os.path.join(_this_dir, ".."))
from key import OPENAI_API_KEY, SRC_ROOT, RESULTS_DIR

from function import draw_bbox, draw_mask
from utils import snake, IMAGE_EXTS

MAIRA_MODEL = "microsoft/maira-2"
MEDSAM_CKPT = ""

def load_config(finding):
    plan_path = os.path.join("plans", f"{finding}_plan.json")
    if not os.path.exists(plan_path):
        print(f"[skip] No plan found for {finding} ({plan_path})")
        return None, None, None
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    with open("toolset.json", "r", encoding="utf-8") as f:
        toolset = json.load(f)
    with open("tasks.json", "r", encoding="utf-8") as f:
        tasks = json.load(f)
    return plan, toolset, tasks.get(finding, {})


def get_definite_cases(finding):
    """Return rows with definite label (0.0 or 1.0) for the given finding."""
    df = pd.read_csv(os.path.join(SRC_ROOT, "subset_chexpert.csv"))
    mask = df[finding].isin([0.0, 1.0])
    return df[mask].reset_index(drop=True)


_meta_df = None

def _load_metadata():
    global _meta_df
    if _meta_df is None:
        _meta_df = pd.read_csv(os.path.join(SRC_ROOT, "subset_metadata.csv"))
    return _meta_df


def get_frontal_image(subject_id, study_id):
    """Return path to the frontal (PA/AP) image for this study using metadata, or None."""
    meta = _load_metadata()
    rows = meta[(meta["subject_id"] == subject_id) & (meta["study_id"] == study_id)]
    # Filter for frontal views: PA or AP
    frontal = rows[rows["ViewPosition"].isin(["PA", "AP"])]
    if frontal.empty:
        return None
    dicom_id = frontal.iloc[0]["dicom_id"]
    img_dir = os.path.join(SRC_ROOT, "imgs", f"p{subject_id}", f"s{study_id}")
    # MIMIC images are named {dicom_id}.jpg
    img_path = os.path.join(img_dir, f"{dicom_id}.jpg")
    if os.path.exists(img_path):
        return img_path
    # Fallback: first jpg in folder
    if os.path.isdir(img_dir):
        imgs = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))
        return os.path.join(img_dir, imgs[0]) if imgs else None
    return None


def step_tool_ids(step):
    """Return list[int] of tool ids used by a plan step."""
    tids = step.get("tool", []) or []
    if not isinstance(tids, list):
        tids = [tids]
    return [int(t) for t in tids]


def output_exists(save_dir, step):
    """Check whether this step's output already exists."""
    out = step.get("output_path", "")
    if not out:
        return False
    path = os.path.join(save_dir, out)
    if path.lower().endswith(".json"):
        # JSON is shared; check if this step's key is present
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # For coding steps, use step_{id} as key
        return f"step_{step['id']}" in data
    # For grounding steps, also check that grounding_data.json has the phrase
    tools = step.get("tool", []) or []
    if not isinstance(tools, list):
        tools = [tools]
    if 2 in [int(t) for t in tools]:
        gd_path = os.path.join(save_dir, "grounding_data.json")
        if not os.path.exists(gd_path):
            return False
        phrase = _get_grounding_phrase(step)
        with open(gd_path, "r", encoding="utf-8") as f:
            gd = json.load(f)
        if phrase not in gd:
            return False
    return os.path.exists(path)


def save_bbox_to_json(save_dir, phrase, bbox):
    """Persist bbox in grounding_data.json keyed by grounding phrase."""
    path = os.path.join(save_dir, "grounding_data.json")
    data = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    data[phrase] = bbox
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_bbox_from_json(save_dir, phrase):
    """Load bbox saved by a grounding step, keyed by phrase."""
    path = os.path.join(save_dir, "grounding_data.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(phrase)


def load_generated_functions(finding):
    """Import tools/GenCode_{finding}.py and return {fn_name: callable}."""
    module_path = os.path.join("tools", f"GenCode_{finding}.py")
    if not os.path.exists(module_path):
        return {}
    module_name = f"tools.GenCode_{finding}"
    mod = import_module(module_name)
    mod = reload(mod)
    registry = {}
    import inspect
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        registry[name] = obj
    return registry


import re

def _get_grounding_phrase(step):
    """Get the grounding phrase from step's 'phrase' field, falling back to action."""
    phrase = step.get("phrase", "").strip()
    if phrase:
        return phrase
    # Fallback: use action text directly
    return step.get("action", "").strip()


def run_grounding(maira, image_path, step, save_dir):
    """Execute a grounding step (tool 2)."""
    phrase = _get_grounding_phrase(step)
    bbox = maira.phrase_grounding(image_path, phrase)
    step_id = step["id"]
    out_name = step.get("output_path", f"grounding_{step_id}.png")
    base, ext = os.path.splitext(out_name)
    if ext.lower() not in {".png", ".jpg", ".jpeg"}:
        out_name = f"{base}.png"
    output_path = os.path.join(save_dir, out_name)

    save_bbox_to_json(save_dir, phrase, bbox)

    if bbox is not None:
        draw_bbox(image_path, bbox, output_path)
    else:
        from shutil import copyfile
        copyfile(image_path, output_path)

    return bbox


def run_segmentation(medsam, image_path, step, plan_by_id, save_dir):
    """Execute a segmentation step (tool 3). Requires bbox from a prior grounding step."""
    deps = step.get("input_type", []) or []
    bbox = None
    for dep in deps:
        dep = int(dep)
        if dep == 0:
            continue
        parent = plan_by_id.get(dep)
        if parent and 2 in step_tool_ids(parent):
            phrase = _get_grounding_phrase(parent)
            bbox = load_bbox_from_json(save_dir, phrase)
        if bbox is not None:
            break

    if bbox is None:
        print(f"  [warn] step {step['id']}: no bbox available from dependencies — skipping segmentation")
        return None

    step_id = step["id"]
    mask_path = os.path.join(save_dir, step.get("output_path", f"seg_{step_id}.png"))

    # MedSAM expects pixel-coord bbox as string "[x1,y1,x2,y2]"
    from PIL import Image as PILImage
    img = PILImage.open(image_path)
    w, h = img.size
    bbox_pixel = [int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h)]
    medsam.predict_mask(image_path, str(bbox_pixel), mask_path)

    # Also create overlay visualisation
    overlay_path = mask_path.replace(".png", "_overlay.png")
    try:
        draw_mask(image_path, mask_path, overlay_path)
    except Exception:
        pass

    return mask_path


def run_coding(step, plan_by_id, fn_registry, image_path, save_dir):
    """Execute a coding step (tool 4) using a pre-generated function."""
    base = step.get("action") or step.get("output_type") or "generated_fn"
    fn_name = f"{snake(base)}_{int(step['id'])}"
    fn = fn_registry.get(fn_name)
    if fn is None:
        print(f"  [warn] step {step['id']}: function '{fn_name}' not found in registry — skipping")
        return None

    # Resolve inputs — pass bbox coordinates directly for grounding deps
    deps = step.get("input_type", []) or []
    resolved = []
    for dep in deps:
        dep = int(dep)
        if dep == 0:
            resolved.append(image_path)
        else:
            prev = plan_by_id.get(dep)
            if prev:
                prev_tools = step_tool_ids(prev)
                if 2 in prev_tools:
                    # Grounding dep: pass bbox [x1,y1,x2,y2] directly, keyed by phrase
                    phrase = _get_grounding_phrase(prev)
                    bbox = load_bbox_from_json(save_dir, phrase)
                    resolved.append(bbox)
                else:
                    prev_out = prev.get("output_path", "")
                    resolved.append(os.path.join(save_dir, prev_out))

    save_name = step.get("output_path", "diagnosis.json")
    try:
        fn(resolved, save_dir, save_name)
    except Exception as e:
        print(f"  [error] step {step['id']} ({fn_name}): {e}")
        return None
    return os.path.join(save_dir, save_name)


def main():
    parser = argparse.ArgumentParser(description="Step 3: pre-compute tool outputs")
    parser.add_argument("--finding", required=True, help="Finding name, e.g. Atelectasis")
    parser.add_argument("--gpu", default="7", help="CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    finding = args.finding
    plan, toolset, task = load_config(finding)
    if plan is None:
        print(f"[skip] Skipping {finding} — no plan file.")
        return
    plan_by_id = {int(s["id"]): s for s in plan}

    # Determine which tool types are present in the plan
    has_grounding = any(2 in step_tool_ids(s) for s in plan)
    has_segmentation = any(3 in step_tool_ids(s) for s in plan)
    has_coding = any(4 in step_tool_ids(s) for s in plan)

    fn_registry = load_generated_functions(finding) if has_coding else {}

    # Load cases
    cases = get_definite_cases(finding)
    print(f"[data] {len(cases)} cases with definite label for {finding}")

    def _iter_cases():
        for idx in range(len(cases)):
            row = cases.iloc[idx]
            subject_id = int(row["subject_id"])
            study_id = int(row["study_id"])
            image_path = get_frontal_image(subject_id, study_id)
            if image_path is None:
                continue
            save_dir = os.path.join(RESULTS_DIR, f"{subject_id}_{study_id}")
            os.makedirs(save_dir, exist_ok=True)
            yield subject_id, study_id, image_path, save_dir

    if has_grounding:
        import importlib.util
        _spec = importlib.util.spec_from_file_location("maira", os.path.join(_this_dir, "tools", "maira.py"))
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        MAIRA = _mod.MAIRA
        print(f"[init] Loading Maira from {MAIRA_MODEL} ...")
        maira = MAIRA(MAIRA_MODEL)

        grounding_steps = [s for s in plan if 2 in step_tool_ids(s)]
        for subj, study, image_path, save_dir in tqdm(_iter_cases(), total=len(cases), desc=f"{finding} grounding"):
            for step in grounding_steps:
                if not output_exists(save_dir, step):
                    run_grounding(maira, image_path, step, save_dir)

        # Release GPU memory
        del maira
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[mem] Maira released")

    if has_segmentation:
        import importlib, importlib.util, types
        _tools_dir = os.path.join(_this_dir, "tools")
        _medsam_dir = os.path.join(_tools_dir, "MedSAM")
        for pkg_name, pkg_dir in [("tools", _tools_dir), ("tools.MedSAM", _medsam_dir)]:
            if pkg_name not in sys.modules:
                _pkg = types.ModuleType(pkg_name)
                _pkg.__path__ = [pkg_dir]
                _pkg.__package__ = pkg_name
                sys.modules[pkg_name] = _pkg
        from tools.MedSAM.model import MedSAM
        print(f"[init] Loading MedSAM from {MEDSAM_CKPT} ...")
        medsam = MedSAM(MEDSAM_CKPT)

        seg_steps = [s for s in plan if 3 in step_tool_ids(s)]
        for subj, study, image_path, save_dir in tqdm(_iter_cases(), total=len(cases), desc=f"{finding} segmentation"):
            for step in seg_steps:
                if not output_exists(save_dir, step):
                    run_segmentation(medsam, image_path, step, plan_by_id, save_dir)

        del medsam
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[mem] MedSAM released")

    if has_coding:
        coding_steps = [s for s in plan if 4 in step_tool_ids(s)]
        for subj, study, image_path, save_dir in tqdm(_iter_cases(), total=len(cases), desc=f"{finding} coding"):
            for step in coding_steps:
                if not output_exists(save_dir, step):
                    run_coding(step, plan_by_id, fn_registry, image_path, save_dir)

    print(f"[done] Pre-computation finished for {finding}.")


if __name__ == "__main__":
    main()
