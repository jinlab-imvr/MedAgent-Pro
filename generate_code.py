import os
import sys
import json
import argparse

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
sys.path.insert(1, os.path.join(_this_dir, ".."))
from key import OPENAI_API_KEY, MODEL_PLANNING

from CodingAgent import Coding_Agent
from utils import snake, inputs_desc, ensure_pkg_inited

MODEL = MODEL_PLANNING


def load_plan_and_toolset(finding):
    plan_path = os.path.join("plans", f"{finding}_plan.json")
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    with open("toolset.json", "r", encoding="utf-8") as f:
        toolset = json.load(f)
    with open("tasks.json", "r", encoding="utf-8") as f:
        tasks = json.load(f)
    return plan, toolset, tasks.get(finding, {})


def build_requirement_and_name(step, plan_by_id, tool_by_id, task_input_desc):
    """Build (function_name, requirement_text) for Coding_Agent."""
    base = step.get("action") or step.get("output_type") or "generated_fn"
    fn_name = f"{snake(base)}_{int(step.get('id', 0))}"

    in_desc_list = inputs_desc(step, plan_by_id, tool_by_id, task_input_desc)
    in_desc_str = ", ".join(in_desc_list)
    out_desc = str(step.get("output_type", "")).strip()
    step_id = int(step.get("id", 0))

    requirement = (
        "Implement a Python function with the EXACT signature:\n"
        f"  {fn_name}(inputs, save_dir, save_name)\n\n"
        "Semantics:\n"
        "- `inputs` is a LIST; elements correspond IN ORDER to the step dependencies.\n"
        f"- Conceptual inputs: {in_desc_str}\n"
        f"- Expected output: {out_desc}\n\n"
        "IMPORTANT about input types:\n"
        "- When a dependency is a GROUNDING step (tool 2), `inputs[i]` is a Python "
        "list [x1_norm, y1_norm, x2_norm, y2_norm] in normalised [0,1] coords, "
        "passed DIRECTLY as an in-memory object (NOT a file path). "
        "Simply use it as: `bbox = inputs[i]`; `x1, y1, x2, y2 = bbox`.\n"
        "- When a dependency is a SEGMENTATION step (tool 3) or other, `inputs[i]` "
        "is a file path (str) to the output image or JSON.\n"
        "- For any quantitative measurement (e.g. cardiothoracic ratio), "
        "compute from the bbox coordinates directly.\n"
        "- For cardiothoracic ratio (CTR): thoracic width is NOT available from "
        "grounding (unreliable). Approximate thoracic width as the full image width "
        "(1.0 in normalised coordinates). CTR_approx = cardiac_bbox_width / 1.0. "
        "This gives a slight underestimate of true CTR.\n\n"
        "Constraints:\n"
        "- Self-contained; add imports inside if needed. No print statements.\n"
        "- Always use `os.path.join(save_dir, save_name)` as the ONLY output file\n"
        "  path for any NON-IMAGE result.\n"
        "- For NON-IMAGE outputs, open/create the JSON file at that path,\n"
        f"  read existing data if present, update key 'step_{step_id}', and write back\n"
        "  with UTF-8 and ensure_ascii=False.\n"
        "- IMAGE outputs may use a distinct file in save_dir.\n"
        "- Add a brief docstring.\n"
    )
    return fn_name, requirement


def main():
    parser = argparse.ArgumentParser(description="Step 2: generate coding functions")
    parser.add_argument("--finding", required=True, help="Finding name, e.g. Cardiomegaly")
    parser.add_argument("--model", default=MODEL, help="OpenAI model for code generation")
    args = parser.parse_args()

    finding = args.finding
    plan, toolset, task = load_plan_and_toolset(finding)

    tool_by_id = {int(t["id"]): t for t in toolset}
    plan_by_id = {int(s["id"]): s for s in plan}

    coding_steps = []
    for step in plan:
        tool_ids = step.get("tool", []) or []
        if not isinstance(tool_ids, list):
            tool_ids = [tool_ids]
        if any(int(tid) == 4 for tid in tool_ids):
            coding_steps.append(step)

    if not coding_steps:
        print(f"[info] No coding steps (tool=4) in {finding} plan. Nothing to generate.")
        return

    # Prepare output file
    ensure_pkg_inited(".")
    code_path = os.path.join("tools", f"GenCode_{finding}.py")
    os.makedirs(os.path.dirname(code_path), exist_ok=True)
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(f"# Auto-generated coding functions for {finding}\n")

    coder = Coding_Agent(OPENAI_API_KEY)
    task_input = task.get("input", "")

    for step in coding_steps:
        fn_name, requirement = build_requirement_and_name(
            step, plan_by_id, tool_by_id, task_input
        )
        print(f"[generate] step {step['id']}: {fn_name}")
        coder.generate_function(
            output_file=code_path,
            requirement=requirement,
            enforce_function_name=fn_name,
            extra_context=(
                "`inputs` is a list; each item may be a file path (str) "
                "or an in-memory object (e.g., numpy array). Handle both."
            ),
            model=args.model,
        )
        print(f"  -> appended to {code_path}")

    print(f"[done] Generated {len(coding_steps)} function(s) in {code_path}")


if __name__ == "__main__":
    main()
