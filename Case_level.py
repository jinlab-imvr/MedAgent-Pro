# main.py
import os
import json
import re
from tqdm import tqdm

from CodingAgent import Coding_Agent
from Decider import GPT_Decider, Pro_Decider
from Summary_Module import Summary_Module

# domain tools
from Glaucoma.tools import *  # segment_optic_cup, segment_optic_disc, ...

# utils
from utils import (
    ensure_pkg_inited,
    register_generated_function,
    command_to_fn_name,
    snake,
    inputs_desc,
    read_prev_output,
    build_qual_prompt,
)

OPENAI_API_KEY = ""

# -------------------- data & configs --------------------
data_root = "Glaucoma"

# 1) load task
task_path = os.path.join(data_root, "task.json")
with open(task_path, "r", encoding="utf-8") as f:
    task = json.load(f)

# 2) load plan
plan_path = os.path.join(data_root, "plan.json")
with open(plan_path, "r", encoding="utf-8") as f:
    plan = json.load(f)

# 3) load toolset
toolset_path = os.path.join(data_root, "toolset.json")
with open(toolset_path, "r", encoding="utf-8") as f:
    toolset = json.load(f)

# 4) build mappings
tool_by_id = {int(t["id"]): t for t in toolset if "id" in t}
plan_by_id = {int(s["id"]): s for s in plan if "id" in s}

# 5) function registry (built-ins)
TOOL_FN_REGISTRY = {
    "segment_optic_cup":  segment_optic_cup,
    "segment_optic_disc": segment_optic_disc,   # alias kept if toolset uses 'disc'
}

# -------------------- code generation (Coding_Agent) --------------------
ensure_pkg_inited(data_root)

coder = Coding_Agent(OPENAI_API_KEY)
code_path = os.path.join(data_root, "tools", "GenCode.py")
os.makedirs(os.path.dirname(code_path), exist_ok=True)
if not os.path.exists(code_path):
    with open(code_path, "w", encoding="utf-8") as f:
        f.write("# Generated code\n")

def build_requirement_and_name(step: dict, plan_by_id: dict, tool_by_id: dict, task_input_desc: str):
    """
    Build (function_name, requirement_text) for Coding_Agent.
    Signature is enforced to: fn(inputs, save_dir, save_name)
    """
    base = step.get("action") or step.get("output_type") or "generated_fn"
    fn_name = f"{snake(base)}_{int(step.get('id', 0))}"

    in_desc_list = inputs_desc(step, plan_by_id, tool_by_id, task_input_desc)
    in_desc_str = ", ".join(in_desc_list)
    out_desc = str(step.get("output_type", "")).strip()
    step_id = int(step.get("id", 0))

    requirement = (
        "Implement a Python function with the EXACT signature:\n"
        f"{fn_name}(inputs, save_dir, save_name)\n\n"
        "Semantics:\n"
        "- `inputs` is a LIST; its elements correspond IN ORDER to the step dependencies.\n"
        f"- Conceptual inputs: {in_desc_str}\n"
        f"- Output (conceptual): {out_desc}\n\n"
        "Constraints:\n"
        "- The function is self-contained; add imports inside if needed. No print statements.\n"
        "- Always use `os.path.join(save_dir, save_name)` as the ONLY output file path for any NON-IMAGE result.\n"
        "- NON-IMAGE includes text/json/metrics/numerical values/tables/etc. Do NOT create separate text/json files.\n"
        "- If the output is an IMAGE (extensions: .png/.jpg/.jpeg/.tif/.tiff/.bmp/.gif/.webp), you may write that image to disk using save_dir/save_name as the filename.\n"
        "- When writing NON-IMAGE results, open/create the JSON file at `os.path.join(save_dir, save_name)`, read existing JSON if present, "
        f"update a unique key for this step (e.g., 'step_{step_id}'), and write back with UTF-8 and ensure_ascii=False.\n"
        "- Add a brief docstring explaining inputs (list), outputs, and side effects.\n"
    )
    return fn_name, requirement

for step in plan:
    # only generate for coding steps
    tool_ids = step.get("tool", []) or []
    if not isinstance(tool_ids, list):
        tool_ids = [tool_ids]
    is_coding_step = any("coding" in str(tool_by_id.get(int(tid), {}).get("type", "")).lower()
                         for tid in tool_ids)
    if not is_coding_step:
        continue

    fn_name, requirement = build_requirement_and_name(
        step=step,
        plan_by_id=plan_by_id,
        tool_by_id=tool_by_id,
        task_input_desc=task.get("input", ""),
    )
    coder.generate_function(
        output_file=code_path,
        requirement=requirement,
        enforce_function_name=fn_name,
        extra_context="`inputs` is a list; each item may be a file path or an in-memory object (e.g., numpy array). Handle both gracefully.",
        model="chatgpt-4o-latest",
    )
    register_generated_function(data_root, TOOL_FN_REGISTRY, fn_name)

# -------------------- case-level execution --------------------
img_dir = "/mnt/data0/ziyue/dataset/Glaucoma/REFUGE2/Training400"
name_list = (
    [f"Glaucoma_{f}" for f in os.listdir(os.path.join(img_dir, "Glaucoma"))]
    + [f"Non-Glaucoma_{f}" for f in os.listdir(os.path.join(img_dir, "Non-Glaucoma"))]
)

Analyzer = GPT_Decider(OPENAI_API_KEY)
Summarizer = Summary_Module(OPENAI_API_KEY)
decider = Pro_Decider(OPENAI_API_KEY)

os.makedirs(os.path.join(data_root, "record"), exist_ok=True)

for idx in tqdm(range(2)):  # demo first 2
    example = name_list[idx]
    subdir, file = example.split("_", 1)

    image_path = os.path.join(img_dir, subdir, file)
    if not os.path.exists(image_path):
        continue

    save_dir = os.path.join(data_root, "record", example.split(".")[0])
    os.makedirs(save_dir, exist_ok=True)

    # ---- run each step ----
    for step in plan:
        at = str(step.get("action_type", "")).lower()
        tool_ids = step.get("tool", []) or []
        if not isinstance(tool_ids, list):
            tool_ids = [tool_ids]
        save_name = step.get("output_path", "")

        if at == "quantitative":
            for tid in tool_ids:
                tool = tool_by_id.get(int(tid))
                if tool is None:
                    print(f"[warn] tool id {tid} not found in toolset")
                    continue

                # resolve callable
                if "coding" in str(tool.get("type", "")).lower():
                    # generated function name is derived from step itself
                    fn_name = command_to_fn_name(step.get("action", ""))  # not used; we use generated name instead
                    # rebuild name exactly as build_requirement_and_name did
                    from utils import snake  # local import to avoid unused earlier
                    fn_name = f"{snake(step.get('action') or step.get('output_type') or 'generated_fn')}_{int(step.get('id', 0))}"
                else:
                    fn_name = command_to_fn_name(tool.get("command", ""))

                fn = TOOL_FN_REGISTRY.get(fn_name)
                if fn is None:
                    print(f"[warn] command '{fn_name}' not registered; add it to TOOL_FN_REGISTRY")
                    continue

                # resolve dependencies into concrete inputs
                deps = step.get("input_type", []) or []
                resolved = []
                for dep in deps:
                    dep = int(dep)
                    if dep == 0:
                        resolved.append(image_path)
                    else:
                        prev = plan_by_id.get(dep)
                        if prev:
                            prev_save = prev.get("output_path", "")
                            resolved.append(os.path.join(save_dir, prev_save))

                try:
                    if "coding" in str(tool.get("type", "")).lower():
                        # generated functions expect (inputs, save_dir, save_name)
                        fn(resolved, save_dir, save_name)
                    else:
                        # classic tools expect a single primary input
                        primary = resolved[0] if resolved else image_path
                        fn(primary, save_dir, save_name)
                except Exception as e:
                    print(f"[error] '{fn_name}' failed on {example}: {e}")

        elif at == "qualitative":
            try:
                image_input, text_input = [], []
                for dep in step.get("input_type", []) or []:
                    dep = int(dep)
                    if dep == 0:
                        image_input.append(image_path)
                    else:
                        prev_step = plan_by_id.get(dep, {})
                        prev_save_name = prev_step.get("output_path", "")
                        t, img = read_prev_output(save_dir, prev_save_name, dep)
                        if img:
                            image_input.append(img)
                        if t:
                            text_input.append(t)

                ques = step.get("action", "")
                full_prompt = build_qual_prompt(ques, text_input)

                Analyzer.decide(
                    output_file=os.path.join(save_dir, save_name),
                    prompt=full_prompt,
                    image_paths=image_input,
                    field=f"step_{step.get('id')}"
                )

                summary_prompt = (
                    f"Based on the above text, please provide a brief summary. "
                    f"The task is: {ques}. Does this patient have the abnormal?"
                )
                Summarizer.summarize(
                    input_file=os.path.join(save_dir, save_name),
                    output_file=os.path.join(save_dir, "brief_diagnosis.json"),
                    prompt=summary_prompt,
                    field=f"step_{step.get('id')}"
                )
            except Exception as e:
                print(f"[error] qualitative step id={step.get('id')} failed on {example}: {e}")

    # ---- final decision using indicators ----
    input_desc   = str(task.get("input", "")).strip()
    disease_goal = str(task.get("disease", "")).strip()

    indicators = []
    brief_path = os.path.join(save_dir, "brief_diagnosis.json")
    if os.path.exists(brief_path):
        with open(brief_path, "r", encoding="utf-8") as f:
            brief_data = json.load(f)
    else:
        brief_data = {}

    for step in plan:
        if str(step.get("output_type", "")).lower() != "final indicator":
            continue
        indicators.append({
            "indicator_name": step.get("action", ""),
            "if_abnormal": brief_data.get(f"step_{step.get('id')}", {}),
        })

    decide_prompt = (
        "You are a clinical decision assistant.\n"
        "Task & context:\n"
        f"- Input: {input_desc}\n"
        f"- Goal: {disease_goal}\n\n"
        "Please propose reasonable weights (sum to 1) and a threshold in [0,1]. "
        "Return ONLY a JSON object with the keys: 'weights' (list of {'indicator_name','weight'}), "
        "'threshold' (float), and an optional 'notes' (short string)."
    )

    final_result = decider.decide(
        output_file=os.path.join(save_dir, "final_diagnosis.json"),
        prompt=decide_prompt,
        indicators=indicators,
        field="overall",
    )
    print(final_result.get("overall"))
