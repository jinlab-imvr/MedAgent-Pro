import os
import json
import re
from importlib import import_module, reload
from tqdm import tqdm
from CodingAgent import Coding_Agent
from Summary_Module import Summary_Module

OPENAI_API_KEY = ''

from Glaucoma.tools import *

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

# 4) build mappings for tools and plans
tool_by_id = {int(t["id"]): t for t in toolset if "id" in t}
plan_by_id = {int(s["id"]): s for s in plan if "id" in s}

# 5) register functions
TOOL_FN_REGISTRY = {
    "segment_optic_cup":  segment_optic_cup,
    "segment_optic_disc": segment_optic_disc,
}

def command_to_fn_name(command: str) -> str:
    # extract function name
    if not command:
        return ""
    s = command.strip()
    if "(" in s:
        s = s.split("(", 1)[0]
    return s.strip()

# to register generated functions
def register_generated_function(fn_name: str):
    """Reload GenCode and add the freshly generated function into TOOL_FN_REGISTRY."""
    module_name = f"{data_root}.tools.GenCode"  # e.g., 'Glaucoma.tools.GenCode'
    try:
        mod = import_module(module_name)
    except ModuleNotFoundError:
        # 首次创建 GenCode.py 后 import；此时再 import 一次
        mod = import_module(module_name)
    # 每次生成代码都 reload，拿到最新追加的函数
    mod = reload(mod)
    if hasattr(mod, fn_name):
        TOOL_FN_REGISTRY[fn_name] = getattr(mod, fn_name)
        print(f"[registry] registered {fn_name} -> TOOL_FN_REGISTRY")
    else:
        print(f"[warn] {fn_name} not found in {module_name} after reload")

# 6) generate new functions by CodingAgent
coder = Coding_Agent(OPENAI_API_KEY)
code_path = os.path.join(data_root, "tools", "GenCode.py")
os.makedirs(os.path.dirname(code_path), exist_ok=True)
if not os.path.exists(code_path):
    with open(code_path, "w", encoding="utf-8") as f:
        f.write("# Generated code\n")

def snake(s: str, fallback="generated_fn"):
    s = re.sub(r"[^0-9a-zA-Z]+", " ", str(s or "")).strip().lower()
    s = "_".join(w for w in s.split() if w)
    return s or fallback

def inputs_desc(step):
    deps = step.get("input_type", []) or []
    descs = []
    for dep in deps:
        try:
            dep = int(dep)
        except Exception:
            continue

        if dep == 0:
            descs.append(str(task.get("input", "")).strip())
            continue

        prev = plan_by_id.get(dep)
        if not prev:
            descs.append(f"[missing step {dep}]")
            continue

        tids = prev.get("tool", []) or []
        if not isinstance(tids, list):
            tids = [tids]

        outs = [str(tool_by_id.get(int(tid), {}).get("output", "")).strip() for tid in tids]
        fallback = str(prev.get("output_type", "")).strip()
        descs.append(" / ".join([o for o in outs if o]) or fallback)

    return [d for d in descs if d]  # <-- 返回列表

def build_requirement_and_name(step):
    # name: derive from action or output_type; ensure uniqueness by step id
    base = step.get("action") or step.get("output_type") or "generated_fn"
    fn_name = f"{snake(base)}_{int(step.get('id', 0))}"

    # inputs: descriptions from dependencies (list, order preserved)
    in_desc_list = inputs_desc(step)             # returns list
    in_desc_str  = ", ".join(in_desc_list)

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
        "- NON-IMAGE includes text/json/markdown/metrics/numerical values/tables/etc. Do NOT create separate text/json files.\n"
        "- If the output is an IMAGE (extensions: .png/.jpg/.jpeg/.tif/.tiff/.bmp/.gif/.webp), you may write that image to disk using save_dir/save_name as the filename (or derive an image filename from it).\n"
        "- When writing NON-IMAGE results, open/create the JSON file at `os.path.join(save_dir, save_name)`, read existing JSON if present, update a unique key for this step (e.g., "
        f"'step_{step_id}'), and atomically write back (write to a temp file then replace). Use UTF-8 and ensure_ascii=False.\n"
        "- Add a brief docstring explaining inputs (list), outputs, and side effects.\n"
    )
    return fn_name, requirement



for step in plan:
    # only steps whose tool type contains 'coding'
    tool_ids = step.get("tool", []) or []
    if not isinstance(tool_ids, list):
        tool_ids = [tool_ids]
    if not any("coding" in str(tool_by_id.get(int(tid), {}).get("type", "")).lower() for tid in tool_ids):
        continue

    fn_name, requirement = build_requirement_and_name(step)
    coder.generate_function(
        output_file=code_path,
        requirement=requirement,
        enforce_function_name=fn_name,
        extra_context="`inputs` is a list; each item may be a file path or an in-memory object (e.g., numpy array). Handle both gracefully.",
        model="chatgpt-4o-latest",
    )
    # register in the TOOL_FN_REGISTRY
    register_generated_function(fn_name)

# 7) Case-level analysis 
img_dir = "/mnt/data0/ziyue/dataset/Glaucoma/REFUGE2/Training400"
name_list = (
    [f"Glaucoma_{f}"      for f in os.listdir(os.path.join(img_dir, "Glaucoma"))]
    + [f"Non-Glaucoma_{f}" for f in os.listdir(os.path.join(img_dir, "Non-Glaucoma"))]
)

os.makedirs(os.path.join(data_root, "record"), exist_ok=True)
for idx in tqdm(range(2)):  
    example = name_list[idx]
    subdir, file = example.split("_", 1)

    image_path = os.path.join(img_dir, subdir, file)
    if not os.path.exists(image_path):
        continue

    save_dir = os.path.join(data_root, "record", example.split(".")[0])
    os.makedirs(save_dir, exist_ok=True)

    for step in plan:
        at = str(step.get("action_type", "")).lower()
        if at == "quantitative":
            save_name = step.get("output_path", "")
            # plan 中 tool 是一个 tool-id 的数组；逐个执行
            tool_ids = step.get("tool", [])
            if not isinstance(tool_ids, list):
                tool_ids = [tool_ids]

            for tid in tool_ids:
                tool = tool_by_id.get(int(tid))
                if tool is None:
                    print(f"[warn] tool id {tid} not found in toolset")
                    continue
                
                if "coding" in tool.get("type", "").lower():
                    fn_name, _ = build_requirement_and_name(step)
                else:
                    fn_name = command_to_fn_name(tool.get("command", ""))
                    
                fn = TOOL_FN_REGISTRY.get(fn_name)
                if fn is None:
                    print(f"[warn] command '{fn_name}' not registered; add it to TOOL_FN_REGISTRY")
                    continue

                try:
                    input_type = step.get("input_type", [])
                    if len(input_type) == 1:
                        dep = input_type[0]
                        if dep == 0:
                            fn(image_path, save_dir, save_name)
                        else:
                            input_path = os.path.join(save_dir, plan[dep]["output_path"])
                            fn(input_path, save_dir, save_name)
                    else:
                        inputs = []
                        for dep in input_type:
                            if dep == 0:
                                inputs.append(image_path)
                            else:
                                prev_step = plan_by_id.get(int(dep))
                                if prev_step:
                                    prev_save_name = prev_step.get("output_path", "")
                                    inputs.append(os.path.join(save_dir, prev_save_name))
                        fn(inputs, save_dir, save_name)
                except Exception as e:
                    print(f"[error] '{fn_name}' failed on {example}: {e}")

        elif at == "qualitative":
            continue
            
