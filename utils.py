# utils.py
import os
import re
import json
from importlib import import_module, reload

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp")


# ---------- package / registry helpers ----------

def ensure_pkg_inited(data_root: str):
    """Ensure {data_root}/ and {data_root}/tools/ are importable packages."""
    for pkg_dir in [data_root, os.path.join(data_root, "tools")]:
        os.makedirs(pkg_dir, exist_ok=True)
        init_py = os.path.join(pkg_dir, "__init__.py")
        if not os.path.exists(init_py):
            with open(init_py, "a", encoding="utf-8"):
                pass  # touch


def register_generated_function(data_root: str, registry: dict, fn_name: str):
    """
    Reload {data_root}.tools.GenCode and register the new function object into registry.
    Example: register_generated_function("Glaucoma", TOOL_FN_REGISTRY, "compute_cdr_6")
    """
    module_name = f"{data_root}.tools.GenCode"
    mod = import_module(module_name) if module_name not in globals() else import_module(module_name)
    mod = reload(mod)
    if hasattr(mod, fn_name):
        registry[fn_name] = getattr(mod, fn_name)
        print(f"[registry] registered {fn_name} -> TOOL_FN_REGISTRY")
    else:
        print(f"[warn] {fn_name} not found in {module_name} after reload")


def command_to_fn_name(command: str) -> str:
    """Extract function name from a string like 'segment_optic_disc()' -> 'segment_optic_disc'."""
    if not command:
        return ""
    s = command.strip()
    return s.split("(", 1)[0].strip() if "(" in s else s


# ---------- naming / prompt helpers ----------

def snake(s: str, fallback: str = "generated_fn") -> str:
    """Convert arbitrary text to snake_case."""
    s = re.sub(r"[^0-9a-zA-Z]+", " ", str(s or "")).strip().lower()
    s = "_".join(w for w in s.split() if w)
    return s or fallback


def inputs_desc(step: dict, plan_by_id: dict, tool_by_id: dict, task_input_desc: str):
    """
    Resolve step['input_type'] into a LIST of human-readable descriptors.
    Priority: previous step's tool.output -> fallback previous step's output_type -> task input.
    """
    deps = step.get("input_type", []) or []
    descs = []
    for dep in deps:
        try:
            dep = int(dep)
        except Exception:
            continue

        if dep == 0:
            descs.append(str(task_input_desc).strip())
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

    return [d for d in descs if d]


# ---------- I/O helpers for qualitative steps ----------

def json_to_text(value, max_chars: int = 2000) -> str:
    """Convert any JSON value (str/num/bool/dict/list) to compact text for prompts."""
    if isinstance(value, str):
        s = value.strip()
    else:
        s = json.dumps(value, ensure_ascii=False)
    return s if len(s) <= max_chars else (s[:max_chars] + " …[truncated]")


def read_prev_output(save_dir: str, filename: str, dep_id: int):
    """
    Return (text, image_path). JSON -> prefer data['step_<dep_id>'] else whole file.
    Image -> (None, path). Text file -> (text, None). Missing -> (None, None).
    """
    if not filename:
        return None, None
    path = os.path.join(save_dir, filename)
    if not os.path.exists(path):
        return None, None

    low = filename.lower()
    if low.endswith(IMAGE_EXTS):
        return None, path
    if low.endswith(".json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            key = f"step_{dep_id}"
            val = data.get(key, data)
            return json_to_text(val), None
        except Exception as e:
            return f"[error reading {filename}: {e}]", None
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip(), None
    except Exception as e:
        return f"[error reading {filename}: {e}]", None


def build_qual_prompt(base_question: str, texts: list[str]) -> str:
    """Attach upstream texts as bullet-point context."""
    if not texts:
        return base_question
    bullets = "\n".join(f"- {t}" for t in texts if t)
    return f"{base_question}\n\nContext:\n{bullets}"
