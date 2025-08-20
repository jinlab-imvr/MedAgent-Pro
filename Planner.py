# planner_agent.py (updated)
import os
import json
import openai

class Planner:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    # ---------- helpers ----------
    def _format_toolset_for_prompt(self, toolset):
        """
        toolset item format:
        { "id": 1, "type": "...", "function": "...", "input": "...", "output": "..." }
        """
        if not isinstance(toolset, list):
            return ""
        lines = []
        for idx, t in enumerate(toolset, 1):
            tid   = t.get("id", "")
            ttype = str(t.get("type", "")).strip()
            func  = str(t.get("function", "")).strip()
            tin   = str(t.get("input", "")).strip()
            tout  = str(t.get("output", "")).strip()
            lines.append(f"{idx}. [tool_id: {tid}] [type: {ttype}] {func} (input: {tin} -> output: {tout})")
        return "\n".join(lines)

    def _build_messages(self, rag_text, prompt, toolset):
        if isinstance(rag_text, list):
            rag_text = "\n\n".join([str(x) for x in rag_text])
        else:
            rag_text = str(rag_text)

        toolset_text = self._format_toolset_for_prompt(toolset) if toolset else ""

        system_msg = (
            "You are a planning assistant for medical diagnosis workflows. "
            "Given RAG text, a task, and an AVAILABLE TOOLSET, return ONLY a strict JSON array of steps. "
            "Each step must contain exactly these fields: "
            "id, tool, action_type, action, input_type, output_type, output_path. "
            "Rules: "
            "(1) id starts from 1 and increases by 1; "
            "(2) tool is an ARRAY of integers; each integer must be a valid tool 'id' from the toolset; "
            "(3) action_type is a STRING: either 'qualitative' or 'quantitative' (lowercase); "
            "(4) input_type is an ARRAY of integers; use 0 for raw/original inputs, or a prior step's id when the input comes from that step's output; "
            "(5) strings for all non-array fields; no extra keys; output ONLY the JSON array. "
            "(6) The field output_type MUST be EXACTLY one of: 'intermediate result' or 'final indicator'. "
            "(7) For any non-image output, set output_path EXACTLY to 'diagnosis.json'. Only image outputs (e.g., .png/.jpg/.jpeg/.tif/.bmp/.gif/.webp) may use distinct file paths."
            "(8) OBSERVE potential QUALITATIVE indicators; list EACH indicator as a SEPARATE step (do not bundle multiple indicators in one step). "
            "(9) Prefer tools of type containing 'vlm' or visual-language capabilities when observing qualitative indicators."
        )

        user_text = (
            "RAG text:\n"
            f"{rag_text}\n\n"
            "Task:\n"
            f"{prompt}\n\n"
            "Available tools:\n"
            f"{toolset_text}\n\n"
            "Return ONLY the JSON array."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": [{"type": "text", "text": user_text}]}
        ]
        return messages


    def _safe_json_parse(self, text):
        t = text.strip()
        if t.startswith("```"):
            parts = sorted(t.split("```"), key=len, reverse=True)
            for c in parts:
                c = c.strip()
                if c.startswith("[") or c.startswith("{"):
                    try:
                        return json.loads(c)
                    except Exception:
                        pass
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            start = t.find("["); end = t.rfind("]")
            if start != -1 and end != -1 and end > start:
                return json.loads(t[start:end+1])
            raise

    def _coerce_int_list(self, val):
        """Coerce to list[int]. Accept list/number/str or JSON-like '[1,2]'."""
        if isinstance(val, list):
            out = []
            for x in val:
                if isinstance(x, (int, float)) and int(x) == x:
                    out.append(int(x))
                else:
                    out.append(int(str(x).strip()))
            return out
        if isinstance(val, (int, float)) and int(val) == val:
            return [int(val)]
        s = str(val).strip()
        if s == "":
            return []
        if s.startswith("["):
            return [int(x) for x in json.loads(s)]
        return [int(s)]

    def _allowed_tool_ids(self, toolset):
        if not isinstance(toolset, list):
            return None
        ids = set()
        for t in toolset:
            try:
                ids.add(int(t.get("id")))
            except Exception:
                pass
        return ids if ids else None

    def _validate_and_clean(self, data, toolset=None):
        required = ["id", "tool", "action_type", "action", "input_type", "output_type", "output_path"]
        if not isinstance(data, list):
            raise ValueError("Planner output is not a JSON array.")

        allowed_ids = self._allowed_tool_ids(toolset)

        cleaned = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Plan item #{i} is not an object.")
            for k in required:
                if k not in item:
                    raise ValueError(f"Missing field '{k}' in plan item #{i}.")

            # id
            try:
                _id = int(item["id"])
            except Exception:
                raise ValueError(f"Field 'id' in plan item #{i} must be an integer.")

            # tool
            try:
                tool_ids = self._coerce_int_list(item["tool"])
            except Exception:
                raise ValueError(f"Field 'tool' in plan item id={_id} must be an int/list-of-ints (tool ids).")
            if allowed_ids is not None:
                for tid in tool_ids:
                    if tid not in allowed_ids:
                        raise ValueError(f"Plan item id={_id} references tool id {tid} not in toolset ids {sorted(allowed_ids)}.")

            # action_type
            at = ("" if item["action_type"] is None else str(item["action_type"]).strip().lower())
            if at not in {"qualitative", "quantitative"}:
                raise ValueError(f"Field 'action_type' in plan item id={_id} must be 'qualitative' or 'quantitative'.")
            action_type = at

            # strings
            action      = "" if item["action"] is None else str(item["action"])
            output_type = "" if item["output_type"] is None else str(item["output_type"])
            output_path = "" if item["output_path"] is None else str(item["output_path"])

            # ---- New: qualitative 单指标检查（防止把多个 indicator 合并成一步）----
            if action_type == "qualitative":
                lowered = action.lower()
                if any(sep in lowered for sep in [",", " and ", " / "]):
                    raise ValueError(
                        f"Qualitative step id={_id} seems to bundle multiple indicators in action='{action}'. "
                        "List EACH indicator as a SEPARATE step."
                    )

            # input_type
            try:
                input_ids = self._coerce_int_list(item["input_type"])
            except Exception:
                raise ValueError(f"Field 'input_type' in plan item id={_id} must be an int/list-of-ints.")

            cleaned.append({
                "id": _id,
                "tool": tool_ids,
                "action_type": action_type,
                "action": action,
                "input_type": input_ids,
                "output_type": output_type,
                "output_path": output_path,
            })

        # id 连续与依赖顺序
        cleaned.sort(key=lambda x: x["id"])
        n = len(cleaned)
        expected_ids = list(range(1, n + 1))
        actual_ids = [r["id"] for r in cleaned]
        if actual_ids != expected_ids:
            raise ValueError(f"ids must be consecutive starting at 1, got {actual_ids}")

        seen = set()
        for rec in cleaned:
            for dep in rec["input_type"]:
                if dep == 0:
                    continue
                if dep not in seen:
                    raise ValueError(f"Step id={rec['id']} has input dependency {dep} not produced by any prior step.")
            seen.add(rec["id"])

        def _is_image_path(p: str) -> bool:
            p = (p or "").lower().strip()
            return any(p.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"))

        ALLOWED_TYPES = {"intermediate result", "final indicator"}
        for rec in cleaned:
            ot = (rec.get("output_type") or "").strip().lower()

            # 规范 output_type
            if ot in ALLOWED_TYPES:
                rec["output_type"] = ot
            else:
                if ("final" in ot) or ("indicator" in ot):
                    rec["output_type"] = "final indicator"
                else:
                    rec["output_type"] = "intermediate result"

            # 非图片：统一 diagnosis.json
            op = (rec.get("output_path") or "").strip()
            if not _is_image_path(op):
                rec["output_path"] = "diagnosis.json"

        return cleaned

    def plan(self, output_path, prompt, rag_text, filename="plan.json",
             model="chatgpt-4o-latest", toolset=None):
        os.makedirs(output_path, exist_ok=True)
        messages = self._build_messages(rag_text, prompt, toolset)

        completion = openai.ChatCompletion.create(model=model, messages=messages)
        raw = completion.choices[0].message.content

        data = self._safe_json_parse(raw)
        plan = self._validate_and_clean(data, toolset=toolset)

        out_file = os.path.join(output_path, filename)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)

        return plan
