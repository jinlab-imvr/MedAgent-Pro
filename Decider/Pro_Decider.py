import os
import json
import openai
import base64

class Pro_Decider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = self.api_key

    # --- optional: support images like GPT_Decider ---
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # --- helpers ---
    def safe_json_parse(self, text: str):
        t = (text or "").strip()
        if t.startswith("```"):
            parts = t.split("```")
            parts = sorted((p.strip() for p in parts), key=len, reverse=True)
            for p in parts:
                if p.startswith("{") and '"threshold"' in p:
                    try:
                        return json.loads(p)
                    except Exception:
                        pass
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            s, e = t.find("{"), t.rfind("}")
            if s != -1 and e != -1 and e > s:
                return json.loads(t[s:e+1])
            raise

    def norm_yesno(self, value):
        if isinstance(value, (int, float)):
            x = float(value)
            return max(0.0, min(1.0, x))
        if isinstance(value, str):
            s = value.strip().lower()
            if s in {"yes", "true", "positive", "abnormal"}:
                return 1.0
            if s in {"no", "false", "negative", "normal"}:
                return 0.0
            # try number-like
            try:
                x = float(s)
                return max(0.0, min(1.0, x))
            except Exception:
                return 0.5
        return 0.5

    def weights_from_model(self, obj, indicators):
        """
        Map model-proposed weights to indicators by name.
        Expect obj['weights'] = [{'indicator_name': str, 'weight': float}, ...]
        Returns a dict name->weight and normalized sum=1 (with small epsilon).
        """
        model_ws = obj.get("weights", [])
        w_map = {}
        for w in model_ws:
            name = str(w.get("indicator_name", "")).strip()
            try:
                val = float(w.get("weight"))
            except Exception:
                continue
            if name:
                w_map[name] = max(0.0, min(1.0, val))
        # ensure all indicators present; if missing, give them equal share of remaining mass
        names = [str(it["indicator_name"]).strip() for it in indicators]
        missing = [n for n in names if n not in w_map]
        total = sum(w_map.values())
        # simple renorm & fill
        if total > 0:
            for k in list(w_map.keys()):
                w_map[k] = w_map[k] / total
        total = sum(w_map.values())
        if missing:
            remain = max(0.0, 1.0 - total)
            share = remain / len(missing) if missing else 0.0
            for n in missing:
                w_map[n] = share
        # final tiny renorm
        total = sum(w_map.values())
        if total == 0:
            # fallback: uniform
            u = 1.0 / max(1, len(names))
            w_map = {n: u for n in names}
        else:
            for k in list(w_map.keys()):
                w_map[k] = w_map[k] / total
        return w_map

    # --- main API ---
    def decide(self, output_file: str, prompt: str, indicators, image_paths=None, field: str = "overall"):
        """
        Ask LLM to allocate weights & threshold; compute final weighted score and write JSON result.

        Args:
            output_file (str): path to write JSON result (merged with existing if present)
            prompt (str): the textual task prompt (already contains task & goal)
            indicators (list[dict]): [{'indicator_name': str, 'if_abnormal': <str|num|dict>}]
            image_paths (str|list[str]|None): optional images
            field (str): key name in the output JSON

        Returns:
            dict: the updated JSON object written to output_file
        """
        if image_paths and not isinstance(image_paths, list):
            image_paths = [image_paths]

        lines = []
        for it in indicators:
            name = str(it.get("indicator_name", "")).strip()
            val  = it.get("if_abnormal", "")
            # keep raw textual value for transparency
            if isinstance(val, (dict, list)):
                val_text = json.dumps(val, ensure_ascii=False)
            else:
                val_text = str(val)
            lines.append(f"- {name}: {val_text}")
        ind_text = "Indicators & current judgements:\n" + "\n".join(lines)

        # Model instruction: return ONLY JSON with weights & threshold
        system_msg = (
            "You are a careful clinical decision assistant. "
            "Given the task and indicator judgements, propose weights that sum to 1 and a threshold in [0,1]. "
            "Return ONLY a JSON object with keys: 'weights' (list of {'indicator_name','weight'}), "
            "'threshold' (float in [0,1]), and optional 'notes'."
        )

        image_messages = []
        if image_paths:
            for path in image_paths:
                base64_image = self.encode_image(path)
                image_messages.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                })

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": image_messages + [{
                "type": "text",
                "text": f"{prompt}\n\n{ind_text}\n\n"
                        "Constraints:\n"
                        "- Sum of weights must be 1.\n"
                        "- Threshold must be in [0,1].\n"
                        "- Do NOT include any explanation outside the JSON object."
            }]}
        ]

        completion = openai.ChatCompletion.create(
            model="chatgpt-4o-latest",
            messages=messages
        )
        raw = completion.choices[0].message.content
        obj = self.safe_json_parse(raw)

        # Pull weights & threshold from model, with robust fallbacks
        w_map = self.weights_from_model(obj, indicators)
        try:
            threshold = float(obj.get("threshold"))
        except Exception:
            threshold = 0.5
        threshold = max(0.0, min(1.0, threshold))

        # Compute weighted score from normalized indicator values
        contribs = []
        score = 0.0
        for it in indicators:
            name = str(it.get("indicator_name", "")).strip()
            val  = self.norm_yesno(it.get("if_abnormal"))
            w    = float(w_map.get(name, 0.0))
            contribs.append({"indicator_name": name, "value": val, "weight": w, "weighted": val * w})
            score += val * w

        diagnosis = "Positive" if score >= threshold else "Negative"

        result_obj = {
            "weights": w_map,
            "threshold": threshold,
            "score": score,
            "diagnosis": diagnosis,
            "contributions": contribs,
            "model_raw": obj.get("notes", "") if isinstance(obj, dict) else "",
        }

        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
        else:
            existing = {}

        existing[field] = result_obj
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=4, ensure_ascii=False)

        return existing
