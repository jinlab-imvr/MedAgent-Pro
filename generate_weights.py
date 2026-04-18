import os, sys, json, re, argparse

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
sys.path.insert(1, os.path.join(_this_dir, ".."))
from key import OPENAI_API_KEY, MODEL
import openai

openai.api_key = OPENAI_API_KEY
PLANS_DIR = "plans"


def action_to_key(action: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", " ", (action or "").strip()).strip().lower()
    return "_".join(s.split()) or "unknown"


def generate_weights(finding, model=MODEL):
    plan_file = os.path.join(PLANS_DIR, f"{finding}_plan.json")
    rag_file = os.path.join(PLANS_DIR, f"{finding}_rag_context.json")

    with open(plan_file, "r", encoding="utf-8") as f:
        plan = json.load(f)

    rag_text = ""
    if os.path.exists(rag_file):
        with open(rag_file, "r", encoding="utf-8") as f:
            rag_data = json.load(f)
        rag_text = rag_data.get("rag_result", "")

    # Get final indicators
    indicators = [s for s in plan
                  if s.get("output_type", "").strip().lower() == "final indicator"]
    if not indicators:
        print(f"[warn] No final indicators in {finding} plan.")
        return None

    keys = [action_to_key(s.get("action", "")) for s in indicators]
    actions = [s.get("action", "") for s in indicators]

    indicator_desc = "\n".join(
        f'{i+1}. key="{keys[i]}"\n   action: {actions[i]}'
        for i in range(len(indicators))
    )

    system = (
        "You are a clinical decision assistant. Assign clinical importance weights "
        "and diagnostic roles to indicators for chest X-ray diagnosis.\n"
        "Weights must sum to 1.0 and reflect how directly each indicator "
        "contributes to diagnosing the target disease on a single frontal CXR."
    )

    user_text = (
        f"Disease: {finding}\n\n"
        f"Clinical guideline:\n{rag_text[:1500]}\n\n"
        f"Diagnostic indicators:\n{indicator_desc}\n\n"
        "For each indicator, assign:\n"
        "  - weight (float): clinical importance, all weights sum to 1.0\n"
        "  - role: 'definitive' if this is a primary/pathognomonic sign, "
        "'supportive' if it is a secondary or associated sign\n\n"
        "Also propose a decision threshold in [0.20, 0.55].\n"
        "  score = Σ(weight_i × value_i), value_i in [0,1].\n"
        "  Positive if score ≥ threshold.\n\n"
        f"Indicator keys (use EXACTLY): {json.dumps(keys)}\n\n"
        "Respond with ONLY a JSON object:\n"
        "{\n"
        '  "indicators": {\n'
        '    "<indicator_key>": {"weight": <float>, "role": "definitive"|"supportive"},\n'
        "    ...\n"
        "  },\n"
        '  "threshold": <float>\n'
        "}"
    )

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    data = json.loads(raw)

    # Extract structured data
    raw_indicators = data.get("indicators", data)
    threshold = float(data.get("threshold", 0.40))

    weights = {}
    roles = {}
    for k in keys:
        entry = raw_indicators.get(k, {})
        if isinstance(entry, (int, float)):
            weights[k] = float(entry)
            roles[k] = "supportive"
        else:
            weights[k] = float(entry.get("weight", 0.0))
            roles[k] = entry.get("role", "supportive")

    # Normalize weights to sum to 1.0
    total = sum(weights.values())
    if total > 0:
        weights = {k: round(v / total, 4) for k, v in weights.items()}

    # Build output
    output = {
        "threshold": round(threshold, 4),
        "indicators": {}
    }
    for k in keys:
        output["indicators"][k] = {
            "weight": weights.get(k, 0.0),
            "role": roles.get(k, "supportive"),
        }

    # Save
    out_path = os.path.join(PLANS_DIR, f"{finding}_weights.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[done] {finding} weights saved to {out_path} (threshold={threshold:.4f}):")
    for k, v in output["indicators"].items():
        print(f"  {k[:60]}: w={v['weight']:.4f} role={v['role']}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finding", nargs="+", required=True)
    args = parser.parse_args()

    for finding in args.finding:
        try:
            generate_weights(finding)
        except Exception as e:
            print(f"[error] {finding}: {e}")


if __name__ == "__main__":
    main()
