import os
import sys
import json
import re
import argparse

import openai

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from key import OPENAI_API_KEY, MODEL_PLANNING, SRC_ROOT

openai.api_key = OPENAI_API_KEY

from RAG import RAG_Module
from Planner import Planner

MODEL = MODEL_PLANNING

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_FILE = os.path.join(SCRIPT_DIR, "tasks.json")
TOOLSET_FILE = os.path.join(SCRIPT_DIR, "toolset.json")
PLANS_DIR = os.path.join(SCRIPT_DIR, "plans")
PHRASE_CANDIDATES_FILE = os.path.join(SCRIPT_DIR, "maira_phrase_candidates.json")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_phrase_list_text():
    """Load maira_phrase_candidates.json and format as a readable phrase list for the prompt."""
    candidates = load_json(PHRASE_CANDIDATES_FILE)
    tiers = candidates.get("tiers", {})
    lines = []

    t1 = tiers.get("tier_1_most_stable", {})
    findings = t1.get("mimic_ms_cxr_8_findings", [])
    devices = t1.get("support_devices_and_related_phrases", [])
    lines.append(f"  * Findings (Tier 1, most reliable): {', '.join(findings)}")
    lines.append(f"  * Support devices: {', '.join(devices)}")

    t2 = tiers.get("tier_2_explicit_examples", {})
    t2_maira = t2.get("maira2_explicit_examples", {})
    anatomy = t2_maira.get("anatomy_or_structure", [])
    report_phrases = t2_maira.get("findings_or_report_phrases", [])
    lines.append(f"  * Anatomy/structure (Tier 2): {', '.join(anatomy)}")
    lines.append(f"  * Lateralised/report findings (Tier 2): {', '.join(report_phrases)}")

    t3 = tiers.get("tier_3_conservative_recommendations", {})
    avoid = t3.get("less_preferred_generic_phrases", [])
    lines.append(f"  * Less preferred (use only if no better alternative): {', '.join(avoid)}")

    return "\n".join(lines)


def _get_tier1_finding(finding):
    """If the finding matches a tier-1 phrase, return it (lowercase); else None."""
    candidates = load_json(PHRASE_CANDIDATES_FILE)
    tier1 = candidates.get("tiers", {}).get("tier_1_most_stable", {})
    tier1_findings = tier1.get("mimic_ms_cxr_8_findings", [])
    finding_lower = finding.strip().lower()
    for t in tier1_findings:
        if t == finding_lower:
            return t
    return None


def generate_plan_for_finding(finding: str, task_info: dict, toolset: list,
                              rag: RAG_Module, planner: Planner):
    """
    Disease-level planning for a single MIMIC finding.

    1. RAG retrieval → clinical guideline
    2. Build planner prompt (aligned with paper Section 3.2)
    3. Generate structured plan → plans/<finding>_plan.json
    """
    input_desc = task_info.get("input", "").strip()
    disease_goal = task_info.get("disease", "").strip()

    print(f"\n[Task_level] RAG retrieval for: {finding}")
    rag_result = rag.query(finding=finding)
    print(f"[Task_level] RAG result length: {len(rag_result)} chars")

    os.makedirs(PLANS_DIR, exist_ok=True)
    rag_file = os.path.join(PLANS_DIR, f"{finding}_rag_context.json")
    with open(rag_file, "w", encoding="utf-8") as f:
        json.dump({"finding": finding, "rag_result": rag_result}, f,
                  indent=2, ensure_ascii=False)

    phrase_list_text = _build_phrase_list_text()
    tier1_phrase = _get_tier1_finding(finding)

    # Tier-1 mandatory grounding rule
    tier1_rule = ""
    if tier1_phrase:
        tier1_rule = (
            f"MANDATORY: Since '{tier1_phrase}' is a Tier-1 finding that the grounding "
            f"model can directly localise, the plan MUST include a grounding step (tool 2) "
            f"with phrase='{tier1_phrase}' to directly detect its presence.\n"
        )

    planner_prompt = (
        "Plan a step-by-step, executable diagnostic workflow for a single frontal "
        "chest X-ray image using ONLY the available tools.\n"
        "IMPORTANT: Only a frontal (PA/AP) chest X-ray is available. "
        "Do NOT assume lateral views exist.\n"
        f"Input: {input_desc}\n"
        f"Goal: {disease_goal}\n\n"
        "Output format (STRICT): A JSON array of step objects with fields:\n"
        "  [id, tool, action_type, action, input_type, output_type, output_path]\n"
        "  PLUS additional fields depending on tool type (see below).\n\n"
        "Field rules:\n"
        "- id: starts from 1 and increases by 1\n"
        "- tool: ARRAY of integers (tool ids from toolset)\n"
        "- action_type: STRING — 'qualitative' ONLY for VLM steps (tool 1), "
        "'quantitative' for ALL other tools (grounding/segmentation/coding)\n"
        "- action: STRING describing what the step does\n"
        "- input_type: ARRAY of integers; 0 = raw input image, "
        "or a prior step's id if using that step's output\n"
        "- output_type: MUST be exactly 'intermediate result' or 'final indicator'\n"
        "- output_path: naming convention (STRICT):\n"
        "  * Grounding steps (tool 2): '{phrase}_bbox.png' where {phrase} is the "
        "grounding phrase with spaces replaced by underscores\n"
        "  * Segmentation steps (tool 3): '{phrase}_mask.png' matching the phrase "
        "of the grounding step it depends on\n"
        "  * Coding steps (tool 4): 'diagnosis.json'\n"
        "  * VLM steps (tool 1): 'diagnosis.json'\n\n"
        "Tool-specific REQUIRED fields:\n"
        "- Grounding steps (tool 2): add a 'phrase' field — a CONCISE noun phrase "
        "for the grounding model to localise. Prefer phrases from the list below "
        "(the grounding model performs best with them), but you may use other "
        "concise radiology noun phrases when none of the listed phrases fit the "
        "intended target:\n"
        f"{phrase_list_text}\n\n"
        f"{tier1_rule}"
        "- VLM steps (tool 1): add a 'prompt' field — a detailed diagnostic "
        "question with clinical background/context so the VLM can make a "
        "well-informed judgement.\n"
        "- IMPORTANT — HOLISTIC FRAMING FOR VLM ACTIONS/PROMPTS:\n"
        "  For VLM steps that depend on grounding results, phrase the action "
        "as a HOLISTIC diagnostic question about the PATIENT'S condition "
        "based on the overall chest X-ray, using the grounding as a visual "
        "REFERENCE/HINT — NOT as a narrow question limited to the bbox region.\n"
        "  BAD: 'Based on the [disease] bounding box, is [disease] present in this region?'\n"
        "  GOOD: 'Using the [disease] detection as a visual reference, does "
        "this patient have [disease] based on the overall chest X-ray?'\n"
        "  The VLM receives BOTH the original CXR and the grounding overlay — "
        "use the grounding as an attention anchor, NOT the sole scope.\n\n"
        "Workflow design rules:\n"
        "- Grounding (tool 2) can localise both DISEASE regions and ANATOMICAL "
        "structures (see phrase list above). Use DISEASE phrases for direct visual "
        "assessment by VLM; use ANATOMICAL phrases for quantitative measurements. "
        "Both types may coexist for the same region if they serve different purposes.\n"
        "- Segmentation (tool 3) can ONLY segment organs/anatomical structures "
        "(NOT lesions). It MUST receive a bounding box from grounding (tool 2) "
        "as input — always run grounding first, then segmentation. "
        "IMPORTANT: Segmentation masks are APPROXIMATE and for VISUALISATION only "
        "— they hint at region boundaries so the VLM can see the ROI overlay. "
        "For any QUANTITATIVE measurement (size, ratio, area), ALWAYS use the "
        "grounding bbox coordinates directly via coding (tool 4), NOT segmentation. "
        "Only add a segmentation step when the VLM genuinely benefits from seeing "
        "an organ boundary overlay. Do NOT add segmentation by default for every "
        "grounding step.\n"

        "- IMPORTANT: Do NOT use grounding for 'thoracic cavity' — the grounding "
        "model cannot reliably detect it.\n"
        "- Use coding module (tool 4) for quantitative indicator computation "
        "from grounding bbox coordinates.\n"
        "- Use VLM (tool 1) for qualitative visual assessment of each indicator. "
        "Each VLM step's action MUST be phrased as a diagnostic question that "
        "can be answered with Yes/No.\n"
        "- Each qualitative indicator must be a SEPARATE step.\n"
        "- INDICATOR COUNT: The plan should have 3–7 final indicators "
        "(steps with output_type='final indicator'), depending on the clinical "
        "complexity of the target finding. Simple findings may need only 3-4; "
        "complex findings with multiple radiographic sub-signs may need up to 7. "
        "Focus on clinically discriminative and reliably assessable indicators "
        "for the target disease on a SINGLE frontal CXR. "
        "Quality over quantity — fewer strong indicators outperform many weak ones.\n"
        "- INDICATOR SELECTION GUIDANCE:\n"
        "  AVOID these types of indicators:\n"
        "  (a) Indicators whose conclusion is a DIFFERENT diagnosis — "
        "do NOT borrow signs that primarily diagnose another disease.\n"
        "  (b) Indicators that measure another disease's primary feature. "
        "E.g., do NOT include an indicator whose primary purpose is diagnosing "
        "a different finding — each finding's plan should focus on its OWN "
        "characteristic signs.\n"
        "  WELCOME these types of indicators:\n"
        "  (a) Primary/pathognomonic signs of the target disease\n"
        "  (b) Well-established secondary findings correlated with the target\n"
        "  (c) Different presentation patterns of the target disease\n"
        "- Quantitative steps must be followed by a qualitative VLM judgement step.\n"
        "- Steps must follow strict logical order; no forward references.\n"
        "- Do NOT ground the SAME target with different near-synonym phrases — pick ONE. "
        "However, disease phrases and anatomical phrases for the same region "
        "target DIFFERENT things and may coexist.\n"
        "- Every segmentation step (tool 3) MUST have its output consumed by at "
        "least one downstream step (as input_type). Do NOT add segmentation steps "
        "whose masks are never used.\n"
        "- Do NOT: (a) subdivide severity into separate Yes/No steps — use ONE "
        "step with a graded scale; (b) create a 'summary' or 'meta-aggregation' "
        "step synthesising all indicators — that is handled downstream; "
        "(c) add differential-diagnosis or etiology steps (e.g., 'cardiogenic "
        "vs non-cardiogenic?'). Focus on indicators that DIRECTLY support or "
        "refute the target finding.\n"
        "- For findings with distinct radiographic sub-signs, create SEPARATE "
        "VLM steps for each sub-sign — these are independent visual features, "
        "not bundled indicators.\n"
        "- Do NOT use coding module (tool 4) for computing lateralisation, "
        "bilateral distribution, or multilobar extent from a SINGLE bounding box — "
        "a single bbox cannot encode spatial distribution. Use VLM visual assessment "
        "for such spatial judgments instead.\n\n"
        "Return ONLY the JSON array."
    )

    print(f"[Task_level] Generating plan for: {finding}")
    plan = planner.plan(
        output_path=PLANS_DIR,
        prompt=planner_prompt,
        rag_text=rag_result,
        filename=f"{finding}_plan.json",
        model=MODEL,
        toolset=toolset,
        max_retries=3,
    )

    print(f"[Task_level] Plan saved: {finding}_plan.json ({len(plan)} steps)")

    rubrics = generate_rubrics(finding, plan, rag_result)
    rubrics_file = os.path.join(PLANS_DIR, f"{finding}_rubrics.json")
    with open(rubrics_file, "w", encoding="utf-8") as f:
        json.dump(rubrics, f, indent=2, ensure_ascii=False)
    print(f"[Task_level] Rubrics saved: {finding}_rubrics.json ({len(rubrics)} indicators)")

    return plan


def _action_to_key(action: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", " ", (action or "").strip()).strip().lower()
    return "_".join(s.split()) or "unknown"


def generate_rubrics(finding: str, plan: list, rag_text: str) -> dict:
    """Generate per-indicator diagnostic rubrics using RAG context + plan structure.

    Returns dict mapping action_key -> rubric text string.
    """
    vlm_steps = [s for s in plan
                 if s.get("output_type", "").strip().lower() == "final indicator"
                 and 1 in ([s.get("tool")] if not isinstance(s.get("tool"), list)
                           else [int(t) for t in s.get("tool", [])])]
    if not vlm_steps:
        return {}

    plan_by_id = {int(s["id"]): s for s in plan}

    def _has_coding_ancestor(step):
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
            tids = parent.get("tool", [])
            if not isinstance(tids, list):
                tids = [tids]
            if 4 in [int(t) for t in tids]:
                return True
            queue.extend(parent.get("input_type", []) or [])
        return False

    indicator_descs = []
    for step in vlm_steps:
        key = _action_to_key(step.get("action", ""))
        action = step.get("action", "")
        is_quantitative = _has_coding_ancestor(step)
        step_type = "quantitative (with computed measurement)" if is_quantitative else "qualitative (visual assessment only)"
        indicator_descs.append(f'- key="{key}", action="{action}", type={step_type}')

    indicators_text = "\n".join(indicator_descs)

    system = (
        "You are a radiology expert creating diagnostic scoring rubrics for "
        "chest X-ray analysis. Generate concise, actionable rubrics that will "
        "help a VLM (vision-language model) score each indicator on a 1-5 scale."
    )

    user = (
        f"Disease: {finding}\n\n"
        f"Clinical guideline from RAG retrieval:\n{rag_text}\n\n"
        f"The diagnostic plan has these final indicators:\n{indicators_text}\n\n"
        "For EACH indicator, generate a diagnostic rubric with:\n"
        "1. KEY POSITIVE findings (what to look for to score 4-5)\n"
        "2. KEY NEGATIVE findings (what indicates score 1-2)\n"
        "3. One main CONFOUNDER to watch for\n"
        "4. For quantitative indicators: how to verify if the computed "
        "measurement is reliable (e.g., check bbox accuracy)\n\n"
        "IMPORTANT RULES:\n"
        "- Every rubric must start with: 'NOTE ON BOUNDING BOX: The highlighted "
        "region is only a region-of-interest from an automated detector. It does "
        "NOT confirm the finding.'\n"
        "- Keep each rubric under 200 words\n"
        "- Be specific to CXR imaging (not clinical symptoms)\n"
        "- For quantitative indicators, add a measurement plausibility check\n\n"
        "Respond with ONLY a JSON object mapping each indicator key to its "
        "rubric text string:\n"
        "{\n"
        '  "<indicator_key>": "<rubric text>",\n'
        "  ...\n"
        "}"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    completion = openai.ChatCompletion.create(model=MODEL, messages=messages)
    raw = completion.choices[0].message.content.strip()

    # Parse
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    try:
        rubrics = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            rubrics = json.loads(m.group())
        else:
            print(f"[WARN] Failed to parse rubrics for {finding}")
            rubrics = {}

    # Ensure all indicator keys are covered
    for step in vlm_steps:
        key = _action_to_key(step.get("action", ""))
        if key not in rubrics:
            rubrics[key] = ""

    return rubrics


def main():
    parser = argparse.ArgumentParser(description="Disease-level planning for MIMIC findings")
    parser.add_argument("--finding", type=str, default=None,
                        help="Generate plan for a single finding (e.g. 'Atelectasis')")
    parser.add_argument("--all", action="store_true", default=True,
                        help="Generate plans for all 12 findings (default)")
    args = parser.parse_args()

    # Load configs
    tasks = load_json(TASKS_FILE)
    toolset = load_json(TOOLSET_FILE)

    # Init modules
    rag = RAG_Module(openai_api_key=OPENAI_API_KEY, model=MODEL)
    planner = Planner(api_key=OPENAI_API_KEY)

    if args.finding:
        findings = [args.finding]
    else:
        findings = list(tasks.keys())

    results = {}
    for finding in findings:
        if finding not in tasks:
            print(f"[WARN] Finding '{finding}' not in tasks.json, skipping")
            continue
        try:
            plan = generate_plan_for_finding(
                finding=finding,
                task_info=tasks[finding],
                toolset=toolset,
                rag=rag,
                planner=planner,
            )
            results[finding] = {"status": "ok", "steps": len(plan)}
        except Exception as e:
            print(f"[ERROR] Failed for {finding}: {e}")
            results[finding] = {"status": "error", "error": str(e)}

    # Print summary
    print(f"\n{'='*60}")
    print("Planning Summary")
    print(f"{'='*60}")
    for finding, info in results.items():
        status = info["status"]
        detail = f"{info['steps']} steps" if status == "ok" else info["error"]
        print(f"  {finding:30s} {status:5s}  {detail}")


if __name__ == "__main__":
    main()