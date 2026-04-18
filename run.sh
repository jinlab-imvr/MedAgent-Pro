#!/usr/bin/env bash
set -e

# ============================================================
# Single-finding, single-run inference pipeline.
#
# Usage:
#   bash run_single.sh --finding Edema --gpu 5
#   bash run_single.sh --finding Edema --gpu 5 --skip-plan --skip-code
# ============================================================

GPU="0"
FINDING="Consolidation"
CONDA_ENV=""
SKIP_PLAN=false
SKIP_CODE=false
SKIP_PRECOMPUTE=false
SKIP_QUALITATIVE=false
SKIP_EVIDENCE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --finding)           FINDING="$2";          shift 2 ;;
        --gpu)               GPU="$2";              shift 2 ;;
        --skip-plan)         SKIP_PLAN=true;        shift ;;
        --skip-code)         SKIP_CODE=true;        shift ;;
        --skip-precompute)   SKIP_PRECOMPUTE=true;  shift ;;
        --skip-qualitative)  SKIP_QUALITATIVE=true; shift ;;
        --skip-evidence)     SKIP_EVIDENCE=true;    shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$FINDING" ]]; then
    echo "Error: --finding is required (e.g. --finding Edema)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PY="conda run --no-capture-output -n $CONDA_ENV python -u"

echo "============================================================"
echo "  Single Run — $(date)"
echo "  Finding: $FINDING   GPU: $GPU"
echo "============================================================"

# ── Stage 1: Plan + rubrics ───────────────────────────────────
if [[ "$SKIP_PLAN" == false ]]; then
    echo ""
    echo ">>> Stage 1: Task-level planning ..."
    $PY Task_level.py --finding "$FINDING"
    echo ">>> Stage 1 done."
else
    echo ">>> Stage 1: Skipped (--skip-plan)"
fi

# ── Stage 2: Code generation ─────────────────────────────────
if [[ "$SKIP_CODE" == false ]]; then
    echo ""
    echo ">>> Stage 2: Code generation ..."
    $PY generate_code.py --finding "$FINDING"
    echo ">>> Stage 2 done."
else
    echo ">>> Stage 2: Skipped (--skip-code)"
fi

# ── Stage 3: Precompute (grounding + segmentation + coding) ──
if [[ "$SKIP_PRECOMPUTE" == false ]]; then
    echo ""
    echo ">>> Stage 3: Precompute tools (GPU=$GPU) ..."
    $PY precompute_tools.py --finding "$FINDING" --gpu "$GPU"
    echo ">>> Stage 3 done."
else
    echo ">>> Stage 3: Skipped (--skip-precompute)"
fi

# ── Stage 4: Qualitative VLM analysis ────────────────────────
if [[ "$SKIP_QUALITATIVE" == false ]]; then
    echo ""
    echo ">>> Stage 4: Qualitative VLM analysis ..."
    $PY qualitative_analysis.py --finding "$FINDING" --skip-existing
    echo ">>> Stage 4 done."
else
    echo ">>> Stage 4: Skipped (--skip-qualitative)"
fi

# ── Stage 5: Evidence evaluation ──────────────────────────────
if [[ "$SKIP_EVIDENCE" == false ]]; then
    echo ""
    echo ">>> Stage 5: Evidence evaluation ..."
    $PY evidence_evaluation.py --finding "$FINDING" --skip-existing
    echo ">>> Stage 5 done."
else
    echo ">>> Stage 5: Skipped (--skip-evidence)"
fi

# ── Stage 6: Weights + decision + evaluate ────────────────────
echo ""
echo ">>> Stage 6: Weight generation + rule decision + evaluate ..."
$PY generate_weights.py --finding "$FINDING"
$PY final_decision_rule.py --finding "$FINDING"
$PY evaluate.py --finding "$FINDING"

echo ""
echo "============================================================"
echo "  Done — $(date)"
echo "============================================================"
