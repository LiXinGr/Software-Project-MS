#!/usr/bin/env bash
#
# Run DINOv3 multi-layer feature study.
#
# This wrapper launches run_thesis_benchmark.sh for multiple transformer blocks
# and then summarizes calibrated metrics from experiments/{run_id}.json.
#
# Note on feat_level:
# - scripts/dinov3_matches.py supports both positive (absolute block index) and
#   negative (index from the end) feat_level values.
# - This wrapper uses the canonical negative mapping to keep config keys aligned
#   with existing runs (e.g., block 23 -> feat_level -1):
#       feat_level = -(24 - block_index)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNNER="$PROJECT_ROOT/run_thesis_benchmark.sh"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments"

DEFAULT_LAYERS="4 8 12 16 20 23"
DEVICE="cuda:0"
MAX_POINTS="2000"
LAYERS_STR="$DEFAULT_LAYERS"
SCENE=""
ALL_SCENES=false
PASSTHROUGH_ARGS=()

print_help() {
    cat << 'EOF'
Usage:
  ./scripts/run_layer_study.sh --scene <scene> [options] [-- extra_args...]
  ./scripts/run_layer_study.sh --all-scenes [options] [-- extra_args...]

Options:
  --scene <name>        Single scene (sacre_coeur, reichstag, st_peters_square)
  --all-scenes          Run all scenes
  --device <dev>        CUDA device (default: cuda:0)
  --layers "<list>"     Space-separated transformer blocks (default: "4 8 12 16 20 23")
  --max_points <N>      Keypoint cap (default: 2000)
  -h, --help            Show this help

Pass-through:
  Any unknown args are forwarded to run_thesis_benchmark.sh.
  Examples: --dry-run, --skip-depth, --limit 10, --ratio_thresh 0.8

Examples:
  ./scripts/run_layer_study.sh --scene sacre_coeur --device cuda:0
  ./scripts/run_layer_study.sh --scene sacre_coeur --layers "4 8 12 16 20 23" --dry-run
  ./scripts/run_layer_study.sh --all-scenes --device cuda:0 --layers "4 8 12 16 20 23"
EOF
}

if [[ ! -x "$RUNNER" ]]; then
    echo "ERROR: Runner not found or not executable: $RUNNER"
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene)
            [[ $# -lt 2 ]] && { echo "ERROR: --scene requires a value"; exit 1; }
            SCENE="$2"
            shift 2
            ;;
        --all-scenes)
            ALL_SCENES=true
            shift
            ;;
        --device)
            [[ $# -lt 2 ]] && { echo "ERROR: --device requires a value"; exit 1; }
            DEVICE="$2"
            shift 2
            ;;
        --layers)
            [[ $# -lt 2 ]] && { echo "ERROR: --layers requires a value"; exit 1; }
            LAYERS_STR="$2"
            shift 2
            ;;
        --max_points|--max-points)
            [[ $# -lt 2 ]] && { echo "ERROR: --max_points requires a value"; exit 1; }
            MAX_POINTS="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                PASSTHROUGH_ARGS+=("$1")
                shift
            done
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -n "$SCENE" && "$ALL_SCENES" == true ]]; then
    echo "ERROR: Use either --scene or --all-scenes, not both."
    exit 1
fi
if [[ -z "$SCENE" && "$ALL_SCENES" == false ]]; then
    echo "ERROR: You must provide --scene or --all-scenes."
    exit 1
fi

# Parse and validate layer list
read -r -a LAYERS <<< "$LAYERS_STR"
if [[ "${#LAYERS[@]}" -eq 0 ]]; then
    echo "ERROR: No layers parsed from --layers \"$LAYERS_STR\""
    exit 1
fi

for layer in "${LAYERS[@]}"; do
    if ! [[ "$layer" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Invalid block index '$layer' (expected integer 0..23)."
        exit 1
    fi
    if (( layer < 0 || layer > 23 )); then
        echo "ERROR: Block index out of range: $layer (expected 0..23)."
        exit 1
    fi
done

for arg in "${PASSTHROUGH_ARGS[@]}"; do
    case "$arg" in
        dinov3|dift|ldm|roma|romav2|superpoint|--run_id|--run-id|--feat_level|--feat-level|--scene|--all-scenes|--device|--max_points|--max-points)
            echo "ERROR: '$arg' is managed by run_layer_study.sh and cannot be passed through."
            exit 1
            ;;
    esac
done

echo "============================================"
echo "DINOv3 Layer Study"
echo "============================================"
if [[ "$ALL_SCENES" == true ]]; then
    echo "Mode:       all scenes"
else
    echo "Scene:      $SCENE"
fi
echo "Device:     $DEVICE"
echo "Layers:     ${LAYERS[*]}"
echo "Max points: $MAX_POINTS"
echo "Runner:     $RUNNER"
echo "Mapping:    feat_level = -(24 - block)"
if [[ "${#PASSTHROUGH_ARGS[@]}" -gt 0 ]]; then
    echo "Extra args: ${PASSTHROUGH_ARGS[*]}"
else
    echo "Extra args: (none)"
fi
echo "============================================"

RUN_IDS=()
FEAT_LEVELS=()

for layer in "${LAYERS[@]}"; do
    feat_level=$((layer - 24))
    run_id="layer_study_b${layer}"

    RUN_IDS+=("$run_id")
    FEAT_LEVELS+=("$feat_level")

    echo ""
    echo ">>> Running block ${layer} (feat_level=${feat_level}), run_id=${run_id}"

    cmd=(
        "$RUNNER" dinov3
        --run_id "$run_id"
        --device "$DEVICE"
        --feat_level "$feat_level"
        --max_points "$MAX_POINTS"
    )
    if [[ "$ALL_SCENES" == true ]]; then
        cmd+=(--all-scenes)
    else
        cmd+=(--scene "$SCENE")
    fi
    if [[ "${#PASSTHROUGH_ARGS[@]}" -gt 0 ]]; then
        cmd+=("${PASSTHROUGH_ARGS[@]}")
    fi

    printf "Command: "
    printf "%q " "${cmd[@]}"
    printf "\n"

    "${cmd[@]}"
done

timestamp="$(date '+%Y%m%d_%H%M%S')"
scene_label="$SCENE"
if [[ "$ALL_SCENES" == true ]]; then
    scene_label="all_scenes"
fi
summary_path="$EXPERIMENTS_DIR/layer_study_summary_${scene_label}_${timestamp}.txt"

layers_csv="$(IFS=,; echo "${LAYERS[*]}")"
feat_levels_csv="$(IFS=,; echo "${FEAT_LEVELS[*]}")"
run_ids_csv="$(IFS=,; echo "${RUN_IDS[*]}")"

python3 - "$EXPERIMENTS_DIR" "$summary_path" "$scene_label" "$ALL_SCENES" "$SCENE" "$layers_csv" "$feat_levels_csv" "$run_ids_csv" "$MAX_POINTS" << 'PYTHON_SUMMARY'
import json
import statistics
import sys
from pathlib import Path

experiments_dir = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
scene_label = sys.argv[3]
all_scenes = sys.argv[4].lower() == "true"
single_scene = sys.argv[5]
layers = [int(x) for x in sys.argv[6].split(",") if x]
feat_levels = [int(x) for x in sys.argv[7].split(",") if x]
run_ids = [x for x in sys.argv[8].split(",") if x]
max_points = sys.argv[9]

rows = []
known_scenes = ["sacre_coeur", "reichstag", "st_peters_square"]
PRIMARY_SOLVER = "3p_ours_shift_scale+12"


def get_calibrated_solver_metrics(scene_data):
    """
    Return calibrated solver metrics as:
        {solver_name: (mAA10, inliers_pct)}
    Falls back to aggregate calibrated metrics under 'aggregate' if per-solver
    data is not available.
    """
    calibrated = scene_data.get("calibrated", {})
    out = {}

    solvers = calibrated.get("solvers", {})
    if isinstance(solvers, dict):
        for solver_name, metrics in solvers.items():
            if not isinstance(metrics, dict):
                continue
            if "mAA10" not in metrics:
                continue
            # New field name is inliers_pct; keep compatibility with older keys.
            inl = metrics.get("inliers_pct", metrics.get("inlier_pct"))
            if inl is None:
                continue
            out[solver_name] = (float(metrics["mAA10"]), float(inl))

    if out:
        return out

    if "mAA10" in calibrated and "inlier_pct" in calibrated:
        out["aggregate"] = (float(calibrated["mAA10"]), float(calibrated["inlier_pct"]))
    return out

for layer, feat_level, run_id in zip(layers, feat_levels, run_ids):
    exp_path = experiments_dir / f"{run_id}.json"
    config_key = f"dinov3_l{feat_level}_sp_mnn_mp{max_points}"
    solver_metrics = {}
    solver_scene_counts = {}

    if exp_path.exists():
        with open(exp_path, "r") as f:
            data = json.load(f)
        config_key = data.get("config_key", config_key)
        scenes = data.get("scenes", {})

        if all_scenes:
            acc = {}
            for s in known_scenes:
                scene_solver_metrics = get_calibrated_solver_metrics(scenes.get(s, {}))
                for solver_name, (m, i) in scene_solver_metrics.items():
                    if solver_name not in acc:
                        acc[solver_name] = {"m": [], "i": []}
                    acc[solver_name]["m"].append(m)
                    acc[solver_name]["i"].append(i)

            for solver_name, vals in acc.items():
                solver_metrics[solver_name] = (
                    statistics.mean(vals["m"]),
                    statistics.mean(vals["i"]),
                )
                solver_scene_counts[solver_name] = len(vals["m"])
        else:
            scene_solver_metrics = get_calibrated_solver_metrics(scenes.get(single_scene, {}))
            solver_metrics.update(scene_solver_metrics)
            for solver_name in scene_solver_metrics:
                solver_scene_counts[solver_name] = 1

    rows.append(
        {
            "block": layer,
            "feat_level": feat_level,
            "run_id": run_id,
            "config_key": config_key,
            "solver_metrics": solver_metrics,
            "solver_scene_counts": solver_scene_counts,
            "json_exists": exp_path.exists(),
        }
    )

# Solver order: primary first, then alphabetical.
all_solver_names = sorted(
    {solver for r in rows for solver in r["solver_metrics"].keys()},
    key=lambda s: (0 if s == PRIMARY_SOLVER else 1, s),
)

if all_scenes:
    title = f"Layer Study Results - {scene_label} (calibrated, all solvers, avg over scenes)"
else:
    title = f"Layer Study Results - {scene_label} (calibrated, all solvers)"

lines = []
lines.append(title)
lines.append("-" * len(title))
lines.append("")

if all_scenes:
    header = f"{'Block':>5}  {'FeatLvl':>7}  {'Config Key':<38}  {'Solver':<38}  {'Avg mAA@10':>10}  {'Avg Inliers%':>12}  {'Scenes':>6}  {'Run ID':<20}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        first = True
        names = all_solver_names if all_solver_names else ["-"]
        for solver_name in names:
            val = r["solver_metrics"].get(solver_name)
            if val is None:
                m_str, i_str = "N/A", "N/A"
                c_str = "0"
            else:
                m_str, i_str = f"{val[0]:.2f}", f"{val[1]:.2f}"
                c_str = str(r["solver_scene_counts"].get(solver_name, 0))
            lines.append(
                f"{(r['block'] if first else ''):>5}  {(r['feat_level'] if first else ''):>7}  {(r['config_key'] if first else ''):<38}  {solver_name:<38}  {m_str:>10}  {i_str:>12}  {c_str:>6}  {(r['run_id'] if first else ''):<20}"
            )
            first = False
else:
    header = f"{'Block':>5}  {'FeatLvl':>7}  {'Config Key':<38}  {'Solver':<38}  {'mAA@10':>8}  {'Inliers%':>9}  {'Run ID':<20}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        first = True
        names = all_solver_names if all_solver_names else ["-"]
        for solver_name in names:
            val = r["solver_metrics"].get(solver_name)
            if val is None:
                m_str, i_str = "N/A", "N/A"
            else:
                m_str, i_str = f"{val[0]:.2f}", f"{val[1]:.2f}"
            lines.append(
                f"{(r['block'] if first else ''):>5}  {(r['feat_level'] if first else ''):>7}  {(r['config_key'] if first else ''):<38}  {solver_name:<38}  {m_str:>8}  {i_str:>9}  {(r['run_id'] if first else ''):<20}"
            )
            first = False

missing = [r["run_id"] for r in rows if not r["json_exists"]]
if missing:
    lines.append("")
    lines.append("Missing experiment logs:")
    for run_id in missing:
        lines.append(f"- {run_id}.json")

output = "\n".join(lines)
print(output)
summary_path.write_text(output + "\n")
print(f"\nSaved summary: {summary_path}")
PYTHON_SUMMARY

echo ""
echo "Layer study complete."
echo "Summary file: $summary_path"
