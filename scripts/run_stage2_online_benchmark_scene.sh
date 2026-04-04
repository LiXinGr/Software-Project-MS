#!/bin/bash

set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <scene> <gpu_id> <run_id> [limit]"
    echo "  scene: sacre_coeur | reichstag | st_peters_square"
    echo "  gpu_id: physical GPU id, e.g. 1"
    echo "  run_id: unique output key used for matches/results"
    echo "  limit: optional pair limit (default: 15000)"
    exit 1
fi

SCENE="$1"
GPU_ID="$2"
RUN_ID="$3"
LIMIT="${4:-15000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DINOV3_PY="/home.stud/gorbuden/.conda/envs/dinov3/bin/python"
REPOSED_PY="/home.stud/gorbuden/.conda/envs/reposed/bin/python"

MAX_POINTS="2048"
FEAT_LEVEL="-8"
DIFT_T="0"
UP_FT_INDEX="2"
ENSEMBLE_SIZE="1"
IMG_H="768"
IMG_W="768"
ALPHA="0.5"

STAGE2_CKPT="$PROJECT_ROOT/external/glue-factory/outputs/training/stage2_dinov3_lg_v1/checkpoint_best.tar"
PROJECTION_CKPT="$PROJECT_ROOT/experiments/phase2_projection_wide/best.pt"

DATASET_ROOT="$PROJECT_ROOT/datasets/phototourism/$SCENE"
IMAGES_DIR="$DATASET_ROOT/images_preprocessed"
SPARSE_DIR="$DATASET_ROOT/dense/sparse"
DEPTH_DIR="$DATASET_ROOT/depth_unidepth"
PREPROCESS_INFO="$IMAGES_DIR/preprocess_info.json"
PAIRS_FILE="$PROJECT_ROOT/output/pairs_${SCENE}.txt"

CONFIG_KEY="${RUN_ID}_mp${MAX_POINTS}"
MATCHES_DIR="$PROJECT_ROOT/output/matches/$CONFIG_KEY/$SCENE"
BENCHMARK_FILE="$PROJECT_ROOT/output/benchmarks/${CONFIG_KEY}_${SCENE}.h5"
RESULTS_DIR="$PROJECT_ROOT/output/results/$RUN_ID/$SCENE"

mkdir -p "$MATCHES_DIR" "$RESULTS_DIR" "$(dirname "$BENCHMARK_FILE")"

if [ ! -f "$STAGE2_CKPT" ]; then
    echo "Missing Stage 2 checkpoint: $STAGE2_CKPT"
    exit 1
fi
if [ ! -f "$PROJECTION_CKPT" ]; then
    echo "Missing projection checkpoint: $PROJECTION_CKPT"
    exit 1
fi

echo "[$(date '+%F %T')] Stage 2 online benchmark"
echo "  scene:        $SCENE"
echo "  gpu_id:       $GPU_ID"
echo "  run_id:       $RUN_ID"
echo "  config_key:   $CONFIG_KEY"
echo "  limit:        $LIMIT"
echo "  matches_dir:  $MATCHES_DIR"
echo "  results_dir:  $RESULTS_DIR"

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$GPU_ID" "$DINOV3_PY" \
    "$PROJECT_ROOT/scripts/lightglue_projection_matches.py" \
    --online_extraction \
    --scene "$SCENE" \
    --pairs_file "$PAIRS_FILE" \
    --images_dir "$IMAGES_DIR" \
    --output_dir "$MATCHES_DIR" \
    --config_key "$CONFIG_KEY" \
    --checkpoint "$PROJECTION_CKPT" \
    --lightglue_checkpoint "$STAGE2_CKPT" \
    --max_points "$MAX_POINTS" \
    --feat_level "$FEAT_LEVEL" \
    --img_size "$IMG_H" "$IMG_W" \
    --t "$DIFT_T" \
    --up_ft_index "$UP_FT_INDEX" \
    --ensemble_size "$ENSEMBLE_SIZE" \
    --alpha "$ALPHA" \
    --device cuda \
    --limit "$LIMIT"

PYTHONNOUSERSITE=1 "$REPOSED_PY" "$PROJECT_ROOT/scripts/pack_benchmark.py" \
    --matches_dir "$MATCHES_DIR" \
    --depth_dir "$DEPTH_DIR" \
    --sparse_dir "$SPARSE_DIR" \
    --pairs_file "$PAIRS_FILE" \
    --output "$BENCHMARK_FILE" \
    --limit "$LIMIT"

MKL_PKG="$(find "$HOME/.conda/pkgs" -maxdepth 3 -name "libmkl_intel_lp64.so.2" 2>/dev/null | head -1 | xargs -I{} dirname {} 2>/dev/null || true)"
OMP_PKG="$(find "$HOME/.conda/pkgs" -maxdepth 3 -name "libiomp5.so" 2>/dev/null | head -1 | xargs -I{} dirname {} 2>/dev/null || true)"
EXTRA_LIBS=""
[ -n "$MKL_PKG" ] && EXTRA_LIBS="$MKL_PKG"
[ -n "$OMP_PKG" ] && EXTRA_LIBS="${EXTRA_LIBS:+$EXTRA_LIBS:}$OMP_PKG"
if [ -n "$EXTRA_LIBS" ]; then
    export LD_LIBRARY_PATH="${EXTRA_LIBS}:${LD_LIBRARY_PATH:-}"
fi

pushd "$PROJECT_ROOT/external/RePoseD" >/dev/null

PYTHONNOUSERSITE=1 "$REPOSED_PY" eval.py "$BENCHMARK_FILE" -nw 8 --thesis --output_dir "$RESULTS_DIR" --preprocess_info "$PREPROCESS_INFO"
PYTHONNOUSERSITE=1 "$REPOSED_PY" eval_shared_f.py "$BENCHMARK_FILE" -nw 8 --thesis --output_dir "$RESULTS_DIR" --preprocess_info "$PREPROCESS_INFO" || echo "Warning: shared_f eval failed"
PYTHONNOUSERSITE=1 "$REPOSED_PY" eval_varying_f.py "$BENCHMARK_FILE" -nw 8 --thesis --output_dir "$RESULTS_DIR" --preprocess_info "$PREPROCESS_INFO" || echo "Warning: varying_f eval failed"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RESULTS_JSON="$RESULTS_DIR/results_stage2_${SCENE}_${TIMESTAMP}.json"
RESULTS_CSV="$RESULTS_DIR/results_stage2_${SCENE}_${TIMESTAMP}.csv"

PYTHONNOUSERSITE=1 "$REPOSED_PY" - "$RESULTS_DIR" "$SCENE" "$RESULTS_JSON" "$CONFIG_KEY" <<'PYTHON_COMBINE'
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
scene = sys.argv[2]
output_path = Path(sys.argv[3])
config_key = sys.argv[4]

basename = f"{config_key}_{scene}"
paths = [
    (results_dir / f"calibrated-{basename}.json", "calibrated"),
    (results_dir / f"shared_focal-{basename}.json", "shared_f"),
    (results_dir / f"varying_focal-{basename}.json", "varying_f"),
]

all_results = []
for path, exp_type in paths:
    if not path.exists():
        continue
    with open(path, "r") as handle:
        data = json.load(handle)
    for row in data:
        if isinstance(row, dict):
            row["exp_type"] = exp_type
    all_results.extend(data)

with open(output_path, "w") as handle:
    json.dump(all_results, handle)
print(output_path)
PYTHON_COMBINE

PYTHONNOUSERSITE=1 "$REPOSED_PY" - "$RESULTS_JSON" "$RESULTS_CSV" <<'PYTHON_CSV'
import csv
import json
import sys
import numpy as np

json_path = sys.argv[1]
csv_path = sys.argv[2]

with open(json_path, "r") as handle:
    results = json.load(handle)

experiments = {}
for row in results:
    if not isinstance(row, dict):
        continue
    exp = row.get("experiment", "unknown")
    exp_type = row.get("exp_type", "calibrated")
    key = (exp, exp_type)
    experiments.setdefault(
        key,
        {"R_err": [], "t_err": [], "runtime": [], "inlier_ratio": [], "f_err": []},
    )
    experiments[key]["R_err"].append(row.get("R_err", float("nan")))
    experiments[key]["t_err"].append(row.get("t_err", float("nan")))
    if "f_err" in row:
        experiments[key]["f_err"].append(row.get("f_err", float("nan")))
    info = row.get("info", {})
    experiments[key]["runtime"].append(info.get("runtime", float("nan")))
    experiments[key]["inlier_ratio"].append(info.get("inlier_ratio", float("nan")))

has_focal = any(data["f_err"] for data in experiments.values())

with open(csv_path, "w", newline="") as handle:
    writer = csv.writer(handle)
    header = [
        "Solver",
        "Exp.Type",
        "Opt.",
        "eps_r_deg",
        "eps_t_deg",
        "mAA@10",
        "runtime_ms",
        "inliers",
        "num_pairs",
    ]
    if has_focal:
        header.insert(6, "mAA_f@10")
    writer.writerow(header)

    for (exp, exp_type), data in sorted(experiments.items()):
        r_err = np.asarray(data["R_err"], dtype=float)
        t_err = np.asarray(data["t_err"], dtype=float)
        pose_err = np.maximum(r_err, t_err)
        pose_err[np.isnan(pose_err)] = 180.0
        med_r = np.nanmedian(r_err)
        med_t = np.nanmedian(t_err)
        maa10 = np.mean([(pose_err < t).mean() for t in range(1, 11)]) * 100.0
        row = [
            exp,
            exp_type,
            "H" if "hybrid" in exp.lower() else "S",
            f"{med_r:.2f}",
            f"{med_t:.2f}",
            f"{maa10:.1f}",
            f"{np.nanmean(np.asarray(data['runtime'], dtype=float)):.1f}",
            f"{np.nanmean(np.asarray(data['inlier_ratio'], dtype=float)) * 100.0:.1f}",
            len(r_err),
        ]
        if has_focal:
            f_err = np.asarray(data["f_err"], dtype=float)
            if f_err.size == 0:
                maa_f = float("nan")
            else:
                f_err[np.isnan(f_err)] = 1.0
                maa_f = np.mean([(f_err < t / 100.0).mean() for t in range(1, 11)]) * 100.0
            row.insert(6, f"{maa_f:.1f}")
        writer.writerow(row)

print(csv_path)
PYTHON_CSV

popd >/dev/null

echo "[$(date '+%F %T')] Done"
echo "  benchmark_file: $BENCHMARK_FILE"
echo "  results_dir:    $RESULTS_DIR"
