#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DINOV3_PY="/home.stud/gorbuden/.conda/envs/dinov3/bin/python"
REPOSED_PY="/home.stud/gorbuden/.conda/envs/reposed/bin/python"

ADAPTIVE_KEY="phase4_lg_zeroshot_proj256"
ADAPTIVE_SWEEP_SOURCE_KEY="${ADAPTIVE_KEY}_ft00src"
NOADAPT_KEY="phase4_lg_zeroshot_noadapt"
SCENES=("sacre_coeur" "reichstag" "st_peters_square")
DEFAULT_LIMIT="${PHASE4_LIMIT:-15000}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_phase4}"

mkdir -p "$PROJECT_ROOT/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$PROJECT_ROOT/logs/phase4_followups_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

threshold_tag() {
    printf 'ft%s' "$(printf '%.2f' "$1" | tr -d '.')"
}

run_pack_eval_csv() {
    local config_key="$1"
    local scene="$2"
    local limit="$3"
    local results_root="$4"

    local matches_dir="$PROJECT_ROOT/output/matches/${config_key}/${scene}"
    local benchmark_file="$PROJECT_ROOT/output/benchmarks/${config_key}_${scene}.h5"
    local results_dir="$PROJECT_ROOT/output/results/${results_root}/${scene}"
    local pairs_file="$PROJECT_ROOT/output/pairs_${scene}.txt"
    local depth_dir="$PROJECT_ROOT/datasets/phototourism/${scene}/depth_unidepth"
    local sparse_dir="$PROJECT_ROOT/datasets/phototourism/${scene}/dense/sparse"
    local preprocess_info="$PROJECT_ROOT/datasets/phototourism/${scene}/images_preprocessed/preprocess_info.json"
    local calibrated_json="$results_dir/calibrated-${config_key}_${scene}.json"
    local calibrated_csv="$results_dir/results_${config_key}_${scene}.csv"

    mkdir -p "$results_dir"

    log "=== ${scene} : packing (${config_key}) ==="
    "$REPOSED_PY" "$PROJECT_ROOT/scripts/pack_benchmark.py" \
        --matches_dir "$matches_dir" \
        --depth_dir "$depth_dir" \
        --sparse_dir "$sparse_dir" \
        --pairs_file "$pairs_file" \
        --output "$benchmark_file" \
        --limit "$limit"

    log "=== ${scene} : calibrated eval (${config_key}) ==="
    "$REPOSED_PY" "$PROJECT_ROOT/external/RePoseD/eval.py" \
        "$benchmark_file" \
        -nw 8 \
        --thesis \
        --output_dir "$results_dir" \
        --preprocess_info "$preprocess_info"

    log "=== ${scene} : calibrated CSV (${config_key}) ==="
    "$REPOSED_PY" "$PROJECT_ROOT/scripts/reconstruct_results_csv.py" \
        --input-json "$calibrated_json" \
        --output-csv "$calibrated_csv" \
        --matcher "$config_key" \
        --max-points 2048 \
        --img-size 768 \
        --feat-level -8 \
        --up-ft-index 2 \
        --dift-t 0
}

summarize_sweep_results() {
    local sweep_root="$1"
    "$REPOSED_PY" - <<'PY' "$PROJECT_ROOT" "$sweep_root" "$ADAPTIVE_KEY"
from pathlib import Path
import csv
import numpy as np
import sys

root = Path(sys.argv[1])
sweep_root = Path(sys.argv[2])
adaptive_key = sys.argv[3]
scenes = ["sacre_coeur", "reichstag", "st_peters_square"]
summary_path = root / "output" / "results" / sweep_root / "summary.tsv"

def read_maa(csv_path: Path) -> float:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["Solver"] == "3p_ours_shift_scale+12":
                return float(row["mAA@10"])
    raise RuntimeError(f"Missing solver row in {csv_path}")

def avg_matches(matches_dir: Path) -> float:
    counts = []
    for path in sorted(matches_dir.glob("*.npz")):
        with np.load(path) as data:
            counts.append(len(data["mkpts0"]))
    if not counts:
        return float("nan")
    return float(np.mean(counts))

rows = []
for threshold_dir in sorted((root / "output" / "results" / sweep_root).glob("ft*/")):
    tag = threshold_dir.name
    scene_scores = []
    scene_matches = []
    for scene in scenes:
        csv_path = threshold_dir / scene / f"results_{adaptive_key}_{tag}_{scene}.csv"
        score = read_maa(csv_path)
        scene_scores.append(score)
        scene_matches.append(avg_matches(root / "output" / "matches" / f"{adaptive_key}_{tag}" / scene))
    threshold = tag.replace("ft", "")
    threshold = f"{int(threshold) / 100:.2f}"
    rows.append((threshold, *scene_scores, float(np.mean(scene_scores)), float(np.mean(scene_matches))))

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(["threshold", "sacre", "reichstag", "st_peters", "avg", "avg_matches"])
    for row in rows:
        writer.writerow([
            row[0],
            f"{row[1]:.1f}",
            f"{row[2]:.1f}",
            f"{row[3]:.1f}",
            f"{row[4]:.1f}",
            f"{row[5]:.1f}",
        ])

print("[SWEEP] Filter threshold sweep results (mAA@10):")
print("[SWEEP] threshold | sacre | reichstag | st_peters | avg | avg_matches")
for row in rows:
    print(
        f"[SWEEP] {row[0]:>8} | {row[1]:5.1f} | {row[2]:9.1f} | "
        f"{row[3]:9.1f} | {row[4]:4.1f} | {row[5]:11.1f}"
    )
print(f"[SWEEP] Summary written to {summary_path}")
PY
}

summarize_noadapt_results() {
    local results_root="$1"
    local log_file="$2"
    "$REPOSED_PY" - <<'PY' "$PROJECT_ROOT" "$results_root" "$NOADAPT_KEY" "$log_file"
from pathlib import Path
import csv
import re
import sys

root = Path(sys.argv[1])
results_root = sys.argv[2]
config_key = sys.argv[3]
log_file = Path(sys.argv[4])
scenes = ["sacre_coeur", "reichstag", "st_peters_square"]

baseline = {"sacre_coeur": 82.5, "reichstag": 74.1, "st_peters_square": 73.3}
adaptive = {"sacre_coeur": 83.1, "reichstag": 71.8, "st_peters_square": 71.3}
target = {"sacre_coeur": 86.2, "reichstag": 84.5, "st_peters_square": 75.3}

def read_maa(csv_path: Path) -> float:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["Solver"] == "3p_ours_shift_scale+12":
                return float(row["mAA@10"])
    raise RuntimeError(f"Missing solver row in {csv_path}")

match_stats = {}
pattern = re.compile(
    r"\[PHASE4-NOADAPT\] (?P<scene>\w+(?:_\w+)*) DONE: "
    r"(?P<pairs>\d+) pairs, avg_matches=(?P<avg_matches>[0-9.]+), "
    r"avg_conf=(?P<avg_conf>[0-9.]+), zero_match_pairs=(?P<zero>\d+)"
)
for line in log_file.read_text(encoding="utf-8").splitlines():
    m = pattern.search(line)
    if m:
        match_stats[m.group("scene")] = {
            "pairs": int(m.group("pairs")),
            "avg_matches": float(m.group("avg_matches")),
            "avg_conf": float(m.group("avg_conf")),
            "zero_match_pairs": int(m.group("zero")),
        }

noadapt = {}
for scene in scenes:
    csv_path = root / "output" / "results" / results_root / scene / f"results_{config_key}_{scene}.csv"
    noadapt[scene] = read_maa(csv_path)

summary_md = root / "output" / "results" / results_root / "summary.md"
summary_md.parent.mkdir(parents=True, exist_ok=True)
avg_noadapt = sum(noadapt.values()) / len(scenes)

lines = [
    "| Configuration | sacre | reichstag | st_peters | avg |",
    "|---|---:|---:|---:|---:|",
    f"| Phase 2a MNN baseline | {baseline['sacre_coeur']:.1f} | {baseline['reichstag']:.1f} | {baseline['st_peters_square']:.1f} | {sum(baseline.values()) / 3:.1f} |",
    f"| LG zero-shot (adaptive) | {adaptive['sacre_coeur']:.1f} | {adaptive['reichstag']:.1f} | {adaptive['st_peters_square']:.1f} | {sum(adaptive.values()) / 3:.1f} |",
    f"| LG zero-shot (no-adapt) | {noadapt['sacre_coeur']:.1f} | {noadapt['reichstag']:.1f} | {noadapt['st_peters_square']:.1f} | {avg_noadapt:.1f} |",
    f"| SP+LG target | {target['sacre_coeur']:.1f} | {target['reichstag']:.1f} | {target['st_peters_square']:.1f} | {sum(target.values()) / 3:.1f} |",
    "",
    "| Scene | avg_matches | avg_conf | zero_match_pairs | pairs_logged |",
    "|---|---:|---:|---:|---:|",
]

for scene in scenes:
    stats = match_stats.get(scene)
    if stats is None:
        lines.append(f"| {scene} | n/a | n/a | n/a | n/a |")
    else:
        lines.append(
            f"| {scene} | {stats['avg_matches']:.1f} | {stats['avg_conf']:.3f} | "
            f"{stats['zero_match_pairs']} | {stats['pairs']} |"
        )

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("[PHASE4-NOADAPT] Final calibrated mAA@10:")
print(
    f"[PHASE4-NOADAPT] sacre={noadapt['sacre_coeur']:.1f}, "
    f"reichstag={noadapt['reichstag']:.1f}, "
    f"st_peters={noadapt['st_peters_square']:.1f}, avg={avg_noadapt:.1f}"
)
for scene in scenes:
    stats = match_stats.get(scene)
    if stats is not None:
        print(
            f"[PHASE4-NOADAPT] {scene}: avg_matches={stats['avg_matches']:.1f}, "
            f"avg_conf={stats['avg_conf']:.3f}, zero_match_pairs={stats['zero_match_pairs']}"
        )
print(f"[PHASE4-NOADAPT] Summary written to {summary_md}")
PY
}

cd "$PROJECT_ROOT"

log "Phase 4 follow-up experiments"
log "Project root: $PROJECT_ROOT"
log "Log file: $LOG_FILE"
if [[ -n "${PHASE4_SWEEP_THRESHOLDS:-}" ]]; then
    read -r -a SWEEP_THRESHOLDS <<<"${PHASE4_SWEEP_THRESHOLDS}"
else
    SWEEP_THRESHOLDS=("0.00" "0.02" "0.05" "0.10" "0.15" "0.20")
fi

log "Experiment A: existing ${ADAPTIVE_KEY} files do not contain scores, so a fresh adaptive source run at filter_threshold=0.0 is required."
log "Experiment A: source config key=${ADAPTIVE_SWEEP_SOURCE_KEY}"
log "Experiment A: thresholds=${SWEEP_THRESHOLDS[*]}"
log "Experiment A: this makes the sweep a full matching run, not a cheap post-process."

rm -rf "$PROJECT_ROOT/output/matches/${ADAPTIVE_SWEEP_SOURCE_KEY}"
rm -rf "$PROJECT_ROOT/output/results/${ADAPTIVE_KEY}_sweep"
rm -f "$PROJECT_ROOT/output/benchmarks/${ADAPTIVE_KEY}"_ft*.h5

for SCENE in "${SCENES[@]}"; do
    log "=== ${SCENE} : adaptive source matching (${ADAPTIVE_SWEEP_SOURCE_KEY}) ==="
    "$DINOV3_PY" "$PROJECT_ROOT/scripts/lightglue_projection_matches.py" \
        --scene "$SCENE" \
        --limit "$DEFAULT_LIMIT" \
        --config_key "$ADAPTIVE_SWEEP_SOURCE_KEY" \
        --filter_threshold 0.0 \
        --depth_confidence 0.95 \
        --width_confidence 0.99 \
        --seed 42 \
        --source_cache_max_points 2000 \
        --save_scores
done

for THRESHOLD in "${SWEEP_THRESHOLDS[@]}"; do
    TAG="$(threshold_tag "$THRESHOLD")"
    THRESH_KEY="${ADAPTIVE_KEY}_${TAG}"
    log "Experiment A: threshold=${THRESHOLD} (${THRESH_KEY})"
    rm -rf "$PROJECT_ROOT/output/matches/${THRESH_KEY}"
    rm -f "$PROJECT_ROOT/output/benchmarks/${THRESH_KEY}"_*.h5

    for SCENE in "${SCENES[@]}"; do
        "$REPOSED_PY" "$PROJECT_ROOT/scripts/rethreshold_lightglue_matches.py" \
            --input_dir "$PROJECT_ROOT/output/matches/${ADAPTIVE_SWEEP_SOURCE_KEY}/${SCENE}" \
            --output_dir "$PROJECT_ROOT/output/matches/${THRESH_KEY}/${SCENE}" \
            --threshold "$THRESHOLD" \
            --scene "$SCENE"
        run_pack_eval_csv "$THRESH_KEY" "$SCENE" "$DEFAULT_LIMIT" "${ADAPTIVE_KEY}_sweep/${TAG}"
    done
done

summarize_sweep_results "${ADAPTIVE_KEY}_sweep"

if [[ "${PHASE4_SKIP_NOADAPT:-0}" == "1" ]]; then
    log "Experiment B: PHASE4_SKIP_NOADAPT=1, skipping no-adapt run after setup."
    exit 0
fi

log "Experiment B: clearing previous no-adapt outputs"
rm -rf "$PROJECT_ROOT/output/matches/${NOADAPT_KEY}"
rm -rf "$PROJECT_ROOT/output/results/${NOADAPT_KEY}"
rm -f "$PROJECT_ROOT/output/benchmarks/${NOADAPT_KEY}"_*.h5

for SCENE in "${SCENES[@]}"; do
    log "=== ${SCENE} : matching (${NOADAPT_KEY}) ==="
    "$DINOV3_PY" "$PROJECT_ROOT/scripts/lightglue_projection_matches.py" \
        --scene "$SCENE" \
        --limit "$DEFAULT_LIMIT" \
        --config_key "$NOADAPT_KEY" \
        --filter_threshold 0.1 \
        --depth_confidence -1 \
        --width_confidence -1 \
        --seed 42 \
        --source_cache_max_points 2000

    run_pack_eval_csv "$NOADAPT_KEY" "$SCENE" "$DEFAULT_LIMIT" "$NOADAPT_KEY"
done

summarize_noadapt_results "$NOADAPT_KEY" "$LOG_FILE"
log "All requested follow-up experiments are complete."
