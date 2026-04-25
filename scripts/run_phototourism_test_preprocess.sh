#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/phototourism_test_common.sh"

GPU_SPEC="auto"
DRY_RUN=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --gpus)
            GPU_SPEC="${2:?missing value for --gpus}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

prepare_runtime_env
require_runtime_prereqs

LOG_FILE="$DEFAULT_LOG_DIR/preprocess_$(date +%Y%m%d_%H%M%S).log"
if [ "$DRY_RUN" -eq 0 ]; then
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

generate_pairs_for_scene() {
    local scene="$1"
    local pairs_file
    pairs_file="$(scene_pairs_file "$scene")"
    "$REPOSED_PY" - "$PROJECT_ROOT" "$scene" "$pairs_file" <<'PY'
import sys
from pathlib import Path

project_root = Path(sys.argv[1])
scene = sys.argv[2]
output_file = Path(sys.argv[3])

sys.path.insert(0, str(project_root / "external" / "RePoseD"))
from utils.read_write_colmap import read_images_binary

sparse_dir = project_root / "datasets" / "phototourism" / scene / "dense" / "sparse"
images = read_images_binary(str(sparse_dir / "images.bin"))

point_to_images = {}
for img in images.values():
    for pt_id in img.point3D_ids:
        if pt_id != -1:
            point_to_images.setdefault(pt_id, set()).add(img.name)

pair_counts = {}
for img_names in point_to_images.values():
    img_list = list(img_names)
    for i in range(len(img_list)):
        for j in range(i + 1, len(img_list)):
            pair = tuple(sorted((img_list[i], img_list[j])))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

pairs = [(pair, count) for pair, count in pair_counts.items() if count >= 100]
pairs.sort(key=lambda item: (-item[1], item[0][0], item[0][1]))

output_file.parent.mkdir(parents=True, exist_ok=True)
with output_file.open("w", encoding="utf-8") as handle:
    for pair, _ in pairs:
        handle.write(f"{pair[0]} {pair[1]}\n")

print(f"{scene}: generated {len(pairs)} pairs -> {output_file}")
PY
}

depth_assignments() {
    local gpu_csv="$1"
    "$DINOV3_PY" - "$PROJECT_ROOT" "$gpu_csv" <<'PY'
import sys
from pathlib import Path

project_root = Path(sys.argv[1])
gpus = [gpu for gpu in sys.argv[2].split(",") if gpu]
scenes = [
    "british_museum",
    "florence_cathedral_side",
    "lincoln_memorial_statue",
    "milan_cathedral",
    "mount_rushmore",
    "piazza_san_marco",
    "sagrada_familia",
    "st_pauls_cathedral",
    "taj_mahal",
    "temple_nara_japan",
]

counts = []
for scene in scenes:
    images_dir = project_root / "datasets" / "phototourism" / scene / "dense" / "images"
    count = sum(1 for path in images_dir.iterdir() if path.is_file())
    counts.append((scene, count))

buckets = {gpu: [] for gpu in gpus}
totals = {gpu: 0 for gpu in gpus}
for scene, count in sorted(counts, key=lambda item: (-item[1], item[0])):
    gpu = min(gpus, key=lambda g: (totals[g], int(g)))
    buckets[gpu].append(scene)
    totals[gpu] += count

for gpu in gpus:
    print(f"{gpu}:{' '.join(buckets[gpu])}")
PY
}

depth_worker() {
    local gpu="$1"
    shift
    local scene img_count depth_count depth_pythonpath
    depth_pythonpath="${PYTHONPATH:-}"
    if [ -n "${UNIDEPTH_EXTRA_PYTHONPATH:-}" ]; then
        if [ -n "$depth_pythonpath" ]; then
            depth_pythonpath="${UNIDEPTH_EXTRA_PYTHONPATH}:$depth_pythonpath"
        else
            depth_pythonpath="${UNIDEPTH_EXTRA_PYTHONPATH}"
        fi
    fi
    for scene in "$@"; do
        img_count="$(scene_image_count "$scene")"
        depth_count="$(scene_depth_count "$scene")"
        if [ "$depth_count" -ge "$img_count" ] && [ "$img_count" -gt 0 ]; then
            log "[DEPTH gpu=$gpu] scene=$scene already complete ($depth_count/$img_count), skipping"
            continue
        fi
        log "[DEPTH gpu=$gpu] scene=$scene starting"
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$depth_pythonpath" "$UNIDEPTH_PY" \
            "$PROJECT_ROOT/scripts/generate_depth_maps.py" \
            --images_dir "$(scene_preprocessed_dir "$scene")" \
            --output_dir "$(scene_depth_dir "$scene")" \
            --backbone vitl14 \
            --device cuda \
            --max_size 1024 \
            --skip_existing
        log "[DEPTH gpu=$gpu] scene=$scene done"
    done
}

log "PhotoTourism test preprocessing"
log "Project root: $PROJECT_ROOT"
log "Dry run: $DRY_RUN"
log "GPU spec: $GPU_SPEC"

log "Step 1/3: preprocess images"
for scene in "${TEST_SCENES[@]}"; do
    img_count="$(scene_image_count "$scene")"
    prep_count="$(scene_preprocessed_count "$scene")"
    info_file="$(scene_preprocess_info "$scene")"
    if [ "$prep_count" -ge "$img_count" ] && [ -f "$info_file" ] && [ "$img_count" -gt 0 ]; then
        log "[PREP] scene=$scene already complete ($prep_count/$img_count), skipping"
        continue
    fi
    cmd=(
        "$DINOV3_PY" "$PROJECT_ROOT/scripts/preprocess_images.py"
        --images_dir "$(scene_images_dir "$scene")"
        --output_dir "$(scene_preprocessed_dir "$scene")"
        --target_size 1120
        --divisibility 16
        --format jpg
        --quality 95
    )
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '[DRY-RUN][PREP]'
        printf ' %q' "${cmd[@]}"
        printf '\n'
    else
        log "[PREP] scene=$scene starting"
        "${cmd[@]}"
        log "[PREP] scene=$scene done"
    fi
done

log "Step 2/3: generate covisibility pairs"
for scene in "${TEST_SCENES[@]}"; do
    pair_count="$(scene_pair_count "$scene")"
    if [ "$pair_count" -gt 0 ]; then
        log "[PAIRS] scene=$scene already present ($pair_count pairs), skipping"
        continue
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        log "[DRY-RUN][PAIRS] scene=$scene -> $(scene_pairs_file "$scene")"
    else
        log "[PAIRS] scene=$scene starting"
        generate_pairs_for_scene "$scene"
        log "[PAIRS] scene=$scene done"
    fi
done

log "Step 3/3: generate UniDepth maps"
mapfile -t GPU_IDS < <(resolve_gpu_ids "$GPU_SPEC")
if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    echo "No GPUs available for UniDepth preprocessing." >&2
    exit 1
fi
log "Depth GPUs: ${GPU_IDS[*]}"

if [ "$DRY_RUN" -eq 1 ]; then
    ASSIGNMENTS="$(depth_assignments "$(IFS=,; echo "${GPU_IDS[*]}")")"
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        log "[DRY-RUN][DEPTH] $line"
    done <<<"$ASSIGNMENTS"
else
    declare -a DEPTH_PIDS=()
    declare -a DEPTH_TAGS=()
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        gpu="${line%%:*}"
        scenes_str="${line#*:}"
        [ -n "$scenes_str" ] || continue
        read -r -a scenes <<<"$scenes_str"
        worker_log="$DEFAULT_LOG_DIR/preprocess_depth_gpu${gpu}.log"
        (
            depth_worker "$gpu" "${scenes[@]}"
        ) > >(tee -a "$worker_log") 2>&1 &
        DEPTH_PIDS+=("$!")
        DEPTH_TAGS+=("gpu${gpu}")
    done < <(depth_assignments "$(IFS=,; echo "${GPU_IDS[*]}")")

    worker_fail=0
    for idx in "${!DEPTH_PIDS[@]}"; do
        if ! wait "${DEPTH_PIDS[$idx]}"; then
            log "[DEPTH] worker ${DEPTH_TAGS[$idx]} failed"
            worker_fail=1
        fi
    done
    [ "$worker_fail" -eq 0 ] || exit 1
fi

log "Verification table"
STATUS_FILE="$DEFAULT_LOG_DIR/preprocess_status_$(date +%Y%m%d_%H%M%S).txt"
if [ "$DRY_RUN" -eq 1 ]; then
    print_readiness_table | tee "$STATUS_FILE" || true
    log "Dry-run complete. No preprocessing jobs were started."
    log "Dry-run status snapshot: $STATUS_FILE"
elif print_readiness_table | tee "$STATUS_FILE"; then
    log "All 10 test scenes are ready for evaluation."
else
    log "At least one scene is still incomplete. See $STATUS_FILE"
    exit 1
fi

if [ "$DRY_RUN" -eq 0 ]; then
    log "Preprocessing complete."
    log "Master log: $LOG_FILE"
fi
