#!/bin/bash
#
# Thesis Benchmark Pipeline Orchestrator
#
# Runs the complete benchmarking pipeline for a given feature matcher:
# 1. Generate depth maps (if not already done)
# 2. Generate matches for all pairs
# 3. Pack into HDF5 format
# 4. Run RePoseD evaluation
#
# Usage: ./run_thesis_benchmark.sh <matcher> [--dry-run] [--skip-depth] [--skip-matches]
#   matcher: dinov3 | dift | ldm | roma | superpoint
#   --dry-run: Process only first 10 pairs
#   --skip-depth: Skip depth generation step
#   --skip-matches: Skip match generation step

set -e

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Dataset paths
ALL_SCENES=("sacre_coeur" "reichstag" "st_peters_square")
SCENE="sacre_coeur"  # Default scene, can be overridden with --scene
IMAGES_DIR="$PROJECT_ROOT/datasets/phototourism/$SCENE/dense/images"
SPARSE_DIR="$PROJECT_ROOT/datasets/phototourism/$SCENE/dense/sparse"
DEPTH_DIR="$PROJECT_ROOT/datasets/phototourism/$SCENE/depth_unidepth"

# Output paths
OUTPUT_BASE="$PROJECT_ROOT/output"
RESULTS_DIR="$OUTPUT_BASE/results"

# Environment names (adjust these to match your conda environments)
ENV_UNIDEPTH="unidepth"
ENV_DINOV3="dinov3"
ENV_DIFT="dift"
ENV_LDM="ldm"
ENV_ROMA="roma"
ENV_SUPERPOINT="superpoint"
ENV_REPOSED="reposed"

# ============================================================================
# Parse Arguments
# ============================================================================

MATCHER=""
DRY_RUN=false
SKIP_DEPTH=false
SKIP_MATCHES=false
LIMIT=""
DEVICE="cuda:0"  # Default to first GPU
ALL_SCENES_MODE=false
CUSTOM_SCENE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        dinov3|dift|ldm|roma|superpoint)
            MATCHER="$1"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            LIMIT="10"
            shift
            ;;
        --skip-depth)
            SKIP_DEPTH=true
            shift
            ;;
        --skip-matches)
            SKIP_MATCHES=true
            shift
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --scene)
            CUSTOM_SCENE="$2"
            shift 2
            ;;
        --all-scenes)
            ALL_SCENES_MODE=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./run_thesis_benchmark.sh <matcher> [options]"
            echo "  --dry-run         Process only first 10 pairs"
            echo "  --skip-depth      Skip depth generation"
            echo "  --skip-matches    Skip match generation"
            echo "  --device <dev>    CUDA device (cuda:0, cuda:1, cuda:2)"
            echo "  --scene <name>    Scene to process (sacre_coeur, reichstag, st_peters_square)"
            echo "  --all-scenes      Process all available scenes"
            exit 1
            ;;
    esac
done

if [ -z "$MATCHER" ]; then
    echo "Error: No matcher specified"
    echo "Usage: ./run_thesis_benchmark.sh <matcher> [options]"
    echo "  matcher: dinov3 | dift | ldm | roma | superpoint"
    echo "  --device cuda:N   Use specific GPU (e.g., cuda:0, cuda:1, cuda:2)"
    echo "  --scene <name>    Scene to process (sacre_coeur, reichstag, st_peters_square)"
    echo "  --all-scenes      Process all available scenes"
    echo "  --dry-run         Process only first 10 pairs"
    exit 1
fi

# Handle --all-scenes: re-invoke this script for each scene
if [ "$ALL_SCENES_MODE" = true ]; then
    BATCH_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    echo "============================================"
    echo "Running pipeline for ALL scenes: ${ALL_SCENES[*]}"
    echo "Batch timestamp: $BATCH_TIMESTAMP"
    echo "============================================"
    
    # Track generated CSV files for aggregation
    GENERATED_CSVS=()
    
    for scene in "${ALL_SCENES[@]}"; do
        args=("$MATCHER" "--scene" "$scene" "--device" "$DEVICE")
        [ "$DRY_RUN" = true ] && args+=("--dry-run")
        [ "$SKIP_DEPTH" = true ] && args+=("--skip-depth")
        [ "$SKIP_MATCHES" = true ] && args+=("--skip-matches")
        [ -n "$LIMIT" ] && args+=("--limit" "$LIMIT")
        
        echo ""
        echo "======== Processing scene: $scene ========"
        "$0" "${args[@]}"
        
        # Find the latest CSV for this scene
        latest_csv=$(ls -t "$RESULTS_DIR/results_${MATCHER}_${scene}_"*.csv 2>/dev/null | head -1)
        if [ -n "$latest_csv" ]; then
            GENERATED_CSVS+=("$latest_csv")
        fi
    done
    
    echo ""
    echo "============================================"
    echo "All scenes completed! Aggregating results..."
    echo "============================================"
    
    # Aggregate results from all scenes
    if [ ${#GENERATED_CSVS[@]} -gt 0 ]; then
        COMBINED_CSV="$RESULTS_DIR/results_${MATCHER}_COMBINED_${BATCH_TIMESTAMP}.csv"
        python3 "$PROJECT_ROOT/scripts/aggregate_results.py" \
            --files "${GENERATED_CSVS[@]}" \
            --output "$COMBINED_CSV" \
            --matcher "$MATCHER"
        
        echo ""
        echo "============================================"
        echo "Combined results saved to: $COMBINED_CSV"
        echo "============================================"
    fi
    
    exit 0
fi

# Handle --scene: update paths for specified scene
if [ -n "$CUSTOM_SCENE" ]; then
    SCENE="$CUSTOM_SCENE"
fi

# Update paths based on SCENE
IMAGES_DIR="$PROJECT_ROOT/datasets/phototourism/$SCENE/dense/images"
SPARSE_DIR="$PROJECT_ROOT/datasets/phototourism/$SCENE/dense/sparse"
DEPTH_DIR="$PROJECT_ROOT/datasets/phototourism/$SCENE/depth_unidepth"

# ============================================================================
# Helper Functions
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

activate_env() {
    local env_name="$1"
    log "Activating environment: $env_name"
    
    # Try to find and source conda
    CONDA_BASE=""
    if [ -n "$CONDA_EXE" ]; then
        CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
    elif [ -d "$HOME/miniconda3" ]; then
        CONDA_BASE="$HOME/miniconda3"
    elif [ -d "$HOME/anaconda3" ]; then
        CONDA_BASE="$HOME/anaconda3"
    elif [ -d "/opt/conda" ]; then
        CONDA_BASE="/opt/conda"
    fi
    
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        # Fallback: try to use conda directly
        eval "$(conda shell.bash hook)"
    fi
    
    conda activate "$env_name"
}

get_matcher_env() {
    case "$1" in
        dinov3) echo "$ENV_DINOV3" ;;
        dift) echo "$ENV_DIFT" ;;
        ldm) echo "$ENV_LDM" ;;
        roma) echo "$ENV_ROMA" ;;
        superpoint) echo "$ENV_SUPERPOINT" ;;
        *) echo "" ;;
    esac
}

# ============================================================================
# Setup
# ============================================================================

log "============================================"
log "Thesis Benchmark Pipeline"
log "============================================"
log "Matcher: $MATCHER"
log "Scene: $SCENE"
log "Device: $DEVICE"
log "Dry-run: $DRY_RUN"
if [ -n "$LIMIT" ]; then
    log "Limit: $LIMIT pairs"
fi
log "============================================"

# Create output directories
MATCHES_DIR="$OUTPUT_BASE/matches/$MATCHER"
mkdir -p "$MATCHES_DIR"
mkdir -p "$RESULTS_DIR"

# ============================================================================
# Step 1: Generate Pairs File
# ============================================================================

PAIRS_FILE="$OUTPUT_BASE/pairs_${SCENE}.txt"

if [ ! -f "$PAIRS_FILE" ]; then
    log "Step 0: Generating pairs file from COLMAP..."
    
    python3 - "$SPARSE_DIR" "$PAIRS_FILE" "$LIMIT" << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, 'external/RePoseD')
from utils.read_write_colmap import read_images_binary
from pathlib import Path

sparse_dir = sys.argv[1]
output_file = sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None

images = read_images_binary(str(Path(sparse_dir) / "images.bin"))

# Build covisibility
point_to_images = {}
for img_id, img in images.items():
    for pt_id in img.point3D_ids:
        if pt_id != -1:
            if pt_id not in point_to_images:
                point_to_images[pt_id] = set()
            point_to_images[pt_id].add(img.name)

# Count common points
pair_counts = {}
for pt_id, img_names in point_to_images.items():
    img_list = list(img_names)
    for i in range(len(img_list)):
        for j in range(i + 1, len(img_list)):
            pair = tuple(sorted([img_list[i], img_list[j]]))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

# Filter and sort pairs
pairs = [(pair, count) for pair, count in pair_counts.items() if count >= 100]
pairs.sort(key=lambda x: -x[1])

if limit:
    pairs = pairs[:limit]

with open(output_file, 'w') as f:
    for pair, _ in pairs:
        f.write(f"{pair[0]} {pair[1]}\n")

print(f"Generated {len(pairs)} pairs")
PYTHON_SCRIPT
    
    log "Pairs file created: $PAIRS_FILE"
else
    log "Step 0: Using existing pairs file: $PAIRS_FILE"
fi

# ============================================================================
# Step 2: Depth Generation
# ============================================================================

if [ "$SKIP_DEPTH" = true ]; then
    log "Step 1: Skipping depth generation (--skip-depth)"
elif [ -d "$DEPTH_DIR" ] && [ "$(ls -A $DEPTH_DIR 2>/dev/null | head -1)" ]; then
    log "Step 1: Depth maps already exist at $DEPTH_DIR, skipping..."
else
    log "Step 1: Generating depth maps with UniDepth..."
    
    activate_env "$ENV_UNIDEPTH"
    
    python3 scripts/generate_depth_maps.py \
        --images_dir "$IMAGES_DIR" \
        --output_dir "$DEPTH_DIR" \
        --backbone vitl14 \
        --device "$DEVICE" \
        --max_size 1024 \
        --skip_existing
    
    log "Step 1: Depth generation complete"
fi

# ============================================================================
# Step 3: Generate Matches
# ============================================================================

if [ "$SKIP_MATCHES" = true ]; then
    log "Step 2: Skipping match generation (--skip-matches)"
else
    log "Step 2: Generating matches with $MATCHER..."
    
    MATCHER_ENV=$(get_matcher_env "$MATCHER")
    activate_env "$MATCHER_ENV"
    
    python3 "scripts/${MATCHER}_matches.py" \
        --pairs_file "$PAIRS_FILE" \
        --images_dir "$IMAGES_DIR" \
        --output_dir "$MATCHES_DIR" \
        --device "$DEVICE" \
        --use_mutual \
        --ratio_thresh 0.8
    
    log "Step 2: Match generation complete"
fi

# ============================================================================
# Step 4: Pack into HDF5
# ============================================================================

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
H5_FILE="$OUTPUT_BASE/benchmark_${MATCHER}_${SCENE}_${TIMESTAMP}.h5"

log "Step 3: Packing data into HDF5..."

# Use reposed environment which has h5py
activate_env "$ENV_REPOSED"

PACK_ARGS="--matches_dir $MATCHES_DIR \
    --depth_dir $DEPTH_DIR \
    --sparse_dir $SPARSE_DIR \
    --pairs_file $PAIRS_FILE \
    --output $H5_FILE"

if [ -n "$LIMIT" ]; then
    PACK_ARGS="$PACK_ARGS --limit $LIMIT"
fi

python3 scripts/pack_benchmark.py $PACK_ARGS

log "Step 3: Packing complete: $H5_FILE"

# ============================================================================
# Step 5: Run RePoseD Evaluation
# ============================================================================

log "Step 4: Running RePoseD evaluation..."

activate_env "$ENV_REPOSED"

cd "$PROJECT_ROOT/external/RePoseD"

RESULTS_JSON="$RESULTS_DIR/results_${MATCHER}_${SCENE}_${TIMESTAMP}.json"
RESULTS_CSV="$RESULTS_DIR/results_${MATCHER}_${SCENE}_${TIMESTAMP}.csv"

# Run evaluation
python3 eval.py "$H5_FILE" -nw 1

# Copy results
if [ -f "results_new/calibrated-benchmark_${MATCHER}_${SCENE}_${TIMESTAMP}.json" ]; then
    cp "results_new/calibrated-benchmark_${MATCHER}_${SCENE}_${TIMESTAMP}.json" "$RESULTS_JSON"
    log "Results saved to: $RESULTS_JSON"
fi

# Convert to CSV
python3 - "$RESULTS_JSON" "$RESULTS_CSV" "$MATCHER" "UniDepth" << 'PYTHON_SCRIPT'
import json
import sys
import csv
import numpy as np

# Read arguments: json_path, csv_path, matcher_name, depth_method
json_path = sys.argv[1]
csv_path = sys.argv[2]
matcher_name = sys.argv[3] if len(sys.argv) > 3 else "unknown"
depth_method = sys.argv[4] if len(sys.argv) > 4 else "UniDepth"

with open(json_path, 'r') as f:
    results = json.load(f)

# Group by experiment
experiments = {}
for r in results:
    if isinstance(r, dict):
        exp = r.get('experiment', 'unknown')
        if exp not in experiments:
            experiments[exp] = {'R_err': [], 't_err': [], 'runtime': [], 'inlier_ratio': []}
        experiments[exp]['R_err'].append(r.get('R_err', float('nan')))
        experiments[exp]['t_err'].append(r.get('t_err', float('nan')))
        info = r.get('info', {})
        experiments[exp]['runtime'].append(info.get('runtime', float('nan')))
        experiments[exp]['inlier_ratio'].append(info.get('inlier_ratio', float('nan')))

# Write CSV matching paper format
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    # Columns: Matches, Depth, Solver, εr(°), εt(°), mAA@5, mAA@10, mAA@20, τ(ms), Inliers, Num_Pairs
    writer.writerow(['Matches', 'Depth', 'Solver', 'εr(°)', 'εt(°)', 'mAA@5', 'mAA@10', 'mAA@20', 'τ(ms)', 'Inliers', 'Num_Pairs'])
    
    for exp, data in sorted(experiments.items()):
        r_err = np.array(data['R_err'])
        t_err = np.array(data['t_err'])
        runtimes = np.array(data['runtime'])
        inliers = np.array(data['inlier_ratio'])
        pose_err = np.maximum(r_err, t_err)
        
        med_r = np.nanmedian(r_err)
        med_t = np.nanmedian(t_err)
        mAA_5 = np.nanmean(pose_err < 5) * 100
        mAA_10 = np.nanmean(pose_err < 10) * 100
        mAA_20 = np.nanmean(pose_err < 20) * 100
        mean_time = np.nanmean(runtimes) * 1000  # Convert to ms
        mean_inliers = np.nanmean(inliers) * 100  # Convert to %
        
        writer.writerow([matcher_name, depth_method, exp, f"{med_r:.2f}", f"{med_t:.2f}", 
                        f"{mAA_5:.1f}", f"{mAA_10:.1f}", f"{mAA_20:.1f}",
                        f"{mean_time:.1f}", f"{mean_inliers:.1f}", len(r_err)])

print(f"CSV saved to: {csv_path}")
PYTHON_SCRIPT

cd "$PROJECT_ROOT"

# ============================================================================
# Summary
# ============================================================================

log "============================================"
log "Pipeline Complete!"
log "============================================"
log "Matcher: $MATCHER"
log "Scene: $SCENE"
log "H5 File: $H5_FILE"
log "Results JSON: $RESULTS_JSON"
log "Results CSV: $RESULTS_CSV"
log "============================================"

# Display results summary
if [ -f "$RESULTS_CSV" ]; then
    log "Results Summary:"
    cat "$RESULTS_CSV" | column -t -s,
fi
