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
RESULTS_DIR=""  # Set dynamically after matcher is known

# Environment names (adjust these to match your conda environments)
ENV_UNIDEPTH="unidepth"
ENV_DINOV3="dinov3"
ENV_DIFT="dift"
ENV_LDM="ldm"
ENV_ROMA="roma"
ENV_ROMAV2="romav2"
ENV_SUPERPOINT="lightglue"
ENV_REPOSED="reposed"

# ============================================================================
# Parse Arguments
# ============================================================================

MATCHER=""
RUN_ID=""
DRY_RUN=false
SKIP_DEPTH=false
SKIP_MATCHES=false
SKIP_PACK=false
FORCE_MATCHES=false
FORCE_BENCHMARK=false
REUSE_MATCHES=""        # Config key of existing matches to reuse
LIMIT=""
MAX_PAIRS="15000"       # Default max pairs limit (override with --limit)
DEVICE="cuda:0"  # Default to first GPU
ALL_SCENES_MODE=false
CUSTOM_SCENE=""

# Hyperparameters (model-specific, tracked in results)
MAX_POINTS="2000"       # Number of keypoints for matching
IMG_SIZE="1120"         # Image size for feature extraction
FEAT_LEVEL="-1"         # DINOv3: ViT block to extract features from (-1 = last)
UP_FT_INDEX="1"         # DIFT: UNet decoder layer (0-3)
DIFT_T="261"            # DIFT: Diffusion timestep
ENSEMBLE_SIZE="8"       # DIFT: Number of noise samples to average (2-8)
ROMA_SETTING="precise"  # RoMaV2: 'precise' (default) or 'fast' mode
RATIO_THRESH=""         # Lowe's ratio test threshold (empty = use default)

while [[ $# -gt 0 ]]; do
    # Debug: uncomment to see argument parsing
    # echo "DEBUG: Processing arg: $1 (next: $2)"
    
    case "$1" in
        dinov3|dift|ldm|roma|romav2|superpoint)
            MATCHER="$1"
            shift
            ;;
        --run_id|--run-id)
            RUN_ID="$2"
            shift 2
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
        --skip-pack)
            SKIP_PACK=true
            shift
            ;;
        --force-matches)
            FORCE_MATCHES=true
            shift
            ;;
        --force-benchmark)
            FORCE_BENCHMARK=true
            shift
            ;;
        --reuse_matches|--reuse-matches)
            REUSE_MATCHES="$2"
            SKIP_MATCHES=true
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --max-pairs)
            MAX_PAIRS="$2"
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
        --max_points|--max-points)
            MAX_POINTS="$2"
            shift 2
            ;;
        --feat_level|--feat-level)
            FEAT_LEVEL="$2"
            shift 2
            ;;
        --up_ft_index|--up-ft-index)
            UP_FT_INDEX="$2"
            shift 2
            ;;
        --dift_t|--dift-t)
            DIFT_T="$2"
            shift 2
            ;;
        --ratio_thresh|--ratio-thresh)
            RATIO_THRESH="$2"
            shift 2
            ;;
        --ensemble_size|--ensemble-size)
            ENSEMBLE_SIZE="$2"
            shift 2
            ;;
        --img_size|--img-size)
            IMG_SIZE="$2"
            shift 2
            ;;
        --roma_setting|--roma-setting)
            ROMA_SETTING="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./run_thesis_benchmark.sh <matcher> --run_id <id> [options]"
            echo "  --run_id <id>       REQUIRED: human label for this run (used for results dir + log)"
            echo "  --dry-run           Process only first 10 pairs"
            echo "  --skip-depth        Skip depth generation"
            echo "  --skip-matches      Skip match generation"
            echo "  --force-matches     Regenerate matches even if they exist"
            echo "  --force-benchmark   Regenerate HDF5 even if it exists"
            echo "  --reuse_matches <k> Use matches from a different config_key (implies --skip-matches)"
            echo "  --limit <N>         Process at most N pairs (default: MAX_PAIRS=$MAX_PAIRS)"
            echo "  --max-pairs <N>     Set the default pair limit (default: 15000)"
            echo "  --device <dev>      CUDA device (cuda:0, cuda:1, cuda:2)"
            echo "  --scene <name>      Scene to process (sacre_coeur, reichstag, st_peters_square)"
            echo "  --all-scenes        Process all available scenes"
            echo "  --max_points <N>    Number of keypoints (default: 2000)"
            echo "  --feat_level <L>    DINOv3 feature level (default: -1)"
            echo "  --up_ft_index <I>   DIFT UNet layer 0-3 (default: 1)"
            echo "  --dift_t <T>        DIFT diffusion timestep (default: 261)"
            echo "  --ensemble_size <N> DIFT noise ensemble size (default: 8)"
            echo "  --ratio_thresh <R>  Lowe's ratio test threshold"
            exit 1
            ;;
    esac
done

if [ -z "$MATCHER" ]; then
    echo "Error: No matcher specified"
    echo "Usage: ./run_thesis_benchmark.sh <matcher> --run_id <id> [options]"
    echo "  matcher: dinov3 | dift | ldm | roma | romav2 | superpoint"
    echo "  --run_id <id>   REQUIRED: human label for this run"
    echo "  --device cuda:N Use specific GPU (e.g., cuda:0, cuda:1, cuda:2)"
    echo "  --scene <name>  Scene to process (sacre_coeur, reichstag, st_peters_square)"
    echo "  --all-scenes    Process all available scenes"
    echo "  --dry-run       Process only first 10 pairs"
    exit 1
fi

if [ -z "$RUN_ID" ]; then
    echo "ERROR: --run_id is required."
    echo "Example: ./run_thesis_benchmark.sh $MATCHER --run_id phase0_bugfix --scene sacre_coeur"
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
        args=("$MATCHER" "--run_id" "$RUN_ID" "--scene" "$scene" "--device" "$DEVICE")
        [ "$DRY_RUN" = true ] && args+=("--dry-run")
        [ "$SKIP_DEPTH" = true ] && args+=("--skip-depth")
        [ "$SKIP_MATCHES" = true ] && args+=("--skip-matches")
        [ "$FORCE_MATCHES" = true ] && args+=("--force-matches")
        [ "$FORCE_BENCHMARK" = true ] && args+=("--force-benchmark")
        [ -n "$REUSE_MATCHES" ] && args+=("--reuse_matches" "$REUSE_MATCHES")
        [ -n "$LIMIT" ] && args+=("--limit" "$LIMIT")

        # Pass hyperparameters to subprocess
        args+=("--max_points" "$MAX_POINTS")
        args+=("--img_size" "$IMG_SIZE")
        args+=("--feat_level" "$FEAT_LEVEL")
        args+=("--up_ft_index" "$UP_FT_INDEX")
        args+=("--dift_t" "$DIFT_T")
        args+=("--ensemble_size" "$ENSEMBLE_SIZE")
        args+=("--roma_setting" "$ROMA_SETTING")
        [ -n "$RATIO_THRESH" ] && args+=("--ratio_thresh" "$RATIO_THRESH")
        
        echo ""
        echo "======== Processing scene: $scene ========"
        
        # Run sub-process but allow failure without killing the loop
        set +e
        "$0" "${args[@]}"
        exit_code=$?
        set -e
        
        if [ $exit_code -ne 0 ]; then
            echo "!!!! Error processing scene $scene (Exit code: $exit_code) !!!!"
            echo "Continuing to next scene..."
        fi
        
        # Find the latest CSV for this scene (new path: output/results/{RUN_ID}/{scene}/)
        scene_results_dir="$OUTPUT_BASE/results/$RUN_ID/$scene"
        latest_csv=$(ls -t "$scene_results_dir/"*.csv 2>/dev/null | head -1)
        if [ -n "$latest_csv" ]; then
            GENERATED_CSVS+=("$latest_csv")
        fi
    done

    echo ""
    echo "============================================"
    echo "All scenes completed! Aggregating results..."
    echo "============================================"

    # Aggregate results from all scenes (use base conda env which has pandas)
    if [ ${#GENERATED_CSVS[@]} -gt 0 ]; then
        COMBINED_CSV="$OUTPUT_BASE/results/$RUN_ID/combined_${BATCH_TIMESTAMP}.csv"
        conda run -n base python3 "$PROJECT_ROOT/scripts/aggregate_results.py" \
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
DATASET_ROOT="$PROJECT_ROOT/datasets/phototourism/$SCENE"
IMAGES_DIR="$DATASET_ROOT/dense/images"
SPARSE_DIR="$DATASET_ROOT/dense/sparse"
DEPTH_DIR="$DATASET_ROOT/depth_unidepth"

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
        romav2) echo "$ENV_ROMAV2" ;;
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
log "Matcher:  $MATCHER"
log "Run ID:   $RUN_ID"
log "Scene:    $SCENE"
log "Device:   $DEVICE"
log "Dry-run:  $DRY_RUN"
if [ -n "$LIMIT" ]; then
    log "Limit: $LIMIT pairs"
else
    log "Limit: $MAX_PAIRS pairs (default)"
fi
log "============================================"

# Apply default limit if no explicit --limit given
if [ -z "$LIMIT" ]; then
    LIMIT="$MAX_PAIRS"
fi

# ============================================================================
# Derive CONFIG_KEY from matcher script (fast, no model loading)
# ============================================================================

get_config_key_for_matcher() {
    # Compute config key in pure bash (mirrors get_config_key() in each Python script).
    # This avoids dependency on conda envs with broken PIL/libjpeg at import time.
    case "$MATCHER" in
        dinov3)
            local rt_suffix=""
            [ -n "$RATIO_THRESH" ] && rt_suffix="_rt${RATIO_THRESH}"
            echo "dinov3_l${FEAT_LEVEL}_sp_mnn${rt_suffix}_mp${MAX_POINTS}"
            ;;
        dift)
            echo "dift_t${DIFT_T}_up${UP_FT_INDEX}_ens${ENSEMBLE_SIZE}_sp_mnn_mp${MAX_POINTS}"
            ;;
        superpoint)
            echo "superpoint_lg_mp${MAX_POINTS}"
            ;;
        roma)
            echo "roma_outdoor_mp${MAX_POINTS}"
            ;;
        romav2)
            echo "romav2_${ROMA_SETTING}_mp${MAX_POINTS}"
            ;;
        *)
            echo "${MATCHER}_mp${MAX_POINTS}"
            ;;
    esac
}

CONFIG_KEY=$(get_config_key_for_matcher)
log "Config key: $CONFIG_KEY"

# ============================================================================
# Output paths (all keyed by CONFIG_KEY or RUN_ID)
# ============================================================================

# Matches: use CONFIG_KEY unless --reuse_matches provides a different key
if [ -n "$REUSE_MATCHES" ]; then
    MATCHES_DIR="$OUTPUT_BASE/matches/${REUSE_MATCHES}/${SCENE}"
    log "Reusing matches from: $MATCHES_DIR"
else
    MATCHES_DIR="$OUTPUT_BASE/matches/${CONFIG_KEY}/${SCENE}"
fi

BENCHMARK_FILE="$OUTPUT_BASE/benchmarks/${CONFIG_KEY}_${SCENE}.h5"
RESULTS_DIR="$OUTPUT_BASE/results/${RUN_ID}/${SCENE}"

mkdir -p "$MATCHES_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$(dirname "$BENCHMARK_FILE")"

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
# Step 1.5: Image Preprocessing (resize + letterbox for fair comparison)
# ============================================================================

PREPROCESSED_DIR="$DATASET_ROOT/images_preprocessed"
PREPROCESS_INFO="$PREPROCESSED_DIR/preprocess_info.json"

if [ -d "$PREPROCESSED_DIR" ] && [ -f "$PREPROCESS_INFO" ]; then
    log "Step 1.5: Preprocessed images already exist at $PREPROCESSED_DIR, skipping..."
else
    log "Step 1.5: Preprocessing images (resize to 1120px + letterbox)..."
    
    # Use dinov3 environment (has PIL with libjpeg)
    activate_env "$ENV_DINOV3"
    
    python3 scripts/preprocess_images.py \
        --images_dir "$IMAGES_DIR" \
        --output_dir "$PREPROCESSED_DIR" \
        --target_size 1120 \
        --divisibility 16 \
        --format jpg \
        --quality 95
    
    log "Step 1.5: Preprocessing complete"
fi

# Use preprocessed images for all subsequent steps
IMAGES_DIR_ORIGINAL="$IMAGES_DIR"
IMAGES_DIR="$PREPROCESSED_DIR"

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

MATCHES_EXIST=false
if compgen -G "$MATCHES_DIR/*.npz" > /dev/null 2>&1; then
    MATCHES_EXIST=true
fi

if [ "$SKIP_MATCHES" = true ]; then
    log "Step 2: Skipping match generation (--skip-matches or --reuse_matches)"
elif [ "$MATCHES_EXIST" = true ] && [ "$FORCE_MATCHES" = false ]; then
    log "Step 2: Matches already exist at $MATCHES_DIR — skipping (use --force-matches to regenerate)"
else
    log "Step 2: Generating matches with $MATCHER..."

    MATCHER_ENV=$(get_matcher_env "$MATCHER")
    activate_env "$MATCHER_ENV"

    # Build matcher arguments
    MATCHER_ARGS="--pairs_file $PAIRS_FILE \
        --images_dir $IMAGES_DIR \
        --output_dir $MATCHES_DIR \
        --device $DEVICE"

    # Add --use_mutual only for matchers that support it (not roma/romav2/ldm which have their own matching)
    if [[ "$MATCHER" != "roma" && "$MATCHER" != "romav2" && "$MATCHER" != "ldm" ]]; then
        MATCHER_ARGS="$MATCHER_ARGS --use_mutual"
    fi

    # Add feature cache keyed by CONFIG_KEY to avoid cross-config contamination
    if [[ "$MATCHER" == "dinov3" || "$MATCHER" == "dift" || "$MATCHER" == "superpoint" ]]; then
        FEATURE_CACHE_DIR="$PROJECT_ROOT/cache/features/${CONFIG_KEY}/${SCENE}"
        mkdir -p "$FEATURE_CACHE_DIR"
        MATCHER_ARGS="$MATCHER_ARGS --feature_cache $FEATURE_CACHE_DIR"
        log "  Feature cache: $FEATURE_CACHE_DIR"
    fi

    # Use SuperPoint keypoints for DINOv3 and DIFT for fair comparison
    if [[ "$MATCHER" == "dinov3" || "$MATCHER" == "dift" ]]; then
        MATCHER_ARGS="$MATCHER_ARGS --use_sp_keypoints"
        log "  Using SuperPoint keypoints for fair comparison"
    fi

    # Add hyperparameters
    MATCHER_ARGS="$MATCHER_ARGS --max_points $MAX_POINTS"
    log "  max_points: $MAX_POINTS"

    # Add img_size for DINOv3 and DIFT (feature extraction resolution)
    if [[ "$MATCHER" == "dinov3" ]]; then
        MATCHER_ARGS="$MATCHER_ARGS --img_size $IMG_SIZE"
        log "  img_size: $IMG_SIZE"
    fi
    if [[ "$MATCHER" == "dift" ]]; then
        MATCHER_ARGS="$MATCHER_ARGS --img_size $IMG_SIZE $IMG_SIZE"
        log "  img_size: ${IMG_SIZE}x${IMG_SIZE}"
    fi

    # DINOv3-specific parameters
    if [[ "$MATCHER" == "dinov3" ]]; then
        MATCHER_ARGS="$MATCHER_ARGS --feat_level $FEAT_LEVEL"
        log "  feat_level: $FEAT_LEVEL"
    fi

    # DIFT-specific parameters
    if [[ "$MATCHER" == "dift" ]]; then
        MATCHER_ARGS="$MATCHER_ARGS --up_ft_index $UP_FT_INDEX --t $DIFT_T --ensemble_size $ENSEMBLE_SIZE"
        log "  up_ft_index: $UP_FT_INDEX, t: $DIFT_T, ensemble_size: $ENSEMBLE_SIZE"
    fi

    # RoMaV2-specific parameters
    if [[ "$MATCHER" == "romav2" ]]; then
        MATCHER_ARGS="$MATCHER_ARGS --setting $ROMA_SETTING"
        log "  setting: $ROMA_SETTING"
    fi

    # Ratio threshold (if specified)
    if [ -n "$RATIO_THRESH" ]; then
        MATCHER_ARGS="$MATCHER_ARGS --ratio_thresh $RATIO_THRESH"
        log "  ratio_thresh: $RATIO_THRESH"
    fi

    # Always pass limit
    MATCHER_ARGS="$MATCHER_ARGS --limit $LIMIT"
    log "  Limiting to $LIMIT pairs"

    # Suppress diffusers warnings for DIFT
    export PYTHONWARNINGS="ignore::UserWarning"

    python3 "scripts/${MATCHER}_matches.py" $MATCHER_ARGS

    log "Step 2: Match generation complete"
fi

# ============================================================================
# Step 4: Pack into HDF5
# ============================================================================

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

if [ "$SKIP_PACK" = true ]; then
    log "Step 3: Skipping packing (--skip-pack)"
elif [ -f "$BENCHMARK_FILE" ] && [ "$FORCE_BENCHMARK" = false ]; then
    log "Step 3: Benchmark already exists: $BENCHMARK_FILE — skipping (use --force-benchmark to regenerate)"
else
    log "Step 3: Packing data into HDF5..."

    # Use reposed environment which has h5py
    activate_env "$ENV_REPOSED"

    PACK_ARGS="--matches_dir $MATCHES_DIR \
        --depth_dir $DEPTH_DIR \
        --sparse_dir $SPARSE_DIR \
        --pairs_file $PAIRS_FILE \
        --output $BENCHMARK_FILE \
        --limit $LIMIT"

    python3 scripts/pack_benchmark.py $PACK_ARGS

    log "Step 3: Packing complete: $BENCHMARK_FILE"
fi

# ============================================================================
# Step 5: Run RePoseD Evaluation
# ============================================================================

log "Step 4: Running RePoseD evaluation..."

activate_env "$ENV_REPOSED"

cd "$PROJECT_ROOT/external/RePoseD"

RESULTS_JSON="$RESULTS_DIR/results_${MATCHER}_${SCENE}_${TIMESTAMP}.json"
RESULTS_CSV="$RESULTS_DIR/results_${MATCHER}_${SCENE}_${TIMESTAMP}.csv"

# Run all 3 experiment types for complete mAA and mAA_f metrics
# 1. Calibrated (known focal lengths) - for mAA pose accuracy
log "  Running calibrated experiments (known focal)..."
python3 eval.py "$BENCHMARK_FILE" -nw 8 --thesis --output_dir "$RESULTS_DIR" --preprocess_info "$PREPROCESS_INFO"

# 2. Shared focal (estimate one shared focal length) - for mAA_f
log "  Running shared focal experiments (for mAA_f)..."
python3 eval_shared_f.py "$BENCHMARK_FILE" -nw 8 --thesis --output_dir "$RESULTS_DIR" --preprocess_info "$PREPROCESS_INFO" || log "  Warning: shared_f eval failed"

# 3. Varying focal (estimate two different focal lengths) - for mAA_f
log "  Running varying focal experiments (for mAA_f)..."
python3 eval_varying_f.py "$BENCHMARK_FILE" -nw 8 --thesis --output_dir "$RESULTS_DIR" --preprocess_info "$PREPROCESS_INFO" || log "  Warning: varying_f eval failed"

# Combine all results into one JSON
python3 - "$RESULTS_DIR" "$MATCHER" "$SCENE" "$RESULTS_JSON" "$CONFIG_KEY" << 'PYTHON_COMBINE'
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
matcher = sys.argv[2]
scene = sys.argv[3]
output_path = sys.argv[4]
config_key = sys.argv[5] if len(sys.argv) > 5 else f"benchmark_{matcher}_{scene}"

# eval.py writes: calibrated-{config_key}_{scene}.json
basename = f"{config_key}_{scene}"
calibrated_path = results_dir / f"calibrated-{basename}.json"
shared_path = results_dir / f"shared_focal-{basename}.json"
varying_path = results_dir / f"varying_focal-{basename}.json"

all_results = []

for path, exp_type in [(calibrated_path, "calibrated"), 
                        (shared_path, "shared_f"), 
                        (varying_path, "varying_f")]:
    if path.exists():
        print(f"Loading {exp_type} results from {path}")
        with open(path, 'r') as f:
            results = json.load(f)
            # Tag each result with experiment type
            for r in results:
                if isinstance(r, dict):
                    r['exp_type'] = exp_type
            all_results.extend(results)
    else:
        print(f"No {exp_type} results found at {path}")

print(f"Combined {len(all_results)} total results")

with open(output_path, 'w') as f:
    json.dump(all_results, f)
print(f"Saved to: {output_path}")
PYTHON_COMBINE

# Convert to CSV with hyperparameters
python3 - "$RESULTS_JSON" "$RESULTS_CSV" "$MATCHER" "UniDepth" "$MAX_POINTS" "$IMG_SIZE" "$FEAT_LEVEL" "$UP_FT_INDEX" "$DIFT_T" "$RATIO_THRESH" << 'PYTHON_SCRIPT'
import json
import sys
import csv
import numpy as np

# Read arguments
json_path = sys.argv[1]
csv_path = sys.argv[2]
matcher_name = sys.argv[3] if len(sys.argv) > 3 else "unknown"
depth_method = sys.argv[4] if len(sys.argv) > 4 else "UniDepth"

# Hyperparameters
max_points = sys.argv[5] if len(sys.argv) > 5 else "2000"
img_size = sys.argv[6] if len(sys.argv) > 6 else "1120"
feat_level = sys.argv[7] if len(sys.argv) > 7 else "-1"
up_ft_index = sys.argv[8] if len(sys.argv) > 8 else "1"
dift_t = sys.argv[9] if len(sys.argv) > 9 else "261"
ratio_thresh = sys.argv[10] if len(sys.argv) > 10 else ""

with open(json_path, 'r') as f:
    results = json.load(f)

# Group by experiment AND exp_type
experiments = {}
for r in results:
    if isinstance(r, dict):
        exp = r.get('experiment', 'unknown')
        exp_type = r.get('exp_type', 'calibrated')
        key = f"{exp}|{exp_type}"
        if key not in experiments:
            experiments[key] = {'R_err': [], 't_err': [], 'runtime': [], 'inlier_ratio': [], 
                               'f_err': [], 'exp': exp, 'exp_type': exp_type}
        experiments[key]['R_err'].append(r.get('R_err', float('nan')))
        experiments[key]['t_err'].append(r.get('t_err', float('nan')))
        # Focal length error (geometric mean of f1_err and f2_err if available)
        if 'f_err' in r:
            experiments[key]['f_err'].append(r.get('f_err', float('nan')))
        info = r.get('info', {})
        experiments[key]['runtime'].append(info.get('runtime', float('nan')))
        experiments[key]['inlier_ratio'].append(info.get('inlier_ratio', float('nan')))

# Check if we have focal length data
has_focal = any(len(data['f_err']) > 0 and not all(np.isnan(data['f_err'])) for data in experiments.values())

# Write CSV matching IMC-PT / RePoseD paper format + hyperparameters
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    # Header with hyperparameter columns
    header = ['Matches', 'Depth', 'Solver', 'Exp.Type', 'Opt.', 'εr(°)', 'εt(°)', 'mAA@10', 'τ(ms)', 'Inliers', 'Num_Pairs', 
              'max_points', 'img_size', 'feat_level', 'up_ft_index', 'dift_t', 'ratio_thresh']
    if has_focal:
        header.insert(8, 'mAA_f@10')
    writer.writerow(header)
    
    for key, data in sorted(experiments.items()):
        exp = data['exp']
        exp_type = data['exp_type']
        r_err = np.array(data['R_err'])
        t_err = np.array(data['t_err'])
        runtimes = np.array(data['runtime'])
        inliers = np.array(data['inlier_ratio'])
        
        # Pose error = max(rotation, translation) per pair
        pose_err = np.maximum(r_err, t_err)
        pose_err[np.isnan(pose_err)] = 180  # Failed poses count as 180°
        
        # Median errors
        med_r = np.nanmedian(r_err)
        med_t = np.nanmedian(t_err)
        
        # mAA@10 (AUC): Average accuracy over thresholds 1° to 10°
        mAA_10 = np.mean([np.sum(pose_err < t) / len(pose_err) for t in range(1, 11)]) * 100
        
        # mAA_f@10: Focal length AUC (thresholds 1% to 10%)
        mAA_f_10 = None
        if has_focal and len(data['f_err']) > 0:
            f_err = np.array(data['f_err'])
            f_err[np.isnan(f_err)] = 1.0  # Failed as 100% error
            mAA_f_10 = np.mean([np.sum(f_err < t/100) / len(f_err) for t in range(1, 11)]) * 100
        
        # Determine optimization type from experiment name
        opt_type = 'H' if 'hybrid' in exp.lower() else 'S'
        
        mean_time = np.nanmean(runtimes)
        mean_inliers = np.nanmean(inliers) * 100
        
        row = [matcher_name, depth_method, exp, exp_type, opt_type, f"{med_r:.2f}", f"{med_t:.2f}", 
               f"{mAA_10:.1f}", f"{mean_time:.1f}", f"{mean_inliers:.1f}", len(r_err),
               max_points, img_size, feat_level, up_ft_index, dift_t, ratio_thresh]
        if has_focal:
            row.insert(8, f"{mAA_f_10:.1f}" if mAA_f_10 is not None else "N/A")
        writer.writerow(row)

print(f"CSV saved to: {csv_path}")
PYTHON_SCRIPT

cd "$PROJECT_ROOT"

# ============================================================================
# Experiment Logging
# ============================================================================

log "Step 5: Logging experiment..."

LOG_ARGS="--run_id $RUN_ID \
    --method $MATCHER \
    --config_key $CONFIG_KEY \
    --scene $SCENE \
    --results_dir $RESULTS_DIR \
    --matches_dir $MATCHES_DIR \
    --benchmark $BENCHMARK_FILE \
    --config feat_level=$FEAT_LEVEL img_size=$IMG_SIZE max_points=$MAX_POINTS \
             up_ft_index=$UP_FT_INDEX dift_t=$DIFT_T ensemble_size=$ENSEMBLE_SIZE \
             roma_setting=$ROMA_SETTING"
[ -n "$RATIO_THRESH" ] && LOG_ARGS="$LOG_ARGS ratio_thresh=$RATIO_THRESH"

conda run -n "$ENV_REPOSED" python3 "$PROJECT_ROOT/experiments/log_experiment.py" \
    $LOG_ARGS || log "  Warning: experiment logging failed (non-fatal)"

# ============================================================================
# Summary
# ============================================================================

log "============================================"
log "Pipeline Complete!"
log "============================================"
log "Matcher:    $MATCHER"
log "Run ID:     $RUN_ID"
log "Config key: $CONFIG_KEY"
log "Scene:      $SCENE"
log "H5 File:    $BENCHMARK_FILE"
log "Results:    $RESULTS_DIR"
log "Exp. log:   experiments/${RUN_ID}.json"
log "============================================"

# Display results summary
if [ -f "$RESULTS_CSV" ]; then
    log "Results Summary:"
    cat "$RESULTS_CSV" | column -t -s,
fi
