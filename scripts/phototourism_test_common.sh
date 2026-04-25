#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TEST_SCENES=(
    british_museum
    florence_cathedral_side
    lincoln_memorial_statue
    milan_cathedral
    mount_rushmore
    piazza_san_marco
    sagrada_familia
    st_pauls_cathedral
    taj_mahal
    temple_nara_japan
)

METHOD_ORDER=(
    test_splg
    test_dinov3_mnn
    test_dift_mnn
    test_ours_151sc_ft010
    test_roma
    test_romav2
)

declare -Ag METHOD_LABELS=(
    [test_dinov3_mnn]="DINOv3 MNN"
    [test_dift_mnn]="DIFT MNN"
    [test_ours_151sc_ft010]="Ours (frozen 151sc)"
    [test_splg]="SP+LG"
    [test_roma]="RoMa"
    [test_romav2]="RoMaV2"
)

declare -Ag VALIDATION_MAA10=(
    [test_dinov3_mnn]="63.6"
    [test_dift_mnn]="66.5"
    [test_ours_151sc_ft010]="81.4"
    [test_splg]="81.7"
    [test_roma]="87.1"
    [test_romav2]="86.4"
)

THESIS_PAIR_LIMIT="${THESIS_PAIR_LIMIT:-15000}"
REPOSED_NUM_WORKERS="${REPOSED_NUM_WORKERS:-8}"

DINOV3_PY="/home.stud/gorbuden/.conda/envs/dinov3/bin/python"
DIFT_PY="/home.stud/gorbuden/.conda/envs/dift/bin/python"
LIGHTGLUE_PY="/home.stud/gorbuden/.conda/envs/lightglue/bin/python"
ROMA_PY="/home.stud/gorbuden/.conda/envs/roma/bin/python"
ROMAV2_PY="/home.stud/gorbuden/.conda/envs/romav2/bin/python"
REPOSED_PY="/home.stud/gorbuden/.conda/envs/reposed/bin/python"
UNIDEPTH_PY="/home.stud/gorbuden/.conda/envs/unidepth/bin/python"

OURS151_LIGHTGLUE_CKPT="$PROJECT_ROOT/external/glue-factory/outputs/training/stage2_dinov3_lg_151scenes_v1/checkpoint_best.tar"
PROJECTION_CKPT="$PROJECT_ROOT/experiments/phase2_projection_wide/best.pt"

DEFAULT_LOG_DIR="$PROJECT_ROOT/logs/phototourism_test"
DEFAULT_RESULTS_ROOT="$PROJECT_ROOT/output/results"
UNIDEPTH_EXTRA_PYTHONPATH="${UNIDEPTH_EXTRA_PYTHONPATH:-}"
ROMAV2_EXTRA_PYTHONPATH="${ROMAV2_EXTRA_PYTHONPATH:-}"
ROMAV2_CUDA_SHIM_DIR="${ROMAV2_CUDA_SHIM_DIR:-}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

ensure_module_cmd() {
    if type module >/dev/null 2>&1; then
        return 0
    fi
    if [ -f /etc/profile.d/lmod.sh ]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/lmod.sh
    elif [ -f /usr/share/lmod/lmod/init/bash ]; then
        # shellcheck disable=SC1091
        source /usr/share/lmod/lmod/init/bash
    fi
    type module >/dev/null 2>&1
}

load_anaconda_module() {
    local module_name
    for module_name in Anaconda3/2020.07 Anaconda3/2022.10 Anaconda3/2024.02-1; do
        if module load "$module_name" >/dev/null 2>&1; then
            return 0
        fi
    done
    return 1
}

prepare_runtime_env() {
    mkdir -p "$DEFAULT_LOG_DIR" "$DEFAULT_RESULTS_ROOT" /tmp/mpl_phototourism_test

    if ensure_module_cmd; then
        load_anaconda_module || true
    fi

    if command -v conda >/dev/null 2>&1; then
        set +u
        source "$(conda info --base)/etc/profile.d/conda.sh" >/dev/null 2>&1 || true
        set -u
    fi

    export PYTHONNOUSERSITE=1
    export HF_HOME="${HF_HOME:-/home.stud/gorbuden/.cache/huggingface}"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_phototourism_test}"

    resolve_usersite_for_python() {
        local python_bin="$1"
        "$python_bin" - <<'PY' 2>/dev/null || true
import sys
from pathlib import Path
print(Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")
PY
    }

    resolve_libcuda_so1() {
        local candidate
        for candidate in \
            /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
            /lib/x86_64-linux-gnu/libcuda.so.1 \
            /usr/lib64/libcuda.so.1 \
            /lib64/libcuda.so.1
        do
            if [ -f "$candidate" ]; then
                printf '%s\n' "$candidate"
                return 0
            fi
        done
        return 1
    }

    # UniDepth and RoMaV2 rely on mpmath from the user's local site-packages.
    # Keep PYTHONNOUSERSITE enabled globally, but expose the exact dependency path explicitly.
    if [ -z "$UNIDEPTH_EXTRA_PYTHONPATH" ] && [ -x "$UNIDEPTH_PY" ]; then
        local unidepth_usersite
        unidepth_usersite="$(resolve_usersite_for_python "$UNIDEPTH_PY")"
        if [ -n "$unidepth_usersite" ] && [ -d "$unidepth_usersite/mpmath" ]; then
            UNIDEPTH_EXTRA_PYTHONPATH="$unidepth_usersite"
        fi
    fi

    if [ -z "$ROMAV2_EXTRA_PYTHONPATH" ] && [ -x "$ROMAV2_PY" ]; then
        local romav2_usersite
        romav2_usersite="$(resolve_usersite_for_python "$ROMAV2_PY")"
        if [ -n "$romav2_usersite" ] && [ -d "$romav2_usersite/mpmath" ]; then
            ROMAV2_EXTRA_PYTHONPATH="$romav2_usersite"
        fi
    fi

    if [ -z "$ROMAV2_CUDA_SHIM_DIR" ]; then
        local libcuda_so1
        libcuda_so1="$(resolve_libcuda_so1 || true)"
        if [ -n "$libcuda_so1" ]; then
            ROMAV2_CUDA_SHIM_DIR="$PROJECT_ROOT/cache/runtime/romav2_cuda_shim"
            mkdir -p "$ROMAV2_CUDA_SHIM_DIR"
            ln -sfn "$libcuda_so1" "$ROMAV2_CUDA_SHIM_DIR/libcuda.so"
        fi
    fi

    export UNIDEPTH_EXTRA_PYTHONPATH
    export ROMAV2_EXTRA_PYTHONPATH
    export ROMAV2_CUDA_SHIM_DIR
}

require_file() {
    if [ ! -f "$1" ]; then
        echo "Missing required file: $1" >&2
        return 1
    fi
}

require_dir() {
    if [ ! -d "$1" ]; then
        echo "Missing required directory: $1" >&2
        return 1
    fi
}

scene_root() {
    printf '%s\n' "$PROJECT_ROOT/datasets/phototourism/$1"
}

scene_images_dir() {
    printf '%s\n' "$PROJECT_ROOT/datasets/phototourism/$1/dense/images"
}

scene_preprocessed_dir() {
    printf '%s\n' "$PROJECT_ROOT/datasets/phototourism/$1/images_preprocessed"
}

scene_preprocess_info() {
    printf '%s\n' "$PROJECT_ROOT/datasets/phototourism/$1/images_preprocessed/preprocess_info.json"
}

scene_sparse_dir() {
    printf '%s\n' "$PROJECT_ROOT/datasets/phototourism/$1/dense/sparse"
}

scene_depth_dir() {
    printf '%s\n' "$PROJECT_ROOT/datasets/phototourism/$1/depth_unidepth"
}

scene_pairs_file() {
    printf '%s\n' "$PROJECT_ROOT/output/pairs_$1.txt"
}

scene_image_count() {
    local dir
    dir="$(scene_images_dir "$1")"
    if [ ! -d "$dir" ]; then
        echo 0
        return
    fi
    find "$dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l
}

scene_preprocessed_count() {
    local dir
    dir="$(scene_preprocessed_dir "$1")"
    if [ ! -d "$dir" ]; then
        echo 0
        return
    fi
    find "$dir" -maxdepth 1 -type f -iname '*.jpg' | wc -l
}

scene_depth_count() {
    local dir
    dir="$(scene_depth_dir "$1")"
    if [ ! -d "$dir" ]; then
        echo 0
        return
    fi
    find "$dir" -maxdepth 1 -type f -name '*_depth.npy' | wc -l
}

scene_pair_count() {
    local file
    file="$(scene_pairs_file "$1")"
    if [ ! -f "$file" ]; then
        echo 0
        return
    fi
    wc -l < "$file"
}

scene_pair_limit() {
    local pair_count
    pair_count="$(scene_pair_count "$1")"
    if [ "$pair_count" -le 0 ]; then
        echo 0
    elif [ "$pair_count" -lt "$THESIS_PAIR_LIMIT" ]; then
        echo "$pair_count"
    else
        echo "$THESIS_PAIR_LIMIT"
    fi
}

print_readiness_table() {
    local status=0
    local scene img prep depth pairs info ready
    for scene in "${TEST_SCENES[@]}"; do
        img="$(scene_image_count "$scene")"
        prep="$(scene_preprocessed_count "$scene")"
        depth="$(scene_depth_count "$scene")"
        pairs="$(scene_pair_count "$scene")"
        if [ -f "$(scene_preprocess_info "$scene")" ]; then
            info="Y"
        else
            info="N"
        fi
        ready="YES"
        [ "$prep" -lt "$img" ] && ready="NO"
        [ "$depth" -lt "$img" ] && ready="NO"
        [ "$pairs" = "0" ] && ready="NO"
        [ "$info" = "N" ] && ready="NO"
        echo "$scene | img=$img | prep=$prep | depth=$depth | pairs=$pairs | info=$info | ready=$ready"
        [ "$ready" = "YES" ] || status=1
    done
    return "$status"
}

resolve_gpu_ids() {
    local spec="${1:-auto}"
    if [ -n "$spec" ] && [ "$spec" != "auto" ]; then
        tr ',' '\n' <<<"$spec" | sed '/^$/d'
        return
    fi
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | awk '{print $1}'
}

require_runtime_prereqs() {
    require_dir "$PROJECT_ROOT/datasets/phototourism"
    require_file "$DINOV3_PY"
    require_file "$DIFT_PY"
    require_file "$LIGHTGLUE_PY"
    require_file "$ROMA_PY"
    require_file "$ROMAV2_PY"
    require_file "$REPOSED_PY"
    require_file "$UNIDEPTH_PY"
    require_file "$OURS151_LIGHTGLUE_CKPT"
    require_file "$PROJECTION_CKPT"

    require_dir "$HF_HOME/hub/models--lpiccinelli--unidepth-v2-vitl14"
    require_dir "$HF_HOME/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5"
    require_dir "$HF_HOME/hub/models--timm--vit_large_patch16_dinov3.lvd1689m"

    require_file "$HOME/.cache/torch/hub/checkpoints/roma_outdoor.pth"
    require_file "$HOME/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth"
    require_file "$HOME/.cache/torch/hub/checkpoints/romav2.pt"
    require_file "$HOME/.cache/torch/hub/checkpoints/superpoint_v1.pth"
    require_file "$HOME/.cache/torch/hub/checkpoints/superpoint_lightglue_v0-1_arxiv.pth"

    local unidepth_pythonpath
    unidepth_pythonpath="${UNIDEPTH_EXTRA_PYTHONPATH:-}"
    if ! PYTHONNOUSERSITE=1 PYTHONPATH="$unidepth_pythonpath" "$UNIDEPTH_PY" - <<'PY' >/dev/null 2>&1
from unidepth.models import UniDepthV2
PY
    then
        echo "UniDepth preflight failed. Import check could not load UniDepthV2." >&2
        echo "UNIDEPTH_PY=$UNIDEPTH_PY" >&2
        echo "UNIDEPTH_EXTRA_PYTHONPATH=${UNIDEPTH_EXTRA_PYTHONPATH:-<empty>}" >&2
        return 1
    fi

    local romav2_pythonpath
    romav2_pythonpath="${ROMAV2_EXTRA_PYTHONPATH:-}"
    if [ -z "${ROMAV2_CUDA_SHIM_DIR:-}" ] || [ ! -f "$ROMAV2_CUDA_SHIM_DIR/libcuda.so" ]; then
        echo "RoMaV2 preflight failed. CUDA linker shim was not prepared." >&2
        echo "ROMAV2_CUDA_SHIM_DIR=${ROMAV2_CUDA_SHIM_DIR:-<empty>}" >&2
        return 1
    fi
    if ! PYTHONNOUSERSITE=1 \
        PYTHONPATH="$romav2_pythonpath" \
        LD_LIBRARY_PATH="$ROMAV2_CUDA_SHIM_DIR:${LD_LIBRARY_PATH:-}" \
        LIBRARY_PATH="$ROMAV2_CUDA_SHIM_DIR:${LIBRARY_PATH:-}" \
        "$ROMAV2_PY" - <<'PY' >/dev/null 2>&1
import sympy
from romav2 import RoMaV2
PY
    then
        echo "RoMaV2 preflight failed. Import check could not load sympy/RoMaV2." >&2
        echo "ROMAV2_PY=$ROMAV2_PY" >&2
        echo "ROMAV2_EXTRA_PYTHONPATH=${ROMAV2_EXTRA_PYTHONPATH:-<empty>}" >&2
        echo "ROMAV2_CUDA_SHIM_DIR=${ROMAV2_CUDA_SHIM_DIR:-<empty>}" >&2
        return 1
    fi

    local scene
    for scene in "${TEST_SCENES[@]}"; do
        require_dir "$(scene_root "$scene")"
        require_dir "$(scene_images_dir "$scene")"
        require_dir "$(scene_sparse_dir "$scene")"
        require_file "$(scene_sparse_dir "$scene")/cameras.bin"
        require_file "$(scene_sparse_dir "$scene")/images.bin"
        require_file "$(scene_sparse_dir "$scene")/points3D.bin"
    done
}

setup_mkl_openmp_env() {
    local mkl_pkg omp_pkg extra_libs
    mkl_pkg="$(find "$HOME/.conda/pkgs" -maxdepth 3 -name 'libmkl_intel_lp64.so.2' 2>/dev/null | head -1 | xargs -I{} dirname {} 2>/dev/null || true)"
    omp_pkg="$(find "$HOME/.conda/pkgs" -maxdepth 3 -name 'libiomp5.so' 2>/dev/null | head -1 | xargs -I{} dirname {} 2>/dev/null || true)"
    extra_libs=""
    [ -n "$mkl_pkg" ] && extra_libs="$mkl_pkg"
    [ -n "$omp_pkg" ] && extra_libs="${extra_libs:+$extra_libs:}$omp_pkg"
    if [ -n "$extra_libs" ]; then
        export LD_LIBRARY_PATH="${extra_libs}:${LD_LIBRARY_PATH:-}"
    fi
}

method_label() {
    printf '%s\n' "${METHOD_LABELS[$1]}"
}
