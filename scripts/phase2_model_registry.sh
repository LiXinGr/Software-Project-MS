#!/usr/bin/env bash
# Central registry of Phase 2a/2b feature models.
#
# Purpose:
# - keep the "current best model" for future experiment branches in one place
# - allow future scripts (including a LightGlue branch) to switch models by alias
# - avoid hardcoding checkpoint paths across multiple launchers
#
# This file does NOT implement the LightGlue phase. It only prepares model
# selection so that future work can reuse the same Phase 2a/2b model aliases.
#
# Usage:
#   source scripts/phase2_model_registry.sh
#   phase2_select_model "${PHASE2_ACTIVE_MODEL}"
#   phase2_print_selected_model
#
# To switch model:
#   export PHASE2_ACTIVE_MODEL=phase2b_lora_fusion_jointwarm10k
#   source scripts/phase2_model_registry.sh
#   phase2_select_model "$PHASE2_ACTIVE_MODEL"

PHASE2_ACTIVE_MODEL="${PHASE2_ACTIVE_MODEL:-phase2a_raw_fusion_wide}"

phase2_list_models() {
    cat <<'EOF'
phase2a_raw_fusion_wide
  Raw DIFT + raw DINOv3 + wide projection head
  Current default for future branches because it is the strongest overall Phase 2 result.

phase2b_lora_fusion_jointwarm10k
  DIFT + LoRA-DINOv3 + projection head, joint warm-start
  Best current LoRA-fusion checkpoint.

phase2b_lora_dino_proj_v2
  LoRA-DINOv3 + projection head, DINO-only
  Adopted best DINO-only LoRA+projection checkpoint.

phase2b_lora_fusion_wide512clean
  DIFT + LoRA-DINOv3 + projection head, frozen-backbone wide settings
  Best clean frozen-backbone LoRA-fusion checkpoint before the joint warm-start run.
EOF
}

phase2_clear_selected_model() {
    unset PHASE2_MODEL_KEY
    unset PHASE2_MODEL_FAMILY
    unset PHASE2_MODEL_DESC
    unset PHASE2_MODEL_CHECKPOINT
    unset PHASE2_MODEL_MATCH_SCRIPT_HINT
    unset PHASE2_MODEL_SOURCE_KIND
    unset PHASE2_MODEL_USE_DINOV3
    unset PHASE2_MODEL_USE_DIFT
    unset PHASE2_MODEL_USE_LORA
    unset PHASE2_MODEL_USE_PROJECTION
    unset PHASE2_MODEL_FEAT_LEVEL
    unset PHASE2_MODEL_FUSION_ALPHA
    unset PHASE2_MODEL_IMG_SIZE
    unset PHASE2_MODEL_MAX_POINTS
    unset PHASE2_MODEL_REPORT_AVG_MAA10
    unset PHASE2_MODEL_STATUS
}

phase2_select_model() {
    local key="${1:-$PHASE2_ACTIVE_MODEL}"
    phase2_clear_selected_model

    case "$key" in
        phase2a_raw_fusion_wide)
            export PHASE2_MODEL_KEY="$key"
            export PHASE2_MODEL_FAMILY="phase2a"
            export PHASE2_MODEL_DESC="Raw DIFT + raw DINOv3 + wide projection head"
            export PHASE2_MODEL_CHECKPOINT="experiments/phase2_projection_wide/best.pt"
            export PHASE2_MODEL_MATCH_SCRIPT_HINT="projection_matches.py"
            export PHASE2_MODEL_SOURCE_KIND="raw_fusion_projection"
            export PHASE2_MODEL_USE_DINOV3="1"
            export PHASE2_MODEL_USE_DIFT="1"
            export PHASE2_MODEL_USE_LORA="0"
            export PHASE2_MODEL_USE_PROJECTION="1"
            export PHASE2_MODEL_FEAT_LEVEL="-8"
            export PHASE2_MODEL_FUSION_ALPHA="0.5"
            export PHASE2_MODEL_IMG_SIZE="768 768"
            export PHASE2_MODEL_MAX_POINTS="2000"
            export PHASE2_MODEL_REPORT_AVG_MAA10="76.7"
            export PHASE2_MODEL_STATUS="default_current_best"
            ;;

        phase2b_lora_fusion_jointwarm10k)
            export PHASE2_MODEL_KEY="$key"
            export PHASE2_MODEL_FAMILY="phase2b"
            export PHASE2_MODEL_DESC="DIFT + LoRA-DINOv3 + projection head, joint warm-start"
            export PHASE2_MODEL_CHECKPOINT="experiments/phase2_lora_joint_fusion_projwarm_jointwarm10k_20260328_184458/best.pt"
            export PHASE2_MODEL_MATCH_SCRIPT_HINT="lora_matches.py"
            export PHASE2_MODEL_SOURCE_KIND="lora_fusion_projection_joint"
            export PHASE2_MODEL_USE_DINOV3="1"
            export PHASE2_MODEL_USE_DIFT="1"
            export PHASE2_MODEL_USE_LORA="1"
            export PHASE2_MODEL_USE_PROJECTION="1"
            export PHASE2_MODEL_FEAT_LEVEL="-8"
            export PHASE2_MODEL_FUSION_ALPHA="0.5"
            export PHASE2_MODEL_IMG_SIZE="768 768"
            export PHASE2_MODEL_MAX_POINTS="2000"
            export PHASE2_MODEL_REPORT_AVG_MAA10="75.8000"
            export PHASE2_MODEL_STATUS="best_current_lora_fusion"
            ;;

        phase2b_lora_dino_proj_v2)
            export PHASE2_MODEL_KEY="$key"
            export PHASE2_MODEL_FAMILY="phase2b"
            export PHASE2_MODEL_DESC="LoRA-DINOv3 + projection head, DINO-only"
            export PHASE2_MODEL_CHECKPOINT="experiments/phase2_lora_proj_dinov3only_v2/best.pt"
            export PHASE2_MODEL_MATCH_SCRIPT_HINT="lora_matches.py"
            export PHASE2_MODEL_SOURCE_KIND="lora_dino_projection"
            export PHASE2_MODEL_USE_DINOV3="1"
            export PHASE2_MODEL_USE_DIFT="0"
            export PHASE2_MODEL_USE_LORA="1"
            export PHASE2_MODEL_USE_PROJECTION="1"
            export PHASE2_MODEL_FEAT_LEVEL="-8"
            export PHASE2_MODEL_FUSION_ALPHA=""
            export PHASE2_MODEL_IMG_SIZE="768 768"
            export PHASE2_MODEL_MAX_POINTS="2000"
            export PHASE2_MODEL_REPORT_AVG_MAA10="74.8830"
            export PHASE2_MODEL_STATUS="best_current_lora_dino_only"
            ;;

        phase2b_lora_fusion_wide512clean)
            export PHASE2_MODEL_KEY="$key"
            export PHASE2_MODEL_FAMILY="phase2b"
            export PHASE2_MODEL_DESC="DIFT + LoRA-DINOv3 + projection head, frozen-backbone wide settings"
            export PHASE2_MODEL_CHECKPOINT="experiments/phase2_lora_proj_fusion_wide512clean_20260327_214806/best.pt"
            export PHASE2_MODEL_MATCH_SCRIPT_HINT="lora_matches.py"
            export PHASE2_MODEL_SOURCE_KIND="lora_fusion_projection_frozen"
            export PHASE2_MODEL_USE_DINOV3="1"
            export PHASE2_MODEL_USE_DIFT="1"
            export PHASE2_MODEL_USE_LORA="1"
            export PHASE2_MODEL_USE_PROJECTION="1"
            export PHASE2_MODEL_FEAT_LEVEL="-8"
            export PHASE2_MODEL_FUSION_ALPHA="0.5"
            export PHASE2_MODEL_IMG_SIZE="768 768"
            export PHASE2_MODEL_MAX_POINTS="2000"
            export PHASE2_MODEL_REPORT_AVG_MAA10="74.5694"
            export PHASE2_MODEL_STATUS="clean_frozen_lora_fusion_reference"
            ;;

        *)
            echo "Unknown Phase 2 model alias: $key" >&2
            echo >&2
            echo "Available aliases:" >&2
            phase2_list_models >&2
            return 1
            ;;
    esac

    export PHASE2_ACTIVE_MODEL="$key"
}

phase2_print_selected_model() {
    cat <<EOF
PHASE2_ACTIVE_MODEL=$PHASE2_ACTIVE_MODEL
PHASE2_MODEL_KEY=$PHASE2_MODEL_KEY
PHASE2_MODEL_FAMILY=$PHASE2_MODEL_FAMILY
PHASE2_MODEL_DESC=$PHASE2_MODEL_DESC
PHASE2_MODEL_CHECKPOINT=$PHASE2_MODEL_CHECKPOINT
PHASE2_MODEL_MATCH_SCRIPT_HINT=$PHASE2_MODEL_MATCH_SCRIPT_HINT
PHASE2_MODEL_SOURCE_KIND=$PHASE2_MODEL_SOURCE_KIND
PHASE2_MODEL_USE_DINOV3=$PHASE2_MODEL_USE_DINOV3
PHASE2_MODEL_USE_DIFT=$PHASE2_MODEL_USE_DIFT
PHASE2_MODEL_USE_LORA=$PHASE2_MODEL_USE_LORA
PHASE2_MODEL_USE_PROJECTION=$PHASE2_MODEL_USE_PROJECTION
PHASE2_MODEL_FEAT_LEVEL=$PHASE2_MODEL_FEAT_LEVEL
PHASE2_MODEL_FUSION_ALPHA=$PHASE2_MODEL_FUSION_ALPHA
PHASE2_MODEL_IMG_SIZE=$PHASE2_MODEL_IMG_SIZE
PHASE2_MODEL_MAX_POINTS=$PHASE2_MODEL_MAX_POINTS
PHASE2_MODEL_REPORT_AVG_MAA10=$PHASE2_MODEL_REPORT_AVG_MAA10
PHASE2_MODEL_STATUS=$PHASE2_MODEL_STATUS
EOF
}

