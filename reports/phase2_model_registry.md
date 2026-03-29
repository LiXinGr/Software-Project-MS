# Phase 2 Model Registry

Purpose:
- prepare the repo for the future LightGlue branch without implementing LightGlue yet
- keep the current Phase 2a/2b candidate models in one switchable place
- avoid copying checkpoint paths and model choices into multiple future scripts

Registry file:
- `scripts/phase2_model_registry.sh`

## Current Default

The current default alias is:

- `phase2a_raw_fusion_wide`

Why:
- it is the strongest overall Phase 2 result at the moment
- it is the safest default base model for a future LightGlue branch
- if a later LoRA model becomes better, only the registry alias needs to change

## Available Model Aliases

- `phase2a_raw_fusion_wide`
  - Raw DIFT + raw DINOv3 + wide projection head
  - checkpoint: `experiments/phase2_projection_wide/best.pt`
  - report avg: `76.7`

- `phase2b_lora_fusion_jointwarm10k`
  - DIFT + LoRA-DINOv3 + projection head, joint warm-start
  - checkpoint: `experiments/phase2_lora_joint_fusion_projwarm_jointwarm10k_20260328_184458/best.pt`
  - report avg: `75.8000`

- `phase2b_lora_dino_proj_v2`
  - LoRA-DINOv3 + projection head, DINO-only
  - checkpoint: `experiments/phase2_lora_proj_dinov3only_v2/best.pt`
  - report avg: `74.8830`

- `phase2b_lora_fusion_wide512clean`
  - DIFT + LoRA-DINOv3 + projection head, frozen-backbone wide settings
  - checkpoint: `experiments/phase2_lora_proj_fusion_wide512clean_20260327_214806/best.pt`
  - report avg: `74.5694`

## How To Use It Later

In a future script:

```bash
source scripts/phase2_model_registry.sh
phase2_select_model "${PHASE2_ACTIVE_MODEL:-phase2a_raw_fusion_wide}"
phase2_print_selected_model
```

To switch model:

```bash
export PHASE2_ACTIVE_MODEL=phase2b_lora_fusion_jointwarm10k
source scripts/phase2_model_registry.sh
phase2_select_model "$PHASE2_ACTIVE_MODEL"
```

This keeps the future LightGlue branch flexible:
- default to the current best Phase 2 model
- switch to a newer LoRA model by alias
- avoid editing multiple files when the preferred base model changes
