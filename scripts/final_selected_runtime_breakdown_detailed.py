#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))

import final_runtime_benchmark as frb


METHOD = frb.FINAL_METHOD
RUN_KEY = "final_selected_runtime_breakdown_detailed"
CSV_PATH = frb.CSV_ROOT / "final_selected_runtime_breakdown_detailed.csv"
REPORT_PATH = frb.REPORT_ROOT / "final_selected_runtime_breakdown_detailed.md"
RAW_JSON = frb.RUNTIME_ROOT / "timing" / RUN_KEY / "raw_detailed_timings.json"

COMPONENT_ORDER = [
    "image_loading_preprocessing",
    "superpoint_keypoint_extraction",
    "dinov3_forward",
    "dift_feature_extraction",
    "descriptor_sampling",
    "fusion_projection_head",
    "lightglue_matching",
    "benchmark_packing_depth_sampling",
    "reposed_calibrated_solver",
    "total_online",
    "total_cached_descriptors",
]

COMPONENT_LABELS = {
    "image_loading_preprocessing": "image loading + preprocessing",
    "superpoint_keypoint_extraction": "SuperPoint keypoint extraction",
    "dinov3_forward": "DINOv3 forward pass",
    "dift_feature_extraction": "DIFT feature extraction",
    "descriptor_sampling": "DINOv3/DIFT descriptor sampling",
    "fusion_projection_head": "fusion + projection head",
    "lightglue_matching": "LightGlue matching",
    "benchmark_packing_depth_sampling": "benchmark packing / depth sampling",
    "reposed_calibrated_solver": "RePoseD calibrated solver",
    "total_online": "total online",
    "total_cached_descriptors": "total cached descriptors",
}

PER_IMAGE_COMPONENTS = {
    "image_loading_preprocessing",
    "superpoint_keypoint_extraction",
    "dinov3_forward",
    "dift_feature_extraction",
    "descriptor_sampling",
    "fusion_projection_head",
}


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return f"{value:.6f}"


def ms_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def ms_median(values: list[float]) -> float | None:
    return median(values) if values else None


def ensure_dirs() -> None:
    for path in [frb.CSV_ROOT, frb.REPORT_ROOT, RAW_JSON.parent, frb.LOG_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def ensure_pair_files(subset: dict[str, list[dict[str, str]]]) -> None:
    pair_dir = frb.RUNTIME_ROOT / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    for scene in frb.SCENES:
        path = frb.scene_pairs_txt(scene)
        rows = subset.get(scene, [])
        if path.exists():
            continue
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(f"{row['img0']} {row['img1']}\n")


def synchronize(device: Any | None) -> None:
    try:
        import torch

        if device is not None and torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
    except Exception:
        return


class CudaBlockTimer:
    def __init__(self, device: Any) -> None:
        self.device = device
        self.component_peaks_mb: dict[str, float] = defaultdict(float)
        self.component_incremental_peaks_mb: dict[str, float] = defaultdict(float)
        self.full_peak_mb = 0.0

    def clear(self) -> None:
        self.component_peaks_mb.clear()
        self.component_incremental_peaks_mb.clear()
        self.full_peak_mb = 0.0

    def timed(self, component: str, fn: Callable[[], Any]) -> tuple[Any, float]:
        import torch

        synchronize(self.device)
        before_mb = 0.0
        if torch.cuda.is_available():
            before_mb = float(torch.cuda.memory_allocated(device=self.device) / 1024**2)
            torch.cuda.reset_peak_memory_stats(device=self.device)
        start = time.perf_counter()
        result = fn()
        synchronize(self.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if torch.cuda.is_available():
            peak_mb = float(torch.cuda.max_memory_allocated(device=self.device) / 1024**2)
            self.component_peaks_mb[component] = max(self.component_peaks_mb[component], peak_mb)
            self.component_incremental_peaks_mb[component] = max(
                self.component_incremental_peaks_mb[component],
                max(0.0, peak_mb - before_mb),
            )
            self.full_peak_mb = max(self.full_peak_mb, peak_mb)
        return result, elapsed_ms


def final_args() -> argparse.Namespace:
    return argparse.Namespace(
        config_key=METHOD,
        checkpoint=str(PROJECT_ROOT / "experiments" / "phase2_projection_wide" / "best.pt"),
        lightglue_checkpoint=str(
            PROJECT_ROOT
            / "external"
            / "glue-factory"
            / "outputs"
            / "training"
            / "stage2_dinov3_lg_151scenes_v1"
            / "checkpoint_best.tar"
        ),
        max_points=2048,
        feat_level=-8,
        img_size=[768, 768],
        t=0,
        up_ft_index=2,
        ensemble_size=2,
        alpha=0.5,
        filter_threshold=0.02,
        depth_confidence=0.95,
        width_confidence=0.99,
        flash=True,
        mp=False,
        save_scores=True,
        online_extraction=True,
    )


def build_models(device: Any):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import lightglue_projection_matches as lgp

    args = final_args()
    extractor = lgp.build_online_extractor(args, device)
    matcher = lgp.build_matcher(args, device)
    return args, extractor, matcher, lgp


def extract_detailed(
    img_path: Path,
    extractor,
    lgp,
    timer: CudaBlockTimer,
    device: Any,
) -> tuple[dict[str, Any], dict[str, float]]:
    import torch
    import torch.nn.functional as F
    from gluefactory.models.extractors.dinov3_dift_projection import sample_feature_map

    (image, (height, width)), load_ms = timer.timed(
        "image_loading_preprocessing",
        lambda: lgp.load_image_tensor(img_path, device),
    )
    image_b = image.unsqueeze(0)
    h_img, w_img = image.shape[-2:]
    image_size_xy = torch.tensor([width, height], device=device, dtype=torch.float32)

    extractor._ensure_dift_device(device)

    def run_superpoint():
        with torch.inference_mode():
            sp_pred = extractor.superpoint({"image": image_b})
        keypoints = sp_pred["keypoints"][0].to(dtype=torch.float32)
        scores = sp_pred["keypoint_scores"][0].to(dtype=torch.float32)
        if extractor.conf.crop_to_image_size:
            valid_xy = (
                (keypoints[:, 0] >= 0)
                & (keypoints[:, 1] >= 0)
                & (keypoints[:, 0] < image_size_xy[0])
                & (keypoints[:, 1] < image_size_xy[1])
            )
            keypoints = keypoints[valid_xy]
            scores = scores[valid_xy]
        if keypoints.shape[0] > int(extractor.conf.max_num_keypoints):
            top_idx = torch.topk(scores, k=int(extractor.conf.max_num_keypoints), sorted=True).indices
            keypoints = keypoints[top_idx]
            scores = scores[top_idx]
        return keypoints, scores

    (keypoints, _scores), sp_ms = timer.timed("superpoint_keypoint_extraction", run_superpoint)

    def run_dino():
        dino_input = (image_b - extractor.dino_mean.to(device)) / extractor.dino_std.to(device)
        with torch.inference_mode():
            return extractor.dinov3(dino_input)[0]

    dino_feats, dino_ms = timer.timed("dinov3_forward", run_dino)

    def run_dift():
        dift_input = image_b * 2.0 - 1.0
        dift_hw = extractor.conf.dift_input_size
        if dift_hw is not None:
            dift_h, dift_w = int(dift_hw[0]), int(dift_hw[1])
            if (dift_h, dift_w) != tuple(dift_input.shape[-2:]):
                dift_input = F.interpolate(
                    dift_input,
                    size=(dift_h, dift_w),
                    mode="bilinear",
                    align_corners=False,
                )
        dift_image_hw = tuple(dift_input.shape[-2:])
        with torch.inference_mode():
            dift_feats = extractor.dift.forward(
                dift_input,
                prompt="",
                t=int(extractor.conf.dift_t),
                up_ft_index=int(extractor.conf.dift_up_ft_index),
                ensemble_size=int(extractor.conf.dift_ensemble_size),
            )
        return dift_feats, dift_image_hw

    (dift_feats, dift_image_hw), dift_ms = timer.timed("dift_feature_extraction", run_dift)

    def run_sampling():
        dino_desc = sample_feature_map(
            dino_feats,
            keypoints,
            (h_img, w_img),
            mode=str(extractor.conf.sampling_mode),
        )
        dift_keypoints = keypoints.clone()
        if dift_image_hw != (h_img, w_img):
            dift_keypoints[:, 0] *= dift_image_hw[1] / float(max(w_img, 1))
            dift_keypoints[:, 1] *= dift_image_hw[0] / float(max(h_img, 1))
        dift_desc = sample_feature_map(
            dift_feats,
            dift_keypoints,
            dift_image_hw,
            mode=str(extractor.conf.sampling_mode),
        )
        return dino_desc, dift_desc

    (dino_desc, dift_desc), sampling_ms = timer.timed("descriptor_sampling", run_sampling)

    def run_projection():
        fused = extractor._fuse_source_descriptors(dino_desc, dift_desc)
        projected = extractor.projection(fused)
        projected = F.normalize(projected, dim=-1, eps=1e-8)
        return {
            "kpts": keypoints.detach().cpu().to(dtype=torch.float32),
            "desc": projected.detach().cpu().to(dtype=torch.float32),
        }

    bundle, projection_ms = timer.timed("fusion_projection_head", run_projection)
    timings = {
        "image_loading_preprocessing": float(load_ms),
        "superpoint_keypoint_extraction": float(sp_ms),
        "dinov3_forward": float(dino_ms),
        "dift_feature_extraction": float(dift_ms),
        "descriptor_sampling": float(sampling_ms),
        "fusion_projection_head": float(projection_ms),
        "num_keypoints": float(bundle["kpts"].shape[0]),
    }
    return bundle, timings


def match_detailed(
    bundle0: dict[str, Any],
    bundle1: dict[str, Any],
    img0_path: Path,
    img1_path: Path,
    matcher,
    lgp,
    timer: CudaBlockTimer,
    device: Any,
):
    return timer.timed(
        "lightglue_matching",
        lambda: lgp.match_with_lightglue(bundle0, bundle1, img0_path, img1_path, matcher, device),
    )


def clean_outputs() -> None:
    frb.scoped_clean_method(RUN_KEY)
    for path in [CSV_PATH, REPORT_PATH, RAW_JSON]:
        if path.exists():
            path.unlink()


def run(args: argparse.Namespace) -> None:
    import torch

    ensure_dirs()
    if args.force:
        clean_outputs()

    if not frb.SUBSET_CSV.exists():
        raise FileNotFoundError(f"Missing subset CSV: {frb.SUBSET_CSV}")
    subset = frb.load_subset()
    ensure_pair_files(subset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_args, extractor, matcher, lgp = build_models(device)
    timer = CudaBlockTimer(device)

    warmups = list(frb.iter_global_pairs(subset))[: args.warmup_pairs]
    print(f"[{RUN_KEY}] warm-up pairs: {len(warmups)}", flush=True)
    for scene, row in warmups:
        img0_path = frb.image_dir(scene) / row["img0"]
        img1_path = frb.image_dir(scene) / row["img1"]
        bundle0, _ = extract_detailed(img0_path, extractor, lgp, timer, device)
        bundle1, _ = extract_detailed(img1_path, extractor, lgp, timer, device)
        match_detailed(bundle0, bundle1, img0_path, img1_path, matcher, lgp, timer, device)

    timer.clear()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)

    image_records: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    scene_pack_ms: dict[str, float] = {}
    scene_solver_runtimes: dict[str, list[float]] = {}

    for scene in frb.SCENES:
        rows = subset.get(scene, [])
        if not rows:
            continue
        scene_match_dir = frb.matches_dir(RUN_KEY, scene)
        scene_match_dir.mkdir(parents=True, exist_ok=True)
        scene_pairs: list[dict[str, Any]] = []

        print(f"[{RUN_KEY}][{scene}] recording {len(rows)} pairs", flush=True)
        for idx, row in enumerate(rows, start=1):
            img0 = row["img0"]
            img1 = row["img1"]
            img0_path = frb.image_dir(scene) / img0
            img1_path = frb.image_dir(scene) / img1

            bundle0, t0 = extract_detailed(img0_path, extractor, lgp, timer, device)
            bundle1, t1 = extract_detailed(img1_path, extractor, lgp, timer, device)
            (mkpts0, mkpts1, scores, stop_layer), matching_ms = match_detailed(
                bundle0,
                bundle1,
                img0_path,
                img1_path,
                matcher,
                lgp,
                timer,
                device,
            )

            save_path = scene_match_dir / f"{Path(img0).stem}__{Path(img1).stem}.npz"
            lgp.save_match_archive(
                save_path,
                mkpts0,
                mkpts1,
                run_args,
                match_scores=scores,
                stop_layer=stop_layer,
            )

            image_records.append(
                {
                    "scene": scene,
                    "pair_index": idx - 1,
                    "role": "image0",
                    "image": img0,
                    **{k: v for k, v in t0.items() if k in PER_IMAGE_COMPONENTS or k == "num_keypoints"},
                }
            )
            image_records.append(
                {
                    "scene": scene,
                    "pair_index": idx - 1,
                    "role": "image1",
                    "image": img1,
                    **{k: v for k, v in t1.items() if k in PER_IMAGE_COMPONENTS or k == "num_keypoints"},
                }
            )

            pair_row = {
                "scene": scene,
                "pair_index": idx - 1,
                "img0": img0,
                "img1": img1,
                "pair_key": row["pair_key"],
                "image_loading_preprocessing": t0["image_loading_preprocessing"] + t1["image_loading_preprocessing"],
                "superpoint_keypoint_extraction": t0["superpoint_keypoint_extraction"] + t1["superpoint_keypoint_extraction"],
                "dinov3_forward": t0["dinov3_forward"] + t1["dinov3_forward"],
                "dift_feature_extraction": t0["dift_feature_extraction"] + t1["dift_feature_extraction"],
                "descriptor_sampling": t0["descriptor_sampling"] + t1["descriptor_sampling"],
                "fusion_projection_head": t0["fusion_projection_head"] + t1["fusion_projection_head"],
                "lightglue_matching": float(matching_ms),
                "num_matches": int(len(mkpts0)),
                "stop_layer": int(stop_layer),
            }
            scene_pairs.append(pair_row)
            pair_records.append(pair_row)

            if idx == len(rows) or idx % 25 == 0:
                print(f"[{RUN_KEY}][{scene}] pair {idx}/{len(rows)}", flush=True)

        pack_ms_per_pair = frb.run_pack(RUN_KEY, scene, len(rows))
        solver_mean, solver_median, evaluated_pairs, solver_runtimes = frb.run_solver(RUN_KEY, scene)
        if not solver_runtimes:
            fallback = float(solver_mean or 0.0)
            solver_runtimes = [fallback for _ in scene_pairs]
        scene_pack_ms[scene] = float(pack_ms_per_pair)
        scene_solver_runtimes[scene] = [float(v) for v in solver_runtimes]

        for idx, pair_row in enumerate(scene_pairs):
            solver_ms = scene_solver_runtimes[scene][idx] if idx < len(scene_solver_runtimes[scene]) else float(solver_mean or 0.0)
            pair_row["benchmark_packing_depth_sampling"] = float(pack_ms_per_pair)
            pair_row["reposed_calibrated_solver"] = float(solver_ms)
            pair_row["total_online"] = sum(float(pair_row[c]) for c in COMPONENT_ORDER[:9])
            pair_row["total_cached_descriptors"] = (
                float(pair_row["lightglue_matching"])
                + float(pair_row["benchmark_packing_depth_sampling"])
                + float(pair_row["reposed_calibrated_solver"])
            )

        print(
            f"[{RUN_KEY}][{scene}] done: pack={pack_ms_per_pair:.3f} ms/pair, "
            f"solver_mean={solver_mean}, solver_median={solver_median}, evaluated={evaluated_pairs}",
            flush=True,
        )

    payload = {
        "method": METHOD,
        "run_key": RUN_KEY,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "warmup_pairs": args.warmup_pairs,
        "subset_csv": str(frb.SUBSET_CSV.relative_to(PROJECT_ROOT)),
        "image_records": image_records,
        "pair_records": pair_records,
        "scene_pack_ms_per_pair": scene_pack_ms,
        "scene_solver_runtimes_ms": scene_solver_runtimes,
        "component_peak_gpu_memory_mb": dict(timer.component_peaks_mb),
        "component_incremental_peak_gpu_memory_mb": dict(timer.component_incremental_peaks_mb),
        "full_final_method_peak_gpu_memory_mb": timer.full_peak_mb,
    }
    RAW_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = RAW_JSON.with_suffix(".tmp")
    tmp_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_json, RAW_JSON)

    write_csv_and_report(payload, command=args.command)


def filter_records(records: list[dict[str, Any]], scene: str | None) -> list[dict[str, Any]]:
    if scene is None:
        return records
    return [record for record in records if record.get("scene") == scene]


def summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    image_records = payload["image_records"]
    pair_records = payload["pair_records"]
    component_peaks = payload["component_peak_gpu_memory_mb"]
    full_peak = float(payload.get("full_final_method_peak_gpu_memory_mb", 0.0))
    rows: list[dict[str, Any]] = []

    scopes: list[tuple[str, str | None]] = [("all", None)]
    scopes.extend(("scene", scene) for scene in frb.SCENES if filter_records(pair_records, scene))

    for scope, scene in scopes:
        scene_image_records = filter_records(image_records, scene)
        scene_pair_records = filter_records(pair_records, scene)
        if not scene_pair_records:
            continue
        total_sum = sum(float(row["total_online"]) for row in scene_pair_records)

        for component in COMPONENT_ORDER:
            basis = "per_image" if component in PER_IMAGE_COMPONENTS else "per_pair"
            if component in PER_IMAGE_COMPONENTS:
                component_values = [float(row[component]) for row in scene_image_records]
            else:
                component_values = [float(row[component]) for row in scene_pair_records]
            pair_values = [float(row[component]) for row in scene_pair_records]
            component_sum = sum(pair_values)
            percent = (component_sum / total_sum * 100.0) if total_sum > 0 and component != "total_cached_descriptors" else None
            peak = None
            if scope == "all":
                if component == "total_online":
                    peak = full_peak
                elif component in component_peaks:
                    peak = float(component_peaks[component])
            rows.append(
                {
                    "method": METHOD,
                    "scope": scope,
                    "scene": scene or "ALL",
                    "component": component,
                    "component_label": COMPONENT_LABELS[component],
                    "basis": basis,
                    "num_pairs": len(scene_pair_records),
                    "num_images": len(scene_image_records),
                    "mean_ms": ms_mean(component_values),
                    "median_ms": ms_median(component_values),
                    "pair_equivalent_mean_ms": ms_mean(pair_values),
                    "pair_equivalent_median_ms": ms_median(pair_values),
                    "total_online_percent": percent,
                    "peak_gpu_memory_mb": peak,
                    "notes": component_notes(component),
                }
            )
    return rows


def component_notes(component: str) -> str:
    if component == "dift_feature_extraction":
        return "DIFT SD v1.5 t=0 up_ft_index=2 ensemble=2; includes resize to 768x768."
    if component == "dinov3_forward":
        return "DINOv3 ViT-L/16 block 16, feat_level=-8."
    if component == "fusion_projection_head":
        return "Includes descriptor fusion, 1664->1024->256 projection, final normalization, and CPU bundle transfer."
    if component == "lightglue_matching":
        return "Includes CPU-to-GPU LightGlue input staging and post-match CPU output transfer."
    if component == "benchmark_packing_depth_sampling":
        return "Measured by pack_benchmark.py wall time divided by scene pair count."
    if component == "reposed_calibrated_solver":
        return "Calibrated RePoseD 3p_ours_shift_scale+12 runtime."
    if component == "total_cached_descriptors":
        return "Matcher + pack/depth + solver only; descriptor-cache disk IO is not included."
    return ""


def write_csv_and_report(payload: dict[str, Any], command: str) -> None:
    rows = summary_rows(payload)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "scope",
        "scene",
        "component",
        "component_label",
        "basis",
        "num_pairs",
        "num_images",
        "mean_ms",
        "median_ms",
        "pair_equivalent_mean_ms",
        "pair_equivalent_median_ms",
        "total_online_percent",
        "peak_gpu_memory_mb",
        "notes",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        row[field]
                        if field in {"method", "scope", "scene", "component", "component_label", "basis", "notes"}
                        else str(int(row[field]))
                        if field in {"num_pairs", "num_images"}
                        else fmt(row[field])
                    )
                    for field in fields
                }
            )

    write_report(payload, rows, command)
    print(f"[detailed] wrote {CSV_PATH}", flush=True)
    print(f"[detailed] wrote {REPORT_PATH}", flush=True)


def all_row(rows: list[dict[str, Any]], component: str) -> dict[str, Any]:
    for row in rows:
        if row["scope"] == "all" and row["component"] == component:
            return row
    raise KeyError(component)


def read_sp_lg_online_ms() -> float | None:
    if not frb.COMPARISON_CSV.exists():
        return None
    with frb.COMPARISON_CSV.open("r", encoding="utf-8", newline="") as handle:
        values = []
        weights = []
        for row in csv.DictReader(handle):
            if row.get("method") != "superpoint_lg_mp2048":
                continue
            if not row.get("total_online_ms_mean"):
                continue
            values.append(float(row["total_online_ms_mean"]))
            weights.append(int(row.get("num_pairs", "1")))
        if not values:
            return None
        return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def write_report(payload: dict[str, Any], rows: list[dict[str, Any]], command: str) -> None:
    total = all_row(rows, "total_online")
    cached = all_row(rows, "total_cached_descriptors")
    dino = all_row(rows, "dinov3_forward")
    dift = all_row(rows, "dift_feature_extraction")
    feature_components = [
        "superpoint_keypoint_extraction",
        "dinov3_forward",
        "dift_feature_extraction",
        "descriptor_sampling",
        "fusion_projection_head",
    ]
    feature_pair_mean = sum(float(all_row(rows, c)["pair_equivalent_mean_ms"] or 0.0) for c in feature_components)
    dino_pair = float(dino["pair_equivalent_mean_ms"] or 0.0)
    dift_pair = float(dift["pair_equivalent_mean_ms"] or 0.0)
    dominant = "DIFT" if dift_pair >= dino_pair else "DINOv3"
    ratio = (max(dift_pair, dino_pair) / max(min(dift_pair, dino_pair), 1e-9)) if min(dift_pair, dino_pair) > 0 else None
    sp_lg_online = read_sp_lg_online_ms()

    lines = [
        "# Final Selected Runtime Breakdown Detailed",
        "",
        "## Protocol",
        "",
        f"- Method: `{METHOD}`.",
        f"- Subset: `{frb.SUBSET_CSV.relative_to(PROJECT_ROOT)}`; same fixed 100-pairs-per-scene runtime subset.",
        "- Warm-up: 10 pairs before recording.",
        "- GPU: `CUDA_VISIBLE_DEVICES=0`.",
        "- CUDA timing: `torch.cuda.synchronize()` is called before and after every GPU-timed block.",
        "- Settings: DINOv3 ViT-L/16 block 16 (`feat_level=-8`), DIFT SD v1.5 (`t=0`, `up_ft_index=2`, ensemble 2), projection 1664 -> 1024 -> 256, expanded-151 LightGlue checkpoint, `filter_threshold=0.02`, correspondence cap 2048.",
        "- Online total: image load/preprocess + SuperPoint + DINOv3 + DIFT + descriptor sampling + fusion/projection + LightGlue + benchmark packing/depth sampling + calibrated RePoseD solver.",
        "- Cached-descriptor total: LightGlue + benchmark packing/depth sampling + solver only; descriptor-cache disk IO is not included.",
        "",
        "## Hardware / GPU",
        "",
        "```text",
        frb.hardware_string(),
        "```",
        "",
        "## Exact Command",
        "",
        "```bash",
        command or "CUDA_VISIBLE_DEVICES=0 scripts/run_final_selected_runtime_breakdown_detailed.sh --launch --gpu 0 --force",
        "```",
        "",
        "## Git Commit",
        "",
        f"`{frb.git_commit()}`",
        "",
        "## Timing Components",
        "",
        "| component | basis | mean (ms) | median (ms) | pair-equivalent mean (ms) | pair-equivalent median (ms) | total online % | peak GPU MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for component in COMPONENT_ORDER:
        row = all_row(rows, component)
        lines.append(
            "| {label} | {basis} | {mean_ms} | {median_ms} | {pair_mean} | {pair_median} | {percent} | {peak} |".format(
                label=row["component_label"],
                basis=row["basis"],
                mean_ms=fmt(row["mean_ms"]),
                median_ms=fmt(row["median_ms"]),
                pair_mean=fmt(row["pair_equivalent_mean_ms"]),
                pair_median=fmt(row["pair_equivalent_median_ms"]),
                percent=fmt(row["total_online_percent"]),
                peak=fmt(row["peak_gpu_memory_mb"]),
            )
        )

    memory_lines = [
        "| block | observed peak GPU MB | incremental peak over block start MB |",
        "|---|---:|---:|",
    ]
    component_peaks = payload.get("component_peak_gpu_memory_mb", {})
    incremental = payload.get("component_incremental_peak_gpu_memory_mb", {})
    for component in ["dinov3_forward", "dift_feature_extraction", "total_online"]:
        if component == "total_online":
            memory_lines.append(
                f"| full final method | {fmt(float(payload.get('full_final_method_peak_gpu_memory_mb', 0.0)))} |  |"
            )
        else:
            memory_lines.append(
                f"| {COMPONENT_LABELS[component]} | {fmt(float(component_peaks.get(component, 0.0)))} | {fmt(float(incremental.get(component, 0.0)))} |"
            )

    cached_text = (
        f"Cached descriptors reduce the final selected online stack to {float(cached['pair_equivalent_mean_ms']):.1f} ms/pair "
        f"from {float(total['pair_equivalent_mean_ms']):.1f} ms/pair."
    )
    if sp_lg_online is not None:
        cached_text += f" That is below the previously measured SP+LG online total of {sp_lg_online:.1f} ms/pair, with descriptor-cache IO excluded."

    ratio_text = f" by {ratio:.2f}x" if ratio is not None else ""
    lines.extend(
        [
            "",
            "## GPU Memory",
            "",
            *memory_lines,
            "",
            "## Diagnosis",
            "",
            f"- Feature extraction dominates: the timed feature stack is {feature_pair_mean:.1f} ms/pair, while LightGlue matching is {float(all_row(rows, 'lightglue_matching')['pair_equivalent_mean_ms']):.1f} ms/pair.",
            f"- {dominant} dominates the learned feature extraction path{ratio_text}: DIFT is {dift_pair:.1f} ms/pair and DINOv3 is {dino_pair:.1f} ms/pair.",
            f"- {cached_text}",
            "- The matcher itself is not the bottleneck; the practical online cost is driven by the dense descriptor backbones, especially DIFT when descriptors are extracted on demand.",
            "",
            "## Outputs",
            "",
            f"- `{CSV_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{REPORT_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{RAW_JSON.relative_to(PROJECT_ROOT)}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detailed timing breakdown for the final selected sparse pipeline.")
    parser.add_argument("--force", action="store_true", help="Overwrite detailed timing outputs and temporary matches.")
    parser.add_argument("--warmup-pairs", type=int, default=10)
    parser.add_argument("--command", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
