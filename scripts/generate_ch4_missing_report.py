#!/usr/bin/env python3

import argparse
import csv
import glob
import json
from pathlib import Path
from statistics import mean


SCENES = ["sacre_coeur", "reichstag", "st_peters_square"]
RATIOS = ["0.75", "0.8", "0.9", "0.95"]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def summary_path(root, key, scene):
    pattern = root / "output_v2" / "results_v2" / key / scene / f"calibrated-*{key}_{scene}-2.0t_summary.json"
    paths = sorted(glob.glob(str(pattern)))
    return Path(paths[-1]) if paths else None


def primary_metrics(root, key, scene):
    path = summary_path(root, key, scene)
    if not path:
        return None
    data = load_json(path)
    exps = data.get("experiments", [])
    exp = next((e for e in exps if e.get("solver") == "3p_ours_shift_scale+12"), exps[0] if exps else {})
    return {
        "path": str(path.relative_to(root)),
        "maa": exp.get("mAA@10"),
        "inlier": exp.get("mean_inlier_ratio"),
        "median_pose": exp.get("median_pose_error"),
        "pairs": exp.get("num_evaluated_pairs"),
    }


def timing_path(root, key, scene):
    path = root / "output_v2" / "timing" / f"{key}_{scene}_timing.json"
    return path if path.exists() else None


def timing_metrics(root, key, scene):
    path = timing_path(root, key, scene)
    if not path:
        return {}
    data = load_json(path)
    pair = data.get("pair_timings", [])
    feat = data.get("feature_timings", [])
    return {
        "path": str(path.relative_to(root)),
        "pair_ms": mean([x.get("time_ms", 0.0) for x in pair]) if pair else None,
        "matches": mean([x.get("num_matches", 0.0) for x in pair]) if pair else None,
        "feat_ms": mean([x.get("extract_ms", 0.0) for x in feat]) if feat else None,
        "feat_miss_ms": mean([x.get("extract_ms", 0.0) for x in feat if not x.get("cache_hit")]) if any(not x.get("cache_hit") for x in feat) else None,
        "peak": data.get("peak_gpu_memory_mb"),
        "feature_count": len(feat),
        "pair_count": len(pair),
        "skipped_existing": data.get("skipped_existing"),
        "coordinate_frame": data.get("coordinate_frame"),
    }


def failure_note(root, key, scene):
    path = root / "output_v2" / "logs" / f"{key}_{scene}.failure.log"
    if path.exists():
        return path.read_text(errors="replace").strip()
    return None


def avg(vals):
    vals = [v for v in vals if v is not None]
    return mean(vals) if vals else None


def fmt_maa(v):
    return "--" if v is None else f"{v:.1f}"


def fmt_pct(v):
    return "--" if v is None else f"{100.0 * v:.1f}%"


def fmt_pose(v):
    return "--" if v is None else f"{v:.2f}"


def fmt_int(v):
    return "--" if v is None else f"{round(v):.0f}"


def fmt_ms(v):
    return "--" if v is None else f"{v:.1f}"


def paths_cell(paths):
    paths = [p for p in paths if p]
    return "<br>".join(f"`{p}`" for p in paths) if paths else "--"


def command_rows(root):
    path = root / "output_v2" / "logs" / "ch4_missing_commands.tsv"
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) >= 5:
                    rows.append({"time": row[0], "stage": row[1], "key": row[2], "scene": row[3], "cmd": row[4]})
    return rows


def command_cell(commands, key, scenes):
    scenes = set(scenes if isinstance(scenes, (list, tuple, set)) else [scenes])
    rows = [r for r in commands if r["key"] == key and r["scene"] in scenes]
    if not rows:
        return "--"
    parts = []
    for r in rows[:3]:
        parts.append(f"`{r['stage']}: {r['cmd']}`")
    if len(rows) > 3:
        parts.append(f"`... {len(rows) - 3} more command log entries`")
    return "<br>".join(parts)


def row_for_key(root, key, scenes=SCENES):
    ms = {s: primary_metrics(root, key, s) for s in scenes}
    ts = {s: timing_metrics(root, key, s) for s in scenes}
    return {
        "maa": {s: (ms[s] or {}).get("maa") for s in scenes},
        "inlier": avg([(ms[s] or {}).get("inlier") for s in scenes]),
        "pose": avg([(ms[s] or {}).get("median_pose") for s in scenes]),
        "matches": avg([(ts[s] or {}).get("matches") for s in scenes]),
        "pair_ms": avg([(ts[s] or {}).get("pair_ms") for s in scenes]),
        "feat_ms": avg([(ts[s] or {}).get("feat_miss_ms") for s in scenes]),
        "peak": avg([(ts[s] or {}).get("peak") for s in scenes]),
        "paths": [(ms[s] or {}).get("path") for s in scenes],
        "all_scenes": all(ms[s] for s in scenes),
    }


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def infer_desc_dim(root, key, scene):
    cache = root / "output_v2" / "feature_cache_raw" / key / scene
    for path in sorted(cache.glob("*.pt"))[:4]:
        try:
            import torch
            obj = torch.load(path, map_location="cpu")
            if hasattr(obj, "shape") and len(obj.shape) >= 1:
                return int(obj.shape[0])
            if isinstance(obj, dict) and "desc" in obj:
                return int(obj["desc"].shape[1])
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    commands = command_rows(root)
    lines = []
    lines.append("# Chapter 4 Missing Final-Protocol Evidence Report")
    lines.append("")
    lines.append("Generated from `output_v2` final raw-image protocol artifacts after the detached `screen` runs `ch4_missing_gpu0` and `ch4_missing_gpu1`. Rows with `--` were not completed or failed under the final protocol. Old `mp2000`, old-preprocessing, 1.0 px threshold, and non-raw-coordinate artifacts are not used.")
    lines.append("")
    lines.append("Protocol: raw dataset images, method-internal resolution only, original-image output coordinates, deterministic top-15000 COLMAP-covisibility pairs with at least 100 shared 3D points, UniDepthV2 depth values only, COLMAP intrinsics in calibrated mode, Sampson threshold 2.0 px, reprojection threshold 16.0 px, 1000 RANSAC iterations, 25 LO iterations, and 2048 correspondence cap.")
    lines.append("")

    lines.append("## 1. DINOv3 Grid-Sampling Diagnostic")
    grid = [
        ("A. SuperPoint bilinear sampling", 16, "original SuperPoint coordinates", "dinov3_l-8_sp_mnn_mp2048"),
        ("B. snapped SP lookup", 16, "snapped patch-center coordinates", "dinov3_l-8_gridalign_outsnapped_mnn_mp2048"),
        ("C. snapped lookup only", 16, "original SuperPoint coordinates", "dinov3_l-8_gridalign_outoriginal_mnn_mp2048"),
        ("D. dense DINOv3 patch grid", 16, "patch-center coordinates", "dinov3_l-8_dense16_mnn_mp2048"),
    ]
    rows = []
    for variant, block, coords, key in grid:
        r = row_for_key(root, key)
        maa_vals = [r["maa"][s] for s in SCENES]
        rows.append([
            variant, str(block), coords,
            fmt_maa(r["maa"]["sacre_coeur"]),
            fmt_maa(r["maa"]["reichstag"]),
            fmt_maa(r["maa"]["st_peters_square"]),
            fmt_maa(avg(maa_vals) if r["all_scenes"] else None),
            fmt_pct(r["inlier"]),
            fmt_int(r["matches"]),
            fmt_pose(r["pose"]),
            paths_cell(r["paths"]),
            command_cell(commands, key, SCENES),
        ])
    lines.append(table(["variant", "DINOv3 block", "reported coordinates", "sacre_coeur mAA@10", "reichstag mAA@10", "st_peters_square mAA@10", "average mAA@10", "inlier ratio", "average matches", "median pose error", "summary JSON paths", "exact command used"], rows))
    lines.append("")
    lines.append("Dense-grid final rows are capped deterministically by taking the top 2048 mutual-nearest-neighbor matches sorted by cosine similarity after MNN.")
    lines.append("")

    lines.append("## 2. DIFT Feature-Level And Timestep Contribution")
    feat_rows = [(261, 1), (261, 2), (0, 1), (0, 2), (0, 3), (0, 0)]
    rows = []
    for t, up in feat_rows:
        key = f"dift_t{t}_up{up}_ens8_sp_mnn_mp2048"
        r = row_for_key(root, key)
        dim = infer_desc_dim(root, key, "sacre_coeur")
        scale = {0: "approx. 1/32", 1: "approx. 1/16", 2: "approx. 1/8", 3: "approx. 1/8"}.get(up, "--")
        rows.append([
            str(t), str(up), str(dim) if dim else "--", scale,
            fmt_maa(r["maa"]["sacre_coeur"]), fmt_maa(r["maa"]["reichstag"]), fmt_maa(r["maa"]["st_peters_square"]),
            fmt_maa(avg([r["maa"][s] for s in SCENES]) if r["all_scenes"] else None),
            fmt_pct(r["inlier"]), fmt_int(r["matches"]), fmt_ms(r["pair_ms"]), fmt_int(r["peak"]),
            paths_cell(r["paths"]), command_cell(commands, key, SCENES),
        ])
    lines.append(table(["timestep", "up_ft_index", "descriptor dimension", "approx feature scale", "sacre_coeur", "reichstag", "st_peters_square", "average mAA@10", "inlier ratio", "average matches", "runtime ms/pair", "peak GPU MB", "summary JSON paths", "exact command used"], rows))
    lines.append("")

    lines.append("## 3. DIFT Timestep Sweep")
    rows = []
    for t in [0, 4, 8, 12, 16, 20, 30, 261]:
        key = f"dift_t{t}_up2_ens8_sp_mnn_mp2048"
        r = row_for_key(root, key)
        rows.append([
            str(t),
            fmt_maa(r["maa"]["sacre_coeur"]), fmt_maa(r["maa"]["reichstag"]), fmt_maa(r["maa"]["st_peters_square"]),
            fmt_maa(avg([r["maa"][s] for s in SCENES]) if r["all_scenes"] else None),
            fmt_pct(r["inlier"]), fmt_pose(r["pose"]), fmt_int(r["matches"]),
            paths_cell(r["paths"]), command_cell(commands, key, SCENES),
        ])
    lines.append(table(["timestep", "sacre_coeur mAA@10", "reichstag mAA@10", "st_peters_square mAA@10", "average mAA@10", "inlier ratio", "median pose error", "matches", "summary JSON paths", "exact command used"], rows))
    lines.append("")

    lines.append("## 4. DIFT Ensemble-Size Sweep")
    rows = []
    for ens in [1, 2, 4, 8, 16]:
        key = f"dift_t0_up2_ens{ens}_sp_mnn_mp2048"
        r = row_for_key(root, key)
        fail = failure_note(root, key, "sacre_coeur")
        rows.append([
            str(ens), fmt_maa(r["maa"]["sacre_coeur"]),
            fmt_maa(avg([r["maa"][s] for s in SCENES]) if r["all_scenes"] else None),
            fmt_pct(r["inlier"]), fmt_int(r["matches"]), fmt_ms(r["feat_ms"]), fmt_ms(r["pair_ms"]), fmt_int(r["peak"]),
            paths_cell(r["paths"][:1]) if r["paths"][0] else (f"`{fail}`" if fail else "--"),
            command_cell(commands, key, ["sacre_coeur"]),
        ])
    lines.append(table(["ensemble size", "sacre_coeur mAA@10", "average mAA@10 if all scenes", "inlier ratio", "matches", "feature extraction ms/image", "matching ms/pair", "peak GPU MB", "summary JSON path or failure log", "exact command used"], rows))
    lines.append("")

    lines.append("## 5. Ratio-Test Ablation")
    methods = [
        ("DINOv3 block 12", "dinov3_l-12_sp_mnn_mp2048", "dinov3_l-12_sp_mnn_rt{rt}_mp2048"),
        ("DIFT t=0 up2 ens8", "dift_t0_up2_ens8_sp_mnn_mp2048", "dift_t0_up2_ens8_sp_mnn_rt{rt}_mp2048"),
        ("DINOv3 block16 + DIFT fusion", "fusion_dinov3b16_dift_t0up2_sp_mnn_mp2048", "fusion_dinov3b16_dift_t0up2_sp_mnn_rt{rt}_mp2048"),
    ]
    rows = []
    for name, base_key, tmpl in methods:
        base = row_for_key(root, base_key, ["sacre_coeur"])
        rows.append([name, "MNN only", fmt_maa(base["maa"]["sacre_coeur"]), fmt_pct(base["inlier"]), fmt_int(base["matches"]), "baseline", paths_cell(base["paths"]), "--"])
        base_maa = base["maa"]["sacre_coeur"]
        for rt in RATIOS:
            key = tmpl.format(rt=rt)
            r = row_for_key(root, key, ["sacre_coeur"])
            maa = r["maa"]["sacre_coeur"]
            if maa is None or base_maa is None:
                effect = "--"
            else:
                diff = maa - base_maa
                effect = f"{'improves' if diff > 0 else 'hurts' if diff < 0 else 'ties'} ({diff:+.2f})"
            rows.append([name, f"MNN + ratio {rt}", fmt_maa(maa), fmt_pct(r["inlier"]), fmt_int(r["matches"]), effect, paths_cell(r["paths"]), command_cell(commands, key, ["sacre_coeur"])])
    lines.append(table(["method", "matching variant", "sacre_coeur mAA@10", "inlier ratio", "average matches", "pose effect vs MNN", "summary JSON paths", "exact command used"], rows))
    lines.append("")

    lines.append("## 6. Timing Sanity Check")
    for key, label in [
        ("dinov3_l-12_sp_mnn_mp2048", "DINOv3 block 12"),
        ("dift_t0_up2_ens8_sp_mnn_mp2048", "DIFT t0/up2/ens8"),
        ("fusion_dinov3b16_dift_t0up2_sp_mnn_mp2048", "Fusion block16+DIFT"),
    ]:
        tm = timing_metrics(root, key, "sacre_coeur")
        lines.append(f"- {label}: timing file `{tm.get('path', '--')}`, pair timings={tm.get('pair_count', 0)}, feature timings={tm.get('feature_count', 0)}, avg pair time={fmt_ms(tm.get('pair_ms'))} ms, avg cache-miss feature time={fmt_ms(tm.get('feat_miss_ms'))} ms/image, peak GPU={fmt_int(tm.get('peak'))} MB.")
    lines.append("")
    lines.append("Fusion timing is not directly comparable with DINOv3 or DIFT alone when the fused descriptor cache already exists. `fusion_matches.py` reports only descriptor-cache preparation and pair matching, stores no per-image feature extraction timings, and usually reuses DINOv3/DIFT feature caches. Therefore the much lower fusion time is a cache-accounting artifact, not evidence that fusion extracts features faster. Recommendation: do not report runtime in Chapter 4 unless timing is rerun from cold caches with the same accounting boundary for all methods.")
    lines.append("")

    supported = []
    unsupported = []
    if row_for_key(root, "dinov3_l-8_gridalign_outsnapped_mnn_mp2048", ["sacre_coeur"])["maa"]["sacre_coeur"] is not None:
        supported.append("DINOv3 snapped-grid and dense-grid diagnostics have final-protocol sacre_coeur evidence for block 16.")
    else:
        unsupported.append("DINOv3 snapped-grid or dense-grid diagnostics are still missing final-protocol results.")
    if row_for_key(root, "dift_t261_up1_ens8_sp_mnn_mp2048", ["sacre_coeur"])["maa"]["sacre_coeur"] is not None:
        supported.append("DIFT t=261/up1 and t=261/up2 can now be compared with t=0 rows under the final raw protocol.")
    else:
        unsupported.append("DIFT t=261 feature-level rows remain unsupported.")
    if row_for_key(root, "dift_t0_up2_ens16_sp_mnn_mp2048", ["sacre_coeur"])["maa"]["sacre_coeur"] is not None:
        supported.append("DIFT ensemble size 16 completed under the final protocol.")
    elif failure_note(root, "dift_t0_up2_ens16_sp_mnn_mp2048", "sacre_coeur"):
        supported.append("DIFT ensemble size 16 has a final-protocol failure log, so memory/runtime failure can be reported with that caveat.")
    else:
        unsupported.append("DIFT ensemble size 16 has neither metrics nor a failure log.")
    if row_for_key(root, "dift_t4_up2_ens8_sp_mnn_mp2048", ["sacre_coeur"])["maa"]["sacre_coeur"] is not None:
        supported.append("The requested DIFT low-timestep sweep includes final-protocol sacre_coeur rows.")
    else:
        unsupported.append("The requested DIFT low-timestep sweep is still incomplete.")
    if row_for_key(root, "fusion_dinov3b16_dift_t0up2_sp_mnn_rt0.90_mp2048", ["sacre_coeur"])["maa"]["sacre_coeur"] is not None:
        supported.append("The ratio-test ablation has final-protocol sacre_coeur evidence for at least the selected fusion method.")
    else:
        unsupported.append("The ratio-test ablation remains unsupported for at least one requested method.")

    lines.append("## Claims Supported By Final-Protocol Evidence")
    lines.extend(f"- {x}" for x in supported) if supported else lines.append("- No new strict claims are supported beyond the previous report.")
    lines.append("")
    lines.append("## Claims Still Unsupported")
    lines.extend(f"- {x}" for x in unsupported) if unsupported else lines.append("- No requested missing-evidence claim remains unsupported by the completed final-protocol artifacts.")
    lines.append("")
    lines.append("## Run Artifacts")
    lines.append("- GPU runner logs: `output_v2/logs/ch4_missing_gpu0.log`, `output_v2/logs/ch4_missing_gpu1.log`.")
    lines.append("- Command log: `output_v2/logs/ch4_missing_commands.tsv`.")
    lines.append("- Failure logs, if any: `output_v2/logs/*failure.log`.")
    lines.append("")

    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
