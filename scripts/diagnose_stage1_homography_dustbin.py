from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GF_ROOT = PROJECT_ROOT / "external" / "glue-factory"

for path in (PROJECT_ROOT, GF_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from gluefactory.datasets import get_dataset  # noqa: E402
from gluefactory.models import get_model  # noqa: E402
from gluefactory.utils.tensor import batch_to_device  # noqa: E402


def build_model_conf(conf):
    model_conf = OmegaConf.create(OmegaConf.to_container(conf.model, resolve=True))
    model_conf.matcher = {"name": None}
    return model_conf


def format_ratio(num: int, den: int) -> float:
    return float(num) / float(max(den, 1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure matched/unmatched/ignored keypoint fractions for Stage 1 homography training."
    )
    parser.add_argument("--conf", type=Path, required=True)
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_args()

    conf = OmegaConf.load(args.conf)
    overrides = OmegaConf.from_cli(args.dotlist)
    conf = OmegaConf.merge(conf, overrides)
    if conf.data.get("num_workers", None) == 0:
        conf.data.prefetch_factor = None

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(conf.data.name)(conf.data)
    loader = dataset.get_data_loader("train")
    model_conf = build_model_conf(conf)
    model = get_model(model_conf.name)(model_conf).to(device)
    model.eval()

    sum_total0 = 0
    sum_total1 = 0
    sum_matched0 = 0
    sum_matched1 = 0
    sum_unmatched0 = 0
    sum_unmatched1 = 0
    sum_ignored0 = 0
    sum_ignored1 = 0
    sum_assignment = 0
    processed = 0

    with torch.inference_mode():
        for batch_idx, data in enumerate(loader, start=1):
            if batch_idx > args.num_batches:
                break

            data = batch_to_device(data, device, non_blocking=True)
            pred0 = model.extract_view(data, "0")
            pred1 = model.extract_view(data, "1")
            pred = {
                **{k + "0": v for k, v in pred0.items()},
                **{k + "1": v for k, v in pred1.items()},
            }
            gt = model.ground_truth({**data, **pred})

            matches0 = gt["matches0"]
            matches1 = gt["matches1"]
            assignment = gt["assignment"]

            total0 = int(matches0.numel())
            total1 = int(matches1.numel())
            matched0 = int((matches0 >= 0).sum().item())
            matched1 = int((matches1 >= 0).sum().item())
            unmatched0 = int((matches0 == -1).sum().item())
            unmatched1 = int((matches1 == -1).sum().item())
            ignored0 = int((matches0 == -2).sum().item())
            ignored1 = int((matches1 == -2).sum().item())
            assignment_count = int(assignment.sum().item())

            sum_total0 += total0
            sum_total1 += total1
            sum_matched0 += matched0
            sum_matched1 += matched1
            sum_unmatched0 += unmatched0
            sum_unmatched1 += unmatched1
            sum_ignored0 += ignored0
            sum_ignored1 += ignored1
            sum_assignment += assignment_count
            processed += 1

            print(
                f"[DUSTBIN] batch={batch_idx} "
                f"view0 total={total0} matched={matched0} unmatched={unmatched0} ignored={ignored0} "
                f"unmatched_frac={format_ratio(unmatched0, total0):.4f} "
                f"nonignored_unmatched_frac={format_ratio(unmatched0, total0 - ignored0):.4f} | "
                f"view1 total={total1} matched={matched1} unmatched={unmatched1} ignored={ignored1} "
                f"unmatched_frac={format_ratio(unmatched1, total1):.4f} "
                f"nonignored_unmatched_frac={format_ratio(unmatched1, total1 - ignored1):.4f} | "
                f"gt_pairs={assignment_count}",
                flush=True,
            )

    if processed == 0:
        raise RuntimeError("No batches were processed.")

    print(
        "[DUSTBIN-SUMMARY] "
        f"batches={processed} "
        f"view0_total={sum_total0} view0_matched={sum_matched0} view0_unmatched={sum_unmatched0} view0_ignored={sum_ignored0} "
        f"view0_unmatched_frac={format_ratio(sum_unmatched0, sum_total0):.4f} "
        f"view0_nonignored_unmatched_frac={format_ratio(sum_unmatched0, sum_total0 - sum_ignored0):.4f} | "
        f"view1_total={sum_total1} view1_matched={sum_matched1} view1_unmatched={sum_unmatched1} view1_ignored={sum_ignored1} "
        f"view1_unmatched_frac={format_ratio(sum_unmatched1, sum_total1):.4f} "
        f"view1_nonignored_unmatched_frac={format_ratio(sum_unmatched1, sum_total1 - sum_ignored1):.4f} | "
        f"gt_pairs={sum_assignment}",
        flush=True,
    )


if __name__ == "__main__":
    main()
