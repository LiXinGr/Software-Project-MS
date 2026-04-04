#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


START_EPOCH_RE = re.compile(r"Starting epoch (\d+)")
TRAIN_RE = re.compile(r"\[E (\d+) \| it (\d+)\] loss \{(.+)\}")
VAL_RE = re.compile(r"\[Validation\] \{(.+)\}")


def parse_metrics(blob: str) -> dict[str, float]:
    metrics = {}
    for part in blob.split(", "):
        if " " not in part:
            continue
        key, value = part.rsplit(" ", 1)
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def aggregate_log(log_path: Path) -> list[dict[str, float | int]]:
    train_metrics = defaultdict(lambda: defaultdict(list))
    val_last = {}
    val_best = {}
    val_counts = defaultdict(int)
    current_epoch = None

    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if match := START_EPOCH_RE.search(line):
                current_epoch = int(match.group(1))
                continue

            if match := TRAIN_RE.search(line):
                epoch = int(match.group(1))
                metrics = parse_metrics(match.group(3))
                for key, value in metrics.items():
                    train_metrics[epoch][key].append(value)
                continue

            if match := VAL_RE.search(line):
                if current_epoch is None:
                    continue
                metrics = parse_metrics(match.group(1))
                val_last[current_epoch] = metrics
                val_counts[current_epoch] += 1
                best = val_best.get(current_epoch)
                if best is None or metrics.get("loss/total", float("inf")) < best.get(
                    "loss/total", float("inf")
                ):
                    val_best[current_epoch] = metrics

    all_epochs = sorted(
        set(train_metrics.keys()) | set(val_last.keys()) | set(val_best.keys())
    )
    rows = []
    for epoch in all_epochs:
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_logs": len(next(iter(train_metrics[epoch].values()), [])),
            "val_updates": val_counts.get(epoch, 0),
        }
        for key, values in sorted(train_metrics[epoch].items()):
            if values:
                row[f"train_mean/{key}"] = sum(values) / len(values)
        for prefix, metrics_by_epoch in [("val_last", val_last), ("val_best", val_best)]:
            metrics = metrics_by_epoch.get(epoch, {})
            for key, value in sorted(metrics.items()):
                row[f"{prefix}/{key}"] = value
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, float | int]], output_csv: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(rows: list[dict[str, float | int]], output_png: Path | None) -> None:
    if output_png is None:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for --output-png: {exc}") from exc

    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    plots = [
        ("train_mean/total", "Train Total"),
        ("val_last/loss/total", "Val Loss/Total"),
        ("val_last/accuracy", "Val Accuracy"),
        ("val_last/average_precision", "Val Average Precision"),
    ]
    for ax, (key, title) in zip(axes.flat, plots):
        ys = [row.get(key) for row in rows]
        valid = [(x, y) for x, y in zip(epochs, ys) if y is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, marker="o", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def maybe_write_tensorboard(
    rows: list[dict[str, float | int]], tensorboard_dir: Path | None
) -> None:
    if tensorboard_dir is None:
        return

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    skip_keys = {"epoch", "train_logs", "val_updates"}
    for row in rows:
        step = int(row["epoch"])
        for key, value in row.items():
            if key in skip_keys:
                continue
            if isinstance(value, (int, float)):
                writer.add_scalar(f"epoch/{key}", float(value), step)
    writer.flush()
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export epoch-level curves from a Glue Factory training log."
    )
    parser.add_argument("--log", required=True, type=Path, help="Path to log.txt")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-png", type=Path)
    parser.add_argument("--tensorboard-dir", type=Path)
    args = parser.parse_args()

    rows = aggregate_log(args.log)
    if not rows:
        raise SystemExit(f"No epoch data found in {args.log}")
    write_csv(rows, args.output_csv)
    maybe_plot(rows, args.output_png)
    maybe_write_tensorboard(rows, args.tensorboard_dir)
    print(f"Wrote {len(rows)} epoch rows to {args.output_csv}")
    if args.output_png:
        print(f"Wrote plot to {args.output_png}")
    if args.tensorboard_dir:
        print(f"Wrote TensorBoard events to {args.tensorboard_dir}")


if __name__ == "__main__":
    main()
