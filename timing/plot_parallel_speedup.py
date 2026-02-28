#!/usr/bin/env python3
import argparse
import glob
import json
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_wall_span(outdir):
    paths = sorted(glob.glob(os.path.join(outdir, "clock_task*.json")))
    if not paths:
        return None
    starts, ends = [], []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        starts.append(float(d["wall_start"]))
        ends.append(float(d["wall_end"]))
    return max(ends) - min(starts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdirs", nargs="+", required=True, help="List of outdirs for different n_tasks")
    ap.add_argument("--labels", nargs="+", required=True, help="Labels (e.g., n_tasks) matching outdirs")
    ap.add_argument("--out", default="outputs/parallel_speedup.png")
    args = ap.parse_args()

    if len(args.outdirs) != len(args.labels):
        raise ValueError("outdirs and labels must be same length")

    spans = []
    for d in args.outdirs:
        span = load_wall_span(d)
        if span is None:
            raise FileNotFoundError(f"No clock_task*.json in {d}")
        spans.append(span)

    labels = args.labels
    times = np.array(spans)
    speedup = times.min() / times

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(labels, times / 60.0, marker="o", lw=2)
    ax.set_xlabel("Parallel configuration")
    ax.set_ylabel("Wall-clock time (min)")
    ax.set_title("Parallel Wall-Clock Scaling (Outer Bootstrap)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=200)
    plt.close(fig)

    out_csv = os.path.splitext(args.out)[0] + ".csv"
    np.savetxt(out_csv, np.c_[labels, times, speedup], fmt="%s", delimiter=",", header="label,wall_seconds,speedup", comments="")
    print(f"Wrote {args.out} and {out_csv}")


if __name__ == "__main__":
    main()
