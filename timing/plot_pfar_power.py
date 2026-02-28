#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Aggregate npz with pfar array")
    ap.add_argument("--out", required=True, help="Output png path")
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--n-ic", type=int, default=200)
    ap.add_argument("--title", default="Pointwise False Alarm Rate")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    pfar = d["pfar"]
    idx = np.arange(len(pfar))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(idx, pfar, lw=2, label="Empirical PFAR / Power")
    ax.axhline(args.alpha, ls="--", lw=2, color="#d95f02", label=f"alpha={args.alpha}")
    ax.axvline(args.n_ic - 0.5, color="k", lw=1.3, ls="--", label="change-point")
    ax.set_xlabel("Sample index i")
    ax.set_ylabel("Rate (log scale)")
    ax.set_title(args.title)
    ax.set_ylim(1e-5, 1e0)
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
