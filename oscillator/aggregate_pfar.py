#!/usr/bin/env python3
import argparse
import glob
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Directory containing pfar_task*.npz")
    ap.add_argument("--alpha", type=float, default=0.001, help="Nominal FAR line")
    ap.add_argument("--outfile-base", default="pfar_agg", help="Base name for outputs")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.outdir, "pfar_task*.npz")))
    if not paths:
        raise FileNotFoundError(f"No pfar_task*.npz found in {args.outdir}")

    exceed_total = None
    R_total = 0
    for p in paths:
        d = np.load(p, allow_pickle=True)
        exceed = d["exceed"]
        R_chunk = int(d["R_chunk"])
        if exceed_total is None:
            exceed_total = np.zeros_like(exceed, dtype=np.int64)
        exceed_total += exceed.astype(np.int64)
        R_total += R_chunk

    pfar = exceed_total / max(R_total, 1)

    base = os.path.join(args.outdir, args.outfile_base)
    np.savez(base + ".npz", pfar=pfar, exceed=exceed_total, R_total=R_total)
    np.savetxt(base + ".csv", np.c_[np.arange(len(pfar)), pfar], delimiter=",", header="i,PFAR", comments="")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(np.arange(len(pfar)), pfar, lw=2, label="Empirical PFAR")
    ax.axhline(args.alpha, ls="--", lw=2, color="#d95f02", label=f"alpha={args.alpha}")
    ax.set_xlabel("Sample index i")
    ax.set_ylabel("PFAR (log scale)")
    ax.set_title("Pointwise False Alarm Rate (Aggregated)")
    ax.set_ylim(1e-5, 1e0)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    plt.tight_layout()
    plt.savefig(base + ".png", dpi=200)
    plt.close(fig)

    print(f"Wrote {base}.npz, {base}.csv, {base}.png (R_total={R_total})")


if __name__ == "__main__":
    main()
