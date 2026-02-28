#!/usr/bin/env python3
"""
Compare pointwise FAR/PD curves for our bootstrap method vs Zhang et al. (two-phase)
on the nonlinear oscillator example at two noise levels.
"""
import argparse
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 20,
        "legend.fontsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    }
)


def load_curve(path):
    d = np.load(path, allow_pickle=True)
    pfar = d["pfar"]
    n_ic = int(d["n_ic"]) if "n_ic" in d else None
    n_oc = int(d["n_oc"]) if "n_oc" in d else None
    return pfar, n_ic, n_oc


def plot_compare(noise, ours_path, kg_path, outdir, alpha_nom=0.001):
    ours, n_ic_ours, n_oc_ours = load_curve(ours_path)
    kg, n_ic_kg, n_oc_kg = load_curve(kg_path)

    n_ic = n_ic_ours or n_ic_kg or 200
    M = min(len(ours), len(kg))
    xs = np.arange(M)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(xs, kg[:M], ls="--", lw=2, color="#d95f02", label="Zhang et al.'s method")
    ax.semilogy(xs, ours[:M], ls="-", lw=2, color="#1f77b4", label="Our bootstrap method")
    ax.axhline(alpha_nom, ls="--", lw=2, color="#7570b3", label=r"Nominal FAR $\alpha=0.001$")
    ax.axvline(n_ic - 0.5, color="k", lw=1.3, ls="--", label="change-point")

    ax.set_xlabel("Sample index i")
    ax.set_ylabel("PFAR / POD (log scale)")
    ax.set_title(
        f"Pointwise FAR and Probability of Detection Comparison (Noise = {noise})",
        pad=8,
    )
    ax.set_ylim(1e-5, 1e0)
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"compare_noise{noise}.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Wrote {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, nargs="+", default=[0.03, 0.25])
    ap.add_argument("--ours-dir", default="outputs/pfar_quick")
    ap.add_argument("--kg-dir", default="outputs/kungang_nonlinear")
    ap.add_argument("--outdir", default="outputs/compare_far")
    args = ap.parse_args()

    for noise in args.noise:
        ours_path = os.path.join(args.ours_dir, f"pfar_noise{noise}.npz")
        kg_path = os.path.join(args.kg_dir, f"kungang_nonlinear_noise{noise}.npz")
        if not os.path.exists(ours_path):
            raise FileNotFoundError(f"Missing ours npz: {ours_path}")
        if not os.path.exists(kg_path):
            raise FileNotFoundError(f"Missing Kungang npz: {kg_path}")
        plot_compare(noise, ours_path, kg_path, args.outdir, alpha_nom=0.001)


if __name__ == "__main__":
    main()
