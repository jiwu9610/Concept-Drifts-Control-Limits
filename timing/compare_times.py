#!/usr/bin/env python3
import argparse
import os
import numpy as np


def load_npz(path):
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


def fmt(sec):
    if sec is None:
        return "n/a"
    return f"{sec:.1f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, nargs="+", default=[0.03, 0.20])
    ap.add_argument("--ours-serial-dir", default="outputs/pfar_quick")
    ap.add_argument("--ours-parallel-dir", default="outputs/nonlinear_osc")
    ap.add_argument("--kg-dir", default="outputs/kungang_nonlinear")
    args = ap.parse_args()

    print("Timing comparison (wall-clock where available):")
    print("Columns: noise | ours-serial bootstrap | ours-serial pfar | ours-parallel serial-est | ours-parallel parallel-est | kungang fit | kungang pfar")

    for noise in args.noise:
        ours_serial = load_npz(os.path.join(args.ours_serial_dir, f"pfar_noise{noise}.npz"))
        ours_parallel = load_npz(os.path.join(args.ours_parallel_dir, "ucl_and_chartparams.npz"))
        kg = load_npz(os.path.join(args.kg_dir, f"kungang_nonlinear_noise{noise}.npz"))

        t_boot = float(ours_serial["t_boot_wall"]) if ours_serial is not None and "t_boot_wall" in ours_serial else None
        t_pfar = float(ours_serial["t_pfar_wall"]) if ours_serial is not None and "t_pfar_wall" in ours_serial else None

        t_serial_est = float(ours_parallel["t_serial_outer"]) if ours_parallel is not None and "t_serial_outer" in ours_parallel else None
        t_parallel_est = float(ours_parallel["t_parallel_outer_est"]) if ours_parallel is not None and "t_parallel_outer_est" in ours_parallel else None

        t_kg_fit = float(kg["t_fit_wall"]) if kg is not None and "t_fit_wall" in kg else None
        t_kg_pfar = float(kg["t_pfar_wall"]) if kg is not None and "t_pfar_wall" in kg else None

        print(
            f"{noise} | {fmt(t_boot)} | {fmt(t_pfar)} | {fmt(t_serial_est)} | {fmt(t_parallel_est)} | {fmt(t_kg_fit)} | {fmt(t_kg_pfar)}"
        )


if __name__ == "__main__":
    main()
