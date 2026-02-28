#!/usr/bin/env python3
import argparse
import glob
import json
import os
from typing import Dict, Optional

import numpy as np


def load_npz(path: str):
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


def read_clock_dir(outdir: str) -> Optional[Dict[str, float]]:
    paths = sorted(glob.glob(os.path.join(outdir, "clock_task*.json")))
    if not paths:
        return None
    starts, ends, durations = [], [], []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        starts.append(float(d["wall_start"]))
        ends.append(float(d["wall_end"]))
        durations.append(float(d["wall_seconds"]))
    return {
        "wall_min_start": float(min(starts)),
        "wall_max_end": float(max(ends)),
        "wall_span": float(max(ends) - min(starts)),
        "wall_sum": float(sum(durations)),
        "n_tasks": float(len(paths)),
    }


def fmt(sec: Optional[float]) -> str:
    if sec is None:
        return "n/a"
    return f"{sec:.1f}"


def fmt_min(sec: Optional[float]) -> str:
    if sec is None:
        return "n/a"
    return f"{sec/60.0:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, nargs="+", default=[0.03, 0.20])
    ap.add_argument("--ours-serial-dir", default="outputs/pfar_quick")
    ap.add_argument("--ours-parallel-root", default="outputs")
    ap.add_argument("--ours-parallel-prefix", default="nonlinear_osc_noise")
    ap.add_argument("--kg-dir", default="outputs/kungang_nonlinear")
    ap.add_argument("--out-csv", default="outputs/compare_times.csv")
    ap.add_argument("--out-tex", default="outputs/compare_times.tex")
    ap.add_argument("--out-md", default="outputs/compare_times.md")
    args = ap.parse_args()

    rows = []
    for noise in args.noise:
        ours_serial = load_npz(os.path.join(args.ours_serial_dir, f"pfar_noise{noise}.npz"))
        ours_parallel_dir = os.path.join(args.ours_parallel_root, f"{args.ours_parallel_prefix}{noise}")
        ours_parallel = load_npz(os.path.join(ours_parallel_dir, "ucl_and_chartparams.npz"))
        chunk_paths = sorted(glob.glob(os.path.join(ours_parallel_dir, "chunk_task*.npz")))
        kg = load_npz(os.path.join(args.kg_dir, f"kungang_nonlinear_noise{noise}.npz"))

        t_boot = float(ours_serial["t_boot_wall"]) if ours_serial is not None and "t_boot_wall" in ours_serial else None
        t_pfar = float(ours_serial["t_pfar_wall"]) if ours_serial is not None and "t_pfar_wall" in ours_serial else None
        t_base = float(ours_serial["t_base_fit"]) if ours_serial is not None and "t_base_fit" in ours_serial else None

        t_serial_est = float(ours_parallel["t_serial_outer"]) if ours_parallel is not None and "t_serial_outer" in ours_parallel else None
        t_parallel_est = float(ours_parallel["t_parallel_outer_est"]) if ours_parallel is not None and "t_parallel_outer_est" in ours_parallel else None

        clock = read_clock_dir(ours_parallel_dir)
        t_parallel_wall = float(clock["wall_span"]) if clock is not None else None

        t_parallel_best = None
        if chunk_paths:
            all_t = []
            for p in chunk_paths:
                d = np.load(p, allow_pickle=True)
                all_t.append(d["t_total"])
            all_t = np.concatenate(all_t)
            t_parallel_best = float(np.max(all_t))

        t_kg_fit = float(kg["t_fit_wall"]) if kg is not None and "t_fit_wall" in kg else None
        t_kg_pfar = float(kg["t_pfar_wall"]) if kg is not None and "t_pfar_wall" in kg else None

        rows.append(
            dict(
                noise=noise,
                ours_serial_boot_s=t_boot,
                ours_serial_boot_min=t_boot / 60.0 if t_boot is not None else None,
                ours_serial_pfar_s=t_pfar,
                ours_serial_pfar_min=t_pfar / 60.0 if t_pfar is not None else None,
                ours_serial_base_fit_s=t_base,
                ours_parallel_serial_est_s=t_serial_est,
                ours_parallel_serial_est_min=t_serial_est / 60.0 if t_serial_est is not None else None,
                ours_parallel_parallel_est_s=t_parallel_est,
                ours_parallel_parallel_est_min=t_parallel_est / 60.0 if t_parallel_est is not None else None,
                ours_parallel_wall_s=t_parallel_wall,
                ours_parallel_wall_min=t_parallel_wall / 60.0 if t_parallel_wall is not None else None,
                ours_parallel_best_s=t_parallel_best,
                ours_parallel_best_min=t_parallel_best / 60.0 if t_parallel_best is not None else None,
                kg_fit_s=t_kg_fit,
                kg_fit_min=t_kg_fit / 60.0 if t_kg_fit is not None else None,
                kg_pfar_s=t_kg_pfar,
                kg_pfar_min=t_kg_pfar / 60.0 if t_kg_pfar is not None else None,
            )
        )

    # CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    headers = list(rows[0].keys()) if rows else []
    with open(args.out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join("" if r[h] is None else str(r[h]) for h in headers) + "\n")

    # Markdown table
    if rows:
        md_lines = []
        md_lines.append("| noise | ours serial (min) | ours parallel est (min) | ours parallel best (min) | Zhang et al. fit (min) |")
        md_lines.append("|---|---:|---:|---:|---:|")
        for r in rows:
            md_lines.append(
                f"| {r['noise']} | {fmt_min(r['ours_serial_boot_s'])} | {fmt_min(r['ours_parallel_parallel_est_s'])} | {fmt_min(r['ours_parallel_best_s'])} | {fmt_min(r['kg_fit_s'])} |"
            )
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

    # LaTeX table (simple)
    if rows:
        tex_lines = []
        tex_lines.append("\\begin{tabular}{lrrrr}")
        tex_lines.append("\\toprule")
        tex_lines.append("Noise $\\sigma$ & Ours--Serial (CL) & Ours--Parallel (est) & Ours--Parallel (best) & Zhang et al.\\ (fit) \\\\")
        tex_lines.append("\\midrule")
        for r in rows:
            tex_lines.append(
                f"{r['noise']} & {fmt_min(r['ours_serial_boot_s'])} & {fmt_min(r['ours_parallel_parallel_est_s'])} & {fmt_min(r['ours_parallel_best_s'])} & {fmt_min(r['kg_fit_s'])} \\\\"
            )
        tex_lines.append("\\bottomrule")
        tex_lines.append("\\end{tabular}")
        os.makedirs(os.path.dirname(args.out_tex) or ".", exist_ok=True)
        with open(args.out_tex, "w", encoding="utf-8") as f:
            f.write("\n".join(tex_lines) + "\n")

    print(f"Wrote {args.out_csv}, {args.out_md}, {args.out_tex}")


if __name__ == "__main__":
    main()
