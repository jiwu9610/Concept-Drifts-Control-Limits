#!/usr/bin/env python3
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    chunk_paths = sorted(glob.glob(os.path.join(args.outdir, "chunk_task*.npz")))
    assert chunk_paths, f"No chunks found in {args.outdir}"

    # load and concatenate res + timings
    res_list = []
    t_total_all = []
    t_total_per_chunk = []
    meta0 = None
    s_bar = None
    Sigma_inv = None
    for p in chunk_paths:
        d = np.load(p, allow_pickle=True)
        res_list.append(d["res"])
        t_total = d["t_total"]
        t_total_all.append(t_total)
        t_total_per_chunk.append(float(np.sum(t_total)))
        meta = json.loads(str(d["meta"]))
        if meta0 is None:
            meta0 = meta
        if os.path.basename(p) == "chunk_task000.npz":
            s_bar = d["s_bar"]
            Sigma_inv = d["Sigma_inv"]

    if s_bar is None or Sigma_inv is None:
        # fall back to first chunk if task000 missing
        d0 = np.load(chunk_paths[0], allow_pickle=True)
        s_bar = d0["s_bar"]
        Sigma_inv = d0["Sigma_inv"]
        print("Warning: chunk_task000.npz not found; using first chunk for s_bar/Sigma_inv.")

    res = np.vstack(res_list)
    t_total_all = np.concatenate(t_total_all)

    perc = float(meta0["perc"])
    UCL = np.percentile(res, perc, axis=0)

    # serial estimate for outer loop only (sum of per-b totals, across all tasks)
    T_serial_outer = float(t_total_all.sum())
    T_parallel_outer_est = float(np.max(t_total_per_chunk)) if t_total_per_chunk else 0.0

    outpath = os.path.join(args.outdir, "ucl_and_chartparams.npz")
    np.savez(
        outpath,
        UCL=UCL,
        s_bar=s_bar,
        Sigma_inv=Sigma_inv,
        t_total_outer=t_total_all,
        t_total_outer_per_chunk=np.array(t_total_per_chunk, dtype=float),
        t_serial_outer=T_serial_outer,
        t_parallel_outer_est=T_parallel_outer_est,
        meta=json.dumps(meta0),
    )

    print(f"Wrote {outpath}")
    print(f"Bootstrap paths total: {res.shape[0]} (= B_outer * B_inner)")
    print(f"T_serial_outer_est (sum over outer b totals) = {T_serial_outer/60:.1f} min")
    if t_total_per_chunk:
        print(f"T_parallel_outer_est (max chunk sum) = {T_parallel_outer_est/60:.1f} min")
    print(f"Median t_total per outer replicate = {np.median(t_total_all):.2f} s")


if __name__ == "__main__":
    main()
