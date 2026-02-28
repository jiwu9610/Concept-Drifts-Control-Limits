#!/usr/bin/env python3
import argparse
import glob
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Directory containing pfar_power_task*.npz")
    ap.add_argument("--outfile-base", default="pfar_power_agg", help="Base name for outputs")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.outdir, "pfar_power_task*.npz")))
    if not paths:
        raise FileNotFoundError(f"No pfar_power_task*.npz found in {args.outdir}")

    exceed_total = None
    R_total = 0
    n_ic = None
    n_oc = None
    for p in paths:
        d = np.load(p, allow_pickle=True)
        exceed = d["exceed"]
        R_chunk = int(d["R_chunk"])
        if exceed_total is None:
            exceed_total = np.zeros_like(exceed, dtype=np.int64)
        exceed_total += exceed.astype(np.int64)
        R_total += R_chunk
        if n_ic is None and "n_ic" in d:
            n_ic = int(d["n_ic"])
        if n_oc is None and "n_oc" in d:
            n_oc = int(d["n_oc"])

    pfar = exceed_total / max(R_total, 1)

    base = os.path.join(args.outdir, args.outfile_base)
    np.savez(base + ".npz", pfar=pfar, exceed=exceed_total, R_total=R_total, n_ic=n_ic, n_oc=n_oc)
    np.savetxt(base + ".csv", np.c_[np.arange(len(pfar)), pfar], delimiter=",", header="i,PFAR", comments="")

    print(f"Wrote {base}.npz, {base}.csv (R_total={R_total})")


if __name__ == "__main__":
    main()
