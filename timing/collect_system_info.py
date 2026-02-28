#!/usr/bin/env python3
import argparse
import json
import os
import platform
import subprocess
from datetime import datetime


def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    args = ap.parse_args()

    info = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "uname": " ".join(platform.uname()),
        "cpu_lscpu": run("lscpu"),
        "gpu_nvidia_smi": run("nvidia-smi"),
        "conda_env": run("echo $CONDA_DEFAULT_ENV"),
        "pip_freeze": run("python -m pip freeze | head -n 200"),
        "torch_version": run("python - <<'PY'\nimport torch\nprint(torch.__version__)\nprint('cuda', torch.cuda.is_available())\nprint('cuda_version', torch.version.cuda)\nPY"),
    }

    os.makedirs(args.outdir, exist_ok=True)
    json_path = os.path.join(args.outdir, "system_info.json")
    txt_path = os.path.join(args.outdir, "system_info.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        for k, v in info.items():
            f.write(f"{k}:\n{v}\n\n")

    print(f"Wrote {json_path} and {txt_path}")


if __name__ == "__main__":
    main()
