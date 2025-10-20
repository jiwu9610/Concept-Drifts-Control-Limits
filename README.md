
# Bootstrapped Control Limits for Score-Based Concept Drift Detection

This repository provides a **reproducible implementation** of the control-limit calibration method proposed in the accompanying manuscript  
**“Control Limits for Concept Drift Detection Based on Score Vectors and Bootstrap Estimation” (blinded)**.

The implementation reproduces key empirical results from the paper — specifically, the **score-based MEWMA chart** with **nested bootstrap control limits** and **0.632-style correction** on two representative examples:
1. **Linear mixture model** (ridge regression scores)
2. **Nonlinear oscillator** (neural network last-layer scores)

Both examples can be reproduced with a single command.

---

## 🧱 Repository Structure

```

score-drift-bootstrap-CL/
├─ README.md
├─ requirements.txt
├─ src/
│  └─ driftcl/
│     ├─ **init**.py
│     ├─ mewma.py              # MEWMA update and T² statistic
│     ├─ bootstrap_cl.py       # Algorithm 1: nested bootstrap + 0.632 correction
│     ├─ scores.py             # Per-observation score extraction (ridge, NN)
│     ├─ monitoring.py         # End-to-end drift monitoring pipeline
│     └─ utils.py              # Regularization, RNG, helpers
├─ examples/
│  ├─ mixture/
│  │  ├─ mixture_final.py      # Linear example (uploaded script)
│  │  └─ run_mixture.py        # One-command reproduction of Figure 2
│  └─ oscillator/
│     ├─ oscilator_final.py    # Nonlinear example (uploaded script)
│     └─ run_oscillator.py     # Reproduce key nonlinear figure
├─ outputs/                    # Figures and tables (generated)
└─ scripts/
├─ reproduce_linear.sh
├─ reproduce_oscillator.sh
└─ make_supplement.tar.sh

````

---

## 🚀 Quick Start (Reproduce a Key Figure)

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

### 2. Run the main example (linear mixture)

```bash
bash scripts/reproduce_linear.sh
```

This script:

* Generates synthetic data (Eqs. 4.1–4.2)
* Fits the baseline ridge model and extracts score vectors
* Computes **time-varying bootstrap control limits**
* Streams post-shift data and produces a **T² vs. UCL plot**

Output is saved in:

```
outputs/mixture/t2_vs_ucl.png
```

You should immediately recognize the figure reproduced from the paper (similar to Figure 2).

---

## 🧩 Python API Example

```python
from driftcl.bootstrap_cl import compute_control_limits
from driftcl.monitoring import monitor_stream
from sklearn.linear_model import Ridge

# Step 1: fit model and compute bootstrap-based control limits
cl = compute_control_limits(
    X_train, y_train,
    model=Ridge(alpha=0.1),
    alpha=1e-3, lambda_=0.01,
    BO=60, BI=120, epsilon_reg=1e-6
)

# Step 2: monitor streaming data
results = monitor_stream(X_stream, y_stream, fitted_model, cl, lambda_=0.01)
```

`results` includes:

* `z_i`: MEWMA vectors
* `T2_i`: test statistics
* `CL_i`: time-varying control limits
* `signal_index`: first detection time (if any)

---

## 📊 Included Examples

| Example                  | Description                                         | Model Type | Figure/Table |
| ------------------------ | --------------------------------------------------- | ---------- | ------------ |
| **Linear Mixture**       | Mean shift in a Gaussian mixture (ridge regression) | Ridge      | Figure 2     |
| **Nonlinear Oscillator** | Dynamical system with neural network model          | MLP        | Figure 4     |

The linear example is the **primary reproducibility target** (required by journal policy), and runs in ≈3 minutes on CPU.

---

## ⚙️ Hyperparameters (Default for Review)

| Parameter    | Meaning                        | Default  |
| ------------ | ------------------------------ | -------- |
| `α`          | Type-I error level             | 1e-3     |
| `λ`          | MEWMA smoothing                | 0.01     |
| `B_O`, `B_I` | Outer / inner bootstrap loops  | 60 / 120 |
| `ε_reg`      | Ridge regularization in Σ      | 1e-6     |
| `γ`          | Ridge penalty (linear example) | 0.1      |

---

## 📦 Supplementary Archive for Review

A single blinded archive can be created via:

```bash
bash scripts/make_supplement.tar.sh
```

This produces:

```
supplement_for_review.tar.gz
```

containing:

* Minimal source code
* `run.sh` (one-command reproduction)
* Output figure and CSV table
* Short README.txt

This ensures compliance with *Technometrics* reproducibility policy:

> “Source a file without modification, wait, and immediately recognize an important figure summarizing results of a challenging example.”

---

## 🧠 Citation

If you use this code, please cite the corresponding manuscript:

> *“Control Limits for Concept Drift Detection Based on Score Vectors and Bootstrap Estimation” (under review).*

---

## 🪄 License

Released under the **MIT License** (blinded).





