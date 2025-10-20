#!/bin/bash
set -e

# Where to output the blinded archive
OUT="supplement_for_review.tar.gz"

# Create a temp staging dir
STAGE="$(mktemp -d)"
echo "[INFO] Staging at: $STAGE"

# 1) Minimal README for reviewers
cat > "$STAGE/README.txt" <<'EOF'
Blinded Supplement (Score-Based MEWMA with Bootstrap Control Limits)

Quick check:
  1) (Optional) python -m venv .venv && source .venv/bin/activate
  2) pip install -r requirements.txt
  3) bash run.sh

This will generate outputs/mixture/t2_vs_ucl.png reproducing the key figure.

Notes:
- This archive is blinded for review.
- Code runs on CPU in a few minutes.
EOF

# 2) Minimal run script (one command)
mkdir -p "$STAGE/scripts"
cat > "$STAGE/run.sh" <<'EOF'
#!/bin/bash
set -e
# Ensure outputs folder exists
mkdir -p outputs/mixture
# Run the main reproduction
python examples/mixture/run_mixture.py
echo "[OK] Reproduction complete. See: outputs/mixture/t2_vs_ucl.png"
EOF
chmod +x "$STAGE/run.sh"

# 3) Requirements
cp requirements.txt "$STAGE/requirements.txt"

# 4) Minimal source required to run the linear example
mkdir -p "$STAGE/src/driftcl"
cp src/driftcl/__init__.py "$STAGE/src/driftcl/__init__.py"
cp src/driftcl/mewma.py      "$STAGE/src/driftcl/mewma.py"
cp src/driftcl/bootstrap_cl.py "$STAGE/src/driftcl/bootstrap_cl.py"
cp src/driftcl/monitoring.py "$STAGE/src/driftcl/monitoring.py"
cp src/driftcl/utils.py      "$STAGE/src/driftcl/utils.py"

# 5) Example script to reproduce the key figure
mkdir -p "$STAGE/examples/mixture"
cp examples/mixture/run_mixture.py "$STAGE/examples/mixture/run_mixture.py"

# 6) (Optional but nice) Include a pre-generated output for immediate recognition
mkdir -p "$STAGE/outputs/mixture"
if [ -f outputs/mixture/t2_vs_ucl.png ]; then
  cp outputs/mixture/t2_vs_ucl.png "$STAGE/outputs/mixture/t2_vs_ucl.png"
fi

# 7) Pack it up
tar -czf "$OUT" -C "$STAGE" .
echo "[OK] Wrote $OUT"

# 8) Cleanup temp staging dir
rm -rf "$STAGE"
