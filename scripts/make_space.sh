#!/usr/bin/env bash
# Package the Streamlit demo as a Hugging Face Space (CPU basic is enough).
#   scripts/make_space.sh build/space
#   huggingface-cli upload <your-user>/picklabel build/space . --repo-type space
# A Space needs its own README front matter, so it is built here instead of living in the repo README.
set -euo pipefail
OUT=${1:-build/space}
rm -rf "$OUT" && mkdir -p "$OUT"
rsync -a --exclude "__pycache__" picklabel app configs examples "$OUT"/
mkdir -p "$OUT/report" && cp -r report/autolabel report/results.md "$OUT/report/" 2>/dev/null || true
cat > "$OUT/README.md" <<'EOF'
---
title: PickLabel
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app/streamlit_app.py
pinned: false
license: mit
---
Auto-label a robot pick from its before/after frames and grasp point.
Code, method and results: https://github.com/Waqar53/picklable
EOF
cat > "$OUT/requirements.txt" <<'EOF'
numpy
opencv-python-headless
scikit-image
scipy
pyyaml
pillow
matplotlib
tqdm
EOF
echo "Space ready in $OUT"
