#!/usr/bin/env bash
# Run after pipeline completes to generate all paper content
# Usage: bash scripts/finalize_results.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "[$(date)] Starting result finalization..."

# Wait for pipeline to be done
if [ ! -f results/.pipeline_done ]; then
    echo 'Pipeline not done yet. Waiting...'
    while [ ! -f results/.pipeline_done ]; do
        sleep 60
    done
fi

echo "[$(date)] Pipeline done. Generating paper content..."

# Generate visualizations
python scripts/collect_and_visualize.py --results_dir results --output_dir to_human

# Generate paper-ready tables and figures
python scripts/generate_paper_content.py --results_dir results --output_dir paper/generated

echo "[$(date)] Finalization complete!"
echo 'Generated files:'
ls -la paper/generated/
ls -la to_human/
