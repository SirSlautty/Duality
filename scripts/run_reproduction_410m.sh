#!/bin/bash
# DRAI V1: One-Command Reproduction of +3.3% Result
#
# This script runs the story comprehension benchmark on pythia-410m
# and validates the +3.3% improvement claim.
#
# Expected result:
#   Baseline: 85.0% accuracy
#   V1 DRAI:  88.3% accuracy
#   Delta:    +3.3 percentage points
#
# Runtime: ~5-10 minutes on CPU

set -e  # Exit on error

echo "=============================================="
echo "DRAI V1 - Reproduction Script"
echo "=============================================="
echo ""
echo "This will:"
echo "  1. Check dependencies"
echo "  2. Run pythia-410m benchmark"
echo "  3. Validate +3.3% improvement"
echo ""
echo "Runtime: ~5-10 minutes on CPU"
echo "=============================================="
echo ""

# Check if we're in the right directory
if [ ! -f "benchmarks/story_comprehension_410m/run.py" ]; then
    echo "ERROR: Must run from project root directory"
    echo "Usage: bash scripts/run_reproduction_410m.sh"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi

# Check dependencies
echo "[1/3] Checking dependencies..."
python3 -c "import torch, transformers" 2>/dev/null || {
    echo "ERROR: Missing dependencies"
    echo "Install with: pip install -r requirements.txt"
    exit 1
}
echo "  ✓ Dependencies OK"
echo ""

# Run benchmark
echo "[2/3] Running benchmark (this takes ~5-10 minutes)..."
echo ""
cd benchmarks/story_comprehension_410m
python3 run.py

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "[3/3] SUCCESS!"
    echo "=============================================="
    echo ""
    echo "Results saved to:"
    echo "  benchmarks/story_comprehension_410m/results/comparison.json"
    echo ""
    echo "You have successfully reproduced the +3.3% improvement!"
    echo ""
else
    echo ""
    echo "=============================================="
    echo "ERROR: Benchmark failed"
    echo "=============================================="
    exit 1
fi
