#!/bin/bash
# ==========================================
# Script: run.sh
# Purpose: Run video tracker app
# ==========================================

# Suppress warnings and set single-threaded behavior
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_MAX_ACTIVE_LEVELS=1
export PYTHONUNBUFFERED=1

# If conda is not initialized in non-interactive shells, initialize it manually:
if [ -f "$(brew --prefix)/etc/profile.d/conda.sh" ]; then
    source "$(brew --prefix)/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

# Activate the environment
# conda init
# conda activate tracker

# Run the tracker with real-time output
# python main.py \
#     --input input/3c.mp4 \
#     --interactive \
#     --save-video \
#     --max-frames 500 \
#     --debug \
#     --output output \
#     --output-video output/3c_clustering.mp4

# Alternate run example:
python main.py \
    --input input/3c1.mp4 \
    --interactive \
    --save-video \
    --max-frames 1000 \
    --debug \
    --output output \
    --output-video output/3c1_clustering.mp4