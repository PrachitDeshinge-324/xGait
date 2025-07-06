#!/bin/bash
# Set environment variables to suppress warnings
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_MAX_ACTIVE_LEVELS=1
export PYTHONUNBUFFERED=1

# Run the application with real-time output (no grep buffering)
python main.py --input input/3c1.mp4 --max-frames 200 --interactive