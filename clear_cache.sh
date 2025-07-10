#!/bin/bash

# Clear pip cache
pip cache purge

# Clear Python __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} +

# Clear .cache directories (common for various tools)
find . -type d -name ".cache" -exec rm -rf {} +

# Clear Jupyter Notebook checkpoints (if any)
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

rm -f $(find . -name ".DS_Store" -type f)

rm -rf debug_gait_parsing 

echo "All cache cleared."

clear 