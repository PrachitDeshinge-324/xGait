#!/usr/bin/env python3
"""
Cleanup script to remove temporary files and cache while preserving visualization_analysis folders
"""

import os
import shutil
from pathlib import Path

def cleanup_project():
    """Clean up the project directory"""
    base_dir = Path(__file__).parent
    
    print("🧹 Starting project cleanup...")
    
    # Remove Python cache files
    cache_patterns = [
        "**/__pycache__",
        "**/*.pyc", 
        "**/*.pyo",
        "**/.ipynb_checkpoints"
    ]
    
    for pattern in cache_patterns:
        for path in base_dir.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                print(f"   Removed directory: {path.relative_to(base_dir)}")
            else:
                path.unlink(missing_ok=True)
                print(f"   Removed file: {path.relative_to(base_dir)}")
    
    # Remove system files
    system_files = [
        "**/.DS_Store",
        "**/Thumbs.db"
    ]
    
    for pattern in system_files:
        for path in base_dir.glob(pattern):
            path.unlink(missing_ok=True)
            print(f"   Removed system file: {path.relative_to(base_dir)}")
    
    # Remove temporary directories (but keep visualization_analysis*)
    temp_dirs = [
        "debug_gait_parsing",
        "gallery_data", 
        "gallery_analysis",
        "clustering_analysis",
        "track_naming_results"
    ]
    
    for temp_dir in temp_dirs:
        temp_path = base_dir / temp_dir
        if temp_path.exists() and temp_path.is_dir():
            shutil.rmtree(temp_path, ignore_errors=True)
            print(f"   Removed temp directory: {temp_dir}")
    
    # Clean up log files (optional)
    log_files = base_dir.glob("*.log")
    for log_file in log_files:
        log_file.unlink(missing_ok=True)
        print(f"   Removed log file: {log_file.name}")
    
    # Keep visualization_analysis* folders
    viz_dirs = list(base_dir.glob("visualization_analysis*"))
    if viz_dirs:
        print(f"   Keeping {len(viz_dirs)} visualization_analysis folders")
    
    print("✅ Cleanup completed!")

if __name__ == "__main__":
    cleanup_project()
