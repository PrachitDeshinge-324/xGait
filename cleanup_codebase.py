#!/usr/bin/env python3
"""
Cleanup script to remove unnecessary files and duplicates from the codebase
"""
import os
import shutil
from pathlib import Path

def cleanup_codebase():
    """Clean up unnecessary files from the codebase"""
    base_dir = Path(__file__).parent
    
    print("🧹 Starting codebase cleanup...")
    
    # Files to remove
    files_to_remove = [
        "track_persons_new.py",         # Temporary file during modularization
        "track_persons_original.py",   # Backup file - no longer needed
        "src/models/xgait_adapter.py", # Moved to src/models/xgait/adapter.py
        "src/models/official_xgait_model.py", # Moved to src/models/xgait/official.py
    ]
    
    # Remove files
    removed_count = 0
    for file_path in files_to_remove:
        full_path = base_dir / file_path
        if full_path.exists():
            try:
                full_path.unlink()
                print(f"✅ Removed: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"⚠️  Failed to remove {file_path}: {e}")
        else:
            print(f"ℹ️  File not found: {file_path}")
    
    # Remove Python cache directories
    cache_dirs = list(base_dir.rglob("__pycache__"))
    for cache_dir in cache_dirs:
        try:
            shutil.rmtree(cache_dir)
            print(f"✅ Removed cache: {cache_dir.relative_to(base_dir)}")
            removed_count += 1
        except Exception as e:
            print(f"⚠️  Failed to remove cache {cache_dir}: {e}")
    
    # Remove .pyc files
    pyc_files = list(base_dir.rglob("*.pyc"))
    for pyc_file in pyc_files:
        try:
            pyc_file.unlink()
            print(f"✅ Removed: {pyc_file.relative_to(base_dir)}")
            removed_count += 1
        except Exception as e:
            print(f"⚠️  Failed to remove {pyc_file}: {e}")
    
    # Remove .DS_Store files (macOS)
    ds_store_files = list(base_dir.rglob(".DS_Store"))
    for ds_file in ds_store_files:
        try:
            ds_file.unlink()
            print(f"✅ Removed: {ds_file.relative_to(base_dir)}")
            removed_count += 1
        except Exception as e:
            print(f"⚠️  Failed to remove {ds_file}: {e}")
    
    print(f"\n📊 Cleanup Summary:")
    print(f"   • Total items removed: {removed_count}")
    
    # Report current state
    print(f"\n📋 Current Main Files:")
    main_files = [
        "main.py",
        "src/app/main_app.py",
        "src/processing/",
        "src/models/xgait_model.py",
        "src/models/xgait/",
    ]
    
    for file_path in main_files:
        full_path = base_dir / file_path
        if full_path.exists():
            if full_path.is_dir():
                print(f"   ✅ Directory: {file_path}")
            else:
                print(f"   ✅ File: {file_path}")
        else:
            print(f"   ❌ Missing: {file_path}")
    
    print(f"\n✅ Codebase cleanup completed!")

if __name__ == "__main__":
    cleanup_codebase()
