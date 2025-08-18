#!/usr/bin/env python3
"""
Test script to demonstrate the identification conclusion matrix functionality
"""

import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import SystemConfig
from src.processing.enhanced_identity_manager import IdentityManager


def test_identification_matrix():
    """Test the identification conclusion matrix with sample data"""
    print("🧪 Testing Identification Conclusion Matrix")
    print("=" * 60)
    
    # Create config and identity manager
    config = SystemConfig()
    identity_manager = IdentityManager(config)
    
    # Simulate some identification data
    print("📝 Simulating identification data...")
    
    # Simulate Track 1: Mostly identified as "Alice"
    for frame in range(10, 50, 5):
        identity_manager.identification_history[1].append((frame, "Alice", 0.95))
        identity_manager.frame_identifications[frame][1] = "Alice"
        identity_manager.person_identification_counts["Alice"] += 1
        identity_manager.track_identification_counts[1]["Alice"] += 1
    
    # Add a few misidentifications for Track 1
    for frame in range(55, 65, 5):
        identity_manager.identification_history[1].append((frame, "Bob", 0.75))
        identity_manager.frame_identifications[frame][1] = "Bob"
        identity_manager.person_identification_counts["Bob"] += 1
        identity_manager.track_identification_counts[1]["Bob"] += 1
    
    # Simulate Track 2: Consistently identified as "Bob"
    for frame in range(20, 100, 10):
        identity_manager.identification_history[2].append((frame, "Bob", 0.92))
        identity_manager.frame_identifications[frame][2] = "Bob"
        identity_manager.person_identification_counts["Bob"] += 1
        identity_manager.track_identification_counts[2]["Bob"] += 1
    
    # Simulate Track 3: Identified as "Charlie"
    for frame in range(30, 80, 8):
        identity_manager.identification_history[3].append((frame, "Charlie", 0.88))
        identity_manager.frame_identifications[frame][3] = "Charlie"
        identity_manager.person_identification_counts["Charlie"] += 1
        identity_manager.track_identification_counts[3]["Charlie"] += 1
    
    # Add some mixed identifications for Track 3
    for frame in range(85, 95, 5):
        identity_manager.identification_history[3].append((frame, "Alice", 0.70))
        identity_manager.frame_identifications[frame][3] = "Alice"
        identity_manager.person_identification_counts["Alice"] += 1
        identity_manager.track_identification_counts[3]["Alice"] += 1
    
    # Simulate Track 4: Split between "Alice" and "David"
    for frame in range(40, 60, 5):
        identity_manager.identification_history[4].append((frame, "Alice", 0.80))
        identity_manager.frame_identifications[frame][4] = "Alice"
        identity_manager.person_identification_counts["Alice"] += 1
        identity_manager.track_identification_counts[4]["Alice"] += 1
    
    for frame in range(65, 85, 5):
        identity_manager.identification_history[4].append((frame, "David", 0.85))
        identity_manager.frame_identifications[frame][4] = "David"
        identity_manager.person_identification_counts["David"] += 1
        identity_manager.track_identification_counts[4]["David"] += 1
    
    print(f"✅ Simulated data for {len(identity_manager.identification_history)} tracks")
    print(f"   Total identifications: {sum(identity_manager.person_identification_counts.values())}")
    print(f"   Unique persons: {len(identity_manager.person_identification_counts)}")
    
    # Test getting basic statistics
    print("\n📊 Testing identification statistics...")
    id_stats = identity_manager.get_identification_statistics()
    print(f"   Statistics method returned {len(id_stats)} fields")
    
    # Test getting conclusion matrix data
    print("\n🔍 Testing conclusion matrix generation...")
    matrix_data = identity_manager.get_identification_conclusion_matrix()
    print(f"   Matrix data contains {len(matrix_data)} fields")
    print(f"   Tracks in matrix: {len(matrix_data['conclusion_matrix'])}")
    print(f"   Persons in matrix: {len(matrix_data['person_summary'])}")
    
    # Display the full conclusion matrix
    print("\n🎯 Displaying full conclusion matrix:")
    identity_manager.print_identification_conclusion_matrix()
    
    # Test final summary
    print("\n📋 Testing final summary:")
    identity_manager.print_final_summary()
    
    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    test_identification_matrix()
