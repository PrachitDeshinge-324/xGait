#!/usr/bin/env python3
"""
Manual Track Merging Interface for Person Identification System
Allows users to manually review tracks, name persons, and merge similar tracks
"""

import sys
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.simple_identity_gallery import SimpleIdentityGallery
from src.utils.enhanced_person_gallery import EnhancedPersonGallery
from src.processing.enhanced_identity_manager import EnhancedIdentityManager
from src.config import SystemConfig


class ManualTrackMerger:
    """Interactive tool for manually merging tracks and managing person identities"""
    
    def __init__(self, config_path: str = None):
        """Initialize the manual track merger"""
        self.config = SystemConfig.load_default() if not config_path else SystemConfig.load_from_file(config_path)
        
        # Initialize identity manager
        self.identity_manager = EnhancedIdentityManager(self.config, use_enhanced_gallery=True)
        self.identity_manager.load_gallery()
        
        # Data structures for manual management
        self.track_embedding_buffer = defaultdict(list)
        self.track_quality_buffer = defaultdict(list)
        self.track_to_person = {}
        self.unassigned_tracks = set()
        
        print("🔧 Manual Track Merger initialized")
        print(f"   Simple gallery: {len(self.identity_manager.simple_gallery.gallery)} persons")
        if self.identity_manager.enhanced_gallery:
            print(f"   Enhanced gallery: {len(self.identity_manager.enhanced_gallery.gallery)} persons")
    
    def load_track_data(self, track_data_path: str = None) -> bool:
        """
        Load track embedding data from processing results
        
        Args:
            track_data_path: Path to track data file (optional)
            
        Returns:
            True if data loaded successfully
        """
        # Try to load from saved track data first
        if self.identity_manager.load_track_data():
            # Use loaded data from identity manager
            self.track_embedding_buffer = dict(self.identity_manager.track_embedding_buffer)
            self.track_quality_buffer = dict(self.identity_manager.track_quality_buffer)
            self.track_to_person = dict(self.identity_manager.track_to_person)
        else:
            # Fall back to loading from identity manager buffers (if available)
            if hasattr(self.identity_manager, 'track_embedding_buffer'):
                self.track_embedding_buffer = dict(self.identity_manager.track_embedding_buffer)
                self.track_quality_buffer = dict(self.identity_manager.track_quality_buffer)
                self.track_to_person = dict(self.identity_manager.track_to_person)
        
        # Identify unassigned tracks
        for track_id in self.track_embedding_buffer.keys():
            if track_id not in self.track_to_person:
                self.unassigned_tracks.add(track_id)
        
        print(f"📊 Loaded track data:")
        print(f"   Total tracks: {len(self.track_embedding_buffer)}")
        print(f"   Assigned tracks: {len(self.track_to_person)}")
        print(f"   Unassigned tracks: {len(self.unassigned_tracks)}")
        
        return len(self.track_embedding_buffer) > 0
    
    def show_track_summary(self) -> None:
        """Display summary of all tracks and their status"""
        print("\n" + "="*60)
        print("📋 TRACK SUMMARY")
        print("="*60)
        
        for track_id in sorted(self.track_embedding_buffer.keys()):
            embeddings = self.track_embedding_buffer[track_id]
            qualities = self.track_quality_buffer.get(track_id, [])
            assigned_person = self.track_to_person.get(track_id, "UNASSIGNED")
            
            avg_quality = np.mean(qualities) if qualities else 0.0
            status = "✅ ASSIGNED" if track_id in self.track_to_person else "❌ UNASSIGNED"
            
            print(f"Track {track_id:2d}: {len(embeddings):2d} embeddings | "
                  f"Avg Quality: {avg_quality:.3f} | "
                  f"Person: {assigned_person:12s} | {status}")
        
        print(f"\n📊 Summary: {len(self.track_to_person)} assigned, {len(self.unassigned_tracks)} unassigned")
    
    def show_person_gallery(self) -> None:
        """Display current person gallery"""
        print("\n" + "="*60)
        print("👥 PERSON GALLERY")
        print("="*60)
        
        gallery = self.identity_manager.simple_gallery.gallery
        if not gallery:
            print("📭 Gallery is empty - no persons created yet")
            return
        
        for person_name, person_data in gallery.items():
            tracks = person_data.track_associations
            embeddings = len(person_data.embeddings)
            avg_quality = np.mean(person_data.qualities) if person_data.qualities else 0.0
            
            print(f"{person_name}: {embeddings} embeddings | "
                  f"Avg Quality: {avg_quality:.3f} | "
                  f"Tracks: {tracks}")
    
    def create_person_from_track(self, track_id: int, person_name: str) -> bool:
        """
        Create a new person from an unassigned track
        
        Args:
            track_id: Track ID to create person from
            person_name: Name for the new person
            
        Returns:
            True if person was created successfully
        """
        if track_id not in self.track_embedding_buffer:
            print(f"❌ Track {track_id} not found")
            return False
        
        if track_id in self.track_to_person:
            print(f"❌ Track {track_id} is already assigned to {self.track_to_person[track_id]}")
            return False
        
        embeddings = self.track_embedding_buffer[track_id]
        qualities = self.track_quality_buffer.get(track_id, [0.5] * len(embeddings))
        
        if not embeddings:
            print(f"❌ Track {track_id} has no embeddings")
            return False
        
        # Create person in simple gallery
        success = self.identity_manager.simple_gallery.create_person_from_track(
            person_name, track_id, embeddings, qualities
        )
        
        if success:
            # Update local tracking
            self.track_to_person[track_id] = person_name
            self.unassigned_tracks.discard(track_id)
            
            # Create in enhanced gallery if available
            if self.identity_manager.enhanced_gallery and self.identity_manager.track_crop_buffer:
                if track_id in self.identity_manager.track_crop_buffer:
                    latest_crop = self.identity_manager.track_crop_buffer[track_id][-1]
                    latest_bbox = self.identity_manager.track_bbox_buffer[track_id][-1]
                    
                    for i, (embedding, quality) in enumerate(zip(embeddings, qualities)):
                        self.identity_manager.enhanced_gallery.add_person_embedding(
                            person_name, track_id, embedding, latest_bbox, 
                            latest_crop, i, quality
                        )
            
            print(f"✅ Created person '{person_name}' from track {track_id} with {len(embeddings)} embeddings")
            return True
        else:
            print(f"❌ Failed to create person '{person_name}' from track {track_id}")
            return False
    
    def merge_tracks(self, primary_track_id: int, secondary_track_id: int) -> bool:
        """
        Merge two tracks - combine embeddings from secondary into primary track's person
        
        Args:
            primary_track_id: Track ID to keep (must be assigned to a person)
            secondary_track_id: Track ID to merge (will be merged into primary's person)
            
        Returns:
            True if merge was successful
        """
        # Validate tracks exist
        if primary_track_id not in self.track_embedding_buffer:
            print(f"❌ Primary track {primary_track_id} not found")
            return False
        
        if secondary_track_id not in self.track_embedding_buffer:
            print(f"❌ Secondary track {secondary_track_id} not found")
            return False
        
        # Primary track must be assigned
        if primary_track_id not in self.track_to_person:
            print(f"❌ Primary track {primary_track_id} must be assigned to a person first")
            return False
        
        primary_person = self.track_to_person[primary_track_id]
        
        # Get secondary track data
        secondary_embeddings = self.track_embedding_buffer[secondary_track_id]
        secondary_qualities = self.track_quality_buffer.get(secondary_track_id, [0.5] * len(secondary_embeddings))
        
        if not secondary_embeddings:
            print(f"❌ Secondary track {secondary_track_id} has no embeddings")
            return False
        
        # Add secondary embeddings to primary person
        for embedding, quality in zip(secondary_embeddings, secondary_qualities):
            self.identity_manager.simple_gallery._add_embedding_to_person(
                primary_person, embedding, quality, secondary_track_id
            )
        
        # Update enhanced gallery if available
        if (self.identity_manager.enhanced_gallery and 
            secondary_track_id in self.identity_manager.track_crop_buffer):
            latest_crop = self.identity_manager.track_crop_buffer[secondary_track_id][-1]
            latest_bbox = self.identity_manager.track_bbox_buffer[secondary_track_id][-1]
            
            for i, (embedding, quality) in enumerate(zip(secondary_embeddings, secondary_qualities)):
                self.identity_manager.enhanced_gallery.add_person_embedding(
                    primary_person, secondary_track_id, embedding, latest_bbox, 
                    latest_crop, i, quality
                )
        
        # Update tracking
        self.track_to_person[secondary_track_id] = primary_person
        self.unassigned_tracks.discard(secondary_track_id)
        
        print(f"✅ Merged track {secondary_track_id} into {primary_person} (primary track: {primary_track_id})")
        print(f"   Added {len(secondary_embeddings)} embeddings")
        
        return True
    
    def merge_persons(self, person1_name: str, person2_name: str) -> bool:
        """
        Merge two persons in the gallery
        
        Args:
            person1_name: Name of first person (will be kept)
            person2_name: Name of second person (will be merged into first)
            
        Returns:
            True if merge was successful
        """
        return self.identity_manager.merge_persons(person1_name, person2_name)
    
    def save_gallery(self) -> bool:
        """Save the current gallery state"""
        try:
            self.identity_manager.save_gallery()
            print("✅ Gallery saved successfully")
            return True
        except Exception as e:
            print(f"❌ Error saving gallery: {e}")
            return False
    
    def interactive_session(self) -> None:
        """Start an interactive session for manual track management"""
        print("\n🎮 Starting Interactive Track Management Session")
        print("="*60)
        print("Commands:")
        print("  summary      - Show track summary")
        print("  gallery      - Show person gallery") 
        print("  create <track_id> <person_name> - Create person from track")
        print("  merge <primary_track> <secondary_track> - Merge tracks")
        print("  merge_persons <person1> <person2> - Merge persons")
        print("  save         - Save gallery")
        print("  help         - Show this help")
        print("  quit         - Exit session")
        print("="*60)
        
        while True:
            try:
                command = input("\n🔧 Enter command: ").strip().lower()
                
                if command == "quit" or command == "exit":
                    break
                elif command == "help":
                    self.interactive_session()  # Show help again
                    break
                elif command == "summary":
                    self.show_track_summary()
                elif command == "gallery":
                    self.show_person_gallery()
                elif command == "save":
                    self.save_gallery()
                elif command.startswith("create"):
                    parts = command.split()
                    if len(parts) >= 3:
                        try:
                            track_id = int(parts[1])
                            person_name = " ".join(parts[2:])
                            self.create_person_from_track(track_id, person_name)
                        except ValueError:
                            print("❌ Invalid track ID. Use: create <track_id> <person_name>")
                    else:
                        print("❌ Usage: create <track_id> <person_name>")
                elif command.startswith("merge ") and "persons" not in command:
                    parts = command.split()
                    if len(parts) == 3:
                        try:
                            primary_track = int(parts[1])
                            secondary_track = int(parts[2])
                            self.merge_tracks(primary_track, secondary_track)
                        except ValueError:
                            print("❌ Invalid track IDs. Use: merge <primary_track> <secondary_track>")
                    else:
                        print("❌ Usage: merge <primary_track> <secondary_track>")
                elif command.startswith("merge_persons"):
                    parts = command.split()
                    if len(parts) == 3:
                        person1 = parts[1]
                        person2 = parts[2]
                        self.merge_persons(person1, person2)
                    else:
                        print("❌ Usage: merge_persons <person1> <person2>")
                else:
                    print("❌ Unknown command. Type 'help' for available commands")
                    
            except KeyboardInterrupt:
                print("\n👋 Session interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("👋 Interactive session ended")


def main():
    """Main function for manual track merging"""
    print("🚀 Manual Track Merging Interface")
    print("="*50)
    
    # Initialize merger
    merger = ManualTrackMerger()
    
    # Load track data
    if not merger.load_track_data():
        print("❌ No track data found. Run the main application first to generate tracks.")
        return
    
    # Show initial state
    merger.show_track_summary()
    merger.show_person_gallery()
    
    # Start interactive session
    merger.interactive_session()
    
    # Final save
    print("\n💾 Final save...")
    merger.save_gallery()
    print("✅ Manual track merging completed")


if __name__ == "__main__":
    main()
