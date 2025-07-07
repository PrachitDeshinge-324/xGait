"""
Visual Track Reviewer for Interactive Person Identification
Shows track crops using matplotlib and allows interactive naming
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple, Optional, Any
import cv2
from pathlib import Path
import pickle
import json
from collections import defaultdict


class VisualTrackReviewer:
    """Visual interface for reviewing and naming tracks"""
    
    def __init__(self, visualization_dir: str = "visualization_analysis"):
        self.visualization_dir = Path(visualization_dir)
        self.track_data = None
        self.track_crops = None
        self.track_embeddings = None
        self.track_qualities = None
        self.track_to_person = {}
        
    def load_track_data(self) -> bool:
        """Load track data from saved files"""
        try:
            # Load track data
            track_data_path = self.visualization_dir / "track_data.json"
            if not track_data_path.exists():
                return False
            
            with open(track_data_path, 'r') as f:
                self.track_data = json.load(f)
            
            # Load track crops
            crops_path = self.visualization_dir / "track_crops.pkl"
            if crops_path.exists():
                with open(crops_path, 'rb') as f:
                    crop_data = pickle.load(f)
                    self.track_crops = crop_data.get('track_crops', {})
            
            # Process track data
            self.track_embeddings = {}
            for track_id, embeddings in self.track_data['track_embeddings'].items():
                self.track_embeddings[int(track_id)] = [np.array(emb) for emb in embeddings]
            
            self.track_qualities = {}
            for track_id, qualities in self.track_data['track_qualities'].items():
                self.track_qualities[int(track_id)] = qualities
            
            self.track_to_person = {int(k): v for k, v in self.track_data['track_to_person'].items()}
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading track data: {e}")
            return False
    
    def get_track_summary(self) -> Dict[int, Dict]:
        """Get summary of all tracks"""
        summary = {}
        
        # Use buffer attributes if available (for in-memory data), otherwise use loaded attributes
        track_embeddings = getattr(self, 'track_embedding_buffer', None) or getattr(self, 'track_embeddings', {})
        track_qualities = getattr(self, 'track_quality_buffer', None) or getattr(self, 'track_qualities', {})
        track_crops = getattr(self, 'track_crop_buffer', None) or getattr(self, 'track_crops', {})
        
        for track_id in track_embeddings.keys():
            embeddings = track_embeddings[track_id]
            qualities = track_qualities.get(track_id, [])
            crops = track_crops.get(track_id, []) if track_crops else []
            assigned_person = self.track_to_person.get(track_id)
            
            summary[track_id] = {
                'num_embeddings': len(embeddings),
                'num_crops': len(crops),
                'avg_quality': np.mean(qualities) if qualities else 0.0,
                'assigned_person': assigned_person,
                'is_assigned': assigned_person is not None
            }
        
        return summary
    
    def show_track_crops(self, track_id: int, max_crops: int = 6) -> bool:
        """Display crops for a specific track using matplotlib"""
        if not self.track_crops or track_id not in self.track_crops:
            print(f"❌ No crops available for track {track_id}")
            return False
        
        crops = self.track_crops[track_id]
        if not crops:
            print(f"❌ No crops available for track {track_id}")
            return False
        
        # Select representative crops
        selected_crops = self._select_representative_crops(crops, max_crops)
        
        # Create figure
        num_crops = len(selected_crops)
        cols = min(3, num_crops)
        rows = (num_crops + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        fig.suptitle(f'Track {track_id} - Person Crops ({num_crops} samples)', fontsize=16)
        
        if rows == 1:
            axes = [axes] if num_crops == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, crop in enumerate(selected_crops):
            if i < len(axes):
                ax = axes[i]
                
                # Convert BGR to RGB for matplotlib
                if len(crop.shape) == 3:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                else:
                    crop_rgb = crop
                
                ax.imshow(crop_rgb)
                ax.set_title(f'Sample {i+1}')
                ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_crops, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save to file for viewing
        output_path = self.visualization_dir / f"track_{track_id}_crops.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📸 Crops saved to: {output_path}")
        
        # Try to show (will work in interactive environments)
        try:
            plt.show()
        except:
            print("💡 Note: Crops saved to file since display is not available in this environment")
        
        plt.close()
        
        return True
    
    def _select_representative_crops(self, crops: List[np.ndarray], max_crops: int) -> List[np.ndarray]:
        """Select representative crops from the available crops"""
        if len(crops) <= max_crops:
            return crops
        
        # Select crops evenly distributed across the sequence
        indices = np.linspace(0, len(crops) - 1, max_crops, dtype=int)
        return [crops[i] for i in indices]
    
    def show_all_tracks_overview(self) -> None:
        """Show overview of all tracks with thumbnail crops"""
        summary = self.get_track_summary()
        
        if not summary:
            print("❌ No tracks found")
            return
        
        print("\n" + "="*70)
        print("👥 TRACK OVERVIEW")
        print("="*70)
        
        unassigned_tracks = []
        assigned_tracks = []
        
        for track_id, info in summary.items():
            status = "✅ ASSIGNED" if info['is_assigned'] else "❌ UNASSIGNED"
            person = info['assigned_person'] or "UNASSIGNED"
            
            print(f"Track {track_id:2d}: {info['num_embeddings']:2d} embeddings | "
                  f"Quality: {info['avg_quality']:.3f} | "
                  f"Crops: {info['num_crops']:2d} | "
                  f"Person: {person:15s} | {status}")
            
            if info['is_assigned']:
                assigned_tracks.append(track_id)
            else:
                unassigned_tracks.append(track_id)
        
        print(f"\n📊 Summary: {len(assigned_tracks)} assigned, {len(unassigned_tracks)} unassigned")
        
        if unassigned_tracks:
            print(f"🔧 Unassigned tracks: {unassigned_tracks}")
    
    def review_track_interactive(self, track_id: int) -> Optional[str]:
        """Review a single track interactively"""
        # Use buffer attributes if available (for in-memory data), otherwise use loaded attributes
        track_embeddings = getattr(self, 'track_embedding_buffer', None) or getattr(self, 'track_embeddings', {})
        
        if track_id not in track_embeddings:
            print(f"❌ Track {track_id} not found")
            return None
        
        # Show track info
        summary = self.get_track_summary()
        info = summary[track_id]
        
        print(f"\n🔍 Reviewing Track {track_id}")
        print(f"   • Embeddings: {info['num_embeddings']}")
        print(f"   • Average Quality: {info['avg_quality']:.3f}")
        print(f"   • Crops Available: {info['num_crops']}")
        print(f"   • Current Assignment: {info['assigned_person'] or 'UNASSIGNED'}")
        
        # Show crops if available
        if info['num_crops'] > 0:
            print(f"📸 Showing crops for Track {track_id}...")
            self.show_track_crops(track_id)
        
        # Get user input
        while True:
            print(f"\n🎯 Track {track_id} - What would you like to do?")
            print("   1. Enter person name")
            print("   2. Skip this track")
            print("   3. Mark as 'Unknown'")
            print("   4. Show crops again")
            
            choice = input("Enter choice (1-4): ").strip()
            
            if choice == '1':
                person_name = input("Enter person name: ").strip()
                if person_name:
                    return person_name
                else:
                    print("❌ Empty name not allowed")
            elif choice == '2':
                print("⏭️ Skipping track")
                return None
            elif choice == '3':
                return "Unknown"
            elif choice == '4':
                if info['num_crops'] > 0:
                    self.show_track_crops(track_id)
                else:
                    print("❌ No crops available for this track")
            else:
                print("❌ Invalid choice. Please enter 1-4")
    
    def review_all_unassigned_tracks(self) -> Dict[int, str]:
        """Review all unassigned tracks interactively"""
        summary = self.get_track_summary()
        unassigned_tracks = [tid for tid, info in summary.items() if not info['is_assigned']]
        
        if not unassigned_tracks:
            print("✅ All tracks are already assigned")
            return {}
        
        print(f"\n🚀 Starting interactive review of {len(unassigned_tracks)} unassigned tracks")
        
        assignments = {}
        
        for i, track_id in enumerate(unassigned_tracks):
            print(f"\n📋 Progress: {i+1}/{len(unassigned_tracks)}")
            person_name = self.review_track_interactive(track_id)
            
            if person_name:
                assignments[track_id] = person_name
                print(f"✅ Track {track_id} assigned to '{person_name}'")
            
            # Ask if user wants to continue
            if i < len(unassigned_tracks) - 1:
                continue_review = input(f"\nContinue reviewing? (y/n, default=y): ").strip().lower()
                if continue_review == 'n':
                    print("🛑 Review session ended by user")
                    break
        
        return assignments
    
    def get_track_data_for_assignment(self, track_id: int) -> Optional[Tuple[List[np.ndarray], List[float]]]:
        """Get embeddings and qualities for a track assignment"""
        # Use buffer attributes if available (for in-memory data), otherwise use loaded attributes
        track_embeddings = getattr(self, 'track_embedding_buffer', None) or getattr(self, 'track_embeddings', {})
        track_qualities = getattr(self, 'track_quality_buffer', None) or getattr(self, 'track_qualities', {})
        
        if track_id not in track_embeddings:
            return None
        
        embeddings = track_embeddings[track_id]
        qualities = track_qualities.get(track_id, [0.5] * len(embeddings))
        
        return embeddings, qualities
    
    def review_all_tracks(self) -> Dict[int, str]:
        """Review all tracks interactively, including assigned ones for confirmation"""
        summary = self.get_track_summary()
        
        if not summary:
            print("❌ No tracks found")
            return {}
        
        print(f"\n🚀 Starting interactive review of all {len(summary)} tracks")
        
        assignments = {}
        
        # Sort tracks by assigned status (assigned first) then by track_id
        all_tracks = sorted(summary.keys(), key=lambda tid: (not summary[tid]['is_assigned'], tid))
        
        for i, track_id in enumerate(all_tracks):
            is_assigned = summary[track_id]['is_assigned']
            current_person = summary[track_id]['assigned_person']
            
            print(f"\n📋 Progress: {i+1}/{len(all_tracks)}")
            
            # For assigned tracks, ask for confirmation
            if is_assigned and current_person:
                person_name = self.review_assigned_track_interactive(track_id, current_person)
            else:
                person_name = self.review_track_interactive(track_id)
            
            if person_name:
                assignments[track_id] = person_name
                print(f"✅ Track {track_id} assigned to '{person_name}'")
            
            # Ask if user wants to continue
            if i < len(all_tracks) - 1:
                continue_review = input(f"\nContinue reviewing? (y/n, default=y): ").strip().lower()
                if continue_review == 'n':
                    print("🛑 Review session ended by user")
                    break
        
        return assignments
        
    def review_assigned_track_interactive(self, track_id: int, current_person: str) -> Optional[str]:
        """Review a track that already has an assigned identity"""
        # Use buffer attributes if available, otherwise use loaded attributes
        track_embeddings = getattr(self, 'track_embedding_buffer', None) or getattr(self, 'track_embeddings', {})
        
        if track_id not in track_embeddings:
            print(f"❌ Track {track_id} not found")
            return None
        
        # Show track info
        summary = self.get_track_summary()
        info = summary[track_id]
        
        print(f"\n🔍 Reviewing Track {track_id} - Currently Identified as '{current_person}'")
        print(f"   • Embeddings: {info['num_embeddings']}")
        print(f"   • Average Quality: {info['avg_quality']:.3f}")
        print(f"   • Crops Available: {info['num_crops']}")
        
        # Show crops if available
        if info['num_crops'] > 0:
            print(f"📸 Showing crops for Track {track_id}...")
            self.show_track_crops(track_id)
        
        # Ask for confirmation
        while True:
            print(f"\n🎯 Track {track_id} - Is '{current_person}' correct?")
            print("   1. Yes, keep this identification")
            print("   2. No, enter new identity")
            print("   3. Show crops again")
            
            choice = input("Enter choice (1-3): ").strip()
            
            if choice == '1':
                print(f"✅ Confirmed: Track {track_id} is '{current_person}'")
                return current_person
            elif choice == '2':
                person_name = input("Enter correct person name: ").strip()
                if person_name:
                    return person_name
                else:
                    print("❌ Empty name not allowed")
            elif choice == '3':
                if info['num_crops'] > 0:
                    self.show_track_crops(track_id)
                else:
                    print("❌ No crops available for this track")
            else:
                print("❌ Invalid choice. Please enter 1-3")
