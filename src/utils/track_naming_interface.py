#!/usr/bin/env python3
"""
Interactive Track Naming Utility
Provides an interactive interface for naming tracks after video processing
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
from datetime import datetime

class InteractiveTrackNaming:
    """
    Interactive interface for naming tracks with sample visualizations
    """
    
    def __init__(self, simple_gallery, track_history: Dict, track_embedding_buffer: Dict, 
                 track_quality_buffer: Dict, debug_output_dir: Optional[Path] = None, 
                 video_path: Optional[str] = None):
        self.simple_gallery = simple_gallery
        self.track_history = track_history
        self.track_embedding_buffer = track_embedding_buffer
        self.track_quality_buffer = track_quality_buffer
        self.debug_output_dir = debug_output_dir or Path("debug_gait_parsing")
        self.video_path = video_path
        
        # Create output directories
        self.samples_dir = Path("track_samples")
        self.samples_dir.mkdir(exist_ok=True)
        
        self.results_dir = Path("track_naming_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Video cache for extracting frames
        self.video_cache = {}
        self.video_cap = None
        
        # Initialize video capture if path provided
        if self.video_path and Path(self.video_path).exists():
            self.video_cap = cv2.VideoCapture(self.video_path)
            if not self.video_cap.isOpened():
                print(f"⚠️  Could not open video: {self.video_path}")
                self.video_cap = None
        
    def __del__(self):
        """Clean up video capture"""
        if self.video_cap:
            self.video_cap.release()
            
    def extract_person_crop(self, frame_num: int, center_x: int, center_y: int, 
                           crop_size: Tuple[int, int] = (128, 256)) -> Optional[np.ndarray]:
        """
        Extract person crop from video frame
        
        Args:
            frame_num: Frame number in video
            center_x: Center X coordinate of person
            center_y: Center Y coordinate of person
            crop_size: Desired crop size (width, height)
            
        Returns:
            Cropped image or None if extraction fails
        """
        if not self.video_cap:
            return None
            
        # Check cache first
        cache_key = f"{frame_num}_{center_x}_{center_y}"
        if cache_key in self.video_cache:
            return self.video_cache[cache_key]
            
        try:
            # Set frame position
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.video_cap.read()
            
            if not ret or frame is None:
                return None
                
            h, w = frame.shape[:2]
            crop_w, crop_h = crop_size
            
            # Calculate crop boundaries with padding
            padding = 20
            x1 = max(0, center_x - crop_w // 2 - padding)
            y1 = max(0, center_y - crop_h // 2 - padding)
            x2 = min(w, center_x + crop_w // 2 + padding)
            y2 = min(h, center_y + crop_h // 2 + padding)
            
            # Extract crop
            crop = frame[y1:y2, x1:x2]
            
            if crop.size > 0:
                # Resize to standard size
                crop_resized = cv2.resize(crop, crop_size)
                
                # Cache the result (limit cache size)
                if len(self.video_cache) < 50:  # Limit cache size
                    self.video_cache[cache_key] = crop_resized
                    
                return crop_resized
                
        except Exception as e:
            print(f"⚠️  Error extracting crop for frame {frame_num}: {e}")
            
        return None
        
    def run_interactive_naming(self) -> int:
        """
        Main method to run the interactive naming process
        
        Returns:
            Number of tracks merged
        """
        return self.run_full_naming_process()[1]
        
    def collect_track_data(self) -> Dict[int, Dict[str, Any]]:
        """
        Collect all track data from gallery and buffers
        
        Returns:
            Dictionary mapping track_id to track data (filters out tracks with zero embeddings)
        """
        track_data = {}
        
        # Get all embeddings by track from the gallery
        for person_name, person_data in self.simple_gallery.gallery.items():
            if person_data.track_associations:
                for track_id in person_data.track_associations:
                    if track_id not in track_data:
                        track_data[track_id] = {
                            'embeddings': [],
                            'qualities': [],
                            'person_names': [],
                            'history': self.track_history.get(track_id, [])
                        }
                    
                    # Add embeddings from this person to the track
                    for i, embedding in enumerate(person_data.embeddings):
                        quality = person_data.qualities[i] if i < len(person_data.qualities) else 0.5
                        track_data[track_id]['embeddings'].append(embedding)
                        track_data[track_id]['qualities'].append(quality)
                        track_data[track_id]['person_names'].append(person_name)
        
        # Also collect from track_embedding_buffer
        for track_id, embeddings in self.track_embedding_buffer.items():
            if track_id not in track_data:
                track_data[track_id] = {
                    'embeddings': [],
                    'qualities': [],
                    'person_names': [],
                    'history': self.track_history.get(track_id, [])
                }
            
            qualities = self.track_quality_buffer.get(track_id, [0.5] * len(embeddings))
            for i, embedding in enumerate(embeddings):
                quality = qualities[i] if i < len(qualities) else 0.5
                track_data[track_id]['embeddings'].append(embedding)
                track_data[track_id]['qualities'].append(quality)
                track_data[track_id]['person_names'].append(f'track_{track_id}')
        
        # Filter out tracks with zero embeddings
        filtered_track_data = {}
        for track_id, data in track_data.items():
            if len(data['embeddings']) > 0:
                filtered_track_data[track_id] = data
            else:
                print(f"⚠️  Filtering out Track {track_id} - zero embeddings")
        
        return filtered_track_data
    
    def find_best_sample_frames(self, track_id: int, track_data: Dict[str, Any], 
                               max_samples: int = 6) -> List[Dict[str, Any]]:
        """
        Find the best sample frames for a track based on history and quality
        
        Args:
            track_id: Track ID
            track_data: Track data dictionary
            max_samples: Maximum number of samples to return
            
        Returns:
            List of sample frame dictionaries
        """
        history = track_data.get('history', [])
        qualities = track_data.get('qualities', [])
        
        if not history:
            return []
        
        # If we have quality scores, use them to select best frames
        if qualities:
            # Combine history with quality scores
            history_with_quality = []
            for i, (center_x, center_y, frame_num, conf) in enumerate(history):
                quality = qualities[i % len(qualities)]  # Wrap around if needed
                history_with_quality.append({
                    'center_x': center_x,
                    'center_y': center_y,
                    'frame_num': frame_num,
                    'confidence': conf,
                    'quality': quality
                })
            
            # Sort by quality descending
            history_with_quality.sort(key=lambda x: x['quality'], reverse=True)
            
            # Take top samples, but ensure they're spread out over time
            selected_samples = []
            used_frame_nums = set()
            min_frame_gap = max(1, len(history) // max_samples)
            
            for sample in history_with_quality:
                if len(selected_samples) >= max_samples:
                    break
                
                # Check if this frame is too close to already selected frames
                too_close = any(abs(sample['frame_num'] - used_frame) < min_frame_gap 
                              for used_frame in used_frame_nums)
                
                if not too_close:
                    selected_samples.append(sample)
                    used_frame_nums.add(sample['frame_num'])
            
            return selected_samples
        
        else:
            # No quality scores, just distribute evenly
            step = max(1, len(history) // max_samples)
            samples = []
            
            for i in range(0, len(history), step):
                if len(samples) >= max_samples:
                    break
                center_x, center_y, frame_num, conf = history[i]
                samples.append({
                    'center_x': center_x,
                    'center_y': center_y,
                    'frame_num': frame_num,
                    'confidence': conf,
                    'quality': 0.5  # Default quality
                })
            
            return samples
    
    def create_track_visualization(self, track_id: int, track_data: Dict[str, Any]) -> Path:
        """
        Create a visualization for a track showing actual person crops and statistics
        
        Args:
            track_id: Track ID
            track_data: Track data dictionary
            
        Returns:
            Path to the saved visualization
        """
        sample_frames = self.find_best_sample_frames(track_id, track_data)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Track {track_id} - Person Crops for Naming', fontsize=20, fontweight='bold')
        
        # Display sample frames with actual crops
        for i, ax in enumerate(axes.flat):
            if i < len(sample_frames):
                frame_info = sample_frames[i]
                
                # Try to extract actual person crop
                crop = self.extract_person_crop(
                    frame_info['frame_num'], 
                    frame_info['center_x'], 
                    frame_info['center_y']
                )
                
                if crop is not None:
                    # Display the actual person crop
                    ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    ax.set_title(f'Frame {frame_info["frame_num"]} (Q: {frame_info.get("quality", 0.5):.3f})', 
                               fontsize=12, fontweight='bold')
                    
                    # Add frame info as text overlay
                    ax.text(0.02, 0.98, f"Conf: {frame_info['confidence']:.3f}", 
                           transform=ax.transAxes, fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                           verticalalignment='top')
                else:
                    # Fallback to text display if crop extraction fails
                    quality = frame_info.get('quality', 0.5)
                    color = plt.cm.RdYlGn(quality)  # Red for low quality, Green for high quality
                    
                    ax.text(0.5, 0.5, f"Frame {frame_info['frame_num']}\n"
                                      f"Position: ({frame_info['center_x']}, {frame_info['center_y']})\n"
                                      f"Confidence: {frame_info['confidence']:.3f}\n"
                                      f"Quality: {quality:.3f}\n"
                                      f"(Image not available)",
                           ha='center', va='center', fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.7))
                    ax.set_title(f'Sample {i+1} (Quality: {quality:.3f})', fontsize=12, fontweight='bold')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
            else:
                ax.text(0.5, 0.5, 'No more samples', ha='center', va='center', fontsize=12)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            
            ax.axis('off')
        
        # Add comprehensive statistics text
        qualities = track_data.get('qualities', [0.5])
        if not qualities:  # Handle empty qualities list
            qualities = [0.5]
        
        stats_text = f"""
Track Statistics:
• Track ID: {track_id}
• Total Embeddings: {len(track_data.get('embeddings', []))}
• Total History Frames: {len(track_data.get('history', []))}
• Average Quality: {np.mean(qualities):.3f}
• Max Quality: {np.max(qualities):.3f}
• Min Quality: {np.min(qualities):.3f}

Sample Frames: {len(sample_frames)}
Frame Numbers: {[f['frame_num'] for f in sample_frames]}

Video: {self.video_path if self.video_path else 'Not available'}
Crops: {'Available' if self.video_cap else 'Not available'}
"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Save the figure
        sample_path = self.samples_dir / f"track_{track_id}_person_crops.png"
        plt.savefig(sample_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return sample_path
    
    def interactive_naming(self) -> Dict[int, str]:
        """
        Interactive naming process for tracks that represent NEW persons only
        Tracks already associated with known persons will be skipped
        
        Returns:
            Dictionary mapping track_id to user-provided name
        """
        print("\n🎯 Interactive Track Naming Process")
        print("=" * 80)
        
        # Collect all track data
        track_data = self.collect_track_data()
        
        if not track_data:
            print("⚠️  No track data found for naming")
            return {}
        
        print(f"📊 Found {len(track_data)} tracks to process")
        
        # Filter tracks to only include NEW persons (not already identified)
        new_person_tracks = {}
        known_person_tracks = {}
        
        for track_id, data in track_data.items():
            # Check if this track is associated with an existing person in the gallery
            associated_person = self.simple_gallery.track_to_person.get(track_id)
            
            if associated_person and associated_person in self.simple_gallery.gallery:
                known_person_tracks[track_id] = associated_person
                print(f"   ✅ Track {track_id} already identified as: {associated_person}")
            else:
                new_person_tracks[track_id] = data
                print(f"   🆕 Track {track_id} is a NEW person (needs naming)")
        
        if not new_person_tracks:
            print("\n✅ All tracks are already identified! No naming needed.")
            return {}
        
        print(f"\n📋 Summary:")
        print(f"   • Known persons: {len(known_person_tracks)} tracks")
        print(f"   • New persons: {len(new_person_tracks)} tracks (will ask for names)")
        
        # Sort new person tracks by number of embeddings (most active first)
        sorted_tracks = sorted(new_person_tracks.items(), 
                             key=lambda x: len(x[1].get('embeddings', [])), 
                             reverse=True)
        
        track_names = {}
        
        for track_id, data in sorted_tracks:
            print(f"\n🎭 Processing Track {track_id}")
            print("-" * 60)
            
            # Create and display visualization
            viz_path = self.create_track_visualization(track_id, data)
            print(f"   📸 Person crops visualization saved: {viz_path}")
            
            # Create comprehensive gallery if video is available
            gallery_path = None
            if self.video_cap:
                gallery_path = self.create_comprehensive_crop_gallery(track_id, data)
                if gallery_path:
                    print(f"   🖼️  Comprehensive crop gallery: {gallery_path}")
            
            # Show crop availability status
            if self.video_cap:
                print(f"   🎬 Video available: {self.video_path}")
                print(f"   🖼️  Person crops: Available for display")
            else:
                print(f"   ⚠️  Video not available - showing frame info only")
            
            # Display track statistics
            embeddings = data.get('embeddings', [])
            qualities = data.get('qualities', [])
            if not qualities:  # Handle empty qualities list
                qualities = [0.5]
            history = data.get('history', [])
            
            print(f"   📊 Track Statistics:")
            print(f"      • Embeddings: {len(embeddings)}")
            print(f"      • History frames: {len(history)}")
            print(f"      • Avg quality: {np.mean(qualities):.3f}")
            print(f"      • Quality range: [{np.min(qualities):.3f}, {np.max(qualities):.3f}]")
            
            if history:
                frame_range = [h[2] for h in history]  # frame numbers
                print(f"      • Frame range: {min(frame_range)} - {max(frame_range)}")
                print(f"      • Duration: {max(frame_range) - min(frame_range)} frames")
                
                # Show sample frame info
                sample_frames = self.find_best_sample_frames(track_id, data)
                if sample_frames:
                    print(f"      • Sample frames: {[f['frame_num'] for f in sample_frames]}")
                    if self.video_cap:
                        print(f"      • Person crops: Generated for {len(sample_frames)} frames")
            
            # Ask for track name
            while True:
                try:
                    print(f"\n🏷️  Options for Track {track_id}:")
                    print(f"      1. Enter a person name")
                    print(f"      2. Type 'skip' to skip this track")
                    print(f"      3. Type 'quit' to stop naming process")
                    print(f"      4. Type 'show' to view person crops again")
                    if gallery_path:
                        print(f"      5. Type 'gallery' to view comprehensive crop gallery")
                        print(f"      6. Type 'open' to open visualization image")
                    else:
                        print(f"      5. Type 'open' to open visualization image")
                    
                    response = input(f"\n➤ Your choice for Track {track_id}: ").strip()
                    
                    if response.lower() == 'quit':
                        print("\n⏹️  Naming process stopped by user")
                        return track_names
                    
                    elif response.lower() == 'skip':
                        print(f"   ⏭️  Skipped Track {track_id}")
                        break
                    
                    elif response.lower() == 'show':
                        print(f"   👀 Please check the person crops at: {viz_path}")
                        if self.video_cap:
                            print(f"   🖼️  Visualization shows actual person crops from video")
                        else:
                            print(f"   ⚠️  Visualization shows frame info only (video not available)")
                        continue
                    
                    elif response.lower() == 'gallery' and gallery_path:
                        print(f"   🖼️  Comprehensive gallery available at: {gallery_path}")
                        print(f"   💡 This shows more person crops with quality indicators")
                        try:
                            import subprocess
                            subprocess.run(['open', str(gallery_path)], check=True)
                            print(f"   📖 Opened comprehensive gallery in default image viewer")
                        except Exception as e:
                            print(f"   ❌ Could not open image: {e}")
                            print(f"   💡 Please manually open: {gallery_path}")
                        continue
                    
                    elif response.lower() == 'open':
                        try:
                            import subprocess
                            subprocess.run(['open', str(viz_path)], check=True)
                            print(f"   📖 Opened visualization in default image viewer")
                        except Exception as e:
                            print(f"   ❌ Could not open image: {e}")
                            print(f"   💡 Please manually open: {viz_path}")
                        continue
                    
                    elif response:
                        # Valid name entered
                        track_names[track_id] = response
                        print(f"   ✅ Track {track_id} named: '{response}'")
                        break
                    
                    else:
                        print("   ❌ Please enter a valid option")
                        
                except KeyboardInterrupt:
                    print("\n\n⏹️  Naming process interrupted by user")
                    return track_names
                except EOFError:
                    print("\n\n⏹️  Naming process stopped (EOF)")
                    return track_names
        
        return track_names
    
    def merge_tracks_by_name(self, track_names: Dict[int, str]) -> int:
        """
        Merge tracks that have the same name
        
        Args:
            track_names: Dictionary mapping track_id to name
            
        Returns:
            Number of tracks merged
        """
        print(f"\n🔗 Merging Tracks with Same Names")
        print("-" * 50)
        
        # Group tracks by name
        name_groups = {}
        for track_id, name in track_names.items():
            if name not in name_groups:
                name_groups[name] = []
            name_groups[name].append(track_id)
        
        merged_count = 0
        
        # Merge tracks with same names
        for name, track_ids in name_groups.items():
            if len(track_ids) > 1:
                print(f"\n🎯 Merging tracks with name '{name}': {track_ids}")
                
                # Find the track with most embeddings to be the primary
                track_data = self.collect_track_data()
                primary_track = max(track_ids, key=lambda tid: len(track_data.get(tid, {}).get('embeddings', [])))
                other_tracks = [tid for tid in track_ids if tid != primary_track]
                
                print(f"   • Primary track: {primary_track}")
                print(f"   • Other tracks: {other_tracks}")
                
                # Merge embeddings into primary track's person
                primary_person = None
                for person_name, person_data in self.simple_gallery.gallery.items():
                    if primary_track in person_data.track_associations:
                        primary_person = person_name
                        break
                
                if primary_person:
                    print(f"   • Primary person: {primary_person}")
                    
                    # Merge other tracks' embeddings into primary person
                    for track_id in other_tracks:
                        # Find person associated with this track
                        other_person = None
                        for person_name, person_data in self.simple_gallery.gallery.items():
                            if track_id in person_data.track_associations:
                                other_person = person_name
                                break
                        
                        if other_person and other_person != primary_person:
                            print(f"   • Merging {other_person} into {primary_person}")
                            
                            # Merge other_person into primary_person
                            primary_data = self.simple_gallery.gallery[primary_person]
                            other_data = self.simple_gallery.gallery[other_person]
                            
                            # Combine embeddings
                            primary_data.embeddings.extend(other_data.embeddings)
                            primary_data.qualities.extend(other_data.qualities)
                            primary_data.track_associations.extend(other_data.track_associations)
                            primary_data.update_count += other_data.update_count
                            
                            # Trim to max size if needed
                            if len(primary_data.embeddings) > self.simple_gallery.max_embeddings_per_person:
                                # Keep the best embeddings
                                indices = np.argsort(primary_data.qualities)[::-1][:self.simple_gallery.max_embeddings_per_person]
                                primary_data.embeddings = [primary_data.embeddings[i] for i in indices]
                                primary_data.qualities = [primary_data.qualities[i] for i in indices]
                            
                            # Recompute prototype
                            primary_data.prototype = self.simple_gallery._compute_prototype(
                                primary_data.embeddings, primary_data.qualities
                            )
                            
                            # Remove the other person
                            del self.simple_gallery.gallery[other_person]
                            
                            # Update track mappings
                            for tid in other_data.track_associations:
                                self.simple_gallery.track_to_person[tid] = primary_person
                            
                            merged_count += 1
                            print(f"   ✅ Merged {other_person} into {primary_person}")
                
                # Update the primary person's name if provided
                if primary_person and name != primary_person:
                    print(f"   • Renaming {primary_person} to {name}")
                    
                    # Create new person with the user-provided name
                    person_data = self.simple_gallery.gallery[primary_person]
                    person_data.person_name = name  # Update the person_name field
                    self.simple_gallery.gallery[name] = person_data
                    del self.simple_gallery.gallery[primary_person]
                    
                    # Update track mappings
                    for tid in self.simple_gallery.gallery[name].track_associations:
                        self.simple_gallery.track_to_person[tid] = name
                    
                    # Update person_to_track mapping
                    if primary_person in self.simple_gallery.person_to_track:
                        track_id = self.simple_gallery.person_to_track[primary_person]
                        del self.simple_gallery.person_to_track[primary_person]
                        self.simple_gallery.person_to_track[name] = track_id
                    
                    print(f"   ✅ Renamed {primary_person} to {name}")
                else:
                    # Even if no renaming needed, ensure the person has the correct user-provided name
                    if primary_person and primary_person != name:
                        # This handles cases where the primary person might have a generic name
                        person_data = self.simple_gallery.gallery[primary_person]
                        person_data.person_name = name  # Update the person_name field
                        self.simple_gallery.gallery[name] = person_data
                        del self.simple_gallery.gallery[primary_person]
                        
                        # Update track mappings
                        for tid in self.simple_gallery.gallery[name].track_associations:
                            self.simple_gallery.track_to_person[tid] = name
                        
                        # Update person_to_track mapping
                        if primary_person in self.simple_gallery.person_to_track:
                            track_id = self.simple_gallery.person_to_track[primary_person]
                            del self.simple_gallery.person_to_track[primary_person]
                            self.simple_gallery.person_to_track[name] = track_id
                        
                        print(f"   ✅ Updated person name from {primary_person} to {name}")
        
        # Handle single tracks that were named (not merged) - CREATE NEW PERSONS
        for track_id, name in track_names.items():
            # Check if this track is already associated with a person
            current_person = self.simple_gallery.track_to_person.get(track_id)
            
            if current_person and current_person in self.simple_gallery.gallery:
                # Track is already assigned - rename the existing person
                if current_person != name:
                    print(f"\n🏷️  Updating existing person name:")
                    print(f"   • Track {track_id}: {current_person} → {name}")
                    
                    # Update the person_name field and create new person with user-provided name
                    person_data = self.simple_gallery.gallery[current_person]
                    person_data.person_name = name  # Update the person_name field
                    self.simple_gallery.gallery[name] = person_data
                    del self.simple_gallery.gallery[current_person]
                    
                    # Update track mappings
                    for tid in self.simple_gallery.gallery[name].track_associations:
                        self.simple_gallery.track_to_person[tid] = name
                    
                    # Update person_to_track mapping
                    if current_person in self.simple_gallery.person_to_track:
                        track_id_mapped = self.simple_gallery.person_to_track[current_person]
                        del self.simple_gallery.person_to_track[current_person]
                        self.simple_gallery.person_to_track[name] = track_id_mapped
                    
                    print(f"   ✅ Updated {current_person} to {name}")
            else:
                # Track is NOT assigned - create a NEW person with user-provided name
                track_data = self.collect_track_data()
                if track_id in track_data:
                    embeddings = track_data[track_id].get('embeddings', [])
                    qualities = track_data[track_id].get('qualities', [])
                    
                    if embeddings:
                        print(f"\n🆕 Creating NEW person:")
                        print(f"   • Track {track_id} → {name}")
                        print(f"   • Embeddings: {len(embeddings)}")
                        
                        # Create new person using the gallery method
                        self.simple_gallery.create_person_from_track(
                            person_name=name,
                            track_id=track_id,
                            embeddings=embeddings,
                            qualities=qualities
                        )
                        
                        print(f"   ✅ Created new person: {name}")
                    else:
                        print(f"   ⚠️  Cannot create person {name} for track {track_id} - no embeddings")
        
        return merged_count
    
    def save_naming_results(self, track_names: Dict[int, str], merged_count: int) -> Path:
        """
        Save the naming results to a JSON file
        
        Args:
            track_names: Dictionary mapping track_id to name
            merged_count: Number of tracks merged
            
        Returns:
            Path to the saved results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"track_naming_results_{timestamp}.json"
        
        results = {
            'timestamp': timestamp,
            'total_tracks_named': len(track_names),
            'tracks_merged': merged_count,
            'final_persons': len(self.simple_gallery.gallery),
            'track_names': {str(k): v for k, v in track_names.items()},  # Convert int keys to strings for JSON
            'final_gallery': {
                person_name: {
                    'track_associations': list(person_data.track_associations),
                    'num_embeddings': len(person_data.embeddings),
                    'avg_quality': float(np.mean(person_data.qualities)) if person_data.qualities else 0.0,
                    'update_count': person_data.update_count
                }
                for person_name, person_data in self.simple_gallery.gallery.items()
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results_file
    
    def run_full_naming_process(self) -> Tuple[Dict[int, str], int]:
        """
        Run the complete interactive naming process
        
        Returns:
            Tuple of (track_names, merged_count)
        """
        print("\n🚀 Starting Full Track Naming Process")
        print("=" * 80)
        
        # Step 1: Interactive naming
        track_names = self.interactive_naming()
        
        if not track_names:
            print("⚠️  No tracks were named")
            return {}, 0
        
        # Step 2: Merge tracks with same names
        merged_count = self.merge_tracks_by_name(track_names)
        
        # Step 3: Save results
        results_file = self.save_naming_results(track_names, merged_count)
        
        # Step 4: Print summary
        print(f"\n✅ Track Naming Process Completed!")
        print(f"   📊 Summary:")
        print(f"      • Tracks named: {len(track_names)}")
        print(f"      • Tracks merged: {merged_count}")
        print(f"      • Final persons: {len(self.simple_gallery.gallery)}")
        print(f"      • Results saved: {results_file}")
        
        return track_names, merged_count
    
    def create_comprehensive_crop_gallery(self, track_id: int, track_data: Dict[str, Any], 
                                         max_crops: int = 12) -> Path:
        """
        Create a comprehensive gallery showing more person crops for better identification
        
        Args:
            track_id: Track ID
            track_data: Track data dictionary
            max_crops: Maximum number of crops to show
            
        Returns:
            Path to the saved gallery
        """
        # Get more samples for comprehensive view
        sample_frames = self.find_best_sample_frames(track_id, track_data, max_samples=max_crops)
        
        if not sample_frames:
            print(f"⚠️  No sample frames available for track {track_id}")
            return None
        
        # Calculate grid layout
        rows = int(np.ceil(len(sample_frames) / 4))
        cols = min(4, len(sample_frames))
        
        # Create comprehensive gallery
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        fig.suptitle(f'Track {track_id} - Comprehensive Person Crop Gallery', 
                    fontsize=18, fontweight='bold')
        
        # Handle single row case properly
        if rows == 1:
            if len(sample_frames) == 1:
                axes = [axes]  # Single subplot case
            else:
                axes = axes.reshape(1, -1)  # Multiple subplots in one row
        
        crops_extracted = 0
        
        for i, frame_info in enumerate(sample_frames):
            row = i // cols
            col = i % cols
            
            # Get the correct axis
            if rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            # Extract person crop
            crop = self.extract_person_crop(
                frame_info['frame_num'], 
                frame_info['center_x'], 
                frame_info['center_y'],
                crop_size=(160, 320)  # Larger crop for better detail
            )
            
            if crop is not None:
                # Display the crop
                ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                ax.set_title(f'Frame {frame_info["frame_num"]}\nQ: {frame_info.get("quality", 0.5):.3f}', 
                           fontsize=10, fontweight='bold')
                crops_extracted += 1
                
                # Add quality indicator border
                quality = frame_info.get('quality', 0.5)
                if quality > 0.7:
                    border_color = 'green'
                elif quality > 0.4:
                    border_color = 'orange'
                else:
                    border_color = 'red'
                
                for spine in ax.spines.values():
                    spine.set_color(border_color)
                    spine.set_linewidth(3)
                    
            else:
                # Show placeholder if crop extraction fails
                ax.text(0.5, 0.5, f'Frame {frame_info["frame_num"]}\nCrop not available', 
                       ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_title(f'Frame {frame_info["frame_num"]}', fontsize=10)
            
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(len(sample_frames), rows * cols):
            row = i // cols
            col = i % cols
            
            # Get the correct axis for unused subplots
            if rows == 1:
                if col < len(axes):
                    ax = axes[col]
                else:
                    continue
            else:
                ax = axes[row, col]
            ax.axis('off')
        
        # Add comprehensive statistics
        qualities = track_data.get('qualities', [0.5])
        if not qualities:  # Handle empty qualities list
            qualities = [0.5]
            
        stats_text = f"""
Track {track_id} Details:
• Embeddings: {len(track_data.get('embeddings', []))}
• History frames: {len(track_data.get('history', []))}
• Crops shown: {crops_extracted}/{len(sample_frames)}
• Avg quality: {np.mean(qualities):.3f}
• Quality range: [{np.min(qualities):.3f}, {np.max(qualities):.3f}]

Border colors: Green (high quality), Orange (medium), Red (low quality)
"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8))
        
        # Save the comprehensive gallery
        gallery_path = self.samples_dir / f"track_{track_id}_comprehensive_gallery.png"
        plt.savefig(gallery_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return gallery_path
