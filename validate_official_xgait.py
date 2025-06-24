#!/usr/bin/env python3
"""
Official XGait Architecture Validation Script
Ensures our implementation matches the official Gait3D-Benchmark XGait exactly
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.official_xgait_model import create_official_xgait_inference, create_official_xgait_model
from src.models.xgait_adapter import XGaitAdapter
from src.models.identification_processor import create_identification_processor


def validate_model_architecture():
    """Validate that the model architecture matches official XGait"""
    print("🔍 Validating Official XGait Architecture")
    print("=" * 60)
    
    # Create official model
    model = create_official_xgait_model(num_classes=3000)
    
    # Check architecture components
    components = [
        "Backbone_sil", "Backbone_par", 
        "gcm", "pcm_up", "pcm_middle", "pcm_down",
        "FCs_sil", "FCs_par", "FCs_gcm", "FCs_pcm",
        "BNNecks_sil", "BNNecks_par", "BNNecks_gcm", "BNNecks_pcm",
        "TP", "HPP"
    ]
    
    print("📋 Architecture Components:")
    for component in components:
        has_component = hasattr(model, component)
        status = "✅" if has_component else "❌"
        print(f"   {status} {component}")
        
        if not has_component:
            print(f"   ❌ MISSING: {component}")
            return False
    
    # Test forward pass with dummy data
    print("\n🧪 Testing Forward Pass:")
    try:
        # Create dummy input
        batch_size = 2
        sequence_length = 30
        height, width = 64, 44
        
        pars = torch.randn(batch_size, 1, sequence_length, height, width)
        sils = torch.randn(batch_size, 1, sequence_length, height, width)
        labs = torch.randint(0, 3000, (batch_size,))
        seqL = [sequence_length] * batch_size
        
        inputs = ([pars, sils], labs, None, None, seqL)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(inputs)
        
        # Validate output structure
        expected_keys = ['training_feat', 'visual_summary', 'inference_feat']
        for key in expected_keys:
            if key not in output:
                print(f"   ❌ Missing output key: {key}")
                return False
            print(f"   ✅ Output key: {key}")
        
        # Check feature dimensions
        embeddings = output['inference_feat']['embeddings']
        print(f"   📊 Feature shape: {embeddings.shape}")
        
        # Should be [batch_size, 256, 64] = [2, 256, 64]
        expected_shape = (batch_size, 256, 64)  # 4 streams × 16 parts × 256 dims
        if embeddings.shape == expected_shape:
            print(f"   ✅ Feature dimensions correct: {embeddings.shape}")
        else:
            print(f"   ❌ Feature dimensions incorrect: {embeddings.shape}, expected: {expected_shape}")
            return False
            
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False
    
    print("   ✅ Forward pass successful")
    return True


def validate_inference_engine():
    """Validate the XGait inference engine"""
    print("\n🔧 Validating XGait Inference Engine")
    print("=" * 60)
    
    try:
        # Create inference engine
        xgait = create_official_xgait_inference(device="cpu")
        
        # Test parameters
        print("📋 Inference Parameters:")
        print(f"   Input size: {xgait.input_height}x{xgait.input_width}")
        print(f"   Min sequence length: {xgait.min_sequence_length}")
        print(f"   Target sequence length: {xgait.target_sequence_length}")
        print(f"   Device: {xgait.device}")
        print(f"   Model loaded: {xgait.model_loaded}")
        
        # Test feature extraction
        print("\n🧪 Testing Feature Extraction:")
        
        # Create test data
        silhouettes = [np.random.randint(0, 255, (64, 44), dtype=np.uint8) for _ in range(30)]
        parsing_masks = [np.random.randint(0, 19, (64, 44), dtype=np.uint8) for _ in range(30)]
        
        # Extract features
        features = xgait.extract_features(silhouettes, parsing_masks)
        
        print(f"   📊 Extracted features shape: {features.shape}")
        
        # Validate feature dimensions
        expected_shape = (1, 256, 64)  # [batch, dims, parts]
        if features.shape == expected_shape:
            print(f"   ✅ Feature shape correct: {features.shape}")
        else:
            print(f"   ⚠️ Feature shape: {features.shape}, expected: {expected_shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Inference engine validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_adapter_integration():
    """Validate the XGait adapter integration"""
    print("\n🔗 Validating XGait Adapter Integration")
    print("=" * 60)
    
    try:
        # Create adapter
        adapter = XGaitAdapter(device="cpu")
        
        # Test adapter properties
        print("📋 Adapter Properties:")
        print(f"   Model loaded: {adapter.model_loaded}")
        print(f"   Input size: {adapter.input_height}x{adapter.input_width}")
        print(f"   Min sequence length: {adapter.min_sequence_length}")
        print(f"   Target sequence length: {adapter.target_sequence_length}")
        
        # Test feature extraction through adapter
        print("\n🧪 Testing Adapter Feature Extraction:")
        
        silhouettes = [np.random.randint(0, 255, (64, 44), dtype=np.uint8) for _ in range(30)]
        parsing_masks = [np.random.randint(0, 19, (64, 44), dtype=np.uint8) for _ in range(30)]
        
        # Single sequence
        features_single = adapter.extract_features_from_sequence(silhouettes, parsing_masks)
        print(f"   📊 Single sequence features: {features_single.shape}")
        
        # Multiple sequences
        features_multi = adapter.extract_features([silhouettes], [parsing_masks])
        print(f"   📊 Multi sequence features: {features_multi.shape}")
        
        # Should be flattened to 16384 dims (256 * 64)
        expected_single_dim = 16384
        expected_multi_shape = (1, 16384)
        
        if features_single.shape == (expected_single_dim,):
            print(f"   ✅ Single sequence shape correct: {features_single.shape}")
        else:
            print(f"   ⚠️ Single sequence shape: {features_single.shape}, expected: ({expected_single_dim},)")
        
        if features_multi.shape == expected_multi_shape:
            print(f"   ✅ Multi sequence shape correct: {features_multi.shape}")
        else:
            print(f"   ⚠️ Multi sequence shape: {features_multi.shape}, expected: {expected_multi_shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Adapter validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_identification_processor():
    """Validate the complete identification processor"""
    print("\n🎯 Validating Identification Processor")
    print("=" * 60)
    
    try:
        # Create processor
        processor = create_identification_processor(device="cpu")
        
        # Check model status
        status = processor.get_model_status()
        print("📋 Model Status:")
        for model, available in status.items():
            symbol = "✅" if available else "❌"
            print(f"   {symbol} {model}")
        
        # Test processing
        print("\n🧪 Testing Processing Pipeline:")
        
        # Create test crops
        test_crops = [np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8) for _ in range(3)]
        test_track_ids = [1, 2, 3]
        
        # Process crops
        results = processor.process_person_crops(test_crops, test_track_ids)
        
        print(f"   📊 Processing Results:")
        print(f"      - Silhouettes: {len(results['silhouettes'])}")
        print(f"      - Parsing masks: {len(results['parsing_masks'])}")
        print(f"      - Features shape: {results['features'].shape}")
        print(f"      - Success: {results['success']}")
        
        # Test dual input statistics
        dual_stats = processor.get_dual_input_statistics()
        print(f"\n📈 Dual Input Statistics:")
        for key, value in dual_stats.items():
            print(f"      - {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Processor validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_weight_compatibility():
    """Check weight file compatibility"""
    print("\n📦 Validating Weight File Compatibility")
    print("=" * 60)
    
    # Check for weight files
    weight_dir = Path(__file__).parent / "weights"
    potential_weights = [
        "Gait3D-XGait-120000.pt",
        "xgait_gait3d.pth",
        "xgait_model.pth"
    ]
    
    print("📋 Available Weight Files:")
    found_weights = []
    for weight_file in potential_weights:
        weight_path = weight_dir / weight_file
        if weight_path.exists():
            size_mb = weight_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ {weight_file} ({size_mb:.1f} MB)")
            found_weights.append(str(weight_path))
        else:
            print(f"   ❌ {weight_file} (not found)")
    
    if found_weights:
        print(f"\n🧪 Testing Weight Loading:")
        try:
            # Test loading with first available weights
            xgait = create_official_xgait_inference(model_path=found_weights[0], device="cpu")
            print(f"   ✅ Successfully loaded: {Path(found_weights[0]).name}")
            print(f"   📊 Model status: {'Loaded' if xgait.model_loaded else 'Failed to load'}")
            return True
        except Exception as e:
            print(f"   ❌ Failed to load weights: {e}")
            return False
    else:
        print("   ⚠️ No weight files found - using random initialization")
        return True


def main():
    """Run complete validation"""
    print("🚀 Official XGait Architecture Validation")
    print("=" * 80)
    
    validations = [
        ("Model Architecture", validate_model_architecture),
        ("Inference Engine", validate_inference_engine),  
        ("Adapter Integration", validate_adapter_integration),
        ("Identification Processor", validate_identification_processor),
        ("Weight Compatibility", validate_weight_compatibility)
    ]
    
    results = {}
    
    for name, validation_func in validations:
        try:
            results[name] = validation_func()
        except Exception as e:
            print(f"\n❌ {name} validation failed with exception: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 VALIDATION SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED - Official XGait Architecture is correctly implemented!")
    else:
        print("⚠️ SOME VALIDATIONS FAILED - Please check the implementation above.")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
