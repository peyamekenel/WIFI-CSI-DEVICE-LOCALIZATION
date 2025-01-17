import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
from tqdm import tqdm

# Add reference implementation path
reference_path = Path('/home/ubuntu/reference_haloc')
sys.path.append(str(reference_path))

# Import both implementations
from HALOC.datasets import HALOC as ReferenceHALOC, csi_vaid_subcarrier_index as ref_subcarrier_index
from resnet_spectrogram_dataloader import ResnetSpectrogramDataset, CSI_SUBCARRIERS

def compare_implementations():
    """Compare our CSI processing with reference implementation."""
    print("\nComparing CSI Processing Implementations:")
    print("----------------------------------------")
    
    # Initialize both implementations with same data
    data_dir = Path('HALOC')
    window_size = 351
    
    print("\n1. Loading implementations...")
    # Our implementation
    our_dataset = ResnetSpectrogramDataset(data_dir, window_size=window_size, split='0.csv')
    
    # Reference implementation
    ref_dataset = ReferenceHALOC(str(data_dir / '0.csv'), windowSize=window_size)
    
    print("\n2. Comparing subcarrier selection...")
    print(f"Our subcarriers: {len(ResnetSpectrogramDataset.csi_valid_subcarrier_index)}")
    print(f"Reference subcarriers: {len(ref_subcarrier_index)}")
    
    # Compare subcarrier indices
    print("\nSubcarrier indices comparison:")
    our_indices = ResnetSpectrogramDataset.csi_valid_subcarrier_index
    ref_indices = ref_subcarrier_index
    
    print(f"First range (6-31):")
    print(f"Our impl:   {our_indices[:26]}")
    print(f"Reference:  {ref_indices[:26]}")
    print(f"\nSecond range (33-58):")
    print(f"Our impl:   {our_indices[26:]}")
    print(f"Reference:  {ref_indices[26:]}")
    
    # Verify exact match
    indices_match = np.array_equal(our_indices, ref_indices)
    print(f"\nIndices match exactly: {indices_match}")
    
    print("\n3. Comparing feature extraction...")
    # Get first sample from both
    our_features = our_dataset.features[0]
    ref_features = ref_dataset.features[0]
    
    print("\nFeature shapes:")
    print(f"Our impl: {our_features.shape}")
    print(f"Reference: {ref_features.shape}")
    
    print("\nFirst 5 feature values:")
    print(f"Our impl: {our_features[:5]}")
    print(f"Reference: {ref_features[:5]}")
    
    # Compare statistics
    print("\n4. Comparing feature statistics...")
    our_stats = {
        'min': np.min(our_features),
        'max': np.max(our_features),
        'mean': np.mean(our_features),
        'std': np.std(our_features)
    }
    ref_stats = {
        'min': np.min(ref_features),
        'max': np.max(ref_features),
        'mean': np.mean(ref_features),
        'std': np.std(ref_features)
    }
    
    print("\nOur implementation:")
    for k, v in our_stats.items():
        print(f"{k}: {v:.6f}")
    
    print("\nReference implementation:")
    for k, v in ref_stats.items():
        print(f"{k}: {v:.6f}")
    
    # Calculate differences
    print("\n5. Calculating differences...")
    abs_diff = np.abs(our_features - ref_features)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    print(f"Maximum absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    
    # Verify window handling
    print("\n6. Verifying window handling...")
    our_window, our_pos = our_dataset[0]
    ref_window, ref_pos = ref_dataset[0]
    
    print("\nWindow shapes:")
    print(f"Our impl: {our_window.shape}")
    print(f"Reference: {ref_window.shape}")
    
    print("\nPosition values:")
    print(f"Our impl: {our_pos}")
    print(f"Reference: {ref_pos}")
    
    # Convert lists to numpy arrays for comparison
    our_indices_np = np.array(our_indices)
    ref_indices_np = np.array(ref_indices)
    
    return {
        'subcarrier_match': indices_match,
        'feature_shape_match': our_features.shape == ref_features.shape,
        'max_difference': float(max_diff),
        'mean_difference': float(mean_diff),
        'window_shape_match': our_window.shape == ref_window.shape,
        'details': {
            'our_indices': our_indices_np.tolist(),
            'ref_indices': ref_indices_np.tolist(),
            'our_features_first3': our_features[:3].tolist() if isinstance(our_features, np.ndarray) else our_features[:3],
            'ref_features_first3': ref_features[:3].tolist() if isinstance(ref_features, np.ndarray) else ref_features[:3]
        }
    }

if __name__ == '__main__':
    results = compare_implementations()
    
    print("\nVerification Results:")
    print("--------------------")
    for k, v in results.items():
        print(f"{k}: {v}")
    
    # Final verdict
    if all([
        results['subcarrier_match'],
        results['feature_shape_match'],
        results['max_difference'] < 1e-6,
        results['window_shape_match']
    ]):
        print("\n✓ Implementation matches reference")
    else:
        print("\n✗ Implementation differences found")
