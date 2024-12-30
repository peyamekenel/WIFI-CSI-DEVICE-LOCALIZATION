import numpy as np
from pathlib import Path

def inspect_feature_files():
    """Inspect the content and structure of saved feature files."""
    feature_dir = Path('csi_features')
    if not feature_dir.exists():
        print(f"Error: Feature directory not found at {feature_dir}")
        return
        
    print("Feature File Analysis")
    print("-" * 50)
    
    for feature_file in sorted(feature_dir.glob('*.npy')):
        try:
            features = np.load(feature_file)
            print(f"\nFile: {feature_file.name}")
            print(f"Array shape: {features.shape}")
            print(f"Array type: {features.dtype}")
            print(f"Value range: [{features.min():.3f}, {features.max():.3f}]")
            print(f"First row shape: {features[0].shape if features.ndim > 1 else 'N/A'}")
            print(f"Memory usage: {features.nbytes / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            print(f"Error processing {feature_file.name}: {str(e)}")

if __name__ == "__main__":
    inspect_feature_files()
