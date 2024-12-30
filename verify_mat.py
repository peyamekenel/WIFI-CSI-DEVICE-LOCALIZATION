import os
import h5py
import numpy as np

def quick_verify_mat(filepath):
    """Quickly verify .mat file structure without loading full arrays."""
    try:
        with h5py.File(filepath, 'r') as f:
            # Check if csi_complex_data exists
            if 'csi_complex_data' not in f:
                return False, "Missing csi_complex_data"
            
            # Get shape without loading data
            shape = f['csi_complex_data'].shape
            return True, shape
    except Exception as e:
        return False, str(e)

def main():
    print("CSI Data Verification Report")
    print("-" * 50)
    
    # Get all .mat files from data directory
    data_dir = 'data'
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat')])
    files = [os.path.join(data_dir, f) for f in files]
    print(f"Found {len(files)} .mat files\n")
    
    angles = set()
    distances = set()
    
    for filename in files:
        # Extract metadata from filename
        parts = filename.replace('.mat', '').split('_')
        angle_str = parts[1].replace('deg', '')
        # Handle negative angles (e.g., 'minus60' -> -60)
        if angle_str.startswith('minus'):
            angle = -int(angle_str.replace('minus', ''))
        else:
            angle = int(angle_str)
        distance = int(parts[2].replace('m', ''))
        
        # Verify file
        success, result = quick_verify_mat(filename)
        
        if success:
            print(f"✓ {filename}")
            print(f"  Shape: {result}")
            print(f"  Angle: {angle}°")
            print(f"  Distance: {distance}m\n")
            angles.add(angle)
            distances.add(distance)
        else:
            print(f"✗ {filename}")
            print(f"  Error: {result}\n")
    
    # Verify dataset completeness
    print("Dataset Summary")
    print("-" * 50)
    print(f"Angles found: {sorted(angles)}°")
    print(f"Distances found: {sorted(distances)}m")
    
    # Check for missing measurements
    expected_angles = {30, -60}
    expected_distances = set(range(1, 6))
    
    missing_angles = expected_angles - angles
    missing_distances = expected_distances - distances
    
    if not missing_angles and not missing_distances:
        print("\n✓ Dataset complete - all angles and distances present")
    else:
        if missing_angles:
            print(f"\n✗ Missing angles: {missing_angles}")
        if missing_distances:
            print(f"\n✗ Missing distances: {missing_distances}")

if __name__ == "__main__":
    main()
