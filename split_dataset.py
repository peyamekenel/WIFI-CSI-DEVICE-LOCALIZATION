import os
import numpy as np
from sklearn.model_selection import train_test_split
import h5py
import pickle

def load_preprocessed_features(feature_dir='csi_features'):
    """Load all preprocessed CSI features and organize into a dataset."""
    X = []  # Features
    y = []  # Labels (distance, angle)
    
    # Process each .mat file in the data directory
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found at {data_dir}")
        
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith('.mat'):
            continue
            
        # Extract distance and angle from filename
        parts = filename.replace('.mat', '').split('_')
        angle_str = parts[1].replace('deg', '')
        angle = -int(angle_str.replace('minus', '')) if angle_str.startswith('minus') else int(angle_str)
        distance = int(parts[2].replace('m', ''))
        
        print(f"Processing {filename} - Distance: {distance}m, Angle: {angle}°")
        
        # Load and preprocess data (using same functions from preprocess_csi.py)
        file_path = os.path.join(data_dir, filename)
        with h5py.File(file_path, 'r') as f:
            # Get total number of packets
            total_packets = f['csi_complex_data'].shape[0]
            # Use random sampling
            max_packets = 10000
            if total_packets > max_packets:
                indices = np.random.choice(total_packets, max_packets, replace=False)
                indices.sort()
                raw_data = f['csi_complex_data'][indices]
            else:
                raw_data = f['csi_complex_data'][:]
            
            # Convert to complex numbers and transpose
            csi_data = raw_data['real'] + 1j * raw_data['imag']
            csi_data = np.transpose(csi_data, (2, 1, 0))
            
            # Extract features (simplified version focusing on key statistics)
            epsilon = 1e-10
            amplitude = np.abs(csi_data)
            amplitude_db = 20 * np.log10(np.maximum(amplitude, epsilon))
            phase = np.angle(csi_data)
            
            # Compute features per packet
            n_packets = amplitude_db.shape[2]
            
            # Reshape amplitude and phase features (3 antennas × 30 subcarriers = 90 features each)
            amplitude_features = amplitude_db.reshape(3 * 30, n_packets).T
            phase_features = phase.reshape(3 * 30, n_packets).T
            
            # Compute statistical features across all antennas and subcarriers
            amplitude_stats = np.vstack([
                np.std(amplitude_db, axis=(0, 1)),   # Standard deviation per packet
                np.var(amplitude_db, axis=(0, 1)),   # Variance per packet
                np.mean(amplitude_db, axis=(0, 1)),  # Mean per packet
                np.percentile(amplitude_db, 75, axis=(0, 1)),  # 75th percentile
                np.percentile(amplitude_db, 25, axis=(0, 1))   # 25th percentile
            ]).T
            
            # Combine all features
            features = np.hstack([
                amplitude_features,  # 90 features
                phase_features,      # 90 features
                amplitude_stats      # 5 features
            ])
            
            # Add features and labels
            X.extend(features)
            y.extend([(distance, angle)] * len(features))
    
    return np.array(X), np.array(y)

def split_and_save_dataset(X, y, test_size=0.2, random_state=42):
    """Split dataset into training and testing sets."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Save splits
    dataset = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    with open('csi_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    
    print("\nDataset Statistics:")
    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")
    
    # Print distribution of distances and angles
    print("\nTraining Set Distribution:")
    for distance, angle in np.unique(y_train, axis=0):
        count = np.sum((y_train == [distance, angle]).all(axis=1))
        print(f"Distance: {distance}m, Angle: {angle}° - {count} samples")
    
    print("\nTesting Set Distribution:")
    for distance, angle in np.unique(y_test, axis=0):
        count = np.sum((y_test == [distance, angle]).all(axis=1))
        print(f"Distance: {distance}m, Angle: {angle}° - {count} samples")

def main():
    print("Loading and splitting CSI dataset")
    print("-" * 50)
    
    # Load preprocessed features
    print("Loading preprocessed features...")
    X, y = load_preprocessed_features()
    print(f"\nTotal dataset size: {len(X)} samples")
    print(f"Feature dimension: {X.shape[1]}")
    
    # Split and save dataset
    print("\nSplitting dataset (80% train, 20% test)...")
    split_and_save_dataset(X, y)
    print("\nDataset saved to 'csi_dataset.pkl'")

if __name__ == "__main__":
    main()
