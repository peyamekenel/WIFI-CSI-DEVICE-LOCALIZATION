from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import h5py
from scipy import stats

def load_csi_data(filepath, max_packets=10000):
    """
    Load CSI data from .mat file and convert to proper format.
    Args:
        filepath: Path to .mat file
        max_packets: Maximum number of packets to load (random sampling)
    """
    with h5py.File(filepath, 'r') as f:
        # Get total number of packets
        total_packets = f['csi_complex_data'].shape[0]
        
        # Random sampling if total_packets > max_packets
        if total_packets > max_packets:
            indices = np.random.choice(total_packets, max_packets, replace=False)
            indices.sort()  # Sort for sequential access
            print(f"Sampling {max_packets} packets from total {total_packets}")
            raw_data = f['csi_complex_data'][indices]
        else:
            print(f"Loading all {total_packets} packets")
            raw_data = f['csi_complex_data'][:]
        
        # Convert to complex numbers
        csi_data = raw_data['real'] + 1j * raw_data['imag']
        # Transpose to (3, 30, N) format
        return np.transpose(csi_data, (2, 1, 0))

def extract_features(csi_data):
    """Extract amplitude, phase, and statistical features from CSI data."""
    # Extract amplitude (in dB) and phase with safety checks
    epsilon = 1e-10  # Small constant to prevent log(0)
    amplitude = np.abs(csi_data)
    # Clip small values to epsilon before log
    amplitude_clipped = np.maximum(amplitude, epsilon)
    amplitude_db = 20 * np.log10(amplitude_clipped)
    phase = np.angle(csi_data)
    
    # Calculate statistical features per antenna and subcarrier
    features = {
        'amplitude_mean': np.mean(amplitude_db, axis=2),  # Mean over packets
        'amplitude_std': np.std(amplitude_db, axis=2),    # Standard deviation
        'amplitude_var': np.var(amplitude_db, axis=2),    # Variance
        'amplitude_skew': stats.skew(amplitude_db, axis=2),  # Skewness
        'amplitude_kurtosis': stats.kurtosis(amplitude_db, axis=2),  # Kurtosis
        'phase_mean': np.mean(phase, axis=2),
        'phase_std': np.std(phase, axis=2),
        'phase_var': np.var(phase, axis=2)
    }
    return features

def normalize_features(features):
    """
    Normalize features to [0,1] range with robust handling of edge cases.
    Uses robust scaling to handle outliers.
    """
    normalized = {}
    for name, feature in features.items():
        # Get robust statistics
        q1 = np.percentile(feature, 1)
        q99 = np.percentile(feature, 99)
        iqr = q99 - q1
        
        if iqr == 0:  # Handle constant features
            normalized[name] = np.zeros_like(feature)
        else:
            # Clip extreme values and scale
            feature_clipped = np.clip(feature, q1, q99)
            normalized[name] = (feature_clipped - q1) / iqr
            # Clip final values to [0,1]
            normalized[name] = np.clip(normalized[name], 0, 1)
    
    return normalized

def create_feature_vector(features):
    """Convert normalized features into a feature vector."""
    n_antennas, n_subcarriers = features['amplitude_mean'].shape
    n_samples = 10000  # Number of samples per file
    
    # Initialize feature array (n_samples, n_features)
    n_features = (n_antennas * n_subcarriers * 2) + 5  # 90 amplitude + 90 phase + 5 stats
    feature_array = np.zeros((n_samples, n_features))
    
    # Fill feature array
    for i in range(n_antennas):
        for j in range(n_subcarriers):
            # Amplitude features (first 90 features)
            feature_idx = i * n_subcarriers + j
            feature_array[:, feature_idx] = features['amplitude_mean'][i, j]
            
            # Phase features (next 90 features)
            phase_idx = (n_antennas * n_subcarriers) + (i * n_subcarriers + j)
            feature_array[:, phase_idx] = features['phase_mean'][i, j]
    
    # Add statistical features (last 5 features)
    stat_start = n_antennas * n_subcarriers * 2
    feature_array[:, stat_start] = features['amplitude_std'].mean(axis=(0,1))
    feature_array[:, stat_start+1] = features['amplitude_var'].mean(axis=(0,1))
    feature_array[:, stat_start+2] = features['amplitude_skew'].mean(axis=(0,1))
    feature_array[:, stat_start+3] = features['amplitude_kurtosis'].mean(axis=(0,1))
    feature_array[:, stat_start+4] = features['phase_std'].mean(axis=(0,1))
    
    return feature_array

def load_and_process_data(data_dir='data'):
    """Load and process all CSI data files directly."""
    X = []  # Features
    y = []  # Labels (distance, angle)
    
    # Process each .mat file in the data directory
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found at {data_dir}")
        
    for mat_file in sorted(data_dir.glob('*.mat')):
        # Extract base name
        parts = mat_file.stem.split('_')
        
        # Extract distance and angle from filename (e.g., loc_minus60deg_4m)
        angle_str = parts[1].replace('deg', '')
        angle = -int(angle_str.replace('minus', '')) if angle_str.startswith('minus') else int(angle_str)
        distance = int(parts[2].replace('m', ''))
        
        print(f"Processing {mat_file.name} - Distance: {distance}m, Angle: {angle}°")
        
        try:
            # Load and process CSI data
            print("Loading CSI data...")
            csi_data = load_csi_data(str(mat_file))
            print(f"Loaded CSI data shape: {csi_data.shape}")
            
            # Extract features
            print("\nExtracting features...")
            features = extract_features(csi_data)
            print("Computed statistical features")
            
            # Normalize features
            print("\nNormalizing features...")
            normalized_features = normalize_features(features)
            
            # Convert to feature vector
            feature_array = create_feature_vector(normalized_features)
            
            n_samples = feature_array.shape[0]
            n_features = feature_array.shape[1]
            print(f"Created feature array with {n_samples} samples and {n_features} features")
            
            # Add features and labels
            X.append(feature_array)
            y.extend([(distance, angle)] * n_samples)
            
        except Exception as e:
            print(f"Error processing {mat_file.name}: {str(e)}")
            continue
    
    if not X:
        raise ValueError("No feature files were successfully processed")
        
    # Stack all features into a single array
    X = np.vstack(X)
    return np.array(X), np.array(y)

def split_and_save_dataset(X, y, test_size=0.2, random_state=42):
    """Split dataset into training and testing sets."""
    # Convert labels to a format suitable for stratification
    y_combined = np.array([f"{d}m_{a}deg" for d, a in y])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y_combined
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
    print("Loading and processing CSI dataset")
    print("-" * 50)
    
    # Load and process raw CSI data
    print("Processing CSI data files...")
    X, y = load_and_process_data()
    print(f"\nTotal dataset size: {len(X)} samples")
    print(f"Feature dimension: {X.shape[1]}")
    
    # Split and save dataset
    print("\nSplitting dataset (80% train, 20% test)...")
    split_and_save_dataset(X, y)
    print("\nDataset saved to 'csi_dataset.pkl'")

if __name__ == "__main__":
    main()
