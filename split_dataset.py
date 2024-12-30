from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import h5py
from scipy import stats
import warnings

def load_csi_data(filepath, max_packets=1000):
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

def normalize_feature_array(feature_array):
    """
    Normalize feature array using robust scaling across all samples.
    
    Args:
        feature_array: numpy array of shape (n_samples, n_features)
    
    Returns:
        normalized array of same shape
    """
    n_samples, n_features = feature_array.shape
    normalized = np.zeros_like(feature_array)
    
    # Normalize each feature column independently
    for j in range(n_features):
        feature_col = feature_array[:, j]
        q1 = np.percentile(feature_col, 1)
        q99 = np.percentile(feature_col, 99)
        iqr = q99 - q1
        
        if iqr == 0:  # Handle constant features
            normalized[:, j] = np.zeros_like(feature_col)
        else:
            # Clip extreme values and scale
            feature_clipped = np.clip(feature_col, q1, q99)
            normalized[:, j] = (feature_clipped - q1) / iqr
            # Clip final values to [0,1]
            normalized[:, j] = np.clip(normalized[:, j], 0, 1)
    
    return normalized

def create_feature_vectors_timed(csi_data, window_size=100, step_size=None):
    """Create feature vectors using non-overlapping windows over CSI data packets.
    Features are designed to be completely independent between windows.
    
    Args:
        csi_data: numpy array of shape (3, 30, N_packets) containing CSI data
        window_size: number of packets to use in each window
        step_size: ignored (windows are non-overlapping)
    
    Returns:
        feature_array: numpy array of shape (n_windows, n_features)
    """
    print(f"Creating feature vectors with window_size={window_size}, step_size={step_size}")
    print(f"Input CSI data shape: {csi_data.shape}")
    n_antennas, n_subcarriers, n_packets = csi_data.shape
    n_windows = (n_packets - window_size) // step_size + 1
    
    # Calculate number of features (90 amplitude + 90 phase + 5 stats = 185 features)
    n_features = (n_antennas * n_subcarriers * 2) + 5  # Match preprocess_csi.py format
    feature_array = np.zeros((n_windows, n_features))
    
    # Create non-overlapping windows
    n_windows = n_packets // window_size
    print(f"Creating {n_windows} non-overlapping windows...")
    
    # Randomly select a subset of windows to prevent having too many samples
    max_windows = min(n_windows, 5)  # Limit to 5 windows per condition
    selected_indices = np.random.choice(n_windows, max_windows, replace=False)
    selected_indices.sort()  # Sort for sequential access
    
    print(f"Selected {max_windows} windows for processing...")
    for window_idx in selected_indices:
        if window_idx % 2 == 0:  # Show progress every other window
            print(f"Processing window {window_idx}/{max_windows}...")
        
        # Use non-overlapping windows
        start_idx = window_idx * window_size
        end_idx = start_idx + window_size
        
        # Create a completely independent window with enhanced noise and augmentation
        window_data = csi_data[:, :, start_idx:end_idx].copy()
        
        # Calculate signal power for proportional noise
        signal_power = np.mean(np.abs(window_data)**2)
        
        # Add significant phase rotation (up to 15 degrees)
        phase_noise = np.random.uniform(-15 * np.pi/180, 15 * np.pi/180, size=window_data.shape)
        window_data = window_data * np.exp(1j * phase_noise)
        
        # Add complex Gaussian noise (15% of signal power)
        noise_std = np.sqrt(0.15 * signal_power)
        complex_noise = (np.random.normal(0, noise_std, window_data.shape) + 
                       1j * np.random.normal(0, noise_std, window_data.shape))
        window_data = window_data + complex_noise
        
        # Add frequency-selective fading with more variation
        freq_fade = np.random.uniform(0.7, 1.3, (1, window_data.shape[1], 1))
        window_data = window_data * freq_fade
        
        # Add random time-varying phase drift
        time_drift = np.linspace(0, np.random.uniform(-10, 10) * np.pi/180, window_data.shape[2])
        time_drift = np.exp(1j * time_drift)
        window_data = window_data * time_drift
        
        # Add random subcarrier-specific attenuation
        subcarrier_atten = np.random.uniform(0.8, 1.2, (1, window_data.shape[1], 1))
        window_data = window_data * subcarrier_atten
        
        # Extract amplitude (in dB) and phase with noise
        epsilon = 1e-10
        amplitude = np.abs(window_data)
        amplitude_db = 20 * np.log10(np.maximum(amplitude, epsilon))
        phase = np.angle(window_data)
        
        # Add more significant amplitude noise (2% of signal)
        amplitude_noise = np.random.normal(0, 0.02 * np.mean(amplitude_db), amplitude_db.shape)
        amplitude_db += amplitude_noise
        
        # Extract robust features with enhanced statistics
        feature_idx = 0
        for i in range(n_antennas):
            for j in range(n_subcarriers):
                # Amplitude features (first 90)
                amp_data = amplitude_db[i, j, :]
                feature_array[window_idx, feature_idx] = np.median(amp_data)  # More robust than mean
                
                # Phase features (next 90) with unwrapping
                phase_data = np.unwrap(phase[i, j, :])
                phase_idx = (n_antennas * n_subcarriers) + feature_idx
                feature_array[window_idx, phase_idx] = stats.circmean(phase_data)
                
                feature_idx += 1
        
        # Enhanced statistical features (last 5)
        stat_start = n_antennas * n_subcarriers * 2
        
        # Use robust statistics
        feature_array[window_idx, stat_start] = stats.iqr(amplitude_db.flatten())  # IQR instead of std
        feature_array[window_idx, stat_start+1] = stats.median_abs_deviation(amplitude_db.flatten())  # MAD
        feature_array[window_idx, stat_start+2] = stats.skew(amplitude_db.flatten())  # Overall skewness
        feature_array[window_idx, stat_start+3] = stats.kurtosis(amplitude_db.flatten())  # Overall kurtosis
        feature_array[window_idx, stat_start+4] = stats.iqr(np.unwrap(phase).flatten())  # Phase variation
    
    return feature_array

def load_and_process_data(data_dir='data', window_size=100, step_size=75):
    """Load and process all CSI data files using sliding windows.
    Features are normalized globally after all samples are collected.
    Uses larger step size and adds significant noise for robustness.
    
    Args:
        data_dir: directory containing .mat files
        window_size: number of packets to use in each window
        step_size: number of packets to slide between windows (increased to reduce correlation)
    """
    X = []  # Features
    y = []  # Labels (distance, angle)
    file_indices = []  # Track which file each sample came from
    
    # Process each .mat file in the data directory
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found at {data_dir}")
    
    mat_files = sorted(data_dir.glob('*.mat'))
    print(f"Found {len(mat_files)} .mat files to process")
    
    for idx, mat_file in enumerate(mat_files):
        # Extract base name
        parts = mat_file.stem.split('_')
        
        # Extract distance and angle from filename (e.g., loc_minus60deg_4m)
        angle_str = parts[1].replace('deg', '')
        angle = -int(angle_str.replace('minus', '')) if angle_str.startswith('minus') else int(angle_str)
        distance = int(parts[2].replace('m', ''))
        
        print(f"\nProcessing {mat_file.name} - Distance: {distance}m, Angle: {angle}°")
        
        try:
            # Load raw CSI data
            print("Loading CSI data...")
            csi_data = load_csi_data(str(mat_file))
            print(f"Loaded CSI data shape: {csi_data.shape}")
            
            # Create feature vectors using sliding windows
            print("\nExtracting features using sliding windows...")
            feature_array = create_feature_vectors_timed(csi_data, window_size, step_size)
            n_samples = feature_array.shape[0]
            n_features = feature_array.shape[1]
            print(f"Created feature array with {n_samples} samples and {n_features} features")
            
            # Add features, labels, and file indices
            X.append(feature_array)
            y.extend([(distance, angle)] * n_samples)
            file_indices.extend([idx] * n_samples)  # Use file index to track samples
            
        except Exception as e:
            print(f"Error processing {mat_file.name}: {str(e)}")
            continue
    
    if not X:
        raise ValueError("No feature files were successfully processed")
    
    # Stack all features into a single array
    X = np.vstack(X)
    y = np.array(y)
    file_indices = np.array(file_indices)
    
    # Normalize features globally
    print("\nNormalizing features globally...")
    X = normalize_feature_array(X)
    
    print(f"\nTotal dataset size: {len(X)} samples")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Number of unique files: {len(np.unique(file_indices))}")
    
    return X, y, file_indices

def split_and_save_dataset(X, y, file_indices, test_size=0.2, random_state=42):
    """Split dataset into training and testing sets ensuring each distance-angle
    combination is represented in both sets. This prevents the model from having
    to generalize to completely unseen conditions.
    
    Args:
        X: feature array
        y: labels array
        file_indices: array indicating which file each sample came from
        test_size: fraction of data to use for testing
        random_state: random seed for reproducibility
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Create sample indices and conditions array
    n_samples = len(X)
    sample_indices = np.arange(n_samples)
    conditions = np.array([f"{d}m_{a}deg" for d, a in y])
    
    # Get unique conditions and files
    unique_conditions = np.unique(conditions)
    print("\nUnique conditions in dataset:")
    for cond in unique_conditions:
        mask = (conditions == cond)
        print(f"{cond}: {np.sum(mask)} samples")
    
    # Initialize train and test indices
    train_indices = []
    test_indices = []
    
    # Split each condition separately to ensure representation
    for condition in unique_conditions:
        condition_mask = (conditions == condition)
        condition_indices = sample_indices[condition_mask]
        
        # Shuffle indices
        np.random.shuffle(condition_indices)
        
        # Split indices
        n_test = int(len(condition_indices) * test_size)
        if n_test == 0:  # Ensure at least one test sample
            n_test = 1
        
        # Add to train and test sets
        test_indices.extend(condition_indices[:n_test])
        train_indices.extend(condition_indices[n_test:])
    
    # Convert to arrays
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    # Shuffle the indices
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    # Create train and test masks
    train_mask = np.zeros(n_samples, dtype=bool)
    test_mask = np.zeros(n_samples, dtype=bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    
    # Print initial split statistics
    print(f"\nInitial split - Train: {len(train_indices)} samples, Test: {len(test_indices)} samples")
    
    # Verify no overlap in masks
    if np.any(train_mask & test_mask):
        raise ValueError("Found samples that appear in both train and test sets!")
    
    # Verify we have samples in both sets
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("Split resulted in empty train or test set! Adjust test_size.")
    
    # Split the data
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    # Print split statistics
    print("\nSplit Statistics:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    print("\nTraining Set Distribution:")
    train_conditions = [f"{d}m_{a}deg" for d, a in y_train]
    for cond in unique_conditions:
        count = sum(1 for c in train_conditions if c == cond)
        print(f"{cond}: {count} samples")
    
    print("\nTesting Set Distribution:")
    test_conditions = [f"{d}m_{a}deg" for d, a in y_test]
    for cond in unique_conditions:
        count = sum(1 for c in test_conditions if c == cond)
        print(f"{cond}: {count} samples")
    
    # Save the dataset
    dataset = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_indices': train_indices,
        'test_indices': test_indices
    }
    
    with open('csi_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    
    print("\nVerification: Split completed successfully")

def main():
    print("Loading and processing CSI dataset")
    print("-" * 50)
    
    # Load and process raw CSI data with non-overlapping windows
    print("Processing CSI data files with non-overlapping windows...")
    X, y, file_indices = load_and_process_data(window_size=100)  # Non-overlapping windows
    
    # Verify feature values
    print("\nFeature Statistics:")
    print(f"Mean value: {np.mean(X):.4f}")
    print(f"Std deviation: {np.std(X):.4f}")
    print(f"Min value: {np.min(X):.4f}")
    print(f"Max value: {np.max(X):.4f}")
    print(f"Number of NaN values: {np.isnan(X).sum()}")
    print(f"Number of Inf values: {np.isinf(X).sum()}")
    
    print(f"\nTotal dataset size: {len(X)} samples")
    print(f"Feature dimension: {X.shape[1]}")
    
    # Split and save dataset
    print("\nSplitting dataset (80% train, 20% test)...")
    split_and_save_dataset(X, y, file_indices)
    print("\nDataset saved to 'csi_dataset.pkl'")

if __name__ == "__main__":
    main()
