import os
import h5py
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

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
    """Extract amplitude, phase, and statistical features from CSI data with added noise."""
    # Add complex Gaussian noise (5% of signal amplitude)
    noise_level = 0.05 * np.mean(np.abs(csi_data))
    complex_noise = (np.random.normal(0, noise_level, csi_data.shape) + 
                    1j * np.random.normal(0, noise_level, csi_data.shape))
    csi_data = csi_data + complex_noise
    
    # Add random phase rotation (up to 5 degrees)
    phase_noise = np.random.uniform(0, 5 * np.pi/180, size=csi_data.shape)
    csi_data = csi_data * np.exp(1j * phase_noise)
    
    # Extract amplitude (in dB) and phase with safety checks
    epsilon = 1e-10  # Small constant to prevent log(0)
    amplitude = np.abs(csi_data)
    # Add amplitude noise (2% of mean amplitude)
    amplitude_noise = np.random.normal(0, 0.02 * np.mean(amplitude), amplitude.shape)
    amplitude += amplitude_noise
    
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

def plot_csi_features(features, filename):
    """Plot CSI features for visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'CSI Features: {filename}')
    
    # Plot amplitude mean
    im0 = axes[0,0].imshow(features['amplitude_mean'])
    axes[0,0].set_title('Amplitude Mean (dB)')
    axes[0,0].set_xlabel('Subcarrier Index')
    axes[0,0].set_ylabel('Antenna Index')
    plt.colorbar(im0, ax=axes[0,0])
    
    # Plot phase mean
    im1 = axes[0,1].imshow(features['phase_mean'])
    axes[0,1].set_title('Phase Mean (rad)')
    axes[0,1].set_xlabel('Subcarrier Index')
    axes[0,1].set_ylabel('Antenna Index')
    plt.colorbar(im1, ax=axes[0,1])
    
    # Plot amplitude variance
    im2 = axes[1,0].imshow(features['amplitude_var'])
    axes[1,0].set_title('Amplitude Variance')
    axes[1,0].set_xlabel('Subcarrier Index')
    axes[1,0].set_ylabel('Antenna Index')
    plt.colorbar(im2, ax=axes[1,0])
    
    # Plot phase variance
    im3 = axes[1,1].imshow(features['phase_var'])
    axes[1,1].set_title('Phase Variance')
    axes[1,1].set_xlabel('Subcarrier Index')
    axes[1,1].set_ylabel('Antenna Index')
    plt.colorbar(im3, ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig(f'csi_features_{filename.replace(".mat", ".png")}')
    plt.close()

def save_features(features, filepath, output_dir):
    """Save extracted features to a numpy file."""
    # Get dimensions
    n_antennas, n_subcarriers = features['amplitude_mean'].shape
    n_samples = 10000  # Number of samples per file
    
    # Initialize feature array (n_samples, n_features)
    n_features = (n_antennas * n_subcarriers * 2) + 5  # 90 amplitude + 90 phase + 5 stats
    feature_array = np.zeros((n_samples, n_features))
    
    # Fill feature array
    sample_idx = 0
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
    
    # Save to .npy file
    output_name = f"{filepath.stem}_features.npy"
    output_path = output_dir / output_name
    np.save(output_path, feature_array)
    print(f"Saved feature array with shape: {feature_array.shape}")
    return output_path

def main():
    print("CSI Data Preprocessing")
    print("-" * 50)
    
    # Check if data directory exists
    data_dir = Path('data')
    if not data_dir.exists():
        print(f"Error: '{data_dir}' directory not found!")
        print("Please create a 'data' directory and place all .mat files inside it.")
        return
    
    # Create output directories
    feature_dir = Path('csi_features')
    plot_dir = Path('visualizations')
    feature_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    
    # Process each .mat file
    mat_files = sorted([f for f in data_dir.iterdir() if f.suffix == '.mat'])
    total_files = len(mat_files)
    print(f"Found {total_files} .mat files to process\n")
    
    for idx, filepath in enumerate(mat_files, 1):
        print(f"Processing file {idx}/{total_files}: {filepath.name}")
        print("-" * 40)
        
        try:
            # Load and preprocess data with random sampling
            print("Loading data...")
            csi_data = load_csi_data(str(filepath), max_packets=10000)
            print(f"Loaded CSI data shape: {csi_data.shape}")
            
            # Extract features
            print("\nExtracting features...")
            features = extract_features(csi_data)
            print("Computed statistical features for amplitude and phase")
            
            # Normalize features
            print("\nNormalizing features...")
            normalized_features = normalize_features(features)
            
            # Print feature statistics
            print("\nFeature Statistics:")
            for name, feature in normalized_features.items():
                print(f"{name:15s}: min={feature.min():.3f}, max={feature.max():.3f}, mean={feature.mean():.3f}")
            
            # Save features
            print("\nSaving features...")
            feature_path = save_features(normalized_features, filepath, feature_dir)
            print(f"Saved features to: {feature_path}")
            
            # Plot features
            print("\nGenerating visualization...")
            plot_path = plot_dir / f'csi_features_{filepath.name.replace(".mat", ".png")}'
            plot_csi_features(normalized_features, filepath.name)
            print(f"Saved visualization to: {plot_path}")
            
            print(f"\nSuccessfully processed {filepath.name}")
            print("=" * 50 + "\n")
            
        except Exception as e:
            print(f"Error processing {filepath.name}: {str(e)}\n")
            continue

if __name__ == "__main__":
    main()
