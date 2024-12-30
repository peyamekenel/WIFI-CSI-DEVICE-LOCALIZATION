import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

def load_csi_dataset(data_dir, max_packets_per_file=10000):
    """
    Load CSI data from all .mat files in the specified directory.
    Args:
        data_dir: Directory containing .mat files
        max_packets_per_file: Maximum number of packets to load per file to manage memory
    Returns:
        - csi_data: list of complex CSI matrices (downsampled)
        - labels: corresponding distance and angle labels
    """
    csi_data = []
    distances = []
    angles = []
    
    # List all .mat files
    mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    
    print(f"Found {len(mat_files)} .mat files")
    print(f"Using max {max_packets_per_file} packets per file for memory efficiency")
    
    for mat_file in sorted(mat_files):
        try:
            # Load .mat file using h5py
            with h5py.File(os.path.join(data_dir, mat_file), 'r') as f:
                # Get CSI data
                # Print dataset structure
                print(f"\nProcessing {mat_file}:")
                print("Available datasets:", list(f.keys()))
                
                # Get CSI data and print its properties
                csi_raw = f['csi_complex_data'][()]
                print(f"Raw data type: {csi_raw.dtype}")
                print(f"Raw data shape: {csi_raw.shape}")
                
                # Check if the data is already in complex format
                if np.issubdtype(csi_raw.dtype, np.complexfloating):
                    csi = csi_raw
                else:
                    # Handle compound type with real/imag parts
                    print("Field names in compound type:", csi_raw.dtype.names)
                    csi = csi_raw['real'] + 1j * csi_raw['imag']
                
                print(f"Complex data shape before transpose: {csi.shape}")
                
                # Transpose to get the correct dimension order (3 x 30 x packets)
                if csi.shape[2] == 3 and csi.shape[1] == 30:
                    csi = csi.transpose(2, 1, 0)
                
                # Downsample if necessary
                if csi.shape[2] > max_packets_per_file:
                    step = csi.shape[2] // max_packets_per_file
                    csi = csi[:, :, ::step][:, :, :max_packets_per_file]
                    print(f"Downsampled to {csi.shape[2]} packets")
                
                print(f"Final data shape: {csi.shape}")
                
                # Verify dimensions (3 x 30 x num_packets)
                if csi.shape[0] != 3 or csi.shape[1] != 30:
                    raise ValueError(f"Data dimensions incorrect after processing: {csi.shape}")
            
            # Extract distance and angle from filename
            parts = mat_file.replace('.mat', '').split('_')
            angle = int(parts[1].replace('deg', ''))
            distance = int(parts[2].replace('m', ''))
            
            print(f"Loaded {mat_file}: shape={csi.shape}, angle={angle}°, distance={distance}m")
            
            # Store data and labels
            csi_data.append(csi)
            distances.append(distance)
            angles.append(angle)
            
        except Exception as e:
            print(f"Error loading {mat_file}: {str(e)}")
    
    return csi_data, np.array(distances), np.array(angles)

# Load the dataset
if __name__ == "__main__":
    data_dir = "/home/ubuntu/attachments"
    # Load dataset with maximum 1000 packets per file for faster verification
    csi_data, distances, angles = load_csi_dataset(data_dir, max_packets_per_file=1000)
    
    # Print dataset summary
    print("\nDataset Summary:")
    print(f"Total number of samples: {len(csi_data)}")
    print(f"Unique distances: {np.unique(distances)} meters")
    print(f"Unique angles: {np.unique(angles)} degrees")
    
    
    # Print detailed information about each sample
    print("\nDetailed Sample Information:")
    for i, (csi, dist, ang) in enumerate(zip(csi_data, distances, angles)):
        print(f"Sample {i+1}: shape={csi.shape}, angle={ang}°, distance={dist}m")
