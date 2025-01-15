import pandas as pd
import numpy as np
from pathlib import Path
import json

class HALOCDataset:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.splits = {
            'train': ['0.csv', '1.csv', '2.csv', '3.csv'],
            'val': ['4.csv'],
            'test': ['5.csv']
        }
        
    def load_split(self, split):
        """Load and combine all files for a specific split."""
        if split not in self.splits:
            raise ValueError(f"Split must be one of {list(self.splits.keys())}")
            
        dfs = []
        for file in self.splits[split]:
            file_path = self.data_dir / file
            df = pd.read_csv(file_path)
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)
    

    
    def unwrap_phase(self, phase):
        """
        Enhanced phase unwrapping with adaptive smoothing and discontinuity detection.
        """
        # Initial unwrapping
        unwrapped = np.unwrap(phase)
        
        # Detect and handle discontinuities
        diff = np.diff(unwrapped)
        jumps = np.where(np.abs(diff) > np.pi)[0]
        
        if len(jumps) > 0:
            # Apply adaptive smoothing around discontinuities
            window_size = 5
            for jump in jumps:
                start = max(0, jump - window_size)
                end = min(len(unwrapped), jump + window_size + 1)
                segment = unwrapped[start:end]
                # Fit linear trend to smooth transition
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, deg=1)
                smooth_segment = np.polyval(coeffs, x)
                unwrapped[start:end] = smooth_segment
        
        # Global trend removal
        x = np.arange(len(unwrapped))
        coeffs = np.polyfit(x, unwrapped, deg=1)
        trend = np.polyval(coeffs, x)
        detrended = unwrapped - trend
        
        return detrended
        
    def process_csi_data(self, csi_str):
        """Convert CSI string data to numpy array with enhanced preprocessing."""
        # Remove brackets and split by commas
        csi_values = csi_str.strip('[]').split(',')
        # Convert to complex numbers
        csi_array = np.array([complex(x) for x in csi_values])
        
        # Extract magnitude and phase
        magnitude = np.abs(csi_array)
        phase = np.angle(csi_array)
        
        # Global standardization of magnitude with scaling
        mean_mag = np.mean(magnitude)
        std_mag = np.std(magnitude)
        normalized_magnitude = (magnitude - mean_mag) / (std_mag + 1e-8)
        
        # Scale to reasonable range and clip
        normalized_magnitude = 5.0 * normalized_magnitude  # Scale factor for better signal range
        normalized_magnitude = np.clip(normalized_magnitude, -3, 3)
        
        # Unwrap phase and normalize
        unwrapped_phase = self.unwrap_phase(phase)
        phase_mean = np.mean(unwrapped_phase)
        phase_std = np.std(unwrapped_phase)
        normalized_phase = (unwrapped_phase - phase_mean) / (phase_std + 1e-8)
        
        # Scale phase to maintain proper range
        normalized_phase = np.pi * normalized_phase
        
        # Reconstruct complex numbers with normalized components
        return normalized_magnitude * np.exp(1j * normalized_phase)
    
    def get_split_info(self):
        """Get information about the dataset splits."""
        split_info = {}
        for split, files in self.splits.items():
            split_info[split] = {
                'files': files,
                'total_samples': sum(
                    pd.read_csv(self.data_dir / file).shape[0] 
                    for file in files
                )
            }
        return split_info

def main():
    """Prepare and analyze the dataset splits."""
    dataset = HALOCDataset('HALOC')
    
    # Get split information
    split_info = dataset.get_split_info()
    
    # Print dataset statistics
    print("Dataset Split Information:")
    print(json.dumps(split_info, indent=2))
    
    # Sample a few rows from training set to verify processing
    train_sample = dataset.load_split('train').head(2)
    print("\nSample from training set:")
    print(train_sample[['x', 'y', 'z', 'data']].head())
    
    # Process one CSI sample to verify
    csi_sample = dataset.process_csi_data(train_sample.iloc[0]['data'])
    print("\nProcessed CSI shape:", csi_sample.shape)
    print("First few CSI values:", csi_sample[:5])

if __name__ == "__main__":
    main()
