import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

# Subcarrier selection for HALOC dataset (same as reference implementation)
csi_valid_subcarrier_index = []
csi_valid_subcarrier_index += [i for i in range(6, 32)]
csi_valid_subcarrier_index += [i for i in range(33, 59)]
CSI_SUBCARRIERS = len(csi_valid_subcarrier_index)

class ResnetSpectrogramDataset(Dataset):
    def __init__(self, data_dir, window_size=351, split='train'):
        """
        Initialize the dataset following reference implementation approach.
        Using single-channel amplitude features and window-based processing.
        
        Args:
            data_dir: Path to HALOC dataset directory
            window_size: Number of packets in each window (default: 351)
            split: One of 'train', 'val', or 'test'
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.window_half = int(window_size/2)  # Reference implementation window size calculation
        
        # Handle individual CSV files or predefined splits
        if split.endswith('.csv'):
            # Direct CSV file
            file_path = self.data_dir / split
            if not file_path.exists():
                raise ValueError(f"CSV file not found: {file_path}")
            self.data = pd.read_csv(file_path)
        else:
            # Predefined splits
            splits = {
                'train': ['0.csv', '1.csv', '2.csv', '3.csv'],
                'val': ['4.csv'],
                'test': ['5.csv']
            }
            if split not in splits:
                raise ValueError(f"Split must be one of {list(splits.keys())} or a CSV filename")
            
            # Load and combine all files for the split
            dfs = []
            for file in splits[split]:
                file_path = self.data_dir / file
                df = pd.read_csv(file_path)
                dfs.append(df)
            self.data = pd.concat(dfs, ignore_index=True)
        
        # Create cache directory for .npy files
        self.cache_dir = self.data_dir / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        # Process and cache CSI data
        cache_file = self.cache_dir / f"{split}_features.npy"
        if os.path.exists(cache_file):
            print(f"Loading cached features from {cache_file}")
            self.features = np.load(cache_file)
        else:
            print(f"Processing and caching features to {cache_file}")
            # Extract CSI features for all samples
            self.features = np.zeros((len(self.data), CSI_SUBCARRIERS), dtype=np.float32)
            for idx in tqdm(range(len(self.data)), desc="Processing CSI data"):
                csi_str = self.data.iloc[idx]['data']
                csi_values = np.array([int(x) for x in csi_str.strip('[]').split(',')])
                
                # Extract valid subcarriers
                features = np.zeros(CSI_SUBCARRIERS, dtype=np.complex64)
                for i, subcarrier_idx in enumerate(csi_valid_subcarrier_index):
                    real_idx = subcarrier_idx * 2
                    imag_idx = real_idx - 1
                    features[i] = complex(csi_values[real_idx], csi_values[imag_idx])
                
                # Store amplitude features
                self.features[idx] = np.abs(features)
            
            # Apply min-max scaling
            self.features = (self.features - np.min(self.features)) / (np.max(self.features) - np.min(self.features))
            
            # Cache processed features
            np.save(cache_file, self.features)
        
        # Compute dataset size excluding border regions
        self.effective_size = len(self.data) - self.window_size
        
    def _get_features(self, idx):
        """Get preprocessed features for a sample."""
        return self.features[idx]
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.effective_size
    
    def __getitem__(self, index):
        """
        Get a window of CSI amplitudes and corresponding position.
        Following reference implementation approach with single-channel amplitude features.
        
        Args: 
            index: Sample index
            
        Returns:
            tuple: (feature_window, position_tensor)
                feature_window: shape (1, CSI_SUBCARRIERS, window_size)
                position_tensor: shape (3,) for (x, y, z)
        """
        # Convert tensor index to Python integer if needed
        if isinstance(index, torch.Tensor):
            index = index.item()
            
        # Add index offset to avoid border regions (reference implementation approach)
        index = index + self.window_half + 1
        
        # Get feature window (reference implementation approach)
        feature_window = self.features[index-self.window_half:index+self.window_half]  # Shape: [window_size, CSI_SUBCARRIERS]
        
        # Transpose to match reference implementation shape [CSI_SUBCARRIERS, window_size]
        feature_window = np.transpose(feature_window)  # Shape: [CSI_SUBCARRIERS, window_size]
        
        # Add channel dimension (reference implementation approach)
        feature_window = np.expand_dims(feature_window, axis=0)  # Shape: [1, CSI_SUBCARRIERS, window_size]
        
        # Get position label for center sample (reference implementation approach)
        position = np.array([
            self.data.iloc[index]['x'],
            self.data.iloc[index]['y'],
            self.data.iloc[index]['z']
        ], dtype=np.float32)
        
        return torch.tensor(feature_window, dtype=torch.float32), torch.tensor(position, dtype=torch.float32)

def create_dataloaders(data_dir, batch_size=128, num_workers=8, window_size=351):
    """
    Create DataLoaders for all splits.
    Using reference implementation configuration.
    
    Args:
        data_dir: Path to HALOC dataset directory
        batch_size: Number of samples per batch (default: 128, reference implementation)
        num_workers: Number of worker processes (default: 8, reference implementation)
        window_size: Number of packets in each window (default: 351, reference implementation)
        
    Returns:
        dict: DataLoaders for train, validation, and test sets
    """
    print(f"\nCreating dataloaders with reference configuration:")
    print(f"- Window size: {window_size}")
    print(f"- Batch size: {batch_size}")
    print(f"- Workers: {num_workers}")
    
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        print(f"\nCreating {split} dataset...")
        dataset = ResnetSpectrogramDataset(data_dir, window_size=window_size, split=split)
        print(f"{split.capitalize()} dataset size: {len(dataset)}")
        shuffle = (split == 'train')  # Only shuffle training data
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        print(f"{split.capitalize()} dataloader created successfully")
    return dataloaders

if __name__ == '__main__':
    # Test the implementation
    data_dir = 'HALOC'
    window_size = 351  # Reference implementation window size
    dataset = ResnetSpectrogramDataset(data_dir, window_size=window_size, split='train')
    
    # Create dataloader with reference implementation batch size
    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    # Get a batch
    feature_batch, position_batch = next(iter(dataloader))
    
    print("\nDataset Configuration (Reference Implementation):")
    print(f"Window size: {window_size}")
    print(f"Batch size: {batch_size}")
    print(f"Dataset size: {len(dataset)}")
    
    print("\nTensor Shapes:")
    print(f"Feature batch: {feature_batch.shape}")  # Should be (batch_size, 1, CSI_SUBCARRIERS, window_size)
    print(f"Position batch: {position_batch.shape}")  # Should be (batch_size, 3)
    
    print("\nFeature Statistics:")
    print("Amplitude range:", 
          f"[{feature_batch.min():.3f}, {feature_batch.max():.3f}]")
    print("Mean amplitude:", f"{feature_batch.mean():.3f}")
    print("Std amplitude:", f"{feature_batch.std():.3f}")
