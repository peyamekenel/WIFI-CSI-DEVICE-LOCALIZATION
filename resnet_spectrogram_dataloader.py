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
    def __init__(self, data_dir, window_size=351, split='train', cache_size=1000):
        """
        Initialize the dataset with lazy loading and LRU cache for CSI data.
        
        Args:
            data_dir: Path to HALOC dataset directory
            window_size: Number of packets in each window (default: 351 ~ 3.51s)
            split: One of 'train', 'val', or 'test'
            cache_size: Number of processed samples to keep in memory
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.window_half = window_size // 2
        self.cache_size = cache_size
        
        # Define split files
        splits = {
            'train': ['0.csv', '1.csv', '2.csv', '3.csv'],
            'val': ['4.csv'],
            'test': ['5.csv']
        }
        
        if split not in splits:
            raise ValueError(f"Split must be one of {list(splits.keys())}")
        
        # Load and combine all files for the split
        dfs = []
        for file in splits[split]:
            file_path = self.data_dir / file
            df = pd.read_csv(file_path)
            dfs.append(df)
        self.data = pd.concat(dfs, ignore_index=True)
        
        # Initialize cache
        self.feature_cache = {}
        self.cache_order = []
        
        # Create cache directory
        self.cache_dir = self.data_dir / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        # Compute dataset size
        self.effective_size = len(self.data) - self.window_size
        
        # Initialize min-max values
        self.min_val = float('inf')
        self.max_val = float('-inf')
        
    def _process_single_sample(self, idx):
        """Process a single CSI sample."""
        # Check cache first
        if idx in self.feature_cache:
            return self.feature_cache[idx]
            
        # Process CSI data
        csi_str = self.data.iloc[idx]['data']
        csi_values = np.array([int(x) for x in csi_str.strip('[]').split(',')])
        
        # Extract valid subcarriers
        features = np.zeros(CSI_SUBCARRIERS, dtype=np.float32)
        for i, subcarrier_idx in enumerate(csi_valid_subcarrier_index):
            real_idx = subcarrier_idx * 2
            imag_idx = real_idx - 1
            complex_val = complex(csi_values[real_idx], csi_values[imag_idx])
            features[i] = np.abs(complex_val)
        
        # Update min-max values
        self.min_val = min(self.min_val, features.min())
        self.max_val = max(self.max_val, features.max())
        
        # Add to cache
        if len(self.cache_order) >= self.cache_size:
            # Remove oldest entry
            oldest = self.cache_order.pop(0)
            del self.feature_cache[oldest]
        
        self.feature_cache[idx] = features
        self.cache_order.append(idx)
        
        return features
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.effective_size
    
    def __getitem__(self, index):
        """
        Get a window of CSI amplitudes and corresponding position.
        
        Args:
            index: Sample index
            
        Returns:
            tuple: (spectrogram_tensor, position_tensor)
                spectrogram_tensor: shape (1, CSI_SUBCARRIERS, window_size)
                position_tensor: shape (3,) for (x, y, z)
        """
        # Convert tensor index to Python integer if needed
        if isinstance(index, torch.Tensor):
            index = index.item()
            
        # Add window_half to avoid border issues
        center_idx = int(index + self.window_half)
        
        # Process window of samples
        feature_window = np.zeros((self.window_size, CSI_SUBCARRIERS), dtype=np.float32)
        for i in range(self.window_size):
            idx = center_idx - self.window_half + i
            feature_window[i] = self._process_single_sample(idx)
        
        # Apply min-max scaling
        if self.max_val > self.min_val:
            feature_window = (feature_window - self.min_val) / (self.max_val - self.min_val)
        
        # Transpose to (CSI_SUBCARRIERS, window_size) and add channel dimension
        feature_window = np.transpose(feature_window)
        feature_window = np.expand_dims(feature_window, axis=0)
        
        # Get position label for center sample
        position = np.array([
            self.data.iloc[center_idx]['x'],
            self.data.iloc[center_idx]['y'],
            self.data.iloc[center_idx]['z']
        ], dtype=np.float32)
        
        return torch.tensor(feature_window, dtype=torch.float32), torch.tensor(position, dtype=torch.float32)

def create_dataloaders(data_dir, batch_size=32, num_workers=1):
    """
    Create DataLoaders for all splits.
    
    Args:
        data_dir: Path to HALOC dataset directory
        batch_size: Number of samples per batch
        num_workers: Number of worker processes
        
    Returns:
        dict: DataLoaders for train, validation, and test sets
    """
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        dataset = ResnetSpectrogramDataset(data_dir, split=split)
        shuffle = (split == 'train')
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    return dataloaders

def test_dataloader():
    """Test the ResnetSpectrogramDataset implementation."""
    # Create dataset
    data_dir = 'HALOC'
    dataset = ResnetSpectrogramDataset(data_dir, split='train')
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Get a batch
    spectrogram_batch, position_batch = next(iter(dataloader))
    
    print("\nDataset size:", len(dataset))
    print("Spectrogram batch shape:", spectrogram_batch.shape)
    print("Position batch shape:", position_batch.shape)
    print("\nSpectrogram value range:", 
          f"[{spectrogram_batch.min():.3f}, {spectrogram_batch.max():.3f}]")
    
    return spectrogram_batch.shape, position_batch.shape

if __name__ == '__main__':
    test_dataloader()
