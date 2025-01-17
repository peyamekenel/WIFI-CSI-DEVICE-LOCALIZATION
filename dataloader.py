import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from prepare_dataset import HALOCDataset

class HALOCTorchDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        Args:
            data_dir: Path to HALOC dataset directory
            split: One of 'train', 'val', or 'test'
        """
        self.haloc = HALOCDataset(data_dir)
        self.data = self.haloc.load_split(split)
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Convert a single sample to tensors.
        Returns:
            tuple: (csi_tensor, position_tensor)
        """
        # Ensure idx is an integer for DataFrame indexing
        idx = int(idx)
        sample = self.data.iloc[idx]
        
        # Process CSI data into complex tensor with phase unwrapping
        csi_array = self.haloc.process_csi_data(sample['data'])
        
        # Convert to tensor of shape (2, 256) for real and imaginary parts
        # Using unwrapped phase information
        csi_tensor = torch.tensor(
            np.stack([csi_array.real, csi_array.imag]),
            dtype=torch.float32
        )
        
        # Convert position to tensor
        position_tensor = torch.tensor(
            [sample['x'], sample['y'], sample['z']],
            dtype=torch.float32
        )
        
        return csi_tensor, position_tensor

def create_dataloaders(data_dir, batch_size=32, num_workers=2):
    """
    Create DataLoaders for all splits.
    
    Args:
        data_dir: Path to HALOC dataset directory
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        
    Returns:
        dict: DataLoaders for train, validation, and test sets
    """
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        dataset = HALOCTorchDataset(data_dir, split)
        shuffle = (split == 'train')  # Only shuffle training data
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True  # Faster data transfer to GPU
        )
    return dataloaders

def main():
    """Test the DataLoader implementation."""
    # Create dataloaders
    data_dir = 'HALOC'
    dataloaders = create_dataloaders(data_dir, batch_size=32)
    
    # Print dataset sizes
    for split, dataloader in dataloaders.items():
        print(f"\n{split.capitalize()} dataset size: {len(dataloader.dataset)}")
        
    # Test loading a batch
    print("\nTesting batch loading...")
    train_loader = dataloaders['train']
    csi_batch, pos_batch = next(iter(train_loader))
    
    print(f"CSI batch shape: {csi_batch.shape}")
    print(f"Position batch shape: {pos_batch.shape}")
    print("\nFirst position in batch:", pos_batch[0])
    
    # Verify tensor properties
    print("\nTensor properties:")
    print(f"CSI dtype: {csi_batch.dtype}")
    print(f"Position dtype: {pos_batch.dtype}")
    print(f"Device: {csi_batch.device}")

if __name__ == "__main__":
    main()
