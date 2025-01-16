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
    

    
    def process_csi_data(self, csi_str):
        """Convert CSI string data to numpy array with minimal preprocessing."""
        # Remove brackets and split by commas
        csi_values = csi_str.strip('[]').split(',')
        # Convert to complex numbers and return directly
        return np.array([complex(x) for x in csi_values])
    
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
