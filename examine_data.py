import pandas as pd
import os
from pathlib import Path

def examine_dataset():
    # Check for required files
    required_files = {
        'training': ['0.csv', '1.csv', '2.csv', '3.csv'],
        'validation': ['4.csv'],
        'testing': ['5.csv']
    }
    
    data_dir = Path('HALOC')
    print("Checking for required files...")
    for split, files in required_files.items():
        print(f"\n{split.upper()} files:")
        for file in files:
            file_path = data_dir / file
            if file_path.exists():
                size_mb = os.path.getsize(str(file_path)) / (1024 * 1024)
                print(f"✓ {file} exists ({size_mb:.2f} MB)")
            else:
                print(f"✗ {file} missing!")

    # Examine data structure of first file
    print("\nExamining data structure from 0.csv...")
    df = pd.read_csv(data_dir / '0.csv', nrows=5)  # Read just first 5 rows
    print("\nColumns in dataset:")
    print(df.columns.tolist())
    
    print("\nSample data (first 2 rows):")
    print(df.head(2))
    
    # Basic statistics for position labels
    print("\nPosition label statistics (first 1000 rows):")
    df_stats = pd.read_csv(data_dir / '0.csv', nrows=1000)
    print("\nRange of positions:")
    print(f"X: [{df_stats['x'].min():.2f}, {df_stats['x'].max():.2f}]")
    print(f"Y: [{df_stats['y'].min():.2f}, {df_stats['y'].max():.2f}]")
    print(f"Z: [{df_stats['z'].min():.2f}, {df_stats['z'].max():.2f}]")

if __name__ == "__main__":
    examine_dataset()
