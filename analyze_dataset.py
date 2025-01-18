import pandas as pd
import numpy as np

# Read the readme file
with open("HALOC/readme.txt", "r") as f:
    print("=== README CONTENTS ===")
    print(f.read())
    print("\n=== CSV STRUCTURE ANALYSIS ===")

# Load and examine the first few rows of the first CSV file
df = pd.read_csv("HALOC/0.csv", nrows=5)
print("\nDataset columns:", df.columns.tolist())
print("\nSample of first 5 rows:")
print(df.head())
print("\nData types of columns:")
print(df.dtypes)

print("\n=== DATASET SIZE ANALYSIS ===")
# Analyze each CSV file
for i in range(6):
    df = pd.read_csv(f"HALOC/{i}.csv")
    print(f"\nFile {i}.csv:")
    print(f"Number of samples: {len(df)}")
    
    # Analyze the CSI data column (first row as example)
    csi_data = df['data'].iloc[0]
    print(f"CSI data sample length: {len(str(csi_data))}")
    print(f"CSI data sample preview: {str(csi_data)[:100]}...")
