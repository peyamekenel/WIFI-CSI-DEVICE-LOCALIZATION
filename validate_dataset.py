import numpy as np
import pickle
import os

def validate_dataset():
    """Validate the preprocessed CSI dataset."""
    print('Dataset Validation')
    print('-' * 50)
    
    # Check if dataset file exists
    if not os.path.exists('csi_dataset.pkl'):
        raise FileNotFoundError("Dataset file 'csi_dataset.pkl' not found!")
    
    # Load the dataset
    print("Loading dataset...")
    with open('csi_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    # Print shapes and statistics
    print('\nDataset Statistics:')
    print('-' * 50)
    print(f'X_train shape: {dataset["X_train"].shape}')
    print(f'X_test shape: {dataset["X_test"].shape}')
    print(f'Feature dimension: {dataset["X_train"].shape[1]}')
    
    # Verify feature statistics
    print('\nFeature Statistics:')
    print('-' * 50)
    print(f'Mean: {np.mean(dataset["X_train"]):.4f}')
    print(f'Std: {np.std(dataset["X_train"]):.4f}')
    print(f'Min: {np.min(dataset["X_train"]):.4f}')
    print(f'Max: {np.max(dataset["X_train"]):.4f}')
    
    # Verify label distribution
    print('\nLabel Distribution:')
    print('-' * 50)
    unique_labels = np.unique(dataset["y_train"], axis=0)
    print(f'Number of unique combinations: {len(unique_labels)}')
    print('\nUnique distance-angle combinations:')
    for distance, angle in unique_labels:
        count = np.sum((dataset["y_train"] == [distance, angle]).all(axis=1))
        print(f'Distance: {distance}m, Angle: {angle}° - {count} samples')

if __name__ == "__main__":
    validate_dataset()
