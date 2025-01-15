import torch
import numpy as np
import time
from pathlib import Path
from model import CSILocalizationNet
from dataloader import create_dataloaders
from prepare_dataset import HALOCDataset

def test_inference_time(model, test_loader, num_runs=100):
    """Test inference time with both single samples and batches."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Get a batch and a single sample
    batch_csi, _ = next(iter(test_loader))
    single_csi = batch_csi[0:1]  # Keep batch dimension
    
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = model(single_csi)
            _ = model(batch_csi)
    
    # Test single sample inference
    single_times = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            _ = model(single_csi)
        single_times.append((time.time() - start) * 1000)  # Convert to ms
    
    # Test batch inference
    batch_times = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            _ = model(batch_csi)
        batch_times.append((time.time() - start) * 1000)  # Convert to ms
    
    return {
        'single_sample': {
            'mean': np.mean(single_times),
            'std': np.std(single_times),
            'max': np.max(single_times)
        },
        'batch': {
            'mean': np.mean(batch_times),
            'std': np.std(batch_times),
            'max': np.max(batch_times),
            'per_sample': np.mean(batch_times) / len(batch_csi)
        }
    }

def test_preprocessing_time(dataset, num_runs=100):
    """Test CSI preprocessing time."""
    # Get a sample
    sample = dataset.load_split('test').iloc[0]
    
    # Warm up
    for _ in range(10):
        _ = dataset.process_csi_data(sample['data'])
    
    # Test preprocessing time
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = dataset.process_csi_data(sample['data'])
        times.append((time.time() - start) * 1000)  # Convert to ms
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'max': np.max(times)
    }

def main():
    # Create model and dataset
    model = CSILocalizationNet()
    dataset = HALOCDataset('HALOC')
    dataloaders = create_dataloaders('HALOC', batch_size=32)
    
    # Test preprocessing time
    print("\nTesting preprocessing performance...")
    preproc_times = test_preprocessing_time(dataset)
    print(f"Preprocessing time (ms):")
    print(f"Mean: {preproc_times['mean']:.2f} ± {preproc_times['std']:.2f}")
    print(f"Max: {preproc_times['max']:.2f}")
    
    # Test inference time
    print("\nTesting inference performance...")
    inf_times = test_inference_time(model, dataloaders['test'])
    
    print("\nSingle sample inference (ms):")
    print(f"Mean: {inf_times['single_sample']['mean']:.2f} ± {inf_times['single_sample']['std']:.2f}")
    print(f"Max: {inf_times['single_sample']['max']:.2f}")
    
    print("\nBatch inference (ms):")
    print(f"Mean: {inf_times['batch']['mean']:.2f} ± {inf_times['batch']['std']:.2f}")
    print(f"Max: {inf_times['batch']['max']:.2f}")
    print(f"Per sample in batch: {inf_times['batch']['per_sample']:.2f}")
    
    # Check model size
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)  # Size in MB
    print(f"\nModel size: {model_size:.2f} MB")
    
    # Verify targets
    print("\nPerformance Targets:")
    print(f"✓ Inference time < 20ms: {inf_times['single_sample']['max'] < 20}")
    print(f"✓ Model size < 10MB: {model_size < 10}")
    print(f"✓ Batch processing supported: {inf_times['batch']['per_sample'] < inf_times['single_sample']['mean']}")
    print(f"✓ CPU-only deployment: {torch.cuda.is_available() == False}")

if __name__ == '__main__':
    main()
