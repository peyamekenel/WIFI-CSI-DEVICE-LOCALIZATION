import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import json
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import create_model
from resnet_spectrogram_dataloader import ResnetSpectrogramDataset, CSI_SUBCARRIERS

def load_best_model(device):
    """Load the best ResNet18 model from checkpoints."""
    # Get the latest checkpoint directory with 'resnet18' in name
    checkpoint_dirs = sorted(Path('checkpoints').glob('resnet18_*'))
    if not checkpoint_dirs:
        raise ValueError("No ResNet18 checkpoint directories found")
    latest_dir = checkpoint_dirs[-1]
    print(f"\nLoading model from checkpoint: {latest_dir}")
    
    # Load model with reference implementation configuration
    model, _, _ = create_model(device)  # We don't need criterion and optimizer for evaluation
    checkpoint = torch.load(latest_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, latest_dir

def evaluate_test_set():
    """Evaluate ResNet18 model performance on test set (5.csv)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load best model and get checkpoint directory
    model, checkpoint_dir = load_best_model(device)
    model.eval()
    
    # Create test dataloader with reference implementation configuration
    window_size = 351  # Reference implementation window size
    batch_size = 128   # Reference implementation batch size
    print(f"\nUsing reference implementation configuration:")
    print(f"- Window size: {window_size}")
    print(f"- Batch size: {batch_size}")
    
    # Create test dataset and dataloader
    test_dataset = ResnetSpectrogramDataset('HALOC', window_size=window_size, split='5.csv')
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,  # Reference implementation workers
        pin_memory=True
    )
    print(f"\nTest dataset size: {len(test_dataset)}")
    
    # Calculate model size
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"\nModel size: {model_size:.2f}MB (target: <10MB)")
    
    # Test preprocessing time
    start_time = time.time()
    for _ in range(100):
        _ = test_dataset._process_single_sample(0)
    preproc_time = (time.time() - start_time) * 10  # ms per sample
    print(f"Preprocessing time: {preproc_time:.2f}ms per sample (target: <1ms)")
    
    # Test inference time
    dummy_input = torch.randn(1, 1, CSI_SUBCARRIERS, window_size).to(device)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    inference_time = (time.time() - start_time) * 10  # ms per sample
    print(f"Inference time: {inference_time:.2f}ms per sample (target: <20ms)")
    
    # Lists to store predictions and ground truth
    all_preds = []
    all_true = []
    test_losses = []
    
    # Evaluate model
    criterion = torch.nn.MSELoss()
    print("\nEvaluating ResNet18 on test subset...")
    
    from tqdm import tqdm
    with torch.no_grad():
        for batch_idx, (spectrograms, positions) in enumerate(tqdm(test_loader, desc="Evaluating")):
            spectrograms = spectrograms.to(device)
            positions = positions.to(device)
            
            # Forward pass
            outputs = model(spectrograms)
            loss = criterion(outputs, positions)
            test_losses.append(loss.item())
            
            # Store predictions and ground truth
            all_preds.extend(outputs.cpu().numpy())
            all_true.extend(positions.cpu().numpy())
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"\nBatch [{batch_idx + 1}/{len(test_loader)}], "
                      f"Loss: {loss.item():.4f}")
    
    # Convert to numpy arrays
    predictions = np.array(all_preds)
    ground_truth = np.array(all_true)
    
    # Calculate error metrics
    errors = np.abs(predictions - ground_truth)
    mean_error = np.mean(errors, axis=0)
    rmse = np.sqrt(np.mean(np.square(errors), axis=0))
    overall_rmse = np.sqrt(np.mean(np.square(errors)))
    
    print("\nResNet18 Test Set Performance:")
    print(f"Average Test Loss: {np.mean(test_losses):.4f}")
    print("\nMean Absolute Error per dimension:")
    print(f"X: {mean_error[0]:.4f} meters")
    print(f"Y: {mean_error[1]:.4f} meters")
    print(f"Z: {mean_error[2]:.4f} meters")
    print("\nRMSE per dimension:")
    print(f"X: {rmse[0]:.4f} meters")
    print(f"Y: {rmse[1]:.4f} meters")
    print(f"Z: {rmse[2]:.4f} meters")
    print(f"\nOverall RMSE: {overall_rmse:.4f} meters")
    
    # Compare with reference implementation
    reference_rmse = 0.197  # meters (from paper)
    print("\nComparison with reference implementation:")
    print("Dimension  Current    Reference  Gap")
    print("-" * 45)
    for dim, current in zip(['x', 'y', 'z'], rmse):
        gap = current - reference_rmse
        print(f"{dim.upper()}-axis:  {current:.4f}m    {reference_rmse:.4f}m    {gap:+.4f}m")
    
    overall_gap = overall_rmse - reference_rmse
    print(f"Overall:   {overall_rmse:.4f}m    {reference_rmse:.4f}m    {overall_gap:+.4f}m")
    
    # Verify technical requirements
    print("\nTechnical Requirements:")
    print(f"✓ Model size: {'PASS' if model_size < 10 else 'FAIL'} ({model_size:.2f}MB < 10MB)")
    print(f"✓ Preprocessing: {'PASS' if preproc_time < 1 else 'FAIL'} ({preproc_time:.2f}ms < 1ms)")
    print(f"✓ Inference: {'PASS' if inference_time < 20 else 'FAIL'} ({inference_time:.2f}ms < 20ms)")
    print(f"✗ Accuracy: FAIL (gap: +{overall_gap:.4f}m)")
    
    # Save detailed results
    results = {
        'rmse': {
            'current': {
                'x': float(rmse[0]),
                'y': float(rmse[1]),
                'z': float(rmse[2]),
                'overall': float(overall_rmse)
            },
            'reference': float(reference_rmse),
            'gap': float(overall_gap)
        },
        'technical': {
            'model_size_mb': float(model_size),
            'preproc_time_ms': float(preproc_time),
            'inference_time_ms': float(inference_time),
            'requirements_met': {
                'size': model_size < 10,
                'preproc': preproc_time < 1,
                'inference': inference_time < 20,
                'accuracy': overall_rmse <= reference_rmse
            }
        }
    }
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    ax = Axes3D(fig)  # This is the correct way to create a 3D axis
    
    # Sample a subset of points for clearer visualization
    sample_size = min(1000, len(ground_truth))
    indices = np.random.choice(len(ground_truth), sample_size, replace=False)
    
    # Plot ground truth
    ax.scatter(ground_truth[indices, 0], 
              ground_truth[indices, 1], 
              ground_truth[indices, 2], 
              c='blue', marker='o', label='Ground Truth', alpha=0.6)
    
    # Plot predictions
    ax.scatter(predictions[indices, 0], 
              predictions[indices, 1], 
              predictions[indices, 2], 
              c='red', marker='^', label='Predictions', alpha=0.6)
    
    # Set labels and title
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('ResNet18: 3D Position Predictions vs Ground Truth')
    ax.legend()
    
    # Adjust the view for better visualization
    ax.view_init(elev=20, azim=45)
    
    # Calculate axis limits
    x_min = float(min(ground_truth[:, 0].min(), predictions[:, 0].min()))
    x_max = float(max(ground_truth[:, 0].max(), predictions[:, 0].max()))
    y_min = float(min(ground_truth[:, 1].min(), predictions[:, 1].min()))
    y_max = float(max(ground_truth[:, 1].max(), predictions[:, 1].max()))
    z_min = float(min(ground_truth[:, 2].min(), predictions[:, 2].min()))
    z_max = float(max(ground_truth[:, 2].max(), predictions[:, 2].max()))
    
    # Set axis limits with some padding
    padding = 0.1
    ax.set_xlim3d(x_min - padding, x_max + padding)
    ax.set_ylim3d(y_min - padding, y_max + padding)
    ax.set_zlim3d(z_min - padding, z_max + padding)
    
    plt.savefig('resnet18_test_visualization.png')
    plt.close()
    
    # Save detailed results (convert numpy types to Python native types)
    results = {
        'test_metrics': {
            'mean_loss': float(np.mean(test_losses)),
            'mean_absolute_error': [float(x) for x in mean_error],
            'rmse': {
                'x': float(rmse[0]),
                'y': float(rmse[1]),
                'z': float(rmse[2]),
                'overall': float(overall_rmse)
            }
        },
        'reference_comparison': {
            'reference_rmse': float(reference_rmse),
            'gap': float(overall_gap)
        },
        'technical_requirements': {
            'model_size_mb': float(model_size),
            'preproc_time_ms': float(preproc_time),
            'inference_time_ms': float(inference_time),
            'requirements_met': {
                'size': model_size < 10,
                'preproc': preproc_time < 1,
                'inference': inference_time < 20,
                'accuracy': overall_rmse <= reference_rmse
            }
        },
        'dataset_info': {
            'num_test_samples': int(len(ground_truth)),
            'window_size': int(window_size),
            'batch_size': int(batch_size),
            'num_workers': 8
        }
    }
    
    # Print analysis of potential issues
    print("\nPerformance Analysis:")
    print("1. Model shows significant regression from baseline")
    print("2. Possible causes:")
    print("   - Window size (351) might be too large")
    print("   - Learning rate (0.0001) might be too low")
    print("   - Need more training epochs")
    print("   - ResNet18 architecture might need modification")
    print("\nRecommended next steps:")
    print("1. Reduce window size to 256")
    print("2. Increase learning rate to 0.001")
    print("3. Add more residual connections")
    print("4. Train for more epochs")
    
    with open('resnet18_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == '__main__':
    evaluate_test_set()
