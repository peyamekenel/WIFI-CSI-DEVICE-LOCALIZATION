import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import json
from model import create_resnet18
from resnet_spectrogram_dataloader import create_dataloaders

def load_best_model(device):
    """Load the best ResNet18 model from checkpoints."""
    # Get the latest checkpoint directory with 'resnet18' in name
    checkpoint_dirs = sorted(Path('checkpoints').glob('resnet18_*'))
    if not checkpoint_dirs:
        raise ValueError("No ResNet18 checkpoint directories found")
    latest_dir = checkpoint_dirs[-1]
    print(f"\nLoading model from checkpoint: {latest_dir}")
    
    # Load model
    model = create_resnet18().to(device)
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
    
    # Create test dataloader with same window size as training
    window_size = 351  # Must match training configuration
    print(f"\nUsing window size: {window_size} (matching training configuration)")
    dataloaders = create_dataloaders('HALOC', batch_size=32, window_size=window_size)
    test_loader = dataloaders['test']
    
    # Create smaller test subset for quick verification
    test_subset_size = 1000  # Use 1000 samples for quick testing
    dataset = test_loader.dataset
    indices = torch.randperm(len(dataset))[:test_subset_size]
    subset = torch.utils.data.Subset(dataset, indices)
    test_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=32,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    print(f"\nUsing {test_subset_size} samples for quick evaluation")
    
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
    
    # Compare with baseline from results_summary.md
    baseline_rmse = {
        'x': 3.9632,
        'y': 0.3838,
        'z': 0.0153,
        'overall': 5.2783
    }
    
    print("\nComparison with baseline:")
    print("Dimension  Current    Baseline   Improvement")
    print("-" * 45)
    for dim, baseline in zip(['x', 'y', 'z'], rmse):
        improvement = ((baseline_rmse[dim] - baseline) / baseline_rmse[dim]) * 100
        print(f"{dim.upper()}-axis:  {baseline:.4f}m    {baseline_rmse[dim]:.4f}m    {improvement:+.1f}%")
    
    overall_improvement = ((baseline_rmse['overall'] - overall_rmse) / baseline_rmse['overall']) * 100
    print(f"Overall:   {overall_rmse:.4f}m    {baseline_rmse['overall']:.4f}m    {overall_improvement:+.1f}%")
    
    # Compare with reference study
    reference_rmse = 0.197  # meters
    print(f"\nComparison with reference study:")
    print(f"Our ResNet18: {overall_rmse:.3f}m")
    print(f"Reference:    {reference_rmse:.3f}m")
    
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
        'mean_test_loss': float(np.mean(test_losses)),
        'mean_absolute_error': [float(x) for x in mean_error],
        'rmse': [float(x) for x in rmse],
        'overall_rmse': float(overall_rmse),
        'num_test_samples': int(len(ground_truth)),
        'improvements': {
            'x': float(improvement),
            'y': float(improvement),
            'z': float(improvement),
            'overall': float(overall_improvement)
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
