import torch
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
from pathlib import Path
import json

from model import CSILocalizationNet
from dataloader import create_dataloaders

# Enable 3D plotting
plt.rcParams['figure.figsize'] = [12, 8]

def load_best_model(device):
    """Load the best model from checkpoints."""
    # Get the latest checkpoint directory
    checkpoint_dirs = sorted(Path('checkpoints').glob('*'))
    if not checkpoint_dirs:
        raise ValueError("No checkpoint directories found")
    latest_dir = checkpoint_dirs[-1]
    
    # Load model
    model = CSILocalizationNet().to(device)
    checkpoint = torch.load(latest_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def evaluate_test_set():
    """Evaluate model performance on test set (5.csv)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load best model
    model = load_best_model(device)
    model.eval()
    
    # Create test dataloader
    dataloaders = create_dataloaders('HALOC', batch_size=32)
    test_loader = dataloaders['test']
    
    # Lists to store predictions and ground truth
    all_preds = []
    all_true = []
    test_losses = []
    
    # Evaluate model
    criterion = torch.nn.MSELoss()
    print("\nEvaluating on test set (5.csv)...")
    
    with torch.no_grad():
        for csi, positions in test_loader:
            csi = csi.to(device)
            positions = positions.to(device)
            
            # Forward pass
            outputs = model(csi)
            loss = criterion(outputs, positions)
            test_losses.append(loss.item())
            
            # Store predictions and ground truth
            all_preds.extend(outputs.cpu().numpy())
            all_true.extend(positions.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_preds)
    ground_truth = np.array(all_true)
    
    # Calculate error metrics
    errors = np.abs(predictions - ground_truth)
    mean_error = np.mean(errors, axis=0)
    rmse = np.sqrt(np.mean(np.square(errors), axis=0))
    
    print("\nTest Set Performance:")
    print(f"Average Test Loss: {np.mean(test_losses):.4f}")
    print("\nMean Absolute Error per dimension:")
    print(f"X: {mean_error[0]:.4f} meters")
    print(f"Y: {mean_error[1]:.4f} meters")
    print(f"Z: {mean_error[2]:.4f} meters")
    print("\nRMSE per dimension:")
    print(f"X: {rmse[0]:.4f} meters")
    print(f"Y: {rmse[1]:.4f} meters")
    print(f"Z: {rmse[2]:.4f} meters")
    
    # Create 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert to float32 to ensure compatibility
    ground_truth = ground_truth.astype(np.float32)
    predictions = predictions.astype(np.float32)
    
    # Plot ground truth
    ax.scatter(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], 
              c='blue', marker='o', label='Ground Truth', alpha=0.6)
    
    # Plot predictions
    ax.scatter(predictions[:, 0], predictions[:, 1], predictions[:, 2], 
              c='red', marker='^', label='Predictions', alpha=0.6)
    
    # Set labels and title
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('3D Position Predictions vs Ground Truth')
    ax.legend()
    
    # Adjust figure layout
    plt.tight_layout()
    
    plt.savefig('test_set_visualization.png')
    plt.close()
    
    # Save detailed results
    results = {
        'mean_test_loss': float(np.mean(test_losses)),
        'mean_absolute_error': mean_error.tolist(),
        'rmse': rmse.tolist(),
        'num_test_samples': len(ground_truth)
    }
    
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == '__main__':
    evaluate_test_set()
