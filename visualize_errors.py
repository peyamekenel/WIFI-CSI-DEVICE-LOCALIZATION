import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import torch
from model import CSILocalizationNet
from dataloader import create_dataloaders

def load_predictions():
    """Load model and generate predictions for test set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load best model
    checkpoint_dirs = sorted(Path('checkpoints').glob('*'))
    if not checkpoint_dirs:
        raise ValueError("No checkpoint directories found")
    latest_dir = checkpoint_dirs[-1]
    
    model = CSILocalizationNet().to(device)
    checkpoint = torch.load(latest_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create test dataloader
    dataloaders = create_dataloaders('HALOC', batch_size=32)
    test_loader = dataloaders['test']
    
    # Collect predictions and ground truth
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for csi, positions in test_loader:
            csi = csi.to(device)
            outputs = model(csi)
            all_preds.extend(outputs.cpu().numpy())
            all_true.extend(positions.cpu().numpy())
    
    return np.array(all_preds), np.array(all_true)

def plot_error_distributions(predictions, ground_truth):
    """Create error distribution plots for each dimension."""
    errors = predictions - ground_truth
    dimensions = ['X', 'Y', 'Z']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Error Distributions by Dimension')
    
    for i, (ax, dim) in enumerate(zip(axes, dimensions)):
        ax.hist(errors[:, i], bins=50, alpha=0.7)
        ax.set_title(f'{dim} Dimension Errors')
        ax.set_xlabel(f'{dim} Error (meters)')
        ax.set_ylabel('Count')
        
        # Add mean and std annotations
        mean = np.mean(errors[:, i])
        std = np.std(errors[:, i])
        ax.axvline(mean, color='r', linestyle='--', alpha=0.5)
        ax.text(0.05, 0.95, f'Mean: {mean:.3f}m\nStd: {std:.3f}m',
                transform=ax.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('error_distributions.png')
    plt.close()

def plot_error_heatmap(predictions, ground_truth):
    """Create 2D heatmap of error magnitudes."""
    errors = np.abs(predictions - ground_truth)
    total_error = np.sqrt(np.sum(np.square(errors), axis=1))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(ground_truth[:, 0], ground_truth[:, 1],
                        c=total_error, cmap='viridis',
                        alpha=0.6, s=50)
    plt.colorbar(scatter, label='Total Error (meters)')
    
    ax.set_xlabel('X Position (meters)')
    ax.set_ylabel('Y Position (meters)')
    ax.set_title('Error Magnitude Distribution in Space')
    
    plt.tight_layout()
    plt.savefig('error_heatmap.png')
    plt.close()

def main():
    print("Loading predictions and ground truth...")
    predictions, ground_truth = load_predictions()
    
    print("\nGenerating error distribution plots...")
    plot_error_distributions(predictions, ground_truth)
    
    print("Creating error heatmap...")
    plot_error_heatmap(predictions, ground_truth)
    
    print("\nVisualization files created:")
    print("- error_distributions.png")
    print("- error_heatmap.png")

if __name__ == '__main__':
    main()
