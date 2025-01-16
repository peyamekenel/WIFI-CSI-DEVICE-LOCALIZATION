import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def monitor_training():
    """Monitor training progress and plot learning curves."""
    # Get latest checkpoint directory
    checkpoint_dirs = sorted(Path('checkpoints').glob('resnet18_*'))
    if not checkpoint_dirs:
        print("No checkpoint directories found")
        return
    latest_dir = checkpoint_dirs[-1]
    
    # Load training history
    history_file = latest_dir / 'history.json'
    if not history_file.exists():
        print("No history.json found")
        return
        
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Extract losses
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    best_val_loss = history.get('best_val_loss', float('inf'))
    
    # Calculate RMSE (sqrt of MSE loss)
    train_rmse = np.sqrt(train_losses)
    val_rmse = np.sqrt(val_losses)
    best_rmse = np.sqrt(best_val_loss)
    
    # Create learning curves plot
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_rmse, 'b-', label='Training RMSE')
    plt.plot(epochs, val_rmse, 'r-', label='Validation RMSE')
    plt.axhline(y=0.197, color='g', linestyle='--', label='Target RMSE (0.197m)')
    
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (meters)')
    plt.legend()
    plt.grid(True)
    
    # Add current stats
    plt.text(0.02, 0.98, f'Current epoch: {len(epochs)}/50\n'
             f'Best val RMSE: {best_rmse:.4f}m\n'
             f'Target RMSE: 0.197m\n'
             f'Gap: {(best_rmse - 0.197):.4f}m',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_file = f'training_progress_{timestamp}.png'
    plt.savefig(plot_file)
    plt.close()
    
    print(f"\nTraining Progress (Epoch {len(epochs)}/50):")
    print(f"Current training RMSE: {train_rmse[-1]:.4f}m")
    print(f"Current validation RMSE: {val_rmse[-1]:.4f}m")
    print(f"Best validation RMSE: {best_rmse:.4f}m")
    print(f"Target RMSE: 0.197m")
    print(f"Gap to target: {(best_rmse - 0.197):.4f}m")
    print(f"\nProgress plot saved as: {plot_file}")

if __name__ == '__main__':
    monitor_training()
