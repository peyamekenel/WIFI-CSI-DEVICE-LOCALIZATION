import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time

def calculate_rmse(mse_loss):
    """Convert MSE loss to RMSE in meters."""
    return np.sqrt(mse_loss)

def monitor_training(interval=60):  # Check every 60 seconds
    """Monitor training progress and plot learning curves."""
    while True:
        print("\nMonitoring Training Progress...")
        
        # Get latest checkpoint directory
        checkpoint_dirs = sorted(Path('checkpoints').glob('resnet18_*'))
        if not checkpoint_dirs:
            print("Waiting for first checkpoint directory...")
            time.sleep(interval)
            continue
            
        latest_dir = checkpoint_dirs[-1]
        print(f"Found checkpoint directory: {latest_dir}")
        
        # Load training history
        history_file = latest_dir / 'history.json'
        if not history_file.exists():
            print("\nWaiting for first epoch to complete...")
            print("Current training status:")
            print("- Model initialized successfully")
            print("- Training dataset: 95,087 samples")
            print("- Validation dataset: 27,760 samples")
            print("- Test dataset: 13,926 samples")
            print("- Batch size: 32")
            print("- Expected batches per epoch: 2,971")
            time.sleep(interval)
            continue
        
        try:
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
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            epochs = range(1, len(train_losses) + 1)
            
            # RMSE plot
            ax1.plot(epochs, train_rmse, 'b-', label='Training RMSE')
            ax1.plot(epochs, val_rmse, 'r-', label='Validation RMSE')
            ax1.axhline(y=0.197, color='g', linestyle='--', label='Target RMSE (0.197m)')
            ax1.set_title('RMSE Progress')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('RMSE (meters)')
            ax1.legend()
            ax1.grid(True)
            
            # Convergence analysis
            window = 5  # Rolling window for convergence check
            if len(val_rmse) >= window:
                rolling_mean = np.convolve(val_rmse, np.ones(window)/window, mode='valid')
                rolling_std = np.array([np.std(val_rmse[i:i+window]) for i in range(len(val_rmse)-window+1)])
                convergence_epochs = range(window, len(val_rmse) + 1)
                
                ax2.plot(convergence_epochs, rolling_mean, 'g-', label='Rolling Mean RMSE')
                ax2.fill_between(convergence_epochs, 
                               rolling_mean - rolling_std,
                               rolling_mean + rolling_std,
                               alpha=0.2, color='g')
                ax2.set_title('Convergence Analysis')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('RMSE (meters)')
                ax2.legend()
                ax2.grid(True)
            
            # Add detailed stats
            stats_text = (
                f'Training Progress:\n'
                f'Current epoch: {len(epochs)}/50\n'
                f'Best val RMSE: {best_rmse:.4f}m\n'
                f'Target RMSE: 0.197m\n'
                f'Gap: {(best_rmse - 0.197):.4f}m\n\n'
                f'Convergence Metrics:\n'
                f'Recent std: {np.std(val_rmse[-5:]):.4f}m\n'
                f'Improvement rate: {((val_rmse[0] - val_rmse[-1])/val_rmse[0]*100):.1f}%\n'
                f'Epochs since best: {len(epochs) - np.argmin(val_rmse)}'
            )
            fig.text(0.02, 0.02, stats_text,
                    transform=fig.transFigure,
                    verticalalignment='bottom',
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
            
            # Check if training is complete
            if len(epochs) >= 50:
                print("\nTraining completed!")
                break
                
        except Exception as e:
            print(f"Error reading history: {e}")
        
        time.sleep(interval)

if __name__ == '__main__':
    try:
        monitor_training(interval=60)  # Check every 60 seconds
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
