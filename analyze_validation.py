import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from model import CSILocalizationNet
from dataloader import create_dataloaders

def load_training_history():
    """Load training history from the latest checkpoint directory."""
    # Get the latest checkpoint directory
    checkpoint_dirs = sorted(Path('checkpoints').glob('*'))
    if not checkpoint_dirs:
        raise ValueError("No checkpoint directories found")
    latest_dir = checkpoint_dirs[-1]
    
    # Load history
    with open(latest_dir / 'history.json', 'r') as f:
        history = json.load(f)
    return history, latest_dir

def plot_training_curves(history):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_losses'], label='Training Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    plt.close()

def analyze_validation_performance():
    """Analyze the model's validation performance."""
    # Load training history
    history, checkpoint_dir = load_training_history()
    
    # Plot training curves
    plot_training_curves(history)
    
    print("\nValidation Performance Analysis:")
    print(f"Best validation loss: {history['best_val_loss']:.4f}")
    
    # Calculate improvement
    initial_val_loss = history['val_losses'][0]
    final_val_loss = history['val_losses'][-1]
    improvement = (initial_val_loss - final_val_loss) / initial_val_loss * 100
    
    print(f"\nTraining Statistics:")
    print(f"Initial validation loss: {initial_val_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Improvement: {improvement:.1f}%")
    
    # Analyze convergence
    val_losses = history['val_losses']
    window_size = 5
    rolling_mean = np.convolve(val_losses, np.ones(window_size)/window_size, mode='valid')
    
    # Check if loss has stabilized
    last_mean = rolling_mean[-1]
    variation = np.std(rolling_mean[-10:]) / last_mean
    
    print("\nConvergence Analysis:")
    print(f"Recent loss variation: {variation:.1%}")
    if variation < 0.05:
        print("Model has converged (loss variation < 5%)")
    else:
        print("Model might benefit from further training")

if __name__ == '__main__':
    analyze_validation_performance()
