import torch
import torch.nn as nn
from model import CSILocalizationNet
from dataloader import create_dataloaders
import json
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_overfitting():
    """Analyze training vs validation loss to detect overfitting."""
    # Load training history
    checkpoint_dirs = sorted(Path('checkpoints').glob('*'))
    if not checkpoint_dirs:
        raise ValueError("No checkpoint directories found")
    latest_dir = checkpoint_dirs[-1]
    
    with open(latest_dir / 'history.json', 'r') as f:
        history = json.load(f)
    
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    
    # Calculate overfitting metrics
    train_final = train_losses[-1]
    val_final = val_losses[-1]
    gap = val_final - train_final
    gap_percentage = (gap / train_final) * 100
    
    print("\nOverfitting Analysis:")
    print(f"Final training loss: {train_final:.4f}")
    print(f"Final validation loss: {val_final:.4f}")
    print(f"Gap between validation and training: {gap:.4f} ({gap_percentage:.1f}%)")
    
    # Plot training vs validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training vs Validation Loss (Overfitting Analysis)')
    plt.legend()
    plt.grid(True)
    plt.savefig('overfitting_analysis.png')
    plt.close()
    
    return gap_percentage

def apply_regularization():
    """Apply additional regularization based on overfitting analysis."""
    gap_percentage = analyze_overfitting()
    
    # Current regularization techniques
    current_techniques = {
        'dropout': 0.2,
        'batch_normalization': True,
        'early_stopping': True
    }
    
    # Recommend additional regularization based on gap
    print("\nRegularization Recommendations:")
    if gap_percentage > 20:
        print("High overfitting detected. Recommendations:")
        print("1. Increase dropout rate to 0.3")
        print("2. Add L2 regularization (weight decay=1e-4)")
        print("3. Consider reducing model capacity")
        
        # Update model with stronger regularization
        model = CSILocalizationNet(dropout_rate=0.3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
    elif gap_percentage > 10:
        print("Moderate overfitting detected. Recommendations:")
        print("1. Increase dropout rate to 0.25")
        print("2. Add L2 regularization (weight decay=1e-5)")
        
        # Update model with moderate regularization
        model = CSILocalizationNet(dropout_rate=0.25)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
    else:
        print("Low overfitting detected. Current regularization appears sufficient:")
        print("1. Keep current dropout rate (0.2)")
        print("2. Maintain batch normalization")
        print("3. Continue with early stopping")
        
        # Keep current model settings
        model = CSILocalizationNet(dropout_rate=0.2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\nCurrent Regularization Techniques:")
    for technique, value in current_techniques.items():
        print(f"- {technique}: {value}")
    
    return model, optimizer

if __name__ == '__main__':
    model, optimizer = apply_regularization()
