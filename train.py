import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import time

from model import create_model
from dataloader import create_dataloaders

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    for batch_idx, (csi, positions) in enumerate(train_loader):
        # Move data to device
        csi = csi.to(device)
        positions = positions.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(csi)
        loss = criterion(outputs, positions)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item()
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f'Batch [{batch_idx + 1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')
    
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for csi, positions in val_loader:
            csi = csi.to(device)
            positions = positions.to(device)
            
            outputs = model(csi)
            loss = criterion(outputs, positions)
            running_loss += loss.item()
    
    return running_loss / len(val_loader)

def train_model(data_dir, num_epochs=5, batch_size=32):
    """
    Train the model with early stopping and L2 regularization monitoring.
    
    Args:
        data_dir: Path to dataset
        num_epochs: Maximum number of epochs
        batch_size: Batch size for training
    """
    # Create save directory for checkpoints
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path('checkpoints') / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model, criterion, and optimizer
    model, criterion, optimizer = create_model(device)
    
    # Create dataloaders with subsets for quick testing
    dataloaders = create_dataloaders(data_dir, batch_size=batch_size)
    
    # Use full datasets
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f'\nStarting full training with {num_epochs} epochs...')
    print(f'Using L2 regularization with weight_decay={optimizer.param_groups[0]["weight_decay"]}')
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train and validate
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        # Save losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, save_dir / 'best_model.pth')
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
        }
        with open(save_dir / 'history.json', 'w') as f:
            json.dump(history, f)
        
        # Print epoch statistics
        time_elapsed = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss:.4f} '
              f'Val Loss: {val_loss:.4f} '
              f'Time: {time_elapsed:.2f}s')
        
        # Early stopping (optional)
        if val_loss > 1.5 * min(val_losses):
            print('Early stopping due to validation loss increase')
            break
    
    print('\nTraining completed!')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Model checkpoints saved in: {save_dir}')
    
    # Calculate L2 regularization impact
    param_norms = torch.stack([torch.norm(p) for p in model.parameters()])
    l2_norm = torch.sum(param_norms ** 2)
    l2_loss = l2_norm.item() * 1e-4  # Same weight decay as in optimizer
    total_loss = train_losses[-1]
    
    print(f'\nL2 Regularization Analysis:')
    print(f'L2 Loss component: {l2_loss:.4f}')
    print(f'Percentage of total loss: {(l2_loss / total_loss * 100):.2f}%')
    print(f'Parameter L2 norm: {torch.sqrt(l2_norm).item():.4f}')
    
    return model, history

if __name__ == '__main__':
    data_dir = 'HALOC'
    model, history = train_model(data_dir)
