import torch
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import time
from tqdm import tqdm

from model import create_model  # Using standard model creation function
from resnet_spectrogram_dataloader import ResnetSpectrogramDataset, CSI_SUBCARRIERS

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch following reference implementation approach."""
    model.train()
    running_loss = 0.0
    
    # Use tqdm for progress tracking (reference implementation style)
    for batch in tqdm(train_loader, desc='Training', unit='batch', leave=False):
        # Unpack batch and move to device
        feature_window, positions = [x.to(device) for x in batch]
        feature_window = feature_window.float()  # Ensure float type
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(feature_window)
        loss = criterion(outputs, positions)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate the model following reference implementation approach."""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating', unit='batch', leave=False):
            # Unpack batch and move to device
            feature_window, positions = [x.to(device) for x in batch]
            feature_window = feature_window.float()  # Ensure float type
            
            # Forward pass
            outputs = model(feature_window)
            loss = criterion(outputs, positions)
            running_loss += loss.item()
    
    return running_loss / len(val_loader)

def train_model(data_dir, num_epochs=50, batch_size=128, window_size=351, num_workers=8):
    """
    Train the ResNet18 model with reference implementation configuration.
    
    Args:
        data_dir: Path to dataset
        num_epochs: Maximum number of epochs (default: 50)
        batch_size: Number of samples per batch (default: 128)
        window_size: Number of packets in each window (default: 351)
        num_workers: Number of dataloader workers (default: 8)
    """
    print(f"\nInitializing training with:")
    print(f"- Batch size: {batch_size}")
    print(f"- Window size: {window_size}")
    print(f"- Number of epochs: {num_epochs}")
    # Create save directory for checkpoints
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path('checkpoints') / f'resnet18_{timestamp}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model, criterion, and optimizer with reference configuration
    model, criterion, optimizer = create_model(device)
    
    print("\nCreating dataloaders...")
    # Training data loader (reference implementation approach)
    print("\nCreating training dataloaders...")
    train_datasets = []
    for seq in ['0.csv', '1.csv', '2.csv', '3.csv']:
        print(f"Loading training sequence: {seq}")
        dataset = ResnetSpectrogramDataset(data_dir, window_size=window_size, split=seq)
        train_datasets.append(dataset)
    train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Reference implementation drops incomplete batches
    )
    print(f"Training dataset size: {len(train_dataset)}")
    
    print("\nCreating validation dataloader...")
    # Validation data loader (reference implementation approach)
    val_dataset = ResnetSpectrogramDataset(data_dir, window_size=window_size, split='4.csv')
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    print(f"Validation dataset size: {len(val_dataset)}")
    
    print("\nCreating test dataloader...")
    # Test data loader (reference implementation approach)
    test_dataset = ResnetSpectrogramDataset(data_dir, window_size=window_size, split='5.csv')
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Training loop with reference implementation configuration
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0  # For early stopping with reference implementation approach
    
    print(f'\nStarting ResNet18 training with reference configuration:')
    print(f'- Epochs: {num_epochs}')
    print(f'- Batch size: {batch_size}')
    print(f'- Window size: {window_size}')
    print(f'- Workers: {num_workers}')
    print(f'- Training sequences: 0.csv, 1.csv, 2.csv, 3.csv')
    print(f'- Validation sequence: 4.csv')
    print(f'- Model size: {sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024):.2f}MB')
    
    # Verify performance targets
    with torch.no_grad():
        dummy_input = torch.randn(1, 1, CSI_SUBCARRIERS, window_size).to(device)
        start_time = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        avg_inference_time = (time.time() - start_time) * 10  # ms per inference
    print(f'- Inference time: {avg_inference_time:.2f}ms')
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train and validate
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        # Save losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save best model (reference implementation approach)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter
            }, save_dir / 'best_model.pth')
            print(f'New best validation loss: {best_val_loss:.4f}')
        
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
        
        # Early stopping with patience (reference implementation approach)
        if val_loss > best_val_loss:
            patience_counter += 1
            if patience_counter >= 5:  # Stop after 5 epochs without improvement
                print('Early stopping: validation loss not improving')
                break
        else:
            patience_counter = 0
            best_val_loss = val_loss
    
    print('\nTraining completed!')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Model checkpoints saved in: {save_dir}')
    
    return model, history

if __name__ == '__main__':
    data_dir = 'HALOC'
    model, history = train_model(data_dir)
