import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from tqdm import tqdm
import sys
import os

# Add HALOC repo to path for using their dataset class
sys.path.append('/home/ubuntu/HALOC')
from datasets import HALOC
from model import HALOCNet, HALOCTrainer

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/ubuntu/datasets/haloc/HALOC', help='directory of the HALOC dataset')
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--ws', type=int, default=351, help='feature window size')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='device to use')
    return parser.parse_args()

def train(args):
    print(f"Using device: {args.device}")
    
    # Load datasets using HALOC's dataset class
    print("Loading datasets...")
    train_datasets = [
        HALOC(os.path.join(args.data, f"{i}.csv"), windowSize=args.ws)
        for i in range(4)  # 0.csv through 3.csv
    ]
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = HALOC(os.path.join(args.data, "4.csv"), windowSize=args.ws)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # Reduced batch size for memory efficiency
        shuffle=True,
        num_workers=2,  # Reduced workers
        pin_memory=False,  # Disabled pin_memory to reduce memory usage
        persistent_workers=True  # Keep workers alive between iterations
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,  # Reduced batch size for memory efficiency
        shuffle=False,
        num_workers=2,  # Reduced workers
        pin_memory=False,  # Disabled pin_memory to reduce memory usage
        persistent_workers=True  # Keep workers alive between iterations
    )
    
    # Initialize model and trainer
    model = HALOCNet().to(args.device)
    trainer = HALOCTrainer(
        model,
        learning_rate=args.lr,
        weight_decay=args.wd
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = 'best_model.pth'
    
    print("Starting training...")
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch_idx, (features, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')):
            features = features.float().to(args.device)  # Ensure float32
            labels = labels.float().to(args.device)  # Ensure float32
            
            loss = trainer.train_step(features, labels)
            train_losses.append(loss)
            
            if batch_idx % 10 == 0:
                print(f'\nBatch {batch_idx}, Training loss: {loss:.6f}')
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # Validation phase
        model.eval()
        val_losses = []
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc='Validation'):
                features = features.float().to(args.device)  # Ensure float32
                labels = labels.float().to(args.device)  # Ensure float32
                
                val_loss, predictions = trainer.validate_step(features, labels)
                val_losses.append(val_loss)
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        trainer.scheduler.step(avg_val_loss)
        
        # Calculate validation metrics
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        mse_per_dimension = np.mean((val_predictions - val_targets) ** 2, axis=0)
        
        # Print and log metrics
        log_msg = f'\nEpoch {epoch+1}/{args.epochs}:\n'
        log_msg += f'Average training loss: {avg_train_loss:.6f}\n'
        log_msg += f'Average validation loss: {avg_val_loss:.6f}\n'
        log_msg += f'MSE per dimension (x,y,z): {mse_per_dimension}\n'
        
        print(log_msg)  # Print to console
        
        # Log to file
        with open('train_model.log', 'a') as f:
            f.write(log_msg)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, best_model_path)
            print(f'Saved new best model with validation loss: {best_val_loss:.6f}')

if __name__ == '__main__':
    args = parse_args()
    train(args)
