"""
ViT model training script for CIFAR-100 classification
"""
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from model import create_model
from utils import (
    get_data_loaders, get_optimizer, save_model, 
    load_model, get_device, compute_accuracy, plot_training_history
)


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    
    Parameters:
        model: Model instance
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Training device
        epoch: Current epoch
        
    Returns:
        train_loss: Average training loss
        train_acc: Average training accuracy
    """
    model.train()
    running_loss = 0.0
    running_top1_acc = 0.0
    running_top5_acc = 0.0
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Training]")
    
    for batch_idx, (data, target) in enumerate(pbar):
        # Transfer data to device
        data, target = data.to(device), target.to(device)
        
        # Forward propagation
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward propagation
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        top1_acc, top5_acc = compute_accuracy(output, target)
        
        # Accumulate statistics
        running_loss += loss.item()
        running_top1_acc += top1_acc
        running_top5_acc += top5_acc
        
        # Update progress bar
        if (batch_idx + 1) % config.LOG_INTERVAL == 0 or (batch_idx + 1) == len(train_loader):
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'top1_acc': f"{top1_acc:.2f}%",
                'top5_acc': f"{top5_acc:.2f}%",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
    
    # Calculate average metrics
    train_loss = running_loss / len(train_loader)
    train_top1_acc = running_top1_acc / len(train_loader)
    train_top5_acc = running_top5_acc / len(train_loader)
    
    return train_loss, train_top1_acc


def validate(model, val_loader, criterion, device):
    """
    Validate the model
    
    Parameters:
        model: Model instance
        val_loader: Validation data loader
        criterion: Loss function
        device: Validation device
        
    Returns:
        val_loss: Average validation loss
        val_acc: Average validation accuracy
    """
    model.eval()
    val_loss = 0.0
    val_top1_acc = 0.0
    val_top5_acc = 0.0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[Validation]")
        for data, target in pbar:
            # Transfer data to device
            data, target = data.to(device), target.to(device)
            
            # Forward propagation
            output = model(data)
            loss = criterion(output, target)
            
            # Calculate accuracy
            top1_acc, top5_acc = compute_accuracy(output, target)
            
            # Accumulate statistics
            val_loss += loss.item()
            val_top1_acc += top1_acc
            val_top5_acc += top5_acc
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'top1_acc': f"{top1_acc:.2f}%",
                'top5_acc': f"{top5_acc:.2f}%"
            })
    
    # Calculate average metrics
    val_loss = val_loss / len(val_loader)
    val_top1_acc = val_top1_acc / len(val_loader)
    val_top5_acc = val_top5_acc / len(val_loader)
    
    print(f"Validation - Loss: {val_loss:.4f}, Top-1 Accuracy: {val_top1_acc:.2f}%, Top-5 Accuracy: {val_top5_acc:.2f}%")
    
    return val_loss, val_top1_acc


def test(model, test_loader, device):
    """
    Test the model
    
    Parameters:
        model: Model instance
        test_loader: Test data loader
        device: Test device
        
    Returns:
        test_top1_acc: Average test Top-1 accuracy
        test_top5_acc: Average test Top-5 accuracy
    """
    model.eval()
    test_top1_acc = 0.0
    test_top5_acc = 0.0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="[Testing]")
        for data, target in pbar:
            # Transfer data to device
            data, target = data.to(device), target.to(device)
            
            # Forward propagation
            output = model(data)
            
            # Calculate accuracy
            top1_acc, top5_acc = compute_accuracy(output, target)
            
            # Accumulate statistics
            test_top1_acc += top1_acc
            test_top5_acc += top5_acc
            
            # Update progress bar
            pbar.set_postfix({
                'top1_acc': f"{top1_acc:.2f}%",
                'top5_acc': f"{top5_acc:.2f}%"
            })
    
    # Calculate average metrics
    test_top1_acc = test_top1_acc / len(test_loader)
    test_top5_acc = test_top5_acc / len(test_loader)
    
    print(f"Test - Top-1 Accuracy: {test_top1_acc:.2f}%, Top-5 Accuracy: {test_top5_acc:.2f}%")
    
    return test_top1_acc, test_top5_acc


def main():
    """
    Main training function
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='CIFAR-100 ViT Training')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='Training epochs')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume training')
    parser.add_argument('--evaluate', action='store_true', help='Evaluation mode only')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path for evaluation mode')
    args = parser.parse_args()
    
    # Update configuration
    if args.batch_size != config.BATCH_SIZE:
        config.BATCH_SIZE = args.batch_size
    if args.lr != config.LEARNING_RATE:
        config.LEARNING_RATE = args.lr
    if args.epochs != config.EPOCHS:
        config.EPOCHS = args.epochs
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create model
    model = create_model()
    model = model.to(device)
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders()
    print(f"Training set size: {len(train_loader.dataset)}, Validation set size: {len(val_loader.dataset)}, Test set size: {len(test_loader.dataset)}")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Get optimizer and scheduler
    optimizer, scheduler = get_optimizer(model)
    
    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Evaluation mode
    if args.evaluate:
        if args.checkpoint:
            model, _, _ = load_model(model, args.checkpoint)
        else:
            print("Evaluation mode requires checkpoint path.")
            return
        
        model.eval()
        test_top1_acc, test_top5_acc = test(model, test_loader, device)
        return
    
    # Resume training
    start_epoch = 0
    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    if args.resume:
        model, start_epoch, best_val_acc = load_model(model, args.resume)
        print(f"Resuming training from epoch {start_epoch+1}")
    
    # Early stopping counter
    patience_counter = 0
    
    # Training loop
    for epoch in range(start_epoch, config.EPOCHS):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate model
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(
                model, epoch, optimizer, scheduler, val_acc,
                os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
            )
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Periodic checkpoint saving
        if (epoch + 1) % config.SAVE_FREQ == 0:
            save_model(
                model, epoch, optimizer, scheduler, val_acc,
                os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch{epoch+1}.pth')
            )
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{config.EPOCHS} - "
              f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%, "
              f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"Early stopping: Validation accuracy did not improve for {config.PATIENCE} epochs")
            break
    
    # Plot training history
    plot_training_history(
        train_losses, val_losses, train_accs, val_accs,
        os.path.join(config.CHECKPOINT_DIR, 'training_history.png')
    )
    
    # Load best model and evaluate on test set
    model, _, _ = load_model(model, os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))
    test_top1_acc, test_top5_acc = test(model, test_loader, device)
    
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%, Test Top-1 Accuracy: {test_top1_acc:.2f}%, Test Top-5 Accuracy: {test_top5_acc:.2f}%")


if __name__ == "__main__":
    main() 