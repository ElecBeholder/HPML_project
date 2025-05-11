"""
Utility functions for CIFAR-100 classification
"""
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import config


def get_data_loaders():
    """
    Create CIFAR-100 data loaders
    
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
    """
    # Create data augmentation transforms
    if config.USE_AUGMENTATION:
        train_transform = transforms.Compose([
            transforms.RandomCrop(config.IMG_SIZE, padding=config.RANDOM_CROP_PADDING),
            transforms.RandomHorizontalFlip() if config.RANDOM_FLIP else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    
    # Test data transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR100(
        root=config.DATASET_PATH, 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    # Split training and validation sets (90% train, 10% validation)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Validation set uses test transforms
    val_dataset.dataset.transform = test_transform
    
    # Test set
    test_dataset = datasets.CIFAR100(
        root=config.DATASET_PATH, 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_optimizer(model):
    """
    Create optimizer and learning rate scheduler
    
    Parameters:
        model: Model instance
        
    Returns:
        optimizer: Optimizer
        scheduler: Learning rate scheduler
    """
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Create learning rate scheduler - cosine decay with warmup
    def warmup_cosine_schedule(epoch):
        if epoch < config.WARMUP_EPOCHS:
            return epoch / config.WARMUP_EPOCHS
        else:
            # Prevent division by zero
            remaining_epochs = max(1, config.EPOCHS - config.WARMUP_EPOCHS)
            return 0.5 * (1 + np.cos(np.pi * (epoch - config.WARMUP_EPOCHS) / remaining_epochs))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_schedule)
    
    return optimizer, scheduler


def save_model(model, epoch, optimizer, scheduler, val_acc, checkpoint_path):
    """
    Save model checkpoint
    
    Parameters:
        model: Model instance
        epoch: Current epoch
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        val_acc: Validation accuracy
        checkpoint_path: Save path
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_acc': val_acc
    }, checkpoint_path)
    
    print(f"Model saved to {checkpoint_path}")


def load_model(model, checkpoint_path):
    """
    Load model checkpoint
    
    Parameters:
        model: Model instance
        checkpoint_path: Checkpoint path
        
    Returns:
        model: Model with loaded checkpoint
        epoch: Epoch when checkpoint was saved
        val_acc: Validation accuracy when checkpoint was saved
    """
    # Use weights_only=False to address new default behavior in PyTorch 2.6
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    
    print(f"Loaded model from {checkpoint_path}, epoch {epoch}, validation accuracy {val_acc:.4f}")
    
    return model, epoch, val_acc


def get_device():
    """
    Get available device
    
    Returns:
        device: Training device (CPU or GPU)
    """
    return torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")


def compute_accuracy(outputs, targets):
    """
    Calculate Top-1 and Top-5 accuracy
    
    Parameters:
        outputs: Model outputs (logits)
        targets: Target labels
        
    Returns:
        top1_acc: Top-1 accuracy
        top5_acc: Top-5 accuracy
    """
    batch_size = targets.size(0)
    
    # Get Top-5 predicted classes
    _, pred = outputs.topk(5, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    
    # Top-1 accuracy
    top1_correct = correct[:1].reshape(-1).float().sum(0)
    top1_acc = top1_correct.item() * 100.0 / batch_size
    
    # Top-5 accuracy
    top5_correct = correct[:5].reshape(-1).float().sum(0)
    top5_acc = top5_correct.item() * 100.0 / batch_size
    
    return top1_acc, top5_acc


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Plot training history charts
    
    Parameters:
        train_losses: Training loss list
        val_losses: Validation loss list
        train_accs: Training accuracy list
        val_accs: Validation accuracy list
        save_path: Save chart path
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss chart
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy chart
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Chart saved to {save_path}")
    
    plt.show() 