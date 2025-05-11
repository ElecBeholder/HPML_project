# Vision Transformer Implementation for CIFAR-100 Classification

This project implements a Vision Transformer (ViT) model for image classification on the CIFAR-100 dataset. The implementation uses PyTorch framework and leverages CUDA acceleration for training.

## Project Goals

- Use Vision Transformer architecture for CIFAR-100 image classification
- Efficiently utilize GPU acceleration for training
- Provide complete training, validation, and testing workflows
- Implement visualization and model evaluation features

## Key Features

### Gaussian Token Downsampling

To improve efficiency while maintaining performance, we implemented a novel token downsampling technique:

1. **Motivation**: Standard ViT processes a large number of tokens, which can be computationally expensive.
2. **Approach**: We reduce token count from h×w to h/2×w/2 using a Gaussian attention mechanism:
   - Original tokens undergo tokenization and positional embedding
   - A small prediction network (2-layer ViT) generates h/2×w/2 embeddings
   - These embeddings predict Gaussian distribution parameters (μx, μy, σx, σy, ρ)
   - Gaussian distributions are used as attention scores to attend to original tokens
   - The attended values form the downsampled token set
3. **Benefits**:
   - Reduces computational complexity while preserving important information
   - The Gaussian distributions learn to focus on the most informative regions
   - Improves efficiency without significant accuracy degradation

## Project Structure

```
├── README.md                # Project documentation
├── model.py                 # Vision Transformer model definition
├── train.py                 # Training and evaluation scripts
├── utils.py                 # Data loading and utility functions
├── config.py                # Configuration parameters
└── data/                    # Dataset storage directory
    └── cifar-100-python/    # CIFAR-100 dataset
```

## Module Descriptions

### model.py
Defines the Vision Transformer (ViT) model architecture, including:
- `PatchEmbedding`: Divides images into fixed-size patches and projects them into embedding space
- `MultiHeadAttention`: Implementation of multi-head attention mechanism
- `TransformerEncoder`: Contains multiple layers of Transformer encoders
- `ViT`: Complete Vision Transformer model
- `GaussianTokenDownsampler`: Implements token downsampling using Gaussian distributions
- `ViTWithGaussianDownsampling`: Vision Transformer with the Gaussian downsampling mechanism

**Input**: Image batch of size (B, C, H, W), where B is batch size, C is number of channels, H and W are height and width
**Output**: Classification predictions of shape (B, num_classes), with num_classes=100 corresponding to CIFAR-100 classes

### train.py
Implements training and evaluation loops, including:
- Data loading and preprocessing
- Model training loop
- Validation evaluation
- Model saving
- Training process visualization

**Input**: Command line arguments to control training hyperparameters
**Output**: Training logs, saved models, and evaluation results

### utils.py
Contains utility functions, such as:
- Data loading and augmentation
- Metrics calculation
- Training helper tools

### config.py
Centralized storage for configuration parameters:
- Model hyperparameters
- Training parameters
- Data augmentation settings
- Other configuration options

## Usage Instructions

### Requirements
- Python 3.6+
- PyTorch 1.7+
- CUDA support
- torchvision
- numpy
- matplotlib

### Training the Model
```bash
python train.py
```

### Training with Gaussian Downsampling
By default, the model uses Gaussian token downsampling. To switch to the standard ViT:
```bash
python train.py --no_gaussian_downsampling
```

### Custom Training Parameters
```bash
python train.py --batch_size 128 --learning_rate 0.001 --epochs 100
```

### Evaluating a Model
```bash
python train.py --evaluate --checkpoint path/to/checkpoint.pth
```

## Performance Metrics
Expected performance on CIFAR-100:
- Top-1 accuracy: ~39%

## Implementation Details

### Gaussian Token Downsampling
The Gaussian token downsampling mechanism provides an efficient way to reduce the number of tokens while preserving important information:

1. **Initialization**: Gaussian centers are initially uniformly distributed across the token grid
2. **Parameter Prediction**: A small network predicts 5 parameters for each Gaussian distribution:
   - μx, μy: Center coordinates (in normalized [-1,1] space)
   - σx, σy: Standard deviations along x and y axes
   - ρ: Correlation coefficient between x and y
3. **Attention Calculation**: We compute attention scores based on the Gaussian probability density at each token position
4. **Token Aggregation**: Original tokens are weighted and combined according to attention scores

This approach adaptively focuses on the most informative regions of the image, improving both efficiency and effectiveness.