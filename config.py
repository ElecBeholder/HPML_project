"""
Configuration parameters for Vision Transformer for CIFAR-100 classification
"""

# Dataset parameters
DATASET_PATH = './data'
NUM_CLASSES = 100
IMG_SIZE = 32  # CIFAR-100 image size
CHANNELS = 3   # RGB images

# ViT model parameters
PATCH_SIZE = 2  # Divide 32x32 image into 8 4x4 patches
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # Number of patches in the image
EMBED_DIM = 384  # Embedding dimension
NUM_HEADS = 6    # Number of attention heads
NUM_LAYERS = 7   # Number of Transformer layers
MLP_RATIO = 4    # Hidden dimension ratio in MLP layers
DROPOUT = 0.1    # Dropout ratio

# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 100
WARMUP_EPOCHS = 2    # Reduced warmup epochs from 5 to 2 to accommodate shorter training cycles
PATIENCE = 100    # Early stopping patience value

# Data augmentation parameters
USE_AUGMENTATION = True
RANDOM_CROP_PADDING = 4  # Random crop padding
RANDOM_FLIP = True       # Random horizontal flip

# Device configuration
DEVICE = 'cuda'  # 'cuda' or 'cpu'

# Model saving parameters
CHECKPOINT_DIR = './checkpoints'
SAVE_FREQ = 10   # Save model every N epochs

# Logging parameters
LOG_INTERVAL = 20  # Log output every N batches 