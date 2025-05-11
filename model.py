"""
Vision Transformer model implementation for CIFAR-100 classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import math
import pdb


class GaussianTokenDownsampler(nn.Module):
    """
    Implements token downsampling using Gaussian distribution
    """
    def __init__(self, img_size=32, patch_size=4, embed_dim=384, downsample_factor=2):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.downsample_factor = downsample_factor
        
        # Original token map dimensions
        self.h = self.w = img_size // patch_size
        # Downsampled token map dimensions
        self.h_down = self.h // downsample_factor
        self.w_down = self.w // downsample_factor
        
        # Small ViT prediction network
        # Split into convolutional and Transformer parts to handle dimensionality issues
        self.conv_part = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=downsample_factor, stride=downsample_factor),
            nn.LayerNorm([embed_dim, self.h_down, self.w_down]),
            nn.GELU()
        )
        
        # Transformer part
        self.transformer_part = nn.Sequential(
            TransformerBlock(embed_dim=embed_dim, num_heads=4),
            TransformerBlock(embed_dim=embed_dim, num_heads=4)
        )
        
        # Gaussian distribution parameter prediction network - predicts 5 parameters for each downsampling location
        self.gaussian_predictor = nn.Linear(embed_dim, 5)
        
        # Create learnable parameters for v projection
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Initialize Gaussian distribution parameters
        self._init_gaussian_params()
        
    def _init_gaussian_params(self):
        # Create uniform initial positions for all downsampling locations
        positions = []
        for i in range(self.h_down):
            for j in range(self.w_down):
                # Map positions to [-1,1] range
                pos_x = 2.0 * (j + 0.5) / self.w - 1.0
                pos_y = 2.0 * (i + 0.5) / self.h - 1.0
                positions.append([pos_x, pos_y])
        
        # Initialize the Gaussian predictor's biases to produce reasonable Gaussian parameters
        # Set appropriate initial bias values for the final linear layer
        with torch.no_grad():
            # Initialize μx, μy (will be adjusted in forward)
            bias_init = torch.zeros(5)
            
            # Initialize standard deviations (σx, σy) to reasonable values (parameterized through log)
            # Use larger initial standard deviation for numerical stability
            sigma_init = 0.5  # Increase initial standard deviation
            bias_init[2] = math.log(sigma_init)
            bias_init[3] = math.log(sigma_init)
            
            # Initialize correlation coefficient ρ to 0
            bias_init[4] = 0.0
            
            # Set biases
            self.gaussian_predictor.bias.data.copy_(bias_init)
            # Initialize weights to small random values rather than zeros to help with training
            nn.init.normal_(self.gaussian_predictor.weight, std=0.01)
    
    def _compute_gaussian_attention(self, gaussian_params, token_positions):
        """
        Compute attention scores based on Gaussian distribution
        
        Args:
            gaussian_params: Gaussian parameters of shape (batch_size, h_down*w_down, 5)
            token_positions: Token positions of shape (h*w, 2)
            
        Returns:
            Attention scores of shape (batch_size, h_down*w_down, h*w)
        """
        batch_size = gaussian_params.shape[0]
        n_gaussians = self.h_down * self.w_down
        n_tokens = self.h * self.w
        
        # Parse Gaussian parameters
        mu_x = gaussian_params[:, :, 0].view(batch_size, n_gaussians, 1)  # (B, G, 1)
        mu_y = gaussian_params[:, :, 1].view(batch_size, n_gaussians, 1)  # (B, G, 1)
        
        # Limit sigma range to avoid numerical instability (not too small or too large)
        sigma_x = torch.exp(gaussian_params[:, :, 2].clamp(min=-4.0, max=4.0)).view(batch_size, n_gaussians, 1)  # (B, G, 1)
        sigma_y = torch.exp(gaussian_params[:, :, 3].clamp(min=-4.0, max=4.0)).view(batch_size, n_gaussians, 1)  # (B, G, 1)
        
        # Restrict rho to [-0.9, 0.9] range to avoid singularity near ±1
        rho = 0.9 * torch.tanh(gaussian_params[:, :, 4]).view(batch_size, n_gaussians, 1)  # (B, G, 1)
        
        # Get token positions (x, y)
        x = token_positions[:, 0].view(1, 1, n_tokens)  # (1, 1, T)
        y = token_positions[:, 1].view(1, 1, n_tokens)  # (1, 1, T)
        
        # Calculate bivariate Gaussian distribution
        z_x = (x - mu_x) / (sigma_x + 1e-5)  # (B, G, T), increased epsilon
        z_y = (y - mu_y) / (sigma_y + 1e-5)  # (B, G, T), increased epsilon
        
        # Calculate Gaussian probability density, compute in log space for numerical stability
        # Avoid directly computing exp(-z) / normalizer which may lead to overflow
        exponent = -0.5 / (1 - rho**2 + 1e-5) * (z_x**2 + z_y**2 - 2 * rho * z_x * z_y)
        
        # Limit exponent range to avoid underflow or overflow
        exponent = torch.clamp(exponent, min=-70.0, max=70.0)
        
        # Calculate log-space normalization constant
        log_norm_const = math.log(2 * math.pi) + torch.log(sigma_x) + torch.log(sigma_y) + 0.5 * torch.log(1 - rho**2 + 1e-5)
        
        # Calculate log probability density
        log_gaussian = exponent - log_norm_const
        
        # Convert back to linear space while preventing numerical overflow
        gaussian = torch.exp(log_gaussian)
        
        # Use softmax for normalization, more numerically stable
        # First compute log probabilities, then subtract max value before exponentiation to avoid overflow
        max_log_gaussian = torch.max(log_gaussian, dim=2, keepdim=True)[0]
        attention_unnorm = torch.exp(log_gaussian - max_log_gaussian)
        attention = attention_unnorm / (attention_unnorm.sum(dim=2, keepdim=True) + 1e-10)
        
        return attention

    def forward(self, x):
        """
        Args:
            x: Token sequence of shape (batch_size, n_patches, embed_dim)
        
        Returns:
            Downsampled tokens of shape (batch_size, h_down*w_down, embed_dim)
        """
        batch_size, n_patches, embed_dim = x.shape
        
        # Reshape token sequence into spatial arrangement
        x_spatial = x.view(batch_size, self.h, self.w, embed_dim)
        x_spatial = x_spatial.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Use convolutional part to get downsampled feature map
        conv_out = self.conv_part(x_spatial)  # (B, C, H/2, W/2)
        
        # Reshape for transformer processing
        transformer_in = conv_out.flatten(2).transpose(1, 2)  # (B, H/2*W/2, C)
        
        # Process through transformer blocks
        y = self.transformer_part(transformer_in)  # (B, H/2*W/2, C)
        
        # Predict Gaussian distribution parameters
        base_params = self.gaussian_predictor(y)  # (B, H/2*W/2, 5)
        
        # Create initial uniform distribution of Gaussian centers
        positions = []
        for i in range(self.h_down):
            for j in range(self.w_down):
                pos_x = 2.0 * j / (self.w_down - 1) - 1.0 if self.w_down > 1 else 0.0
                pos_y = 2.0 * i / (self.h_down - 1) - 1.0 if self.h_down > 1 else 0.0
                positions.append([pos_x, pos_y])
        positions = torch.tensor(positions, device=x.device)  # (H/2*W/2, 2)
        
        # Create final Gaussian parameters, adding predefined positions
        gaussian_params = torch.zeros(batch_size, self.h_down * self.w_down, 5, device=x.device)
        
        # Use predefined positions as initial μx and μy, and add offsets predicted by the network
        # Limit offset magnitude to prevent Gaussian centers from moving too far
        offset_scale = 0.2  # Limit offset range
        gaussian_params[:, :, 0] = positions[:, 0].unsqueeze(0) + offset_scale * torch.tanh(base_params[:, :, 0])  # μx
        gaussian_params[:, :, 1] = positions[:, 1].unsqueeze(0) + offset_scale * torch.tanh(base_params[:, :, 1])  # μy
        # Directly use network-predicted standard deviations and correlation coefficient
        gaussian_params[:, :, 2:] = base_params[:, :, 2:]  # σx, σy, ρ
        
        # Calculate token positions (normalized to [-1,1] range)
        token_positions = []
        for i in range(self.h):
            for j in range(self.w):
                # Correctly calculate original token positions, uniformly distributed in [-1,1] range
                pos_x = 2.0 * j / (self.w - 1) - 1.0 if self.w > 1 else 0.0
                pos_y = 2.0 * i / (self.h - 1) - 1.0 if self.h > 1 else 0.0
                token_positions.append([pos_x, pos_y])
        token_positions = torch.tensor(token_positions, device=x.device)  # (H*W, 2)
        
        # Compute attention scores based on Gaussian distribution
        attention = self._compute_gaussian_attention(gaussian_params, token_positions)  # (B, H/2*W/2, H*W)
        
        # Map original tokens to values
        v = self.v_proj(x)  # (B, H*W, C)
        
        # Use attention scores to weight average values
        z = torch.bmm(attention, v)  # (B, H/2*W/2, C)
        
        return z


class PatchEmbedding(nn.Module):
    """
    Divides image into fixed-size patches and projects them into embedding space
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Use convolutional layer to implement image slicing and linear projection
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        """
        Input: (batch_size, in_channels, img_size, img_size)
        Output: (batch_size, n_patches, embed_dim)
        """
        x = self.proj(x)  # (batch_size, embed_dim, h, w)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    """
    def __init__(self, embed_dim=384, num_heads=6, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Input: (batch_size, seq_len, embed_dim)
        Output: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Calculate query, key, value
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention calculation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Output projection
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """
    Multi-layer Perceptron
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * config.MLP_RATIO
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block
    """
    def __init__(self, embed_dim=384, num_heads=6, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, dropout)
        
    def forward(self, x):
        # First residual connection - self-attention
        x = x + self.attn(self.norm1(x))
        # Second residual connection - MLP
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """
    Vision Transformer model
    """
    def __init__(
        self, 
        img_size=config.IMG_SIZE,
        patch_size=config.PATCH_SIZE, 
        in_channels=config.CHANNELS,
        num_classes=config.NUM_CLASSES,
        embed_dim=config.EMBED_DIM, 
        depth=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        mlp_ratio=config.MLP_RATIO,
        dropout=config.DROPOUT
    ):
        super().__init__()
        
        # Image slicing and embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        self.num_patches = self.patch_embed.n_patches
        
        # Learnable class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize parameters
        self._init_weights()
        
    def _init_weights(self):
        # Initialize learnable embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize module weights
        self.apply(self._init_module_weights)
    
    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Get batch size
        batch_size = x.shape[0]
        
        # Perform image patch embedding
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)
        
        # Add class token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (batch_size, n_patches+1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Process through Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Use representation of [CLS] token for classification
        x = self.norm(x)
        x = x[:, 0]  # Take [CLS] token
        x = self.head(x)
        
        return x


class ViTWithGaussianDownsampling(nn.Module):
    """
    Vision Transformer model with Gaussian downsampling
    """
    def __init__(
        self, 
        img_size=config.IMG_SIZE,
        patch_size=config.PATCH_SIZE, 
        in_channels=config.CHANNELS,
        num_classes=config.NUM_CLASSES,
        embed_dim=config.EMBED_DIM, 
        depth=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        mlp_ratio=config.MLP_RATIO,
        dropout=config.DROPOUT,
        downsample_factor=2  # Downsampling factor
    ):
        super().__init__()
        
        # Image slicing and embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        self.num_patches = self.patch_embed.n_patches
        
        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Gaussian token downsampling module
        self.token_downsampler = GaussianTokenDownsampler(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            downsample_factor=downsample_factor
        )
        
        # Downsampled token count
        self.num_downsampled_patches = (img_size // patch_size // downsample_factor) ** 2
        
        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Downsampled positional embedding
        self.downsampled_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_downsampled_patches + 1, embed_dim)
        )
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize parameters
        self._init_weights()
        
    def _init_weights(self):
        # Initialize learnable embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.downsampled_pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize module weights
        self.apply(self._init_module_weights)
    
    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Get batch size
        batch_size = x.shape[0]
        
        # Perform image patch embedding
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Use Gaussian distribution to downsample tokens
        z = self.token_downsampler(x)  # (batch_size, n_patches/4, embed_dim)
        
        # Add class token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        z = torch.cat((cls_token, z), dim=1)  # (batch_size, n_patches/4+1, embed_dim)
        
        # Add downsampled positional embedding
        z = z + self.downsampled_pos_embed
        z = self.pos_drop(z)
        
        # Process through Transformer blocks
        for block in self.blocks:
            z = block(z)
        
        # Use representation of [CLS] token for classification
        z = self.norm(z)
        z = z[:, 0]  # Take [CLS] token
        z = self.head(z)
        
        return z


def create_model(use_gaussian_downsampling=True):
    """
    Create and return model
    """
    if use_gaussian_downsampling:
        model = ViTWithGaussianDownsampling(
            img_size=config.IMG_SIZE,
            patch_size=config.PATCH_SIZE,
            in_channels=config.CHANNELS,
            num_classes=config.NUM_CLASSES,
            embed_dim=config.EMBED_DIM,
            depth=config.NUM_LAYERS,
            num_heads=config.NUM_HEADS,
            mlp_ratio=config.MLP_RATIO,
            dropout=config.DROPOUT
        )
    else:
        model = ViT(
            img_size=config.IMG_SIZE,
            patch_size=config.PATCH_SIZE,
            in_channels=config.CHANNELS,
            num_classes=config.NUM_CLASSES,
            embed_dim=config.EMBED_DIM,
            depth=config.NUM_LAYERS,
            num_heads=config.NUM_HEADS,
            mlp_ratio=config.MLP_RATIO,
            dropout=config.DROPOUT
        )
    return model


if __name__ == "__main__":
    # Model test
    model = create_model(use_gaussian_downsampling=True)
    x = torch.randn(2, config.CHANNELS, config.IMG_SIZE, config.IMG_SIZE)
    output = model(x)
    print(f"Model output shape: {output.shape}")  # Should be [2, 100]
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}") 