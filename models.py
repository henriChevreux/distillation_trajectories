import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for timestep encoding
    """
    def __init__(self, dim):
        super().__init__()
        # Ensure minimum dimension size
        self.dim = max(dim, 2)  # Minimum dimension of 2 to avoid division by zero

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        # Ensure half_dim is at least 1 to avoid division by zero
        half_dim = max(half_dim, 1)
        embeddings = math.log(10000) / (half_dim - 1 + 1e-8)  # Add small epsilon to avoid division by zero
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # Ensure time has shape (batch_size,)
        if time.dim() > 1:
            time = time.squeeze(-1)  # Remove extra dimensions
        
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # Pad or trim to match the original requested dimension
        if embeddings.shape[-1] != self.dim:
            if embeddings.shape[-1] < self.dim:
                # Pad with zeros if needed
                padding = torch.zeros(embeddings.shape[0], self.dim - embeddings.shape[-1], device=device)
                embeddings = torch.cat((embeddings, padding), dim=-1)
            else:
                # Trim if needed
                embeddings = embeddings[:, :self.dim]
        return embeddings

class Block(nn.Module):
    """
    Basic convolutional block with double convolution and residual connection
    """
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch) if time_emb_dim else None
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        
        if in_ch != out_ch:
            self.residual_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x, time_emb=None):
        residual = self.residual_conv(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        
        if time_emb is not None and self.time_mlp is not None:
            time_emb = self.relu(self.time_mlp(time_emb))
            # Reshape time embedding to match spatial dimensions
            if time_emb.dim() == 2:
                time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
            else:
                # If time_emb is already more than 2D, reshape it properly
                time_emb = time_emb.view(time_emb.size(0), -1).unsqueeze(-1).unsqueeze(-1)
            
            # Expand to match x's spatial dimensions
            time_emb = time_emb.expand(-1, -1, x.size(2), x.size(3))
            x = x + time_emb
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        
        return x + residual

class DiffusionUNet(nn.Module):
    """
    A unified U-Net architecture for diffusion models with configurable size
    
    This architecture is used for both teacher and student models, with the only
    difference being the channel dimensions (controlled by size_factor).
    
    All models have the same number of encoder/decoder levels and maintain the
    same spatial resolution flow to ensure consistent output dimensions.
    """
    def __init__(self, config, size_factor=1.0):
        super().__init__()
        
        # Model dimensions
        self.channels = config.channels
        self.size_factor = size_factor
        self.time_emb_dim = max(int(256 * size_factor), 16)  # Minimum of 16 dimensions
        
        # Base channel dimensions
        self.base_channels = max(int(128 * size_factor), 16)  # Minimum of 16 base channels
        
        # Channel multipliers (same for all models)
        self.channel_multipliers = [1, 2, 2, 2]
        
        # Calculate channel dimensions based on multipliers and size factor
        self.dims = [max(16, int(self.base_channels * mult)) for mult in self.channel_multipliers]
        
        # Print model information
        print(f"Model size factor: {size_factor}")
        print(f"Model dimensions: {self.dims}")
        
        # Dropout (same for all models)
        self.dropout = nn.Dropout(config.dropout)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.ReLU()
        )
        
        # Condition embedding (for classifier-free guidance)
        self.cond_emb = nn.Sequential(
            nn.Linear(1, self.time_emb_dim),
            nn.ReLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )
        
        # Downsampling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Encoder (4 levels for all models)
        self.enc1 = Block(self.channels, self.dims[0], self.time_emb_dim)
        self.enc2 = Block(self.dims[0], self.dims[1], self.time_emb_dim)
        self.enc3 = Block(self.dims[1], self.dims[2], self.time_emb_dim)
        self.enc4 = Block(self.dims[2], self.dims[3], self.time_emb_dim)
        
        # Bottleneck
        self.bottleneck = Block(self.dims[3], self.dims[3], self.time_emb_dim)
        
        # Decoder (with proper input channel dimensions for concatenation)
        # After upsampling bottleneck and concatenating with enc4
        self.dec3 = Block(self.dims[3] + self.dims[3], self.dims[2], self.time_emb_dim)
        
        # After upsampling dec3 output and concatenating with enc3
        self.dec2 = Block(self.dims[2] + self.dims[2], self.dims[1], self.time_emb_dim)
        
        # After upsampling dec2 output and concatenating with enc2
        self.dec1 = Block(self.dims[1] + self.dims[1], self.dims[0], self.time_emb_dim)
        
        # Final layer
        self.final = nn.Conv2d(self.dims[0], self.channels, 1)
    
    def forward(self, x, t, cond=None):
        """
        Forward pass through the U-Net
        
        Resolution flow for 32x32 input:
        32x32 → 16x16 → 8x8 → 4x4 → 2x2 → 4x4 → 8x8 → 16x16 → 32x32
        
        All models (teacher and students) follow the same resolution flow,
        ensuring consistent output dimensions.
        
        Args:
            x: Input tensor [B, C, H, W]
            t: Timestep tensor [B]
            cond: Conditioning tensor [B, 1] for classifier-free guidance (optional)
        """
        # Time embedding
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        # Ensure t has the right shape (batch_size, 1)
        if t.dim() > 2:
            t = t.view(t.size(0), -1)[:, 0:1]
        time_emb = self.time_mlp(t)
        
        # Condition embedding (if provided)
        if cond is not None:
            cond_emb = self.cond_emb(cond)
            # Add condition embedding to time embedding
            time_emb = time_emb + cond_emb
        
        # Encoder
        x1 = self.enc1(x, time_emb)
        x1 = self.dropout(x1)
        
        x2 = self.enc2(self.pool(x1), time_emb)
        x2 = self.dropout(x2)
        
        x3 = self.enc3(self.pool(x2), time_emb)
        x3 = self.dropout(x3)
        
        x4 = self.enc4(self.pool(x3), time_emb)
        x4 = self.dropout(x4)
        
        # Bottleneck
        x = self.bottleneck(self.pool(x4), time_emb)
        x = self.dropout(x)
        
        # Decoder with skip connections
        x = self.upsample(x)
        x = torch.cat([x, x4], dim=1)  # Concatenate bottleneck output with enc4 features
        x = self.dec3(x, time_emb)
        x = self.dropout(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x3], dim=1)  # Concatenate dec3 output with enc3 features
        x = self.dec2(x, time_emb)
        x = self.dropout(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)  # Concatenate dec2 output with enc2 features
        x = self.dec1(x, time_emb)
        x = self.dropout(x)
        
        # Final upsampling to match input resolution
        x = self.upsample(x)
        
        # Final layer
        return self.final(x)

# For backward compatibility
class SimpleUNet(DiffusionUNet):
    """
    Alias for DiffusionUNet with size_factor=1.0 (teacher model)
    """
    def __init__(self, config):
        super().__init__(config, size_factor=1.0)

class StudentUNet(DiffusionUNet):
    """
    Alias for DiffusionUNet with configurable size_factor (student model)
    """
    def __init__(self, config, size_factor=1.0, architecture_type=None):
        # architecture_type is ignored, as we now use a unified architecture
        if architecture_type is not None:
            print(f"Warning: architecture_type '{architecture_type}' is ignored in the new unified model architecture")
        super().__init__(config, size_factor=size_factor)

# This is a new line comment
