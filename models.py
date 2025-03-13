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
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # Pad or trim to match the original requested dimension
        if embeddings.shape[-1] != self.dim:
            if embeddings.shape[-1] < self.dim:
                # Pad with zeros if needed - ensure padding has same number of dimensions
                padding = torch.zeros(embeddings.shape[0], self.dim - embeddings.shape[-1], device=device)
                # Make sure padding has the same number of dimensions as embeddings
                if embeddings.dim() > 2:
                    # Reshape padding to match embeddings dimensions
                    padding_shape = list(embeddings.shape)
                    padding_shape[-1] = self.dim - embeddings.shape[-1]
                    padding = torch.zeros(padding_shape, device=device)
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
            # Handle both 2D and 3D time embeddings
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

class SimpleUNet(nn.Module):
    """
    A simplified U-Net model for diffusion
    """
    def __init__(self, config):
        super().__init__()
        
        # Model dimensions
        self.channels = config.channels
        self.time_emb_dim = 256
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.ReLU()
        )
        
        # Encoder
        self.enc1 = Block(config.channels, 64, self.time_emb_dim)
        self.enc2 = Block(64, 128, self.time_emb_dim)
        self.enc3 = Block(128, 256, self.time_emb_dim)
        
        # Bottleneck
        self.bottleneck = Block(256, 256, self.time_emb_dim)
        
        # Decoder
        self.dec3 = Block(512, 128, self.time_emb_dim)
        self.dec2 = Block(256, 64, self.time_emb_dim)
        self.dec1 = Block(128, 64, self.time_emb_dim)
        
        # Downsampling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final layer
        self.final = nn.Conv2d(64, config.channels, 1)
    
    def forward(self, x, t):
        # Time embedding
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        # Ensure t has the right shape (batch_size, 1)
        if t.dim() > 2:
            t = t.view(t.size(0), -1)[:, 0:1]
        time_emb = self.time_mlp(t)
        
        # Encoder
        x1 = self.enc1(x, time_emb)
        x2 = self.enc2(self.pool(x1), time_emb)
        x3 = self.enc3(self.pool(x2), time_emb)
        
        # Bottleneck
        x = self.bottleneck(self.pool(x3), time_emb)
        
        # Decoder with skip connections
        x = self.upsample(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x, time_emb)
        
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x, time_emb)
        
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x, time_emb)
        
        # Final layer
        return self.final(x)

class StudentUNet(nn.Module):
    """
    A student U-Net model for diffusion with configurable size
    """
    def __init__(self, config, size_factor=1.0, architecture_type='full'):
        super().__init__()
        
        # Model dimensions
        self.channels = config.channels
        # Ensure minimum time embedding dimension
        self.time_emb_dim = max(int(256 * size_factor), 16)  # Minimum of 16 dimensions
        self.size_factor = size_factor
        self.architecture_type = architecture_type
        
        # Set image size based on architecture type
        if architecture_type == 'tiny':
            # Tiny models use smaller image size
            self.image_size = 8
        else:
            # Other models use the standard image size
            self.image_size = config.image_size
        
        # Get architecture dimensions based on type
        if hasattr(config, 'student_architectures') and architecture_type in config.student_architectures:
            dims = config.student_architectures[architecture_type]
        else:
            # Default architectures if not specified in config
            architectures = {
                'tiny': [32, 64],           # 2 layers instead of 3
                'small': [32, 64, 128],     # 3 layers but smaller dimensions
                'medium': [48, 96, 192],    # 3 layers with 75% of teacher dimensions
                'full': [64, 128, 256]      # Same as teacher
            }
            dims = architectures.get(architecture_type, architectures['full'])
        
        # Apply size factor to dimensions
        dims = [int(d * size_factor) for d in dims]
        
        # Ensure minimum dimension size
        dims = [max(16, d) for d in dims]
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.ReLU()
        )
        
        # Encoder
        self.enc1 = Block(config.channels, dims[0], self.time_emb_dim)
        
        # Create encoder blocks based on architecture
        self.encoder_blocks = nn.ModuleList([self.enc1])
        for i in range(1, len(dims)):
            self.encoder_blocks.append(Block(dims[i-1], dims[i], self.time_emb_dim))
        
        # Bottleneck
        self.bottleneck = Block(dims[-1], dims[-1], self.time_emb_dim)
        
        # Decoder
        decoder_dims = []
        for i in range(len(dims)-1, 0, -1):
            decoder_dims.append(Block(dims[i] * 2, dims[i-1], self.time_emb_dim))
        
        self.decoder_blocks = nn.ModuleList(decoder_dims)
        
        # Downsampling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final layer
        self.final = nn.Conv2d(dims[0], config.channels, 1)
    
    def forward(self, x, t):
        # Time embedding
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        # Ensure t has the right shape (batch_size, 1)
        if t.dim() > 2:
            t = t.view(t.size(0), -1)[:, 0:1]
        time_emb = self.time_mlp(t)
        
        # Encoder
        skip_connections = []
        for block in self.encoder_blocks:
            x = block(x, time_emb)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x, time_emb)
        
        # Decoder with skip connections
        for i, block in enumerate(self.decoder_blocks):
            x = self.upsample(x)
            skip_idx = len(skip_connections) - i - 1
            x = torch.cat([x, skip_connections[skip_idx]], dim=1)
            x = block(x, time_emb)
        
        # Final layer
        return self.final(x) # Test comment

# This is a new line comment
