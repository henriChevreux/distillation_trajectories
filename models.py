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
        self.base_channels = 128  # Base channels set to 128
        self.channel_multipliers = [1, 2, 2, 2]  # Channel multipliers as per specifications
        self.dropout = nn.Dropout(config.dropout)  # Dropout rate of 0.3
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.ReLU()
        )
        
        # Calculate channel dimensions based on multipliers
        self.channels_list = [self.base_channels * mult for mult in self.channel_multipliers]
        
        # Encoder
        self.enc1 = Block(config.channels, self.channels_list[0], self.time_emb_dim)
        self.enc2 = Block(self.channels_list[0], self.channels_list[1], self.time_emb_dim)
        self.enc3 = Block(self.channels_list[1], self.channels_list[2], self.time_emb_dim)
        
        # Bottleneck
        self.bottleneck = Block(self.channels_list[2], self.channels_list[3], self.time_emb_dim)
        
        # Decoder - Fixed channel dimensions for concatenated inputs
        self.dec3 = Block(self.channels_list[3] + self.channels_list[2], self.channels_list[2], self.time_emb_dim)
        self.dec2 = Block(self.channels_list[2] + self.channels_list[1], self.channels_list[1], self.time_emb_dim)
        self.dec1 = Block(self.channels_list[1] + self.channels_list[0], self.channels_list[0], self.time_emb_dim)
        
        # Final decoder block
        self.final_decoder = Block(self.channels_list[0] + config.channels, self.channels_list[0], self.time_emb_dim)
        
        # Downsampling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final layer
        self.final = nn.Conv2d(self.channels_list[0], config.channels, 1)
    
    def forward(self, x, t):
        # Time embedding
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        # Ensure t has the right shape (batch_size, 1)
        if t.dim() > 2:
            t = t.view(t.size(0), -1)[:, 0:1]
        time_emb = self.time_mlp(t)
        
        # Encoder
        x1 = self.enc1(x, time_emb)
        x1 = self.dropout(x1)  # Apply dropout after each block
        
        x2 = self.enc2(self.pool(x1), time_emb)
        x2 = self.dropout(x2)  # Apply dropout
        
        x3 = self.enc3(self.pool(x2), time_emb)
        x3 = self.dropout(x3)  # Apply dropout
        
        # Bottleneck
        x = self.bottleneck(self.pool(x3), time_emb)
        x = self.dropout(x)  # Apply dropout
        
        # Decoder with skip connections
        x = self.upsample(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x, time_emb)
        x = self.dropout(x)  # Apply dropout
        
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x, time_emb)
        x = self.dropout(x)  # Apply dropout
        
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x, time_emb)
        x = self.dropout(x)  # Apply dropout
        
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
        
        # Override the input architecture_type based on size_factor
        if size_factor < 0.1:
            self.architecture_type = 'tiny'
        elif size_factor < 0.3:
            self.architecture_type = 'small'
        elif size_factor < 0.7:
            self.architecture_type = 'medium'
        else:
            self.architecture_type = 'full'
        
        print(f"Using architecture type: {self.architecture_type} for size factor {size_factor}")
        
        self.dropout = nn.Dropout(config.dropout)  # Use same dropout as teacher
        
        # Set image size based on architecture type
        self.image_size = config.image_size  # Should be 32 for CIFAR10
            
        # Default student architectures - use round numbers for better alignment
        architectures = {
            'tiny': [32, 64],                   # 2 layers
            'small': [32, 64, 128],             # 3 layers but smaller dimensions
            'medium': [48, 96, 192],            # 3 layers with 75% of teacher dimensions
            'full': [128, 256, 256, 256]        # Same as teacher
        }
        
        # Get architecture dimensions based on type
        if hasattr(config, 'student_architectures') and self.architecture_type in config.student_architectures:
            base_dims = config.student_architectures[self.architecture_type]
        else:
            base_dims = architectures.get(self.architecture_type, architectures['full'])
            
        # Better size scaling for consistent dimensions
        # For small factors (0.1-0.3), use a more careful approach to dimension scaling
        if self.architecture_type == 'small' and size_factor < 0.3:
            # Use fixed minimum sizes for small models
            if size_factor <= 0.1:
                self.dims = [16, 32]  # Tiny model equivalent
            else:  # 0.1 < size_factor < 0.3
                # Ensure even multiples of 8 for better alignment
                self.dims = [16, 32, 64]  # Fixed dimensions for small models
            print(f"Small model with fixed dimensions: {self.dims}")
        elif self.architecture_type == 'medium':
            # Use fixed dimensions for medium models for consistency
            if size_factor == 0.3:
                self.dims = [16, 32, 64]  # Same as small 0.2 for smooth transition
            elif size_factor <= 0.4:
                self.dims = [24, 48, 96]
            elif size_factor <= 0.5:
                self.dims = [32, 64, 128]
            else:  # 0.5 < size_factor < 0.7
                self.dims = [40, 80, 160]
            print(f"Medium model with fixed dimensions: {self.dims}")
        else:
            # Apply size factor to dimensions and ensure minimum size
            self.dims = [max(16, int(d * size_factor)) for d in base_dims]
            # Round to multiples of 8 for better alignment (except for tiny models)
            if self.architecture_type != 'tiny':
                self.dims = [((d + 7) // 8) * 8 for d in self.dims]  # Round up to nearest multiple of 8
            print(f"Model dimensions after scaling: {self.dims}")
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.ReLU()
        )
        
        # Build differently based on architecture type
        if self.architecture_type == 'tiny' or (self.architecture_type == 'small' and size_factor <= 0.1):
            # Simpler model for tiny architecture (just 2 levels)
            # Encoder
            self.enc1 = Block(config.channels, self.dims[0], self.time_emb_dim)
            self.enc2 = Block(self.dims[0], self.dims[1], self.time_emb_dim)
            
            # Bottleneck (use the second dimension)
            self.bottleneck = Block(self.dims[1], self.dims[1], self.time_emb_dim)
            
            # Only one decoder for tiny model - make sure dimensions match
            self.dec1 = Block(self.dims[1] + self.dims[0], self.dims[0], self.time_emb_dim) 
            
            # Final layer
            self.final = nn.Conv2d(self.dims[0], config.channels, 1)
            
        elif self.architecture_type == 'small':
            # Special handling for small architecture (3 levels)
            # Encoder
            self.enc1 = Block(config.channels, self.dims[0], self.time_emb_dim)
            self.enc2 = Block(self.dims[0], self.dims[1], self.time_emb_dim)
            self.enc3 = Block(self.dims[1], self.dims[2], self.time_emb_dim)
            
            # Bottleneck - critically important to match the output dimensions
            if self.size_factor < 0.3:
                # Size factor 0.2 bottleneck stays at 64 channels
                self.bottleneck = Block(self.dims[2], self.dims[2], self.time_emb_dim)
                
                # First decoder concatenates bottleneck output + encoder level 3 output
                # For the 0.2 size factor this is 64 + 64 = 128 channels, outputting 32 channels
                self.dec2 = Block(self.dims[2] + self.dims[2], self.dims[1], self.time_emb_dim)
                
                # Second decoder concatenates first decoder output + encoder level 2 output
                # For the 0.2 size factor this is 32 + 32 = 64 channels, outputting 16 channels
                self.dec1 = Block(self.dims[1] + self.dims[1], self.dims[0], self.time_emb_dim)
                
                print(f"Small model decoder dimensions: dec2: {self.dims[2] + self.dims[2]} -> {self.dims[1]}, "
                      f"dec1: {self.dims[1] + self.dims[1]} -> {self.dims[0]}")
            else:
                # Normal handling for other small models
                self.bottleneck = Block(self.dims[2], self.dims[2], self.time_emb_dim)
                self.dec2 = Block(self.dims[2] + self.dims[1], self.dims[1], self.time_emb_dim)
                self.dec1 = Block(self.dims[1] + self.dims[0], self.dims[0], self.time_emb_dim)
            
            # Final layer
            self.final = nn.Conv2d(self.dims[0], config.channels, 1)
            
        elif self.architecture_type == 'medium':
            # Special handling for medium architecture (3 levels)
            # Encoder
            self.enc1 = Block(config.channels, self.dims[0], self.time_emb_dim)
            self.enc2 = Block(self.dims[0], self.dims[1], self.time_emb_dim)
            self.enc3 = Block(self.dims[1], self.dims[2], self.time_emb_dim)
            
            # Bottleneck
            self.bottleneck = Block(self.dims[2], self.dims[2], self.time_emb_dim)
            
            # Decoder - explicit definition to avoid dimension mismatch
            # When concatenating, we get bottleneck + encoder features
            self.dec2 = Block(self.dims[2] + self.dims[2], self.dims[1], self.time_emb_dim)
            
            # For the second decoder, we need to use dims[1] + dims[1] (not dims[1] + dims[0])
            # because the output of dec2 has dims[1] channels and the skip connection also has dims[1] channels
            self.dec1 = Block(self.dims[1] + self.dims[1], self.dims[0], self.time_emb_dim)
            
            print(f"Medium model decoder dimensions: dec2: {self.dims[2] + self.dims[2]} -> {self.dims[1]}, "
                  f"dec1: {self.dims[1] + self.dims[1]} -> {self.dims[0]}")
            
            # Final layer
            self.final = nn.Conv2d(self.dims[0], config.channels, 1)
            
        else:
            # Standard model with dynamic number of levels
            # Encoder blocks
            self.encoder_blocks = nn.ModuleList()
            in_channels = config.channels
            for dim in self.dims:
                self.encoder_blocks.append(Block(in_channels, dim, self.time_emb_dim))
                in_channels = dim
            
            # Bottleneck
            self.bottleneck = Block(self.dims[-1], self.dims[-1], self.time_emb_dim)
            
            # Decoder blocks
            self.decoder_blocks = nn.ModuleList()
            for i in range(len(self.dims)-1, 0, -1):
                in_channels = self.dims[i] + self.dims[i]
                out_channels = self.dims[i-1]
                self.decoder_blocks.append(Block(in_channels, out_channels, self.time_emb_dim))
            
            # Final layer
            self.final = nn.Conv2d(self.dims[0], config.channels, 1)
        
        # Downsampling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x, t):
        # Time embedding
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        # Ensure t has the right shape (batch_size, 1)
        if t.dim() > 2:
            t = t.view(t.size(0), -1)[:, 0:1]
        time_emb = self.time_mlp(t)
        
        # Handle differently based on architecture type
        if self.architecture_type == 'tiny' or (self.architecture_type == 'small' and self.size_factor <= 0.1):
            # Simple 2-level architecture
            # Encoder
            x1 = self.enc1(x, time_emb)
            x1 = self.dropout(x1)
            
            x2 = self.enc2(self.pool(x1), time_emb)
            x2 = self.dropout(x2)
            
            # Bottleneck
            x = self.bottleneck(x2, time_emb)
            x = self.dropout(x)
            
            # Decoder - single step
            x = self.upsample(x)
            # Ensure x and x1 have the same spatial dimensions before concatenation
            if x.shape[2:] != x1.shape[2:]:
                x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, x1], dim=1)
            x = self.dec1(x, time_emb)
            x = self.dropout(x)
            
            # Final layer
            return self.final(x)
            
        elif self.architecture_type == 'small':
            # Special handling for small architecture (3 levels)
            # Encoder
            x1 = self.enc1(x, time_emb)
            x1 = self.dropout(x1)
            
            x2 = self.enc2(self.pool(x1), time_emb)
            x2 = self.dropout(x2)
            
            x3 = self.enc3(self.pool(x2), time_emb)
            x3 = self.dropout(x3)
            
            # Bottleneck
            x = self.bottleneck(self.pool(x3), time_emb)
            x = self.dropout(x)
            
            # Carefully handle dimensions for decoder path
            # Debug dimensions at each step
            print(f"Bottleneck output shape: {x.shape}")
            print(f"Skip connection x3 shape: {x3.shape}")
            
            # Decoder
            x = self.upsample(x)
            # Ensure spatial dimensions match
            if x.shape[2:] != x3.shape[2:]:
                x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=True)
                
            print(f"After upsampling shape: {x.shape}")
            
            # For debugging, print info about the upcoming concatenation
            print(f"About to concatenate tensors with shapes {x.shape} and {x3.shape}")
            
            x = torch.cat([x, x3], dim=1)
            print(f"After concatenation shape: {x.shape}")
            print(f"Dec2 expected input channels: {self.dec2.conv1.in_channels}")
            
            x = self.dec2(x, time_emb)
            print(f"After dec2 shape: {x.shape}")
            x = self.dropout(x)
            
            x = self.upsample(x)
            # Ensure spatial dimensions match
            if x.shape[2:] != x2.shape[2:]:
                x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=True)
                
            print(f"Before second concat: x shape {x.shape}, x2 shape {x2.shape}")
            x = torch.cat([x, x2], dim=1)
            x = self.dec1(x, time_emb)
            x = self.dropout(x)
            
            # Final layer
            return self.final(x)
            
        elif self.architecture_type == 'medium':
            # Special handling for medium architecture (3 levels)
            # Encoder
            x1 = self.enc1(x, time_emb)
            x1 = self.dropout(x1)
            
            x2 = self.enc2(self.pool(x1), time_emb)
            x2 = self.dropout(x2)
            
            x3 = self.enc3(self.pool(x2), time_emb)
            x3 = self.dropout(x3)
            
            # Bottleneck
            x = self.bottleneck(self.pool(x3), time_emb)
            x = self.dropout(x)
            
            # Debug dimensions
            print(f"Medium model - bottleneck output shape: {x.shape}")
            print(f"Medium model - skip connection x3 shape: {x3.shape}")
            
            # Decoder
            x = self.upsample(x)
            # Ensure spatial dimensions match
            if x.shape[2:] != x3.shape[2:]:
                x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=True)
                
            print(f"Medium model - before concat1: {x.shape} and {x3.shape}")
            x = torch.cat([x, x3], dim=1)
            print(f"Medium model - after concat1: {x.shape}, dec2 expects: {self.dec2.conv1.in_channels}")
            
            x = self.dec2(x, time_emb)
            x = self.dropout(x)
            
            x = self.upsample(x)
            # Ensure spatial dimensions match
            if x.shape[2:] != x2.shape[2:]:
                x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=True)
                
            print(f"Medium model - before concat2: {x.shape} and {x2.shape}")
            x = torch.cat([x, x2], dim=1)
            x = self.dec1(x, time_emb)
            x = self.dropout(x)
            
            # Final layer
            return self.final(x)
            
        else:
            # Standard multi-level architecture
            # Store skip connections
            skip_connections = []
            
            # Encoder pathway
            x_current = x
            for block in self.encoder_blocks:
                x_current = block(x_current, time_emb)
                x_current = self.dropout(x_current)
                skip_connections.append(x_current)
                x_current = self.pool(x_current)
            
            # Bottleneck
            x_current = self.bottleneck(x_current, time_emb)
            x_current = self.dropout(x_current)
            
            # Debug prints
            print(f"Full architecture - bottleneck output shape: {x_current.shape}")
            
            # Decoder pathway with skip connections
            for i, block in enumerate(self.decoder_blocks):
                x_current = self.upsample(x_current)
                skip_idx = len(skip_connections) - i - 1
                skip_connection = skip_connections[skip_idx]
                
                # Debug prints
                print(f"Level {i+1} - Upsampled shape: {x_current.shape}, Skip connection shape: {skip_connection.shape}")
                
                # Ensure dimensions match before concatenation
                if x_current.shape[2:] != skip_connection.shape[2:]:
                    x_current = F.interpolate(x_current, size=skip_connection.shape[2:], 
                                              mode='bilinear', align_corners=True)
                    
                print(f"Level {i+1} - Before concat: {x_current.shape} + {skip_connection.shape}")
                x_current = torch.cat([x_current, skip_connection], dim=1)
                print(f"Level {i+1} - After concat: {x_current.shape}, Block expects: {block.conv1.in_channels}")
                
                x_current = block(x_current, time_emb)
                x_current = self.dropout(x_current)
            
            # Final layer
            return self.final(x_current)

# This is a new line comment
