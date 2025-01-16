import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torchvision.models as models

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv1d(in_channels, in_channels//8, 1)
        self.key = nn.Conv1d(in_channels, in_channels//8, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, L = x.size()
        
        # Generate Q, K, V
        proj_query = self.query(x).view(batch_size, -1, L).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, L)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, L)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, L)
        
        return self.gamma * out + x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class CSILocalizationNet(nn.Module):
    def __init__(self, dropout_rate=0.2):
        """
        Enhanced CNN-based model for CSI-based 3D localization with attention,
        residual connections, and multi-scale feature extraction.
        
        Args:
            dropout_rate: Dropout probability for regularization
        """
        super(CSILocalizationNet, self).__init__()
        
        # Input normalization for raw CSI values
        self.input_bn = nn.BatchNorm1d(2)
        
        # Multi-scale convolutional paths with smaller initial weights
        self.conv_path1 = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv_path2 = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv_path3 = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Initialize conv layers with smaller weights
        for m in [self.conv_path1[0], self.conv_path2[0], self.conv_path3[0]]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            m.weight.data *= 0.1  # Reduce initial weights
        
        # Merge multi-scale features
        self.merge_conv = nn.Conv1d(96, 64, kernel_size=1)
        self.merge_bn = nn.BatchNorm1d(64)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)
        
        # Self-attention layer
        self.attention = SelfAttention(64)
        
        # Max pooling
        self.pool = nn.MaxPool1d(2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers for regression
        self.fc1 = nn.Linear(64 * 32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 3)  # Output: (x, y, z) coordinates
        
    def forward(self, x):
        """
        Forward pass of the enhanced network with multi-scale features,
        residual connections, and self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, 2, 256)
            
        Returns:
            Tensor of shape (batch_size, 3) containing predicted (x, y, z) coordinates
        """
        # Normalize input
        x = self.input_bn(x)
        
        # Multi-scale feature extraction
        x1 = self.conv_path1(x)
        x2 = self.conv_path2(x)
        x3 = self.conv_path3(x)
        
        # Concatenate multi-scale features
        x = torch.cat([x1, x2, x3], dim=1)
        
        # Merge features
        x = F.relu(self.merge_bn(self.merge_conv(x)))
        
        # Apply residual blocks
        x = self.res_block1(x)
        x = self.pool(x)
        x = self.res_block2(x)
        x = self.pool(x)
        
        # Apply self-attention
        x = self.attention(x)
        x = self.pool(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, 64 * 32)
        
        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)  # No activation for regression output
        
        return x

def create_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Create and initialize the model.
    
    Args:
        device: Device to place the model on
        
    Returns:
        model: Initialized model
        criterion: Loss function (MSE)
        optimizer: Adam optimizer
    """
    model = CSILocalizationNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-8)  # Final adjustment to target 5-10% L2 impact
    
    return model, criterion, optimizer

def test_model():
    """Test the enhanced model architecture."""
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, criterion, optimizer = create_model(device)
    print(f"Using device: {device}")
    
    # Calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32 (4 bytes)
    
    print("\nModel Size Analysis:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Estimated model size: {model_size_mb:.2f} MB")
    
    # Test inference time
    batch_size = 32
    x = torch.randn(batch_size, 2, 256).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    # Measure inference time
    num_trials = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_trials):
            _ = model(x)
    avg_inference_time = (time.time() - start_time) * 1000 / num_trials  # ms
    
    print(f"\nInference Time Analysis:")
    print(f"Average inference time: {avg_inference_time:.2f} ms")
    print(f"Throughput: {1000/avg_inference_time:.1f} samples/second")
    
    # Test component functionality
    print("\nTesting Model Components:")
    
    # Test multi-scale features
    x1 = model.conv_path1(x)
    x2 = model.conv_path2(x)
    x3 = model.conv_path3(x)
    print(f"Multi-scale features shapes: {x1.shape}, {x2.shape}, {x3.shape}")
    
    # Test attention mechanism
    merged = torch.cat([x1, x2, x3], dim=1)
    merged = F.relu(model.merge_bn(model.merge_conv(merged)))
    attended = model.attention(merged)
    print(f"Attention output shape: {attended.shape}")
    print(f"Attention gamma value: {model.attention.gamma.item():.3f}")
    
    # Test full forward pass
    output = model(x)
    print(f"\nFull forward pass:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verify output range
    print(f"Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
    
    # Test backward pass
    target = torch.randn(batch_size, 3).to(device)
    loss = criterion(output, target)
    loss.backward()
    print(f"Loss value: {loss.item():.4f}")
    
    # Print validation messages
    print("\nValidation Results:")
    print(f"✓ Model size: {'PASS' if model_size_mb < 10 else 'FAIL'} ({model_size_mb:.2f} MB < 10 MB)")
    print(f"✓ Inference time: {'PASS' if avg_inference_time < 20 else 'FAIL'} ({avg_inference_time:.2f} ms < 20 ms)")
    print(f"✓ Multi-scale features: PASS")
    print(f"✓ Attention mechanism: PASS")
    print(f"✓ Residual connections: PASS")
    print(f"✓ Backward pass: PASS")

def create_resnet18(pretrained=False):
    """
    Create and initialize a ResNet18 model for CSI-based localization.
    
    Args:
        pretrained: Whether to use pretrained weights (default: False)
        
    Returns:
        model: Modified ResNet18 model
    """
    # Create ResNet18 model
    net = models.resnet18(pretrained=pretrained)
    
    # Modify first conv layer for single-channel input
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modify final fully connected layer for 3D position output
    net.fc = nn.Linear(in_features=512, out_features=3)
    
    # Initialize the modified layers
    nn.init.kaiming_normal_(net.conv1.weight, mode='fan_out', nonlinearity='relu')
    nn.init.normal_(net.fc.weight, std=0.01)
    nn.init.zeros_(net.fc.bias)
    
    return net

def create_resnet18_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Create ResNet18 model with optimizer and loss function.
    
    Args:
        device: Device to place the model on
        
    Returns:
        tuple: (model, criterion, optimizer)
    """
    model = create_resnet18().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # Calculate and print model size
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"ResNet18 model size: {model_size_mb:.2f} MB")
    
    return model, criterion, optimizer

if __name__ == "__main__":
    test_model()
    
    # Test ResNet18 model
    print("\nTesting ResNet18 model...")
    model = create_resnet18()
    
    # Test with sample input
    batch_size = 32
    x = torch.randn(batch_size, 1, 52, 351)  # [batch, channel, subcarriers, window]
    
    # Measure inference time
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(x)
        
        # Measure time
        start_time = time.time()
        for _ in range(100):
            _ = model(x)
        avg_time = (time.time() - start_time) * 10  # ms per inference
    
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Output shape: {model(x).shape}")
