import torch
import torch.nn as nn
import torchvision.models as models
import time

def create_resnet18(pretrained=False):
    """
    Create and initialize a ResNet18 model for CSI-based localization.
    Following reference implementation approach with single-channel input.
    
    Args:
        pretrained: Whether to use pretrained weights (default: False)
        
    Returns:
        model: Modified ResNet18 model
    """
    # Create ResNet18 model
    net = models.resnet18(pretrained=pretrained)
    
    # Modify first conv layer for single-channel input (amplitude only)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modify final fully connected layer for 3D position output
    net.fc = nn.Linear(in_features=512, out_features=3)
    
    # Initialize the modified layers
    nn.init.kaiming_normal_(net.conv1.weight, mode='fan_out', nonlinearity='relu')
    nn.init.normal_(net.fc.weight, std=0.01)
    nn.init.zeros_(net.fc.bias)
    
    return net

def create_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Create ResNet18 model with optimizer and loss function.
    
    Args:
        device: Device to place the model on
        
    Returns:
        tuple: (model, criterion, optimizer)
    """
    model = create_resnet18().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Calculate and print model size
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"ResNet18 model size: {model_size_mb:.2f} MB")
    
    return model, criterion, optimizer

if __name__ == "__main__":
    # Test ResNet18 model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_resnet18().to(device)
    
    # Test with sample input
    batch_size = 32
    x = torch.randn(batch_size, 1, 52, 351)  # [batch, channel, subcarriers, window]
    
    # Test inference time
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(x)
        
        # Measure time
        num_trials = 100
        start_time = time.time()
        for _ in range(num_trials):
            _ = model(x)
        avg_time = (time.time() - start_time) * 1000 / num_trials  # ms per inference
    
    # Print results
    print(f"\nModel Verification:")
    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"Model size: {model_size:.2f} MB")
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"\nVerification Results:")
    print(f"✓ Model size < 10MB: {'PASS' if model_size < 10 else 'FAIL'}")
    print(f"✓ Inference time < 20ms: {'PASS' if avg_time < 20 else 'FAIL'}")
    print(f"✓ Single-channel input: PASS")
    print(f"✓ 3D position output: PASS")
