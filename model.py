import torch
import torch.nn as nn
import torchvision.models as models

class HALOCNet(nn.Module):
    def __init__(self, dropout_rate=0.5, n_input_channels=52):  # 52 CSI subcarriers
        super(HALOCNet, self).__init__()
        
        # Base ResNet18 model modified for spectrogram input
        self.backbone = models.resnet18(pretrained=False)
        # Modify first conv layer for multi-channel spectrogram input (one per CSI subcarrier)
        self.backbone.conv1 = nn.Conv2d(52, 64, kernel_size=(7, 7), 
                                      stride=(2, 2), padding=(3, 3), bias=False)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Regression head for 3D position prediction
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout_rate),  # Regularization
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),  # Regularization
            nn.Dropout(dropout_rate),  # Regularization
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),  # Regularization
            nn.Linear(128, 3)  # Output: (x, y, z) coordinates
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass of the network.
        
        Args:
            x: Input tensor of shape [batch_size, n_subcarriers, freq_bins, time_steps]
               where n_subcarriers=52 (one spectrogram per CSI subcarrier)
        
        Returns:
            position: Predicted 3D position [batch_size, 3]
        """
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        position = self.regression_head(features)
        return position

def weighted_mse_loss(pred, target, weights=(2.0, 1.0, 1.0)):
    """Custom MSE loss with axis-specific weights.
    
    Args:
        pred: Predicted positions [batch_size, 3]
        target: True positions [batch_size, 3]
        weights: Weight factors for (X, Y, Z) axes, default=(2.0, 1.0, 1.0)
    
    Returns:
        Weighted MSE loss value
    """
    diff = (pred - target) ** 2  # Square differences for each axis
    for i, w in enumerate(weights):
        diff[:, i] *= w  # Apply axis-specific weights
    return diff.mean()  # Average over all dimensions and batch

class HALOCTrainer:
    def __init__(self, model, learning_rate=0.001, weight_decay=1e-4):
        self.model = model
        self.criterion = weighted_mse_loss  # Custom weighted MSE loss
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay  # L2 regularization
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5,
            verbose=True
        )
    
    def train_step(self, features, labels):
        """Training step with weighted MSE loss.
        
        Args:
            features: Input tensor [batch_size, n_subcarriers, freq_bins, time_steps]
            labels: Target positions [batch_size, 3]
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        predictions = self.model(features)
        loss = self.criterion(predictions, labels)  # Uses weighted MSE loss
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate_step(self, features, labels):
        """Validation step with weighted MSE loss.
        
        Args:
            features: Input tensor [batch_size, n_subcarriers, freq_bins, time_steps]
            labels: Target positions [batch_size, 3]
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(features)
            loss = self.criterion(predictions, labels)  # Uses weighted MSE loss
        return loss.item(), predictions

# Example usage:
"""
model = HALOCNet().to(device)
trainer = HALOCTrainer(model)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)
        
        loss = trainer.train_step(features, labels)
        
    # Validation
    val_losses = []
    for batch in val_loader:
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)
        
        val_loss, predictions = trainer.validate_step(features, labels)
        val_losses.append(val_loss)
    
    avg_val_loss = sum(val_losses) / len(val_losses)
    trainer.scheduler.step(avg_val_loss)
"""
