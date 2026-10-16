import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    Compact Convolutional Neural Network for EEG Motor Imagery Classification.
    Reference: Lawhern et al., 2018.
    """
    def __init__(self, channels: int = 64, samples: int = 480, num_classes: int = 2, F1: int = 8, D: int = 2, F2: int = 16, kernel_length: int = 64, dropout_rate: float = 0.25):
        super(EEGNet, self).__init__()

        # Block 1: Temporal & Depthwise Spatial Convolutions
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        
        # Depthwise Spatial Convolution (learns spatial filter weights per electrode)
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.pooling1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 2: Separable Convolution (Temporal + Pointwise)
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        )
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.pooling2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Calculate flattened features shape dynamically
        flattened_size = F2 * (samples // 32)
        
        # Classification Head
        self.classifier = nn.Linear(flattened_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input Shape: (Batch, 1, Channels, Time_Samples)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.pooling1(x)
        x = self.dropout1(x)

        x = self.separableConv(x)
        x = self.batchnorm3(x)
        x = self.elu(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        x = x.flatten(start_dim=1)
        out = self.classifier(x)
        return out


if __name__ == "__main__":
    # Test forward pass with dummy input matching Phase 2 tensor dimensions
    dummy_input = torch.randn(8, 1, 64, 480)
    model = EEGNet(channels=64, samples=480, num_classes=2)
    output = model(dummy_input)
    
    print("\n================ EEGNET MODEL INITIALIZED ================")
    print(f"Input Shape:        {dummy_input.shape}")
    print(f"Output Shape:       {output.shape} -> (Batch Size, Num Classes)")
    print(f"Total Parameters:   {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("==========================================================\n")