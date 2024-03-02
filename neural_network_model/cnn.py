from torch import nn
import torch


class CNNNetwork(nn.Module):
    def __init__(self, metadata_dim):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 19 * 1 + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x, metadata):
        x = self.conv_layers(x)
        x = self.flatten(x)
        metadata_features = self.metadata_fc(metadata)
        combined_features = torch.cat((x, metadata_features), dim=1)
        x = self.fc_layers(combined_features)
        return x
