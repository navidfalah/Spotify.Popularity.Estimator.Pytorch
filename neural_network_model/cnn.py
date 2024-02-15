from torch import nn
from torchsummary import summary


class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output size: (16, 32, 2) assuming input size (1, 64, 4)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Removed one MaxPool2d to avoid reducing the width too much
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Added MaxPool2d with stride=1 to prevent width reduction
            nn.MaxPool2d(kernel_size=2, stride=1),  # Adjusted pooling to manage small width
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # No pooling here to preserve spatial dimensions
        )
        self.flatten = nn.Flatten()
        # Adjusted linear layer to match the output of the convolutional layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 32 * 2, 512),  # Adjusted based on the expected output size
            nn.ReLU(),
            nn.Linear(512, 1)  # Output a single value for regression
        )

    def forward(self, x):
        print(f"Input size: {x.size()}")
        x = self.conv_layers(x)
        print(f"Input size: {x.size()}")
        x = self.flatten(x)
        print(f"Input size: {x.size()}")
        x = self.fc_layers(x)
        print(f"Input size: {x.size()}")
        return x
if __name__ == "__main__":
    cnn = CNNNetwork()
    cnn.cuda()  # Assuming you are using a CUDA-enabled GPU
    # Calculate the size of the flattened output after the conv and pooling layers
    # This is necessary to define the first Linear layer properly
    # You might need to adjust the calculation based on your actual input size and architecture
    summary(cnn, (1, 64, 44))
