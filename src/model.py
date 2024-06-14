import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DTo2D(nn.Module):
    def __init__(self):
        super(Conv3DTo2D, self).__init__()
        # Define the 3D convolution layers
        self.pool = nn.MaxPool3d((2, 2, 2))

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=3, padding=1)

        # Define the 2D convolution layers for mapping to 2D latent representation
        self.conv4 = nn.Conv2d(
            in_channels=48, out_channels=32, kernel_size=3, padding=1
        )

        # Define the final convolutional layer for resizing to (256, 256)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        # Input shape: (batch_size, channels=1, depth=96, height=512, width=512)
        # Apply 3D convolutions
        x = self.pool(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Reshape for 2D convolutions
        # x = x.view(-1, 64, 96, 512, 512)  # Reshape to (batch_size, channels, depth, height, width)
        # x = torch.mean(x, dim=2)  # Take the mean along the depth dimension
        x = x.permute(0, 4, 2, 3, 1).squeeze(4)
        # Apply 2D convolutions
        x = F.relu(self.conv4(x))

        # Apply final convolutional layer for resizing
        x = self.conv5(x)

        # Output shape: (batch_size, 1, 256, 256)
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Forward pass through the network
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class TACEnet(nn.Module):
    def __init__(self):
        super(TACEnet, self).__init__()
        self.lrm = Conv3DTo2D()
        self.net = ConvNet()

    def forward(self, VesselVolume, DRR):
        latent_representation = self.lrm(VesselVolume)
        x = torch.concatenate((DRR, latent_representation), dim=1)
        x = self.net(x)
        return x, latent_representation
