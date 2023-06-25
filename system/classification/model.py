import torch
import torch.nn as nn
import torch.nn.functional as F


# Creating a CNN class
class CharacterClassifier(nn.Module):
    def __init__(self):

        """The character classifier is a convolutional neural network that takes in a 195x256 grayscale image
        and outputs a 35 dimensional. The 35 dimensions represent the 35 characters in the
        dataset.

        The network has 2 convolutional layers, 2 max pooling layers, and 2 fully connected layers.

        The first convolutional layer has 32 filters of size 3x3 with a stride of 1 and padding of 1.
        The second convolutional layer has 64 filters of size 3x3 with a stride of 1 and padding of 1.

        The first fully connected layer has 512 neurons.
        The second fully connected layer has 35 neurons.

        The output of the network is a 35 dimensional vector of probabilities for each character."""

        super(CharacterClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=64 * 64 * 48, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=35)

    def forward(self, x) -> torch.Tensor:

        """The forward function takes in a 195x256 grayscale image and outputs a 35 dimensional vector of
        Args:
            x: 195x256 grayscale image

        Returns:
            35 dimensional vector of probabilities for each character"""

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 64 * 48)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
