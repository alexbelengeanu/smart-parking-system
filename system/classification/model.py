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

        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=9, stride=1)
        self.conv2d_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=9, stride=1)
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.do_1 = nn.Dropout(p=0.2)

        self.conv2d_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1)
        self.conv2d_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1)
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.do_2 = nn.Dropout(p=0.2)

        self.conv2d_5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1)
        self.conv2d_6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1)
        self.maxpool2d_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.do_3 = nn.Dropout(p=0.2)

        self.conv2d_7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.conv2d_8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.conv2d_9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.maxpool2d_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.do_4 = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(in_features=256 * 4 * 8, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.do_5 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=256, out_features=35)

    def forward(self, x) -> torch.Tensor:

        """The forward function takes in a 195x256 grayscale image and outputs a 35 dimensional vector of
        Args:
            x: 195x256 grayscale image

        Returns:
            35 dimensional vector of probabilities for each character"""

        x = F.relu(self.conv2d_1(x))
        x = F.relu(self.conv2d_2(x))
        x = self.maxpool2d_1(x)
        #x = self.do_1(x)

        x = F.relu(self.conv2d_3(x))
        x = F.relu(self.conv2d_4(x))
        x = self.maxpool2d_2(x)
        #x = self.do_2(x)

        x = F.relu(self.conv2d_5(x))
        x = F.relu(self.conv2d_6(x))
        x = self.maxpool2d_3(x)
        #x = self.do_3(x)

        x = F.relu(self.conv2d_7(x))
        x = F.relu(self.conv2d_8(x))
        x = F.relu(self.conv2d_9(x))
        x = self.maxpool2d_4(x)
        #x = self.do_4(x)

        x = x.view(-1, 256 * 4 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.do_5(x)
        x = F.softmax(self.fc3(x), dim=1)
        return x
