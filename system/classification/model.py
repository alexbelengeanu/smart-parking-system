import torch
import torch.nn as nn
import torch.nn.functional as F


# Creating a CNN class
class CharacterClassifier(nn.Module):
    def __init__(self):
        super(CharacterClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=64 * 64 * 48, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=35)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 64 * 48)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
