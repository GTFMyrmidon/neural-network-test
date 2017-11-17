import torch
import torch.nn as nn
from torch.autograd import Variable

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()

        # Input (3, 32, 32)
        # Conv Output (16, 32, 32)
        # Output (16, 16, 16)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Input (16, 16, 16)
        # Conv Output (32, 16, 16)
        # Output (32, 8, 8)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Fully-connected layer,
        self.fc = nn.Linear(8*8*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
