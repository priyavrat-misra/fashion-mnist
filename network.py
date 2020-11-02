import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8,
                      kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=7*7*16, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU()
        )
        self.out = nn.Linear(in_features=64, out_features=10)

    def forward(self, t):
        t = self.layer1(t)
        t = self.layer2(t)
        t = t.reshape(t.size(0), -1)
        t = self.fc1(t)
        t = self.fc2(t)
        t = self.out(t)

        return t
