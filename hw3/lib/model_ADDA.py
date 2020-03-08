import torch
import torch.nn as nn


class D(nn.Module):
    def __init__(self, in_size=256*4*4):
        super(D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, 256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.net(x).view(-1)


class D_CNN(nn.Module):
    def __init__(self, in_ch=256):
        super(D_CNN, self).__init__()
        self.net = nn.Sequential(  # (b, in_ch, 4, 4)
            nn.Conv2d(in_ch, 128, kernel_size=4, stride=2, padding=1),  # (b, 128, 2, 2)
            nn.BatchNorm2d(128),
            nn.Dropout2d(),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),  # (b, 64, 1, 1)
            nn.BatchNorm2d(64),
            nn.Dropout2d(),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),  # (b, 1, 1, 1)
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1)


class E(nn.Module):
    def __init__(self, in_ch=3, out_ch=256):
        super(E, self).__init__()
        self.net = nn.Sequential(  # (b, 3, 28, 28)
            nn.Conv2d(in_ch, 64, kernel_size=4, stride=2, padding=1, bias=False),  # (b, 64, 14, 14)
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # (b, 128, 7, 7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),  # (b, 128, 7, 7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(),
            nn.Conv2d(128, out_ch, kernel_size=4, stride=1, padding=0, bias=False),  # (b, out_ch, 4, 4)
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class C(nn.Module):
    def __init__(self, in_size=256*4*4, n_classes=10):
        super(C, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, 100),
            nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(100, n_classes),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.net(x)
