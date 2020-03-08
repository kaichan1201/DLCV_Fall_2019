import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.hidden = nn.Sequential(
            nn.Linear(self.nz, 512*4*4),
            nn.LeakyReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),  # (b, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),  # (b, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),  # (b, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),  # (b, 3, 64, 64)
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.hidden(x.view(x.shape[0], -1))
        x = x.view(x.shape[0], 512, 4, 4)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(  # (b, 3, 64, 64)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),  # (b, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # (b, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),  # (b, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),  # (b, 512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.hidden = nn.Linear(512*4*4, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.net(x).view(-1)
        # x = self.net(x)
        # x = x.view(x.shape[0], -1)
        # x = self.sig(self.hidden(x))
        # return x.view(-1)


class AC_Generator(nn.Module):
    def __init__(self, nz):
        super(AC_Generator, self).__init__()
        self.nz = nz
        self.hidden = nn.Sequential(
            nn.Linear(self.nz, 512 * 4 * 4),
            nn.LeakyReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),  # (b, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),  # (b, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),  # (b, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),  # (b, 3, 64, 64)
            nn.Tanh(),
        )

    def forward(self, x, c):
        c = c.view(-1, 1).float()
        x_c = torch.cat((x.view(x.shape[0], -1), c), dim=1)
        x_c = self.hidden(x_c).view(-1, 512, 4, 4)
        return self.net(x_c)


class AC_Discriminator(nn.Module):
    def __init__(self):
        super(AC_Discriminator, self).__init__()
        self.extractor = nn.Sequential(  # (b, 3, 64, 64)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),  # (b, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # (b, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),  # (b, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),  # (b, 512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Conv2d(512, 2, kernel_size=4, bias=False)  # (b, 2, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.extractor(x)
        adv_out = self.fc(x.view(x.shape[0], -1)).view(-1)
        # adv_out = self.net(x).view(-1)
        cls_out = self.classifier(x).view(x.shape[0], -1)
        return adv_out, cls_out


'''
DANN
'''


class ReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANN(nn.Module):
    def __init__(self, n_class=10):
        super(DANN, self).__init__()
        self.n_class = n_class
        self.extractor = nn.Sequential(  # (b, 3, 28, 28)
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1, bias=False),  # (b, 128, 14, 14)
            nn.BatchNorm2d(128),
            nn.Dropout2d(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),  # (b, 64, 7, 7)
            nn.BatchNorm2d(64),
            nn.Dropout2d(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, kernel_size=5),  # (b, 32, 3, 3)
            nn.BatchNorm2d(32),
            nn.Dropout2d(),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.lbl_classifier = nn.Sequential(
            nn.Linear(32*3*3, 100),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(100, self.n_class),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(32*3*3, 100),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, alpha):
        x = self.extractor(x)
        rev_x = ReverseLayer.apply(x, alpha)
        cls_out = self.lbl_classifier(x.view(x.shape[0], -1))
        domain_out = self.domain_classifier(rev_x.reshape(rev_x.shape[0], -1))
        return cls_out, domain_out.view(-1)


class DANN_NoTransfer(DANN):
    def __init__(self, n_class=10):
        super(DANN_NoTransfer, self).__init__(n_class=n_class)

    def forward(self, x):
        x = self.extractor(x)
        return self.lbl_classifier(x.view(x.shape[0], -1))
