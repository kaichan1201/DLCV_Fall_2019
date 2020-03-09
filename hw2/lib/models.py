import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18


def get_model(args):
    if args.use_model == 'BASE':
        model = Net(args)
    elif args.use_model == 'ASPP':
        model = ASPP_NET(args)
    elif args.use_model == 'ASPP_DEC':
        model = ASPP_DEC_NET(args)
    elif args.use_model == 'ASPP_DEC_RES':
        model = ASPP_DEC_NET_RES(args)
    else:
        print('{} does not exist!'.format(args.use_model))
        raise NotImplementedError
    return model


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()  
        self.resnet = resnet18(pretrained=True)
        self.transpose_conv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.transpose_conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.transpose_conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.transpose_conv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.transpose_conv5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(16, 9, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, img):
        # extract features using resnet18
        x = self.resnet.conv1(img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # upsample
        x = self.transpose_conv1(x)
        x = self.transpose_conv2(x)
        x = self.transpose_conv3(x)
        x = self.transpose_conv4(x)
        x = self.transpose_conv5(x)
        x = self.conv1(x)
        return x


class _ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, r in enumerate(rates):
            self.add_module("conv_{}_rate{}".format(i, r),
                            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=r, dilation=r, bias=True)
                            )
        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = sum([stage(x) for stage in self.children()])
        return out


class ASPP_NET(Net):
    def __init__(self, args):
        super(ASPP_NET, self).__init__(args)
        self.aspp = _ASPP(in_ch=512, out_ch=9, rates=[6, 12, 18, 24])
        # self.aspp = _ASPP(in_ch=512, out_ch=9, rates=[1, 4, 8, 12])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img):
        # extracting features using resnet18
        x = self.resnet.conv1(img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = F.interpolate(self.aspp(x), scale_factor=32, mode='bilinear')

        return x


class ASPP_DEC_NET(Net):
    def __init__(self, args):
        super(ASPP_DEC_NET, self).__init__(args)
        self.aspp = _ASPP(in_ch=512, out_ch=64, rates=[1, 4, 8, 12])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img):
        # extracting features using resnet18
        x = self.resnet.conv1(img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.relu(self.aspp(x))

        x = self.transpose_conv4(x)
        x = self.transpose_conv5(x)
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=8, mode='bilinear')

        return x


class ASPP_DEC_NET_RES(ASPP_DEC_NET):
    def __init__(self, args):
        super(ASPP_DEC_NET_RES, self).__init__(args)
        self.transpose_conv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.transpose_conv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.transpose_conv3 = nn.Sequential(
            nn.ConvTranspose2d(16, 9, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.transpose_conv4 = nn.Sequential(
            nn.ConvTranspose2d(9, 9, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.transpose_conv5 = nn.Sequential(
            nn.ConvTranspose2d(9, 9, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.conv1 = nn.Conv2d(64, 9, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, img):
        # extracting features using resnet18
        x = self.resnet.conv1(img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # coarse feature
        x_coarse = self.aspp(x)
        x_coarse = F.interpolate(x_coarse, scale_factor=32, mode='bilinear')

        x = self.relu(self.aspp(x))

        x = self.transpose_conv1(x)
        x = self.transpose_conv2(x)
        x = self.transpose_conv3(x)
        x = self.transpose_conv4(x)
        x = self.transpose_conv5(x)

        x += self.conv1(x_coarse)

        return x
