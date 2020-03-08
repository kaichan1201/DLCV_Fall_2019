import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class E_Resnet50(nn.Module):
    def __init__(self, pretrained=True):
        super(E_Resnet50, self).__init__()
        self.resnet = resnet50(pretrained=pretrained)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        return x.view(x.shape[0], x.shape[1])


class C(nn.Module):
    def __init__(self, in_size=8192, n_classes=11):
        super(C, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class RNN(nn.Module):
    def __init__(self, in_size=2048, n_classes=11):
        super(RNN, self).__init__()
        # self.lstm = nn.LSTM(
        #     input_size=in_size,
        #     hidden_size=256,
        #     num_layers=2,
        #     batch_first=True,
        #     dropout=0.3
        # )
        # self.classifier = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.BatchNorm1d(128),
        #     # nn.Dropout(),
        #     nn.ReLU(True),
        #     nn.Linear(128, 64),
        #     nn.BatchNorm1d(64),
        #     nn.Dropout(0.3),
        #     nn.ReLU(True),
        #     nn.Linear(64, n_classes),
        # )

        self.gru = nn.GRU(
            input_size=in_size,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x):
        assert isinstance(x, list)  # {(T_i, 2048), 0 < i < B}
        lengths = [t.shape[0] for t in x]
        x = pad_sequence(x, batch_first=True)

        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # r_out, (_, _) = self.lstm(x)
        r_out, _ = self.gru(x, None)
        r_out, lengths = pad_packed_sequence(r_out, batch_first=True)

        iters = torch.arange(r_out.shape[0])
        out = self.classifier(r_out[iters, lengths-1, :])
        return out


class RNN_Seq(nn.Module):
    def __init__(self, in_size=2048, n_classes=11):
        super(RNN_Seq, self).__init__()
        self.gru = nn.GRU(
            input_size=in_size,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.7
        )
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x):  # x: (T, feat_size)
        x = x.view(1, *x.shape)
        r_out, _ = self.gru(x, None)
        outs = []
        for i in range(x.shape[1]):  # for every time step
            outs.append(self.classifier(r_out[:, i, :]))
        return torch.cat(outs, dim=0)
