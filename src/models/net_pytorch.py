import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Vgg(nn.Module):
    def __init__(self):
        super(Vgg, self).__init__()
        self.conv1 = nn.Sequential(
            conv2d_bn_relu(1, 32, 5),
            nn.MaxPool2d(3, 2)
        )
        self.conv2 = nn.Sequential(
            conv2d_bn_relu(32, 64, 3),
            conv2d_bn_relu(64, 64, 3),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            conv2d_bn_relu(64, 128, 3),
            conv2d_bn_relu(128, 128, 3),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        out = F.sigmoid(x)

        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            conv2d_bn_relu(1, 32, 5),
            nn.MaxPool2d(3, 2)
        )
        self.conv2 = Res_block(32, 64)
        self.conv2 = Res_block(64, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.block1 = Incep_res_block(1, 32, 5)
        self.block2 = Incep_res_block(32, 64, 3)
        self.block3 = Incep_res_block(64, 128, 3)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x


def conv2d_bn_relu(in_channels, out_channels, kernel_size,
                   stride=1, padding=0, dilation=1, groups=1, bias=True,
                   norm2d_fn=nn.BatchNorm2d, actication_fn=nn.ReLU()):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  stride, padding, dilation, groups, bias),
        norm2d_fn(out_channels),
        actication_fn
    )


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_block, self).__init__()
        self.conv1 = conv2d_bn_relu(in_channels, out_channels, 3, padding=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        res = self.conv3(x)
        res = self.bn3(x)

        x = x + res

        return x


class Incep_res_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Incep_res_block, self).__init__()
        self.conv_a1 = nn.Conv2d(in_channels, 128, 1)
        self.conv_a2 = nn.Conv2d(128, 160, (1, kernel_size))
        self.conv_a3 = nn.Conv2d(160, 192, (kernel_size, 1))

        self.conv_b1 = nn.Conv2d(in_channels, 192, 1)

        self.conv_c1 = nn.Conv2d(2*192, out_channels, 1)

        self.conv_d1 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x_a = self.conv_a1(x)
        x_a = self.conv_a2(x_a)
        x_a = self.conv_a3(x_a)

        x_b = self.conv_b1(x)

        x_c = torch.cat((x_a, x_b), 3)
        x_c = self.conv_c1(x_c)

        x = self.conv_d1(x)

        out = F.relu(x + x_c)

        return out


def pred_to_01(pred):
    pred_class = (np.sign(pred.detach().clone().cpu()-0.5)+1)/2
    pred_class[pred_class == 0.5] = 1
    return pred_class
