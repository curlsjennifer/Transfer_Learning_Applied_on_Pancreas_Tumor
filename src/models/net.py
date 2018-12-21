import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class res_2dcnn(nn.Module):
    def __init__(self, in_chan, dim):
        super(res_2dcnn, self).__init__()
        self.conv1 = conv2d_norm_act(in_chan, dim)
        self.conv2 = conv2d_norm_act(dim, 2*dim)
        self.resConv1 = ResidualBlock(2*dim, 4*dim)
        self.resConv2 = ResidualBlock(4*dim, 8*dim)
        self.final = nn.Linear(225792, 1)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 3, 1)
        x = F.max_pool2d(self.conv2(x), 3, 1)
        # x = self.resConv1(x)
        # x = self.resConv2(x)

        x = x.view(x.size(0), -1)
        outs = F.sigmoid(self.final(x))

        return outs


def conv2d_norm_act(in_dim, out_dim, kernel_size=3, stride=1, padding=0, bias=True,
                    norm=nn.BatchNorm2d, act=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=True),
        norm(out_dim),
        act())


def pred_to_01(pred):
    pred_class = (np.sign(pred.detach().clone().cpu()-0.5)+1)/2
    pred_class[pred_class == 0.5] = 1
    return pred_class


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)
