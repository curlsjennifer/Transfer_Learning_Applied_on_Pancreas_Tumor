import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Vgg', 'ResNet', 'Inception', 'SENet']

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
            nn.Linear(512, 1024),
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
        self.conv3 = Res_block(64, 128)

        self.avg_pool = nn.AvgPool2d(2, 2)

        self.fc1 = nn.Linear(128*121, 1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.sigmoid(x)

        return x


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.conv1 = conv2d_bn_relu(1, 32, 7, 2)
        self.block1 = Incep_res_block(32, 64, 5)
        self.block2 = Incep_res_block(64, 128, 3)

        self.avg_pool = nn.AvgPool2d(2, 2)

        self.fc1 = nn.Linear(128*11*11, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)

        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.sigmoid(x)

        return x


class SENet(nn.Module):
    def __init__(self):
        super(SENet, self).__init__()
        self.conv1 = nn.Sequential(
            conv2d_bn_relu(1, 32, 5),
            nn.MaxPool2d(3, 2)
        )
        self.conv2 = SE_Res_block(32, 64)
        self.conv3 = SE_Res_block(64, 128)

        self.avg_pool = nn.AvgPool2d(2, 2)

        self.fc1 = nn.Linear(128*121, 1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.sigmoid(x)

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
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        res = self.conv3(res)
        res = self.bn3(res)
        x = x + res

        return x


class Incep_res_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Incep_res_block, self).__init__()
        self.conv_a1 = nn.Conv2d(in_channels, 128, 1)
        self.conv_a2 = nn.Conv2d(128, 160, (1, kernel_size), padding=(0, (kernel_size-1)//2))
        self.conv_a3 = nn.Conv2d(160, 192, (kernel_size, 1), padding=((kernel_size-1)//2, 0))

        self.conv_b1 = nn.Conv2d(in_channels, 192, 1)

        self.conv_c1 = nn.Conv2d(2*192, out_channels, 1)

        self.conv_d1 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x_a = self.conv_a1(x)
        x_a = self.conv_a2(x_a)
        x_a = self.conv_a3(x_a)

        x_b = self.conv_b1(x)

        x_c = torch.cat((x_a, x_b), 1)
        x_c = self.conv_c1(x_c)

        x = self.conv_d1(x)

        out = F.relu(x + x_c)

        return out


class SE_Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, scale_ratio=8):
        super(SE_Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.exc_fc1 = nn.Linear(out_channels, out_channels//scale_ratio, bias=False)
        self.exc_fc2 = nn.Linear(out_channels//scale_ratio, out_channels, bias=False)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # squeeze
        z = F.avg_pool2d(x, x.size()[2:])
        z = z.view(z.size(0), -1)
        # excite
        z = self.exc_fc1(z)
        z = self.exc_fc2(z)
        z = F.sigmoid(z)
        z = z.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3))
        # rescale feature maps
        x = x * z

        res = self.conv3(res)
        res = self.bn3(res)

        x = x + res

        return x


def pred_to_01(pred):
    pred_class = (np.sign(pred.detach().clone().cpu()-0.5)+1)/2
    pred_class[pred_class == 0.5] = 1
    return pred_class
