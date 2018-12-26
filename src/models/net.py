import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, Dense

""" 
-------------- PyTorch model --------------
"""


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
        self.resConv1 = ResidualBlock(2*dim, 2*dim)
        self.resConv2 = ResidualBlock(2*dim, 2*dim)
        self.final = nn.Linear(225792, 1)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 3, 1)
        x = F.max_pool2d(self.conv2(x), 3, 1)
        x = self.resConv1(x)
        x = self.resConv2(x)

	# flatten
        x = x.view(x.size(0), -1)
        print(x.size())
        outs = F.sigmoid(self.final(x))

        return outs

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class DSCNN(nn.Module):
    def __init__(self, in_chan, dim):
        super(DSCNN, self).__init__()

        self.dsc1 = nn.Sequential(SeparableConv2d(in_chan, 2 * dim, 3), nn.BatchNorm2d(2 * dim), nn.ReLU())
        self.dsc2 = nn.Sequential(SeparableConv2d(2 * dim, 2 * dim, 3, padding=1), nn.BatchNorm2d(2 * dim), nn.ReLU())
        self.dsc3 = nn.Sequential(SeparableConv2d(2 * dim, 4 * dim, 3), nn.BatchNorm2d(4 * dim), nn.ReLU())

        self.fc1 = nn.Linear(541696, 1)

    def forward(self, x):
        x = self.dsc1(x)
        res = x
        x = self.dsc2(x)
        x = res + x
        x = self.dsc3(x)

        x = x.view(x.size(0), -1)
        outs = F.sigmoid(self.fc1(x))

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


""" 
-------------- Keras model --------------
"""


def simple_cnn_keras(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', 
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model
