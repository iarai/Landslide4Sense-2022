import torch.nn as nn


class Conv2dBn(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBn, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        return self.conv(x)


class Conv2dRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Conv2dBnRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBnRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


'''
Unlike BasicBlock in torchvision.models.resnet, it has no dilation parameter and cannot form Dilated ResNet
'''


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2dBnRelu(in_ch, out_ch, kernel_size=3,
                                  stride=stride, padding=dilation, dilation=dilation, bias=False)

        self.conv2 = Conv2dBn(in_ch, out_ch, kernel_size=3,
                              stride=1, padding=dilation, dilation=dilation, bias=False)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()

        self.conv1 = Conv2dBnRelu(in_ch, out_ch, kernel_size=1, bias=False)

        self.conv2 = Conv2dBnRelu(out_ch, out_ch, kernel_size=3, stride=stride,
                                  padding=dilation, dilation=dilation, bias=False)

        self.conv3 = Conv2dBn(out_ch, out_ch * 4, kernel_size=1, bias=False)

        self.downsample = downsample

        self.relu = nn.ReLU(inplace=True)

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
