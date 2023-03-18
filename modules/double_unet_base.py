import torch
import torch.nn as nn

class Residual_Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return y


class Squeeze_Excite(nn.Module):
    def __init__(self, channel, reduction):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(channel // reduction, channel, bias=False),
                                    nn.Sigmoid()
                                    )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excite(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_enc=True, residual=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.SE = Squeeze_Excite(out_channels, 8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shortcut = Residual_Shortcut(in_channels, out_channels)

        self.is_enc = is_enc
        self.residual = residual

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.SE(y)

        if self.residual:
            y = y + self.shortcut(x)

        if self.is_enc:
            y = self.pool(y)

        return y


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)

        self.conv5 = nn.Conv2d(self.out_channels * 5, self.out_channels, kernel_size=1)

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]

        y_pool0 = nn.AdaptiveAvgPool2d(output_size=1)(x)
        y_conv0 = self.conv0(y_pool0)
        y_conv0 = self.bn(y_conv0)
        y_conv0 = self.relu(y_conv0)
        y_conv0 = nn.Upsample(size=(h, w), mode='bilinear')(y_conv0)

        y_conv1 = self.conv1(x)
        y_conv1 = self.bn(y_conv1)
        y_conv1 = self.relu(y_conv1)

        y_conv2 = self.conv2(x)
        y_conv2 = self.bn(y_conv2)
        y_conv2 = self.relu(y_conv2)

        y_conv3 = self.conv3(x)
        y_conv3 = self.bn(y_conv3)
        y_conv3 = self.relu(y_conv3)

        y_conv4 = self.conv4(x)
        y_conv4 = self.bn(y_conv4)
        y_conv4 = self.relu(y_conv4)

        y = torch.cat([y_conv0, y_conv1, y_conv2, y_conv3, y_conv4], 1)
        y = self.conv5(y)
        y = self.bn(y)
        y = self.relu(y)

        return y


def output_block():
    Layer = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1)), nn.Sigmoid())
    return Layer
  