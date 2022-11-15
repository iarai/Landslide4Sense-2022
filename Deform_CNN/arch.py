# -*- coding: utf-8 -*-
from model.deform_conv_v2 import *


class DeformCNN(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        # self.relu = nn.ReLU(inplace=True)
        # self.sigmoid = nn.Sigmoid()
        # self.pool = nn.MaxPool2d((1, 2))  # 2 pool layer (1,2) --> (1,4)
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        features = []
        inplanes = 14
        outplanes = 32

        # args.dcn = number of layer using deform conv
        for y in range(args.dcn):
            # Creating 2 conv layers
            features.append(nn.Conv2d(in_channels=inplanes, out_channels=outplanes,
                                      kernel_size=(1, 3), padding=(0, 1), bias=False))
            features.append(nn.ReLU(inplace=True))
            features.append(nn.MaxPool2d((1, 2)))
            features.append(nn.BatchNorm2d(outplanes))

            # args.cvn = number of layer using conv
            for i in range(args.cvn - 1):
                features.append(nn.Conv2d(in_channels=outplanes, out_channels=outplanes,
                                          kernel_size=(1, 3), padding=(0, 1), bias=False))
                features.append(nn.ReLU(inplace=True))
                features.append(nn.MaxPool2d((1, 2)))
                features.append(nn.BatchNorm2d(outplanes))

            # Adding deformable convolution
            features.append(DeformConv2d(inc=outplanes, outc=outplanes, kernel_size=3,
                                         padding=1, bias=False, modulation=args.modulation))
            features.append(nn.ReLU(inplace=True))
            features.append(nn.BatchNorm2d(outplanes))

            if y != args.dcn - 1:
                inplanes = outplanes
                outplanes *= 2

        self.features = nn.Sequential(*features)

        # self.fc1 = nn.Linear(in_features=outplanes, out_features=outplanes)
        self.fc0 = nn.Linear(in_features=outplanes, out_features=num_classes)

    def forward(self, input):
        x = self.features(input)
        # x = nn.AdaptiveAvgPool2d(1)(x)
        # x = x.view(x.shape[0], -1)
        # output = self.fc0(x)
        # return output
        return x
