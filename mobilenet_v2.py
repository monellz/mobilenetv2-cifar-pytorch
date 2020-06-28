import torch
import torch.nn as nn


class MobileNetV2(nn.Module):
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        super(MobileNetV2, self).__init__()
        input_channel = input_shape[-1]
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=3, stride=1, bias=False)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.25)
        self.dropout4 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, bias=False)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.conv3 = nn.Conv2d(1280, num_classes, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bottleneck(x, 32, 16, expansion=1, stride=1)

        x = self.bottleneck(x, 16, 24, expansion=6, stride=1)
        x = self.bottleneck(x, 24, 24, expansion=6, stride=1)

        x = self.bottleneck(x, 24, 32, expansion=6, stride=2)
        x = self.bottleneck(x, 32, 32, expansion=6, stride=1)
        x = self.bottleneck(x, 32, 32, expansion=6, stride=1)

        x = self.bottleneck(x, 32, 64, expansion=6, stride=2)
        x = self.bottleneck(x, 64, 64, expansion=6, stride=1)
        x = self.bottleneck(x, 64, 64, expansion=6, stride=1)
        x = self.bottleneck(x, 64, 64, expansion=6, stride=1)
        x = self.dropout1(x)

        x = self.bottleneck(x, 64, 96, expansion=6, stride=1)
        x = self.bottleneck(x, 96, 96, expansion=6, stride=1)
        x = self.bottleneck(x, 96, 96, expansion=6, stride=1)
        x = self.dropout2(x)

        x = self.bottleneck(x, 96, 160, expansion=6, stride=2)
        x = self.bottleneck(x, 160, 160, expansion=6, stride=1)
        x = self.bottleneck(x, 160, 160, expansion=6, stride=1)
        x = self.dropout3(x)

        x = self.bottleneck(x, 160, 320, expansion=6, stride=1)
        x = self.dropout4(x)

        x = self.conv2(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)

        return x

    def bottleneck(self, x, input_channel, output_channel, expansion, stride):
        expand_channel = input_channel * expansion
        x_ = x
        x = nn.Conv2d(input_channel, expand_channel, kernel_size=1, bias=False)(x)
        x = nn.BatchNorm2d(expand_channel)(x)
        x = nn.ReLU(inplace=True)(x) # TODO:ReLU6?

        x = nn.Conv2d(expand_channel, expand_channel, kernel_size=3, stride=stride, padding=1, bias=False, groups=expand_channel)(x)
        x = nn.BatchNorm2d(expand_channel)(x)
        x = nn.ReLU(inplace=True)(x)

        x = nn.Conv2d(expand_channel, output_channel, kernel_size=1, stride=stride, bias=False)(x)
        x = nn.BatchNorm2d(output_channel)(x)

        x += x_
        return x
        














