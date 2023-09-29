import torch.nn as nn
import torch

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn(x)))
        out = torch.cat((x, out), 1)
        return out

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.avgpool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn(x)))
        out = self.avgpool(out)
        return out

class DenseNet3D(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_classes=5):
        super(DenseNet3D, self).__init__()
        self.conv1 = nn.Conv3d(24, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        num_init_features = 64
        self.features = nn.Sequential()
        self.features.add_module('conv1', self.conv1)
        self.features.add_module('bn1', self.bn1)
        self.features.add_module('relu', self.relu)
        self.features.add_module('maxpool', self.maxpool)

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = self._make_dense_block(num_features, growth_rate, num_layers)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = self._make_transition_block(num_features, num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(num_features, num_classes)

    def _make_dense_block(self, in_channels, growth_rate, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(DenseBlock(in_channels + i * growth_rate, growth_rate))
        return nn.Sequential(*layers)

    def _make_transition_block(self, in_channels, out_channels):
        return TransitionBlock(in_channels, out_channels)

    def forward(self, x):
        features = self.features(x)
        out = self.avgpool(features)
        out = out.view(features.size(0), -1)
        out = self.fc(out)
        return out


