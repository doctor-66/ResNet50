# print("Please define your ResNet50 in this file.")
import torch
import torch.nn as nn

EXPANSION = 4


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.dp1 = nn.Dropout(0.3)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dp2 = nn.Dropout(0.3)
        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(planes, EXPANSION * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(EXPANSION * planes)
        self.dp3 = nn.Dropout(0.3)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual=x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dp1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dp2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dp3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(EXPANSION * 512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != EXPANSION * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, EXPANSION * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(EXPANSION * planes),
            )

        layers = []
        layers.append(Bottleneck(self.in_planes, planes, stride, downsample))
        self.in_planes = EXPANSION * planes
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(self.in_planes, planes))

        return nn.Sequential(*layers)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # x = self.dropout(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # x = self.dropout(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        # x = self.softmax(x)
        return x


def ResNet50():
    return ResNet([3, 4, 6, 3])

# print(ResNet50().eval())