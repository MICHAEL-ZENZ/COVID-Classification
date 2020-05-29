import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

from model import ResNet
from model.layer import dcpLayer

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class dcpBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, 
                R1_ratio,R2_ratio,
                stride=1, downsample=None, groups=1,
                base_width=64, dilation=1, norm_layer=None):
        super(dcpBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.R1_ratio=R1_ratio
        self.R2_ratio=R2_ratio
        if groups != 1 or base_width != 64:
            raise ValueError('dcpBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in dcpBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = dcpLayer.tuckerDCP_Conv2D(inplanes, planes, kernel_size=3, stride=stride,
                    R1_ratio=self.R1_ratio, R2_ratio=self.R2_ratio,
                    padding=1, groups=1, bias=False, dilation=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = dcpLayer.tuckerDCP_Conv2D(planes, planes, kernel_size=3, stride=1,
                    R1_ratio=self.R1_ratio, R2_ratio=self.R2_ratio,
                    padding=1, groups=1, bias=False, dilation=1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class dcpBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, 
                R1_ratio,R2_ratio,
                stride=1, downsample=None, groups=1,
                base_width=64, dilation=1, norm_layer=None):
        super(dcpBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.R1_ratio=R1_ratio
        self.R2_ratio=R2_ratio
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = dcpLayer.tuckerDCP_Conv2D(inplanes, width, kernel_size=1,
                    R1_ratio=self.R1_ratio, R2_ratio=self.R2_ratio,
                    stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = dcpLayer.tuckerDCP_Conv2D(width, width, kernel_size=3, stride=stride,
                        R1_ratio=self.R1_ratio, R2_ratio=self.R2_ratio,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = dcpLayer.tuckerDCP_Conv2D(width, planes * self.expansion, kernel_size=1,
                        R1_ratio=self.R1_ratio, R2_ratio=self.R2_ratio,
                        stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

dcpBlock2ResBlockMapper={
    dcpBasicBlock:ResNet.BasicBlock,
    dcpBottleneck:ResNet.Bottleneck
}

class dcpResNet(nn.Module):

    def __init__(self, block, layers, R1_ratio=1.0, R2_ratio=1.0,
                num_classes=2, zero_init_residual=False,
                groups=1, width_per_group=64, replace_stride_with_dilation=None,
                norm_layer=None):
        super(dcpResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.R1_ratio=R1_ratio
        self.R2_ratio=R2_ratio
        self._norm_layer = norm_layer
        self.block = block
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7,
                                stride=2, padding=3,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, dcpBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, dcpBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                dcpLayer.tuckerDCP_Conv2D(self.inplanes, planes * block.expansion,
                R1_ratio=self.R1_ratio, R2_ratio=self.R2_ratio,
                kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.R1_ratio, self.R2_ratio,
                             stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.R1_ratio, self.R2_ratio,
                                groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def change_cls_number(self,num_classes):
        print("Changing Full Connected")
        self.fc = nn.Linear(512 * self.block.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def loadStateFromModel(self,resnet:ResNet.ResNet):
        gen1,gen2=self.modules(),resnet.modules()

        while True:
            m1,m2=next(gen1,None), next(gen2,None)
            if not m1 or not m2:
                if m1!=m2:
                    raise ValueError("It seems the 2 models does not share same structure")
                break

            if isinstance(m2, nn.Conv2d):
                if isinstance(m1,dcpLayer.dcpConv2D):
                    m1.loadStateFromModule(m2)
                    next(gen1)
                    next(gen1)
                    next(gen1)
                else:
                    m1.weight.data.copy_(m2.weight.data)
                    if m1.bias is not None:
                        m1.bias.data.copy_(m2.bias.data)
            elif isinstance(m1, (nn.BatchNorm2d, nn.GroupNorm)):
                m1.weight.data.copy_(m2.weight.data)
                m1.bias.data.copy_(m2.bias.data)
                m1.running_mean.data.copy_(m2.running_mean.data)
                m1.running_var.data.copy_(m2.running_var.data)

            elif isinstance(m1, nn.Linear):
                m1.weight.data.copy_(m2.weight.data)
                m1.bias.data.copy_(m2.bias.data)

def _dcpresnet(arch, block, layers, pretrained, progress, **kwargs):

    return dcpResNet(block, layers, **kwargs)


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dcpresnet('resnet18', dcpBasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dcpresnet('resnet34', dcpBasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dcpresnet('resnet50', dcpBottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dcpresnet('resnet101', dcpBottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dcpresnet('resnet152', dcpBottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _dcpresnet('resnext50_32x4d', dcpBottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _dcpresnet('resnext101_32x8d', dcpBottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _dcpresnet('wide_resnet50_2', dcpBottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _dcpresnet('wide_resnet101_2', dcpBottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
