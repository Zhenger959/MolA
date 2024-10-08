'''
Author: Jiaxin Zheng
Date: 2023-12-05 14:14:22
LastEditors: Jiaxin Zheng
LastEditTime: 2024-01-02 16:44:48
Description: 
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

https://download.pytorch.org/models/resnet50-11ad3fa6.pth
/data/zhengjiaxin/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth
'''
import torch
from torch import nn
import math 
import numpy as np 
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights


def get_point_of_trisection_coordinates(segment, ratio_b, ratio_e):
    #segment = np.array(segment)
    #trisection = (ratio_e*segment + ratio_b*segment)/(ratio_b + ratio_e)
    
    x = (ratio_e*segment[0][0] + ratio_b*segment[1][0])/(ratio_b + ratio_e)
    y = (ratio_e*segment[0][1] + ratio_b*segment[1][1])/(ratio_b + ratio_e)
    return torch.tensor([x, y], requires_grad=False)

def get_positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                        "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe

def get_positional_encoding_2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model)) 
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe


class Backbone(nn.Module):
    """Base class for backbone networks. Handles freezing layers etc.
    args:
        frozen_layers  -  Name of layers to freeze. Either list of strings, 'none' or 'all'. Default: 'none'.
    """
    def __init__(self, frozen_layers=()):
        super().__init__()

        if isinstance(frozen_layers, str):
            if frozen_layers.lower() == 'none':
                frozen_layers = ()
            elif frozen_layers.lower() != 'all':
                raise ValueError('Unknown option for frozen layers: \"{}\". Should be \"all\", \"none\" or list of layer names.'.format(frozen_layers))

        self.frozen_layers = frozen_layers
        self._is_frozen_nograd = False


    def train(self, mode=True):
        super().train(mode)
        if mode == True:
            self._set_frozen_to_eval()
        if not self._is_frozen_nograd:
            self._set_frozen_to_nograd()
            self._is_frozen_nograd = True
        return self


    def _set_frozen_to_eval(self):
        if isinstance(self.frozen_layers, str) and self.frozen_layers.lower() == 'all':
            self.eval()
        else:
            for layer in self.frozen_layers:
                getattr(self, layer).eval()


    def _set_frozen_to_nograd(self):
        if isinstance(self.frozen_layers, str) and self.frozen_layers.lower() == 'all':
            for p in self.parameters():
                p.requires_grad_(False)
        else:
            for layer in self.frozen_layers:
                for p in getattr(self, layer).parameters():
                    p.requires_grad_(False)


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)

        if use_bn:
            self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        if self.use_bn:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(Backbone):
    """ ResNet network module. Allows extracting specific feature blocks."""
    def __init__(self, block, layers, output_layers, conv1_stride=2, num_classes=1000, inplanes=64, dilation_factor=1, frozen_layers=()):
        self.inplanes = inplanes
        super(ResNet, self).__init__(frozen_layers=frozen_layers)
        self.output_layers = output_layers
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=conv1_stride, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stride = [1 + (dilation_factor < l) for l in (8, 4, 2)]
        self.layer1 = self._make_layer(block, inplanes, layers[0], dilation=max(dilation_factor//8, 1))
        self.layer2 = self._make_layer(block, inplanes*2, layers[1], stride=stride[0], dilation=max(dilation_factor//4, 1))
        self.layer3 = self._make_layer(block, inplanes*4, layers[2], stride=stride[1], dilation=max(dilation_factor//2, 1))
        self.layer4 = self._make_layer(block, inplanes*8, layers[3], stride=stride[2], dilation=dilation_factor)

        out_feature_strides = {'conv1': 4, 'layer1': 4, 'layer2': 4*stride[0], 'layer3': 4*stride[0]*stride[1],
                               'layer4': 4*stride[0]*stride[1]*stride[2]}

        if isinstance(self.layer1[0], BasicBlock):
            out_feature_channels = {'conv1': inplanes, 'layer1': inplanes, 'layer2': inplanes*2, 'layer3': inplanes*4,
                               'layer4': inplanes*8}
        elif isinstance(self.layer1[0], Bottleneck):
            base_num_channels = 4 * inplanes
            out_feature_channels = {'conv1': inplanes, 'layer1': base_num_channels, 'layer2': base_num_channels * 2,
                                    'layer3': base_num_channels * 4, 'layer4': base_num_channels * 8}
        else:
            raise Exception('block not supported')

        self._out_feature_strides = out_feature_strides
        self._out_feature_channels = out_feature_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(inplanes*8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def out_feature_strides(self, layer=None):
        if layer is None:
            return self._out_feature_strides
        else:
            return self._out_feature_strides[layer]

    def out_feature_channels(self, layer=None):
        if layer is None:
            return self._out_feature_channels
        else:
            return self._out_feature_channels[layer]

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, x, output_layers=None):
        """ Forward pass with input x. The output_layers specify the feature blocks which must be returned """
        outputs = OrderedDict()

        if output_layers is None:
            output_layers = self.output_layers

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self._add_output_and_check('conv1', x, outputs, output_layers):
            return outputs

        x = self.maxpool(x)

        x = self.layer1(x)

        if self._add_output_and_check('layer1', x, outputs, output_layers):
            return outputs

        x = self.layer2(x)

        if self._add_output_and_check('layer2', x, outputs, output_layers):
            return outputs

        x = self.layer3(x)

        if self._add_output_and_check('layer3', x, outputs, output_layers):
            return outputs

        x = self.layer4(x)

        if self._add_output_and_check('layer4', x, outputs, output_layers):
            return outputs

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if self._add_output_and_check('fc', x, outputs, output_layers):
            return outputs

        if len(output_layers) == 1 and output_layers[0] == 'default':
            return x

        raise ValueError('output_layer is wrong.')


def resnet18(output_layers=None, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(BasicBlock, [2, 2, 2, 2], output_layers, **kwargs)

    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
        state_dict = torch.hub.load_state_dict_from_url(weights.url)
        model.load_state_dict(state_dict)
    return model


def resnet50(output_layers=None, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(Bottleneck, [3, 4, 6, 3], output_layers, **kwargs)
    if pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V2
        state_dict = torch.hub.load_state_dict_from_url(weights.url)
        model.load_state_dict(state_dict)
    return model


def resnet101(output_layers=None, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(Bottleneck, [3, 4, 6, 3], output_layers, **kwargs)
    if pretrained:
        weights = ResNet101_Weights.IMAGENET1K_V2
        state_dict = torch.hub.load_state_dict_from_url(weights.url)
        model.load_state_dict(state_dict)
    return model