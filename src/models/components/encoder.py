'''
Author: Jiaxin Zheng
Date: 2023-09-01 10:49:48
LastEditors: Jiaxin Zheng
LastEditTime: 2024-10-01 15:53:23
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from src.models.components.swin_transformer import swin_base
from src.models.components.resnet import resnet18,resnet50

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_name = args.encoder
        self.model_name = model_name
        if model_name.startswith('resnet'):
            self.model_type = 'resnet'
            # self.cnn = timm.create_model(model_name, pretrained=args.pretrained)
            # self.n_features = self.cnn.num_features  # encoder_dim
            # self.cnn.global_pool = nn.Identity()
            # self.cnn.fc = nn.Identity()
            
            if model_name=='resnet18':
                self.cnn=resnet18(
                    pretrained = False, 
                    output_layers = ['layer4'], 
                    dilation_factor = 8,
                    conv1_stride = 2
                ) # 512
                
            if model_name=='resnet50':
                self.cnn = resnet50(
                pretrained = args.pretrained, 
                output_layers = ['layer4'], 
                dilation_factor = 2,
                conv1_stride = 2
                )
                
                self.cnn.layer4 = nn.Identity()
                self.cnn.avgpool = nn.Identity()
                self.cnn.fc = nn.Identity()
                
                self.resnet_channels=list(self.cnn.children())[-3][-1].conv3.out_channels
                self.cnn_out_channels=args.encoder_dim
                self.resnet_linear = nn.Linear(576,self.cnn_out_channels,bias=False)
            
        elif model_name.startswith('swin'):
            self.model_type = 'swin'
            self.transformer = timm.create_model(model_name, pretrained=args.pretrained, pretrained_strict=False,
                                                 use_checkpoint=args.use_checkpoint)
            self.n_features = self.transformer.num_features
            self.transformer.head = nn.Identity()
        elif 'efficientnet' in model_name:
            self.model_type = 'efficientnet'
            self.cnn = timm.create_model(model_name, pretrained=args.pretrained)
            self.n_features = self.cnn.num_features
            self.cnn.global_pool = nn.Identity()
            self.cnn.classifier = nn.Identity()
        else:
            raise NotImplemented

    def swin_forward(self, transformer, x):
        x = transformer.patch_embed(x)
        if transformer.absolute_pos_embed is not None:
            x = x + transformer.absolute_pos_embed
        x = transformer.pos_drop(x)

        def layer_forward(layer, x, hiddens):
            for blk in layer.blocks:
                if not torch.jit.is_scripting() and layer.use_checkpoint:
                    x = torch.utils.checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
            H, W = layer.input_resolution
            B, L, C = x.shape
            hiddens.append(x.view(B, H, W, C))
            if layer.downsample is not None:
                x = layer.downsample(x)
            return x, hiddens

        hiddens = []
        for layer in transformer.layers:
            x, hiddens = layer_forward(layer, x, hiddens)
        x = transformer.norm(x)  # B L C
        hiddens[-1] = x.view_as(hiddens[-1])
        return x, hiddens

    def forward(self, x, refs=None):
        # z:  x(4,3,384,384)  features(4,144,1024)  len(hiddens)=4
        if self.model_type == 'resnet':
            features = self.cnn(x)['layer4']
            bs = features.shape[0]
            features = features.reshape(bs,self.resnet_channels,-1)
            features = self.resnet_linear(features)
            hiddens = []
        # if self.model_type == 'resnet':
        #     features = self.cnn(x)['layer3']
        #     features = self.convert_conv(features)
        #     bs = features.shape[0]
        #     features = features.reshape(bs,-1,self.cnn_out_channels)
        #     hiddens = []
        elif self.model_type == 'swin':
            if 'patch' in self.model_name:
                features, hiddens = self.swin_forward(self.transformer, x)
            else:
                features, hiddens = self.transformer(x)
        else:
            raise NotImplemented
        return features, hiddens