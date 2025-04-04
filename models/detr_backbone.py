# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Backbone modules.
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import models.resnetv3 as resnetv3



class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        # if return_interm_layers:
        #     return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        #     self.strides = [8, 16, 32]
        #     self.num_channels = [512, 1024, 2048]
        # else:
        #     return_layers = {'layer4': "0"}
        #     self.strides = [32]
        #     self.num_channels = [2048]
        # self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.body = backbone

    def forward(self, tensor, preact=False):
        return self.body(tensor, preact=False)
        # xs = self.body(tensor)
        # out: Dict[str, tensor] = {}
        # for name, x in xs.items():
        #     out[name] = x
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,):
        
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(resnetv3, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True, norm_layer=norm_layer)
        
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"

        super().__init__(backbone, train_backbone, return_interm_layers)

        if dilation:
            self.strides[-1] = self.strides[-1] // 2


def build_backbone(input_channels, **kwargs):
    return_interm_layers = True
    train_backbone = True
    dilation = False
    backbone = Backbone('resnet50', train_backbone, return_interm_layers, dilation)

    if input_channels != 3:
        # 修改backbone第一个卷积的输入通道数
        conv1_3ch = backbone.body.conv1
        new_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=conv1_3ch.out_channels,
            kernel_size=conv1_3ch.kernel_size,
            stride=conv1_3ch.stride,
            padding=conv1_3ch.padding,
            dilation=conv1_3ch.dilation,
            groups=conv1_3ch.groups,
            bias=(conv1_3ch.bias is not None)
        )
        backbone.body.conv1 = new_conv
    return backbone
