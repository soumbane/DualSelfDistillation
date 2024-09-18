# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from scipy import spatial
from typing import Union
import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Conv, Dropout, Norm, split_args
from monai.utils import deprecated_arg  # type: ignore

from monai.utils import UpsampleMode, InterpolateMode # type: ignore

# Relative import for final training model
from .deepUp import DeepUp

# Absolute import for testing this script
# from deepUp import DeepUp


def get_acti_layer(act: tuple[str, dict] | str, nchan: int = 0):
    if act == "prelu":
        act = ("prelu", {"num_parameters": nchan})
    act_name, act_args = split_args(act)
    act_type = Act[act_name]
    return act_type(**act_args)


class LUConv(nn.Module):

    def __init__(self, spatial_dims: int, nchan: int, act: tuple[str, dict] | str, bias: bool = False):
        super().__init__()

        self.act_function = get_acti_layer(act, nchan)
        self.conv_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=nchan,
            out_channels=nchan,
            kernel_size=5,
            act=None,
            norm=Norm.BATCH,
            bias=bias,
        )

    def forward(self, x):
        out = self.conv_block(x)
        out = self.act_function(out)
        return out


def _make_nconv(spatial_dims: int, nchan: int, depth: int, act: tuple[str, dict] | str, bias: bool = False):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(spatial_dims, nchan, act, bias))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):

    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, act: tuple[str, dict] | str, bias: bool = False
    ):
        super().__init__()

        if out_channels % in_channels != 0:
            raise ValueError(
                f"out channels should be divisible by in_channels. Got in_channels={in_channels}, out_channels={out_channels}."
            )

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_function = get_acti_layer(act, out_channels)
        self.conv_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            act=None,
            norm=Norm.BATCH,
            bias=bias,
        )

    def forward(self, x):
        out = self.conv_block(x)
        repeat_num = self.out_channels // self.in_channels
        x16 = x.repeat([1, repeat_num, 1, 1, 1][: self.spatial_dims + 2])
        out = self.act_function(torch.add(out, x16))
        return out


class DownTransition(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        nconvs: int,
        act: tuple[str, dict] | str,
        dropout_prob: float | None = None,
        dropout_dim: int = 3,
        bias: bool = False,
    ):
        super().__init__()

        conv_type: type[nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        norm_type: type[nn.BatchNorm2d | nn.BatchNorm3d] = Norm[Norm.BATCH, spatial_dims]
        dropout_type: type[nn.Dropout | nn.Dropout2d | nn.Dropout3d] = Dropout[Dropout.DROPOUT, dropout_dim]

        out_channels = 2 * in_channels
        self.down_conv = conv_type(in_channels, out_channels, kernel_size=2, stride=2, bias=bias)
        self.bn1 = norm_type(out_channels)
        self.act_function1 = get_acti_layer(act, out_channels)
        self.act_function2 = get_acti_layer(act, out_channels)
        self.ops = _make_nconv(spatial_dims, out_channels, nconvs, act, bias)
        self.dropout = dropout_type(dropout_prob) if dropout_prob is not None else None

    def forward(self, x):
        down = self.act_function1(self.bn1(self.down_conv(x)))
        if self.dropout is not None:
            out = self.dropout(down)
        else:
            out = down
        out = self.ops(out)
        out = self.act_function2(torch.add(out, down))
        return out


class UpTransition(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        nconvs: int,
        act: tuple[str, dict] | str,
        dropout_prob: tuple[float | None, float] = (None, 0.5),
        dropout_dim: int = 3,
    ):
        super().__init__()

        conv_trans_type: type[nn.ConvTranspose2d | nn.ConvTranspose3d] = Conv[Conv.CONVTRANS, spatial_dims]
        norm_type: type[nn.BatchNorm2d | nn.BatchNorm3d] = Norm[Norm.BATCH, spatial_dims]
        dropout_type: type[nn.Dropout | nn.Dropout2d | nn.Dropout3d] = Dropout[Dropout.DROPOUT, dropout_dim]

        self.up_conv = conv_trans_type(in_channels, out_channels // 2, kernel_size=2, stride=2)
        self.bn1 = norm_type(out_channels // 2)
        self.dropout = dropout_type(dropout_prob[0]) if dropout_prob[0] is not None else None
        self.dropout2 = dropout_type(dropout_prob[1])
        self.act_function1 = get_acti_layer(act, out_channels // 2)
        self.act_function2 = get_acti_layer(act, out_channels)
        self.ops = _make_nconv(spatial_dims, out_channels, nconvs, act)

    def forward(self, x, skipx):
        if self.dropout is not None:
            out = self.dropout(x)
        else:
            out = x
        skipxdo = self.dropout2(skipx)
        out = self.act_function1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.act_function2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):

    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, act: tuple[str, dict] | str, bias: bool = False
    ):
        super().__init__()

        conv_type: type[nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]

        self.act_function1 = get_acti_layer(act, out_channels)
        self.conv_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            act=None,
            norm=Norm.BATCH,
            bias=bias,
        )
        self.conv2 = conv_type(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.conv_block(x)
        out = self.act_function1(out)
        out = self.conv2(out)
        return out


class SelfDistillVNet(nn.Module):
    """
    V-Net based on `Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        in_channels: number of input channels for the network. Defaults to 1.
            The value should meet the condition that ``16 % in_channels == 0``.
        out_channels: number of output channels for the network. Defaults to 1.
        act: activation type in the network. Defaults to ``("elu", {"inplace": True})``.
        dropout_prob_down: dropout ratio for DownTransition blocks. Defaults to 0.5.
        dropout_prob_up: dropout ratio for UpTransition blocks. Defaults to (0.5, 0.5).
        dropout_dim: determine the dimensions of dropout. Defaults to (0.5, 0.5).

            - ``dropout_dim = 1``, randomly zeroes some of the elements for each channel.
            - ``dropout_dim = 2``, Randomly zeroes out entire channels (a channel is a 2D feature map).
            - ``dropout_dim = 3``, Randomly zeroes out entire channels (a channel is a 3D feature map).
        bias: whether to have a bias term in convolution blocks. Defaults to False.
            According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
            if a conv layer is directly followed by a batch norm layer, bias should be False.

    """

    @deprecated_arg(
        name="dropout_prob",
        since="1.2",
        new_name="dropout_prob_down",
        msg_suffix="please use `dropout_prob_down` instead.",
    )
    @deprecated_arg(
        name="dropout_prob", since="1.2", new_name="dropout_prob_up", msg_suffix="please use `dropout_prob_up` instead."
    )
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        self_distillation: bool = False,
        mode: Union[UpsampleMode, str] = UpsampleMode.DECONV,
        interp_mode: Union[InterpolateMode, str] = InterpolateMode.LINEAR,
        multiple_upsample: bool = True,
        act: tuple[str, dict] | str = ("elu", {"inplace": True}),
        dropout_prob: float | None = 0.5,  # deprecated
        dropout_prob_down: float | None = 0.5,
        dropout_prob_up: tuple[float | None, float] = (0.5, 0.5),
        dropout_dim: int = 3,
        bias: bool = False,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise AssertionError("spatial_dims can only be 2 or 3.")

        self.in_tr = InputTransition(spatial_dims, in_channels, 16, act, bias=bias)
        self.down_tr32 = DownTransition(spatial_dims, 16, 1, act, bias=bias)
        self.down_tr64 = DownTransition(spatial_dims, 32, 2, act, bias=bias)
        self.down_tr128 = DownTransition(spatial_dims, 64, 3, act, dropout_prob=dropout_prob_down, bias=bias)
        self.down_tr256 = DownTransition(spatial_dims, 128, 2, act, dropout_prob=dropout_prob_down, bias=bias)
        self.up_tr256 = UpTransition(spatial_dims, 256, 256, 2, act, dropout_prob=dropout_prob_up)
        self.up_tr128 = UpTransition(spatial_dims, 256, 128, 2, act, dropout_prob=dropout_prob_up)
        self.up_tr64 = UpTransition(spatial_dims, 128, 64, 1, act)
        self.up_tr32 = UpTransition(spatial_dims, 64, 32, 1, act)
        self.out_tr = OutputTransition(spatial_dims, 32, out_channels, act, bias=bias)

        self.self_distillation = self_distillation

        #########################################
        # Upsample blocks for Self Distillation #
        #########################################

        if self.self_distillation:            
            self.deep_down_tr128_enc = DeepUp(
            spatial_dims = 3,
            in_channels = 128,
            out_channels = out_channels,
            scale_factor = 8,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )

            self.deep_down_tr256_dec = DeepUp(
            spatial_dims = 3,
            in_channels = 256,
            out_channels = out_channels,
            scale_factor = 16,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )       

            self.deep_down_tr64_enc = DeepUp(
            spatial_dims = 3,
            in_channels = 64,
            out_channels = out_channels,
            scale_factor = 4,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )

            self.deep_up_tr256_dec = DeepUp(
            spatial_dims = 3,
            in_channels = 256,
            out_channels = out_channels,
            scale_factor = 8,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )

            self.deep_down_tr32_enc = DeepUp(
            spatial_dims = 3,
            in_channels = 32,
            out_channels = out_channels,
            scale_factor = 2,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )

            self.deep_up_tr128_dec = DeepUp(
            spatial_dims = 3,
            in_channels = 128,
            out_channels = out_channels,
            scale_factor = 4,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )

            self.deep_in_tr_enc = DeepUp(
            spatial_dims = 3,
            in_channels = 16,
            out_channels = out_channels,
            scale_factor = 1,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )

            self.deep_up_tr64_dec = DeepUp(
            spatial_dims = 3,
            in_channels = 64,
            out_channels = out_channels,
            scale_factor = 2,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )

            self.deep_up_tr32_dec = DeepUp(
            spatial_dims = 3,
            in_channels = 32,
            out_channels = out_channels,
            scale_factor = 1,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )

    def forward(self, x):
        
        out16 = self.in_tr(x)
        # print(f'Output of in_tr: {out16.shape}')

        if self.self_distillation:
            out_enc1 = self.deep_in_tr_enc(out16) 
        else:
            out_enc1 = None   

        out32 = self.down_tr32(out16)
        # print(f'Output of down_tr32: {out32.shape}')

        if self.self_distillation:
            out_enc2 = self.deep_down_tr32_enc(out32) 
        else:
            out_enc2 = None

        out64 = self.down_tr64(out32)
        # print(f'Output of down_tr64: {out64.shape}')

        if self.self_distillation:
            out_enc3 = self.deep_down_tr64_enc(out64) 
        else:
            out_enc3 = None

        out128 = self.down_tr128(out64)
        # print(f'Output of down_tr128: {out128.shape}')

        if self.self_distillation:
            out_enc4 = self.deep_down_tr128_enc(out128) 
        else:
            out_enc4 = None

        out256 = self.down_tr256(out128)
        # print(f'Output of down_tr256: {out256.shape}')

        if self.self_distillation:
            out_dec4 = self.deep_down_tr256_dec(out256) 
        else:
            out_dec4 = None
        
        x = self.up_tr256(out256, out128)
        # print(f'Output of up_tr256: {x.shape}')

        if self.self_distillation:
            out_dec3 = self.deep_up_tr256_dec(x) 
        else:
            out_dec3 = None

        x = self.up_tr128(x, out64)
        # print(f'Output of up_tr128: {x.shape}')

        if self.self_distillation:
            out_dec2 = self.deep_up_tr128_dec(x) 
        else:
            out_dec2 = None

        x = self.up_tr64(x, out32)
        # print(f'Output of up_tr64: {x.shape}')

        if self.self_distillation:
            out_dec1 = self.deep_up_tr64_dec(x) 
            # print(f'Output of out_dec1: {out_dec1.shape}')
        else:
            out_dec1 = None

        x = self.up_tr32(x, out16)
        # print(f'Output of up_tr32: {x.shape}')

        out_main = self.out_tr(x)

        # For Self Distillation (ONLY during training)
        if self.training and self.self_distillation:
            ## For Self Distillation
            # Encoders:out_enc4: deepest encoder and out_enc3, out_enc2, out_enc1: shallow encoders
            # Decoders:out_dec1: deepest decoder and out_dec2, out_dec3, out_dec4: shallow decoders         
                        
            # Full KL Div WITHOUT feature maps - includes both encoders and decoders
            out = (out_main, out_dec4, out_dec3, out_dec2, out_dec1, out_enc1, out_enc2, out_enc3, out_enc4)
            
        elif self.training and not self.self_distillation:
            # For Basic VNET ONLY (NO Self-Distillation)
            out = out_main  
        else:
            # For validation/testing (NO Self-Distillation)
            out = out_main

        return out    


if __name__ == '__main__':
    vnet_with_self_distil = SelfDistillVNet(
        spatial_dims = 3,
        in_channels = 4,
        out_channels = 4,
        self_distillation = True,
        )

    ## Count model parameters
    total_params = sum(p.numel() for p in vnet_with_self_distil.parameters() if p.requires_grad)
    print(f'The total number of model parameter is: {total_params}')
    

    x1 = torch.rand((1, 4, 128, 128, 128)) # (B,num_ch,x,y,z)
    print("VNet input shape: ", x1.shape)

    # x4 = vnet_with_self_distil(x1)
    # print("Self Distil VNet output shape: ", x4.shape)

    x4 = vnet_with_self_distil(x1)
    print("Self Distil VNet output main shape: ", x4[0].shape)
    print("Self Distil VNet output dec shape: ", x4[1].shape)
    print("Self Distil VNet output enc shape: ", x4[8].shape)