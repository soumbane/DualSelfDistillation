# pyright: reportPrivateImportUsage=false
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer, UnetOutBlock
from monai.networks.blocks import UpSample
from monai.utils import UpsampleMode

class DeepUp(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        scale_factor: int
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            scale_factor: scale by which upsampling is needed
        """

        super().__init__()

        self.conv_block = UnetOutBlock(
            spatial_dims=spatial_dims, 
            in_channels=in_channels, 
            out_channels=out_channels
        )
        
        self.transp_conv = UpSample(
            spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            scale_factor=scale_factor,
            mode=UpsampleMode.DECONV,
            bias=True,
            apply_pad_pool=True,
        )

    def forward(self, inp):        
        out = self.conv_block(inp)
        out = self.transp_conv(out)
        return out

if __name__ == '__main__':
    block_connector = DeepUp(
        spatial_dims = 3,
        in_channels = 1,
        out_channels = 8,
        kernel_size = 1, # type:ignore
        upsample_kernel_size = 2, # type:ignore
        norm_name = 'instance', # type:ignore
        res_block = False # type:ignore
        )  # type:ignore

    x1 = torch.rand((1, 1, 96, 96, 96)) # (B,in_ch,x,y,z)
    print("Deep Upsampling block input shape: ", x1.shape)

    x3 = block_connector(x1)
    print("Deep Upsampling block output shape: ", x3.shape)
