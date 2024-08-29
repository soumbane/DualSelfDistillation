from __future__ import annotations
from typing import Any, Optional, Union, Sequence

import torch

from monai.networks.nets.vnet import VNet
from .self_distill_vnet import SelfDistillVNet


class VNetWithDictOutput(VNet):
    """
    A VNet that returns a dictionary during training
    
    * extends: `VNet`
    """

    def forward(self, x_in: torch.Tensor) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        #print("Input shape:", x_in)
        # repeat modality for input tensor
        return {"out": super().forward(x_in)} if self.training else super().forward(x_in)


class SelfDistillVNetWithDictOutput(SelfDistillVNet):
    """
    A VNet that returns a dictionary during training
    
    * extends: `SelfDistilVNet`
    """

    def forward(self, x_in: torch.Tensor) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        # return {"out": super().forward(x_in)} if self.training else super().forward(x_in)        
        y = super().forward(x_in)
        return {"out": y, "dist_map": y} if self.training else y # type:ignore

