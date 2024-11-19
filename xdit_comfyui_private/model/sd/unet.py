#Original code can be found on: https://github.com/black-forest-labs/flux

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel #, forward_timestep_embed, apply_control

class xFuserUnet(UNetModel):
    """
    UNet model for xFuser CFG parallel.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_orig(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        return super().forward(x, timesteps, context, y, control, transformer_options, **kwargs)

    def forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        out = self.forward_orig(x, timesteps, context, y, control, transformer_options, **kwargs)
        return out