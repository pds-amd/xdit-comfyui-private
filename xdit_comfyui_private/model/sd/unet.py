#Original code can be found on: https://github.com/black-forest-labs/flux

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn
# from comfy.ldm.modules.diffusionmodules.util import timestep_embedding
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
        # bs, c, h, w = x.shape

        # rank = dist.get_rank()
        # world_size = dist.get_world_size()

        # if bs == world_size and world_size > 1:
        #     x = x[rank:rank+1]
        #     timesteps = timesteps[rank:rank+1]
        #     y = y[rank:rank+1]
        #     context = context[rank:rank+1]
        #     transformer_options['cond_or_uncond'] = transformer_options['cond_or_uncond'][rank:rank+1]
        #     out = super().forward(x, timesteps, context, y, control, transformer_options, **kwargs)
        #     out_list = [
        #         torch.empty_like(out) for _ in range(dist.get_world_size())
        #     ]
        #     dist.all_gather(out_list, out)
        #     out = torch.cat(out_list, dim=0)
        # else:
        #     out = super().forward(x, timesteps, context, y, control, transformer_options, **kwargs)
        out = self.forward_orig(x, timesteps, context, y, control, transformer_options, **kwargs)
        return out