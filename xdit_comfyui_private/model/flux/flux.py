#Original code can be found on: https://github.com/black-forest-labs/flux

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn

from comfy.ldm.flux.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)

from comfy.ldm.flux.model import Flux, FluxParams

from einops import rearrange, repeat
import comfy.ldm.common_dit
from .layers import xFuserDoubleStreamBlock, xFuesrSingleStreamBlock

class FluxSegmentation(nn.Module):
    def __init__(self,):
        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def forward(self, x):
        num_tokens = x.shape[1] + self.world_size - 1
        token_start_idx = num_tokens // self.world_size * self.rank
        token_end_idx = min(num_tokens // self.world_size * (self.rank + 1), x.shape[1])
        return x[:, token_start_idx:token_end_idx, :]


class xFuserFlux(Flux):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        super().__init__(image_model=image_model, final_layer=final_layer, dtype=dtype, device=device, operations=operations, **kwargs)
        params = FluxParams(**kwargs)
        self.segment_image_text = FluxSegmentation()
        self.double_blocks = nn.ModuleList(
            [
                xFuserDoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for _ in range(params.depth)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                xFuesrSingleStreamBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control=None,
        neg_mode=None,
        block_controlnet_hidden_states=None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.segment_image_text(img)
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        vec = vec + self.vector_in(y)
        
        # txt = self.segment_image_text(txt)
        txt = self.txt_in(txt)

        # txt_ids = self.segment_image_text(txt_ids)
        img_ids = self.segment_image_text(img_ids)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)


        for i, block in enumerate(self.double_blocks):
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
            if block_controlnet_hidden_states is not None:
                block_controlnet_hidden_state = self.segment_image_text(block_controlnet_hidden_states[i % 2])
                img = img + block_controlnet_hidden_state

        img = torch.cat((txt, img), 1)

        for i, block in enumerate(self.single_blocks):
            img = block(img, vec=vec, pe=pe)

        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward(self, x, timestep, context, y, guidance, control=None, **kwargs):
        bs, c, h, w = x.shape
        patch_size = 2
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        out = self.forward_orig(img, img_ids, context, txt_ids, timestep, y, guidance, control)
        if dist.get_world_size() > 1:
            out_list = [
                torch.empty_like(out) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(out_list, out)
            out = torch.cat(out_list, dim=1)
        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
