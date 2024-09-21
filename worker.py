import ray
import torch
from comfy.ldm.flux.model import Flux, FluxParams

class FluxWorker:
    def __init__(self, **kwargs):
        self.world_size = kwargs.pop('world_size', 1)
        self.ulysses_degree = kwargs.pop('ulysses_degree', 1)
        self.ring_degree = kwargs.pop('ring_degree', 1)
        self.rank = self.local_rank = ray.get_gpu_ids()[0]
        print(f'{self.rank=}, {kwargs}')
        self.device = "cuda:0"
        self.flux = Flux(**kwargs).to(self.device)

    def forward(self, x, timestep, context, y, guidance, control=None, **kwargs):
        with torch.no_grad():
            x_worker = x.to(self.device)
            timestep_worker = timestep.to(self.device)
            context_worker = context.to(self.device)
            y_worker = y.to(self.device)
            guidance_worker = guidance.to(self.device)
            configs_worker = {'transformer_options': {'cond_or_uncond': [0], 'sigmas': torch.tensor([1.], device=self.device)}}
            output = self.flux.forward(x_worker, timestep_worker, context_worker, y_worker, guidance_worker, control, **configs_worker)

        return output

    def load_state_dict(self, sd, strict=False):
        m, u = self.flux.load_state_dict(sd, strict=strict)
        return m, u

    def state_dict(self):
        return self.flux.state_dict()

    def execute_method(self, method, *args, **kwargs):
        return getattr(self, method)(*args, **kwargs)
