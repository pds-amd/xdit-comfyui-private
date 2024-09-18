import ray
import torch
from comfy.ldm.flux.model import Flux, FluxParams

@ray.remote(num_gpus=1)
class FluxWorker:
    def __init__(self, **kwargs):
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