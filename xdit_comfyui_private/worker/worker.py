import ray
import torch
import torch.distributed as dist
from xdit_comfyui_private.model.flux.flux import xFuserFlux
from xdit_comfyui_private.distributed.parallel_state import init_distributed_enviroment, init_model_parallel

class FluxWorker:
    modules_to_convert = []

    def __init__(self, **kwargs):
        self.world_size = kwargs.pop('world_size', 1)
        self.ulysses_degree = kwargs.pop('ulysses_degree', 1)
        self.ring_degree = kwargs.pop('ring_degree', 1)
        self.rank = self.local_rank = ray.get_gpu_ids()[0]
        self.device = "cuda:0"

        init_distributed_enviroment(kwargs.pop('distributed_init_method', 'env://'), self.world_size, self.rank)
        init_model_parallel(self.ulysses_degree, self.ring_degree, self.rank, self.world_size)
        self.flux = xFuserFlux(**kwargs).to(self.device)


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

    # call after all the weight are loaded(including LoRa)
    def parallelize_model(self):
        self._parallelize_flux_model()
        self.flux.to(self.device)

    def state_dict(self):
        return self.flux.state_dict()

    def execute_method(self, method, *args, **kwargs):
        return getattr(self, method)(*args, **kwargs)

    def _parallelize_flux_model(self):
        
        pass
