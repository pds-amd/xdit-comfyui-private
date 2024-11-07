import torch
import numpy as np

class ControlNetContainer:
    def __init__(
            self, controlnet, controlnet_cond, 
            controlnet_gs, controlnet_start_step,
            controlnet_end_step,
            
            ):
        self.controlnet_cond = controlnet_cond
        self.controlnet_gs = controlnet_gs
        self.controlnet_start_step = controlnet_start_step
        self.controlnet_end_step = controlnet_end_step
        self.controlnet = controlnet

class LATENT_PROCESSOR_COMFY:
    def __init__(self):
        self.scale_factor = 0.3611
        self.shift_factor = 0.1159
        self.latent_rgb_factors =[
                    [-0.0404,  0.0159,  0.0609],
                    [ 0.0043,  0.0298,  0.0850],
                    [ 0.0328, -0.0749, -0.0503],
                    [-0.0245,  0.0085,  0.0549],
                    [ 0.0966,  0.0894,  0.0530],
                    [ 0.0035,  0.0399,  0.0123],
                    [ 0.0583,  0.1184,  0.1262],
                    [-0.0191, -0.0206, -0.0306],
                    [-0.0324,  0.0055,  0.1001],
                    [ 0.0955,  0.0659, -0.0545],
                    [-0.0504,  0.0231, -0.0013],
                    [ 0.0500, -0.0008, -0.0088],
                    [ 0.0982,  0.0941,  0.0976],
                    [-0.1233, -0.0280, -0.0897],
                    [-0.0005, -0.0530, -0.0020],
                    [-0.1273, -0.0932, -0.0680]
                ]
    def __call__(self, x):
        return (x / self.scale_factor) + self.shift_factor
    def go_back(self, x):
        return (x - self.shift_factor) * self.scale_factor
    
def optimized_tensor_to_numpy(tensor):        
    cpu_tensor = torch.empty(tensor.shape,
                           dtype=torch.float32,
                           pin_memory=True)
    cpu_tensor.copy_(tensor.to(torch.float32))
    
    torch.cuda.synchronize()
    
    return cpu_tensor.numpy()