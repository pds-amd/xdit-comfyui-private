import time
import os
import torch
import folder_paths

from .utils import load_diffusion_model


def cleanprint(a):
    print(a)
    return a

class XfuserUNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                              "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
                            }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "XDiT"

    def load_unet(self, unet_name, weight_dtype):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        model = load_diffusion_model(unet_path, model_options=model_options)
        return (model,)

NODE_CLASS_MAPPINGS = {
    "XfuserUNETLoader": XfuserUNETLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XfuserUNETLoader": "Load Diffusion Model(Xfuser)",
}
