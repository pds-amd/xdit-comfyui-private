import time
import os
import torch
import folder_paths
import comfy
import latent_preview
import comfy.model_management as mm

from .utils import load_diffusion_model, load_checkpoint_guess_config, any_typ
from xdit_comfyui_private.modules.controlnets.sampling import get_noise, prepare, get_schedule, denoise, denoise_controlnet, unpack
from xdit_comfyui_private.modules.controlnets.utils import LATENT_PROCESSOR_COMFY, ControlNetContainer

dir_xdit = os.path.join(folder_paths.models_dir, "xdit")
os.makedirs(dir_xdit, exist_ok=True)
dir_xdit_loras = os.path.join(dir_xdit, "loras")
os.makedirs(dir_xdit_loras, exist_ok=True)
dir_xdit_controlnets = os.path.join(dir_xdit, "controlnets")
os.makedirs(dir_xdit_controlnets, exist_ok=True)

folder_paths.folder_names_and_paths["xdit"] = ([dir_xdit], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["xdit_loras"] = ([dir_xdit_loras], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["xdit_controlnets"] = ([dir_xdit_controlnets], folder_paths.supported_pt_extensions)

def cleanprint(a):
    # print(a)
    return a

class XDiTUNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                              "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
                            }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "XDiTNodes"

    def load_unet(self, unet_name, weight_dtype):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        model = load_diffusion_model(unet_path, model_options=model_options)
        return (model,)

def print_if_not_empty(a):
    b = list(a.items())
    if len(b)<1:
        return "{}"
    return b[0]

class XDiTFluxLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "lora_name": (cleanprint(folder_paths.get_filename_list("xdit_loras")), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                              }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "loadmodel"
    CATEGORY = "XDiTNodes"

    def loadmodel(self, model, lora_name, strength_model):
        bi = model.clone()
        lora_path = os.path.join(dir_xdit_loras, lora_name)
        bi.lora_cache[lora_path] = strength_model
        return (bi,)

class XDiTSamplerCustomAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"noise": ("NOISE", ),
                    "guider": ("GUIDER", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "latent_image": ("LATENT", ),
                     }
                }

    RETURN_TYPES = ("LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output")

    FUNCTION = "sample"

    CATEGORY = "XDiTNodes"

    def sample(self, noise, guider, sampler, sigmas, latent_image):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        latent["samples"] = latent_image

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        if hasattr(guider.model_patcher.model.diffusion_model, 'clean_lora'):
            guider.model_patcher.model.diffusion_model.clean_lora()
            for lora_path, strength_model in guider.model_patcher.lora_cache.items():
                print(f"Applying Lora: {lora_path} with strength {strength_model}")
                guider.model_patcher.model.diffusion_model.load_lora(lora_path, strength_model)

        samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        
        return (out, out_denoised)

class XDiTSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "model": ("MODEL",),
                    "conditioning": ("CONDITIONING",),
                    "neg_conditioning": ("CONDITIONING",),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT",  {"default": 20, "min": 1, "max": 100}),
                    "timestep_to_start_cfg": ("INT",  {"default": 20, "min": 0, "max": 100}),
                    "true_gs": ("FLOAT",  {"default": 3, "min": 0, "max": 100}),
                    "image_to_image_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                },
            "optional": {
                    "latent_image": ("LATENT", {"default": None}),
                    "controlnet_condition": ("ControlNetCondition", {"default": None}),
                }
            }
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sampling"
    CATEGORY = "XDiTNodes"

    def sampling(self, model, conditioning, neg_conditioning,
                 noise_seed, steps, timestep_to_start_cfg, true_gs,
                 image_to_image_strength, denoise_strength,
                 latent_image=None, controlnet_condition=None
                 ):
        additional_steps = 11 if controlnet_condition is None else 12
        mm.load_model_gpu(model)

        if hasattr(model.model.diffusion_model, 'clean_lora'):
            model.model.diffusion_model.clean_lora()
            for lora_path, strength_model in model.lora_cache.items():
                print(f"Applying Lora: {lora_path} with strength {strength_model}")
                model.model.diffusion_model.load_lora(lora_path, strength_model)        
        
        inmodel = model.model
        #print(conditioning[0][0].shape) #//t5
        #print(conditioning[0][1]['pooled_output'].shape) #//clip
        #print(latent_image['samples'].shape) #// torch.Size([1, 4, 64, 64]) // bc, 4, w//8, h//8
        try:
            guidance = conditioning[0][1]['guidance']
        except:
            guidance = 1.0

        device=mm.get_torch_device()
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        if torch.cuda.is_bf16_supported():
            dtype_model = torch.bfloat16
        else:
            dtype_model = torch.float16
        #dtype_model = torch.bfloat16#model.model.diffusion_model.img_in.weight.dtype
        offload_device=mm.unet_offload_device()

        torch.manual_seed(noise_seed)

        bc, c, h, w = latent_image['samples'].shape
        height = (h//2) * 16
        width = (w//2) * 16

        x = get_noise(
            bc, height, width, device=device,
            dtype=dtype_model, seed=noise_seed
        )
        orig_x = None
        if c==16:
            orig_x=latent_image['samples']
            lat_processor2 = LATENT_PROCESSOR_COMFY()
            orig_x=lat_processor2.go_back(orig_x)
            orig_x=orig_x.to(device, dtype=dtype_model)

        
        timesteps = get_schedule(
            steps,
            (width // 8) * (height // 8) // 4,
            shift=True,
        )
        try:
            inmodel.to(device)
        except:
            pass
        x.to(device)
        
        # inmodel.diffusion_model.to(device)
        inp_cond = prepare(conditioning[0][0], conditioning[0][1]['pooled_output'], img=x)
        neg_inp_cond = prepare(neg_conditioning[0][0], neg_conditioning[0][1]['pooled_output'], img=x)

        if denoise_strength<=0.99:
            try:
                timesteps=timesteps[:int(len(timesteps)*denoise_strength)]
            except:
                pass
        # for sampler preview
        x0_output = {}
        callback = latent_preview.prepare_callback(model, len(timesteps) - 1, x0_output)
        
        
        if controlnet_condition is None:
            x = denoise(
                inmodel.diffusion_model, **inp_cond, timesteps=timesteps, guidance=guidance,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=neg_inp_cond['txt'],
                neg_txt_ids=neg_inp_cond['txt_ids'],
                neg_vec=neg_inp_cond['vec'],
                true_gs=true_gs,
                image2image_strength=image_to_image_strength,
                orig_image=orig_x,
                callback=callback,
                width=width,
                height=height,
            )

        else:
            def prepare_controlnet_condition(controlnet_condition):
                controlnet = controlnet_condition['model']
                controlnet_image = controlnet_condition['img']
                controlnet_image = torch.nn.functional.interpolate(
                    controlnet_image, size=(height, width), scale_factor=None, mode='bicubic',)
                controlnet_strength = controlnet_condition['controlnet_strength']
                controlnet_start = controlnet_condition['start']
                controlnet_end = controlnet_condition['end']
                controlnet.to(device, dtype=dtype_model)
                controlnet_image=controlnet_image.to(device, dtype=dtype_model)
                return {
                    "img": controlnet_image,
                    "controlnet_strength": controlnet_strength,
                    "model": controlnet,
                    "start": controlnet_start,
                    "end": controlnet_end,
                }


            cnet_conditions = [prepare_controlnet_condition(el) for el in controlnet_condition]
            containers = []
            for el in cnet_conditions:
                start_step = int(el['start']*len(timesteps))
                end_step = int(el['end']*len(timesteps))
                container = ControlNetContainer(el['model'], el['img'], el['controlnet_strength'], start_step, end_step)
                containers.append(container)

            mm.load_models_gpu([model,])
            #mm.load_model_gpu(controlnet)

            total_steps = len(timesteps)

            x = denoise_controlnet(
                inmodel.diffusion_model, **inp_cond, 
                controlnets_container=containers,
                timesteps=timesteps, guidance=guidance,
                #controlnet_cond=controlnet_image,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=neg_inp_cond['txt'],
                neg_txt_ids=neg_inp_cond['txt_ids'],
                neg_vec=neg_inp_cond['vec'],
                true_gs=true_gs,
                #controlnet_gs=controlnet_strength,
                image2image_strength=image_to_image_strength,
                orig_image=orig_x,
                callback=callback,
                width=width,
                height=height,
                #controlnet_start_step=start_step,
                #controlnet_end_step=end_step
            )
            #controlnet.to(offload_device)

        x = unpack(x, height, width)
        lat_processor = LATENT_PROCESSOR_COMFY()
        x = lat_processor(x)
        lat_ret = {"samples": x}

        #model.model.to(offload_device)
        return (lat_ret,)

class XDiTCheckpointLoaderSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.", 
                       "The CLIP model used for encoding text prompts.", 
                       "The VAE model used for encoding and decoding images to and from latent space.")
    FUNCTION = "load_checkpoint"

    CATEGORY = "XDiTNodes"
    DESCRIPTION = "Loads a diffusion model checkpoint, diffusion models are used to denoise latents."

    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out[:3]


class XDitCompileModel:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (any_typ,),
                "is_patcher": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
                "object_to_patch": (
                    "STRING",
                    {
                        "default": "diffusion_model",
                    },
                ),
                "compiler": (
                    "STRING",
                    {
                        "default": "torch.compile",
                    }
                ),
                "fullgraph": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
                "dynamic": ("BOOLEAN", {"default": False}),
                "mode": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
                "options": (
                    "STRING",
                    {
                        "multiline": True,
                        # "default": "{}",
                    },
                ),
                "disable": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
                "backend": (
                    "STRING",
                    {
                        "default": "inductor",
                    },
                ),
            }
        }

    RETURN_TYPES = (any_typ,)
    FUNCTION = "patch"

    CATEGORY = "XDiTNodes"

    def patch(
        self,
        model,
        is_patcher,
        object_to_patch,
        compiler,
        fullgraph,
        dynamic,
        mode,
        options,
        disable,
        backend,
    ):
        import importlib
        import json

        import_path, function_name = compiler.rsplit(".", 1)
        module = importlib.import_module(import_path)
        compile_function = getattr(module, function_name)

        mode = mode if mode else None
        options = json.loads(options) if options else None
        
        from xdit_comfyui_private.utils import patch_for_torch_compile
        patch_for_torch_compile()

        if is_patcher:
            patcher = model.clone()
        else:
            patcher = model.patcher
            patcher = patcher.clone()

        patcher.add_object_patch(
            object_to_patch,
            compile_function(
                patcher.get_model_object(object_to_patch),
                fullgraph=fullgraph,
                dynamic=dynamic,
                mode=mode,
                options=options,
                disable=disable,
                backend=backend,
            ),
        )

        if is_patcher:
            return (patcher,)
        else:
            model.patcher = patcher
            return (model,)

NODE_CLASS_MAPPINGS = {
    "XDiTUNETLoader": XDiTUNETLoader,
    "XDiTFluxLoraLoader": XDiTFluxLoraLoader,
    "XDiTSamplerCustomAdvanced": XDiTSamplerCustomAdvanced,
    "XDiTSampler": XDiTSampler,
    "XDiTCheckpointLoaderSimple": XDiTCheckpointLoaderSimple,
    "XDitCompileModel": XDitCompileModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XDiTUNETLoader": "XDiTUNETLoader",
    "XDiTFluxLoraLoader": "XDiTFluxLoraLoader",
    "XDiTSamplerCustomAdvanced": "XDiTSamplerCustomAdvanced",
    "XDiTSampler": "XDiTSampler",
    "XDiTCheckpointLoaderSimple": "XDiTCheckpointLoaderSimple",
    "XDitCompileModel": "XDitCompileModel",
}