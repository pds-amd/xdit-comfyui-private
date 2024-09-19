# xdit-comfyui-private
## Environment
### Install ComfyUI
Follow the [ComfyUI](https://github.com/comfyanonymous/ComfyUI) repo to install the dependencies.

You can simply install the dependencies by running the following command:
```bash
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
pip install -r requirements.txt
```

### Install [Ray](https://docs.ray.io/en/latest/ray-overview/installation.html)
```bash
pip install -U "ray[data,train,tune,serve]"
```

### Put the xdit-comfyui-private folder into the ComfyUI/custom_nodes folder
You can put the xdit-comfyui-private folder into the ComfyUI/custom_nodes folder by running the following command:
```bash
cd ${ComfyUI}/custom_nodes
git clone git@github.com:xdit-project/xdit-comfyui-private.git
```

## Prepare models checkpoint
Please follow the [Flux Examples] (https://comfyanonymous.github.io/ComfyUI_examples/flux/) to prepare the corresponding checkpoints:
1. Put in the `${ComfyUI}/models/clip` folder: https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main
2. Put in the `${ComfyUI}/models/vae` folder: https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/ae.safetensors
3. Put in the `${ComfyUI}/models/unet` folder: https://huggingface.co/black-forest-labs/FLUX.1-dev

## Run the demo
You can run the demo by running the following command:
```bash
cd ${ComfyUI}
python main.py
```

You can load the default workflow in the xdit-comfyui-private/workflows folder: `xdit-comfyui-flux1-dev.json`
