import math
import os
import numpy as np
import time
import torch
import gc
from contextlib import nullcontext
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import load_and_preprocess
import omegaconf
from PIL import Image
from rich import print 
from transformers import AutoFeatureExtractor
from torch import autocast
from torchvision import transforms
from ldm.models.diffusion.ddpm import LatentDiffusion

_GLOBAL_MODELS = None
_GLOBAL_DEVICE = None
_GLOBAL_SAMPLER = None

def load_zero123_ld(state_dict_path='./checkpoints', config_path='./config/latent_diffusion.yml', **kwargs):
    config = omegaconf.OmegaConf.load(config_path)
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    with init_empty_weights():
        model = LatentDiffusion(**config['model']['params'])
    model = load_checkpoint_and_dispatch(model, state_dict_path, **kwargs)
    return model

@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta, x, y, z):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            device = input_im.device
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([
                math.radians(x),
                math.sin(math.radians(y)),
                math.cos(math.radians(y)),
                z
            ], device=device)[None, None, :].repeat(n_samples, 1, 1)

            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)

            cond = {
                'c_crossattn': [c],
                'c_concat': [model.encode_first_stage(input_im).mode().detach().repeat(n_samples, 1, 1, 1)]
            }

            uc = None
            if scale != 1.0:
                uc = {
                    'c_concat': [torch.zeros(n_samples, 4, h // 8, w // 8, device=device)],
                    'c_crossattn': [torch.zeros_like(c, device=device)]
                }

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(
                S=ddim_steps,
                conditioning=cond,
                batch_size=n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
                eta=ddim_eta,
                x_T=None
            )

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            del samples_ddim, cond, uc, c, T
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()

def preprocess_image(models, input_im, preprocess_flag, device):
    print(f'Original input_im size: {input_im.size}')
    start_time = time.time()

    if preprocess_flag:
        input_im = load_and_preprocess(models['carvekit'], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
    else:
        input_im = input_im.resize([256, 256], Image.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im
        input_im = input_im[:, :, 0:3]

    print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
    return input_im

def _initialize_zero123_models(device_str: str = 'cuda:0', ckpt_path: str = './checkpoints', device_map_path: str = None):
    global _GLOBAL_MODELS, _GLOBAL_DEVICE, _GLOBAL_SAMPLER

    if _GLOBAL_MODELS is not None:
        print("Models already initialized. Skipping re-initialization.")
        return _GLOBAL_MODELS, _GLOBAL_DEVICE

    print("Initializing Zero123 models...")
    _GLOBAL_MODELS = dict()
    _GLOBAL_DEVICE = device_str

    device_map = None
    if device_map_path and os.path.isfile(device_map_path):
        device_map = omegaconf.OmegaConf.load(device_map_path)

    _GLOBAL_MODELS['turncam'] = load_zero123_ld(
        state_dict_path=ckpt_path,
        device_map=device_map,
        offload_folder='/tmp'
    )
    _GLOBAL_SAMPLER = DDIMSampler(_GLOBAL_MODELS['turncam'])
    print('Instantiating AutoFeatureExtractor...')
    _GLOBAL_MODELS['clip_fe'] = AutoFeatureExtractor.from_pretrained('CompVis/stable-diffusion-safety-checker')

    print(f"Models initialized on {_GLOBAL_DEVICE}.")
    return _GLOBAL_MODELS, _GLOBAL_DEVICE

def generate_novel_views(
        img_path: str,
        n_steps: int = 45,
        guidance_scale: float = 3.0,
        zoom: float = 0.0,
        preprocess: bool = True,
        output_height: int = 256,
        output_width: int = 256,
        device: str = 'cuda:0',
        ckpt_path: str = './checkpoints',
        device_map_path: str = None
    ) -> list[Image.Image]:

    global _GLOBAL_MODELS, _GLOBAL_DEVICE, _GLOBAL_SAMPLER

    if _GLOBAL_MODELS is None or _GLOBAL_DEVICE != device:
        _initialize_zero123_models(device_str=device, ckpt_path=ckpt_path, device_map_path=device_map_path)

    models = _GLOBAL_MODELS
    inference_device = _GLOBAL_DEVICE
    sampler = _GLOBAL_SAMPLER

    try:
        img = Image.open(img_path).convert('RGBA')
        print(f"Loaded input image from: {img_path}")
    except FileNotFoundError:
        print(f"Error: Input image not found at {img_path}")
        return []
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return []

    input_im_array = preprocess_image(models, img, preprocess, inference_device)
    input_im_tensor = torch.from_numpy(input_im_array).permute(2, 0, 1).unsqueeze(0).to(inference_device)
    input_im_tensor = input_im_tensor * 2 - 1
    input_im_tensor = transforms.functional.resize(input_im_tensor, [output_height, output_width])

    # Define the 4 target angles
    view_angles = [(-30, 0), (30, 0), (0, -30), (0, 30)]
    output_ims = []

    with torch.no_grad():
        for ver_angle, hor_angle in view_angles:
            print(f"Generating view: hor_angle={hor_angle}, ver_angle={ver_angle}, zoom={zoom}")
            x_samples_ddim = sample_model(
                input_im_tensor,
                models['turncam'],
                sampler,
                precision='fp32',
                h=output_height,
                w=output_width,
                ddim_steps=n_steps,
                n_samples=1,
                scale=guidance_scale,
                ddim_eta=1.0,
                x=ver_angle,
                y=hor_angle,
                z=zoom
            )
            for x_sample in x_samples_ddim:
                x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

    del input_im_tensor, x_samples_ddim
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return output_ims

if __name__ == '__main__':
    fire.Fire(generate_novel_views)
