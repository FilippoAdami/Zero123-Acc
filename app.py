import math
import os
import fire
import numpy as np
import time
import torch
from contextlib import nullcontext
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import create_carvekit_interface, load_and_preprocess
import omegaconf
from PIL import Image
from rich import print 
from transformers import AutoFeatureExtractor
from torch import autocast
from torchvision import transforms
from ldm.models.diffusion.ddpm import LatentDiffusion

# Global variable to store initialized models
# This is a common pattern with fire when models are heavy to load,
# to avoid re-loading for every command if you were to structure your CLI differently
# For a simple single command call, it still works by loading them once.
_GLOBAL_MODELS = None
_GLOBAL_DEVICE = None

def load_zero123_ld(
    state_dict_path='./checkpoints',
    config_path='./config/latent_diffusion.yml',
    **kwargs
):
    config = omegaconf.OmegaConf.load(config_path)
    # Ensure init_empty_weights is imported and available, it's from accelerate
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    with init_empty_weights():
        model = LatentDiffusion(**config['model']['params'])
    model = load_checkpoint_and_dispatch(model, state_dict_path, **kwargs)
    return model


@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, x, y, z):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([math.radians(x), math.sin(
                math.radians(y)), math.cos(math.radians(y)), z])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            print(f"Sampled DDIM shape: {samples_ddim.shape}") # Using f-string for clarity
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def preprocess_image(models, input_im, preprocess_flag, device):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''
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
    # Assuming 'lo' (lovely_numpy) is imported if you want to use it for debugging
    # print('new input_im:', lo(input_im))
    return input_im


def _initialize_zero123_models(
    device_str: str = 'cuda:0',
    ckpt_path: str = './checkpoints',
    device_map_path: str = None
):
    """
    Initializes and returns a dictionary of the necessary Zero123 models.
    This function is meant to be called internally or once.
    """
    global _GLOBAL_MODELS, _GLOBAL_DEVICE

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
    _GLOBAL_MODELS['carvekit'] = create_carvekit_interface()
    print('Instantiating AutoFeatureExtractor...')
    _GLOBAL_MODELS['clip_fe'] = AutoFeatureExtractor.from_pretrained(
        'CompVis/stable-diffusion-safety-checker')

    print(f"Models initialized on {_GLOBAL_DEVICE}.")
    return _GLOBAL_MODELS, _GLOBAL_DEVICE


def generate_novel_view(
    img_path: str,
    n_steps: int = 75,
    guidance_scale: float = 3.0,
    hor_angle: float = 0.0, # Azimuth angle
    ver_angle: float = 0.0, # Polar angle
    zoom: float = 0.0,      # Radius
    preprocess: bool = True,
    output_height: int = 256,
    output_width: int = 256,
    output_dir: str = 'output_images',
    device: str = 'cuda:0',
    ckpt_path: str = './checkpoints',
    device_map_path: str = None
) -> None:
    """
    Generates a new view of an object from a given input image and saves it.

    Args:
        img_path: Path to the input PIL Image of the object.
        n_steps: Number of diffusion inference steps.
        guidance_scale: Diffusion guidance scale.
        hor_angle: Horizontal rotation (azimuth angle) in degrees (-180 to 180).
        ver_angle: Vertical rotation (polar angle) in degrees (-90 to 90).
        zoom: Relative distance from the center (zoom) in [-0.5, 0.5].
        preprocess: If True, preprocesses the image to remove background and recenter.
        output_height: The height of the output images.
        output_width: The width of the output images.
        output_dir: Directory to save the generated image(s).
        device: The device to run inference on (e.g., 'cuda:0', 'cpu').
        ckpt_path: Path to the directory containing model checkpoints.
        device_map_path: Path to the device map YAML file for accelerate.
    """
    global _GLOBAL_MODELS, _GLOBAL_DEVICE

    # Initialize models if not already initialized
    if _GLOBAL_MODELS is None or _GLOBAL_DEVICE != device:
        _initialize_zero123_models(device_str=device, ckpt_path=ckpt_path, device_map_path=device_map_path)
    models = _GLOBAL_MODELS
    inference_device = _GLOBAL_DEVICE


    torch.cuda.empty_cache()

    # Load image
    try:
        img = Image.open(img_path).convert('RGBA')
        print(f"Loaded input image from: {img_path}")
    except FileNotFoundError:
        print(f"Error: Input image not found at {img_path}")
        return
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return
    
    # Preprocess image
    input_im_array = preprocess_image(models, img, preprocess, inference_device)
    input_im_tensor = transforms.ToTensor()(input_im_array).unsqueeze(0).to(inference_device)
    input_im_tensor = input_im_tensor * 2 - 1
    input_im_tensor = transforms.functional.resize(input_im_tensor, [output_height, output_width])

    # Sample model
    sampler = DDIMSampler(models['turncam'])

    print(f"\nGenerating new view for image '{os.path.basename(img_path)}' with parameters:")
    print(f"  Horizontal Angle: {hor_angle}°")
    print(f"  Vertical Angle: {ver_angle}°")
    print(f"  Zoom: {zoom}")
    print(f"  Steps: {n_steps}, Guidance Scale: {guidance_scale}")

    # The polar angle 'x' in sample_model corresponds to 'ver_angle'
    # The azimuth angle 'y' in sample_model corresponds to 'hor_angle'
    # The zoom 'z' in sample_model corresponds to 'zoom'
    x_samples_ddim = sample_model(
        input_im_tensor,
        models['turncam'],
        sampler,
        precision='fp32', # Hardcoded as in original app.py
        h=output_height,
        w=output_width,
        ddim_steps=n_steps,
        n_samples=1, # This CLI function will generate one image per call
        scale=guidance_scale,
        ddim_eta=1.0, # Hardcoded as in original app.py
        x=ver_angle,
        y=hor_angle,
        z=zoom
    )

    output_ims = []
    for i, x_sample in enumerate(x_samples_ddim):
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

    # Save output images
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    for i, out_img in enumerate(output_ims):
        output_filename = f"{base_name}_H{hor_angle}_V{ver_angle}_Z{zoom}_{i}.png"
        output_path = os.path.join(output_dir, output_filename)
        out_img.save(output_path)
        print(f"Generated image saved to: {output_path}")


if __name__ == '__main__':
    # When using fire.Fire(), it automatically turns your function into a CLI.
    # No need to call _initialize_zero123_models explicitly here,
    # it will be called by generate_novel_view if models are not loaded.
    fire.Fire(generate_novel_view)