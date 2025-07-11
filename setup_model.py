import os
import sys
from app import _initialize_zero123_models

ZERO123_ACC_REPO_PATH = '/content/Zero123-Acc'
CONFIG_PATH = os.path.join(ZERO123_ACC_REPO_PATH, 'config', 'latent_diffusion.yml')
DEVICE_MAP_PATH = os.path.join(ZERO123_ACC_REPO_PATH, 'config', 'device_map.yml')
CHECKPOINTS_PATH = os.path.join(ZERO123_ACC_REPO_PATH, 'checkpoints')
OUTPUT_BASE_DIR = '/content/generated_views'

desired_paths = [
    ZERO123_ACC_REPO_PATH,
    os.path.join(ZERO123_ACC_REPO_PATH, 'zero123', 'zero123'),
    os.path.join(ZERO123_ACC_REPO_PATH, 'taming-transformers'),
    os.path.join(ZERO123_ACC_REPO_PATH, 'CLIP'),
    os.path.join(ZERO123_ACC_REPO_PATH, 'image-background-remove-tool'),
]

# Defensive cleaning
sys.path = [p for p in sys.path if p not in desired_paths]
for p in reversed(desired_paths):
    if p not in sys.path:
        sys.path.insert(0, p)

print("✅ Python path setup complete.")


# --- Initialize models ---
INFERENCE_DEVICE = 'cuda:0'
print(f"⚙️ Initializing models on {INFERENCE_DEVICE}...")

_initialize_zero123_models(
    device_str=INFERENCE_DEVICE,
    ckpt_path=CHECKPOINTS_PATH,
    device_map_path=DEVICE_MAP_PATH
)

print("✅ Models and sampler are loaded and ready.")
