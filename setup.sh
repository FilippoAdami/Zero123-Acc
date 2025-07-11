#!/bin/bash

set -e  # Exit if any command fails

echo "🚧 Uninstalling conflicting packages..."
pip uninstall -y sentence-transformers torchaudio peft jax jaxlib flax pydantic || true

echo "📦 Installing required dependencies..."
pip install requirements.txt --no-cache-dir

echo "📁 Cloning Zero123-accelerate repo..."
git clone https://github.com/FilippoAdami/Zero123-Acc
cd /content/Zero123-Acc

echo "🔁 Resetting submodules to known good commits..."
cd taming-transformers && git reset --hard 3ba01b2 && cd ..
cd CLIP && git reset --hard a9b1bf5 && cd ..
cd image-background-remove-tool && git reset --hard 2935e46 && cd ..
cd zero123 && git reset --hard 78bc429 && cd ..

echo "🩹 Applying patch files..."
cd ./zero123/zero123/ldm/models/diffusion && patch < ../../../../../patches/ldm_ddpm.patch && cd -
cd ./zero123/zero123/ldm && patch < ../../../patches/ldm_util.patch && cd -

echo "📦 Downloading model checkpoints..."
mkdir -p checkpoints
wget -q https://huggingface.co/learningdisorder/zero123/resolve/main/zero-123-sharded-5gb/shard-00001-of-00002.bin -P checkpoints
wget -q https://huggingface.co/learningdisorder/zero123/resolve/main/zero-123-sharded-5gb/shard-00002-of-00002.bin -P checkpoints
wget -q https://huggingface.co/learningdisorder/zero123/resolve/main/zero-123-sharded-5gb/.index.json -P checkpoints

echo "✅ Setup complete!"