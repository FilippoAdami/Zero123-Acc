#!/bin/bash

pin_and_add_path() {
  local module="$1"
  local commit_sha="$2"
  local output=$(cd "$module" && git reset --hard "$commit_sha" && realpath . 2>&1)
  local module_path=$(echo "$output" | tail -n 1)
  if [ $? -eq 0 ]; then
    export PYTHONPATH="${PYTHONPATH}:${module_path}"
    printf "Added %s with commit %s to PYTHONPATH\n" "$module" "$commit_sha"
  else
    printf "Error: %s\n" "module_path"
  fi
}

# Add submodules to PYTHONPATH. Pin to specific commits.
pin_and_add_path "taming-transformers" "3ba01b2"
pin_and_add_path "CLIP" "a9b1bf5"
pin_and_add_path "image-background-remove-tool" "2935e46"
#
pin_and_add_path "zero123" "78bc429"
# Path to the vendored ldm code on the original zero123 repo
zero123_path=$(echo $PYTHONPATH | awk -F: '{print $NF}')
export PYTHONPATH="${PYTHONPATH}:${zero123_path}/zero123"

# One requirements file to rule them all
pip install -r requirements.txt

# Patching existing code - don't fork or vendor just for a couple of lines
#
# Make zero123's ldm source compatible with newer versions of lightning
(cd ./zero123/zero123/ldm/models/diffusion && patch < ../../../../../patches/ldm_ddpm.patch)
# Compatibility with newer PIL
(cd ./zero123/zero123/ldm && patch < ../../../patches/ldm_util.patch)

echo $PYTHONPATH
