"""
A lightweight tool for sharding state dicts that accelerate can read from.
"""

from typing import Callable, Dict, Optional, Union
import json
import os

import torch
from transformers.modeling_utils import shard_checkpoint


def save_sharded(
        save_directory: Union[str, os.PathLike],
        shards: dict,
        index: dict,
        save_function: Callable = torch.save,
        variant: Optional[str] = None,
        weights_name: Optional[str] = None,
        weights_index_name: Optional[str] = None,
):
    """Saves shards and index, outputs of shard_checkpoint(). No safe serialization
    Adapted from https://github.com/huggingface/transformers/blob/04ab5605fbb4ef207b10bf2772d88c53fc242e83/src/transformers/modeling_utils.py#L1837"""
    # Save the model
    for shard_file, shard in shards.items():
        save_function(shard, os.path.join(save_directory, shard_file))

    # Save the index
    save_index_file = os.path.join(save_directory, weights_index_name)
    with open(save_index_file, "w", encoding="utf-8") as f:
        content = json.dumps(index, indent=2, sort_keys=True) + "\n"
        f.write(content)


if __name__ == '__main__':
    max_shard_size = "5GB"
    pl_sd = torch.load('your_local_path_to/105000.ckpt', map_location="cpu")

    save_dir = 'your_local_destination_path/105000-5GB.ckpt/'
    weights_name = 'shard.bin'
    shards, index = shard_checkpoint(
        state_dict=pl_sd['state_dict'], max_shard_size=max_shard_size, weights_name=weights_name
    )
    save_sharded(
        save_dir,
        shards,
        index,
        weights_name=weights_name,
        weights_index_name='.index.json'
    )
