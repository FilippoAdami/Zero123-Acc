import torch

from ldm.modules.encoders.modules import FrozenCLIPImageEmbedder
from clip.model import CLIP


class LazyFrozenCLIPImageEmbedder(FrozenCLIPImageEmbedder):
    """Allowing to instantiate CLIP without immediately loading pretrained model states"""
    def __init__(
            self,
            model='ViT-L/14',
            jit=False,
            device='cpu',
            antialias=False,
            from_pretrained=True,
            clip_constructor_args=None
        ):
        for cls in FrozenCLIPImageEmbedder.__bases__:
             cls.__init__(self)
        if from_pretrained:
            self.model, _ = clip.load(name=model, device=device, jit=jit)
        else:
            # Allowing to instantiate CLIP without loading pretrained model states
            self.model = CLIP(**clip_constructor_args)
        # We don't use the text part so delete it
        del self.model.transformer
        self.antialias = antialias
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
