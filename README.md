## Zero-shot One Image to 3D Object on Accelerate

[Demo](https://zero123.cs.columbia.edu/), which originally required 22GB of VRAM, now got leaner and can actually run in Colab on a single Tesla T4. This means you can explore large-scale diffusion models for novel view generation or 3D object reconstruction on less expensive hardware.

For an overall reduced memory footprint, inference is done with [Accelerate](https://huggingface.co/docs/accelerate/), using sharded checkpoints of the underlying large-scale diffusion model.

[Colab notebook](https://colab.research.google.com/drive/1iNpZqSlu8SMaDMVXLxp8a6jtwPB7LsUJ)

[Sharded model checkpoints](https://huggingface.co/learningdisorder/zero123/tree/main)
