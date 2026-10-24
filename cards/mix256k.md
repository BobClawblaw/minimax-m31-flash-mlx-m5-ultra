---
license: other
license_name: minimax-community
license_link: LICENSE
base_model: MiniMaxAI/MiniMax-M3.1-Flash
library_name: tensorfold
tags:
- tensorfold
- mlx
- apple-silicon
- minimax
---

# MiniMax-M3.1-Flash, mixed 3-bit sized for a 256k window (mix256k)

[MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) (revision `54e4e69d`)
converted for [TensorFold](https://github.com/ashhart/TensorFold)'s native Metal engine (its `minimax_m3` family), to
serve on one Apple M5 Ultra with 256 GB. Sized so a 262,144-token window fits beside it on 256 GB with images, the drafter and an int8 key/value cache; mix34 trades that window for quality.

**Status:** the engine support is not published yet (a TensorFold pull request and a serving recipe follow). This
checkpoint does not load in mlx-lm or mlx-vlm: the layout is MLX affine quantization, but the model is TensorFold's.

## What is in it

- **Routed experts:** 4-bit affine, groups of 128, on layers 22, 24 and 59 (the three that gain most); 3-bit affine, groups of 64, on the other 57. Each width is a least-squares refit of the release's NVFP4 experts (scales and
  biases fitted to the NVFP4 values, not a plain round-to-nearest).
- **Attention (q, k, v and the lightning indexer's projections), shared expert, dense layers, embeddings and the
  output head:** 8-bit affine, groups of 64, from the release's MXFP8 (exact in bf16 before quantizing).
- **Router:** fp32, as released.
- **Vision tower and projector** (`model-vision.safetensors`): bf16 as released (the patch convolution fp32).
- **Drafters:** the release's own DSpark head (`model-dspark.safetensors`) and
  [incoai/MiniMax-M3.1-Flash-DFlash2](https://huggingface.co/incoai/MiniMax-M3.1-Flash-DFlash2) (revision
  `ac07a7c1`, Apache-2.0, `model-dflash2.safetensors`), its projections in the same affine layout.
- About 189 GiB in all (DSpark and DFlash2 included; the engine loads one drafter).

## Quality

Next-token KL divergence and top-1 agreement against the release's own NVFP4 weights (an MLX reference of
transformers' modeling code), over the 1,024 positions of one chat, the engine's default int8 key/value cache:

| Checkpoint | Mean KL | Top-1 agreement |
|---|---|---|
| mix34 | 0.250 | 80.0% |
| mix256k | 0.284 | 78.2% |
| all experts 3-bit (not published) | 0.300 | 77.3% |

No benchmark suite has been run on it yet; the recipe will report one.

## License

The [MiniMax Community License](LICENSE): free for non-commercial use; commercial use needs the attribution and the
notice or authorization it describes. The DFlash2 drafter's weights are incoai's, under Apache-2.0.
