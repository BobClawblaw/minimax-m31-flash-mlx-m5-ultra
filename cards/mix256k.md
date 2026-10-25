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

# MiniMax-M3.1-Flash, mixed 3-bit for a long window (mix256k)

[MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) (revision `54e4e69d`)
converted for [TensorFold](https://github.com/ashhart/TensorFold)'s native Metal engine (its `minimax_m3` family), to
serve on one Apple M5 Ultra with 256 GB. Sized for the longest window: beside it, the exact bf16 key/value cache fits 150,400 tokens on 256 GB with images and the drafter (the recipe serves 147,456; the name is from the 262,144 an earlier int8 cache gave it). mix34 trades that window for quality.

**Serving:** the engine's `minimax_m3` family is in
[ashhart/TensorFold#1235](https://github.com/ashhart/TensorFold/pull/1235); the serving recipe, with its launcher,
settings and measurements, is [BobClawblaw/minimax-m31-flash-mlx-m5-ultra](https://github.com/BobClawblaw/minimax-m31-flash-mlx-m5-ultra).
This checkpoint does not load in mlx-lm or mlx-vlm: the layout is MLX affine quantization, but the model is
TensorFold's.

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
transformers' modeling code), over the 1,024 positions of one chat, with the engine's exact bf16 key/value cache:

| Checkpoint | Mean KL | Top-1 agreement |
|---|---|---|
| mix34 | 0.246 | 81.1% |
| mix256k | 0.301 | 78.3% |

Through the recipe harness's suite, thinking off, with the engine's bf16 cache: GSM8K 0.976, MGSM 0.821, MMLU 0.819, MATH-500 0.900, IFEval 89/150 (loose), tools 56/60, JSON 30/30 (with the earlier int8 cache: MGSM 0.875, MATH-500 0.908, IFEval 86/150, tools 55/60, the rest the same). Speeds, the other variant's numbers and how they were measured are in the
[recipe's README](https://github.com/BobClawblaw/minimax-m31-flash-mlx-m5-ultra#measured).

## License

The [MiniMax Community License](LICENSE): free for non-commercial use; commercial use needs the attribution and the
notice or authorization it describes. The DFlash2 drafter's weights are incoai's, under Apache-2.0.
