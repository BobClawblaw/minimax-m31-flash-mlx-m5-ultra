"""transformers' MiniMax-M3 image tower and projector on CPU: one image's patches, grid and text rows.

The release's own image processor (image_processor.py, which preprocessor_config.json names) prepares the image, and
transformers' MiniMaxM3VLVisionModel and MiniMaxM3VLMultiModalProjector run it with the source checkpoint's weights.

    python tools/vision_ref.py SRC_DIR IMAGE OUT_DIR

writes OUT_DIR/pixels.npy ([P, 1176] fp32, merge order), grid.npy ([t, h, w]), rows_bf16.npy and rows_f32.npy
([P / 4, 6144] fp32: the tower in bf16 with an fp32 patch conv, as transformers runs it, and all in fp32).
"""

import importlib.util
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers.models.minimax_m3_vl import configuration_minimax_m3_vl as conf
from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl as mod

RENAME = [
    ("vision_tower.vision_model.embeddings.patch_embedding.", "tower.embeddings.proj."),
    ("vision_tower.vision_model.encoder.layers.", "tower.layers."),
    ("vision_tower.vision_model.", "tower."),
    ("multi_modal_projector.linear_", "projector.linear_"),
    ("patch_merge_mlp.linear_", "projector.merge_linear_"),
]


def processor(src):
    spec = importlib.util.spec_from_file_location("mm_image_processor", os.path.join(src, "image_processor.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.MiniMaxM3VLImageProcessor()


def weights(src):
    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for name, shard in index.items():
        for old, new in RENAME:
            if name.startswith(old):
                with safe_open(os.path.join(src, shard), "pt") as f:
                    out[new + name[len(old):]] = f.get_tensor(name)
                break
    return out


def main():
    src, image, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    config = conf.MiniMaxM3VLConfig.from_pretrained(src)
    batch = processor(src)(images=[Image.open(image).convert("RGB")], return_tensors="pt")
    pixels, grid = batch["pixel_values"], batch["image_grid_thw"]
    np.save(os.path.join(out, "pixels.npy"), pixels.float().numpy())
    np.save(os.path.join(out, "grid.npy"), grid[0].numpy())
    w = weights(src)
    for tag, dtype in (("f32", torch.float32), ("bf16", torch.bfloat16)):
        config.vision_config._attn_implementation = "sdpa"
        tower = mod.MiniMaxM3VLVisionModel(config.vision_config).to(dtype).eval()
        projector = mod.MiniMaxM3VLMultiModalProjector(config).to(dtype).eval()
        tower.load_state_dict({k[len("tower."):]: v for k, v in w.items() if k.startswith("tower.")}, strict=True)
        projector.load_state_dict({k[len("projector."):]: v for k, v in w.items() if k.startswith("projector.")}, strict=True)
        tower.embeddings.proj.float()  # the checkpoint's conv is fp32, and transformers runs it so
        with torch.no_grad():
            hidden = tower(pixel_values=pixels, grid_thw=grid).last_hidden_state.squeeze(0)
            rows = projector(hidden.to(dtype))
        np.save(os.path.join(out, f"rows_{tag}.npy"), rows.float().numpy())
        print(tag, "grid", grid[0].tolist(), "rows", tuple(rows.shape), "mean |row|", float(rows.float().abs().mean()))


if __name__ == "__main__":
    main()
