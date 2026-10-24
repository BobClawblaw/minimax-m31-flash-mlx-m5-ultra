"""A converted checkpoint with its non-expert projections at another width, the routed experts copied as they are.

  python tools/requant_dense.py HF_DIR BASE_DIR OUT_DIR --bits 4 [--head-bits 8] [--index IDX]

Every block's `self_attn.qkv_proj`, `self_attn.o_proj`, the dense layers' and shared experts' gate|up and down are
quantized again from the mxfp8 source (exact in bf16); embeddings, the head (unless --head-bits), norms, routers,
experts and the vision tower are linked from BASE_DIR.
"""

import argparse
import json
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import LM, Source, bf16, dense, quant, stacked  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("hf")
    p.add_argument("base")
    p.add_argument("out")
    p.add_argument("--bits", type=int, required=True)
    p.add_argument("--group", type=int, default=64)
    p.add_argument("--head-bits", type=int, default=0)
    p.add_argument("--index", default="model.safetensors.index.json")
    a = p.parse_args()
    a.dense_bits, a.dense_group = a.bits, a.group
    src = Source(a.hf, a.index)
    os.makedirs(a.out, exist_ok=True)
    index = json.load(open(os.path.join(a.base, "model.safetensors.index.json")))
    cfg = json.load(open(os.path.join(a.base, "config.json")))
    q = cfg["quantization"]
    text = json.load(open(os.path.join(a.hf, "config.json")))["text_config"]
    for f in sorted(set(index["weight_map"].values())):
        dst = os.path.join(a.out, f)
        if os.path.exists(dst):
            continue
        if not f.startswith("model-0") and not (f == "model-head.safetensors" and a.head_bits):
            os.symlink(os.path.join(a.base, f), dst)
            continue
        t = dict(mx.load(os.path.join(a.base, f)))
        if f == "model-head.safetensors":
            for k in ("lm_head.weight", "lm_head.scales", "lm_head.biases"):
                del t[k]
            quant(bf16(src.np(f"{LM}lm_head.weight")), a.head_bits, a.group, t, "lm_head")
            q["lm_head"] = {"bits": a.head_bits, "group_size": a.group}
        else:
            i = int(f[len("model-"):-len(".safetensors")])
            p_, o = f"{LM}model.layers.{i}.", f"model.layers.{i}."
            redo = {o + "self_attn.qkv_proj": [p_ + "self_attn." + n for n in ("q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj")],
                    o + "self_attn.o_proj": [p_ + "self_attn.o_proj"]}
            if text["moe_layer_freq"][i]:
                m = p_ + "block_sparse_moe.shared_experts."
                redo[o + "mlp.shared_experts.gate_up_proj"] = [m + "gate_proj", m + "up_proj"]
                redo[o + "mlp.shared_experts.down_proj"] = [m + "down_proj"]
            else:
                redo[o + "mlp.gate_up_proj"] = [p_ + "mlp.gate_proj", p_ + "mlp.up_proj"]
                redo[o + "mlp.down_proj"] = [p_ + "mlp.down_proj"]
            for to, names in redo.items():
                (stacked if len(names) > 1 else lambda s, n, a_, t_, to_: dense(s, n[0], a_.bits, a_.group, t_, to_))(src, names, a, t, to)
                q[to] = {"bits": a.bits, "group_size": a.group}
        mx.save_safetensors(dst, t, metadata={"format": "mlx"})
        print(f, flush=True)
    cfg["quantization"] = q
    cfg.setdefault("source", {})["block_dense_bits"] = a.bits
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    json.dump(index, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
    for f in os.listdir(a.base):
        if not f.startswith("model") and f != "config.json" and not os.path.lexists(os.path.join(a.out, f)):
            os.symlink(os.path.join(a.base, f), os.path.join(a.out, f))


if __name__ == "__main__":
    main()
