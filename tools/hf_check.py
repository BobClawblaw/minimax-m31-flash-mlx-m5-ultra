"""transformers' own MiniMax-M3 text model, its first N layers on the CPU from the release's weights (mxfp8 exact,
nvfp4 experts dequantized), for a check of ref/model.py's reading of it.

  python tools/hf_check.py HF_DIR N --ids @f.json --out f.npy [--index IDX]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import LM, Source, bf16, dequant_mxfp8, dequant_nvfp4  # noqa: E402


def t(x) -> torch.Tensor:
    return torch.from_numpy(np.array(x.astype(__import__("mlx.core").core.float32))).to(torch.bfloat16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("hf")
    p.add_argument("n", type=int)
    p.add_argument("--ids", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--index", default="model.safetensors.index.json")
    a = p.parse_args()
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLForCausalLM

    cfg = json.load(open(os.path.join(a.hf, "config.json")))["text_config"]
    cfg = dict(cfg)
    cfg["num_hidden_layers"] = a.n
    cfg["moe_layer_freq"] = cfg["moe_layer_freq"][: a.n]
    sp = dict(cfg["sparse_attention_config"])
    for k in ("sparse_attention_freq", "sparse_disable_index_value"):
        sp[k] = sp[k][: a.n]
    cfg["sparse_attention_config"] = sp
    cfg["rope_parameters"] = {"rope_type": "default", "rope_theta": cfg["rope_theta"], "partial_rotary_factor": cfg["partial_rotary_factor"]}
    for k in ("architectures", "model_type"):
        cfg.pop(k, None)
    config = MiniMaxM3VLTextConfig(**cfg)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = MiniMaxM3VLForCausalLM(config)
    src = Source(a.hf, a.index)
    sd = {}
    mx8 = lambda n: t(dequant_mxfp8(src.np(n + ".weight"), src.np(n + ".weight_scale_inv")))
    raw = lambda n: torch.from_numpy(np.ascontiguousarray(src.np(n)).astype(np.float32))
    sd["model.embed_tokens.weight"] = t(bf16(src.np(LM + "model.embed_tokens.weight")))
    sd["model.norm.weight"] = t(bf16(src.np(LM + "model.norm.weight")))
    sd["lm_head.weight"] = t(bf16(src.np(LM + "lm_head.weight")))
    for i in range(a.n):
        s, d = f"{LM}model.layers.{i}.", f"model.layers.{i}."
        for n in ("input_layernorm", "post_attention_layernorm"):
            sd[d + n + ".weight"] = t(bf16(src.np(s + n + ".weight")))
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            sd[d + "self_attn." + n + ".weight"] = mx8(s + "self_attn." + n)
        for n in ("q_norm", "k_norm"):
            sd[d + "self_attn." + n + ".weight"] = t(bf16(src.np(s + "self_attn." + n + ".weight")))
        sd[d + "self_attn.indexer.q_proj.weight"] = mx8(s + "self_attn.index_q_proj")
        sd[d + "self_attn.indexer.k_proj.weight"] = mx8(s + "self_attn.index_k_proj")
        sd[d + "self_attn.indexer.q_norm.weight"] = t(bf16(src.np(s + "self_attn.index_q_norm.weight")))
        sd[d + "self_attn.indexer.k_norm.weight"] = t(bf16(src.np(s + "self_attn.index_k_norm.weight")))
        if not cfg["moe_layer_freq"][i]:
            sd[d + "mlp.gate_up_proj.weight"] = torch.cat([mx8(s + "mlp.gate_proj"), mx8(s + "mlp.up_proj")])
            sd[d + "mlp.down_proj.weight"] = mx8(s + "mlp.down_proj")
            continue
        m = s + "block_sparse_moe."
        sd[d + "mlp.gate.weight"] = raw(m + "gate.weight")
        sd[d + "mlp.gate.e_score_correction_bias"] = raw(m + "e_score_correction_bias")
        sh = m + "shared_experts."
        sd[d + "mlp.shared_experts.gate_up_proj.weight"] = torch.cat([mx8(sh + "gate_proj"), mx8(sh + "up_proj")])
        sd[d + "mlp.shared_experts.down_proj.weight"] = mx8(sh + "down_proj")
        E = cfg["num_local_experts"]
        def ex(w):
            g = lambda k: np.stack([src.np(f"{m}experts.{e}.{w}.{k}") for e in range(E)])
            return t(dequant_nvfp4(g("weight_packed"), g("weight_scale"), g("weight_global_scale")))
        sd[d + "mlp.experts.gate_up_proj"] = torch.cat([ex("w1"), ex("w3")], dim=1)
        sd[d + "mlp.experts.down_proj"] = ex("w2")
    sd = {k: (v.float() if k.endswith("mlp.gate.weight") or k.endswith("e_score_correction_bias") else v) for k, v in sd.items()}
    model.load_state_dict(sd, strict=True, assign=True)
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLRotaryEmbedding
    model.model.rotary_emb = MiniMaxM3VLRotaryEmbedding(config)  # its non-persistent table, built off the meta device
    model.eval()
    ids = json.load(open(a.ids[1:])) if a.ids.startswith("@") else [int(x) for x in a.ids.split(",")]
    with torch.no_grad():
        logits = model(torch.tensor([ids]), use_cache=False).logits[0].float().numpy()
    np.save(a.out, logits)
    print("argmax", logits.argmax(-1).tolist())


if __name__ == "__main__":
    main()
