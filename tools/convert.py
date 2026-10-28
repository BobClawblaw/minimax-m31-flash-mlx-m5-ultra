"""MiniMax-M3.1-Flash (HF: mxfp8 projections, nvfp4 routed experts) -> the TensorFold `minimax_m3` checkpoint.

  python tools/convert.py HF_DIR OUT_DIR [--expert-bits 3] [--expert-group 64] [--dense-bits 8]
                          [--layer-bits 3-10:2,50:4] [--only blocks,head,vision,dspark] [--layers 0-3]

Output, all little-endian safetensors (MLX's own naming and affine layout):
- `model-NNNNN.safetensors` + `model.safetensors.index.json`, one shard per block (plus `model-head`, `model-vision`,
  `model-dspark`), the `language_model.` prefix dropped. Projections are MLX affine (`.weight` u32 codes,
  `.scales`/`.biases` bf16); routed experts are stacked per layer as `mlp.switch_mlp.{gate,up,down}_proj` [E, out, in].
  Norms (Gemma's: the forward scales by 1 + w, the stored w unchanged), the fp32 router and its score bias keep their
  checkpoint dtypes.
- `config.json`: the release's own (`model_type` minimax_m3_vl, the language model under `text_config`), its
  `quantization_config` replaced by MLX's `quantization` block (global width = dense width, a key per module that
  differs), with the drafters' `dflash2` and `dspark` blocks and the conversion's `source` beside it.

mxfp8 (e4m3 x 2^k) is exact in bf16; nvfp4 is e2m1 x e4m3 / a global fp32 scale, so its bf16 is the rounded product.
Projections' affine codes come from mx.quantize on that bf16, the experts' from a least-squares refit (refit).
"""

import argparse
import json
import os
import shutil
import struct
import time

import mlx.core as mx
import numpy as np

FP4 = mx.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=mx.float32)
LM = "language_model."
REVISION = "54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e"


class Source:
    """The HF shards by tensor name, as numpy views of the mapped files."""

    NP = {"F8_E4M3": np.uint8, "U8": np.uint8, "BF16": np.uint16, "F32": np.float32}

    def __init__(self, d, index):
        self.dir = d
        self.where = json.load(open(os.path.join(d, index)))["weight_map"]
        self.maps = {}
        self.headers = {}

    def _header(self, f):
        if f not in self.headers:
            with open(os.path.join(self.dir, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n))
            h.pop("__metadata__", None)
            self.headers[f] = (h, 8 + n)
        return self.headers[f]

    def info(self, name):
        return self._header(self.where[name])[0][name]

    def np(self, name):
        f = self.where[name]
        h, base = self._header(f)
        if f not in self.maps:
            self.maps[f] = np.memmap(os.path.join(self.dir, f), dtype=np.uint8, mode="r")
        i = h[name]
        a, b = i["data_offsets"]
        return self.maps[f][base + a : base + b].view(self.NP[i["dtype"]]).reshape(i["shape"])

    def names(self, prefix=""):
        return sorted(n for n in self.where if n.startswith(prefix))

    def ready(self, names):
        return all(os.path.exists(os.path.join(self.dir, self.where[n])) for n in names)


def e8m0(a: np.ndarray) -> mx.array:
    return mx.power(2.0, mx.array(a.astype(np.int32)) - 127).astype(mx.float32)


def bf16(a: np.ndarray) -> mx.array:
    return mx.array(a.view(np.uint16)).view(mx.bfloat16)


def fp8_e4m3(a: np.ndarray) -> mx.array:
    """e4m3fn bytes to fp32 (no infinities; 0x7f/0xff are NaN and never appear in weights)."""
    u = mx.array(a.astype(np.uint32))
    sign = mx.where((u & 0x80) != 0, -1.0, 1.0)
    e = (u >> 3) & 0xF
    m = (u & 0x7).astype(mx.float32)
    normal = (1.0 + m / 8.0) * mx.power(2.0, e.astype(mx.float32) - 7.0)
    sub = (m / 8.0) * (2.0**-6)
    return sign * mx.where(e == 0, sub, normal)


def dequant_mxfp8(w: np.ndarray, s: np.ndarray, block=32) -> mx.array:
    """[out, in] e4m3 with e8m0 scales [out, in/32] -> bf16 (exact)."""
    return (fp8_e4m3(w) * mx.repeat(e8m0(s), block, axis=-1)).astype(mx.bfloat16)


def dequant_nvfp4(w: np.ndarray, s: np.ndarray, g: np.ndarray, block=16) -> mx.array:
    """[..., out, in/2] packed e2m1 (low nibble first), e4m3 scales [..., out, in/16], fp32 global [...] -> bf16."""
    u = mx.array(w)
    vals = mx.stack([FP4[(u & 0x0F).astype(mx.int32)], FP4[(u >> 4).astype(mx.int32)]], axis=-1)
    vals = vals.reshape(*w.shape[:-1], w.shape[-1] * 2)
    scale = fp8_e4m3(s) / mx.array(g.astype(np.float32)).reshape(*g.shape, *([1] * (s.ndim - g.ndim)))
    return (vals * mx.repeat(scale, block, axis=-1)).astype(mx.bfloat16)


def quant(w: mx.array, bits: int, group: int, out: dict, name: str):
    q, s, b = mx.quantize(w, group_size=group, bits=bits)
    out[name + ".weight"], out[name + ".scales"], out[name + ".biases"] = q, s, b
    mx.eval(q, s, b)


def pack(q: mx.array, bits: int) -> mx.array:
    """Codes [..., K] (0 <= q < 2^bits) in MLX's affine words: each 32 codes fill `bits` u32 words, low bits first."""
    c = np.array(q.astype(mx.uint32)).astype(np.uint64)
    k = c.shape[-1]
    c = c.reshape(*c.shape[:-1], k // 32, 32)
    words = np.zeros((*c.shape[:-1], bits), dtype=np.uint64)
    for i in range(32):
        o = i * bits
        j, sh = o // 32, o % 32
        words[..., j] |= (c[..., i] << np.uint64(sh)) & np.uint64(0xFFFFFFFF)
        if sh + bits > 32:
            words[..., j + 1] |= c[..., i] >> np.uint64(32 - sh)
    return mx.array(words.astype(np.uint32).reshape(*c.shape[:-2], k // 32 * bits))


def refit(w: mx.array, bits: int, group: int, iters: int = 8, packed: bool = True):
    """Affine codes, scales and biases by least squares: from min/max, alternate the codes and each group's line fit.

    The affine error of nvfp4 experts at 3 bits falls from 0.196 to 0.179 (relative, layer 3), at 2 bits 0.41 to 0.34.
    """
    x = w.astype(mx.float32).reshape(-1, group)
    n = 2**bits - 1
    lo, hi = x.min(1, keepdims=True), x.max(1, keepdims=True)
    s, b = mx.maximum((hi - lo) / n, 1e-8), lo
    xm = x.mean(1, keepdims=True)
    for _ in range(iters):
        q = mx.clip(mx.round((x - b) / s), 0, n)
        qm = q.mean(1, keepdims=True)
        var = ((q - qm) ** 2).sum(1, keepdims=True)
        fit = ((q - qm) * (x - xm)).sum(1, keepdims=True) / mx.maximum(var, 1e-12)
        s = mx.where((var > 0) & (fit > 1e-8), fit, s)
        b = xm - s * qm
    s, b = s.astype(mx.bfloat16), b.astype(mx.bfloat16)
    q = mx.clip(mx.round((x - b.astype(mx.float32)) / s.astype(mx.float32)), 0, n)
    if not packed:  # the dequantized values alone (a study's), bf16 as the kernels compute them
        return (q * s.astype(mx.float32) + b.astype(mx.float32)).astype(mx.bfloat16).reshape(w.shape)
    shape = (*w.shape[:-1], w.shape[-1] // group)
    return pack(q.reshape(w.shape), bits), s.reshape(shape), b.reshape(shape)


def parse_ranges(spec: str) -> dict[int, str]:
    out = {}
    for part in filter(None, spec.split(",")):
        rng, _, val = part.partition(":")
        lo, _, hi = rng.partition("-")
        for i in range(int(lo), int(hi or lo) + 1):
            out[i] = val or "1"
    return out


def width(spec: str, group: int) -> tuple[int, int]:
    """'4' or '4g128': expert bits and group (the default group when unnamed)."""
    bits, _, g = spec.partition("g")
    return int(bits), int(g) if g else group


class Writer:
    def __init__(self, out_dir):
        self.out = out_dir
        self.weight_map = {}
        index = os.path.join(out_dir, "model.safetensors.index.json")
        if os.path.exists(index):  # a partial run replaces its shards' entries and keeps the others
            self.weight_map = json.load(open(index))["weight_map"]

    def save(self, shard: str, tensors: dict):
        path = os.path.join(self.out, f"model-{shard}.safetensors")
        part = os.path.join(self.out, f"model-{shard}.part.safetensors")
        mx.save_safetensors(part, tensors, metadata={"format": "mlx"})
        os.replace(part, path)
        name = os.path.basename(path)
        self.weight_map = {k: v for k, v in self.weight_map.items() if v != name}
        for n in tensors:
            self.weight_map[n] = name
        self.finish()

    def finish(self):
        files = [os.path.join(self.out, f) for f in set(self.weight_map.values())]
        total = sum(os.path.getsize(f) for f in files if os.path.exists(f))  # a partial folder lists shards made elsewhere
        tmp = os.path.join(self.out, "model.safetensors.index.json.part")
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(self.weight_map.items()))},
                  open(tmp, "w"), indent=1)
        os.replace(tmp, os.path.join(self.out, "model.safetensors.index.json"))


def kept(src: Source, name: str, out: dict, to: str):
    info = src.info(name)
    if info["dtype"] == "BF16":
        out[to] = bf16(src.np(name))
    elif info["dtype"] == "F32":
        out[to] = mx.array(np.ascontiguousarray(src.np(name)))
    else:
        raise SystemExit(f"unhandled tensor {name} {info['dtype']}")


def dense(src: Source, name: str, bits: int, group: int, out: dict, to: str):
    """An mxfp8 projection `name` (.weight + .weight_scale_inv) as affine `to`."""
    quant(dequant_mxfp8(src.np(name + ".weight"), src.np(name + ".weight_scale_inv")), bits, group, out, to)


def stacked(src: Source, names: list[str], a, out: dict, to: str):
    """mxfp8 projections sharing an input, their rows stacked in order as one affine matrix (rows quantize alone)."""
    w = mx.concatenate([dequant_mxfp8(src.np(n + ".weight"), src.np(n + ".weight_scale_inv")) for n in names])
    quant(w, a.dense_bits, a.dense_group, out, to)


def block(src: Source, i: int, text: dict, expert_bits: int, expert_group: int, a) -> tuple[dict, dict]:
    """Layer i's tensors and the quantization entries that differ from the dense width."""
    p = f"{LM}model.layers.{i}."
    q = f"model.layers.{i}."
    t, qc = {}, {}
    for n in ("input_layernorm", "post_attention_layernorm"):
        kept(src, p + n + ".weight", t, q + n + ".weight")
    for n in ("q_norm", "k_norm", "index_q_norm", "index_k_norm"):
        name = p + "self_attn." + n + ".weight"
        if name in src.where:
            kept(src, name, t, q + "self_attn." + n + ".weight")
    stacked(src, [p + "self_attn." + n for n in ("q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj")],
            a, t, q + "self_attn.qkv_proj")
    dense(src, p + "self_attn.o_proj", a.dense_bits, a.dense_group, t, q + "self_attn.o_proj")
    if not text["moe_layer_freq"][i]:
        stacked(src, [p + "mlp.gate_proj", p + "mlp.up_proj"], a, t, q + "mlp.gate_up_proj")
        dense(src, p + "mlp.down_proj", a.dense_bits, a.dense_group, t, q + "mlp.down_proj")
        return t, qc
    m = p + "block_sparse_moe."
    kept(src, m + "gate.weight", t, q + "mlp.gate.weight")
    kept(src, m + "e_score_correction_bias", t, q + "mlp.e_score_correction_bias")
    stacked(src, [m + "shared_experts.gate_proj", m + "shared_experts.up_proj"], a, t, q + "mlp.shared_experts.gate_up_proj")
    dense(src, m + "shared_experts.down_proj", a.dense_bits, a.dense_group, t, q + "mlp.shared_experts.down_proj")
    experts = text["num_local_experts"]
    for ws_, n in ((("w1", "w3"), "gate_up_proj"), (("w2",), "down_proj")):
        codes, scales, biases = [], [], []
        for e0 in range(0, experts, 16):
            es = range(e0, min(e0 + 16, experts))
            parts = []
            for w in ws_:  # gate (w1) rows then up (w3) rows of each expert
                ws = np.stack([src.np(f"{m}experts.{e}.{w}.weight_packed") for e in es])
                ss = np.stack([src.np(f"{m}experts.{e}.{w}.weight_scale") for e in es])
                gs = np.stack([src.np(f"{m}experts.{e}.{w}.weight_global_scale") for e in es])
                parts.append(dequant_nvfp4(ws, ss, gs))
            cq, cs, cb = refit(mx.concatenate(parts, axis=1), expert_bits, expert_group)
            mx.eval(cq, cs, cb)
            codes.append(cq), scales.append(cs), biases.append(cb)
        base = q + "mlp.switch_mlp." + n
        t[base + ".weight"] = mx.concatenate(codes)
        t[base + ".scales"] = mx.concatenate(scales)
        t[base + ".biases"] = mx.concatenate(biases)
        if expert_bits != a.dense_bits or expert_group != a.dense_group:
            qc[base] = {"bits": expert_bits, "group_size": expert_group}
    return t, qc


def dspark(hf: str, a) -> tuple[dict, dict]:
    """The DSpark drafter (dspark/): its five layers' projections affine, q|k|v and gate|up stacked, every layer's k|v
    stacked again as `ctx_kv_proj` (context rows project through all five at once), the Markov tables (w1 a bf16
    lookup, w2 affine), fc affine; norms as stored. The confidence head is not converted (blocks are verified whole)."""
    src = Source(os.path.join(hf, "dspark"), "model.safetensors.index.json")
    cfg = json.load(open(os.path.join(hf, "dspark", "config.json")))["text_config"]
    p, t = LM + "model.dspark.", {}
    quant(bf16(src.np(p + "fc.weight")), a.dense_bits, a.dense_group, t, "dspark.fc")
    for n in ("hidden_norm", "final_norm"):
        kept(src, p + n + ".weight", t, "dspark." + ("norm" if n == "final_norm" else n) + ".weight")
    kept(src, p + "markov_head.markov_w1.weight", t, "dspark.markov_w1.weight")
    quant(bf16(src.np(p + "markov_head.markov_w2.weight")), a.dense_bits, a.dense_group, t, "dspark.markov_w2")
    ctx = []
    for i in range(cfg["num_hidden_layers"]):
        s, d = f"{p}layers.{i}.decoder_layer.", f"dspark.layers.{i}."
        for n in ("input_layernorm", "post_attention_layernorm"):
            kept(src, s + n + ".weight", t, d + n + ".weight")
        for n in ("q_norm", "k_norm"):
            kept(src, s + "self_attn." + n + ".weight", t, d + "self_attn." + n + ".weight")
        stacked(src, [s + "self_attn." + n for n in ("q_proj", "k_proj", "v_proj")], a, t, d + "self_attn.qkv_proj")
        dense(src, s + "self_attn.o_proj", a.dense_bits, a.dense_group, t, d + "self_attn.o_proj")
        stacked(src, [s + "mlp.gate_proj", s + "mlp.up_proj"], a, t, d + "mlp.gate_up_proj")
        dense(src, s + "mlp.down_proj", a.dense_bits, a.dense_group, t, d + "mlp.down_proj")
        ctx += [s + "self_attn.k_proj", s + "self_attn.v_proj"]
    stacked(src, ctx, a, t, "dspark.ctx_kv_proj")
    keep = ("num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "hidden_size", "sliding_window",
            "dspark_block_size", "dspark_noise_token_id", "dspark_markov_rank", "dspark_target_layer_ids",
            "dspark_ffn_hidden_size", "rope_theta", "rotary_dim", "rms_norm_eps", "swiglu_alpha", "swiglu_limit")
    return t, {k: cfg[k] for k in keep}


def dflash2(path: str, a) -> tuple[dict, dict]:
    """incoai's DFlash2 drafter (bf16): its projections affine (q|k|v and gate|up stacked, every layer's k|v again as
    `ctx_kv_proj`), the conv kernels' projections affine beside their bf16 base kernels, the selector's codebooks bf16
    lookups and its hidden projection affine."""
    m = mx.load(os.path.join(path, "model.safetensors"))
    cfg = json.load(open(os.path.join(path, "config.json")))
    t = {}
    q = lambda name, to: quant(m[name], a.dense_bits, a.dense_group, t, to)
    q("fc.weight", "dflash2.fc")
    q("candidate_selector.hidden_projection.weight", "dflash2.hidden_projection")
    for n in ("hidden_norm", "norm"):
        t[f"dflash2.{n}.weight"] = m[f"{n}.weight"]
    for n in ("predecessor_codebook", "successor_codebook"):
        t[f"dflash2.{n}"] = m[f"candidate_selector.{n}"]
    ctx = []
    for i in range(cfg["num_hidden_layers"]):
        s, d = f"layers.{i}.", f"dflash2.layers.{i}."
        for n in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
            t[d + n + ".weight"] = m[s + n + ".weight"]
        quant(mx.concatenate([m[s + f"self_attn.{n}.weight"] for n in ("q_proj", "k_proj", "v_proj")]), a.dense_bits, a.dense_group, t, d + "self_attn.qkv_proj")
        q(s + "self_attn.o_proj.weight", d + "self_attn.o_proj")
        quant(mx.concatenate([m[s + "mlp.gate_proj.weight"], m[s + "mlp.up_proj.weight"]]), a.dense_bits, a.dense_group, t, d + "mlp.gate_up_proj")
        q(s + "mlp.down_proj.weight", d + "mlp.down_proj")
        for c in ("attention_conv", "mlp_conv"):
            t[d + c + ".base_kernel"] = m[s + c + ".base_kernel"]
            q(s + c + ".kernel_projection.weight", d + c + ".kernel_projection")
        ctx += [m[s + "self_attn.k_proj.weight"], m[s + "self_attn.v_proj.weight"]]
    quant(mx.concatenate(ctx), a.dense_bits, a.dense_group, t, "dflash2.ctx_kv_proj")
    d = cfg["dflash_config"]
    keep = {"num_hidden_layers": cfg["num_hidden_layers"], "num_attention_heads": cfg["num_attention_heads"],
            "num_key_value_heads": cfg["num_key_value_heads"], "head_dim": cfg["head_dim"], "hidden_size": cfg["hidden_size"],
            "intermediate_size": cfg["intermediate_size"], "sliding_window": cfg["sliding_window"],
            "rms_norm_eps": cfg["rms_norm_eps"], "rope_theta": cfg["rope_parameters"]["rope_theta"], **d}
    return t, keep


def block_names(src: Source, i: int) -> list[str]:
    return src.names(f"{LM}model.layers.{i}.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("hf")
    p.add_argument("out")
    p.add_argument("--expert-bits", type=int, default=3)
    p.add_argument("--expert-group", type=int, default=64)
    p.add_argument("--dense-bits", type=int, default=8)
    p.add_argument("--dense-group", type=int, default=64)
    p.add_argument("--layer-bits", default="", help="expert widths by layer, e.g. 3-10:2,50:4g128")
    p.add_argument("--layers", default="", help="only these blocks, e.g. 0-3 (default all)")
    p.add_argument("--only", default="", help="comma-separated parts: blocks,head,vision,dspark")
    p.add_argument("--index", default="", help="the weight index, if not yet in HF_DIR (a download in progress)")
    p.add_argument("--dflash2", default="", help="incoai's DFlash2 drafter folder, for --only dflash2")
    p.add_argument("--wait", action="store_true", help="convert blocks as their shards finish downloading")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    src = Source(a.hf, a.index or "model.safetensors.index.json")
    hf_cfg = json.load(open(os.path.join(a.hf, "config.json")))
    text = hf_cfg["text_config"]
    parts = set(a.only.split(",")) if a.only else {"blocks", "head", "vision", "dspark"}
    layer_bits = parse_ranges(a.layer_bits)
    layers = sorted(parse_ranges(a.layers)) if a.layers else list(range(text["num_hidden_layers"]))
    cfg_path = os.path.join(a.out, "config.json")
    old = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    quant_cfg = {k: v for k, v in old.get("quantization", {}).items() if isinstance(v, dict)}
    quant_cfg.update({"group_size": a.dense_group, "bits": a.dense_bits, "mode": "affine"})
    wr = Writer(a.out)
    t0 = time.time()

    if "head" in parts:
        t = {}
        kept(src, f"{LM}model.norm.weight", t, "model.norm.weight")
        quant(bf16(src.np(f"{LM}model.embed_tokens.weight")), a.dense_bits, a.dense_group, t, "model.embed_tokens")
        quant(bf16(src.np(f"{LM}lm_head.weight")), a.dense_bits, a.dense_group, t, "lm_head")
        wr.save("head", t)
        print(f"head {time.time() - t0:.0f}s", flush=True)
    if "vision" in parts:
        t = {}
        for n in src.names("vision_tower.") + src.names("multi_modal_projector.") + src.names("patch_merge_mlp."):
            kept(src, n, t, n)
        wr.save("vision", t)
        print(f"vision {time.time() - t0:.0f}s", flush=True)
    dspark_cfg = None
    if "dspark" in parts and os.path.exists(os.path.join(a.hf, "dspark", "config.json")):
        t, dspark_cfg = dspark(a.hf, a)
        wr.save("dspark", t)
        print(f"dspark {time.time() - t0:.0f}s", flush=True)
    dflash2_cfg = None
    if "dflash2" in parts:
        if not a.dflash2:
            raise SystemExit("--only dflash2 needs --dflash2 DIR (incoai/MiniMax-M3.1-Flash-DFlash2)")
        t, dflash2_cfg = dflash2(a.dflash2, a)
        wr.save("dflash2", t)
        print(f"dflash2 {time.time() - t0:.0f}s", flush=True)
    if "blocks" in parts:
        todo = list(layers)
        while todo:
            ready = [i for i in todo if src.ready(block_names(src, i))]
            if not ready:
                if not a.wait:
                    raise SystemExit(f"blocks {todo} are not downloaded yet (--wait converts them as they land)")
                time.sleep(30)
                continue
            for i in ready:
                bits, group = width(layer_bits.get(i, str(a.expert_bits)), a.expert_group)
                t, qc = block(src, i, text, bits, group, a)
                for k in list(quant_cfg):
                    if k.startswith(f"model.layers.{i}."):
                        del quant_cfg[k]
                quant_cfg.update(qc)
                wr.save(f"{i:05d}", t)
                todo.remove(i)
                kind = f"{bits}-bit g{group} experts" if text["moe_layer_freq"][i] else "dense"
                print(f"layer {i} ({kind}) {time.time() - t0:.0f}s", flush=True)

    # the release's config.json as published, its quantization_config (MXFP8/NVFP4) replaced by MLX's block below
    cfg = {k: v for k, v in hf_cfg.items() if k != "quantization_config"}
    if dflash2_cfg is not None:
        cfg["dflash2"] = dflash2_cfg
    elif "dflash2" in old:
        cfg["dflash2"] = old["dflash2"]
    if dspark_cfg is not None:
        cfg["dspark"] = dspark_cfg
    elif "dspark" in old:
        cfg["dspark"] = old["dspark"]
    cfg["quantization"] = dict(sorted(quant_cfg.items(), key=lambda kv: (isinstance(kv[1], dict), kv[0])))
    src_info = old.get("source", {})
    src_info.update({"repo": "MiniMaxAI/MiniMax-M3.1-Flash", "revision": REVISION, "expert_bits": a.expert_bits,
                     "expert_group": a.expert_group, "dense_bits": a.dense_bits, "dense_group": a.dense_group})
    if a.layer_bits:
        src_info["layer_bits"] = a.layer_bits
    cfg["source"] = src_info
    json.dump(cfg, open(cfg_path + ".part", "w"), indent=1)
    os.replace(cfg_path + ".part", cfg_path)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
              "chat_template.jinja", "generation_config.json", "preprocessor_config.json", "LICENSE"):
        if os.path.exists(os.path.join(a.hf, f)):
            shutil.copyfile(os.path.join(a.hf, f), os.path.join(a.out, f))
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
