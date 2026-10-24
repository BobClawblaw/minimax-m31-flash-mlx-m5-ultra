"""A reference MiniMax-M3.1-Flash text forward in MLX on the converted checkpoint, following transformers'
modeling_minimax_m3_vl.py (eager attention, bf16 activations): the oracle the native port is checked against.

  python ref/model.py MODEL_DIR logits  --ids 1,2,3 [--out f.npy]   every position's fp32 logits (one pass)
  python ref/model.py MODEL_DIR steps   --ids 1,2,3 --n 32          greedy tokens, one token at a time after the prompt
  python ref/model.py MODEL_DIR forced  --ids 1,2,3 --follow 4,5,6 [--out f.npy]   logits along given tokens, a token a step
MM_LAYERS=N runs the first N blocks only (then the final norm and head), so a check fits beside a served model.
"""

import argparse
import json
import math
import os
import sys

import mlx.core as mx
import numpy as np

F32, BF16 = mx.float32, mx.bfloat16
if os.environ.get("MM_F32"):  # fp32 activations: a check of the bf16 runs' own rounding, not the release's arithmetic
    BF16 = mx.float32


CAPTURE = {} if os.environ.get("MM_CAPTURE") else None
TAPS = None  # ref/dspark.py sets a dict: each layer's residual after it
NVFP4 = None
if os.environ.get("MM_NVFP4"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    from convert import Source as _Source
    NVFP4 = _Source(os.environ["MM_NVFP4"], os.environ.get("MM_NVFP4_INDEX", "model.safetensors.index.json"))


class Weights:
    def __init__(self, d):
        self.dir = d
        self.cfg = json.load(open(os.path.join(d, "config.json")))
        self.where = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.cache = {}
        q = self.cfg["quantization"]
        self.default = (q["bits"], q["group_size"])
        self.per = {k: (v["bits"], v["group_size"]) for k, v in q.items() if isinstance(v, dict)}

    def shard(self, f):
        if f not in self.cache:
            self.cache = {f: mx.load(os.path.join(self.dir, f))}  # one shard at a time: blocks load in order
        return self.cache[f]

    def get(self, name):
        return self.shard(self.where[name])[name]

    def linear(self, base):
        q, s, b = self.get(base + ".weight"), self.get(base + ".scales"), self.get(base + ".biases")
        if base in self.per:
            bits, group = self.per[base]
        else:  # at the default group, the bits follow from the stored shapes (the config may not list a partial run)
            group = self.default[1]
            bits = q.shape[-1] * 32 // (s.shape[-1] * group)
        return (q, s, b, bits, group)


def qmm(x, w):
    q, s, b, bits, group = w
    return mx.quantized_matmul(x, q, s.astype(x.dtype), b.astype(x.dtype), transpose=True, group_size=group, bits=bits)


def gemma_norm(x, w, eps):
    """fp32 rms norm scaled by 1 + w (in fp32), then the input's dtype."""
    xf = x.astype(F32)
    out = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (out * (1.0 + w.astype(F32))).astype(x.dtype)


def rope_tables(positions, rotary_dim, theta):
    inv = 1.0 / (theta ** (mx.arange(0, rotary_dim, 2, dtype=F32) / rotary_dim))
    f = mx.array(positions, dtype=F32)[:, None] * inv[None, :]
    emb = mx.concatenate([f, f], axis=-1)
    return mx.cos(emb).astype(BF16), mx.sin(emb).astype(BF16)


def rope(x, cos, sin):
    """x [T, H, D]; the first cos.shape[-1] dims rotate NeoX-style (halves), the rest pass."""
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    h = r // 2
    rot = mx.concatenate([-xr[..., h:], xr[..., :h]], axis=-1)
    out = xr * cos[:, None, :] + rot * sin[:, None, :]
    return mx.concatenate([out, xp], axis=-1)


def swiglu_oai(gu, alpha, limit):
    gate, up = mx.split(gu, 2, axis=-1)
    gate = mx.minimum(gate, limit)
    up = mx.clip(up, -limit, limit)
    glu = gate * mx.sigmoid(gate * alpha)
    return (up + 1.0) * glu


class Layer:
    def __init__(self, W, i, c):
        p = f"model.layers.{i}."
        self.i = i
        self.in_norm = W.get(p + "input_layernorm.weight")
        self.post_norm = W.get(p + "post_attention_layernorm.weight")
        a = p + "self_attn."
        self.qkv, self.o = W.linear(a + "qkv_proj"), W.linear(a + "o_proj")
        self.q_norm, self.k_norm = W.get(a + "q_norm.weight"), W.get(a + "k_norm.weight")
        self.sparse = bool(c["sparse_attention_config"]["sparse_attention_freq"][i])
        assert self.sparse
        self.iq_norm, self.ik_norm = W.get(a + "index_q_norm.weight"), W.get(a + "index_k_norm.weight")
        self.moe = bool(c["moe_layer_freq"][i])
        m = p + "mlp."
        if self.moe:
            self.router = W.get(m + "gate.weight").astype(F32)
            self.bias = W.get(m + "e_score_correction_bias").astype(F32)
            self.sgu, self.sd = W.linear(m + "shared_experts.gate_up_proj"), W.linear(m + "shared_experts.down_proj")
            self.egu, self.ed = W.linear(m + "switch_mlp.gate_up_proj"), W.linear(m + "switch_mlp.down_proj")
        else:
            self.gu, self.d = W.linear(m + "gate_up_proj"), W.linear(m + "down_proj")
        mx.eval([v for v in vars(self).values() if isinstance(v, mx.array)])
        self.keys = self.values = self.idx_keys = None


class Model:
    def __init__(self, d):
        self.W = W = Weights(d)
        c = self.c = W.cfg
        n = int(os.environ.get("MM_LAYERS", c["num_hidden_layers"]))
        self.eps = c["rms_norm_eps"]
        self.embed = W.linear("model.embed_tokens")
        self.head = W.linear("lm_head")
        self.norm = W.get("model.norm.weight")
        mx.eval(self.norm, *self.embed[:3], *self.head[:3])
        self.layers = [Layer(W, i, c) for i in range(n)]
        sp = c["sparse_attention_config"]
        self.block, self.topk, self.local = sp["sparse_block_size"], sp["sparse_topk_blocks"], sp["sparse_local_block"]
        assert sp.get("sparse_init_block", 0) == 0 and sp.get("sparse_score_type", "max") == "max"
        self.pos = 0

    def reset(self):
        self.pos = 0
        for l in self.layers:
            l.keys = l.values = l.idx_keys = None

    def mlp(self, l, x):
        c = self.c
        a, lim = c["swiglu_alpha"], c["swiglu_limit"]
        if not l.moe:
            return qmm(swiglu_oai(qmm(x, l.gu), a, lim), l.d)
        shared = qmm(swiglu_oai(qmm(x, l.sgu), a, lim), l.sd)
        logits = x.astype(F32) @ l.router.T
        w = mx.sigmoid(logits)
        choice = w + l.bias
        k = c["num_experts_per_tok"]
        idx = mx.argsort(-choice, axis=-1)[:, :k]
        tw = mx.take_along_axis(w, idx, axis=-1)
        tw = tw / tw.sum(-1, keepdims=True)
        # transformers' loop: experts in ascending id, each one's product (bf16 x fp32 weight) added in bf16
        out = mx.zeros_like(x)
        idx_np = np.array(idx)
        if NVFP4 is not None:  # the release's own nvfp4 experts, dequantized exactly per use (MM_NVFP4=HF_DIR)
            return self.exact_experts(l, x, idx_np, tw, a, lim, shared)
        for e in sorted(set(idx_np.reshape(-1).tolist())):
            rows, slot = np.nonzero(idx_np == e)
            xe = x[mx.array(rows)]
            sel = lambda w: tuple(t[e] if j < 3 else t for j, t in enumerate(w))
            h = swiglu_oai(qmm(xe, sel(l.egu)), a, lim)
            y = (qmm(h, sel(l.ed)).astype(F32) * tw[mx.array(rows), mx.array(slot)][:, None]).astype(BF16)
            out = out.at[mx.array(rows)].add(y)
        return out * c["routed_scaling_factor"] + shared

    def exact_experts(self, l, x, idx_np, tw, a, lim, shared):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
        from convert import dequant_nvfp4
        m = f"language_model.model.layers.{l.i}.block_sparse_moe.experts."
        def w(e, n):
            g = lambda k: NVFP4.np(f"{m}{e}.{n}.{k}")
            return dequant_nvfp4(g("weight_packed")[None], g("weight_scale")[None], g("weight_global_scale")[None])[0]
        out = mx.zeros_like(x)
        for e in sorted(set(idx_np.reshape(-1).tolist())):
            rows, slot = np.nonzero(idx_np == e)
            xe = x[mx.array(rows)]
            gu = mx.concatenate([xe @ w(e, "w1").T, xe @ w(e, "w3").T], axis=-1)
            y = (swiglu_oai(gu, a, lim) @ w(e, "w2").T).astype(F32) * tw[mx.array(rows), mx.array(slot)][:, None]
            out = out.at[mx.array(rows)].add(y.astype(BF16))
        return out * self.c["routed_scaling_factor"] + shared

    def attention(self, l, x, cos, sin, positions):
        c = self.c
        T = x.shape[0]
        H, KVH, D = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        IH = c["sparse_attention_config"]["sparse_num_index_heads"]
        ID = c["sparse_attention_config"]["sparse_index_dim"]
        cuts = np.cumsum([H * D, KVH * D, KVH * D, IH * ID]).tolist()
        qq, kk_, vv, iqq, ikk = mx.split(qmm(x, l.qkv), cuts, axis=-1)
        q = gemma_norm(qq.reshape(T, H, D), l.q_norm, self.eps)
        k = gemma_norm(kk_.reshape(T, KVH, D), l.k_norm, self.eps)
        v = vv.reshape(T, KVH, D)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        l.keys = k if l.keys is None else mx.concatenate([l.keys, k])
        l.values = v if l.values is None else mx.concatenate([l.values, v])
        S = l.keys.shape[0]
        kpos = mx.arange(S)
        qpos = mx.array(positions)
        future = kpos[None, :] > qpos[:, None]  # [T, S]
        keep = mx.broadcast_to(~future[None], (KVH, T, S))
        if l.sparse:
            iq = gemma_norm(iqq.reshape(T, IH, ID), l.iq_norm, self.eps)
            ik = gemma_norm(ikk.reshape(T, 1, ID), l.ik_norm, self.eps)
            iq, ik = rope(iq, cos, sin), rope(ik, cos, sin)
            l.idx_keys = ik if l.idx_keys is None else mx.concatenate([l.idx_keys, ik])
            scores = mx.einsum("thd,sd->hts", iq.astype(F32), l.idx_keys[:, 0].astype(F32))
            scores = mx.where(future[None], -mx.inf, scores)
            nb = -(-S // self.block)
            pad = nb * self.block - S
            if pad:
                scores = mx.concatenate([scores, mx.full((IH, T, pad), -mx.inf)], axis=-1)
            bs = scores.reshape(IH, T, nb, self.block).max(-1)
            qb = qpos // self.block
            for j in range(self.local):
                lb = mx.maximum(qb - j, 0)
                bs = mx.where(mx.arange(nb)[None, None, :] == lb[None, :, None], mx.inf, bs)
            kk = min(self.topk, nb)
            order = mx.argsort(-bs, axis=-1)[..., :kk]
            top = mx.take_along_axis(bs, order, axis=-1)
            chosen = mx.zeros((IH, T, nb + 1), dtype=mx.bool_)
            safe = mx.where(top == -mx.inf, nb, order)
            chosen = mx.put_along_axis(chosen, safe, mx.array(True), axis=-1)[..., :nb]
            block_keep = mx.repeat(chosen, self.block, axis=-1)[..., :S]
            assert IH == KVH
            keep = keep & block_keep
        g = H // KVH
        kr = mx.repeat(l.keys.transpose(1, 0, 2), g, axis=0)  # [H, S, D]
        vr = mx.repeat(l.values.transpose(1, 0, 2), g, axis=0)
        qh = q.transpose(1, 0, 2)  # [H, T, D]
        w = (qh @ kr.transpose(0, 2, 1)) * (D**-0.5)
        mask = mx.repeat(keep, g, axis=0)
        w = mx.where(mask, w.astype(F32) + 0.0, mx.finfo(mx.bfloat16).min)
        w = mx.softmax(w.astype(F32), axis=-1).astype(BF16)
        out = (w @ vr).transpose(1, 0, 2).reshape(T, H * D)
        return qmm(out, l.o)

    def forward(self, ids):
        c = self.c
        T = len(ids)
        positions = list(range(self.pos, self.pos + T))
        q, s, b, bits, group = self.embed
        r = mx.array(ids)
        x = mx.dequantize(q[r], s[r], b[r], group_size=group, bits=bits).astype(BF16)
        cos, sin = rope_tables(positions, c["rotary_dim"], c["rope_theta"])
        for l in self.layers:
            h = x + self.attention(l, gemma_norm(x, l.in_norm, self.eps), cos, sin, positions)
            xn = gemma_norm(h, l.post_norm, self.eps)
            if CAPTURE is not None and l.moe:  # a layer's MoE input and its residual, for the requant study
                CAPTURE[f"x{l.i}"], CAPTURE[f"h{l.i}"] = np.array(xn.astype(F32)), np.array(h.astype(F32))
            x = h + self.mlp(l, xn)
            if TAPS is not None:  # the residual after this layer (the DSpark drafter's features)
                TAPS[l.i] = np.array(x.astype(F32))
            mx.eval(x, l.keys, l.values, *( [l.idx_keys] if l.sparse else []))
        self.pos += T
        return qmm(gemma_norm(x, self.norm, self.eps), self.head).astype(F32)


def ints(s):
    return [int(t) for t in s.split(",") if t]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("mode", choices=["logits", "steps", "forced"])
    p.add_argument("--ids", required=True, help="comma-separated token ids, or @file of them")
    p.add_argument("--follow", default="")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--out", default="")
    a = p.parse_args()
    read = lambda s: ints(open(s[1:]).read().strip("[] \n").replace("\n", ",")) if s.startswith("@") else ints(s)
    ids = read(a.ids)
    m = Model(a.model)
    if a.mode == "logits":
        lg = np.array(m.forward(ids))
        if CAPTURE is not None:
            np.savez(os.environ["MM_CAPTURE"], **CAPTURE)
        if a.out:
            np.save(a.out, lg)
        print("argmax", lg.argmax(-1).tolist())
    elif a.mode == "steps":
        lg = m.forward(ids)[-1]
        out = []
        for _ in range(a.n):
            t = int(mx.argmax(lg))
            out.append(t)
            lg = m.forward([t])[-1]
        print(",".join(map(str, out)))
    else:
        follow = read(a.follow)
        rows = [np.array(m.forward(ids)[-1])]
        for t in follow[:-1]:
            rows.append(np.array(m.forward([t])[-1]))
        lg = np.stack(rows)
        if a.out:
            np.save(a.out, lg)
        print("argmax", lg.argmax(-1).tolist())


if __name__ == "__main__":
    sys.exit(main())
