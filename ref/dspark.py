"""A reference DSpark drafter (dspark/) in MLX, after TokenSpeed's DSparkMiniMaxDraftModel and omlx's port: the
target's residual after `dspark_target_layer_ids` projected (fc, hidden_norm) into each draft layer's keys and values
over a sliding window; the block [anchor, noise x 6] attends to them and to itself both ways; every block row proposes
through the target head, plus the Markov bias of the token before it, greedily.

  python ref/dspark.py MODEL_DIR --ids @f.json [--from N] [--every K]
Reports, for anchors at positions from N on (every K), how many of a block's drafts match the text's next tokens.
"""

import argparse
import json
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as tm  # noqa: E402

F32, BF16 = mx.float32, mx.bfloat16


class Draft:
    def __init__(self, W: tm.Weights, target: tm.Model):
        c = self.c = W.cfg["dspark"]
        self.t = target
        self.eps = c["rms_norm_eps"]
        g = W.get
        self.fc = W.linear("dspark.fc")
        self.hidden_norm, self.norm = g("dspark.hidden_norm.weight"), g("dspark.norm.weight")
        self.w1, self.w2 = g("dspark.markov_w1.weight"), W.linear("dspark.markov_w2")
        self.layers = []
        for i in range(c["num_hidden_layers"]):
            d = f"dspark.layers.{i}."
            self.layers.append({
                "in": g(d + "input_layernorm.weight"), "post": g(d + "post_attention_layernorm.weight"),
                "qn": g(d + "self_attn.q_norm.weight"), "kn": g(d + "self_attn.k_norm.weight"),
                "qkv": W.linear(d + "self_attn.qkv_proj"), "o": W.linear(d + "self_attn.o_proj"),
                "gu": W.linear(d + "mlp.gate_up_proj"), "down": W.linear(d + "mlp.down_proj")})

    def block(self, taps: mx.array, ctx_pos: list[int], ids: list[int], pos: list[int]) -> list[int]:
        """Greedy drafts from a block whose context is `taps` [n, 5 x hidden] at positions `ctx_pos`."""
        c = self.c
        H, KVH, D = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        ctx = tm.gemma_norm(tm.qmm(taps.astype(BF16), self.fc), self.hidden_norm, self.eps)
        ccos, csin = tm.rope_tables(ctx_pos, c["rotary_dim"], c["rope_theta"])
        bcos, bsin = tm.rope_tables(pos, c["rotary_dim"], c["rope_theta"])
        q_, s_, b_, bits, group = self.t.embed
        r = mx.array(ids)
        x = mx.dequantize(q_[r], s_[r], b_[r], group_size=group, bits=bits).astype(BF16)
        T, N = len(ids), ctx.shape[0]
        for L in self.layers:
            h = tm.gemma_norm(x, L["in"], self.eps)
            q, k, v = mx.split(tm.qmm(h, L["qkv"]), [H * D, (H + KVH) * D], axis=-1)
            q = tm.rope(tm.gemma_norm(q.reshape(T, H, D), L["qn"], self.eps), bcos, bsin)
            k = tm.rope(tm.gemma_norm(k.reshape(T, KVH, D), L["kn"], self.eps), bcos, bsin)
            v = v.reshape(T, KVH, D)
            _, ck, cv = mx.split(tm.qmm(ctx, L["qkv"]), [H * D, (H + KVH) * D], axis=-1)
            ck = tm.rope(tm.gemma_norm(ck.reshape(N, KVH, D), L["kn"], self.eps), ccos, csin)
            keys = mx.concatenate([ck, k]).transpose(1, 0, 2)
            vals = mx.concatenate([cv.reshape(N, KVH, D), v]).transpose(1, 0, 2)
            g_ = H // KVH
            w = (q.transpose(1, 0, 2) @ mx.repeat(keys, g_, axis=0).transpose(0, 2, 1)) * (D**-0.5)
            w = mx.softmax(w.astype(F32), axis=-1).astype(BF16)
            a = (w @ mx.repeat(vals, g_, axis=0)).transpose(1, 0, 2).reshape(T, H * D)
            x = x + tm.qmm(a, L["o"])
            h = tm.gemma_norm(x, L["post"], self.eps)
            x = x + tm.qmm(tm.swiglu_oai(tm.qmm(h, L["gu"]), c["swiglu_alpha"], c["swiglu_limit"]), L["down"])
        logits = tm.qmm(tm.gemma_norm(x, self.norm, self.eps), self.t.head).astype(F32)
        out, prev = [], ids[0]
        for j in range(T):
            bias = tm.qmm(self.w1[prev][None], self.w2)[0].astype(F32)
            prev = int(mx.argmax(logits[j] + bias))
            out.append(prev)
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--ids", required=True)
    p.add_argument("--from", dest="start", type=int, default=64)
    p.add_argument("--every", type=int, default=8)
    a = p.parse_args()
    ids = json.load(open(a.ids[1:]))
    tm.TAPS = {}
    target = tm.Model(a.model)
    c = target.W.cfg["dspark"]
    target.forward(ids)
    taps = mx.concatenate([mx.array(tm.TAPS[i]) for i in c["dspark_target_layer_ids"]], axis=-1)  # [T, 5 x hidden]
    d = Draft(target.W, target)
    n, landed, hist = 0, 0, [0] * (c["dspark_block_size"] + 1)
    for P in range(a.start, len(ids) - c["dspark_block_size"] - 1, a.every):
        lo = max(0, P - c["sliding_window"])
        block = [ids[P]] + [c["dspark_noise_token_id"]] * (c["dspark_block_size"] - 1)
        drafts = d.block(taps[lo:P], list(range(lo, P)), block, list(range(P, P + len(block))))
        k = 0
        while k < len(drafts) and drafts[k] == ids[P + 1 + k]:
            k += 1
        n, landed = n + 1, landed + k
        hist[k] += 1
    print(f"{n} blocks: {landed / n:.2f} drafts land a block (of {c['dspark_block_size']}); by count {hist}")


if __name__ == "__main__":
    main()
