"""Which layers' experts gain most from 4 bits: each routed layer's output on captured inputs (ref/model.py's
MM_CAPTURE), the release's nvfp4 against 3-bit (g64) and 4-bit (g128) refits, the error over the residual's norm.

  python tools/requant_study.py HF_DIR ACTS.npz OUT.json
"""

import json
import sys
import os

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import LM, Source, dequant_nvfp4, refit  # noqa: E402


def main():
    src = Source(sys.argv[1], "model.safetensors.index.json")
    acts = np.load(sys.argv[2])
    cfg = json.load(open(os.path.join(sys.argv[1], "config.json")))["text_config"]
    out = {}
    for i in range(cfg["num_hidden_layers"]):
        if f"x{i}" not in acts:
            continue
        x = mx.array(acts[f"x{i}"]).astype(mx.bfloat16)
        h = mx.array(acts[f"h{i}"])
        m = f"{LM}model.layers.{i}.block_sparse_moe."
        logits = x.astype(mx.float32) @ mx.array(src.np(m + "gate.weight")).T
        w = mx.sigmoid(logits)
        idx = np.array(mx.argsort(-(w + mx.array(src.np(m + "e_score_correction_bias"))), axis=-1)[:, :4])
        tw = np.array(mx.take_along_axis(w, mx.array(idx), axis=-1))
        tw = tw / tw.sum(-1, keepdims=True)
        ys = {k: mx.zeros(x.shape, dtype=mx.float32) for k in ("nv", "q3", "q4")}
        for e in sorted(set(idx.reshape(-1).tolist())):
            rows, slot = np.nonzero(idx == e)
            xe = x[mx.array(rows)]
            g = lambda n, k: src.np(f"{m}experts.{e}.{n}.{k}")[None]
            W = {n: dequant_nvfp4(g(n, "weight_packed"), g(n, "weight_scale"), g(n, "weight_global_scale"))[0] for n in ("w1", "w3", "w2")}
            for k, (bits, group) in (("nv", (None, None)), ("q3", (3, 64)), ("q4", (4, 128))):
                if bits is None:
                    V = W
                else:
                    V = {}
                    for n, M in W.items():
                        V[n] = refit(M, bits, group, packed=False)
                gate, up = xe @ V["w1"].T, xe @ V["w3"].T
                gate, up = mx.minimum(gate, 7.0), mx.clip(up, -7.0, 7.0)
                y = ((up + 1) * (gate * mx.sigmoid(gate * 1.702))) @ V["w2"].T
                ys[k] = ys[k].at[mx.array(rows)].add(y.astype(mx.float32) * mx.array(tw[rows, slot])[:, None] * 2.0)
        hn = mx.sqrt(mx.sum(h * h, axis=-1))
        err = {k: float(mx.mean(mx.sqrt(mx.sum((ys[k] - ys["nv"]) ** 2, axis=-1)) / hn)) for k in ("q3", "q4")}
        out[i] = err
        print(i, "3-bit %.4f  4-bit %.4f  gain %.4f" % (err["q3"], err["q4"], err["q3"] - err["q4"]), flush=True)
    json.dump(out, open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
