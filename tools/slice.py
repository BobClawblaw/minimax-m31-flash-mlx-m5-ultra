"""A truncated checkpoint for layer checks: the first N blocks of a converted folder, linked, with a config of N layers.

  python tools/slice.py OUT N SRC_DIR [SRC_DIR...]   (shards are looked up in each SRC_DIR in turn)
"""

import json
import os
import sys


def main():
    out, n, srcs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    os.makedirs(out, exist_ok=True)
    weight_map = {}
    for d in srcs:
        idx = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(idx):
            for k, f in json.load(open(idx))["weight_map"].items():
                weight_map.setdefault(k, os.path.join(d, f))
    keep = {}
    for k, path in weight_map.items():
        if k.startswith("model.layers."):
            if int(k.split(".")[2]) >= n:
                continue
        keep[k] = os.path.basename(path)
        link = os.path.join(out, os.path.basename(path))
        if not os.path.lexists(link):
            os.symlink(path, link)
    missing = [i for i in range(n) if f"model.layers.{i}.input_layernorm.weight" not in keep]
    if missing or "lm_head.weight" not in keep:
        raise SystemExit(f"missing blocks {missing} or the head")
    json.dump({"metadata": {}, "weight_map": keep}, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
    cfg = json.load(open(os.path.join(srcs[0], "config.json")))
    cfg["num_hidden_layers"] = n
    cfg["moe_layer_freq"] = cfg["moe_layer_freq"][:n]
    for k in ("sparse_attention_freq", "sparse_disable_index_value"):
        cfg["sparse_attention_config"][k] = cfg["sparse_attention_config"][k][:n]
    q = cfg["quantization"]
    for d in srcs[1:]:
        path = os.path.join(d, "config.json")
        for k, v in (json.load(open(path)).get("quantization", {}) if os.path.exists(path) else {}).items():
            q.setdefault(k, v)
    cfg["quantization"] = {k: v for k, v in q.items() if not (k.startswith("model.layers.") and int(k.split(".")[2]) >= n)}
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    for f in os.listdir(srcs[0]):
        if not f.startswith("model") and not f.endswith(".part") and not os.path.lexists(os.path.join(out, f)):
            os.symlink(os.path.join(srcs[0], f), os.path.join(out, f))


if __name__ == "__main__":
    main()
