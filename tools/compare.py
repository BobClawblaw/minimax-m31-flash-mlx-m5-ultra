"""Native bf16 logits (tf-minimax-run) against the reference's fp32 .npy: per-row KL, top-1 agreement, worst diff.

  python tools/compare.py REF.npy NATIVE.bin [--rows a:b]
"""

import sys

import numpy as np


def bf16_file(path, vocab):
    u = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).reshape(-1, vocab)


def logsoftmax(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def main():
    ref = np.load(sys.argv[1]).astype(np.float32)
    ref = ref.reshape(-1, ref.shape[-1])
    nat = bf16_file(sys.argv[2], ref.shape[-1])[: ref.shape[0]]
    ref = ref[: nat.shape[0]]
    lr, ln = logsoftmax(ref), logsoftmax(nat)
    kl = (np.exp(lr) * (lr - ln)).sum(-1)
    top_r, top_n = ref.argmax(-1), nat.argmax(-1)
    srt = np.sort(ref, -1)
    margin = srt[:, -1] - srt[:, -2]
    for i in range(len(kl)):
        flag = "" if top_r[i] == top_n[i] else f"  top differs (ref {top_r[i]} margin {margin[i]:.3f}, native {top_n[i]})"
        print(f"row {i:4d} KL {kl[i]:.5f} max|d| {np.abs(ref[i] - nat[i]).max():.3f}{flag}")
    print(f"rows {len(kl)}: mean KL {kl.mean():.5f}, max KL {kl.max():.5f}, top-1 {np.mean(top_r == top_n) * 100:.1f}%")


if __name__ == "__main__":
    main()
