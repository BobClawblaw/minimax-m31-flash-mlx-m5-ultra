# MiniMax-M3.1-Flash on one M5 Ultra (TensorFold native)

Serves [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash), a 428B-A23B mixture of experts with
block-sparse attention and a vision tower, with TensorFold's native Metal engine on one 256 GB Mac, as an
OpenAI-compatible server with images, tool calls, structured output, thinking on or off, DFlash2 drafts, the prompt
cache and two streams.

MiniMax's release stores its experts in NVFP4, which Metal has no kernels for, so this recipe serves two conversions
of it to MLX-style affine groups (`tools/convert.py`), both public:

| variant | checkpoint | experts | on disk | window on 256 GB |
|---|---|---|---:|---:|
| `mix34` | [`BobClawblaw/MiniMax-M3.1-Flash-MLX-mix34`](https://huggingface.co/BobClawblaw/MiniMax-M3.1-Flash-MLX-mix34) | 4-bit (g128) on 26 of the 60 layers (`tools/requant_study.py`), 3-bit (g64) on the rest | 203.7 GiB | 65,536 |
| `mix256k` (default) | [`BobClawblaw/MiniMax-M3.1-Flash-MLX-mix256k`](https://huggingface.co/BobClawblaw/MiniMax-M3.1-Flash-MLX-mix256k) | 4-bit on 3 layers, 3-bit on the rest | 189.2 GiB | 262,144 |

mix256k is the default: on the quality suites below it scores level with mix34, and its window is four times
as long. mix34 stays a little closer to the release (KL 0.250 against 0.284) and drafts code faster (66 against 58
tokens a second) for work that fits in 65,536 tokens.

Attention, the shared expert, the dense layers and the head are 8-bit in both. Each checkpoint also carries the
release's DSpark head and incoai's [DFlash2 drafter](https://huggingface.co/incoai/MiniMax-M3.1-Flash-DFlash2).

The engine's `minimax_m3` family is in [ashhart/TensorFold#1235](https://github.com/ashhart/TensorFold/pull/1235);
until it merges, this recipe builds the pinned commit of its branch (5aff9a36d, `minimax-m3` on
[BobClawblaw/TensorFold](https://github.com/BobClawblaw/TensorFold/tree/minimax-m3)). Its first commit is the MoE
routing change of [ashhart/TensorFold#1177](https://github.com/ashhart/TensorFold/pull/1177).

## Quick start

Needs an M5 Mac with 256 GB (the M5's tensor units; nothing smaller holds the weights), Python 3.11+, git, cmake,
`hf` and [Zig 0.17.0](https://ziglang.org/download/).

```bash
git clone https://github.com/BobClawblaw/TensorFold ~/github/TensorFold
cp local.example.toml local.toml      # then set zig and the variant (and paths, if yours differ)
python3 -m mac download               # the variant's checkpoint at its pinned revision, into ~/models
python3 -m mac build                  # the pinned engine, libtfgrammar and the JPEG/WebP decoders
python3 -m mac up                     # http://127.0.0.1:8897/v1, model "MiniMax-M3.1-Flash"
python3 -m mac chat "Explain block-sparse attention in three sentences."
```

`python3 -m mac down|status|logs|command|validate` stop, inspect and check it. To serve the LAN on login:
`python3 -m mac launchd --port 8897` writes `~/Library/LaunchAgents/com.tensorfold.minimax-m31-flash.plist` (not
loaded), then `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.tensorfold.minimax-m31-flash.plist`.

Run one copy of the model at a time: two need more memory than the Mac has. Stop other large models first.

## Settings

`recipe.toml` holds the defaults; `local.toml` overrides them for one Mac, and an environment variable of the same
name in upper case overrides both (`VARIANT=mix34 python3 -m mac up`).

| setting | default | |
|---|---|---|
| `variant` | mix256k | `mix256k` for the 262,144-token window, `mix34` for a little more fidelity at 65,536 |
| `context` | 0 | a stream's window; 0 takes the variant's (65,536 or 262,144) |
| `parallel` | 2 | streams decoded together; they share the cache memory |
| `kv_cache` | int8 | keys and values cached as int8 with a scale a row; `bf16` keeps the full width (and needs 1.75x the memory) |
| `load_limit_gb` | 226 | weights, scratch and caches together (`MINIMAX_LOAD_LIMIT_GB`) |
| `drafts`, `drafter` | true, dflash2 | DFlash2 drafts one stream; `dspark` uses the release's own head; greedy replies are identical with drafts off |
| `thinking` | false | the default for requests; `chat_template_kwargs: {"enable_thinking": true}` turns it on for one |
| `max_tokens` | 32768 | a reply's default ceiling |
| `prompt_cache_gib` | 4 | kept prompt states between turns; they give way to streams |
| `vision`, `vision_image_tokens`, `vision_max_images` | true, 4096, 4 | images a request may hold, and their tokens in all |

**Memory.** The engine's default limit (75% of RAM) holds the weights but no caches, so the recipe sets
`load_limit_gb = 226`. The int8 cache takes 77 KiB a token (bf16: 135 KiB) across the 60 layers, 19.6 GiB for a
262,144-token stream with its index. At a 251,713-token prompt on mix256k the Mac showed 212 GiB wired and about
4 GiB free: that is the edge, and other apps' memory comes out of it. mix34 at 226 GB fits 68,352 tokens; the
recipe's 65,536 leaves room for images. Shorter windows or `parallel = 1` leave more.

## What works, and how it was checked

All on one M5 Ultra (256 GB, macOS 27), checkpoints at the revisions in `mac/config.py`. The measurements below were
taken on the engine branch before its last rebases onto TensorFold main (mix34's decode runs one commit earlier, before
the forced-tool-call grammar). The pinned 5aff9a36d is the same code on a newer main, plus a docs commit; after the
rebase the harness's `quick` plan on mix34 (the same code but for main's last four CUDA-only commits) gave the same
one-stream speed (41.9 / 65.2 / 46.8 / 56.5 tokens a second), structured 18/18, drafted == plain 12/12, tools 5/5,
seeded sampling 9/9, turns resumed == cold, and the suite's soak 60/60, drafts 12/12, concurrency 12/12 and
behaviour checks (`evidence/mix34-quick-rebased`).

- **The math.** Four layers of the engine against transformers' own model: KL at most 0.005. The full model against
  an MLX reference (`ref/model.py`) at the same weights: KL 0.015, smaller than the reference's own bf16 noise.
- **The conversion's cost.** 1,024 rows teacher-forced against the NVFP4 release (the int8 cache and bf16 cache):

  | checkpoint | cache | mean KL | top-1 agreement |
  |---|---|---:|---:|
  | mix34 | bf16 | 0.246 | 81.1% |
  | mix34 | int8 | 0.250 | 80.0% |
  | mix256k | bf16 | 0.301 | 78.3% |
  | mix256k | int8 | 0.284 | 78.2% |

- **Long context.** On mix256k with the int8 cache and images on, a needle came back at 103,770 tokens (156 s prompt)
  and 251,713 tokens (545 s, 462 tokens a second; decode afterwards 45 tokens a second, drafted).
- **Images.** The tower equals transformers' (relative error 0.030 against fp32, where torch's bf16 gives 0.039);
  86 ms for a 540-token image. Through the chat API the model reads shapes, colours and text in PNG, JPEG and WebP
  images, two images in one prompt work, and replies are identical with the cache on and off and drafted or plain.
- **Tools and structured output.** The harness's structured check (json_object, json_schema, regex, choice, grammar)
  18/18 on both variants, drafted == plain 12/12; its tool cases (auto, required, named, thinking, streamed) 5/5.
  Forced tool calls (`tool_choice: required` or a named tool) are constrained in MiniMax's own call markup.
- **Prompt cache.** A turn resumed from a kept state equals the same request sent again cold; replies are identical
  with the cache off.
- **Drafts.** Drafted replies equal plain ones token for token (DFlash2 and DSpark; the quality suite's drafts check
  12/12).
- **Concurrency.** Two streams' replies equal each run alone (the quality suite's streaming check).

## Measured

Through [recipe-harness](https://github.com/BobClawblaw/recipe-harness) (targets `minimax-m31-mix34-m5` and
`minimax-m31-mix256k-m5`, the qwen Spark recipe's ruler and suite at commit 277c842a1481), greedy, 256-token replies,
the int8 cache; the raw results are in `evidence/`.

**Decode**, tokens a second, all streams together (prose / code / structured / list):

| variant | streams | DFlash2 drafts | plain |
|---|---:|---|---|
| mix34 | 1 | 42.1 / 65.9 / 47.5 / 56.8 | 43.4 / 43.3 / 43.4 / 43.5 |
| mix34 | 2 | 49.7 / 65.8 / 53.9 / 76.1 | 62.2 / 61.6 / 61.8 / 62.1 |
| mix256k | 1 | 42.8 / 57.9 / 48.7 / 47.9 | 44.1 / 44.0 / 44.0 / 44.1 |
| mix256k | 2 | 52.2 / 65.5 / 58.1 / 74.7 | 63.3 / 62.8 / 64.1 / 63.6 |

Drafts help one stream on code and lists and roughly break even on prose. With two streams they cost speed on prose
and structured text; `drafts = false` is the faster choice for a server
that mostly runs two prose streams at once.

**Prompts**, cold, thinking off (a fresh server, each prompt starting with words no earlier one did):

| variant | 8k | 32k | 60k | 128k |
|---|---:|---:|---:|---:|
| mix34 | 1,177 tok/s (7.1 s) | 1,011 (32.6 s) | 886 (69.6 s) | |
| mix256k | 1,206 (7.0 s) | 1,020 (32.3 s) | | 703 (186.8 s) |

With thinking on, within 3%. A fresh server's first prompt also pays its warm-up (a 2k prompt: 4.6 s first, 2.0 s
after). The 251,713-token needle prompt above took 545 s. A multi-turn chat's resumed turn starts in 0.33 s against
4.25 s cold (mix34; mix256k 0.31 s against 6.2 s).

**Quality**, thinking off (the harness's `dsv41-quality-off` plan):

| | mix34 | mix256k |
|---|---:|---:|
| GSM8K (250) | 0.964 | 0.976 |
| MGSM (240) | 0.829 | 0.875 |
| MMLU (342) | 0.830 | 0.819 |
| MATH-500 | 0.908 | 0.908 |
| IFEval, loose (150) | 0.573 | 0.573 |
| tools (60) | 0.95 | 0.917 |
| JSON (30) | 1.0 | 1.0 |

IFEval is low for a model this size: the suite caps replies at 1,200 tokens, MiniMax writes long (on mix34, 34 of
150 replies hit the cap, 23 of them failing), and many of the rest break the format they were asked for. Thinking
does not rescue many (3 of 12 failed prompts pass with it on). No thinking-on quality run was made.

**TensorFold leaderboard.** The board's community-v1 suite on mix256k:
[result](https://tensorfold.dev/benchmarks?result=bc110e01-c245-4dec-aa40-63d3e432c456) (code 32.0 / 51.4 and chat
28.1 / 45.7 tokens a second at temperature 1 / 0, as the client receives them; the engine commit plus the client's
`--suite community-v1` protocol). mix34 could not complete the suite: its code fixture is an instruction sent as a raw
completion, and at temperature 0 mix34 ends it at the first token (an empty reply), as it did on 10 of 12 tries at
temperature 1.

## Limits

- **Images:** at most 576 tokens an image (672x672, the release processor's cap; larger images are scaled down).
  Each image is encoded on its own. JPEG and WebP need the pinned decoders `python3 -m mac build` makes
  (libjpeg-turbo 3.1.4.1, libwebp 1.6.0); PNG works without them. Images come as data URLs: the server's
  `--vision-urls` (fetching public HTTPS images) is off in this recipe and was not checked. Video is not served.
- **Context:** the model's window is 1,048,576 tokens; on 256 GB only mix256k's 262,144 fits, and only just.
- **Streams:** two were checked. More fit only with shorter windows.
- Only the M5 Ultra was checked.

## Reproducing the checkpoints

`tools/convert.py` converts the release (revision 54e4e69d) to either variant; `tools/compare.py` and `ref/` hold
the KL checks above.

## License

The recipe is MIT. The weights are under MiniMax's community license (non-commercial use; see the model cards);
incoai's DFlash2 drafter is Apache-2.0; TensorFold is Apache-2.0.
