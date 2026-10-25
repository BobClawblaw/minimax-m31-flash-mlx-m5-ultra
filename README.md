# MiniMax-M3.1-Flash on one M5 Ultra (TensorFold native)

Serves [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash), a 428B-A23B mixture of experts with
block-sparse attention and a vision tower, with TensorFold's native Metal engine on one 256 GB Mac, as an
OpenAI-compatible server with images, tool calls, structured output, thinking on or off, DFlash2 drafts, the prompt
cache and two streams.

MiniMax's release stores its experts in NVFP4, which Metal has no kernels for, so this recipe serves two conversions
of it to MLX-style affine groups, both public (the engine's `tools/zig/minimax_m3_convert.py` rebuilds them from
the pinned release and checks every file against the published hashes):

| variant | checkpoint | experts | on disk | window on 256 GB |
|---|---|---|---:|---:|
| `mix34` | [`BobClawblaw/MiniMax-M3.1-Flash-MLX-mix34`](https://huggingface.co/BobClawblaw/MiniMax-M3.1-Flash-MLX-mix34) | 4-bit (g128) on 26 of the 60 layers (`tools/requant_study.py`), 3-bit (g64) on the rest | 203.7 GiB | 32,768 |
| `mix256k` (default) | [`BobClawblaw/MiniMax-M3.1-Flash-MLX-mix256k`](https://huggingface.co/BobClawblaw/MiniMax-M3.1-Flash-MLX-mix256k) | 4-bit on 3 layers, 3-bit on the rest | 189.2 GiB | 147,456 |

The key/value cache is bf16, exact (TensorFold ships no lossy caches), and it is what limits the window: the
weights take 189-204 GiB of the 256. mix256k is the default: on the quality suites below it scores level with mix34,
and its window is four and a half times as long. mix34 stays a little closer to the release (KL 0.246 against 0.301)
for work that fits in 32,768 tokens. (mix256k is named for the 262,144-token window an earlier int8 cache gave it.)

Attention, the shared expert, the dense layers and the head are 8-bit in both. Each checkpoint also carries the
release's DSpark head and incoai's [DFlash2 drafter](https://huggingface.co/incoai/MiniMax-M3.1-Flash-DFlash2).

The engine's `minimax_m3` family is in [ashhart/TensorFold#1235](https://github.com/ashhart/TensorFold/pull/1235);
until it merges, this recipe builds the pinned commit of its branch (c1ace3229, `minimax-m3` on
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
| `variant` | mix256k | `mix256k` for the 147,456-token window, `mix34` for a little more fidelity at 32,768 |
| `context` | 0 | a stream's window; 0 takes the variant's (32,768 or 147,456) |
| `parallel` | 2 | streams decoded together; they share the cache memory |
| `load_limit_gb` | 226 | weights, scratch and caches together (`MINIMAX_LOAD_LIMIT_GB`) |
| `drafts`, `drafter` | true, dflash2 | DFlash2 drafts one stream; `dspark` uses the release's own head; greedy replies are identical with drafts off |
| `thinking` | false | the default for requests; `chat_template_kwargs: {"enable_thinking": true}` turns it on for one |
| `max_tokens` | 32768 | a reply's default ceiling |
| `prompt_cache_gib` | 4 | kept prompt states between turns; they give way to streams |
| `vision`, `vision_image_tokens`, `vision_max_images` | true, 4096, 4 | images a request may hold, and their tokens in all |

**Memory.** The engine's default limit (75% of RAM) holds the weights but no caches, so the recipe sets
`load_limit_gb = 226`. The bf16 cache takes 135 KiB a token across the 60 layers. Under 226 GB the caches fit
150,400 tokens on mix256k and 37,376 on mix34 (the engine's own fit, with images and the drafter); the windows
leave a little of that spare. At a 145,203-token prompt on mix256k the Mac showed 210 GiB wired: near the edge, and
other apps' memory comes out of it. mix34 needs the Mac nearly idle to load (with other apps holding about 36 GiB,
macOS killed it while it mapped its weights). Shorter windows or `parallel = 1` leave more.

## What works, and how it was checked

All on one M5 Ultra (256 GB, macOS 27), checkpoints at the revisions in `mac/config.py`. The engine's bf16-only
commit was measured for KL, the window, the needle, decode and prompt speed and the harness's `quick` plan on both
variants (`evidence/*-bf16`): structured 18/18, drafted == plain 12/12, tools 5/5, seeded sampling 9/9, turns
resumed == cold, and the suite's soak 60/60, drafts 12/12, concurrency 12/12 and behaviour checks. The pinned
c1ace3229 is that code rebased onto newer TensorFold main twice (first as e6b3b3bd0, then onto main faccd11c8):
each time its prompt logits and greedy tokens are byte-identical, and `quick` on mix256k passed again
(`evidence/mix256k-quick-pinned`, `evidence/mix256k-quick-c1ace3229`). The bf16 path is also byte-identical to the
earlier engine's bf16 option, so the math and image checks below (run on that engine) were not repeated; the
quality suite was, on both variants (on e6b3b3bd0).

- **The math.** Four layers of the engine against transformers' own model: KL at most 0.005. The full model against
  an MLX reference (`ref/model.py`) at the same weights: KL 0.015, smaller than the reference's own bf16 noise.
- **The conversion's cost.** 1,024 rows teacher-forced against the NVFP4 release, on the pinned engine: mix34 mean
  KL 0.246 (81.1% top-1 agreement), mix256k 0.301 (78.3%).
- **Long context.** On mix256k with images on, a needle came back from a 145,203-token prompt (115 s cold, 210 GiB
  wired at the peak; `evidence/mix256k-needle-bf16`).
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
the bf16 cache; the raw results are in `evidence/*-bf16`.

**Decode**, tokens a second, all streams together (prose / code / structured / list):

| variant | streams | DFlash2 drafts |
|---|---:|---|
| mix34 | 1 | 42.7 / 58.5 / 50.3 / 55.8 |
| mix34 | 2 | 51.0 / 65.1 / 57.9 / 83.3 |
| mix256k | 1 | 43.1 / 58.9 / 49.6 / 50.3 |
| mix256k | 2 | 57.6 / 63.3 / 61.3 / 75.5 |

Against the earlier int8 cache: mix256k level or faster (two streams' prose +10%, their code -3%), mix34's code
slower (58.5 against 65.9 with one stream). Plain decode (`drafts = false`) was not measured again on the bf16 cache; on the int8 cache it ran 43-44
tokens a second with one stream and 62-64 with two, faster than drafts for two prose streams.

**Prompts**, cold, thinking off (a fresh server, each prompt starting with words no earlier one did):

| variant | 8k | 16k | 31k | 32k | 128k |
|---|---:|---:|---:|---:|---:|
| mix34 | 1,441 tok/s (5.8 s) | 1,452 (11.4 s) | 1,451 (21.3 s) | | |
| mix256k | 1,483 (5.7 s) | | | 1,491 (22.1 s) | 1,379 (95.2 s) |

With the int8 cache: mix256k 1,429 / 1,420 / 1,319, and before the engine's prompt-speed commits 1,206 / 1,020 / 703
(186.8 s at 128k). With thinking on, within 2%. A fresh server's first prompt also pays its warm-up (a 2k prompt:
3.9-4.1 s first, 1.7 s after). The 145,203-token needle prompt took 115 s. A multi-turn chat's resumed turn starts in
0.26-0.30 s against 3.6-5.6 s cold.

**Quality**, thinking off (the harness's `dsv41-quality-off` plan). Both variants were run again on the pinned bf16
cache (`evidence/mix256k-quality-off-bf16`, `evidence/mix34-quality-off-bf16`); the int8 columns are the earlier
engine's int8 cache, kept for comparison:

| | mix256k, bf16 | mix256k, int8 | mix34, bf16 | mix34, int8 |
|---|---:|---:|---:|---:|
| GSM8K (250) | 0.976 | 0.976 | 0.968 | 0.964 |
| MGSM (240) | 0.821 | 0.875 | 0.846 | 0.829 |
| MMLU (342) | 0.819 | 0.819 | 0.827 | 0.830 |
| MATH-500 | 0.900 | 0.908 | 0.890 | 0.908 |
| IFEval, loose (150) | 0.593 | 0.573 | 0.580 | 0.573 |
| tools (60) | 0.933 | 0.917 | 0.95 | 0.95 |
| JSON (30) | 1.0 | 1.0 | 1.0 | 1.0 |

The cache swap moves MGSM in opposite directions on the two variants. mix256k's bf16 loss is nearly all in Japanese
(22/30 against 27) and Swahili (15/30 against 20); mix34's bf16 gain is mostly Bengali (26/30 against 21), with
Swahili level at 17. With 30 questions a language, these read as greedy paths diverging, not as a cost of either cache.
mix34's MATH-500 drop is mostly unfinished replies: 27 of 500 give no boxed answer, against 17 with int8.

IFEval is low for a model this size: the suite caps replies at 1,200 tokens, MiniMax writes long (on mix34, 34 of
150 replies hit the cap, 23 of them failing), and many of the rest break the format they were asked for. Thinking
does not rescue many (3 of 12 failed prompts pass with it on). No thinking-on quality run was made.

**TensorFold leaderboard.** The board's community-v1 suite on mix256k:
[result](https://tensorfold.dev/benchmarks?result=bc110e01-c245-4dec-aa40-63d3e432c456) (code 32.0 / 51.4 and chat
28.1 / 45.7 tokens a second at temperature 1 / 0, as the client receives them; the engine commit plus the client's
`--suite community-v1` protocol). mix34 could not complete the suite: its code fixture is an instruction sent as a raw
completion, and at temperature 0 mix34 ends it at the first token (an empty reply), as it did on 4 of 6 tries at
temperature 1.

## Limits

- **Images:** at most 576 tokens an image (672x672, the release processor's cap; larger images are scaled down).
  Each image is encoded on its own. JPEG and WebP need the pinned decoders `python3 -m mac build` makes
  (libjpeg-turbo 3.1.4.1, libwebp 1.6.0); PNG works without them. Images come as data URLs: the server's
  `--vision-urls` (fetching public HTTPS images) is off in this recipe and was not checked. Video is not served.
- **Context:** the model's window is 1,048,576 tokens; on 256 GB the exact cache fits 150,400 beside mix256k and
  37,376 beside mix34.
- **Streams:** two were checked. More fit only with shorter windows.
- Only the M5 Ultra was checked.

## Reproducing the checkpoints

The engine's `tools/zig/minimax_m3_convert.py` (from this recipe's `tools/convert.py`) converts the release at revision
54e4e69d and incoai's DFlash2 at ac07a7c1 to either variant (`--variant mix34|mix256k`), and `--check` compares a
folder with the published files' hashes; a full mix256k rebuild checks 74/74 (the commands are on the engine's
recipe page). `tools/requant_study.py` ranked the 4-bit layers; `tools/compare.py` and `ref/` hold the KL checks.

## License

The recipe is MIT. The weights are under MiniMax's community license (non-commercial use; see the model cards);
incoai's DFlash2 drafter is Apache-2.0; TensorFold is Apache-2.0.
