"""Settings: recipe.toml's defaults, local.toml's overrides for this Mac, then environment variables in upper case."""

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JPEG_VERSION = "3.1.4.1"  # the libjpeg-turbo release the engine's JPEG decoder pins (Pillow's, bit for bit)
WEBP_VERSION = "1.6.0"  # the libwebp release its WebP decoder pins

# Each conversion: its public repo, the revision the README measured, and the window its memory allows on 256 GB
VARIANTS = {
    "mix34": ("BobClawblaw/MiniMax-M3.1-Flash-MLX-mix34", "136e775c89e996cac20c05befbcb26886ad24782", 32768),
    "mix256k": ("BobClawblaw/MiniMax-M3.1-Flash-MLX-mix256k", "c7defa69b0e8b57426a9945eb3f9eb91cfef9f8c", 147456),
}


@dataclass
class Settings:
    source_repo: str = ""
    source_revision: str = ""
    served_name: str = ""
    variant: str = "mix256k"
    model_dir: str = ""
    revision: str = ""
    tensorfold_src: str = ""
    tensorfold_sha: str = ""
    zig: str = "zig"
    zig_cpu: str = "native"
    binary: str = ""
    host: str = "127.0.0.1"
    port: int = 8897
    tf_cache: str = "~/.cache/minimax-m31-flash"
    context: int = 0
    parallel: int = 2
    drafts: bool = True
    drafter: str = "dflash2"
    load_limit_gb: float = 226
    thinking: bool = False
    max_tokens: int = 32768
    temperature: str = ""
    prompt_cache_gib: float = 4
    vision: bool = True
    vision_image_tokens: int = 4096
    vision_max_images: int = 4
    keep_warm: int = 86400
    api_key: str = ""
    extra_args: str = ""

    @staticmethod
    def path(p: str) -> Path:
        return Path(os.path.expanduser(p))

    @property
    def repo(self) -> str:
        return VARIANTS[self.variant][0]

    @property
    def pinned(self) -> str:
        return self.revision or VARIANTS[self.variant][1]

    @property
    def window(self) -> int:
        return self.context or VARIANTS[self.variant][2]

    @property
    def model(self) -> Path:
        return self.path(self.model_dir or f"~/models/MiniMax-M3.1-Flash-MLX-{self.variant}")

    @property
    def cache(self) -> Path:
        return self.path(self.tf_cache)

    @property
    def worktree(self) -> Path:
        return self.cache / "engine" / self.tensorfold_sha[:12]

    @property
    def built_binary(self) -> Path:
        return self.worktree / "zig-out" / "native" / "bin" / "tensorfold-native"

    @property
    def libs(self) -> Path:
        """The libraries the server loads beside its binary (../lib): libtfgrammar and the pinned image decoders."""
        return self.run_binary.parent.parent / "lib"

    @property
    def run_binary(self) -> Path:
        return self.path(self.binary) if self.binary else self.built_binary

    @property
    def jpeg_lib(self) -> Path:
        return self.libs / "libturbojpeg.0.dylib"

    @property
    def webp_lib(self) -> Path:
        return self.libs / "libwebpdecoder.3.dylib"

    @property
    def pid_file(self) -> Path:
        return self.cache / "server.pid"

    @property
    def log_file(self) -> Path:
        return self.cache / "server.log"

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{self.port}/v1"

    def serve_args(self) -> list[str]:
        a = [str(self.run_binary), "serve", "--host", self.host, "--port", str(self.port), "--name", self.served_name,
             "--context", str(self.window), "--max-tokens", str(self.max_tokens), "--parallel", str(self.parallel),
             "--keep-warm", str(self.keep_warm), "--no-update-check"]
        if self.temperature:
            a += ["--temperature", self.temperature]
        a.append("--thinking" if self.thinking else "--no-thinking")
        if not self.drafts:
            a.append("--no-drafts")
        if self.prompt_cache_gib >= 0:
            a += ["--prompt-cache-gib", f"{self.prompt_cache_gib:g}"]
        if self.vision:
            a += ["--vision", "--vision-image-tokens", str(self.vision_image_tokens),
                  "--vision-max-images", str(self.vision_max_images)]
        if self.api_key:
            a += ["--api-key", self.api_key]
        a += self.extra_args.split()
        a.append(str(self.model))
        return a

    def server_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["MINIMAX_LOAD_LIMIT_GB"] = f"{self.load_limit_gb:g}"
        env["MINIMAX_DRAFTER"] = self.drafter
        return env


def _coerce(kind: type, raw):
    if kind is bool:
        return raw if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "yes", "on")
    return kind(raw)


def load(local_file: Path | None = None, env: dict[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    merged: dict = {}
    for f in (ROOT / "recipe.toml", local_file or ROOT / "local.toml"):
        if f.exists():
            for section in tomllib.loads(f.read_text()).values():
                merged.update(section)
    s = Settings()
    for fld in fields(Settings):
        kind = type(getattr(s, fld.name))
        if fld.name in merged:
            setattr(s, fld.name, _coerce(kind, merged[fld.name]))
        if fld.name.upper() in env:
            setattr(s, fld.name, _coerce(kind, env[fld.name.upper()]))
    return s


def validate(s: Settings) -> list[str]:
    """The settings' problems, worded for the user (empty: none)."""
    out = []
    if s.variant not in VARIANTS:
        return [f"variant must be one of {', '.join(VARIANTS)}"]
    if not s.binary and not s.tensorfold_sha:
        out.append("no engine: set tensorfold_sha (local.toml) or binary")
    if not (s.model / "config.json").exists():
        out.append(f"no checkpoint at {s.model}: python3 -m mac download")
    if s.drafter not in ("dflash2", "dspark"):
        out.append('drafter must be "dflash2" or "dspark"')
    if s.parallel < 1 or s.parallel > 32:
        out.append("parallel must be 1 to 32 (the streams share the cache memory, README)")
    if s.window < 1 or s.window > 1048576:
        out.append("context must be 1 to 1048576 (what fits depends on the variant, README)")
    if not s.run_binary.exists():
        out.append(f"no engine binary at {s.run_binary}: python3 -m mac build")
    if s.vision and not (s.jpeg_lib.exists() and s.webp_lib.exists()):
        out.append(f"vision = true but the pinned JPEG and WebP decoders are not in {s.libs}: python3 -m mac build")
    return out
