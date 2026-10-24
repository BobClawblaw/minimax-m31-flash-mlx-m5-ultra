"""The engine: a worktree of the TensorFold checkout at the pinned commit, built under tf_cache (never in the
checkout), libtfgrammar (structured output) from xgrammar at the tag the engine pins, and the pinned image decoders
(libjpeg-turbo 3.1.4.1, libwebp 1.6.0) built from their releases into the lib folder the server loads from."""

import shutil
import ssl
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from .config import JPEG_VERSION, WEBP_VERSION, Settings

# xgrammar's release the engine's structured output is built against (zig/build/grammar.zig pins the tag), and its commit
XGRAMMAR_TAG = "v0.2.8"
XGRAMMAR_COMMIT = "97787376faee5ed8466cfad57c99855e4ce2f6aa"
XGRAMMAR_REPO = "https://github.com/mlc-ai/xgrammar"

JPEG_URL = f"https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/{JPEG_VERSION}/libjpeg-turbo-{JPEG_VERSION}.tar.gz"
WEBP_URL = f"https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-{WEBP_VERSION}.tar.gz"
WEBP_FLAGS = ["-DBUILD_SHARED_LIBS=ON"] + [f"-DWEBP_BUILD_{t}=OFF" for t in
                                           ("ANIM_UTILS", "CWEBP", "DWEBP", "GIF2WEBP", "IMG2WEBP", "VWEBP", "WEBPINFO", "WEBPMUX", "EXTRAS")]


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def build_engine(s: Settings, rebuild: bool = False) -> Path:
    if s.binary:
        return s.run_binary
    if not s.tensorfold_sha:
        raise SystemExit("no tensorfold_sha pinned (local.toml)")
    src = s.path(s.tensorfold_src)
    if not s.worktree.exists():
        s.worktree.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "-C", str(src), "worktree", "add", "--detach", str(s.worktree), s.tensorfold_sha])
    if rebuild or not s.built_binary.exists():
        _run([s.path(s.zig).as_posix(), "build", "native", "-Doptimize=ReleaseFast", f"-Dcpu={s.zig_cpu}"], cwd=s.worktree)
    return s.built_binary


def _release(s: Settings, url: str, name: str, flags: list[str], built: str, target: Path) -> Path:
    """A decoder library from its release tarball, built with cmake under tf_cache and copied to `target`."""
    if target.exists():
        return target
    work = s.cache / f"{name}-src"
    work.mkdir(parents=True, exist_ok=True)
    tgz = work / f"{name}.tar.gz"
    if not tgz.exists():
        print("downloading", url, flush=True)
        # python.org's Python ships no CA store: the system's bundle
        ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
        with urllib.request.urlopen(url, context=ctx) as r, open(tgz, "wb") as f:
            shutil.copyfileobj(r, f)
    with tarfile.open(tgz) as t:
        t.extractall(work, filter="data")
    build = work / "build"
    _run(["cmake", "-S", str(work / name), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"] + flags)
    _run(["cmake", "--build", str(build), "-j", "8"])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2((build / built).resolve(), target)
    return target


def build_decoders(s: Settings) -> None:
    _release(s, JPEG_URL, f"libjpeg-turbo-{JPEG_VERSION}", ["-DENABLE_STATIC=OFF", "-DWITH_JAVA=OFF"], "libturbojpeg.0.dylib", s.jpeg_lib)
    _release(s, WEBP_URL, f"libwebp-{WEBP_VERSION}", WEBP_FLAGS, "libwebpdecoder.3.dylib", s.webp_lib)


def build_grammar(s: Settings, rebuild: bool = False) -> Path:
    """libtfgrammar beside the binary (the server loads it at the first structured request): xgrammar cloned once into
    tf_cache with its submodules, then zig build grammar in the worktree."""
    built = s.libs / "libtfgrammar.dylib"
    if s.binary or (built.exists() and not rebuild):
        return built
    root = s.cache / f"xgrammar-{XGRAMMAR_TAG}"
    if not (root / ".git").exists():
        _run(["git", "clone", "--branch", XGRAMMAR_TAG, "--recurse-submodules", XGRAMMAR_REPO, str(root)])
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != XGRAMMAR_COMMIT:
        raise SystemExit(f"{root} is at {head[:8]}, not xgrammar {XGRAMMAR_TAG}'s {XGRAMMAR_COMMIT[:8]}; remove it to clone again")
    _run([s.path(s.zig).as_posix(), "build", "grammar", f"-Dxgrammar={root}", f"-Dcpu={s.zig_cpu}"], cwd=s.worktree)
    if not built.exists():
        raise SystemExit(f"the build left no {built}")
    return built


def build(s: Settings, rebuild: bool = False) -> None:
    build_engine(s, rebuild)
    build_grammar(s, rebuild)
    if s.vision:
        build_decoders(s)
