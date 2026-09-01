"""Build an additive portable LATO.2 runtime without mutating upstream source.

``materialize_portable_runtime`` is the setup-facing contract.  It validates
the exact pinned upstream files, copies the complete tree to a staging
directory, applies narrowly scoped compatibility patches, overlays the helper
modules in ``portable_overrides/``, and atomically publishes the result.

The exact upstream checkout remains available next to this generated copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from .constants import LATO_SOURCE_REVISION
from .integrity import (
    TreeIntegrityError,
    TreeInventory,
    entry_exists,
    inventories_equal,
    inventory_tree,
    remove_owned_entry,
)


SPACE_SOURCE_REVISION = "25ec65da46236e5ef46b88d9a510a0fe33b2bc63"
PORTABLE_SCHEMA_VERSION = 3
PORTABLE_MARKER = ".modly-portable.json"

# Hashes are for LoHhhha/LATO.2 at LATO_SOURCE_REVISION.  Refusing to patch an
# unexpected file prevents a future upstream update from being half-patched.
# Complete regular-file inventory for LoHhhha/LATO.2 at
# ``LATO_SOURCE_REVISION``.  Unlike the generated marker in models_dir, this
# extension-controlled inventory cannot be rewritten alongside a compromised
# portable copy to make it attest itself.
EXPECTED_SOURCE_SHA256 = {
    ".gitignore": "e911b61b1d84f19f4d08e9e4de458916e55d60a0551cfdfd8f527b995ae74b8c",
    "LICENSE": "aa0d6cc3dff4b317ca27e27997736173b893c36fd6dc253a52dbd5745ac5d9ba",
    "assets/example_mesh/crocodile.glb": "6f6e3db8a580db37a6e9145f68b3cc765143a5b081d6dc6ce6713949cbad21ca",
    "assets/example_mesh/dragon.glb": "347d0a6f76b3b6afa34b13b75de14e73ab2e45ef5fdeb17c62a7d8216d358fd2",
    "assets/example_mesh/spaceman.glb": "20a48887b1d682b91f8a19e9ec9ec83e9bf0c402ef062ccdd26085a626b2eca1",
    "assets/teaser.png": "3a8551b9f4b9b0c5becfa6225ca554237d10f3eea283e1b51790288e27776bc4",
    "dataset/mesh_render.py": "68800b9527edd0528ab1b4ea8c520cc90e4ed27175fbafc1d7806abdccf8d87b",
    "dataset/topo_dataset.py": "ecb18a3e971cc6b1093e065a2bb7f428d661e4689e8571481f32ecabddfc8022",
    "dataset/utils.py": "f10df4aa1fb69a42113c8092af96aa4a3484a477f636656f9f0bb7a89a561e89",
    "dataset/voxel_dataset.py": "9256a3b96b4e935f99bf9d2ae44a2102990a57f3cd941d5ecc01453d08627843",
    "models/__init__.py": "8c2f4e39a2fc37e944004eeedae0ddd1908a71aac55f9a95e3afa192a96f5c34",
    "models/dino_encoder.py": "9442771487c13e8c73ff182d6f9ef3b942b72265bca1f1cdf5515d75f66f1f4b",
    "models/flow_sampler.py": "f5c9d92f5c60201cf5b9e816205918f88002249c0678b0806b1511fd81103317",
    "models/offset_head.py": "2a0f1b1f0df23e3f43d42c5183ccce824e8789ef6464e6a9552b9e3ad7c26387",
    "models/topo_autoencoder.py": "82d85c5f05cd6859f0ca0db629e436170b04980b0402fe1b31a8154d706c518b",
    "models/topo_flow.py": "e7a139f9fdd400d4c735b44fa8d0c3ec0183a0f12d828214b4f6f3fc66f651aa",
    "models/vdf_encoder.py": "91614927ebc09f509f6716d29e61d1e84f212ddd6e2e4520a9f8f01988949298",
    "models/vertex_autoencoder.py": "3ce7ae053c28353f83ace1d7250cc57c7822d2383266f8500497c3f6050aad8f",
    "models/vertex_structured_flow.py": "0e57c46ce7ead0846f0ba5dacfbf3460fcc898bff6e5fea93a6d5e5f21e55550",
    "models/voxel_encoder.py": "f63779caca781e7131ce922fe95b5b9caaf8d87a5760ccbe2b56251c1b2b70af",
    "modules/attention.py": "86537a47aad37d690732d2e3e96275023fb21c50ffeea2c24f49023d9f506237",
    "modules/norm.py": "e584a2fd4f6145adc0bb15bc3c9cf0fa3d316339dcba5a6359084a18fcb2ef73",
    "modules/pointnet.py": "f60142d35c070ecd6c128bbcbf9b37db8e11da731d23b94665bdf5e2fce2d05d",
    "modules/sparse/__init__.py": "6d8ad5e4a3d1ed8c57fbfaec6475782739203002949c6a60d5012716bacf320f",
    "modules/sparse/attention/__init__.py": "3cab06d5b92f3d7dfd4655f3e29e562c72669257280945b94167d867cafa8576",
    "modules/sparse/attention/modules.py": "9ef9200d30bde37cc015a53403f19e05927cf23eb1a40a948ffbe5c043d797bc",
    "modules/sparse/basic.py": "6f2459793e58ff16a83d13113ec2080df5187a600c846eafa2939039614eac66",
    "modules/sparse/blocks.py": "1252b2c3faa2ca56820d9cb566c3f9a45f98c09b6ccf677c6d6a0e8e863fd81f",
    "modules/sparse/conv/__init__.py": "88d2abea7dac6ed4484b038446917f45f9e2141092252ebab2847eb81678c03e",
    "modules/sparse/conv/conv_spconv.py": "e7db88193a8a2eafca6cc051577b9f033356edcd3d8c456ed208a450786b4005",
    "modules/sparse/conv/conv_torchsparse.py": "c68c9bf3ca294438edef7ef500ebabebf201386d9c0b16320bb63428759c26e5",
    "modules/sparse/attention/full_attn.py": "90999a4e564302231b691d31ac446ba30bb3e652b68d19b731f14f6548b22f66",
    "modules/sparse/attention/serialized_attn.py": "8fcd176485c293fd257f6bea1e091b8ee2c967d28c46f0b3d40c70c0bbedeaac",
    "modules/sparse/attention/windowed_attn.py": "be9522bc98b00954af25aaa6f0916884feb05a84dd09a3edfcc148bc01225f7f",
    "modules/sparse/linear.py": "488b665af6dfb9ca3ad55fd5c176aa6f128f43da7e92910edbdb40bb31077bf3",
    "modules/sparse/nonlinearity.py": "cd22b6f9c6984d95aa1eda095e63b754d7a7bae1b1bd1df0a265fe56710bcbc9",
    "modules/sparse/norm.py": "9e563af7884be4bfb88b4545289ad177bbee6f81170f6dc411f12e41f0c6b37d",
    "modules/sparse/spatial.py": "c987aecb6be79fe843cfe8dd1efff5b549ff686a6e32f22a2c3f6141a36aea23",
    "modules/sparse/transformer/__init__.py": "d3436faa9b80e49a654e7ef4698c7c2579bb8630dbee7d6594ae30a89fe58da8",
    "modules/sparse/transformer/bases.py": "b951be482ce2a21815bd699d75bd42cda60928c1719a82a36c0cfec06adfb281",
    "modules/sparse/transformer/blocks.py": "6f8931fe3aed5acd8dc4cbf04aafd4309f26862eb6069303f9a686a832f776a1",
    "modules/sparse/transformer/modulated.py": "52765a1ff6f4be8457957b2b1e90b592e652e54803100da4a04ee9b9e1966b82",
    "modules/transformer/__init__.py": "5ad3f7cf472be756ff69761c3279d115eba744d748c608a0af23a998da47e0bf",
    "modules/transformer/blocks.py": "9cc45ca97ad6877df7de48185a8162a55da3324daa7fed6a7dc4596342b6b035",
    "modules/transformer/hybrid.py": "d85c92e4ecac5000c8fd7e0168a2affff42914a7cbf124f963c5db7305f0fce2",
    "modules/utils.py": "a3ef89e970819a3f70960d87ef2f7dc8b3ea0bc1b8d91c6022c7707e5dea0aa1",
    "readme.md": "eabacc571327f29b5b1771995bddbe865ae72e23a2fca131b02aad1d57b56b09",
    "scripts/ckpt_download.py": "b99e369534d04179a5f8b6267cc99a478d2d160a80bc9b905056d057dbacabb0",
    "scripts/e2e_inference.py": "6672052a11c7fe1130d84e25cd679411133408ab9c102c75e13abb149e7c2f67",
    "scripts/tflow_inference.py": "c4c02c7ca1abe80d816d0155d11f069e608368587b547b2dabc779324a2e93db",
    "scripts/vflow_inference.py": "7b750f4e10922bee4b100c4035fc57e04e0e0a6911dc93a09dd3b8a9d63532ca",
    "scripts/vvae_inference.py": "0f3c30fac197a2e3a6e40270e7c2fa3a67f5108cc931c840a8505c26201275d8",
    "setup.sh": "228fa7297c72ccfd06623caa69ed2b1bbbdecae600da714d26709e56e7829f45",
    "utils/export.py": "17230d9dea32a910d89f9819a211aa0f74ab4d092aca101b2f12015c8c6c386e",
    "utils/inference.py": "64a586dd1c47cb121ac9643976400e5d9f8e9e42cbfac776ceb2c24cd7c16361",
    "utils/load.py": "c4fb0928d7191cf3d8080c0f2a63938d26be45a0873d16e2e832cd47c73b56bd",
    "utils/logging.py": "c99a8534b6374cc14c53645ad8c902a74cd0f1bd98bc80dd567bea0e61c387c3",
}


@dataclass(frozen=True)
class PortableBuildReport:
    portable_root: str
    upstream_revision: str
    space_revision: str
    reused: bool
    portable_precision_env: bool
    patched_files: tuple[str, ...]
    overlaid_files: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


class PortablePatchError(RuntimeError):
    """Raised when the pinned source cannot be patched safely."""


def _ignored_source_entry(relative: str, is_directory: bool) -> bool:
    parts = Path(relative).parts
    if ".git" in parts or "__pycache__" in parts:
        return True
    return not is_directory and Path(relative).suffix == ".pyc"


def _ignored_runtime_entry(relative: str, is_directory: bool) -> bool:
    # Upstream scripts are launched with ``python -I -B``. Therefore bytecode
    # caches are neither expected nor harmless here: an injected .pyc or native
    # binary must invalidate the generated runtime instead of being hidden by
    # an ignore rule. Only the deterministic marker itself is excluded.
    return not is_directory and relative == PORTABLE_MARKER


def _directories_for_files(files: Iterable[str]) -> tuple[str, ...]:
    directories: set[str] = set()
    for relative in files:
        path = Path(relative)
        directories.update(
            parent.as_posix() for parent in path.parents if parent != Path(".")
        )
    return tuple(sorted(directories))


def _source_hashes() -> dict[str, str]:
    normalized: dict[str, str] = {}
    for relative, digest in EXPECTED_SOURCE_SHA256.items():
        canonical = Path(relative).as_posix()
        canonical = Path(os.path.normpath(canonical)).as_posix()
        if canonical in normalized:
            raise AssertionError(f"duplicate source hash entry: {canonical}")
        normalized[canonical] = digest
    return normalized


def _validate_upstream(upstream_root: Path) -> TreeInventory:
    if not upstream_root.is_dir():
        raise PortablePatchError(f"LATO.2 source directory is missing: {upstream_root}")
    expected_files = _source_hashes()
    expected = TreeInventory(
        files=expected_files,
        directories=_directories_for_files(expected_files),
    )
    try:
        actual = inventory_tree(upstream_root, ignore=_ignored_source_entry)
    except TreeIntegrityError as exc:
        raise PortablePatchError(str(exc)) from exc
    if not inventories_equal(actual, expected):
        missing = sorted(set(expected.files) - set(actual.files))
        extra = sorted(set(actual.files) - set(expected.files))
        changed = sorted(
            relative
            for relative in set(expected.files) & set(actual.files)
            if expected.files[relative] != actual.files[relative]
        )
        details = []
        if missing:
            details.append(f"missing files: {', '.join(missing[:8])}")
        if extra:
            details.append(f"unexpected files: {', '.join(extra[:8])}")
        if changed:
            details.append(f"changed files: {', '.join(changed[:8])}")
        if actual.directories != expected.directories:
            details.append("directory inventory differs")
        raise PortablePatchError(
            "LATO.2 source does not match the extension-pinned complete inventory; "
            f"refusing to apply a partial portable overlay ({'; '.join(details)})"
        )
    return actual


def _read_text(root: Path, relative: str) -> str:
    return (root / relative).read_text(encoding="utf-8")


def _write_text(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.write_text(text, encoding="utf-8", newline="")


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise PortablePatchError(
            f"portable patch {label!r} expected one source match, found {count}"
        )
    return text.replace(old, new, 1)


def _patch_file(
    root: Path, relative: str, transform: Callable[[str], str]
) -> None:
    original = _read_text(root, relative)
    patched = transform(original)
    if patched == original:
        raise PortablePatchError(f"portable patch made no change to {relative}")
    _write_text(root, relative, patched)


def _patch_sparse_init(text: str) -> str:
    text = _replace_once(
        text,
        "BACKEND = 'spconv' \nDEBUG = False\nATTN = 'flash_attn'",
        "# Portable defaults; spconv/torchsparse and flash_attn/xformers remain selectable.\n"
        "BACKEND = 'torch'\nDEBUG = False\nATTN = 'sdpa'",
        "portable sparse defaults",
    )
    text = _replace_once(
        text,
        "env_sparse_backend in ['spconv', 'torchsparse']",
        "env_sparse_backend in ['spconv', 'torchsparse', 'torch']",
        "torch sparse environment option",
    )
    text = _replace_once(
        text,
        "env_sparse_attn in ['xformers', 'flash_attn']",
        "env_sparse_attn in ['xformers', 'flash_attn', 'sdpa']",
        "SDPA environment option",
    )
    text = _replace_once(
        text,
        "Literal['spconv', 'torchsparse']",
        "Literal['spconv', 'torchsparse', 'torch']",
        "torch backend type",
    )
    return _replace_once(
        text,
        "Literal['xformers', 'flash_attn']",
        "Literal['xformers', 'flash_attn', 'sdpa']",
        "SDPA backend type",
    )


def _patch_sparse_basic(text: str) -> str:
    text = _replace_once(
        text,
        "            elif BACKEND == 'spconv':\n"
        "                SparseTensorData = importlib.import_module('spconv.pytorch').SparseConvTensor\n",
        "            elif BACKEND == 'spconv':\n"
        "                SparseTensorData = importlib.import_module('spconv.pytorch').SparseConvTensor\n"
        "            elif BACKEND == 'torch':\n"
        "                SparseTensorData = importlib.import_module('.lite', __package__).LiteSparseConvTensor\n",
        "pure PyTorch tensor class",
    )
    import_block = (
        "            elif BACKEND == 'spconv':\n"
        "                SparseTensorData = importlib.import_module('spconv.pytorch').SparseConvTensor\n"
        "            elif BACKEND == 'torch':\n"
        "                SparseTensorData = importlib.import_module('.lite', __package__).LiteSparseConvTensor\n"
    )
    head, separator, tail = text.partition(import_block)
    if not separator:
        raise PortablePatchError("portable sparse basic import block disappeared")
    remaining = tail.count("elif BACKEND == 'spconv':")
    if remaining != 7:
        raise PortablePatchError(
            f"portable sparse basic patch expected 7 spconv branches, found {remaining}"
        )
    tail = tail.replace(
        "elif BACKEND == 'spconv':", "elif BACKEND in ('spconv', 'torch'):"
    )
    return head + separator + tail


def _patch_sparse_conv_init(text: str) -> str:
    return _replace_once(
        text,
        "elif BACKEND == 'spconv':\n    from .conv_spconv import *",
        "elif BACKEND == 'spconv':\n    from .conv_spconv import *\n"
        "elif BACKEND == 'torch':\n    from .conv_torch import *",
        "pure PyTorch sparse convolution import",
    )


_SCATTER_FALLBACK = '''try:
    from torch_scatter import scatter_mean
except ImportError:
    # From the official LATO.2 Space at 25ec65d; used only when the compiled
    # torch-scatter dependency has no wheel for the active platform.
    def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
        d = dim if dim >= 0 else src.dim() + dim
        idx = index
        while idx.dim() < src.dim():
            idx = idx.unsqueeze(0)
        idx = idx.expand_as(src)
        if out is None:
            size = list(src.shape)
            size[d] = int(dim_size) if dim_size is not None else int(idx.max()) + 1
            out = src.new_zeros(size)
        sums = out.scatter_add(d, idx, src)
        counts = torch.zeros_like(out).scatter_add_(d, idx, torch.ones_like(src))
        result = sums / counts.clamp(min=1)
        out.copy_(result)
        return out
'''


def _patch_pointnet(text: str) -> str:
    return _replace_once(
        text,
        "from torch_scatter import scatter_mean\n",
        _SCATTER_FALLBACK,
        "torch-scatter portable fallback",
    )


def _patch_attention_import(text: str) -> str:
    return _replace_once(
        text,
        "elif ATTN == 'flash_attn':\n    import flash_attn\nelse:\n",
        "elif ATTN == 'flash_attn':\n    import flash_attn\n"
        "elif ATTN == 'sdpa':\n    from .sdpa import dense_sdpa, packed_sdpa\n"
        "else:\n",
        "SDPA attention import",
    )


def _patch_full_attention(text: str) -> str:
    text = _patch_attention_import(text)
    return _replace_once(
        text,
        "    else:\n        raise ValueError(f\"Unknown attention module: {ATTN}\")\n    \n"
        "    if s is not None:\n",
        "    elif ATTN == 'sdpa':\n"
        "        if num_all_args == 1:\n"
        "            q, k, v = qkv.unbind(dim=1)\n"
        "        elif num_all_args == 2:\n"
        "            k, v = kv.unbind(dim=1)\n"
        "        out = packed_sdpa(q, k, v, q_seqlen, kv_seqlen)\n"
        "    else:\n        raise ValueError(f\"Unknown attention module: {ATTN}\")\n    \n"
        "    if s is not None:\n",
        "full sparse SDPA branch",
    )


def _patch_window_attention(text: str) -> str:
    text = _patch_attention_import(text)
    text = _replace_once(
        text,
        "        elif ATTN == 'flash_attn':\n"
        "            out = flash_attn.flash_attn_qkvpacked_func(qkv_feats)   # [B, N, H, C]\n"
        "        else:\n",
        "        elif ATTN == 'flash_attn':\n"
        "            out = flash_attn.flash_attn_qkvpacked_func(qkv_feats)   # [B, N, H, C]\n"
        "        elif ATTN == 'sdpa':\n"
        "            q, k, v = qkv_feats.unbind(dim=2)\n"
        "            out = dense_sdpa(q, k, v)\n"
        "        else:\n",
        "fixed-length sparse SDPA branch",
    )
    return _replace_once(
        text,
        "        elif ATTN == 'flash_attn':\n"
        "            cu_seqlens = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seq_lens), dim=0)], dim=0) \\\n"
        "                        .to(qkv.device).int()\n"
        "            out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv_feats, cu_seqlens, max(seq_lens)) # [M, H, C]\n",
        "        elif ATTN == 'flash_attn':\n"
        "            cu_seqlens = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seq_lens), dim=0)], dim=0) \\\n"
        "                        .to(qkv.device).int()\n"
        "            out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv_feats, cu_seqlens, max(seq_lens)) # [M, H, C]\n"
        "        elif ATTN == 'sdpa':\n"
        "            q, k, v = qkv_feats.unbind(dim=1)\n"
        "            out = packed_sdpa(q, k, v, seq_lens)\n",
        "variable-length sparse SDPA branch",
    )


def _patch_precision_script(text: str) -> str:
    headless_block = (
        'os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/runtime-root")\n'
        'os.makedirs(os.environ["XDG_RUNTIME_DIR"], exist_ok=True)\n'
        "# Open3D headless rendering: without this the default EGL platform can hang\n"
        "# (e.g. when every GPU is busy or no display device is exposed).\n"
        'os.environ.setdefault("EGL_PLATFORM", "surfaceless")\n'
    )
    if headless_block in text:
        text = _replace_once(
            text,
            headless_block,
            "# XDG/EGL are Unix headless-rendering controls. Setting /tmp on Windows\n"
            "# can target an unwritable drive root and EGL is not Open3D's Windows path.\n"
            'if os.name != "nt":\n'
            '    os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/runtime-root")\n'
            '    os.makedirs(os.environ["XDG_RUNTIME_DIR"], exist_ok=True)\n'
            '    os.environ.setdefault("EGL_PLATFORM", "surfaceless")\n',
            "Windows-safe Open3D environment",
        )
    text = _replace_once(
        text,
        "from utils.load import load_latov2_model\n",
        "from utils.load import load_latov2_model\n"
        "from utils.precision import autocast_context, precision_name\n",
        "precision helper import",
    )
    text = _replace_once(
        text,
        '    device = torch.device("cuda")\n',
        '    device = torch.device("cuda")\n'
        '    logging.info(f"autocast precision: {precision_name()}")\n',
        "precision selection log",
    )
    count = text.count('torch.autocast("cuda", dtype=torch.bfloat16)')
    if count < 1:
        raise PortablePatchError("precision patch found no upstream BF16 context")
    return text.replace(
        'torch.autocast("cuda", dtype=torch.bfloat16)', "autocast_context(device)"
    )


def _patch_renderer_import(text: str) -> str:
    return _replace_once(
        text,
        "            from dataset.mesh_render import WhiteModelRenderer\n",
        "            from dataset.renderer_compat import WhiteModelRenderer\n",
        "Open3D-first renderer adapter",
    )


PATCHES: tuple[tuple[str, Callable[[str], str]], ...] = (
    ("modules/sparse/__init__.py", _patch_sparse_init),
    ("modules/sparse/basic.py", _patch_sparse_basic),
    ("modules/sparse/conv/__init__.py", _patch_sparse_conv_init),
    ("modules/pointnet.py", _patch_pointnet),
    ("modules/sparse/attention/full_attn.py", _patch_full_attention),
    ("modules/sparse/attention/serialized_attn.py", _patch_window_attention),
    ("modules/sparse/attention/windowed_attn.py", _patch_window_attention),
    ("scripts/e2e_inference.py", _patch_precision_script),
    ("scripts/vflow_inference.py", _patch_precision_script),
    ("scripts/vvae_inference.py", _patch_precision_script),
    ("scripts/tflow_inference.py", _patch_precision_script),
    ("dataset/voxel_dataset.py", _patch_renderer_import),
)


def _copy_overlay(overlay_root: Path, destination: Path, files: Iterable[str]) -> None:
    for relative in files:
        source = overlay_root / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _runtime_inventory(root: Path) -> TreeInventory:
    try:
        return inventory_tree(root, ignore=_ignored_runtime_entry)
    except TreeIntegrityError as exc:
        raise PortablePatchError(str(exc)) from exc


def _expected_runtime_inventory(
    upstream_root: Path,
    overlay_root: Path,
) -> tuple[TreeInventory, TreeInventory, TreeInventory]:
    upstream = _validate_upstream(upstream_root)
    try:
        overlay = inventory_tree(overlay_root, ignore=_ignored_source_entry)
    except TreeIntegrityError as exc:
        raise PortablePatchError(str(exc)) from exc

    files = dict(upstream.files)
    for relative, transform in PATCHES:
        try:
            raw = (upstream_root / relative).read_bytes()
            if hashlib.sha256(raw).hexdigest() != _source_hashes()[relative]:
                raise PortablePatchError(
                    f"pinned patch input changed during materialization: {relative}"
                )
            original = raw.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise PortablePatchError(f"cannot read pinned patch input: {relative}") from exc
        patched = transform(original)
        if patched == original:
            raise PortablePatchError(f"portable patch made no change to {relative}")
        files[relative] = hashlib.sha256(patched.encode("utf-8")).hexdigest()
    files.update(overlay.files)
    expected = TreeInventory(
        files=dict(sorted(files.items())),
        directories=_directories_for_files(files),
    )
    return upstream, overlay, expected


def _marker_payload(
    upstream: TreeInventory,
    overlay: TreeInventory,
    output: TreeInventory,
) -> dict[str, object]:
    return {
        "schema_version": PORTABLE_SCHEMA_VERSION,
        "upstream_revision": LATO_SOURCE_REVISION,
        "space_revision": SPACE_SOURCE_REVISION,
        "upstream_tree_sha256": upstream.digest,
        "overlay_sha256": overlay.digest,
        "portable_precision_env": True,
        "default_sparse_backend": "torch",
        "default_attention_backend": "sdpa",
        "renderer_policy": "open3d-first-software-fallback",
        "output_tree_sha256": output.digest,
        "output_sha256": dict(output.files),
    }


def _read_marker(portable_root: Path) -> dict | None:
    try:
        marker_path = portable_root / PORTABLE_MARKER
        info = marker_path.lstat()
        if info.st_size > 1024 * 1024:
            return None
        # The full runtime inventory performs the cross-platform alias check.
        marker = json.loads(marker_path.read_text("utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return marker if isinstance(marker, dict) else None


def validate_portable_runtime(
    portable_root: os.PathLike[str] | str,
    *,
    overlay_root: os.PathLike[str] | str | None = None,
    upstream_root: os.PathLike[str] | str | None = None,
) -> bool:
    """Return ``True`` only for a runtime derived from trusted fixed inputs.

    Runtime callers may omit ``upstream_root`` because Modly stores the exact
    source as the sibling ``LATO.2`` tree. Tests and alternate layouts can pass
    it explicitly. The marker is descriptive only; expected hashes are always
    recomputed from the extension-pinned upstream inventory and overlay.
    """
    destination = Path(portable_root).absolute()
    overlay = (
        Path(overlay_root).absolute()
        if overlay_root is not None
        else Path(__file__).resolve().parent.parent / "portable_overrides"
    )
    upstream = (
        Path(upstream_root).absolute()
        if upstream_root is not None
        else destination.parent / "LATO.2"
    )
    try:
        expected_upstream, expected_overlay, expected_output = _expected_runtime_inventory(
            upstream, overlay
        )
        actual_output = _runtime_inventory(destination)
    except (OSError, PortablePatchError):
        return False
    if not inventories_equal(actual_output, expected_output):
        return False
    marker = _read_marker(destination)
    return marker == _marker_payload(
        expected_upstream, expected_overlay, expected_output
    )


def materialize_portable_runtime(
    upstream_root: os.PathLike[str] | str,
    portable_root: os.PathLike[str] | str,
    *,
    overlay_root: os.PathLike[str] | str | None = None,
) -> PortableBuildReport:
    """Create or reuse the verified portable copy used by Modly setup.

    The function imports no ML dependency and is safe to call before installing
    the extension venv.  ``upstream_root`` and ``portable_root`` must be sibling
    trees; the destination may not be nested inside the source.
    """
    source = Path(upstream_root).absolute()
    destination = Path(portable_root).absolute()
    overlay = (
        Path(overlay_root).absolute()
        if overlay_root is not None
        else Path(__file__).resolve().parent.parent / "portable_overrides"
    )
    if source == destination or source in destination.parents:
        raise PortablePatchError("portable_root must not be inside upstream_root")
    upstream_inventory, overlay_inventory, output_inventory = (
        _expected_runtime_inventory(source, overlay)
    )
    overlay_files = tuple(overlay_inventory.files)
    patched_files = tuple(relative for relative, _ in PATCHES)
    if len(patched_files) != len(set(patched_files)):
        raise PortablePatchError("portable patch table contains duplicate target paths")

    if validate_portable_runtime(
        destination, overlay_root=overlay, upstream_root=source
    ):
        return PortableBuildReport(
            portable_root=str(destination),
            upstream_revision=LATO_SOURCE_REVISION,
            space_revision=SPACE_SOURCE_REVISION,
            reused=True,
            portable_precision_env=True,
            patched_files=patched_files,
            overlaid_files=overlay_files,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staging = destination.parent / f".{destination.name}.staging-{token}"
    backup = destination.parent / f".{destination.name}.backup-{token}"
    try:
        shutil.copytree(
            source,
            staging,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        )
        for relative, transform in PATCHES:
            _patch_file(staging, relative, transform)
        _copy_overlay(overlay, staging, overlay_files)

        actual_staging = _runtime_inventory(staging)
        if not inventories_equal(actual_staging, output_inventory):
            raise PortablePatchError(
                "staged portable runtime differs from its trusted derivation"
            )
        marker = _marker_payload(
            upstream_inventory, overlay_inventory, output_inventory
        )
        (staging / PORTABLE_MARKER).write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="",
        )

        if entry_exists(destination):
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except BaseException:
            if entry_exists(backup) and not entry_exists(destination):
                os.replace(backup, destination)
            raise
        if entry_exists(backup):
            remove_owned_entry(backup, destination.parent)
    finally:
        if entry_exists(staging):
            remove_owned_entry(staging, destination.parent)

    if not validate_portable_runtime(
        destination, overlay_root=overlay, upstream_root=source
    ):
        raise PortablePatchError("portable runtime failed post-publication validation")
    return PortableBuildReport(
        portable_root=str(destination),
        upstream_revision=LATO_SOURCE_REVISION,
        space_revision=SPACE_SOURCE_REVISION,
        reused=False,
        portable_precision_env=True,
        patched_files=patched_files,
        overlaid_files=overlay_files,
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--portable-root", required=True)
    parser.add_argument("--overlay-root")
    args = parser.parse_args(argv)
    try:
        report = materialize_portable_runtime(
            args.upstream_root,
            args.portable_root,
            overlay_root=args.overlay_root,
        )
    except Exception as exc:
        print(f"portable LATO.2 materialization failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
