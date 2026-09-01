"""Immutable source and model inventory for the complete LATO.2 runtime."""

from __future__ import annotations

from dataclasses import dataclass


EXTENSION_ID = "modly-lato2-extension"
EXTENSION_VERSION = "1.1.0"

LATO_REPO = "LoHhhha/LATO.2"
LATO_SOURCE_REVISION = "fbb1f5a5755e6db8700cf6922fd506830b7cdccd"
LATO_MODEL_REPO = "0x4c48/LATO.2"
LATO_MODEL_REVISION = "a91090e8077b9318ab87ac08fd9eb905903d4515"

DINO_REPO = "facebookresearch/dinov2"
DINO_SOURCE_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_CHECKPOINT_FILENAME = "dinov2_vitl14_reg4_pretrain.pth"
DINO_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/"
    "dinov2_vitl14_reg4_pretrain.pth"
    "?versionId=HLmbhvcd2hPq9CNLwMvwswbRlzZRuOeA"
)

# Human-readable short revisions are used only as an owned directory name. The
# complete hashes above remain the authority for every download and marker.
REVISION_ID = "lato-a91090e-dino-7764ea0f"
READY_MARKER_FILENAME = "complete.json"
RUNTIME_CONFIG_FILENAME = "runtime_config.json"
SETUP_LOCK_FILENAME = ".setup.lock"
READY_SCHEMA_VERSION = 1
RUNTIME_CONFIG_SCHEMA_VERSION = 1

LATO_SOURCE_PATH = "source/LATO.2"
DINO_HUB_DIR_PATH = "dinov2"
DINO_SOURCE_PATH = "dinov2/facebookresearch_dinov2_main"
DINO_CHECKPOINT_PATH = f"dinov2/checkpoints/{DINO_CHECKPOINT_FILENAME}"


@dataclass(frozen=True)
class AssetSpec:
    """One exact downloadable file in the immutable snapshot."""

    relative_path: str
    size: int
    sha256: str
    url: str
    role: str


@dataclass(frozen=True)
class SourceArchiveSpec:
    """A verified source ZIP and its exact extracted destination."""

    asset_path: str
    destination: str
    expected_archive_root: str


def _hf_url(filename: str) -> str:
    return (
        f"https://huggingface.co/{LATO_MODEL_REPO}/resolve/"
        f"{LATO_MODEL_REVISION}/{filename}?download=true"
    )


LATO_CHECKPOINT_SPECS = (
    AssetSpec(
        "ckpt/offset_head.pt",
        4_229_424,
        "b9ab60321ffa1026813b4cb79eba7a9d224ccd77e01e6b2c84fd29cad301c9df",
        _hf_url("offset_head.pt"),
        "lato-checkpoint",
    ),
    AssetSpec(
        "ckpt/tflow.pt",
        882_821_662,
        "5ac0033544b813ec046f00d5f12d3b488a91a98c1bfef0560948fb4c6b52c6ad",
        _hf_url("tflow.pt"),
        "lato-checkpoint",
    ),
    AssetSpec(
        "ckpt/tvae.pt",
        713_397_214,
        "b109bcf2bc42abea89a3563e3fefa9509f9cd523501d5c62d78108380a3db17c",
        _hf_url("tvae.pt"),
        "lato-checkpoint",
    ),
    AssetSpec(
        "ckpt/vdf_encoder.pt",
        6_673_155,
        "79c461652bc9307126ef02e5ada4a4e6c6434bc00b9d6c05043fe39a9f0819ce",
        _hf_url("vdf_encoder.pt"),
        "lato-checkpoint",
    ),
    AssetSpec(
        "ckpt/vflow.pt",
        649_153_594,
        "a5c660601af9ce9909feb5157cfd456d81b0498835d7ba4129a7d4267a8927d3",
        _hf_url("vflow.pt"),
        "lato-checkpoint",
    ),
    AssetSpec(
        "ckpt/voxel_encoder.pt",
        61_373_424,
        "c304e36651f2a8c63d5b30d0716a4a8ab4ef505fe39311d060e04e5ab253d4ce",
        _hf_url("voxel_encoder.pt"),
        "lato-checkpoint",
    ),
    AssetSpec(
        "ckpt/vvae.pt",
        1_261_420_852,
        "b1192e3e6301dee5ee3dac37d411f82141ee1a8764707f0ea5b7746052342d44",
        _hf_url("vvae.pt"),
        "lato-checkpoint",
    ),
)

SOURCE_ARCHIVE_ASSETS = (
    AssetSpec(
        "_archives/lato2-fbb1f5a5755e6db8700cf6922fd506830b7cdccd.zip",
        17_750_926,
        "ccda4965de16f77406e7101d08eeace3191de9b106d7efb69392225d89f55138",
        (
            "https://codeload.github.com/LoHhhha/LATO.2/zip/"
            "fbb1f5a5755e6db8700cf6922fd506830b7cdccd"
        ),
        "source-archive",
    ),
    AssetSpec(
        "_archives/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8.zip",
        3_001_681,
        "04276715cddb29d45d05bff3a6fc132224dc27749b279ac98ad2ce4620e20d48",
        (
            "https://codeload.github.com/facebookresearch/dinov2/zip/"
            "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
        ),
        "source-archive",
    ),
)

DINO_CHECKPOINT_SPEC = AssetSpec(
    DINO_CHECKPOINT_PATH,
    1_217_607_321,
    "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51",
    DINO_CHECKPOINT_URL,
    "dinov2-checkpoint",
)

ASSETS = LATO_CHECKPOINT_SPECS + SOURCE_ARCHIVE_ASSETS + (DINO_CHECKPOINT_SPEC,)

SOURCE_ARCHIVES = (
    SourceArchiveSpec(
        SOURCE_ARCHIVE_ASSETS[0].relative_path,
        LATO_SOURCE_PATH,
        f"LATO.2-{LATO_SOURCE_REVISION}",
    ),
    SourceArchiveSpec(
        SOURCE_ARCHIVE_ASSETS[1].relative_path,
        DINO_SOURCE_PATH,
        f"dinov2-{DINO_SOURCE_REVISION}",
    ),
)

LATO_CHECKPOINT_PATHS = {
    spec.relative_path.removeprefix("ckpt/").removesuffix(".pt"): spec.relative_path
    for spec in LATO_CHECKPOINT_SPECS
}

# Runtime authentication follows the exact checkpoint routes used by each
# pinned upstream entry point.  Keeping this mapping beside the immutable
# hashes makes additions fail closed instead of silently running an
# unauthenticated checkpoint.
NODE_LATO_CHECKPOINTS = {
    "lato2-e2e": (
        "vflow",
        "vvae",
        "offset_head",
        "tflow",
        "tvae",
        "voxel_encoder",
    ),
    "lato2-vflow": ("vflow", "vvae", "vdf_encoder", "offset_head"),
    "lato2-vvae": ("vvae", "vdf_encoder", "offset_head"),
    "lato2-tflow": ("tflow", "tvae", "voxel_encoder"),
}

DINO_NODE_IDS = frozenset({"lato2-e2e", "lato2-vflow"})
