from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.error import HTTPError
from zipfile import ZipFile

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from lato2_modly import assets
from lato2_modly.constants import (
    ASSETS,
    DINO_CHECKPOINT_SPEC,
    DINO_SOURCE_REVISION,
    LATO_CHECKPOINT_SPECS,
    LATO_MODEL_REPO,
    LATO_MODEL_REVISION,
    LATO_SOURCE_REVISION,
    REVISION_ID,
    SOURCE_ARCHIVE_ASSETS,
    AssetSpec,
    SourceArchiveSpec,
)
from lato2_modly.paths import PathContractError, resolve_models_root
from lato2_modly.state import StateError, read_runtime_config, write_runtime_config
from lato2_modly.integrity import TreeIntegrityError, inventory_tree, remove_owned_entry


class FakeResponse:
    def __init__(
        self,
        data: bytes,
        *,
        status: int = 200,
        content_range: str = "",
        include_length: bool = True,
    ) -> None:
        self.stream = io.BytesIO(data)
        self.status = status
        self.headers: dict[str, str] = {}
        if include_length:
            self.headers["Content-Length"] = str(len(data))
        if content_range:
            self.headers["Content-Range"] = content_range

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.stream.read(size)


def make_spec(data: bytes, path: str = "asset.bin") -> AssetSpec:
    return AssetSpec(
        path,
        len(data),
        hashlib.sha256(data).hexdigest(),
        "https://example.test/a",
        "test",
    )


class AssetPathStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_upstream_inventory_is_exact_and_public(self) -> None:
        self.assertEqual(LATO_MODEL_REPO, "0x4c48/LATO.2")
        self.assertEqual(LATO_MODEL_REVISION, "a91090e8077b9318ab87ac08fd9eb905903d4515")
        self.assertEqual(LATO_SOURCE_REVISION, "fbb1f5a5755e6db8700cf6922fd506830b7cdccd")
        self.assertEqual(DINO_SOURCE_REVISION, "7764ea0f912e53c92e82eb78a2a1631e92725fc8")
        self.assertEqual(REVISION_ID, "lato-a91090e-dino-7764ea0f")
        self.assertEqual(len(LATO_CHECKPOINT_SPECS), 7)
        self.assertEqual(
            {Path(spec.relative_path).name for spec in LATO_CHECKPOINT_SPECS},
            {
                "offset_head.pt",
                "tflow.pt",
                "tvae.pt",
                "vdf_encoder.pt",
                "vflow.pt",
                "voxel_encoder.pt",
                "vvae.pt",
            },
        )
        self.assertEqual(len(ASSETS), 10)
        self.assertTrue(all(len(spec.sha256) == 64 and spec.size > 0 for spec in ASSETS))
        for spec in LATO_CHECKPOINT_SPECS:
            self.assertTrue(
                spec.url.startswith(
                    "https://huggingface.co/0x4c48/LATO.2/resolve/"
                    "a91090e8077b9318ab87ac08fd9eb905903d4515/"
                )
            )
            self.assertNotIn("/main/", spec.url)
            self.assertNotIn("token=", spec.url.casefold())
        self.assertEqual(SOURCE_ARCHIVE_ASSETS[0].size, 17_750_926)
        self.assertEqual(
            SOURCE_ARCHIVE_ASSETS[0].sha256,
            "ccda4965de16f77406e7101d08eeace3191de9b106d7efb69392225d89f55138",
        )
        self.assertEqual(SOURCE_ARCHIVE_ASSETS[1].size, 3_001_681)
        self.assertEqual(
            SOURCE_ARCHIVE_ASSETS[1].sha256,
            "04276715cddb29d45d05bff3a6fc132224dc27749b279ac98ad2ce4620e20d48",
        )
        self.assertEqual(DINO_CHECKPOINT_SPEC.size, 1_217_607_321)
        self.assertEqual(
            DINO_CHECKPOINT_SPEC.sha256,
            "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51",
        )

    def test_valid_asset_is_zero_network_and_removes_stale_part(self) -> None:
        data = b"already verified"
        spec = make_spec(data)
        (self.root / spec.relative_path).write_bytes(data)
        (self.root / f"{spec.relative_path}.part").write_bytes(b"stale")

        def forbidden(*_args: object, **_kwargs: object) -> object:
            self.fail("network must not be opened")

        result = assets.ensure_asset(
            self.root, spec, opener=forbidden, log=lambda _message: None
        )
        self.assertEqual(result.read_bytes(), data)
        self.assertFalse((self.root / f"{spec.relative_path}.part").exists())

    def test_partial_download_uses_validated_range_without_authorization(self) -> None:
        data = b"abcdefghij"
        spec = make_spec(data)
        (self.root / "asset.bin.part").write_bytes(data[:3])

        def opener(request: object, *, timeout: float) -> FakeResponse:
            self.assertEqual(request.headers["Range"], "bytes=3-")  # type: ignore[attr-defined]
            self.assertNotIn("Authorization", request.headers)  # type: ignore[attr-defined]
            return FakeResponse(data[3:], status=206, content_range="bytes 3-9/10")

        result = assets.ensure_asset(
            self.root, spec, opener=opener, log=lambda _message: None
        )
        self.assertEqual(result.read_bytes(), data)

    def test_hardlinked_partial_is_rejected_without_modifying_external_file(self) -> None:
        data = b"abcdefghij"
        spec = make_spec(data)
        outside = self.root / "outside.bin"
        outside.write_bytes(b"SECRET")
        part = self.root / "asset.bin.part"
        try:
            os.link(outside, part)
        except OSError as exc:
            self.skipTest(f"hardlinks unavailable: {exc}")
        with self.assertRaises(assets.AssetError) as raised:
            assets.ensure_asset(
                self.root,
                spec,
                opener=lambda *_args, **_kwargs: FakeResponse(data),
                log=lambda _message: None,
                retries=1,
            )
        self.assertEqual(raised.exception.code, "ASSET_PART_INVALID")
        self.assertEqual(outside.read_bytes(), b"SECRET")

    def test_hardlinked_complete_asset_is_not_authenticated(self) -> None:
        data = b"verified bytes"
        spec = make_spec(data)
        outside = self.root / "outside.bin"
        outside.write_bytes(data)
        target = self.root / spec.relative_path
        try:
            os.link(outside, target)
        except OSError as exc:
            self.skipTest(f"hardlinks unavailable: {exc}")
        valid, reason = assets.verify_asset(target, spec)
        self.assertFalse(valid)
        self.assertIn("regular local file", reason)

    def test_owned_removal_rejects_symlink_parent_without_deleting_target(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        victim = outside / "victim.txt"
        victim.write_text("keep", encoding="utf-8")
        alias = self.root / "alias"
        try:
            alias.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"directory symlinks unavailable: {exc}")
        with self.assertRaises(TreeIntegrityError):
            remove_owned_entry(alias / "victim.txt", alias)
        self.assertEqual(victim.read_text(encoding="utf-8"), "keep")

    def test_inventory_rejects_hardlink_even_when_the_entry_is_ignored(self) -> None:
        tree = self.root / "tree"
        ignored = tree / "ignored"
        ignored.mkdir(parents=True)
        outside = self.root / "outside.bin"
        outside.write_bytes(b"keep")
        try:
            os.link(outside, ignored / "cached.bin")
        except OSError as exc:
            self.skipTest(f"hardlinks unavailable: {exc}")
        with self.assertRaisesRegex(TreeIntegrityError, "hardlink"):
            inventory_tree(tree, ignore=lambda _relative, _directory: True)

    def test_wrong_content_range_preserves_useful_partial(self) -> None:
        data = b"abcdefghij"
        spec = make_spec(data)
        part = self.root / "asset.bin.part"
        part.write_bytes(data[:3])
        with self.assertRaisesRegex(assets.AssetError, "ASSET_DOWNLOAD_FAILED"):
            assets.ensure_asset(
                self.root,
                spec,
                opener=lambda *_args, **_kwargs: FakeResponse(
                    data[3:], status=206, content_range="bytes 2-9/10"
                ),
                log=lambda _message: None,
                retries=1,
            )
        self.assertEqual(part.read_bytes(), data[:3])

    def test_http_416_restarts_once(self) -> None:
        data = b"abcdefghij"
        spec = make_spec(data)
        (self.root / "asset.bin.part").write_bytes(data[:3])
        calls = 0

        def opener(request: object, *, timeout: float) -> FakeResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise HTTPError(spec.url, 416, "range rejected", None, None)
            self.assertNotIn("Range", request.headers)  # type: ignore[attr-defined]
            return FakeResponse(data)

        result = assets.ensure_asset(
            self.root, spec, opener=opener, log=lambda _message: None
        )
        self.assertEqual(result.read_bytes(), data)
        self.assertEqual(calls, 2)

    def test_source_archive_extracts_exact_tree_and_rejects_extra_file(self) -> None:
        archive = self.root / "source.zip"
        with ZipFile(archive, "w") as bundle:
            bundle.writestr("repo-pin/", b"")
            bundle.writestr("repo-pin/LICENSE", b"license\n")
            bundle.writestr("repo-pin/package/__init__.py", b"VALUE = 1\n")
        source = SourceArchiveSpec("source.zip", "source/Repo", "repo-pin")

        installed = assets.ensure_source_tree(self.root, source, log=lambda _message: None)
        self.assertEqual((installed / "LICENSE").read_text(encoding="utf-8"), "license\n")
        self.assertEqual(assets.verify_source_tree(self.root, source), (True, "valid"))
        (installed / "unexpected.txt").write_text("no", encoding="utf-8")
        valid, reason = assets.verify_source_tree(self.root, source)
        self.assertFalse(valid)
        self.assertIn("inventory", reason)

    def test_models_root_precedence_payload_modly_env_legacy_api_manual(self) -> None:
        extension = self.root / "extensions" / "modly-lato2-extension"
        extension.mkdir(parents=True)
        roots = {name: self.root / name for name in ("payload", "modly", "legacy", "api", "manual")}
        for path in roots.values():
            path.mkdir()

        def forbidden(*_args: object, **_kwargs: object) -> object:
            self.fail("higher-precedence path must avoid the API")

        self.assertEqual(
            resolve_models_root(
                {"models_dir": str(roots["payload"])},
                extension,
                environ={"MODLY_MODELS_DIR": str(roots["modly"])},
                opener=forbidden,
            ),
            roots["payload"],
        )
        self.assertEqual(
            resolve_models_root(
                {},
                extension,
                environ={"MODLY_MODELS_DIR": str(roots["modly"])},
                opener=forbidden,
            ),
            roots["modly"],
        )
        self.assertEqual(
            resolve_models_root(
                {},
                extension,
                environ={
                    "MODELS_DIR": str(roots["legacy"]),
                    "WORKSPACE_DIR": str(self.root / "work"),
                },
                opener=forbidden,
            ),
            roots["legacy"],
        )
        self.assertEqual(
            resolve_models_root(
                {},
                extension,
                environ={"MODELS_DIR": str(roots["manual"])},
                opener=lambda *_args, **_kwargs: self._api_response(roots["api"]),
            ),
            roots["api"],
        )
        self.assertEqual(
            resolve_models_root(
                {},
                extension,
                environ={"MODELS_DIR": str(roots["manual"])},
                opener=lambda *_args, **_kwargs: self._api_response(roots["api"], status=503),
            ),
            roots["manual"],
        )

    def _api_response(self, models: Path, status: int = 200) -> FakeResponse:
        body = json.dumps(
            {"models_dir": str(models), "workspace_dir": str(models.parent / "work")}
        ).encode()
        return FakeResponse(body, status=status)

    def test_conflicting_payload_model_paths_fail(self) -> None:
        first = self.root / "first"
        second = self.root / "second"
        first.mkdir()
        second.mkdir()

        def forbidden(*_args: object, **_kwargs: object) -> object:
            self.fail("conflict must fail before network")

        with self.assertRaisesRegex(PathContractError, "PATH_MODELS_CONFLICT"):
            resolve_models_root(
                {"models_dir": str(first), "modelsDir": str(second)},
                self.root,
                environ={},
                opener=forbidden,
            )

    def test_runtime_config_roundtrip_is_atomic_and_extensible(self) -> None:
        extension = self.root / "extension"
        models = self.root / "models"
        revision = models / "modly-lato2-extension" / "lato2" / "revisions" / REVISION_ID
        extension.mkdir()
        revision.mkdir(parents=True)
        path = write_runtime_config(
            extension,
            models,
            revision,
            extra={"default_backend": "upstream", "available_backends": ["upstream"]},
        )
        parsed = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(parsed["schema_version"], 1)
        self.assertEqual(parsed["default_backend"], "upstream")
        loaded = read_runtime_config(extension)
        self.assertEqual(loaded.models_dir, models.resolve())
        self.assertEqual(loaded.revision_dir, revision.resolve())

    def test_runtime_config_rejects_secret_fields(self) -> None:
        extension = self.root / "extension"
        models = self.root / "models"
        revision = models / "modly-lato2-extension" / "lato2" / "revisions" / REVISION_ID
        extension.mkdir()
        revision.mkdir(parents=True)
        with self.assertRaisesRegex(StateError, "STATE_SECRET_REJECTED"):
            write_runtime_config(extension, models, revision, extra={"hf_token": "secret"})

    def test_runtime_config_reader_rejects_hardlinks(self) -> None:
        extension = self.root / "extension"
        models = self.root / "models"
        revision = models / "modly-lato2-extension" / "lato2" / "revisions" / REVISION_ID
        extension.mkdir()
        revision.mkdir(parents=True)
        path = write_runtime_config(extension, models, revision)
        outside = self.root / "outside.json"
        path.replace(outside)
        try:
            os.link(outside, path)
        except OSError as exc:
            self.skipTest(f"hardlinks unavailable: {exc}")
        with self.assertRaisesRegex(StateError, "STATE_FILE_INVALID"):
            read_runtime_config(extension)


if __name__ == "__main__":
    unittest.main()
