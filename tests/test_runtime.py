from __future__ import annotations

import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock
from zipfile import ZipFile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lato2_modly import runtime  # noqa: E402
from lato2_modly.constants import (  # noqa: E402
    AssetSpec,
    DINO_CHECKPOINT_SPEC,
    DINO_NODE_IDS,
    DINO_SOURCE_PATH,
    LATO_CHECKPOINT_PATHS,
    LATO_SOURCE_PATH,
    NODE_LATO_CHECKPOINTS,
    SourceArchiveSpec,
)


def _fake_state(root: Path, *, available=("upstream", "portable"), default="upstream"):
    revision = root / "revision"
    source = revision / "source" / "LATO.2"
    portable = revision / "source" / "LATO.2-portable"
    checkpoints = revision / "ckpt"
    dino = revision / "dinov2"
    for tree in (source, portable):
        (tree / "scripts").mkdir(parents=True, exist_ok=True)
        for name in ("e2e_inference.py", "vflow_inference.py", "vvae_inference.py", "tflow_inference.py"):
            (tree / "scripts" / name).write_text("# fixture\n", encoding="utf-8")
    checkpoints.mkdir(parents=True)
    for relative in LATO_CHECKPOINT_PATHS.values():
        target = revision.joinpath(*relative.split("/"))
        target.write_bytes(b"fixture")
    dino.mkdir(parents=True)
    return runtime.RuntimeState(
        models_root=root / "models",
        revision_root=revision,
        source_root=source,
        checkpoints=checkpoints,
        dino_hub=dino,
        default_backend=default,
        available_backends=frozenset(available),
        attention_backend="xformers" if "upstream" in available else "sdpa",
        portable_precision_env="portable" in available,
        portable_precisions=frozenset({"auto", "bfloat16", "float16"}),
    )


def _request(root: Path, node_id: str, *, backend="upstream", overrides=None):
    state = _fake_state(root)
    params = dict(runtime.NODE_DEFAULTS[node_id])
    params.update(overrides or {})
    params["backend"] = backend
    for name in runtime.BOOL_PARAMETERS & params.keys():
        params[name] = params[name] == "true" if isinstance(params[name], str) else params[name]
    return runtime.ValidatedRequest(
        node_id=node_id,
        input_path=root / "input.glb",
        params=params,
        workspace_dir=root / "workspace",
        temp_dir=root / "temp",
        state=state,
        backend=backend,
        precision=params["precision"],
    )


class ManifestRuntimeContractTests(unittest.TestCase):
    def test_frozen_upstream_tunable_inventory_is_fully_exposed(self):
        common = {"seed", "num_workers"}
        expected = {
            "lato2-e2e": common
            | {
                "inference_threshold", "vflow_steps", "cfg_strength", "rescale_t",
                "vert_num", "use_gt_vert_count", "scaler", "min_verts", "max_verts",
                "tflow_steps", "edge_threshold", "chunk_size", "fill_quad_rings",
                "render_azimuth", "render_elevation", "img_res",
            },
            "lato2-vflow": common
            | {
                "pc_sample_number", "sample_type", "inference_threshold", "reconstruct",
                "sample_posterior", "vflow_steps", "cfg_strength", "rescale_t", "vert_num",
                "use_gt_vert_count", "scaler", "min_verts", "max_verts", "render_azimuth",
                "render_elevation", "img_res",
            },
            "lato2-vvae": common
            | {"pc_sample_number", "sample_type", "inference_threshold", "sample_posterior"},
            "lato2-tflow": common
            | {"tflow_steps", "use_cond", "edge_threshold", "chunk_size", "fill_quad_rings", "save_voxel_field"},
        }
        for node_id, tunables in expected.items():
            self.assertEqual(
                set(runtime.NODE_DEFAULTS[node_id]) - {"backend", "precision"},
                tunables,
            )

    def test_all_manifest_defaults_match_runtime(self):
        manifest = json.loads((REPO_ROOT / "manifest.json").read_text("utf-8"))
        self.assertEqual([node["id"] for node in manifest["nodes"]], list(runtime.NODE_IDS))
        for node in manifest["nodes"]:
            defaults = {item["id"]: item["default"] for item in node["params_schema"]}
            self.assertEqual(defaults, runtime.NODE_DEFAULTS[node["id"]])
            self.assertEqual(node["input"], "mesh")
            self.assertEqual(node["output"], "mesh")

    def test_every_exposed_upstream_parameter_is_forwarded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "one-input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            for node_id, defaults in runtime.NODE_DEFAULTS.items():
                request = _request(root / node_id, node_id)
                command = runtime._command(request, input_dir, output_dir)
                self.assertEqual(command[1:3], ["-I", "-B"])
                self.assertEqual(command[3:5], ["-X", "utf8"])
                self.assertEqual(command[command.index("--batch_size") + 1], "1")
                self.assertEqual(command[command.index("--num_samples") + 1], "1")
                for name in defaults:
                    if name in {"backend", "precision"}:
                        continue
                    cli_name = (
                        "steps"
                        if node_id in {"lato2-vflow", "lato2-tflow"}
                        and name in {"vflow_steps", "tflow_steps"}
                        else name
                    )
                    positive = f"--{cli_name}"
                    negative = f"--no-{cli_name}"
                    self.assertTrue(
                        positive in command or negative in command,
                        f"{node_id} does not forward {name}",
                    )

    def test_checkpoint_and_dinov2_routes_are_exact(self):
        expected = {
            node_id: set(checkpoint_names)
            for node_id, checkpoint_names in NODE_LATO_CHECKPOINTS.items()
        }
        seen = set()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "in").mkdir()
            (root / "out").mkdir()
            for node_id, checkpoint_names in expected.items():
                request = _request(root / node_id, node_id)
                command = runtime._command(request, root / "in", root / "out")
                routed = {
                    token.removeprefix("--").removesuffix("_ckpt")
                    for token in command
                    if token.startswith("--") and token.endswith("_ckpt")
                }
                self.assertEqual(routed, checkpoint_names)
                for name in checkpoint_names:
                    flag = f"--{name}_ckpt"
                    self.assertIn(flag, command)
                    self.assertEqual(Path(command[command.index(flag) + 1]).name, f"{name}.pt")
                    seen.add(name)
                if node_id in DINO_NODE_IDS:
                    self.assertEqual(
                        Path(command[command.index("--dino_hub_dir") + 1]).name,
                        "dinov2",
                    )
                else:
                    self.assertNotIn("--dino_hub_dir", command)
            self.assertEqual(seen, set(LATO_CHECKPOINT_PATHS))

    def test_boolean_optional_flags_forward_both_values(self):
        boolean_nodes = {
            node: sorted(runtime.BOOL_PARAMETERS & defaults.keys())
            for node, defaults in runtime.NODE_DEFAULTS.items()
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "in").mkdir()
            (root / "out").mkdir()
            for node_id, names in boolean_nodes.items():
                for name in names:
                    for value, expected in (("true", f"--{name}"), ("false", f"--no-{name}")):
                        request = _request(root / f"{node_id}-{name}-{value}", node_id, overrides={name: value})
                        command = runtime._command(request, root / "in", root / "out")
                    self.assertIn(expected, command)


class RuntimeAuthenticationTests(unittest.TestCase):
    def test_same_size_checkpoint_replacement_is_rejected_by_sha256(self):
        original = b"A" * 64
        replacement = b"B" * len(original)
        spec = AssetSpec(
            relative_path="ckpt/fixture.pt",
            size=len(original),
            sha256=hashlib.sha256(original).hexdigest(),
            url="https://invalid.example/fixture.pt",
            role="lato-checkpoint",
        )
        with tempfile.TemporaryDirectory() as temp:
            revision = Path(temp)
            target = revision / "ckpt" / "fixture.pt"
            target.parent.mkdir()
            target.write_bytes(original)
            runtime._authenticate_asset(revision, spec)
            target.write_bytes(replacement)
            self.assertEqual(target.stat().st_size, spec.size)
            with self.assertRaisesRegex(runtime.ProcessFailure, "SETUP_INVALID"):
                runtime._authenticate_asset(revision, spec)

    def test_same_size_source_tree_and_archive_replacements_are_rejected(self):
        original = b"print(1)\n"
        replacement = b"print(2)\n"
        source_spec = SourceArchiveSpec(
            asset_path="_archives/source.zip",
            destination="source/example",
            expected_archive_root="example-deadbeef",
        )
        with tempfile.TemporaryDirectory() as temp:
            revision = Path(temp)
            archive = revision / "_archives" / "source.zip"
            archive.parent.mkdir()
            with ZipFile(archive, "w") as bundle:
                bundle.writestr("example-deadbeef/", b"")
                bundle.writestr("example-deadbeef/module.py", original)
            extracted = revision / "source" / "example"
            extracted.mkdir(parents=True)
            target = extracted / "module.py"
            target.write_bytes(original)
            archive_spec = AssetSpec(
                relative_path=source_spec.asset_path,
                size=archive.stat().st_size,
                sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                url="https://invalid.example/source.zip",
                role="source-archive",
            )
            with mock.patch.object(runtime, "_pinned_asset_spec", return_value=archive_spec):
                runtime._authenticate_source_tree(revision, source_spec)
                target.write_bytes(replacement)
                self.assertEqual(target.stat().st_size, len(original))
                with self.assertRaisesRegex(runtime.ProcessFailure, "SETUP_INVALID"):
                    runtime._authenticate_source_tree(revision, source_spec)
                target.write_bytes(original)
                archive_bytes = bytearray(archive.read_bytes())
                archive_bytes[-1] ^= 1
                archive.write_bytes(archive_bytes)
                self.assertEqual(archive.stat().st_size, archive_spec.size)
                with self.assertRaisesRegex(runtime.ProcessFailure, "SETUP_INVALID"):
                    runtime._authenticate_source_tree(revision, source_spec)

    def test_each_node_authenticates_exactly_the_assets_it_routes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for node_id, checkpoint_names in NODE_LATO_CHECKPOINTS.items():
                request = _request(root / node_id, node_id)
                authenticated_assets = []
                authenticated_sources = []
                with (
                    mock.patch.object(
                        runtime,
                        "_authenticate_asset",
                        side_effect=lambda _root, spec: authenticated_assets.append(spec),
                    ),
                    mock.patch.object(
                        runtime,
                        "_authenticate_source_tree",
                        side_effect=lambda _root, spec: authenticated_sources.append(spec),
                    ),
                    mock.patch.object(runtime, "validate_portable_runtime") as portable_check,
                ):
                    runtime._authenticate_request(request)
                expected_paths = [LATO_CHECKPOINT_PATHS[name] for name in checkpoint_names]
                if node_id in DINO_NODE_IDS:
                    expected_paths.append(DINO_CHECKPOINT_SPEC.relative_path)
                self.assertEqual(
                    [spec.relative_path for spec in authenticated_assets],
                    expected_paths,
                )
                expected_sources = [LATO_SOURCE_PATH]
                if node_id in DINO_NODE_IDS:
                    expected_sources.append(DINO_SOURCE_PATH)
                self.assertEqual(
                    [spec.destination for spec in authenticated_sources],
                    expected_sources,
                )
                portable_check.assert_not_called()

    def test_portable_backend_authenticates_the_tree_it_executes(self):
        with tempfile.TemporaryDirectory() as temp:
            request = _request(Path(temp), "lato2-tflow", backend="portable")
            with (
                mock.patch.object(runtime, "_authenticate_asset"),
                mock.patch.object(runtime, "_authenticate_source_tree"),
                mock.patch.object(
                    runtime,
                    "validate_portable_runtime",
                    return_value=True,
                ) as portable_check,
            ):
                runtime._authenticate_request(request)
            portable_check.assert_called_once_with(
                request.state.revision_root / "source" / "LATO.2-portable"
            )


class RuntimeConfigIdentityTests(unittest.TestCase):
    def _config(self, revision: Path) -> dict[str, object]:
        return {
            "extension_id": runtime.EXTENSION_ID,
            "extension_version": runtime.EXTENSION_VERSION,
            "revision_id": runtime.REVISION_ID,
            "ready_marker": str(revision / runtime.READY_MARKER_FILENAME),
            "runtime_cache_dir": str(revision / "runtime-cache"),
        }

    def test_runtime_config_identity_is_bound_to_this_exact_release(self):
        with tempfile.TemporaryDirectory() as temp:
            revision = Path(temp) / "revision"
            revision.mkdir()
            (revision / "runtime-cache").mkdir()
            (revision / runtime.READY_MARKER_FILENAME).write_text("{}", encoding="utf-8")
            config = self._config(revision)
            runtime._validate_runtime_config_identity(config, revision)
            for key, value in (
                ("extension_id", "stale-extension"),
                ("extension_version", "0.0.0"),
                ("revision_id", "stale-revision"),
            ):
                with self.subTest(key=key):
                    tampered = dict(config)
                    tampered[key] = value
                    with self.assertRaisesRegex(runtime.ProcessFailure, "SETUP_INVALID"):
                        runtime._validate_runtime_config_identity(tampered, revision)

    def test_runtime_config_rejects_missing_or_diverted_owned_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            revision = root / "revision"
            revision.mkdir()
            (revision / "runtime-cache").mkdir()
            (revision / runtime.READY_MARKER_FILENAME).write_text("{}", encoding="utf-8")
            config = self._config(revision)
            for key in ("ready_marker", "runtime_cache_dir"):
                with self.subTest(key=key, variant="missing"):
                    missing = dict(config)
                    missing.pop(key)
                    with self.assertRaisesRegex(runtime.ProcessFailure, "SETUP_INVALID"):
                        runtime._validate_runtime_config_identity(missing, revision)
            other_cache = root / "other-cache"
            other_cache.mkdir()
            other_marker = root / "other-marker.json"
            other_marker.write_text("{}", encoding="utf-8")
            for key, value in (
                ("ready_marker", other_marker),
                ("runtime_cache_dir", other_cache),
            ):
                with self.subTest(key=key, variant="diverted"):
                    diverted = dict(config)
                    diverted[key] = str(value)
                    with self.assertRaisesRegex(runtime.ProcessFailure, "SETUP_INVALID"):
                        runtime._validate_runtime_config_identity(diverted, revision)

    def test_runtime_json_reader_rejects_hardlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outside = root / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            linked = root / "linked.json"
            try:
                os.link(outside, linked)
            except OSError as exc:
                self.skipTest(f"hardlinks unavailable: {exc}")
            with self.assertRaisesRegex(runtime.ProcessFailure, "SETUP_INVALID"):
                runtime._read_json_file(linked)


class InputBundleSecurityTests(unittest.TestCase):
    @staticmethod
    def _write_gltf(path: Path, *, buffer_uri: str, image_uri: str | None = None) -> None:
        document: dict[str, object] = {
            "asset": {"version": "2.0"},
            "buffers": [{"uri": buffer_uri, "byteLength": 4}],
        }
        if image_uri is not None:
            document["images"] = [{"uri": image_uri}]
        path.write_text(json.dumps(document), encoding="utf-8")

    @staticmethod
    def _write_glb(path: Path, document: dict[str, object]) -> None:
        encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
        encoded += b" " * ((-len(encoded)) % 4)
        total = 12 + 8 + len(encoded)
        path.write_bytes(
            b"glTF"
            + (2).to_bytes(4, "little")
            + total.to_bytes(4, "little")
            + len(encoded).to_bytes(4, "little")
            + (0x4E4F534A).to_bytes(4, "little")
            + encoded
        )

    def test_valid_gltf_buffer_is_confined_copied_and_uri_rewritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            mesh = source / "model.gltf"
            self._write_gltf(
                mesh,
                buffer_uri="buffers/mesh%20data.bin",
                image_uri="data:image/png;base64,AA==",
            )
            buffer = source / "buffers" / "mesh data.bin"
            buffer.parent.mkdir()
            buffer.write_bytes(b"mesh")
            staged = root / "staged"
            main = runtime._stage_input_bundle(mesh, staged)
            document = json.loads(main.read_text("utf-8"))
            self.assertEqual(document["buffers"][0]["uri"], "buffers/mesh data.bin")
            self.assertEqual((staged / "buffers" / "mesh data.bin").read_bytes(), b"mesh")

    def test_gltf_rejects_escape_absolute_url_and_windows_path_forms(self):
        unsafe = (
            "../outside.bin",
            "%2e%2e/outside.bin",
            "%252e%252e/outside.bin",
            "%252e./outside.bin",
            ".%252e/outside.bin",
            "/absolute.bin",
            "C:/absolute.bin",
            "C%3A/absolute.bin",
            "//server/share.bin",
            "\\\\server\\share.bin",
            "https://example.invalid/buffer.bin",
            "file:///outside.bin",
            "safe.bin?query=1",
            "safe.bin#fragment",
            "safe%2fescape.bin",
            "safe%252fescape.bin",
        )
        for index, uri in enumerate(unsafe):
            with self.subTest(uri=uri), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "source"
                source.mkdir()
                mesh = source / f"model-{index}.gltf"
                self._write_gltf(mesh, buffer_uri=uri)
                with self.assertRaisesRegex(runtime.ProcessFailure, "REQUEST_INPUT"):
                    runtime._stage_input_bundle(mesh, root / "staged")

    def test_gltf_rejects_missing_alias_hardlink_and_aggregate_overflow(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            mesh = source / "missing.gltf"
            self._write_gltf(mesh, buffer_uri="missing.bin")
            with self.assertRaisesRegex(runtime.ProcessFailure, "REQUEST_INPUT"):
                runtime._stage_input_bundle(mesh, root / "missing-stage")

            outside = root / "outside.bin"
            outside.write_bytes(b"mesh")
            for kind in ("symlink", "hardlink"):
                with self.subTest(kind=kind):
                    target = source / f"{kind}.bin"
                    if kind == "symlink":
                        try:
                            target.symlink_to(outside)
                        except OSError as exc:
                            self.skipTest(f"symlinks unavailable: {exc}")
                    else:
                        os.link(outside, target)
                    alias_mesh = source / f"{kind}.gltf"
                    self._write_gltf(alias_mesh, buffer_uri=target.name)
                    with self.assertRaisesRegex(runtime.ProcessFailure, "REQUEST_INPUT"):
                        runtime._stage_input_bundle(alias_mesh, root / f"{kind}-stage")

            regular = source / "regular.bin"
            regular.write_bytes(b"mesh")
            aggregate_mesh = source / "aggregate.gltf"
            self._write_gltf(aggregate_mesh, buffer_uri=regular.name)
            too_small = aggregate_mesh.stat().st_size + regular.stat().st_size - 1
            with (
                mock.patch.object(runtime, "MAX_INPUT_BYTES", too_small),
                self.assertRaisesRegex(runtime.ProcessFailure, "REQUEST_INPUT"),
            ):
                runtime._stage_input_bundle(aggregate_mesh, root / "aggregate-stage")

    def test_only_supported_base64_data_buffers_are_classified_as_embedded(self):
        self.assertIsNone(
            runtime._relative_reference_parts(
                "data:application/octet-stream;base64,AA==",
                uri=True,
            )
        )
        for uri in (
            "data:application/octet-stream,AAAA",
            "data:text/plain;base64,%%%",
            "data:;base64,A",
        ):
            with self.subTest(uri=uri):
                with self.assertRaisesRegex(runtime.ProcessFailure, "REQUEST_INPUT"):
                    runtime._relative_reference_parts(uri, uri=True)

    def test_component_swap_to_symlink_is_caught_by_post_copy_revalidation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            nested = source / "nested"
            nested.mkdir(parents=True)
            mesh = source / "model.gltf"
            self._write_gltf(mesh, buffer_uri="nested/buffer.bin")
            (nested / "buffer.bin").write_bytes(b"safe")
            external = root / "external"
            external.mkdir()
            (external / "buffer.bin").write_bytes(b"outside-secret")
            original_copy = runtime._copy_verified_input_file
            swapped = False

            def copy_then_swap(*args, **kwargs):
                nonlocal swapped
                result = original_copy(*args, **kwargs)
                source_path = args[0]
                if source_path.name == "buffer.bin" and not swapped:
                    backup = source / "nested-backup"
                    nested.rename(backup)
                    try:
                        nested.symlink_to(external, target_is_directory=True)
                    except OSError as exc:
                        backup.rename(nested)
                        self.skipTest(f"directory symlinks unavailable: {exc}")
                    swapped = True
                return result

            with mock.patch.object(
                runtime,
                "_copy_verified_input_file",
                side_effect=copy_then_swap,
            ):
                with self.assertRaisesRegex(runtime.ProcessFailure, "REQUEST_INPUT"):
                    runtime._stage_input_bundle(mesh, root / "staged")

    def test_glb_external_buffer_preserves_literal_percent_uri_for_pinned_resolver(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            mesh = source / "model.glb"
            self._write_glb(
                mesh,
                {
                    "asset": {"version": "2.0"},
                    "buffers": [{"uri": "mesh%20data.bin", "byteLength": 4}],
                },
            )
            (source / "mesh data.bin").write_bytes(b"mesh")
            staged = root / "staged"
            runtime._stage_input_bundle(mesh, staged)
            self.assertEqual((staged / "mesh%20data.bin").read_bytes(), b"mesh")
            self.assertEqual((staged / "model.glb").read_bytes(), mesh.read_bytes())

    def test_glb_external_buffer_rejects_missing_and_escape(self):
        for index, uri in enumerate(("missing.bin", "../outside.bin", "https://x.invalid/a")):
            with self.subTest(uri=uri), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "source"
                source.mkdir()
                mesh = source / f"model-{index}.glb"
                self._write_glb(
                    mesh,
                    {
                        "asset": {"version": "2.0"},
                        "buffers": [{"uri": uri, "byteLength": 4}],
                    },
                )
                with self.assertRaisesRegex(runtime.ProcessFailure, "REQUEST_INPUT"):
                    runtime._stage_input_bundle(mesh, root / "staged")

    def test_glb_external_buffer_rejects_alias_and_aggregate_overflow(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            outside = root / "outside.bin"
            outside.write_bytes(b"mesh")
            alias = source / "alias.bin"
            try:
                alias.symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            alias_glb = source / "alias.glb"
            self._write_glb(
                alias_glb,
                {
                    "asset": {"version": "2.0"},
                    "buffers": [{"uri": alias.name, "byteLength": 4}],
                },
            )
            with self.assertRaisesRegex(runtime.ProcessFailure, "REQUEST_INPUT"):
                runtime._stage_input_bundle(alias_glb, root / "alias-stage")

            regular = source / "regular.bin"
            regular.write_bytes(b"mesh")
            aggregate_glb = source / "aggregate.glb"
            self._write_glb(
                aggregate_glb,
                {
                    "asset": {"version": "2.0"},
                    "buffers": [{"uri": regular.name, "byteLength": 4}],
                },
            )
            too_small = aggregate_glb.stat().st_size + regular.stat().st_size - 1
            with (
                mock.patch.object(runtime, "MAX_INPUT_BYTES", too_small),
                self.assertRaisesRegex(runtime.ProcessFailure, "REQUEST_INPUT"),
            ):
                runtime._stage_input_bundle(aggregate_glb, root / "aggregate-stage")

    def test_material_and_image_resolvers_are_disabled_for_geometry_normalization(self):
        class FakeTrimesh:
            def __init__(self, vertices=None, faces=None, process=False):
                self.vertices = np.asarray(
                    vertices
                    if vertices is not None
                    else [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                )
                self.faces = np.asarray(faces if faces is not None else [[0, 1, 2]])

            def export(self, file_type):
                if file_type != "glb":
                    raise AssertionError(file_type)
                return b"glTF" + b"\0" * 32

        opened_sidecars: list[str] = []

        def fake_load(_path, **kwargs):
            if not kwargs.get("skip_materials"):
                opened_sidecars.append("material-or-image")
            return FakeTrimesh()

        fake_module = ModuleType("trimesh")
        fake_module.Scene = type("FakeScene", (), {})
        fake_module.Trimesh = FakeTrimesh
        fake_module.load = fake_load
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(
            sys.modules, {"trimesh": fake_module}
        ):
            root = Path(temp)
            source = root / "source"
            temporary = root / "temporary"
            source.mkdir()
            temporary.mkdir()
            fixtures = {
                "mesh.obj": b"# mtllib /outside.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                "mesh.ply": (
                    b"ply\nformat ascii 1.0\ncomment TextureFile /outside.png\n"
                    b"element vertex 0\nend_header\n"
                ),
            }
            for name, payload in fixtures.items():
                path = source / name
                path.write_bytes(payload)
                run_root, _normalized = runtime._prepare_input(path, temporary)
                shutil.rmtree(run_root)
            gltf = source / "mesh.gltf"
            self._write_gltf(
                gltf,
                buffer_uri="data:application/octet-stream;base64,",
                image_uri="/outside.png",
            )
            run_root, _normalized = runtime._prepare_input(gltf, temporary)
            shutil.rmtree(run_root)
            glb = source / "image.glb"
            self._write_glb(
                glb,
                {
                    "asset": {"version": "2.0"},
                    "buffers": [{"uri": "data:application/octet-stream;base64,", "byteLength": 0}],
                    "images": [{"uri": "/outside.png"}],
                },
            )
            run_root, _normalized = runtime._prepare_input(glb, temporary)
            shutil.rmtree(run_root)
        self.assertEqual(opened_sidecars, [])


class RequestAndProtocolTests(unittest.TestCase):
    def test_runtime_setup_lease_rejects_hardlinked_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outside = root / "outside.lock"
            outside.write_bytes(b"\0")
            try:
                os.link(outside, root / ".setup.lock")
            except OSError as exc:
                self.skipTest(f"hardlinks unavailable: {exc}")
            with self.assertRaisesRegex(runtime.ProcessFailure, "SETUP_INVALID"):
                with runtime._setup_read_lock(root, timeout=0):
                    self.fail("a hardlinked setup lock must not be leased")

    def test_runtime_setup_lease_reports_busy_writer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".setup.lock").write_bytes(b"\0")
            with mock.patch.object(runtime, "_try_setup_read_lock", return_value=False):
                with self.assertRaisesRegex(runtime.ProcessFailure, "SETUP_BUSY"):
                    with runtime._setup_read_lock(root, timeout=0):
                        self.fail("a writer-held setup lock must not be leased")

    def test_windows_runtime_lease_uses_shared_lockfileex(self):
        calls: list[tuple[str, tuple[object, ...]]] = []

        class Function:
            argtypes: object = None
            restype: object = None

            def __init__(self, name: str):
                self.name = name

            def __call__(self, *args: object) -> int:
                calls.append((self.name, args))
                return 1

        kernel = SimpleNamespace(
            LockFileEx=Function("lock"),
            UnlockFileEx=Function("unlock"),
        )
        fake_msvcrt = SimpleNamespace(get_osfhandle=lambda _fd: 123)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lock"
            path.write_bytes(b"\0")
            with (
                path.open("r+b") as handle,
                mock.patch.object(runtime, "_windows_kernel32", return_value=kernel),
                mock.patch.dict(sys.modules, {"msvcrt": fake_msvcrt}),
            ):
                self.assertTrue(runtime._try_setup_read_lock(handle, "win32"))
                runtime._release_setup_read_lock(handle, "win32")
        self.assertEqual([name for name, _args in calls], ["lock", "unlock"])
        lock_flags = calls[0][1][1]
        self.assertEqual(lock_flags, 0x00000001)
        self.assertEqual(int(lock_flags) & 0x00000002, 0)

    def _payload(self, root: Path, node="lato2-e2e", params=None):
        for name in ("workspace", "temp"):
            (root / name).mkdir(exist_ok=True)
        mesh = root / "mesh.glb"
        mesh.write_bytes(b"glTF fixture")
        return {
            "input": {"filePath": str(mesh.resolve()), "nodeId": "unrelated-producer"},
            "params": {} if params is None else params,
            "nodeId": node,
            "workspaceDir": str((root / "workspace").resolve()),
            "tempDir": str((root / "temp").resolve()),
        }

    def test_dispatch_uses_payload_node_id_not_input_node_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state = _fake_state(root)
            with mock.patch.object(runtime, "_validate_state", return_value=state):
                request = runtime.validate_request(self._payload(root, node="lato2-vvae"))
            self.assertEqual(request.node_id, "lato2-vvae")

    def test_workspace_and_temp_reject_filesystem_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state = _fake_state(root / "state")
            filesystem_root = Path(root.anchor)
            for key in ("workspaceDir", "tempDir"):
                with self.subTest(key=key):
                    payload = self._payload(root)
                    payload[key] = str(filesystem_root)
                    with mock.patch.object(runtime, "_validate_state", return_value=state):
                        with self.assertRaisesRegex(runtime.ProcessFailure, "REQUEST_PATHS"):
                            runtime.validate_request(payload)

    def test_workspace_and_temp_may_not_overlap_extension_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            extension = root / "extension"
            nested = extension / "nested"
            nested.mkdir(parents=True)
            state = _fake_state(root / "state")
            for key in ("workspaceDir", "tempDir"):
                for unsafe in (extension, nested, root):
                    with self.subTest(key=key, unsafe=unsafe.name):
                        payload = self._payload(root)
                        payload[key] = str(unsafe.resolve())
                        with (
                            mock.patch.object(runtime, "ROOT", extension),
                            mock.patch.object(runtime, "_validate_state", return_value=state),
                        ):
                            with self.assertRaisesRegex(runtime.ProcessFailure, "REQUEST_PATHS"):
                                runtime.validate_request(payload)

    def test_backend_and_precision_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            portable = _fake_state(root, available=("portable",), default="portable")
            with mock.patch.object(runtime, "_validate_state", return_value=portable):
                with self.assertRaisesRegex(runtime.ProcessFailure, "BACKEND_UNAVAILABLE"):
                    runtime.validate_request(self._payload(root, params={"backend": "upstream"}))
            upstream = _fake_state(root / "exact", available=("upstream",), default="upstream")
            with mock.patch.object(runtime, "_validate_state", return_value=upstream):
                with self.assertRaisesRegex(runtime.ProcessFailure, "PRECISION_UNAVAILABLE"):
                    runtime.validate_request(self._payload(root, params={"precision": "float16"}))

    def test_negative_seed_and_unsupported_portable_bf16_fail_before_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            portable = _fake_state(root, available=("portable",), default="portable")
            portable = runtime.RuntimeState(
                **{
                    **portable.__dict__,
                    "portable_precisions": frozenset({"auto", "float16"}),
                }
            )
            with mock.patch.object(runtime, "_validate_state", return_value=portable):
                with self.assertRaisesRegex(runtime.ProcessFailure, "REQUEST_PARAMS"):
                    runtime.validate_request(self._payload(root, params={"seed": -1}))
                with self.assertRaisesRegex(runtime.ProcessFailure, "PRECISION_UNAVAILABLE"):
                    runtime.validate_request(
                        self._payload(root, params={"precision": "bfloat16"})
                    )

    def test_protocol_has_one_terminal_record(self):
        output = io.StringIO()

        def handler(payload, emitter):
            self.assertEqual(payload["nodeId"], "lato2-tflow")
            emitter.progress(20, "fixture")
            return Path("/tmp/fixture.glb")

        code = runtime.run_protocol(
            io.StringIO(json.dumps({"nodeId": "lato2-tflow"}) + "\n"),
            output,
            handler,
        )
        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(code, 0)
        self.assertEqual([item["type"] for item in messages], ["progress", "done"])

    def test_authentication_failure_never_claims_assets_are_authenticated(self):
        with tempfile.TemporaryDirectory() as temp:
            request = _request(Path(temp), "lato2-tflow")
            output = io.StringIO()
            emitter = runtime.ProtocolEmitter(output)
            with (
                mock.patch.object(runtime, "validate_request", return_value=request),
                mock.patch.object(
                    runtime,
                    "_authenticate_request",
                    side_effect=runtime.ProcessFailure("SETUP_INVALID"),
                ),
            ):
                with self.assertRaisesRegex(runtime.ProcessFailure, "SETUP_INVALID"):
                    runtime.handle_request({}, emitter)
            labels = [
                message["label"]
                for message in map(json.loads, output.getvalue().splitlines())
                if message["type"] == "progress"
            ]
            self.assertTrue(any("Authenticating" in label for label in labels))
            self.assertFalse(any("Authenticated" in label for label in labels))

    def test_exact_backend_neutralizes_portable_renderer_override(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            exact = _request(root / "exact", "lato2-e2e", backend="upstream")
            portable = _request(root / "portable", "lato2-e2e", backend="portable")
            with mock.patch.dict(os.environ, {"LATO2_RENDERER": "software"}):
                self.assertNotIn("LATO2_RENDERER", runtime._runtime_environment(exact))
                self.assertEqual(
                    runtime._runtime_environment(portable)["LATO2_RENDERER"],
                    "software",
                )

    def test_inference_environment_is_minimal_and_uses_owned_home_and_temp(self):
        with tempfile.TemporaryDirectory() as temp:
            request = _request(Path(temp), "lato2-tflow", backend="upstream")
            inherited = {
                "PATH": "/safe/bin",
                "LD_LIBRARY_PATH": "/safe/lib",
                "CUDA_VISIBLE_DEVICES": "0",
                "NCCL_DEBUG": "WARN",
                "HOME": "/untrusted/home",
                "USERPROFILE": "C:\\untrusted",
                "PYTHONPATH": "/inject/python",
                "PYTHONHOME": "/inject/home",
                "PIP_CONFIG_FILE": "/secret/pip.ini",
                "PIP_INDEX_URL": "https://user:password@example.invalid/simple",
                "HTTP_PROXY": "http://user:password@proxy.invalid:8080",
                "HTTPS_PROXY": "http://user:password@proxy.invalid:8080",
                "ACCESS_TOKEN": "secret-token",
                "DATABASE_PASSWD": "secret-password",
                "SESSION_COOKIE": "secret-cookie",
                "CUSTOM_INHERITED": "not-needed",
            }
            with mock.patch.dict(os.environ, inherited, clear=True):
                env = runtime._runtime_environment(request)
            self.assertEqual(env["PATH"], inherited["PATH"])
            self.assertEqual(env["LD_LIBRARY_PATH"], inherited["LD_LIBRARY_PATH"])
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(env["NCCL_DEBUG"], "WARN")
            for key in (
                "PYTHONPATH",
                "PYTHONHOME",
                "PIP_CONFIG_FILE",
                "PIP_INDEX_URL",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ACCESS_TOKEN",
                "DATABASE_PASSWD",
                "SESSION_COOKIE",
                "CUSTOM_INHERITED",
            ):
                self.assertNotIn(key, env)
            controlled = request.state.revision_root / "runtime-cache"
            self.assertEqual(Path(env["HOME"]), controlled / "home")
            self.assertEqual(Path(env["USERPROFILE"]), controlled / "home")
            self.assertEqual(Path(env["TMPDIR"]), controlled / "temporary")

    def test_run_metadata_records_renderer_and_visible_topology_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root / "state", "lato2-e2e", backend="portable")
            request.workspace_dir.mkdir(parents=True)
            request.temp_dir.mkdir(parents=True)
            prepared = request.temp_dir / "prepared"
            normalized = prepared / "input" / "input.glb"
            normalized.parent.mkdir(parents=True)
            normalized.write_bytes(b"glTF fixture")

            def fake_upstream(*_args, log_path, **_kwargs):
                log_path.parent.joinpath("input_pred.ply").write_bytes(b"ply fixture")
                return "software"

            def fake_glb(_source, destination, *, points):
                self.assertTrue(points)
                destination.write_bytes(b"glTF" + b"\0" * 16)

            protocol = io.StringIO()
            with (
                mock.patch.object(runtime, "validate_request", return_value=request),
                mock.patch.object(runtime, "_authenticate_request"),
                mock.patch.object(
                    runtime,
                    "_prepare_input",
                    return_value=(prepared, normalized),
                ),
                mock.patch.object(runtime, "_runtime_environment", return_value={}),
                mock.patch.object(runtime, "_run_upstream", side_effect=fake_upstream),
                mock.patch.object(runtime, "_convert_to_glb", side_effect=fake_glb),
            ):
                output = runtime.handle_request({}, runtime.ProtocolEmitter(protocol))
            metadata = json.loads((output.parent / "run.json").read_text("utf-8"))
            self.assertEqual(metadata["renderer"], "software")
            self.assertEqual(metadata["resultKind"], "points")
            self.assertFalse(metadata["topologyDecoded"])
            messages = [json.loads(line) for line in protocol.getvalue().splitlines()]
            self.assertTrue(
                any(
                    message.get("type") == "log"
                    and "no topology faces" in message.get("message", "")
                    for message in messages
                )
            )

    def test_inference_failure_publishes_only_sanitized_bounded_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root / "state", "lato2-tflow", backend="upstream")
            request.workspace_dir.mkdir(parents=True)
            request.temp_dir.mkdir(parents=True)
            prepared = request.temp_dir / "prepared"
            normalized = prepared / "input" / "input.glb"
            normalized.parent.mkdir(parents=True)
            normalized.write_bytes(b"glTF fixture")

            def fail_upstream(*_args, log_path, **_kwargs):
                log_path.write_text(
                    f"snapshot={request.state.revision_root} workspace={request.workspace_dir} "
                    "token=top-secret-value "
                    "proxy=http://alice:proxy-password@proxy.invalid:8080\n",
                    encoding="utf-8",
                )
                log_path.parent.joinpath("partial.obj").write_text("untrusted", encoding="utf-8")
                raise runtime.ProcessFailure("INFERENCE_FAILED")

            with (
                mock.patch.object(runtime, "validate_request", return_value=request),
                mock.patch.object(runtime, "_authenticate_request"),
                mock.patch.object(runtime, "_prepare_input", return_value=(prepared, normalized)),
                mock.patch.object(runtime, "_runtime_environment", return_value={}),
                mock.patch.object(runtime, "_run_upstream", side_effect=fail_upstream),
            ):
                with self.assertRaisesRegex(runtime.ProcessFailure, "INFERENCE_FAILED"):
                    runtime.handle_request({}, runtime.ProtocolEmitter(io.StringIO()))

            output_parent = request.workspace_dir / "Workflows" / "LATO2"
            diagnostics = [path for path in output_parent.iterdir() if path.name.startswith("failed-")]
            self.assertEqual(len(diagnostics), 1)
            diagnostic = diagnostics[0]
            self.assertEqual(
                {path.name for path in diagnostic.iterdir()},
                {"upstream.log", "run-failure.json"},
            )
            persisted = (diagnostic / "upstream.log").read_text("utf-8")
            for secret in (
                str(request.state.revision_root),
                str(request.workspace_dir),
                "top-secret-value",
                "alice",
                "proxy-password",
            ):
                self.assertNotIn(secret, persisted)
            metadata = json.loads((diagnostic / "run-failure.json").read_text("utf-8"))
            self.assertEqual(metadata["errorCode"], "INFERENCE_FAILED")
            self.assertNotIn(str(root), json.dumps(metadata))
            self.assertLessEqual(
                (diagnostic / "upstream.log").stat().st_size,
                runtime.MAX_LOG_BYTES + runtime.MAX_LOG_OVERHEAD_BYTES,
            )

    def test_post_inference_output_failure_keeps_real_error_code_in_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root / "state", "lato2-tflow", backend="upstream")
            request.workspace_dir.mkdir(parents=True)
            request.temp_dir.mkdir(parents=True)
            prepared = request.temp_dir / "prepared"
            normalized = prepared / "input" / "input.glb"
            normalized.parent.mkdir(parents=True)
            normalized.write_bytes(b"glTF fixture")

            def finish_without_artifact(*_args, log_path, **_kwargs):
                log_path.write_text("completed inference\n", encoding="utf-8")
                return "no-render"

            with (
                mock.patch.object(runtime, "validate_request", return_value=request),
                mock.patch.object(runtime, "_authenticate_request"),
                mock.patch.object(runtime, "_prepare_input", return_value=(prepared, normalized)),
                mock.patch.object(runtime, "_runtime_environment", return_value={}),
                mock.patch.object(runtime, "_run_upstream", side_effect=finish_without_artifact),
            ):
                with self.assertRaisesRegex(runtime.ProcessFailure, "OUTPUT_MISSING"):
                    runtime.handle_request({}, runtime.ProtocolEmitter(io.StringIO()))
            output_parent = request.workspace_dir / "Workflows" / "LATO2"
            diagnostic = next(path for path in output_parent.iterdir() if path.name.startswith("failed-"))
            metadata = json.loads((diagnostic / "run-failure.json").read_text("utf-8"))
            self.assertEqual(metadata["errorCode"], "OUTPUT_MISSING")
            self.assertEqual(metadata["stage"], runtime.ERRORS["OUTPUT_MISSING"][0])

    def test_diagnostic_publication_failure_never_masks_original_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root / "state", "lato2-tflow", backend="upstream")
            request.workspace_dir.mkdir(parents=True)
            request.temp_dir.mkdir(parents=True)
            prepared = request.temp_dir / "prepared"
            normalized = prepared / "input" / "input.glb"
            normalized.parent.mkdir(parents=True)
            normalized.write_bytes(b"glTF fixture")

            def fail_upstream(*_args, log_path, **_kwargs):
                log_path.write_text("failed\n", encoding="utf-8")
                raise runtime.ProcessFailure("INFERENCE_FAILED")

            with (
                mock.patch.object(runtime, "validate_request", return_value=request),
                mock.patch.object(runtime, "_authenticate_request"),
                mock.patch.object(runtime, "_prepare_input", return_value=(prepared, normalized)),
                mock.patch.object(runtime, "_runtime_environment", return_value={}),
                mock.patch.object(runtime, "_run_upstream", side_effect=fail_upstream),
                mock.patch.object(
                    runtime,
                    "_publish_failure_diagnostic",
                    side_effect=OSError("diagnostic storage unavailable"),
                ),
            ):
                with self.assertRaises(runtime.ProcessFailure) as captured:
                    runtime.handle_request({}, runtime.ProtocolEmitter(io.StringIO()))
            self.assertEqual(captured.exception.code, "INFERENCE_FAILED")

    def test_failure_diagnostic_rejects_collisions_and_log_aliases_without_partials(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root / "state", "lato2-tflow", backend="upstream")
            output_parent = root / "output"
            output_parent.mkdir()
            log = root / "upstream.log"
            log.write_text("bounded failure\n", encoding="utf-8")
            stamp = "20260830T120000"
            token = "0123456789ab"
            collision = output_parent / f"failed-{request.node_id}-{stamp}-{token}"
            collision.mkdir()
            (collision / "sentinel").write_text("keep", encoding="utf-8")
            self.assertFalse(
                runtime._publish_failure_diagnostic(
                    source_log=log,
                    output_parent=output_parent,
                    request=request,
                    stamp=stamp,
                    token=token,
                    replacements={},
                    failure_code="INFERENCE_FAILED",
                )
            )
            self.assertEqual((collision / "sentinel").read_text("utf-8"), "keep")

            outside = root / "outside.log"
            outside.write_text("do not publish", encoding="utf-8")
            alias = root / "alias.log"
            try:
                alias.symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            self.assertFalse(
                runtime._publish_failure_diagnostic(
                    source_log=alias,
                    output_parent=output_parent,
                    request=request,
                    stamp=stamp,
                    token="fedcba987654",
                    replacements={},
                    failure_code="INFERENCE_FAILED",
                )
            )
            self.assertFalse(
                any(path.name.startswith(".") for path in output_parent.iterdir())
            )

    def test_real_bootstrap_rejects_invalid_json_without_stdout_noise(self):
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / "processor.py")],
            input="not-json\n",
            text=True,
            capture_output=True,
            timeout=30,
        )
        lines = completed.stdout.splitlines()
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["type"], "error")

    def test_upstream_log_is_bounded_and_drained(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = io.StringIO()
            emitter = runtime.ProtocolEmitter(output)
            renderer = runtime._run_upstream(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.buffer.write(b'x' * (3 * 1024 * 1024))",
                ],
                cwd=root,
                env=dict(os.environ),
                log_path=root / "upstream.log",
                emitter=emitter,
                replacements={},
                renderer_applicable=False,
            )
            captured = (root / "upstream.log").read_bytes()
            self.assertEqual(renderer, "no-render")
            self.assertLessEqual(
                len(captured),
                runtime.MAX_LOG_BYTES + runtime.MAX_LOG_OVERHEAD_BYTES,
            )
            self.assertIn(b"Modly retained", captured[:100])
            self.assertEqual(captured.count(b"[Modly retained"), 1)
            self.assertTrue(captured.endswith(b"[Modly effective renderer: no-render]\n"))

    def test_successful_upstream_log_is_sanitized_before_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            private_path = str(root / "private" / "snapshot")
            secret_line = (
                f"path={private_path} token=super-secret-token-value "
                "proxy=http://alice:proxy-password@proxy.invalid:8080 "
                "Authorization: Bearer abcdefghijklmnopqrstuvwxyz\n"
            )
            runtime._run_upstream(
                [sys.executable, "-c", f"print({secret_line!r}, end='')"],
                cwd=root,
                env=dict(os.environ),
                log_path=root / "upstream.log",
                emitter=runtime.ProtocolEmitter(io.StringIO()),
                replacements={private_path: "[model snapshot]"},
                renderer_applicable=False,
            )
            persisted = (root / "upstream.log").read_text("utf-8")
            self.assertIn("[model snapshot]", persisted)
            self.assertIn("[redacted]", persisted)
            for secret in (
                private_path,
                "super-secret-token-value",
                "alice",
                "proxy-password",
                "abcdefghijklmnopqrstuvwxyz",
            ):
                self.assertNotIn(secret, persisted)
            self.assertLessEqual(
                len(persisted.encode("utf-8")),
                runtime.MAX_LOG_BYTES + runtime.MAX_LOG_OVERHEAD_BYTES,
            )
            self.assertTrue(
                persisted.endswith("[Modly effective renderer: no-render]\n")
            )

    def test_early_renderer_warning_survives_bounded_log_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = io.StringIO()
            emitter = runtime.ProtocolEmitter(output)
            renderer = runtime._run_upstream(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        "sys.stdout.buffer.write(b'portable software fallback\\n' "
                        "+ b'x' * (3 * 1024 * 1024))"
                    ),
                ],
                cwd=root,
                env=dict(os.environ),
                log_path=root / "upstream.log",
                emitter=emitter,
                replacements={},
                renderer_applicable=True,
            )
            self.assertEqual(renderer, "software")
            captured = (root / "upstream.log").read_bytes()
            self.assertNotIn(b"portable software fallback", captured)
            self.assertTrue(captured.endswith(b"[Modly effective renderer: software]\n"))
            self.assertLessEqual(
                len(captured),
                runtime.MAX_LOG_BYTES + runtime.MAX_LOG_OVERHEAD_BYTES,
            )
            messages = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertTrue(
                any(
                    message["type"] == "log"
                    and "software conditioning renderer" in message["message"]
                    for message in messages
                )
            )

    def test_explicit_software_renderer_is_recorded_without_warning_needle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = io.StringIO()
            renderer = runtime._run_upstream(
                [sys.executable, "-c", "print('rendered')"],
                cwd=root,
                env={**os.environ, "LATO2_RENDERER": " Software "},
                log_path=root / "upstream.log",
                emitter=runtime.ProtocolEmitter(output),
                replacements={},
                renderer_applicable=True,
            )
            self.assertEqual(renderer, "software")
            self.assertTrue(
                (root / "upstream.log").read_bytes().endswith(
                    b"[Modly effective renderer: software]\n"
                )
            )
            messages = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertTrue(any(message["type"] == "log" for message in messages))

    def test_default_conditioning_renderer_is_recorded_as_open3d(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clean_env = {key: value for key, value in os.environ.items() if key != "LATO2_RENDERER"}
            renderer = runtime._run_upstream(
                [sys.executable, "-c", "print('rendered')"],
                cwd=root,
                env=clean_env,
                log_path=root / "upstream.log",
                emitter=runtime.ProtocolEmitter(io.StringIO()),
                replacements={},
                renderer_applicable=True,
            )
            self.assertEqual(renderer, "open3d")
            self.assertTrue(
                (root / "upstream.log").read_bytes().endswith(
                    b"[Modly effective renderer: open3d]\n"
                )
            )

    def test_child_spawn_isolated_without_shell_on_posix_and_windows(self):
        self.assertEqual(
            runtime._process_group_spawn_options("linux"),
            {"start_new_session": True},
        )
        with mock.patch.object(
            subprocess,
            "CREATE_NEW_PROCESS_GROUP",
            0x200,
            create=True,
        ):
            self.assertEqual(
                runtime._process_group_spawn_options("win32"),
                {"creationflags": 0x200},
            )

    def test_sigterm_is_converted_to_managed_child_cleanup(self):
        with (
            mock.patch.object(runtime.signal, "getsignal", return_value=runtime.signal.SIG_DFL),
            mock.patch.object(runtime.signal, "signal") as install,
        ):
            handlers = runtime._install_child_cancel_handlers()
            cancel = next(
                call.args[1]
                for call in install.call_args_list
                if call.args[0] == runtime.signal.SIGTERM
            )
            with self.assertRaises(runtime._ChildCancellation):
                cancel(int(runtime.signal.SIGTERM), None)
            install.reset_mock()
            runtime._restore_child_cancel_handlers(handlers)
            install.assert_any_call(int(runtime.signal.SIGTERM), runtime.signal.SIG_DFL)

    def test_posix_cleanup_signals_the_isolated_process_group(self):
        process = mock.Mock()
        process.pid = 424242
        process.poll.return_value = 0
        process.wait.return_value = 0
        with (
            mock.patch.object(runtime.os, "getpgrp", return_value=31337),
            mock.patch.object(runtime.os, "killpg") as killpg,
        ):
            runtime._terminate_process_group(process, "linux")
        killpg.assert_has_calls(
            [
                mock.call(process.pid, runtime.signal.SIGTERM),
                mock.call(process.pid, 0),
                mock.call(process.pid, runtime.signal.SIGKILL),
            ]
        )

    def test_windows_cleanup_uses_ctrl_break_and_taskkill_tree(self):
        process = mock.Mock()
        process.pid = 424242
        process.poll.side_effect = (None, 0)
        process.wait.return_value = 0
        with (
            mock.patch.object(runtime.signal, "CTRL_BREAK_EVENT", 123, create=True),
            mock.patch.object(runtime.subprocess, "run") as taskkill,
        ):
            runtime._terminate_process_group(process, "win32")
        process.send_signal.assert_called_once_with(123)
        taskkill.assert_called_once()
        args, kwargs = taskkill.call_args
        self.assertEqual(args[0], ["taskkill.exe", "/PID", "424242", "/T", "/F"])
        self.assertFalse(kwargs["shell"])

    def test_nonzero_upstream_exit_still_runs_group_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            emitter = runtime.ProtocolEmitter(io.StringIO())
            with mock.patch.object(runtime, "_terminate_process_group") as terminate:
                with self.assertRaisesRegex(runtime.ProcessFailure, "INFERENCE_FAILED"):
                    runtime._run_upstream(
                        [sys.executable, "-c", "raise SystemExit(7)"],
                        cwd=root,
                        env=dict(os.environ),
                        log_path=root / "failed.log",
                        emitter=emitter,
                        replacements={},
                        renderer_applicable=False,
                    )
            terminate.assert_called_once()


class OutputTests(unittest.TestCase):
    def test_output_selection_prefers_faces_then_point_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "input_pred.ply").write_bytes(b"ply fixture")
            selected, points = runtime._upstream_result("lato2-e2e", root)
            self.assertEqual(selected.name, "input_pred.ply")
            self.assertTrue(points)
            (root / "input_pred.obj").write_bytes(b"obj fixture")
            selected, points = runtime._upstream_result("lato2-e2e", root)
            self.assertEqual(selected.name, "input_pred.obj")
            self.assertFalse(points)

    def test_invalid_topology_output_never_falls_back_silently(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "input_pred.ply").write_bytes(b"ply fixture")
            (root / "input_pred.obj").mkdir()
            with self.assertRaisesRegex(runtime.ProcessFailure, "OUTPUT_INVALID"):
                runtime._upstream_result("lato2-e2e", root)

    def test_mesh_and_point_outputs_become_glb(self):
        class Geometry:
            vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            faces = np.asarray([[0, 1, 2]])
            colors = np.empty((0, 4), dtype=np.uint8)

        class Scene:
            def __init__(self, value):
                self.value = value

            def export(self, file_type):
                assert file_type == "glb"
                return b"glTF" + b"\0" * 32

        calls = []
        fake = ModuleType("trimesh")
        fake.load = lambda *_args, **_kwargs: Geometry()
        fake.Scene = Scene
        fake.Trimesh = lambda **kwargs: calls.append(("mesh", kwargs)) or Geometry()
        fake.points = SimpleNamespace(
            PointCloud=lambda **kwargs: calls.append(("points", kwargs)) or Geometry()
        )
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(sys.modules, {"trimesh": fake}):
            root = Path(temp)
            source = root / "source.ply"
            source.write_bytes(b"fixture")
            runtime._convert_to_glb(source, root / "mesh.glb", points=False)
            runtime._convert_to_glb(source, root / "points.glb", points=True)
            self.assertEqual([kind for kind, _ in calls], ["mesh", "points"])
            self.assertNotIn("colors", calls[1][1])
            self.assertEqual((root / "mesh.glb").read_bytes()[:4], b"glTF")
            self.assertEqual((root / "points.glb").read_bytes()[:4], b"glTF")

    def test_real_colorless_upstream_ply_converts_to_point_glb(self):
        if importlib.util.find_spec("trimesh") is None:
            self.skipTest("trimesh unavailable in the host test interpreter")
        ply = """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
end_header
0 0 0
1 0 0
0 1 0
"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "generated vertices.ply"
            destination = root / "result.glb"
            source.write_text(ply, encoding="ascii")
            runtime._convert_to_glb(source, destination, points=True)
            self.assertEqual(destination.read_bytes()[:4], b"glTF")


if __name__ == "__main__":
    unittest.main()
