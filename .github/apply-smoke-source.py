from pathlib import Path
import hashlib
p=Path('lato2_modly/dependencies.py')
b=p.read_bytes()
assert hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()=='d94463485914a58f9a0fb22a012b292de32792cc'
s=b.decode()
s=s.replace('from . import binary_wheels as binaries\n', 'from . import binary_wheels as binaries\nfrom .dependency_probe import CONDITIONAL_DISTRIBUTION_MARKERS, IMPORT_PROBE, failure_detail\n',1)
s=s.replace('        "constraints": list(constraint_requirements(plan)),', '        "constraints": list(constraint_requirements(plan)),\n        "conditionalDistributionMarkers": dict(CONDITIONAL_DISTRIBUTION_MARKERS),',1)
start=s.index('def verify_dependencies(\n')
original=s[start:]
header=original[:original.index("    script = r'''\n")]
header=header.replace('def verify_dependencies(', 'def _dependency_probe_environment(',1).replace(') -> dict[str, object]:', ') -> dict[str, str]:',1).replace('"""Run import, version, native-symbol and CUDA smoke checks in the venv."""','"""Prepare the same target, marker policy and environment for every probe."""',1)
flags_start=original.index('    smoke_env["MODLY_LATO2_EXPECTED_DISTS"]')
flags_end=original.index('    try:\n        completed = subprocess.run(',flags_start)
flags=original[flags_start:flags_end]+'    smoke_env["MODLY_LATO2_CONDITIONAL_DISTS"] = json.dumps(CONDITIONAL_DISTRIBUTION_MARKERS)\n    return smoke_env\n\n\n'
executor='''def _run_dependency_probe(python: Path, script: str, smoke_env: Mapping[str, str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            [str(python), "-I", "-X", "utf8", "-c", script],
            check=True, capture_output=True, text=True,
            encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
            timeout=20 * 60, env=dict(smoke_env),
        )
    except (subprocess.SubprocessError, OSError) as exc:
        detail = failure_detail(exc, paths={str(python.parent.parent): "<extension-venv>",
                                           smoke_env.get("MODLY_LATO2_CACHE_DIR", ""): "<runtime-cache>"})
        raise DependencyError("DEPENDENCY_SMOKE_FAILED", "dependency smoke failed; " + detail) from exc
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        evidence = subprocess.CalledProcessError(0, "dependency-probe", output=completed.stdout, stderr=completed.stderr)
        raise DependencyError("DEPENDENCY_SMOKE_FAILED", "dependency smoke returned invalid JSON; " + failure_detail(evidence)) from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise DependencyError("DEPENDENCY_SMOKE_INVALID", "dependency smoke result is invalid")
    return payload


def verify_dependency_imports(
    python: Path, plan: DependencyPlan, cache_root: Path, *,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Diagnostic/CI phase only. Does NOT replace production CUDA validation."""
    python = _validated_venv_python_path(Path(python))
    smoke_env = _dependency_probe_environment(python, plan, cache_root, env=env)
    # Renderer context creation belongs to GPU/native acceptance, not imports.
    smoke_env["MODLY_LATO2_OPEN3D_RENDER_SMOKE"] = "0"
    script = IMPORT_PROBE + "\\nprint(json.dumps({'ok': True, 'scope': 'metadata-and-imports-only', 'cuda_tested': False, 'conditional_absent': conditional_absent}))\\n"
    return _run_dependency_probe(python, script, smoke_env)


'''
func=original[:original.index('    smoke_env = sanitize_subprocess_environment(')]
func+='    python = _validated_venv_python_path(Path(python))\n    smoke_env = _dependency_probe_environment(python, plan, cache_root, env=env)\n    script = IMPORT_PROBE + r\'\'\'\n_stage("cuda:availability")\n'
cuda=original[original.index('if not torch.cuda.is_available():'):flags_start]
cuda=cuda.replace('device = torch.device("cuda")', '_stage("cuda:matmul")\ndevice = torch.device("cuda")',1)
cuda=cuda.replace('if native:\n', 'if native:\n    _stage("native:imports")\n',1)
cuda=cuda.replace('    indices = torch.tensor(', '    _stage("native:spconv")\n    indices = torch.tensor(',1)
cuda=cuda.replace('    scatter_source = torch.tensor(', '    _stage("native:torch-scatter")\n    scatter_source = torch.tensor(',1)
cuda=cuda.replace('    ovo_vertices = torch.tensor(', '    _stage("native:o-voxel")\n    ovo_vertices = torch.tensor(',1)
cuda=cuda.replace('    attention_dtype = torch.bfloat16', '    _stage("native:xformers")\n    attention_dtype = torch.bfloat16',1)
cuda=cuda.replace('        import flash_attn', '        _stage("native:flash-attn")\n        import flash_attn',1)
remaining=original[original.index('    capability = payload.get("capability")'):]
s=s[:start]+header+flags+executor+func+cuda+'    payload = _run_dependency_probe(python, script, smoke_env)\n'+remaining
assert 'verify_dependency_imports' in s
compile(s,str(p),'exec')
p.write_text(s,encoding='utf-8')
for path,sha,old,new in [
    ('lato2_modly/constants.py','56c20e8bb7a1c6bfc86fc79dca00d15f804c3606','EXTENSION_VERSION = "1.2.0"','EXTENSION_VERSION = "1.2.1"'),
    ('manifest.json','a2f04eba0c29336cf28ec8058507958deee669a6','"version": "1.2.0"','"version": "1.2.1"'),
]:
    p=Path(path); b=p.read_bytes()
    assert hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()==sha
    text=b.decode(); assert text.count(old)==1
    p.write_text(text.replace(old,new),encoding='utf-8')
