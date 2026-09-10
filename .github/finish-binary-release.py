"""One-time authenticated release assembly for the reviewed binary transition."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path.cwd()))
from lato2_modly import binary_wheels as binary

REPO = 'DrHepa/modly-lato2-extension'
BRANCH = 'fix/prebuilt-portable-runtime'
TAG = 'ovoxel-cpu-post2-20260910.1'
assert os.environ['GITHUB_REPOSITORY'] == REPO
assert os.environ['GITHUB_REF'] == 'refs/heads/' + BRANCH

def run(*args, **kwargs):
    return subprocess.run(list(args), check=True, **kwargs)

def api(path):
    return json.loads(subprocess.check_output(['gh', 'api', f'repos/{REPO}/{path}'], text=True))

# The existing smoke test mocked every subprocess as the same operator result.
# Account for the new, preceding installed-file verification without bypassing it.
p = Path('tests/test_dependencies.py')
raw = p.read_bytes()
assert hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() == '9004152bda48b156d3a1ad4a41a426edfc80564d'
s = raw.decode()
edits = [(
'''            return completed

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python = fake_venv_python(root, symlink=True)
            base_python = python.resolve(strict=True)
            with mock.patch.object(
                deps, "cpu_build_environment", return_value={"PATH": ""}
            ), mock.patch.object(deps.subprocess, "run", side_effect=fake_run):''',
'''            if "MODLY_LATO2_BINARY_SPEC" in kwargs["env"]:
                spec = json.loads(kwargs["env"]["MODLY_LATO2_BINARY_SPEC"])
                result = mock.Mock()
                result.stdout = json.dumps({"wheelSha256": spec["sha256"], "filesVerified": len(spec["files"])})
                return result
            return completed

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python = fake_venv_python(root, symlink=True)
            base_python = python.resolve(strict=True)
            with mock.patch.object(
                deps, "cpu_build_environment", side_effect=AssertionError("compiler requested")
            ), mock.patch.object(deps.subprocess, "run", side_effect=fake_run):'''),(
'''        self.assertEqual([command[0] for command in calls], [str(python), str(python)])
        self.assertNotIn(str(base_python), [command[0] for command in calls])
        smoke_script = calls[1][-1]''',
'''        self.assertEqual([command[0] for command in calls], [str(python)] * 3)
        self.assertNotIn(str(base_python), [command[0] for command in calls])
        self.assertIn("MODLY_LATO2_BINARY_SPEC", environments[0])
        self.assertIn("hashlib.file_digest", calls[0][-1])
        self.assertEqual(calls[1][-1], "check")
        smoke_script = calls[2][-1]'''),(
'''            environments[1]["MODLY_LATO2_OVOXEL_CPU_IDENTITY"]''',
'''            environments[2]["MODLY_LATO2_OVOXEL_CPU_IDENTITY"]''')]
for old, new in edits:
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
p.write_text(s, encoding='utf-8')

for build, sha, pattern, count in [
    (34464232916, 'd7592043c3db5e50fef030afb2334e69925f4239', 'wheel-linux-*', 8),
    (34465721218, '08af54551be69225096cde2bd8e4e25c3f0202d8', 'wheel-windows-*', 4),
]:
    info = api(f'actions/runs/{build}')
    assert info['head_sha'] == sha and info['head_repository']['full_name'] == REPO
    jobs = api(f'actions/runs/{build}/jobs?per_page=100')['jobs']
    selected = [j for j in jobs if j['name'].startswith('linux ' if count == 8 else 'windows ')]
    assert len(selected) == count and all(j['conclusion'] == 'success' for j in selected)
    run('gh', 'run', 'download', str(build), '--pattern', pattern, '--dir', 'collected')

records = []
out = Path('release-pack'); out.mkdir()
for path in sorted(Path('collected').rglob('*.whl.json')):
    value = json.loads(path.read_text())
    spec = binary.parse_spec(value)
    expected = 'd7592043c3db5e50fef030afb2334e69925f4239' if spec.key.startswith('linux/') else '08af54551be69225096cde2bd8e4e25c3f0202d8'
    assert spec.source_commit == expected and spec.release == TAG
    wheel = path.with_name(spec.filename)
    binary.verify_wheel(wheel, spec)
    smoke = json.loads((path.parent / 'smoke.json').read_text())
    assert smoke['key'] == spec.key and smoke['consumer']['ok'] is True
    assert smoke['consumer']['torch'] == spec.key.split('/')[-1]
    assert smoke['consumer']['voxelCount'] > 0
    shutil.copyfile(wheel, out / spec.filename)
    records.append(value)
expected = {f'{system}/{arch}/cp{py}/{torch}' for system, arch, torches in [
    ('win32', 'x64', ('2.6.0+cu124', '2.9.1+cu128')),
    ('linux', 'x64', ('2.6.0+cu124', '2.9.1+cu128')),
    ('linux', 'arm64', ('2.6.0+cu126', '2.9.1+cu128')),
] for py in ('311', '312') for torch in torches}
assert len(records) == 12 and {r['key'] for r in records} == expected
old = {s.key: s for s in binary.load_inventory()}
for record in records:
    if record['key'] in old:
        assert binary.parse_spec(record) == old[record['key']]
text = json.dumps({'schema': binary.SCHEMA, 'artifacts': sorted(records, key=lambda r: r['key'])}, indent=2) + '\n'
binary.INDEX.write_text(text, encoding='utf-8')
(out / 'binary-wheels.json').write_text(text, encoding='utf-8')
(out / 'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['filename']}\n" for r in sorted(records, key=lambda r: r['filename'])), encoding='utf-8')
assert len(binary.load_inventory()) == 12
run(sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v')
run('git', 'diff', '--check')
remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/' + BRANCH], text=True).split()[0]
assert remote == os.environ['GITHUB_SHA']
run('git', 'config', 'user.name', 'github-actions[bot]')
run('git', 'config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com')
run('git', 'add', '--', 'lato2_modly/binary-wheels.json', 'tests/test_dependencies.py')
run('git', 'commit', '-m', 'release: pin twelve verified wheels and exercise binary verification in smoke fixture')
run('git', 'push', 'origin', 'HEAD:refs/heads/' + BRANCH)
assert not any(r['tag_name'] == TAG for r in api('releases?per_page=100')), 'Refusing to overwrite existing release'
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
notes = ('Auxiliary CPU operator binary pack for Windows x64, Linux x64 and Linux ARM64, CPython 3.11/3.12 and the exact PyTorch builds in binary-wheels.json. '
         'All twelve wheels passed real CPU operator tests in separate venvs with compiler PATH removed. '
         'Source commits, pinned TRELLIS.2/Eigen provenance, build recipe and licenses are available in the repository; archive/member SHA-256 values are in the inventory. '
         'These are not model weights or a new Modly app release. The compiler-free production installer is on fix/prebuilt-portable-runtime pending review. '
         'No clean-OS or full GPU inference certification is claimed. Linux wheels retain linux_* tags and were built on glibc 2.28, without auditwheel certification. '
         'Install through the extension rather than manually guessing Python/Torch compatibility.')
run('gh', 'release', 'create', TAG, '--target', commit, '--draft', '--prerelease', '--title', 'Portable o-voxel CPU binary pack — CPython 3.11/3.12', '--notes', notes)
run('gh', 'release', 'upload', TAG, *sorted(str(p) for p in out.iterdir()))
run('gh', 'release', 'download', TAG, '--dir', 'redownloaded')
for spec in binary.load_inventory():
    binary.verify_wheel(Path('redownloaded') / spec.filename, spec)
assert (Path('redownloaded') / 'binary-wheels.json').read_bytes() == binary.INDEX.read_bytes()
run('gh', 'release', 'edit', TAG, '--draft=false', '--latest=false')
run('git', 'archive', '--format=zip', '--output=release-pack/extension-source.zip', 'HEAD')
print('Published and redownload-verified all 12 wheels at', TAG)
