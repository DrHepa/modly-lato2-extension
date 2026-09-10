"""Maintainer-only CPU operator wheel build; no model assets or GPU required."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lato2_modly import dependencies as deps
from lato2_modly import ovoxel_cpu as op

SMOKE = r'''
import json, torch
from lato2_ovoxel_cpu import _C
v = torch.tensor([[.5,.5,.5],[1.5,.5,.5],[.5,1.5,.5],[.5,.5,1.5]], dtype=torch.float32)
f = torch.tensor([[0,2,1],[0,1,3],[0,3,2],[1,2,3]], dtype=torch.int32)
s = torch.tensor([.25,.25,.25], dtype=torch.float32)
r = torch.tensor([[0,0,0],[8,8,8]], dtype=torch.int32)
a = _C.mesh_to_flexible_dual_grid_cpu(v, f, s, r, 1., 1., .1, False)
b = _C.mesh_to_flexible_dual_grid_cpu(v, f, s, r, 1., 1., .1, False)
assert len(a) == 3 and a[0].shape[0] > 0
for i, dtype in enumerate((torch.int32,torch.float32,torch.bool)):
    assert a[i].device.type == 'cpu' and a[i].dtype == dtype
    assert a[i].shape == (a[0].shape[0],3)
# Hash-table traversal need not be stable; sort by coordinates for comparison.
def ordered(outputs):
    c=outputs[0]; keys=c[:,0]*64+c[:,1]*8+c[:,2]; order=keys.argsort()
    return [t[order] for t in outputs]
for x, y in zip(ordered(a), ordered(b)):
    assert torch.equal(x, y)
assert torch.isfinite(a[1]).all()
print(json.dumps({'ok':True,'voxelCount':len(a[0]),'torch':torch.__version__}))
'''


def run(command, **kwargs):
    print('+', ' '.join(map(str,command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, timeout=1800, **kwargs)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--torch', required=True)
    parser.add_argument('--release', required=True)
    parser.add_argument('--output', default='wheel-output')
    args=parser.parse_args()
    import torch
    if torch.__version__ != args.torch:
        raise RuntimeError(f'Wrong Torch build: {torch.__version__}')
    output=Path(args.output).resolve(); output.mkdir(exist_ok=True, parents=True)
    cache=ROOT/'wheel-build-cache'; cache.mkdir(exist_ok=True)
    system='win32' if os.name=='nt' else 'linux'
    arch='arm64' if platform.machine().lower() in {'arm64','aarch64'} else 'x64'
    sm=120 if args.torch.endswith('cu128') else 90 if arch=='arm64' else 75
    plan=deps.select_dependency_plan(dict(platform=system, arch=arch, gpu_sm=sm, cuda_version=128, accelerator='cuda'), 'portable')
    if plan.torch_requirements[0] != f'torch=={args.torch}':
        raise RuntimeError('Build does not match Modly dependency plan')
    sources=deps.prepare_portable_cpu_sources(cache)
    report=op.materialize_ovoxel_cpu_build(sources.ovoxel, sources.eigen, cache/'operator-source')
    # A separate workspace preserves the authenticated materialized source.
    work=deps.prepare_build_workspace(Path(report.build_root), cache, plan, 'binary-dist')
    cpp=work/'src'/'flexible_dual_grid.cpp'
    source=cpp.read_text(encoding='utf-8')
    # MSVC does not accept GCC's redundant double-literal 'd' suffix.
    # Removing it preserves the numeric type and value on all platforms.
    for old,new in [('1e-6d','1e-6'),('0.0d','0.0')]:
        if source.count(old) != 1:
            raise RuntimeError(f'Unreviewed upstream source: {old}')
        source=source.replace(old,new)
    cpp.write_text(source,encoding='utf-8')
    env=deps.cpu_build_environment(plan,cache)
    env['MAX_JOBS']='1'
    build='1torch'+args.torch.replace('.','').replace('+','')
    run([sys.executable,'setup.py','bdist_wheel','--build-number',build,'--dist-dir',output],cwd=work,env=env)
    wheels=list(output.glob('*.whl'))
    if len(wheels)!=1: raise RuntimeError('Expected exactly one wheel')
    wheel=wheels[0]
    # Consumer test: separate venv, wheel-only pip installation, no compiler PATH.
    consumer=cache/'consumer'
    run([sys.executable,'-m','venv',consumer])
    python=consumer/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    index='https://download.pytorch.org/whl/'+args.torch.split('+')[1]
    run([python,'-m','pip','install','--only-binary=:all:','numpy==2.2.6'])
    run([python,'-m','pip','install','--only-binary=:all:',f'torch=={args.torch}','--index-url',index])
    run([python,'-m','pip','install','--no-deps','--no-index','--only-binary=:all:',wheel])
    runtime_env=deps.sanitize_subprocess_environment()
    for key in list(runtime_env):
        if key.upper().startswith(('VC','VS','CUDA','INCLUDE','LIB','CC','CXX','CL','CMAKE','CPATH','COMPILER')):
            runtime_env.pop(key,None)
    runtime_env['PATH']=os.pathsep.join([str(python.parent), str(Path(os.environ.get('SYSTEMROOT','C:/Windows'))/'System32')]) if os.name=='nt' else str(python.parent)
    result=subprocess.run([str(python),'-I','-c',SMOKE],check=True,capture_output=True,text=True,env=runtime_env,timeout=180)
    smoke=json.loads(result.stdout)
    if smoke.get('ok') is not True: raise RuntimeError('CPU operator test failed')
    with ZipFile(wheel) as archive:
        files={n:hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist() if not n.endswith(('/','/RECORD'))}
    record=dict(key=f'{system}/{arch}/{plan.python_abi.lane}/{args.torch}',filename=wheel.name,release=args.release,
                sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),size=wheel.stat().st_size,
                build_identity=op.OVOXEL_CPU_BUILD_IDENTITY,template_sha256=op.TEMPLATE_TREE_SHA256,
                source_commit=os.environ['GITHUB_SHA'],files=files,
                glibc_min=list(map(int,platform.libc_ver()[1].split('.')[:2])) if system=='linux' else None)
    (output/(wheel.name+'.json')).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    (output/'smoke.json').write_text(json.dumps({'key':record['key'], 'consumer':smoke,'scope':'CPU operator only; compiler PATH removed, not a clean OS or GPU inference test'},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))
    print(json.dumps(smoke))

if __name__=='__main__':
    main()
