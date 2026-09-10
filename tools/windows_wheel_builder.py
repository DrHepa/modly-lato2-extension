"""Activate CI's compiler explicitly; consumers never execute this tool."""
import os
from pathlib import Path
import subprocess
from build_portable_wheel import main

if __name__ == '__main__':
    vswhere = Path(os.environ['ProgramFiles(x86)']) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    result = subprocess.run([str(vswhere), '-latest', '-version', '[17.0,18.0)', '-products', '*', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], check=True, capture_output=True, text=True)
    vcvars = Path(result.stdout.strip()) / 'VC/Auxiliary/Build/vcvars64.bat'
    if not vcvars.is_file():
        raise RuntimeError('The CI image lacks the reviewed MSVC v143 toolchain')
    # cmd.exe uses different quoting from the MS C-runtime argument parser.
    # Passing a list would escape embedded quotes as backslash-quote, which
    # cmd interprets literally. Supply its trusted command line unchanged.
    cmd = os.environ.get('COMSPEC', r'C:\Windows\System32\cmd.exe')
    result = subprocess.run(f'"{cmd}" /d /s /c "call "{vcvars}" >nul && set"', capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError('MSVC activation failed: ' + result.stderr)
    allowed = {'PATH','INCLUDE','LIB','LIBPATH','WINDOWSSDKDIR','VCINSTALLDIR','VSINSTALLDIR','VCTOOLSINSTALLDIR'}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition('=')
        if sep and key.upper() in allowed:
            os.environ[key.upper()] = value
    os.environ['DISTUTILS_USE_SDK'] = '1'
    main()
