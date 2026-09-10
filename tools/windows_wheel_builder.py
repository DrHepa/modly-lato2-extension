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
    result = subprocess.run(['cmd.exe', '/d', '/s', '/c', f'call "{vcvars}" >nul && set'], check=True, capture_output=True, text=True)
    # os.environ is case-insensitive on Windows; ordinary dicts are not.
    # Export only build-tool settings, never credentials or arbitrary variables.
    allowed = {'PATH','INCLUDE','LIB','LIBPATH','WINDOWSSDKDIR','VCINSTALLDIR','VSINSTALLDIR','VCTOOLSINSTALLDIR'}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition('=')
        if sep and key.upper() in allowed:
            os.environ[key.upper()] = value
    main()
