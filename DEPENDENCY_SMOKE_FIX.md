# Dependency smoke repair (1.2.1)

## Defect

The reported Windows installation completes model downloads, pip installation and
`pip check`, then fails with `DEPENDENCY_SMOKE_FAILED`. The old verifier treated
every constraints entry as an unconditional installed requirement, and discarded
the subprocess stderr when it raised its generic error.

In particular, huggingface_hub 0.36.0 declares hf-xet conditionally:

```
hf-xet>=1.1.3,<2.0.0; platform_machine=='x86_64' or platform_machine=='amd64' or platform_machine=='arm64' or platform_machine=='aarch64'
```

Source: https://github.com/huggingface/huggingface_hub/blob/v0.36.0/setup.py

Windows CPython commonly reports `AMD64`, which is not the lowercase `amd64`
marker value. The ABI probe's normalized `amd64` must not be substituted for the
raw environment used by pip's marker evaluation. Constraints cap versions; they
do not force installation. This explains a successful pip check followed by a
PackageNotFoundError in our old unconditional version loop.

## Correction

The target interpreter evaluates the audited hf-xet marker using packaging. An
inactive marker permits the package to be absent; if it is installed, the exact
pinned version is still checked. Active markers still require it, and all other
required distributions remain mandatory. No package pins or wheel artifacts are
changed. The marker policy is part of the dependency lock identity.

Production verification still checks actual CUDA availability, CUDA matrix
execution, synchronization and GPU capability. Exact-upstream native checks are
retained. A missing GPU or broken DLL is not converted into a passing smoke.

The probe now writes flushed stage markers before imports and native operations.
Failures retain the last stage, process exit/timeout and bounded, sanitized
stderr evidence. Secrets, user paths and the full inline command are not echoed.
An example failure is now `DEPENDENCY_SMOKE_FAILED; stage=import:torch; exit=1`
followed by the actual exception, rather than only a generic import/CUDA message.

## Tests and scope

The regression suite tests Windows uppercase AMD64 versus active lowercase/Linux
markers, required missing packages, wrong installed versions, secret redaction,
bounded logs, timeouts/native exit codes, and a real failing subprocess.

`tools/check_dependency_closure.py` is a CI/diagnostic tool for a **disposable**
venv named `venv.__modly_staging`. It uses the real complete portable dependency
installer, package metadata, imports, pip check and binary operator verification.
Only the GPU phase is replaced with the shared imports-only probe in this tool.
There is no skip-CUDA environment variable in production setup. Its JSON report
always says `cuda_tested: false`; it is not full Modly/model acceptance.

The workflow includes Windows Python 3.11/3.12 with both supported Torch builds,
Modly's exact standalone Windows CPython 3.11.9, and representative Linux x64 and
native ARM64 lanes. CI compiler-discovery helpers are forbidden. Test results
belong to the specific workflow run, not to this checklist.

Do not delete models_dir to apply this fix. The model revision IDs/hashes and
published CPU wheels remain unchanged. Update extension code, then Repair; valid
model assets are reused. Package cache reuse still depends on cache availability.
