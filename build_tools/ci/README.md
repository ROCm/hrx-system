# Reproducing CI Jobs

The source-tree CI command runner is
[`build_tools/devtools/ci.py`](../devtools/ci.py). Run commands from the
repository root after the [developer tool setup](../../BUILDING.md#quick-start).
Ordinary development uses the build and test commands in that guide; CI profiles
also select feature sets, test resources, and exclusions for a particular job.

## Watch and classify a Loom workflow

`github_actions_watch.py` reports the current Loom workflow for a pull request,
groups the same failure across configurations, and distinguishes exact external
infrastructure failures from repository failures and explicitly tracked flakes.
It is read-only: it does not retry, cancel, or otherwise mutate GitHub state.

```bash
python -m build_tools.ci.github_actions_watch 2198
python -m build_tools.ci.github_actions_watch 2198 --watch
python -m build_tools.ci.github_actions_watch run:39665280602 --json
```

The command authenticates with `GH_TOKEN`, `GITHUB_TOKEN`, or the current `gh`
login. A snapshot exits 3 while work remains. Terminal repository or unknown
failures exit 1, tool/API errors exit 2, and exact infrastructure or tracked
failures exit 4. Clean and proven-obsolete runs exit 0. While watching, output
appears on state changes and as a five-minute heartbeat for long runner queues.
The human view expands the four oldest active jobs and summarizes the rest;
`--json` retains every active job and failure link.

## Find the job's configuration

The workflow is the source of truth for its command, compiler environment,
device target, and assigned resources:

- [IREE source-tree jobs](../../.github/workflows/ci_iree.yml).
- [libHRX jobs](../../.github/workflows/ci_libhrx.yml).
- [Source-tree profile definitions](../devtools/ci_config.py) and
  [command planning](../devtools/ci.py).

Copy the failing job's command and relevant environment, including `CC`, `CXX`,
`AR`, SDK paths, and any coverage options. A reproduction with a different
compiler or enabled dependency set exercises a different build. Host compiler
selection and the ROCm device compiler are independent.

Inspect the planned commands before running a profile:

```bash
python build_tools/devtools/ci.py --help
python build_tools/devtools/ci.py iree-bazel-cpu-asan --dry-run
```

Profiles can regenerate `.bazelrc.configured`; save the configuration command
for the development build before running one. CMake profiles use separate trees
under `build/ci/`.

## Select a reproduction

Command names follow `iree-<build-system>-<target-group>[-<configuration>]`.
Use the configuration from the failing job; individual sanitizer commands let
one failure be reproduced without running the aggregate sanitizer profile.

```bash
python build_tools/devtools/ci.py iree-bazel-cpu-asan --target //runtime/src/iree/base/...
python build_tools/devtools/ci.py iree-bazel-vulkan --target //runtime/...
python build_tools/devtools/ci.py iree-bazel-amdgpu-tsan --amdgpu-target gfx942
python build_tools/devtools/ci.py iree-bazel-xdna-asan
python build_tools/devtools/ci.py iree-cmake-cpu-ubsan
```

`--target` can narrow profiles that accept Bazel target patterns. CMake profiles
select their tests through CTest names and labels. `--verbose` prints the
underlying commands; `--keep-going` runs independent phases after a failure and
reports the failed phases at the end.

The native Windows GPU profile requires an exact GPU target:

```powershell
python build_tools/devtools/ci.py iree-bazel-amdgpu-native-asan --amdgpu-target gfx1201
```

This profile uses native libamdf GPU queues without requiring ROCr or an NPU.
The `iree-bazel-amd-client[-asan]` profiles cover the GPU/NPU client configuration;
their admitted resources depend on the host platform. The Linux
`iree-bazel-amdgpu[-<sanitizer>]` and `iree-cmake-amdgpu[-<sanitizer>]` profiles
also use native libamdf GPU queues, with the legacy AMDGPU HAL disabled. They
retain the ROCm device compiler for HIP code-object fixtures; execution does
not load the ROCr-backed HAL.
The [profile definitions](../devtools/ci_config.py) specify the package scope
and resources for each.

AMD profiles forward `HAL_VISIBLE_XDNA_DEVICES`, `HAL_VISIBLE_RDNA_DEVICES`,
`HAL_VISIBLE_CDNA_DEVICES`, and `HAL_VISIBLE_AMDGPU2_DEVICES` into Bazel test
actions. Each list selects that HAL scheme's unmasked supported device ordinals;
the lists are independent, and empty values remain empty. For example, a runner
can set `HAL_VISIBLE_RDNA_DEVICES=1` so ordinary `--device=rdna` callers use its
second supported RDNA GPU. Native libamdf CTS selection remains explicit through
`--amdf_gpu_target` and `AMDF_CTS_GPU_NATIVE_IDENTITY`; HAL visibility does not
modify libamdf's topology.

Compiler-only AMDGPU checks use `iree-bazel-loom-amdgpu` or
`iree-cmake-loom-amdgpu`; they need no GPU. Repository-wide compile checks use
`iree-bazel-repository-build`. Platform-incompatible targets are excluded by
their build constraints.

## Hardware and test admission

Every `iree-run-requirement=...` declared by a test must be available. A test
using multiple APIs or devices needs all of them; one matching participant is
insufficient. The runner derives exclusions from these declarations and rejects
unknown requirement IDs. Configuration makes tests eligible for selection; it
does not install drivers, provide device permissions, or reserve hardware.

Tests sharing a device declare `GPU_DEVICE_RESOURCE_GROUP` from
[`test_resources.bzl`](../bazel/test_resources.bzl). CTest uses the `gpu-device`
lock, and Bazel uses local exclusivity. Separate invocations still require
exclusive device assignment by the job or developer. A successful invocation
with every hardware case skipped does not establish hardware coverage: inspect
the test results for cases that actually executed.

## Preserve results

Bazel profiles can retain per-phase test logs and XML in a fresh directory:

```bash
python build_tools/devtools/ci.py iree-bazel-cpu-asan \
  --target //runtime/src/iree/base/... \
  --bazel-test-results-dir build/ci-results/cpu-asan
```

Each phase has its own directory so repeated runs of a test with different
filters or settings remain distinguishable. `--bazel-profile-dir` similarly
captures one Bazel execution profile per phase. The workflows upload these
artifacts; the test logs and phase exit status identify the failing operation.

Repository hygiene and commit checks are documented in the
[Lefthook guide](../lefthook/README.md).
