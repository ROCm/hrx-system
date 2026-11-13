---
hide:
  - navigation
---

# Targets

Loom can compile for a target without that hardware attached. Running the result
also requires an execution path and a compatible device or host. The tables
below separate those capabilities from how thoroughly we have exercised and
optimized them. This page tracks current development; installed releases may
lag behind it.

**RDNA 3, RDNA 3.5, RDNA 4, CDNA 3, and CDNA 4 are our primary supported
targets.** The other targets are experimental, with different amounts of
working functionality, correctness coverage, and performance tuning. Loom as
a whole remains a tech preview.

**Kernels** means dispatchable `kernel.def` programs. **Pipelines** means
resident `pipeline.def` programs with independently progressing strands and
channels; ordinary sequences of kernel launches use
[command programs](../guide/command-programs.md).
The [maturity ratings](#reading-the-ratings) are maintained engineering
assessments, rather than percentages of tests passed or instructions implemented.

## AMD GPUs

Native execution uses the operating system's AMD driver through libamdf.
RDNA runs on Linux and Windows, including a WSL execution path; CDNA targets
Linux. The [driver requirements](https://github.com/ROCm/hrx-system/tree/main/libamdf#native-drivers)
describe the platform prerequisites. Select `--device=amdgpu`, or use
`--device=rdna` and `--device=cdna` to select one family.

### RDNA

<div class="loom-target-matrix" markdown>

| Target family | Compilation | Execution | Kernels | Pipelines | Correctness | Performance |
| --- | --- | --- | --- | --- | --- | --- |
| **RDNA 3**<br>`gfx1100`–`gfx1103` | HSACO | Native | Yes | — | Broad | Tuned |
| **RDNA 3.5**<br>`gfx1150`–`gfx1153` | HSACO | Native | Yes | — | Broad | Tuned |
| **RDNA 4**<br>`gfx1200`, `gfx1201` | HSACO | Native | Yes | — | Broad | Tuned |
| **GFX11.7 / RDNA 4m**<br>`gfx1170`–`gfx1172` | HSACO | Native | Yes | — | Focused | Baseline |

</div>

### CDNA

<div class="loom-target-matrix" markdown>

| Target family | Compilation | Execution | Kernels | Pipelines | Correctness | Performance |
| --- | --- | --- | --- | --- | --- | --- |
| **CDNA 2**<br>`gfx90a` | HSACO | Native | Yes | — | Focused | Baseline |
| **CDNA 3**<br>`gfx940`, `gfx941`, `gfx942` | HSACO | Native | Yes | — | Broad | Tuned |
| **CDNA 4**<br>`gfx950` | HSACO | Native | Yes | — | Broad | Tuned |
| **CDNA 5 / GFX12.5**<br>`gfx1250`, `gfx1250-strict`, `gfx1251` | HSACO | Native | Yes | — | Focused | Baseline |

</div>

The MI450 generation uses [CDNA 5](https://rocm.blogs.amd.com/ecosystems-and-partners/cdna5-helios/README.html).
For `gfx1250`, revision A0 selects `gfx1250-strict`; live device selection
preserves that distinction. An offline compile must name the appropriate
profile.

Generic profiles retain the common capabilities of a family:
`gfx11-generic` for RDNA 3/3.5, `gfx12-generic` for RDNA 4,
`gfx9-4-generic` for CDNA 3/4, and `gfx12-5-generic` for GFX12.5.
An exact profile can expose additional instructions and scheduling facts.

GCN, CDNA 1, and RDNA 1/2 do not currently have native Loom compilation paths.
A compatible Vulkan device can use the [SPIR-V path](#portable-execution)
instead; Vulkan features and driver support determine its available operations.

## AMD XDNA NPUs

Loom emits native AIE2P tile programs and their array configuration, packaged in
an XDNA executable. Execution uses the system's NPU driver through libamdf on
Linux or Windows, selected with `--device=xdna`.

<div class="loom-target-matrix" markdown>

| Target family | Compilation | Execution | Kernels | Pipelines | Correctness | Performance |
| --- | --- | --- | --- | --- | --- | --- |
| **XDNA 2 / AIE2P**<br>Strix, Strix Halo, Krackan | XDNA ELF | Native | Array entries | Partial | Focused | Baseline |

</div>

Array entries own the device configuration and worker programs; they are not
the GPU work-item execution model. The current resident pipeline path supports
explicit worker placement, specialized storage, and bounded channel protocols.
Worker code must fit the tile instruction budget. The
[resident pipelines walkthrough](xdna-pipelines.md) follows a complete program
and explains these boundaries.

Use an exact deployment profile after the `amd.xdna.aie2p:` family prefix:

| Device | Profile |
| --- | --- |
| Strix / NPU4 | `amd.xdna.strix.17f0_10` |
| Strix Halo / NPU5 | `amd.xdna.strix_halo.17f0_11` |
| Krackan / NPU6 | `amd.xdna.krackan.17f0_20` |

These profiles describe XDNA 2 devices. First-generation XDNA is not covered.

## x86 CPUs

The x86 backend emits native code for AMD and Intel CPUs. The task runtime
executes kernels through a shared CPU thread pool; select `--device=task`.
Ordinary functions can also be emitted as relocatable or loadable ELF objects
for integration into another application.

<div class="loom-target-matrix" markdown>

| Target family | Compilation | Execution | Kernels | Pipelines | Correctness | Performance |
| --- | --- | --- | --- | --- | --- | --- |
| **x86-64**<br>Scalar, AVX2, AVX-512 | ELF | Native | Yes | — | Growing | Targeted |

</div>

`x86:scalar`, `x86:avx2`, and `x86:avx512` select explicit execution baselines.
Named processor profiles such as `x86:znver5` also supply processor features and
scheduling information. Live device selection chooses a compatible baseline.
The ordinary native-function interface defaults to the SysV ABI; kernel
artifacts use the task runtime's ABI. See
[native callable functions](../workflows/compile-artifacts.md#emit-native-callable-functions)
for the distinction.

## Portable execution

SPIR-V targets Vulkan devices, including GPUs from AMD, Intel, and NVIDIA.
WebAssembly targets a Wasm host such as a browser or standalone engine.
The driver or host supplies the machine-specific execution.

<div class="loom-target-matrix" markdown>

| Target family | Compilation | Execution | Kernels | Pipelines | Correctness | Performance |
| --- | --- | --- | --- | --- | --- | --- |
| **SPIR-V / Vulkan** | SPIR-V | Vulkan | Yes | — | Growing | Targeted |
| **WebAssembly** | Wasm | Wasm host | — | — | Focused | Baseline |

</div>

The Vulkan path requires Vulkan 1.3 and buffer device address support. Scalar
types, subgroup operations, and cooperative matrix/vector operations depend on
the features and properties exposed by the device. Live selection carries those
facts into compilation. A working model on one Vulkan device does not establish
the same feature coverage or performance on every vendor's driver.

WebAssembly currently compiles ordinary functions, structured control flow, and
supported scalar/vector operations. It does not yet emit dispatchable Loom
kernels. Its `wasm:simd128` profile and
[module workflow](../workflows/compile-artifacts.md#emit-a-webassembly-module)
describe the current host interface.

### VM functions

`vm:core` compiles ordinary functions into `.vm` bytecode for the IREE VM.
This is a software execution target for scalar, buffer, and managed-reference
programs. Its compiler path is experimental, with focused correctness coverage
and baseline performance. It does not provide kernel/workgroup execution,
resident pipelines, or HAL command programs. The
[VM target guide](https://github.com/ROCm/hrx-system/tree/main/loom/src/loom/target/arch/vm)
describes the callable module boundary.

## Select a target

For execution, choose a device and let Loom derive its target facts. Inspect
the drivers compiled into your installation and the devices they can discover:

```sh
loom-run --list_drivers
loom-run --list_devices=amdgpu
loom-run --list_devices=xdna
loom-run --list_devices=vulkan
```

The same device choices apply to `loom-run`, `loom-test`, `loom-benchmark`, and
`loom-serve`. Discovery is distinct from execution permission and driver
compatibility; the [installation guide](../getting-started/acquiring-loom.md)
covers those requirements.

For offline compilation, `--target=family:selector` selects a profile without
opening a device. These are representative selections:

| Output | Compiler selection | Execution selection |
| --- | --- | --- |
| AMD GPU kernels | `--target=amdgpu:gfx1151` | `--device=amdgpu` |
| XDNA array program | `--target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11` | `--device=xdna` |
| CPU kernels or functions | `--target=x86:avx2` | `--device=task` for kernels |
| Vulkan kernels | `--target=spirv:vulkan1.3+bda` | `--device=vulkan` |
| WebAssembly functions | `--target=wasm:simd128` | Wasm host |
| VM functions | `--target=vm:core` | IREE VM host |

The input program must fit the selected target. Compilation cannot substitute a
different execution model for target-specific source. The
[artifact workflow](../workflows/compile-artifacts.md) covers root selection,
generic versus exact profiles, and output formats.

### Enable targets in a custom SDK

Ordinary compiler builds include AMD GPU, XDNA, x86, SPIR-V, and VM support.
WebAssembly is opt-in. An installed SDK's capabilities are fixed by its build;
a runtime `--target` flag selects an included backend rather than enabling a
missing one.

The [build configuration guide](https://github.com/ROCm/hrx-system/blob/main/BUILDING.md#shared-project-configuration)
owns source-build setup. Its target presets are:

| Compiler target set | Bazel preset | CMake option |
| --- | --- | --- |
| Ordinary targets | `--config=loom-target-default` | `-DLOOM_TARGET_DEFAULT=ON` |
| All targets, including Wasm | `--config=loom-target-all` | `-DLOOM_TARGET_ALL=ON` |
| AMD GPU | `--config=loom-target-amd-gpu` | `-DLOOM_TARGET_ALL_AMD_GPU=ON` |
| AMD NPU | `--config=loom-target-amd-npu` | `-DLOOM_TARGET_ALL_AMD_NPU=ON` |
| x86 CPU | `--config=loom-target-cpu` | `-DLOOM_TARGET_ALL_CPU=ON` |
| SPIR-V | `--config=loom-target-spirv` | `-DLOOM_TARGET_ALL_SPIRV=ON` |
| WebAssembly | `--config=loom-target-wasm` | `-DLOOM_TARGET_ALL_WASM=ON` |
| VM | `--config=loom-target-vm` | `-DLOOM_TARGET_ALL_VM=ON` |

Bazel presets replace the complete target selection; use one per invocation.
AMDGPU profiles can be selected separately with
`--//loom/config/arch/amdgpu:targets=...` or `LOOM_ARCH_AMDGPU_TARGETS`.
The default `loom_defaults` includes all native AMDGPU profiles listed above.

Compiler enablement and execution-driver enablement are separate. Native AMD
GPU and NPU execution use `IREE_HAL_DRIVER_AMD_GPU` and
`IREE_HAL_DRIVER_AMD_XDNA`; CPU kernels use `IREE_HAL_DRIVER_TASK`, and Vulkan
uses `IREE_HAL_DRIVER_VULKAN`. In Bazel's complete
`--//runtime/config/hal:drivers=...` list, their selectors are `amd-gpu`,
`amd-xdna`, `task`, and `vulkan`; `amd` selects both native AMD engines.
The build guide covers libamdf and loader prerequisites.

## Reading the ratings

The feature columns describe implemented paths. **—** means that programming
model has no current implementation on the target. **Partial** names a working
subset whose constraints are described in the target section.

| Axis | Level | Meaning |
| --- | --- | --- |
| Correctness | **Broad** | Representative workloads exercise a wide range of inputs, types, layouts, and control flow. |
| Correctness | **Growing** | Multiple workloads and operation families work; coverage is still uneven across inputs and features. |
| Correctness | **Focused** | Specific programs and feature families have been checked; broad workload coverage has not been established. |
| Performance | **Tuned** | Substantial target-specific optimization across representative workloads. |
| Performance | **Targeted** | Selected operations or workloads have received tuning; performance remains uneven. |
| Performance | **Baseline** | A functional path exists; broad performance tuning has not been established. |

These judgments describe the target implementation, not every model, device
SKU, or driver release. A high rating on one axis does not imply a high rating
on the other. A model's own validation and benchmark results establish its
coverage and performance for a particular deployment.
