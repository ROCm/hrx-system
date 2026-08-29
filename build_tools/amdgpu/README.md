# Shared AMDGPU Build Tools

This package owns the AMDGPU target catalog and the small Bazel/CMake helpers
used by runtime, libhrx, and future runtime-side consumers that need AMDGPU
code objects. The center of gravity is target selection, not any one runtime
package:

- `target_map_data.py` is the source of truth for exact GPU architectures,
  code-object targets, target-ID feature support, and qualified device-binary
  artifacts. `target_map.py` owns family selectors and generates the consumer
  projections.
- `selectors.bzl` and `selectors.cmake` validate selectors and expand them to
  exact HSA ISA targets, compatible code-object targets, or the complete
  device-binary artifact set.
- `binary.bzl` and `binary.cmake` build C sources into AMDGPU ELF shared
  objects with clang, `llvm-link`, `lld`, and optionally `llvm-objcopy`.

## Target Vocabulary

Selectors accepted by the shared helpers fall into three groups:

| Selector kind | Example | Meaning |
| --- | --- | --- |
| Exact target | `gfx942` | HSA ISA architecture suffix reported by the runtime device. |
| Code-object target | `gfx9-4-generic` | LLVM `-march` value used to build one compatible code object. |
| Family selector | `gfx94X-all`, `dgpu-all`, `igpu-all` | TheRock-style selector that expands through exact targets. |

There are three expansion modes:

- `code-object` expands any selector to the smallest known compatible set of
  code-object targets. Runtime builtin blobs and libhrx CTS HSACOs use this
  mode so `gfx942` and `gfx94X-all` both select `gfx9-4-generic`.
- `device-binary` preserves the code-object fallback set and prepends any
  exact-target artifacts required for safe runtime selection. This is the mode
  for embedded support libraries whose codegen requirements can vary within a
  code-object family.
- `exact` expands selectors to exact HSA ISA targets. This mode is for tools or
  tests that must name precise device architectures.

The current generic-family map is:

| Family | Exact targets | Code-object target |
| --- | --- | --- |
| `gfx9` GCN | `gfx900`, `gfx902`, `gfx904`, `gfx906`, `gfx909`, `gfx90c` | `gfx9-generic` |
| `gfx9-4` CDNA | `gfx940`, `gfx941`, `gfx942`, `gfx950` | `gfx9-4-generic` |
| `gfx10.1` RDNA | `gfx1010`, `gfx1011`, `gfx1012`, `gfx1013` | `gfx10-1-generic` |
| `gfx10.3` RDNA | `gfx1030`, `gfx1031`, `gfx1032`, `gfx1033`, `gfx1034`, `gfx1035`, `gfx1036` | `gfx10-3-generic` |
| `gfx11` RDNA/APU | `gfx1100`, `gfx1101`, `gfx1102`, `gfx1103`, `gfx1150`, `gfx1151`, `gfx1152`, `gfx1153` | `gfx11-generic` |
| `gfx11.7` RDNA 4m | `gfx1170`, `gfx1171`, `gfx1172` | exact target only |
| `gfx12` RDNA | `gfx1200`, `gfx1201` | `gfx12-generic` |
| `gfx12.5` RDNA | `gfx1250`, `gfx1251` | `gfx12-5-generic` |

`gfx11.7` processors are not folded into `gfx11-generic` because LLVM models
them as a separate compatible family that the pinned device toolchain does not
yet expose. `gfx12-5-generic` is
available as an explicit selector, but consumers decide whether it belongs in
their default checked-in artifact sets. The `device-binary` expansion also
includes `gfx1250-a0` because the generic gfx12.5 artifact is B0-qualified and
cannot safely serve that physical revision.

## Generated Files

Running `target_map.py` emits generated fragments consumed by multiple layers:

| Generated file | Consumer |
| --- | --- |
| `build_tools/amdgpu/target_map.bzl` | Bazel selector helpers. |
| `build_tools/amdgpu/target_map.cmake` | CMake selector helpers. |
| `build_tools/amdgpu/elf_machine_map.inl` | C/C++ ELF machine decode tables for runtime, libhrx, and Loom. |
| `build_tools/amdgpu/target_map.h` | C/C++ tests that need exact-to-code-object lookup. |
| `runtime/src/iree/hal/executable/amdgpu/target_id_map.inl` | Runtime target-ID mappings and processor facts. |
| `runtime/src/iree/hal/drivers/amdgpu/util/device_library_target_map.inl` | Runtime qualified device-library artifact lookup. |

The generated files are checked in. The presubmit check runs:

```bash
python3 build_tools/amdgpu/target_map.py --check
```

Architecture updates start in `AMDGPU_EXACT_TARGET_INFOS`,
`AMDGPU_GENERIC_CODE_OBJECT_INFOS`, or `AMDGPU_DEVICE_BINARY_VARIANTS` in
`target_map_data.py`, and in `TARGET_FAMILIES` in `target_map.py`. The evidence
to check before changing the map is TheRock's
`cmake/therock_amdgpu_targets.cmake` for selector and family membership, and
LLVM AMDGPU generic processor documentation/tablegen data for generic
code-object compatibility and variant codegen options.

After editing the map:

```bash
python3 build_tools/amdgpu/target_map.py
buildifier build_tools/amdgpu/BUILD.bazel build_tools/amdgpu/*.bzl
```

The runtime `target_id_map.inl` is generated from the same map so the loader and
build rules agree about exact-to-code-object compatibility. The
`elf_machine_map.inl` fragment is generated from a separate ELF decode table:
it may recognize legacy AMDGPU machine values that are not exposed as build
selectors.

## Bazel Integration

Packages define their own selector flags with:

```python
iree_amdgpu_target_selectors_flag(
    name = "targets",
    build_setting_default = ["gfx9-4-generic"],
)
```

Consumers then use:

- `iree_amdgpu_expand_target_selectors(...)` for Starlark-time expansion.
- `iree_amdgpu_target_selector_config_settings(...)` when a `select()` must
  ask whether a code-object target was requested.
- `iree_amdgpu_binary(...)` for a single AMDGPU binary.
- `iree_amdgpu_binary_variants(...)` for one binary per code-object target plus
  an aggregate filegroup containing the selected variants.
- `iree_amdgpu_binary_variants_embed_data(...)` when the selected binaries are
  private test/runtime assets that should be linked into the consuming
  executable or library instead of discovered through runfiles or install
  paths.

The runtime build owns the shared AMDGPU support selector:

| Bazel flag | Consumer |
| --- | --- |
| `//runtime/src/iree/hal/drivers/amdgpu:targets` | Runtime AMDGPU support: embedded device libraries, AMDGPU HAL CTS, source-built util test/benchmark assets, and libhrx CTS native executable assets. |

Source-built binaries use the optional `@iree_amdgpu_device_toolchain`
repository. It is inert by default. A real producer is selected with:

```bash
--repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm
--repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=llvm-tools
--repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=llvm-project
--repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=auto
```

Useful path overrides include `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_PATH`,
`IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_TOOLS_DIR`, `IREE_ROCM_PATH`, and
`IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_BINARY`. `auto` retains the same search
surface and additionally checks `PATH` after configured tool and ROCm roots. If
that search does not find one complete coherent toolchain, source-built targets
remain incompatible while unrelated targets continue to configure. Explicit
`rocm` and `llvm-tools` modes reject incomplete configurations.

The selected Clang is the toolchain anchor. The repository canonicalizes it,
verifies AMDGPU target support, and asks that executable for `llvm-ar`,
`llvm-link`, `ld.lld`, `llvm-objcopy`, the Clang resource directory, and the
optional offload bundler. Compatibility paths may participate in discovery,
but only Clang-reported canonical files are exposed to Bazel; tools from
different LLVM installations are never mixed.

On ELF hosts, local tools run with their exact non-glibc dynamic-library
closure. The repository inspects only the selected executables and does not
import an LLVM or ROCm library tree. Ambient loader variables are replaced
during discovery and execution, so the closure does not depend on the shell
that configured the worktree. Windows tools retain their native `.exe` names
and execute directly from the selected installation; no POSIX launcher or ELF
inspection participates in Windows repository setup.

When the toolchain repository is inert, selected source-built binaries are
incompatible instead of referencing missing tool labels.

`iree_amdgpu_binary_variants[_embed_data]` accepts `source_format = "hip"` for
device-only HIP fixtures that need ROCm device libraries such as `ocml.bc` and
`ockl.bc`. HIP fixture generation is an optional extension of the base AMDGPU
toolchain and is enabled only when both those device libraries and the matching
Clang offload bundler are available. Missing HIP pieces do not disable base
AMDGPU device compilation. Host C and C++ compilation continues to use the
configured host compiler.

## CMake Integration

Projects add this directory to `CMAKE_MODULE_PATH` and include the small module
names:

```cmake
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/build_tools/amdgpu")
include(binary)
include(selectors)
```

Selector expansion is explicit at the consuming package:

```cmake
iree_amdgpu_expand_target_selectors(
  _TARGETS
  "${IREE_AMDGPU_TARGET_EXPANSION_CODE_OBJECT}"
  ${MY_AMDGPU_TARGETS}
)
```

The matching CMake selector variable is:

| CMake variable | Consumer |
| --- | --- |
| `IREE_HAL_AMDGPU_TARGETS` | Runtime AMDGPU support: embedded device libraries, AMDGPU HAL CTS, source-built util test/benchmark assets, and libhrx CTS native executable assets. |

`iree_amdgpu_binary(...)` builds a single AMDGPU ELF shared object.
`iree_amdgpu_binary_variants(...)` builds one shared object per selected
code-object target and can return both the aggregate CMake targets and generated
file names. `iree_amdgpu_binary_variants_embed_data(...)` builds the same
selected variants and immediately feeds them through `iree_c_embed_data`, giving
the consuming C/C++ code a generated table of contents. Test-owned HSACOs should
use the embed-data form so installed tests, Bazel runfiles, and local CMake
builds all consume the same in-process bytes.

These rules consume the resolved
`IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_*` capability configured by the containing
project. The device clang, llvm-ar, llvm-link, lld, llvm-objcopy, and clang
resource headers form one tool set independent of `CMAKE_C_COMPILER`. Direct
source-producing rules fail when that capability is unavailable. Variant and
CTS aggregates expose no source-built artifacts, allowing normal builds to use
checked-in device binaries without discovering LLVM or ROCm tools.

## Consumer Shape

The shared layer deliberately stops at generic AMDGPU target and binary
mechanics. Runtime, libhrx, and future Loom code should own their policy:

- default selector lists;
- whether a selector enables a test suite or only provides data;
- artifact installation and packaging layout;
- whether source-built artifacts are required or optional.

That split keeps this package usable by several products without turning it
into a runtime-specific configuration surface.
