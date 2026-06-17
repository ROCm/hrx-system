# Shared AMDGPU Build Tools

This package owns the AMDGPU target catalog and the small Bazel/CMake helpers
used by runtime, libhrx, and future runtime-side consumers that need AMDGPU
code objects. The center of gravity is target selection, not any one runtime
package:

- `target_map.py` is the source of truth for exact GPU architectures,
  code-object targets, and family selectors.
- `selectors.bzl` and `selectors.cmake` validate selectors and expand them to
  either exact HSA ISA targets or compatible code-object targets.
- `binary.bzl` and `binary.cmake` build C sources into AMDGPU ELF shared
  objects with clang, `llvm-link`, `lld`, and optionally `llvm-objcopy`.

## Target Vocabulary

Selectors accepted by the shared helpers fall into three groups:

| Selector kind | Example | Meaning |
| --- | --- | --- |
| Exact target | `gfx942` | HSA ISA architecture suffix reported by the runtime device. |
| Code-object target | `gfx9-4-generic` | LLVM `-march` value used to build one compatible code object. |
| Family selector | `gfx94X-all`, `dgpu-all`, `igpu-all` | TheRock-style selector that expands through exact targets. |

There are two expansion modes:

- `code-object` expands any selector to the smallest known compatible set of
  code-object targets. Runtime builtin blobs and libhrx CTS HSACOs use this
  mode so `gfx942` and `gfx94X-all` both select `gfx9-4-generic`.
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

`gfx11.7` processors are available as explicit exact selectors, but they are
not folded into `gfx11-generic` because LLVM models them as separate processors
outside the `gfx11-generic` compatibility set today. `gfx12-5-generic` is
available as an explicit selector, but consumers decide whether it belongs in
their default checked-in artifact sets.

## Generated Files

Running `target_map.py` emits generated fragments consumed by multiple layers:

| Generated file | Consumer |
| --- | --- |
| `build_tools/amdgpu/target_map.bzl` | Bazel selector helpers. |
| `build_tools/amdgpu/target_map.cmake` | CMake selector helpers. |
| `build_tools/amdgpu/target_map.h` | C/C++ tests that need exact-to-code-object lookup. |
| `runtime/src/iree/hal/drivers/amdgpu/util/target_id_map.inl` | Runtime AMDGPU device-library lookup. |

The generated files are checked in. The presubmit check runs:

```bash
python build_tools/amdgpu/target_map.py --check
```

Architecture updates start in `EXACT_TARGET_CODE_OBJECTS` and
`TARGET_FAMILIES` in `target_map.py`. The evidence to check before changing the
map is TheRock's `cmake/therock_amdgpu_targets.cmake` for selector and family
membership, and LLVM AMDGPU generic processor documentation/tablegen data for
generic code-object compatibility.

After editing the map:

```bash
python build_tools/amdgpu/target_map.py
buildifier build_tools/amdgpu/BUILD.bazel build_tools/amdgpu/*.bzl
```

The runtime `target_id_map.inl` is generated from the same map so the loader and
build rules agree about exact-to-code-object compatibility.

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
| `//runtime/src/iree/hal/drivers/amdgpu:targets` | Runtime AMDGPU support: embedded device libraries, AMDGPU/HIP HAL CTS, source-built util test/benchmark assets, and libhrx CTS native executable assets. |

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
per-tool overrides such as `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_BINARY`.
ROCm distributions may expose `clang`/`amdclang` launcher shims that exec a
versioned driver next to their observed `argv[0]` path. The repository resolves
those shims to the matching versioned driver before exposing a Bazel executable
target, because Bazel wraps local tools through generated symlinks and must not
change the driver's sibling lookup behavior.
When the toolchain repository is inert, selected source-built binaries are
incompatible instead of referencing missing tool labels.

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
| `IREE_HAL_AMDGPU_TARGETS` | Runtime AMDGPU support: embedded device libraries, AMDGPU/HIP HAL CTS, source-built util test/benchmark assets, and libhrx CTS native executable assets. |

`iree_amdgpu_binary(...)` builds a single AMDGPU ELF shared object.
`iree_amdgpu_binary_variants(...)` builds one shared object per selected
code-object target and can return both the aggregate CMake targets and generated
file names. `iree_amdgpu_binary_variants_embed_data(...)` builds the same
selected variants and immediately feeds them through `iree_c_embed_data`, giving
the consuming C/C++ code a generated table of contents. Test-owned HSACOs should
use the embed-data form so installed tests, Bazel runfiles, and local CMake
builds all consume the same in-process bytes.

These rules require configured LLVM/ROCm tools through the existing CMake
variables: `IREE_CLANG_BINARY`, `IREE_LLVM_LINK_BINARY`, `IREE_LLD_BINARY`, and
`IREE_CLANG_BUILTIN_HEADERS_PATH`, with `IREE_LLVM_OBJCOPY_BINARY` required when
`MINIMIZE` is used.

## Consumer Shape

The shared layer deliberately stops at generic AMDGPU target and binary
mechanics. Runtime, libhrx, and future Loom code should own their policy:

- default selector lists;
- whether a selector enables a test suite or only provides data;
- artifact naming and installation layout;
- whether source-built artifacts are required or optional.

That split keeps this package usable by several products without turning it
into a runtime-specific configuration surface.
