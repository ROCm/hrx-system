# AMDGPU Device Binaries

This package builds the small device-side AMDGPU support library embedded into
the HAL runtime. These code objects contain blit kernels and runtime utility
kernels; they are selected by runtime ISA lookup when an AMDGPU HAL device is
created.

The shared AMDGPU target vocabulary, exact-to-code-object map, generated
Bazel/CMake/C fragments, and architecture update flow are documented in
[`build_tools/amdgpu/README.md`](../../../../../../../../build_tools/amdgpu/README.md).
This package owns the runtime-specific policy: which embedded blobs are checked
in by default, how they are packed into the runtime table of contents, and how
to regenerate the prebuilt artifacts.

## Regenerating Code Objects

This generator supports the checked-in code object flow where the runtime build
consumes prebuilt blobs instead of building LLVM. Regenerate those code objects
only when the builtin device sources change or when adding/removing an
architecture from the checked-in set:

```bash
python build_tools/scripts/amdgpu_device_binaries.py \
  --output-dir runtime/src/iree/hal/drivers/amdgpu/device/binaries/prebuilt \
  --rocm-path /path/to/rocm-or-therock-sdk \
  --targets gfx9-generic,gfx90a,gfx9-4-generic,gfx10-1-generic,gfx10-3-generic,gfx11-generic,gfx12-generic
```

The script accepts the shared selector vocabulary: exact targets such as
`gfx1100`, code-object targets such as `gfx11-generic`, and families such as
`gfx94X-all`. If `--targets` is omitted, the script uses
`IREE_HAL_AMDGPU_TARGETS` when set, otherwise it builds the checked-in
generic-family set plus `gfx90a`. Pass `--all-targets` to build every known
code-object target. Some ROCm releases may not yet support every generic target
recorded in the map; in that case the script fails before compilation and
reports the unsupported target names.

Tool discovery is intentionally compatible with both in-tree and out-of-tree
LLVM flows. Explicit `--clang`, `--llvm-link`, `--lld`, and `--llvm-objcopy`
flags win. Otherwise the script checks individual
`IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_*`, `IREE_*`, and LLVM environment variables,
host-tool directories such as
`IREE_HOST_BIN_DIR`, `IREE_HOST_TOOLS`, `IREE_BINARY_DIR`,
`IREE_LLVM_TOOLS_DIR`, and `LLVM_TOOLS_BINARY_DIR`, then ROCm roots: explicit
`--rocm-path` entries, `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_PATH`,
`IREE_ROCM_PATH`, the configure-style `ROCM_PATH`, `ROCM_ROOT`, `ROCM_HOME`,
`HIP_PATH` variables, a root inferred from `hipcc` on `PATH`, and `/opt/rocm`.
For ROCm installs the script searches standard layouts such as `llvm/bin`,
`lib/llvm/bin`, and `bin`; after finding `clang` or `amdclang` it also asks
that compiler where its companion LLVM tools live with `--print-prog-name`.
When the discovered compiler is a ROCm launcher shim with a matching versioned
driver next to it, the generator uses the versioned driver. This matches the
Bazel source-build path and avoids depending on shim-relative sibling lookup
after another build system has wrapped or symlinked the executable.

By default the generator keeps only the `.kd` kernel descriptor symbols in the
regular symbol tables. `llvm-strip --strip-all` and
`llvm-objcopy --strip-sections` can remove symbol or section metadata that ROCr
may still load or inspect when resolving builtin kernels by name. The device
kernel declarations carry the liveness contract via
`IREE_AMDGPU_ATTRIBUTE_KERNEL`; the generator and live source builds only
provide a local-all version script and remove local symbols after linking.

## Build Modes

Normal runtime builds use checked-in blobs from `prebuilt/`. The optional Bazel
AMDGPU device toolchain repository defaults to an inert stub and source builds
must opt into a real producer with
`--repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm`, `llvm-tools`,
`llvm-project`, or `auto`. This keeps AMDGPU HAL runtime rebuilds independent
of the LLVM submodule and ROCm tools unless a developer explicitly asks to
rebuild the device code objects from source.

Bazel source mode is enabled with:

```bash
iree-bazel-build \
  --config=amdgpu_device_binaries_source_rocm \
  --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_PATH=/path/to/rocm-or-therock-sdk \
  --//runtime/src/iree/hal/drivers/amdgpu:targets=gfx11-generic \
  //runtime/src/iree/hal/drivers/amdgpu/device/binaries:toc
```

Use `--config=amdgpu_device_binaries_source_llvm_project` when deliberately
building through the in-tree `@llvm-project` repository instead of ROCm tools.
Both Bazel source modes invoke `build_tools/scripts/amdgpu_device_binaries.py`;
the target selector flag accepts the shared exact, code-object, and family
selectors.

CMake uses the matching cache variables:

- `IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE=prebuilt|source`, default
  `prebuilt`.
- `IREE_HAL_AMDGPU_TARGETS`, default
  `gfx9-generic;gfx90a;gfx9-4-generic;gfx10-1-generic;gfx10-3-generic;gfx11-generic;gfx12-generic`.
- `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=auto|rocm|llvm-tools|llvm-project`, default
  `auto` for source mode.
- `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_PATH` for a ROCm or TheRock SDK root.
- `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_TOOLS_DIR` for a directory containing
  `clang`, `llvm-link`, `lld`, and `llvm-objcopy`.
- `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_BINARY`,
  `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_LINK_BINARY`,
  `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLD_BINARY`,
  `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_OBJCOPY_BINARY`, and
  `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_RESOURCE_INCLUDE` for exact per-tool
  overrides.

In source mode CMake invokes `build_tools/scripts/amdgpu_device_binaries.py`.
The `rocm` and `auto` modes search standard ROCm installs, ask `clang` or
`amdclang` for companion tool locations, check `hipcc` on `PATH`, and check
`/opt/rocm`. The `llvm-tools` mode searches explicit LLVM/IREE tool directories
and per-tool overrides. The `llvm-project` mode uses tools already configured by
the containing IREE build.
