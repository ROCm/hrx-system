# Sharing the compiler with libloomc

Bazel can link `loom-compile` and `loom-serve` against the shared compiler.
From the HRX repository root:

```bash
bazel build --loom_link_loomc \
  //loom/src/loom/tools/loom-compile \
  //loom/src/loom/tools/loom-serve
```

Build just the shared library with
`bazel build //loom/binding/c:loomc_shared_library`.

Static linkage remains the default. `--loom_link_loomc` selects shared linkage
for the public `:loomc` alias; `--loom_link_loomc=false` selects static linkage.
The underlying setting is `//loom/config:link_loomc`; an outer Bazel project
uses its repository-qualified label, for example
`--@hrx_system//loom/config:link_loomc=true`. The short flag is an alias in HRX's
`.bazelrc`, so outer projects must define their own alias if they want it.

## Artifacts and dependencies

- `//loom/binding/c:loomc_shared_library` provides the shared library file:
  `bazel-bin/loom/binding/c/libloomc.so` on Linux.
- `//loom/binding/c:loomc` is the public configured compiler dependency. Its
  alias selects the static aggregate or shared compiler according to the flag.
- `loomc_core` owns the target-independent compiler implementation and the C++
  importer when enabled. It has restricted implementation visibility.
- `loomc_static` aggregates that core, the configured provider set, the command
  compiler, and enabled VM, AMDGPU, SPIR-V, XDNA, and Wasm bindings. X86 compiler
  providers are included through the configured provider set.
- Each compiler target has one private implementation library. Public target
  binding labels alias `:loomc`; they do not make separate linkage decisions.
- HAL adapters, task helpers, execution runtimes, and device drivers remain
  separate libraries. HAL adapters obtain their compiler interfaces through
  `:loomc` instead of linking the native compiler dependencies themselves.

The existing `//loom/config/target:enable` and advanced architecture/emission
settings select the target implementations inside the aggregate. Building
`loom-compile` does not acquire HAL runtime dependencies through this aggregate.

The shared dependency forwards the aggregate's compilation context (headers,
include paths, defines, and exported module maps), while taking its link
interface from the shared-library import and runtime support. The aggregate
re-exports the component contexts directly so Bazel layering checks continue
to enforce header dependencies in both linkage modes.

The shared library exports public and internal `loomc_*` symbols, plus a small
list of native compiler entry points and type descriptors used by the separate
HAL adapters. Internal declarations in `src/` retain `LOOMC_API_PRIVATE` to
distinguish them from the supported public API. Internal interfaces require
bindings built from the same source revision and provide no stable ABI guarantee.
The ELF export list is in `cmake/loomc_exports.lds`; Bazel's macOS export flags
must be kept consistent with it.

### Tool coverage

`loom-compile` and `loom-serve` consume the configured compiler through this
shared boundary. The `iree-run-loom`, `iree-test-loom`, and `iree-benchmark-loom`
target bindings use it as well. Those runners also contain native compiler
clients in their tooling helpers, which still link some compiler code directly;
this change centralizes target composition, not every native tooling dependency.
Compiler-internal tools such as `loom-opt` retain static compiler linkage.

Bazel supplies the library through its normal solib/runfiles layout. A
redistributor must package the library alongside matching tools and arrange
its installed loader search path. The switch does not choose an installation
layout or package Python bindings.

## Build-system support

This Bazel composition has Linux and macOS linker settings. Windows shared
linkage is disabled pending an explicit public export list. Linux is the
platform used for the initial builds.

CMake uses the same generated static aggregate to select the object components
of `loomc_shared`. Tool linkage remains static under CMake. The public `:loomc`
alias uses `defaulting_select` to preserve that behavior; target aliases forward
to it unconditionally. Their `cmake-library-alias` tag tells `bazel_to_cmake` to
emit source-free interface libraries. Bazel shared-artifact rules carry
`skip-bazel_to_cmake`; CMake supplies the shared link and platform export settings.
