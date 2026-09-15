# libamdf Working Contract

`libamdf` is a portable C ABI between runtime clients and native AMD GPU and
XDNA device services. Keep the boundary independent of `runtime/`, `loom/`, and
`libhrx/`; those projects may consume libamdf, but libamdf must not consume
them.

## Commit subjects

Commits editing `libamdf/` use the `[libamdf]` subject prefix, including native
GPU/XDNA providers, public headers, build metadata, tests, and this guide. This
subtree rule takes precedence over the repository's broader subsystem prefixes.
Runtime-only consumers outside `libamdf/` retain their own subsystem prefix.

## API and ownership

- Public declarations live under `include/amdf/`. Public headers are the ABI
  specification and document ownership, lifetime, threading, and cost.
- Exported functions use fixed-width C types and explicit ABI negotiation.
- Loading the shared library performs no library-owned initialization or
  dependent-library loading. Querying its API table performs no allocation,
  device discovery, library loading, system call, or other initialization.
- Provider state has explicit creation and destruction boundaries. No global
  process state, one-time initialization, or non-unloadable dependency belongs
  in this library.
- The instance owns shared native driver connections, not memory resources or
  instance-wide memory registries, tracking lists, or deferred cleanup lists.
  Each memory resource owns its allocation-specific state. Construction owns
  local rollback before publication; a cleanup error does not transfer that
  ownership to the instance. Native failure handling follows the operation's
  resource-consumption and progress contract, not a generic later-retry policy.
- Fallible constructors and initializers publish complete outputs only after
  success is guaranteed. Failure leaves caller output storage unchanged and
  creates no cleanup obligation. A non-success result publishes data only when
  the API explicitly defines that result-bearing protocol.
- Names expose retained state. `create` and fallible `initialize` operations
  are transactional. Operations that may retain native progress after failure
  act on an already-live owner, use a state-transition verb such as `attach`,
  `prepare`, `realize`, or `bootstrap`, and document the retry and teardown
  obligation without naming live state as an `out_*` parameter.
- GPU, XDNA, and platform implementation packages remain dependency-isolated.
  Common code may not acquire a device-family dependency by convenience.
- Compiler and scalar portability policy lives beside the shared primitives in
  `include/amdf/base.h`. Callers use those primitives without local platform
  branches. Native OS headers such as `windows.h` belong only in explicitly
  platform-specific implementation or test files, never in common core files
  or portable headers.
- Extension entry points and shared family code are UMD-agnostic. They operate
  on the implementation selected for the live object, without inspecting UMD
  identity or using preprocessor branches to change behavior for a UMD.
- UMD boundaries preserve the ownership and dispatch seams needed for multiple
  implementations, including native backends, mocks, and simulators such as
  rocJITsu. Static UMD selection is acceptable during bringup; runtime
  pluggability is not a prerequisite for landing. Shared APIs remain
  independent of that temporary implementation-binding choice.
- Unsupported services use callable operations with capability discovery and
  explicit unsupported results, or live in a separate absent extension. The
  public and internal contracts consistently follow the selected model.

## Build structure

- Bazel `BUILD.bazel` files are the source of truth for in-tree package
  declarations. Regenerate `CMakeLists.txt` files with
  `python build_tools/bazel_to_cmake/bazel_to_cmake.py --recursive_dir libamdf`.
- Project-specific build rules live under `build_tools/bazel/` and
  `build_tools/cmake/`. Do not add build logic to the repository root or to a
  handwritten package `CMakeLists.txt`.
- `packaging/package_smoke/CMakeLists.txt` is the one handwritten exception. It
  is an external consumer used to validate the installed package and never
  participates in the repository build graph.
- The public distribution consists of the shared library, the explicit static
  library, and headers under `include/amdf/`. Tests and examples are not part
  of the installed package.
- Shared-linked, static-linked, and runtime-loaded CTS modes execute the same
  test corpus. New API suites plug into that matrix instead of creating
  bespoke test programs.

## Verification

Exercise both Bazel and CMake for build-system changes. Installation changes
also require the standalone consumer under `packaging/package_smoke/` to build
against the installed package rather than build-tree targets.
