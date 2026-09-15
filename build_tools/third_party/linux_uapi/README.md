# Linux driver protocol headers

This package provides compile-time DRM, AMDGPU, KFD, and XDNA protocol headers.
It builds no driver or libdrm library and introduces no runtime dependency.
Consumers use the `//third_party:linux_*_uapi` facades; the source identities and
checksums live in `../deps.MODULE.bazel` and the generated root CMake lock.

The header sources have distinct export contracts:

- DRM and AMDGPU use libdrm's exported userspace headers. Raw kernel DRM headers
  contain kernel annotations and are not directly usable as application headers.
- KFD uses Linux's public ioctl and topology definitions.
- XDNA uses the driver's public `include/uapi/drm/amdxdna_accel.h`, not its
  private `drm_local` protocol. Its Linux macro helpers are pinned alongside
  KFD so an older platform sysroot can compile the selected public definitions.

The target sysroot still supplies fundamental Linux types and ioctl encoding.
The package exposes both `drm/` and `libdrm/` include spellings because upstream
driver headers use both. GPU-only header targets are separate from the shared
DRM and XDNA targets; disabled providers do not fetch KFD inputs.

`linux_uapi.cmake` assembles the same header surface as Bazel. Project admission
selects the required targets. Embedding builds that disallow pinned downloads
provide the corresponding `iree::third_party::linux_*_uapi` interface targets.
These are private build inputs and are absent from libamdf's exported package
dependencies, including the static library's interface.

Updating a snapshot changes which protocols can be compiled, not which services
a running driver supports. Native providers retain their version and capability
queries before using optional protocols. Qualification covers the real provider
tests, both build systems, and the installed shared/static consumers.
