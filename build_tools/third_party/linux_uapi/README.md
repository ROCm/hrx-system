# Linux syscall and driver protocol headers

This package provides compile-time I/O, udmabuf, DRM, AMDGPU, KFD, and XDNA
protocol headers.
It builds no driver or libdrm library and introduces no runtime dependency.
Consumers use the `//third_party:linux_*_uapi` facades; the source identities and
checksums live in `../deps.MODULE.bazel` and the generated root CMake lock.
The Linux-sourced headers share the Linux 7.3-rc6 revision, including KFD,
io_uring, statx, and structural macros. The shared source URL keeps revision
updates together; each file retains its own content checksum.

The header sources have distinct export contracts:

- udmabuf uses Linux's public memfd-to-DMA-BUF export definitions. The separate
  target needs neither GPU driver headers nor io_uring declarations.
- DRM and AMDGPU use libdrm's exported userspace headers. Raw kernel DRM headers
  contain kernel annotations and are not directly usable as application headers.
- KFD uses Linux's public ioctl and topology definitions.
- I/O uses Linux's io_uring and statx definitions, including caller-owned
  rings and direct-I/O alignment queries. Consumers include the pinned Linux
  structural macros before io_uring declarations; fundamental types remain
  supplied by the target sysroot. This adds no liburing or libc-wrapper dependency.
  The `io_uring/zcrx.h` dependency comes from the upstream header's include
  contract; exposing its declarations does not enable zero-copy network I/O.
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
a running kernel or driver supports. The header revision is not a minimum
runtime kernel version. Native providers retain their version and capability
queries before using optional protocols. Qualification covers the real provider
tests, both build systems, and the installed shared/static consumers.
