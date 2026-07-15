# amdxdna NPU shim (vendored)

This directory tree contains a minimal, exception-free user-space shim for the
AMD XDNA NPU kernel driver (`amdxdna.ko`). The amdxdna HAL driver
(`../native_linux_kmq.cc` and the HAL layer above it) talks to the device
exclusively through this shim. The shim is **self-contained**: it links no
external user-space runtime library and talks to the kernel driver directly
through its DRM ioctl ABI. It is a small static library built from the sources
here so the HAL driver can link it directly.

The shim is **vendored code, not first-party**, and the vendored sources live
under the platform subdirectory:

- [`linux/kmq/`](linux/kmq/) — the Linux KMQ implementation. See
  [`linux/kmq/README.md`](linux/kmq/README.md) for the authoritative provenance,
  the verbatim-ABI-vs-rewritten breakdown, and the per-file license map (this
  subtree contains files under three different licenses, including a GPL UAPI
  header and a dual-licensed ABI header).

The `linux/` and `linux/kmq/` `CMakeLists.txt` license headers cover those build
files only; per-file SPDX headers on the sources are authoritative.
