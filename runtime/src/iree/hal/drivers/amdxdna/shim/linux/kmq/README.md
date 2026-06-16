# amdxdna Linux KMQ shim — provenance & licensing

This directory holds the vendored, exception-free user-space shim for the AMD
XDNA NPU kernel driver (`amdxdna.ko`) on Linux via the KMQ (kernel-managed
queue) interface. It is **vendored code, not first-party**, and it contains
files under **three different licenses** — this README is the authoritative
description; the per-file SPDX headers are the authoritative license of record.

## ABI / source roots

```
amd/xdna-driver   src/shim/...      — user-space XRT SHIM for XDNA (rewritten here)
Xilinx/XRT        ert.h             — ERT command-packet ABI (verbatim)
Linux kernel UAPI amdxdna_accel.h   — DRM ioctl ABI for amdxdna (verbatim)
```

Their copyright is carried in the per-file SPDX headers; the roots are recorded
here for completeness and for resync.

## Verbatim ABI vs. rewritten

Two headers are **verbatim ABI contracts** — they must byte-match the
kernel/firmware ABI, so they are not rewritable and retain their upstream
licenses unchanged:

| File | Kind | License | © |
|------|------|---------|---|
| `amdxdna_accel.h` | verbatim kernel DRM UAPI | `GPL-2.0 WITH Linux-syscall-note` | AMD |
| `ert.h`           | verbatim XRT ERT ABI     | dual `Apache-2.0 / GPL-2.0`        | Xilinx |

The `Linux-syscall-note` exception on `amdxdna_accel.h` explicitly permits
non-GPL user-space (this shim, and the Apache-licensed HAL above it) to include
the header and use the ioctl interface. `ert.h` is dual-licensed; we use it
under Apache-2.0.

Everything else (`bo`, `device`, `fence`, `hwctx`, `hwq`, `kernel`,
`shim_debug`, `bo_flags`) is a **rewrite** of XRT's exception-based shim into an
errno/return-code style with no `xrt_core` dependency. These are derivative
works of the upstream shim and carry the per-file SPDX headers inherited from
the vendored sources (a mix of `Apache-2.0` (© AMD) and
`Apache-2.0 WITH LLVM-exception` (© The IREE Authors); see each file).

## Maintenance notes

- **Per-file SPDX headers are authoritative. Do not strip them**, and do not
  apply a single tree-wide license claim over this directory — it contains files
  under three different licenses.
- When resyncing, keep the ABI roots above explicit and record the source
  snapshot used for the import.
- If you ever need to drop the vendored copies of the two ABI headers, source
  them from the system instead (`<uapi/drm/amdxdna_accel.h>` from kernel-headers,
  `ert.h` from an XRT install) rather than relicensing them. Note neither is on a
  default include path on all systems, which is why they are vendored here.
