// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_DRIVER_H_
#define IREE_HAL_DRIVERS_AMD_DRIVER_H_

#include "amdf/base.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates an independent AMD driver and native libamdf fabric. Devices retain
// that fabric independently of the driver and registry. Discovery creates no
// schedulable device state. GPU scheme filtering uses query-only endpoint
// identity and queue-family descriptors; activation occurs only when the
// selected device is created.
// |native_lifetime| is passed unchanged to libamdf. PROCESS supports Linux KFD
// host registration and requires coordination with foreign KFD VM mutations.
// Its primary VM can survive fabric destruction and prevent fresh acquisition;
// retain one driver/fabric for all GPU devices using that process context.
// INSTANCE bounds acquired native state to this fabric's lifetime; unsupported
// native services fail explicitly. It does not promise isolated GPU addresses.
//
// The execution schemes are `xdna`, `rdna`, `cdna`, and `amdgpu2`. Each has an
// independent ordinal namespace: NPUs, RDNA GPUs, CDNA GPUs, and all supported GPUs,
// respectively. GPU discovery requires a usable native compute family;
// architecture alone is insufficient. A bare scheme selects visible ordinal
// zero; `rdna:N` and `rdna://N` index the same visible list.
//
// Driver creation captures HAL_VISIBLE_XDNA_DEVICES, HAL_VISIBLE_RDNA_DEVICES,
// HAL_VISIBLE_CDNA_DEVICES, and HAL_VISIBLE_AMDGPU2_DEVICES. Each list selects
// and orders ordinals from that scheme's unmasked supported endpoints. Unset
// exposes all; empty or whitespace-only exposes none. Duplicate, malformed, and
// out-of-range ordinals fail driver creation with INVALID_ARGUMENT. Variables
// are independent: filtering amdgpu2 does not filter rdna. Changing environment
// values afterward does not change this driver or its native topology.
//
// Enumeration emits `id/<32 hexadecimal digits>` paths naming
// exact native endpoint identities. These identities may become stale after
// native device removal, reset, or reboot; they are not persistent machine
// identifiers. Identity selection still requires the endpoint to match the
// requested scheme's visible set. An unsupported or hidden identity fails with
// UNAVAILABLE before native activation. Opaque numeric device IDs returned by
// enumeration name native endpoints without a scheme and retain their meaning
// across aliases; they are not the public URI ordinals. Visibility is device
// selection policy, not access control.
// The `columns` XDNA parameter selects context width (default 1). GPU options
// are `queue_count` (default 1), `queue_family` (an exact native family
// ordinal; otherwise prefer AQL on CDNA and PM4 on RDNA), and `host_numa_node`
// (an explicit CPU NUMA node; otherwise placement is unspecified).
iree_status_t iree_hal_amd_driver_create(amdf_native_lifetime_t native_lifetime,
                                         iree_allocator_t host_allocator,
                                         iree_hal_driver_t** out_driver);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_DRIVER_H_
