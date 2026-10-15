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
// identity; activation occurs only when the selected device is created.
// |native_lifetime| is passed unchanged to libamdf. PROCESS supports Linux KFD
// host registration and requires coordination with foreign KFD VM mutations.
// Its primary VM can survive fabric destruction and prevent fresh acquisition;
// retain one driver/fabric for all GPU devices using that process context.
// INSTANCE bounds acquired native state to this fabric's lifetime; unsupported
// native services fail explicitly. It does not promise isolated GPU addresses.
//
// The execution schemes are `xdna` and `rdna`. A bare scheme selects its first
// endpoint; `rdna:N` and `rdna://N` index the same architecture-local list.
// Enumeration emits `id/<32 hexadecimal digits>` paths naming exact native
// endpoint identities. These identities may become stale after native device
// removal, reset, or reboot; they are not persistent machine identifiers.
// Identity selection still requires the endpoint to match the requested family.
// The `columns` XDNA parameter selects context width (default 1). GPU options
// are `queue_count` (default 1), `queue_family` (an exact native family
// ordinal; otherwise the first PM4 USER host-producer family), and
// `host_numa_node` (an explicit CPU NUMA node; otherwise placement is
// unspecified).
iree_status_t iree_hal_amd_driver_create(amdf_native_lifetime_t native_lifetime,
                                         iree_allocator_t host_allocator,
                                         iree_hal_driver_t** out_driver);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_DRIVER_H_
