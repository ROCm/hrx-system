// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_LIBHRX_VMM_SLAB_PROVIDER_H_
#define HRX_LIBHRX_VMM_SLAB_PROVIDER_H_

#include "hrx_runtime.h"
#include "iree/hal/allocator.h"
#include "iree/hal/memory/slab_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a slab provider that backs each slab with one virtual reservation and
// one mapped physical allocation. Materialized buffers are subspans of that
// reservation, so every pool allocation has a stable device-visible address.
//
// |physical_buffer_params| must require device-local memory and may not request
// host access. The provider retains |allocator| and releases it at destruction.
// Virtual memory must be supported by |allocator|.
iree_status_t hrx_vmm_slab_provider_create(
    iree_hal_allocator_t* allocator,
    iree_hal_buffer_params_t physical_buffer_params,
    iree_allocator_t host_allocator, iree_hal_slab_provider_t** out_provider);

// Grants the agents represented by |accessor_allocator| access to all current
// and future slabs. |device_ordinal| identifies the persistent policy slot.
// This function is only valid for providers created above.
iree_status_t hrx_vmm_slab_provider_set_access(
    iree_hal_slab_provider_t* provider, iree_host_size_t device_ordinal,
    hrx_allocator_t accessor_allocator,
    iree_hal_memory_protection_t protection);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_LIBHRX_VMM_SLAB_PROVIDER_H_
