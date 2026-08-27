// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_LIBHRX_VMM_SLAB_PROVIDER_H_
#define HRX_LIBHRX_VMM_SLAB_PROVIDER_H_

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

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_LIBHRX_VMM_SLAB_PROVIDER_H_
