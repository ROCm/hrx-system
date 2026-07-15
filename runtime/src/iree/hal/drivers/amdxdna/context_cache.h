// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_CONTEXT_CACHE_H_
#define IREE_HAL_DRIVERS_AMDXDNA_CONTEXT_CACHE_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdxdna/native.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdxdna_device iree_hal_amdxdna_device;
typedef struct iree_hal_amdxdna_device_context_cache_t
    iree_hal_amdxdna_device_context_cache_t;

iree_hal_amdxdna_device_context_cache_t*
iree_hal_amdxdna_device_context_cache_create(iree_allocator_t host_allocator);

void iree_hal_amdxdna_device_context_cache_destroy(
    iree_hal_amdxdna_device_context_cache_t* context_cache);

void iree_hal_amdxdna_device_context_cache_clear(
    iree_hal_amdxdna_device_context_cache_t* context_cache);

// Returns a shared native context for the (non-empty) control-packet bootstrap
// `pdi`/`xclbin` and CU/export name, creating and caching it on first use.
iree_status_t iree_hal_amdxdna_device_get_or_create_context(
    iree_hal_amdxdna_device* device, iree_const_byte_span_t pdi,
    iree_const_byte_span_t xclbin, iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_context_ref_t** out_context_ref);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_CONTEXT_CACHE_H_
