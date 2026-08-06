// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_STREAMING_MANAGED_GLOBAL_H_
#define IREE_HAL_STREAMING_MANAGED_GLOBAL_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_streaming_context_symbol_map_t
    iree_hal_streaming_context_symbol_map_t;
typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_hal_streaming_symbol_t iree_hal_streaming_symbol_t;

// Allocates the NUL-terminated device storage name corresponding to
// |pointer_name|. The caller owns the result with |host_allocator|.
iree_status_t iree_hal_streaming_managed_global_storage_name(
    iree_allocator_t host_allocator, iree_string_view_t pointer_name,
    char** out_storage_name);

// Initializes |storage_symbol| from the registered host storage and publishes
// its device address through |pointer_symbol|.
iree_status_t iree_hal_streaming_managed_global_initialize(
    iree_hal_streaming_context_t* context, iree_string_view_t pointer_name,
    const void* initial_value, iree_host_size_t initial_value_size,
    iree_hal_streaming_symbol_t* pointer_symbol,
    iree_hal_streaming_symbol_t* storage_symbol);

// Refreshes registered managed host storage after queued device work has
// completed.
iree_status_t iree_hal_streaming_context_symbol_map_synchronize_managed_data(
    iree_hal_streaming_context_symbol_map_t* map);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_STREAMING_MANAGED_GLOBAL_H_
