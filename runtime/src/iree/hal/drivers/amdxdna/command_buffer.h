// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_AMDXDNA_COMMAND_BUFFER_H_

#include <stdbool.h>

#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdxdna/native.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates an amdxdna-owned recorded command buffer. Recording only captures HAL
// commands and retains referenced resources; execution happens later on the
// amdxdna queue worker by applying records to a live direct command buffer.
iree_status_t iree_hal_amdxdna_command_buffer_create(
    iree_hal_allocator_t* device_allocator,
    const iree_hal_amdxdna_native_c_device_caps_t* native_caps,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

bool iree_hal_amdxdna_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

iree_status_t iree_hal_amdxdna_command_buffer_apply(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_COMMAND_BUFFER_H_
