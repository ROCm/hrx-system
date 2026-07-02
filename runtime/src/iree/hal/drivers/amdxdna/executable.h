// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_AMDXDNA_EXECUTABLE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

struct iree_hal_amdxdna_executable;
struct iree_hal_amdxdna_device;
struct iree_hal_amdxdna_native_device_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

iree_string_view_t iree_hal_amdxdna_executable_format();

bool iree_hal_amdxdna_executable_format_supported(
    iree_string_view_t executable_format);

// `out_executable` must be released by the caller (see
// iree_hal_executable_release).
iree_status_t iree_hal_amdxdna_native_executable_create(
    struct iree_hal_amdxdna_native_device_t* native_device,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

// Preloads native contexts for self-contained entry points. This is executable
// preparation work, not queue execution work: it may create native hardware
// contexts and must therefore never run from device_queue_* entry points.
iree_status_t iree_hal_amdxdna_executable_preload_contexts(
    struct iree_hal_amdxdna_device* device,
    iree_hal_executable_t* base_executable);

iree_status_t iree_hal_amdxdna_native_executable_infer_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size);

struct iree_hal_amdxdna_executable* iree_hal_amdxdna_executable_cast(
    iree_hal_executable_t* base_executable);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_EXECUTABLE_H_
