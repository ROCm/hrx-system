// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_AMDXDNA_EXECUTABLE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdxdna/program_image.h"

struct iree_hal_amdxdna_executable;
struct iree_hal_amdxdna_device;
struct iree_hal_amdxdna_native_device_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

iree_string_view_t iree_hal_amdxdna_executable_format();

bool iree_hal_amdxdna_executable_format_supported(
    iree_string_view_t executable_format);

// Records the HAL device that owns this executable's buffer and context
// caches. Call it after a successful create and before any failure path that
// releases the executable.
void iree_hal_amdxdna_executable_bind_device(
    iree_hal_executable_t* executable, struct iree_hal_amdxdna_device* device);

// `out_executable` must be released by the caller (see
// iree_hal_executable_release). Context program images are interned in
// |image_store| when provided so executables with identical images share one
// host copy; a NULL store gives the executable private copies.
iree_status_t iree_hal_amdxdna_native_executable_create(
    struct iree_hal_amdxdna_native_device_t* native_device,
    iree_hal_amdxdna_program_image_store_t* image_store,
    const iree_hal_executable_load_params_t* load_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

#define IREE_HAL_AMDXDNA_EXECUTABLE_FOOTPRINT_MAX_IMAGES 4

// Host memory retained by an executable.
typedef struct iree_hal_amdxdna_executable_footprint_t {
  // Bytes owned only by the executable: descriptor storage, control code,
  // patch and relocation tables, and names.
  iree_host_size_t exclusive_host_bytes;
  // Distinct program images referenced by the executable. Their bytes are not
  // included in |exclusive_host_bytes| because other executables may share
  // them.
  iree_host_size_t image_count;
  const iree_hal_amdxdna_program_image_t*
      images[IREE_HAL_AMDXDNA_EXECUTABLE_FOOTPRINT_MAX_IMAGES];
} iree_hal_amdxdna_executable_footprint_t;

// Computes the host footprint of |executable|. Returns false when it
// references more than IREE_HAL_AMDXDNA_EXECUTABLE_FOOTPRINT_MAX_IMAGES
// distinct images.
bool iree_hal_amdxdna_executable_query_footprint(
    iree_hal_executable_t* executable,
    iree_hal_amdxdna_executable_footprint_t* out_footprint);

// Returns the control code of an executable with exactly one entry point that
// has exactly one run, or an empty span for any other shape.
iree_const_byte_span_t iree_hal_amdxdna_executable_single_run_control_code(
    iree_hal_executable_t* executable);

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
