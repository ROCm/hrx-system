// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_API_H_
#define IREE_HAL_DRIVERS_AMDXDNA_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct iree_hal_amdxdna_device_params {
  // Number of core tile rows/cols to use. 0 discovers the hardware defaults.
  int32_t n_core_rows;
  int32_t n_core_cols;
  // Optional native device identifier passed to the platform backend. Empty
  // selects the backend's default device (e.g. the first DRM accel node on
  // Linux KMQ, or the default NPU adapter on Windows MCDM).
  iree_string_view_t device_path;
  iree_string_view_t power_mode;
};

IREE_API_EXPORT void iree_hal_amdxdna_device_options_initialize(
    struct iree_hal_amdxdna_device_params* out_params);

IREE_API_EXPORT iree_status_t iree_hal_amdxdna_device_options_parse(
    struct iree_hal_amdxdna_device_params* params, iree_host_size_t pairs_size,
    const iree_string_pair_t* pairs);

struct iree_hal_amdxdna_driver_options {
  // Default device options used when creating devices from this driver.
  struct iree_hal_amdxdna_device_params default_device_params;
};

IREE_API_EXPORT void iree_hal_amdxdna_driver_options_initialize(
    struct iree_hal_amdxdna_driver_options* out_options);

// The provided `identifier` will be used by programs to distinguish the device
// type from other HAL implementations. If compiling programs with the IREE
// compiler this must match the value used by IREE::HAL::TargetDevice.
//
// `out_driver` must be released by the caller (see iree_hal_driver_release).
IREE_API_EXPORT iree_status_t iree_hal_amdxdna_driver_create(
    iree_string_view_t identifier,
    const struct iree_hal_amdxdna_driver_options* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver);

IREE_API_EXPORT iree_status_t iree_hal_amdxdna_device_create(
    iree_string_view_t identifier,
    const struct iree_hal_amdxdna_device_params* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_API_H_
