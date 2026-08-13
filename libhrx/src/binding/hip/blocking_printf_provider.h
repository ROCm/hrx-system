// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_HIP_BLOCKING_PRINTF_PROVIDER_H_
#define HRX_BINDING_HIP_BLOCKING_PRINTF_PROVIDER_H_

#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// HIP blocking printf provider configuration.
typedef struct iree_hip_blocking_printf_provider_t {
  // Host allocator used for physical-device provider contexts.
  iree_allocator_t host_allocator;

  // Custom HRX event sink, or an invalid sink for HIP's default stdio output.
  iree_hal_device_event_sink_t event_sink;

  // HAL device-creation extension referencing this provider.
  iree_hal_hostcall_provider_extension_t device_extension;
} iree_hip_blocking_printf_provider_t;

// Initializes one provider configuration that remains immutable while active.
void iree_hip_blocking_printf_provider_initialize(
    iree_hal_device_event_sink_t event_sink, iree_allocator_t host_allocator,
    iree_hip_blocking_printf_provider_t* out_provider);

// Returns the HAL device-creation extension owned by |provider|.
const iree_hal_device_create_params_extension_t*
iree_hip_blocking_printf_provider_device_extension(
    const iree_hip_blocking_printf_provider_t* provider);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_HIP_BLOCKING_PRINTF_PROVIDER_H_
