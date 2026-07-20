// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_CPU_DEVICE_SPEC_H_
#define IREE_HAL_LOCAL_CPU_DEVICE_SPEC_H_

#include <stdint.h>

#include "iree/base/cpu_data.h"
#include "iree/hal/device_spec.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define IREE_HAL_CPU_DEVICE_SPEC_SCHEMA_ID "iree.hal.cpu.device"
#define IREE_HAL_CPU_DEVICE_SPEC_SCHEMA_VERSION 1u

// Stable CPU device spec flags.
typedef uint32_t iree_hal_cpu_device_spec_flags_t;
typedef enum iree_hal_cpu_device_spec_flag_bits_e {
  // No CPU device spec flags are present.
  IREE_HAL_CPU_DEVICE_SPEC_FLAG_NONE = 0u,
} iree_hal_cpu_device_spec_flag_bits_t;

// Pointer-free CPU facts preserved in a device spec facet.
typedef struct iree_hal_cpu_device_spec_t {
  // Architecture and fixed-width CPU capability fields.
  iree_cpu_data_t cpu_data;
  // Stable CPU device spec flags.
  iree_hal_cpu_device_spec_flags_t flags;
} iree_hal_cpu_device_spec_t;

// Returns the canonical byte size of an encoded CPU device spec payload.
IREE_API_EXPORT iree_host_size_t iree_hal_cpu_device_spec_payload_size(void);

// Encodes |spec| into the canonical fixed-width little-endian |payload|.
IREE_API_EXPORT iree_status_t iree_hal_cpu_device_spec_encode(
    const iree_hal_cpu_device_spec_t* spec, iree_byte_span_t payload);

// Decodes the fixed-width little-endian |payload| into |out_spec| after
// validating its schema envelope.
IREE_API_EXPORT iree_status_t iree_hal_cpu_device_spec_decode(
    iree_const_byte_span_t payload, iree_hal_cpu_device_spec_t* out_spec);

// Finds the CPU device spec facet in |device_spec| or NULL.
IREE_API_EXPORT const iree_hal_device_spec_facet_t*
iree_hal_cpu_device_spec_find_facet(const iree_hal_device_spec_t* device_spec);

// Decodes a CPU device spec |facet| into |out_spec|.
IREE_API_EXPORT iree_status_t
iree_hal_cpu_device_spec_decode_facet(const iree_hal_device_spec_facet_t* facet,
                                      iree_hal_cpu_device_spec_t* out_spec);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_CPU_DEVICE_SPEC_H_
