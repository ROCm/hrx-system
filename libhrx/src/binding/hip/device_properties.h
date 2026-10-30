// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_HIP_DEVICE_PROPERTIES_H_
#define LIBHRX_SRC_BINDING_HIP_DEVICE_PROPERTIES_H_

#include <stdbool.h>

#include "iree/hal/device_spec.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Parses the numeric architecture from a gfx target identifier. Feature
// suffixes beginning with ':' are accepted. Returns false without modifying
// |out_architecture| when the identifier is malformed or out of range.
bool iree_hip_parse_gcn_arch_name(const char* name, int* out_architecture);

// Translates a provider UUID to HIP's 16-byte representation. AMD native and
// HSA drivers store their 64-bit identifier in the first eight HAL bytes,
// most-significant byte first; HIP uses its sixteen lowercase ASCII hex digits.
// Other providers retain their complete opaque 128-bit UUID representation.
void iree_hip_format_device_uuid(iree_string_view_t driver_id,
                                 const iree_hal_uuid_t* source,
                                 uint8_t out_bytes[16]);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LIBHRX_SRC_BINDING_HIP_DEVICE_PROPERTIES_H_
