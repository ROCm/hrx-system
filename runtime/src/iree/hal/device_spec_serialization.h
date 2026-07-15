// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DEVICE_SPEC_SERIALIZATION_H_
#define IREE_HAL_DEVICE_SPEC_SERIALIZATION_H_

#include "iree/hal/device_spec.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Computes a stable non-cryptographic fingerprint of the canonical device spec
// encoding.
//
// This traverses the spec without allocating the serialized image.
iree_status_t iree_hal_device_spec_compute_fingerprint(
    const iree_hal_device_spec_t* spec, uint64_t* out_fingerprint);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DEVICE_SPEC_SERIALIZATION_H_
