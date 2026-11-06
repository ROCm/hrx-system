// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_IMAGE_HSACO_H_
#define IREE_HAL_DRIVER_AMD_GPU_IMAGE_HSACO_H_

#include "iree/hal/driver/amd/gpu/image/image.h"

#ifdef __cplusplus
extern "C" {
#endif

// Admits a linked AMDHSA ELF64 little-endian code object (V3 through V6).
//
// Uses program headers and dynamic tables; section headers and debug sections
// are not required. Structural metadata is bounded and decoded once. Payloads
// stay in source_sequence until loaded, without a whole-image contiguous copy.
// The source is retained on success. Failure leaves out_image zeroed.
//
// The image records target, resource, import, dependency and runtime-service
// requirements. Admission does not assert that a particular device/runtime can
// execute them. Native code is not sandboxed or instruction-validated.
iree_status_t iree_hal_amd_gpu_hsaco_image_initialize(
    iree_byte_sequence_t* source_sequence, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_image_t* out_image);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_IMAGE_HSACO_H_
