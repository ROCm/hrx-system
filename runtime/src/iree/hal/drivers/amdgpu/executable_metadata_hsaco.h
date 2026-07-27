// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_METADATA_HSACO_H_
#define IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_METADATA_HSACO_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/executable_metadata.h"
#include "iree/hal/drivers/amdgpu/util/hsaco_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Calculates common executable metadata storage counts for |hsaco_metadata|.
iree_status_t iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts(
    const iree_hal_amdgpu_hsaco_metadata_t* hsaco_metadata,
    iree_hal_amdgpu_executable_metadata_counts_t* out_counts);

// Populates |metadata| from decoded HSACO metadata.
//
// |metadata| must have been allocated with counts returned from
// iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts.
//
// |hsaco_metadata| must have been parsed from the original source ELF and all
// decoded string views must point within hsaco_metadata->elf_data. Retained
// views populated into |metadata| are rebased to borrow from
// |loaded_code_object_data| at the same absolute byte offsets. The loaded bytes
// are owned by the HSA executable and must remain alive for the lifetime of
// |metadata|.
iree_status_t iree_hal_amdgpu_executable_metadata_populate_from_hsaco(
    const iree_hal_amdgpu_hsaco_metadata_t* hsaco_metadata,
    iree_const_byte_span_t loaded_code_object_data,
    iree_hal_amdgpu_executable_metadata_t* metadata);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_METADATA_HSACO_H_
