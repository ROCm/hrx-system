// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// External admission of native storage, indexed uses and invocation ranges.

#ifndef IREE_HAL_DRIVER_AMD_XDNA_IMAGE_VALIDATION_H_
#define IREE_HAL_DRIVER_AMD_XDNA_IMAGE_VALIDATION_H_

#include "iree/hal/driver/amd/xdna/image/aie2p/target.h"
#include "iree/hal/driver/amd/xdna/image/directory.h"
#include "iree/hal/driver/amd/xdna/image/tables.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validates target requirements and all relationships in bounded metadata.
// Only the external image admission path calls this; consumers of an admitted
// image use its indexed relationships directly.
iree_status_t iree_hal_amd_xdna_image_validate(
    const iree_hal_amd_xdna_image_tables_t* tables,
    const iree_hal_amd_xdna_image_directory_t* directory,
    const iree_hal_amd_xdna_aie2p_target_t* target);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_XDNA_IMAGE_VALIDATION_H_
