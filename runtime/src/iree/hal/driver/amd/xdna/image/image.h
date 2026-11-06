// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable native XDNA executable descriptions, without device resources.

#ifndef IREE_HAL_DRIVER_AMD_XDNA_IMAGE_IMAGE_H_
#define IREE_HAL_DRIVER_AMD_XDNA_IMAGE_IMAGE_H_

#include "iree/base/byte_sequence.h"
#include "iree/hal/driver/amd/xdna/image/aie2p/target.h"
#include "iree/hal/driver/amd/xdna/image/directory.h"
#include "iree/hal/driver/amd/xdna/image/tables.h"

#ifdef __cplusplus
extern "C" {
#endif

// Owns compact metadata and the structural directory retaining image source.
// The image is immutable and can be queried concurrently. Loaded native backing
// and all indirectly referenced resources remain owned by the caller.
typedef struct iree_hal_amd_xdna_image_t iree_hal_amd_xdna_image_t;

// Admits an image's structure and target requirements, retaining its source.
// Native instruction bytes are opaque executable code, not sandboxed input.
iree_status_t iree_hal_amd_xdna_image_allocate(
    iree_byte_sequence_t* source_sequence,
    const iree_hal_amd_xdna_aie2p_target_t* target,
    iree_allocator_t host_allocator, iree_hal_amd_xdna_image_t** out_image);

// Releases the description and retained source; performs no device operation.
void iree_hal_amd_xdna_image_free(iree_hal_amd_xdna_image_t* image);

// Borrows validated tables for the lifetime of the image.
const iree_hal_amd_xdna_image_tables_t* iree_hal_amd_xdna_image_tables(
    const iree_hal_amd_xdna_image_t* image);

// Borrows the structural load directory for the lifetime of the image.
const iree_hal_amd_xdna_image_directory_t* iree_hal_amd_xdna_image_directory(
    const iree_hal_amd_xdna_image_t* image);

// Resolves an exact public export name to its dense entry ordinal.
iree_status_t iree_hal_amd_xdna_image_find_entry(
    const iree_hal_amd_xdna_image_t* image, iree_string_view_t name,
    uint32_t* out_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_XDNA_IMAGE_IMAGE_H_
