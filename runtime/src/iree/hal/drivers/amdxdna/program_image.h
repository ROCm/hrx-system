// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_PROGRAM_IMAGE_H_
#define IREE_HAL_DRIVERS_AMDXDNA_PROGRAM_IMAGE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Immutable host copy of a context program image (xclbin wrapper or PDI).
//
// Shape-specialized producers create many executables whose control code
// differs but whose context images are byte-identical. Executables reference
// a shared image instead of carrying a private copy so host memory scales with
// the number of distinct images rather than the number of live executables.
typedef struct iree_hal_amdxdna_program_image_t
    iree_hal_amdxdna_program_image_t;

// Exact-content intern table of program images. Reference counted: the owning
// device and every image acquired from the store hold a reference, so images
// may outlive the device that loaded them.
typedef struct iree_hal_amdxdna_program_image_store_t
    iree_hal_amdxdna_program_image_store_t;

iree_status_t iree_hal_amdxdna_program_image_store_create(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_program_image_store_t** out_store);

void iree_hal_amdxdna_program_image_store_retain(
    iree_hal_amdxdna_program_image_store_t* store);

void iree_hal_amdxdna_program_image_store_release(
    iree_hal_amdxdna_program_image_store_t* store);

typedef struct iree_hal_amdxdna_program_image_store_stats_t {
  // Number of distinct live images.
  iree_host_size_t image_count;
  // Sum of the content lengths of all live images.
  iree_host_size_t content_bytes;
} iree_hal_amdxdna_program_image_store_stats_t;

void iree_hal_amdxdna_program_image_store_query_stats(
    iree_hal_amdxdna_program_image_store_t* store,
    iree_hal_amdxdna_program_image_store_stats_t* out_stats);

// Returns a retained image whose contents equal |contents|. With a non-NULL
// |store| an existing live image with identical bytes is shared and new images
// are allocated from the store's allocator; with a NULL |store| a private image
// is allocated from |host_allocator|. Empty |contents| yield a NULL image.
iree_status_t iree_hal_amdxdna_program_image_acquire(
    iree_hal_amdxdna_program_image_store_t* store,
    iree_const_byte_span_t contents, iree_allocator_t host_allocator,
    iree_hal_amdxdna_program_image_t** out_image);

void iree_hal_amdxdna_program_image_retain(
    iree_hal_amdxdna_program_image_t* image);

void iree_hal_amdxdna_program_image_release(
    iree_hal_amdxdna_program_image_t* image);

// Returns the image bytes, or an empty span for a NULL image.
iree_const_byte_span_t iree_hal_amdxdna_program_image_contents(
    const iree_hal_amdxdna_program_image_t* image);

// Host bytes owned by |image| including its header.
iree_host_size_t iree_hal_amdxdna_program_image_host_bytes(
    const iree_hal_amdxdna_program_image_t* image);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_PROGRAM_IMAGE_H_
