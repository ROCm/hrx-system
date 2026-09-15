// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_ALLOCATOR_H_
#define AMDF_SRC_ALLOCATOR_H_

#include <stddef.h>

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns the built-in cross-module-safe system allocator.
amdf_allocator_t amdf_allocator_system(void);

// Validates `requested_allocator` and resolves the built-in system allocator.
// Failure leaves `out_allocator` unchanged.
amdf_status_t amdf_allocator_resolve(
    const amdf_allocator_t* requested_allocator,
    amdf_allocator_t* out_allocator);

// Allocates one uninitialized aligned byte range.
// Failure leaves `out_pointer` unchanged.
amdf_status_t amdf_malloc(amdf_allocator_t allocator, size_t byte_length,
                          size_t minimum_alignment, void** out_pointer);

// Allocates and zeroes one aligned byte range.
// Failure leaves `out_pointer` unchanged.
amdf_status_t amdf_calloc(amdf_allocator_t allocator, size_t byte_length,
                          size_t minimum_alignment, void** out_pointer);

// Allocates and zeroes an overflow-checked aligned array.
// Failure leaves `out_pointer` unchanged.
amdf_status_t amdf_calloc_array(amdf_allocator_t allocator, size_t count,
                                size_t element_size, size_t minimum_alignment,
                                void** out_pointer);

// Allocates and zeroes an overflow-checked header plus trailing byte range.
// Failure leaves `out_pointer` unchanged.
amdf_status_t amdf_calloc_with_trailing(amdf_allocator_t allocator,
                                        size_t header_byte_length,
                                        size_t trailing_byte_length,
                                        size_t minimum_alignment,
                                        void** out_pointer);

// Resizes an aligned allocation while preserving the old allocation on
// failure. `old_byte_length` is zero exactly when `*inout_pointer` is NULL.
amdf_status_t amdf_realloc(amdf_allocator_t allocator, size_t old_byte_length,
                           size_t new_byte_length, size_t minimum_alignment,
                           void** inout_pointer);

// Frees a pointer returned by the same allocator. NULL is a no-op.
void amdf_free(amdf_allocator_t allocator, void* pointer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_ALLOCATOR_H_
