// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_BUFFER_VIEW_UTIL_H_
#define IREE_HAL_BUFFER_VIEW_UTIL_H_

#include <stdint.h>
#include <stdio.h>

#include "iree/base/api.h"
#include "iree/hal/allocator.h"
#include "iree/hal/buffer_view.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Buffer view math
//===----------------------------------------------------------------------===//

// Calculates the allocation size of a buffer view.
IREE_API_EXPORT iree_status_t iree_hal_buffer_compute_view_size(
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type,
    iree_device_size_t* out_allocation_size);

// Calculates a byte offset into a buffer at the given indices.
// Only works with densely-packed representations.
IREE_API_EXPORT iree_status_t iree_hal_buffer_compute_view_offset(
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type, iree_host_size_t indices_count,
    const iree_hal_dim_t* indices, iree_device_size_t* out_offset);

// Calculates a byte range into a buffer of the given contiguous range.
// Only works with densely-packed representations.
IREE_API_EXPORT iree_status_t iree_hal_buffer_compute_view_range(
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type, iree_host_size_t indices_count,
    const iree_hal_dim_t* start_indices, iree_host_size_t lengths_count,
    const iree_hal_dim_t* lengths, iree_device_size_t* out_start_offset,
    iree_device_size_t* out_length);

//===----------------------------------------------------------------------===//
// Buffer view allocation and generation
//===----------------------------------------------------------------------===//

// Allocates an uninitialized buffer from |allocator| using |buffer_params| and
// wraps it in a buffer view.
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_allocate(
    iree_hal_allocator_t* allocator, iree_hal_buffer_params_t buffer_params,
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type,
    iree_hal_buffer_view_t** out_buffer_view);

// Allocates an uninitialized buffer view with the same shape, element type, and
// encoding as |source_buffer_view|.
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_allocate_like(
    iree_hal_allocator_t* allocator, iree_hal_buffer_params_t buffer_params,
    const iree_hal_buffer_view_t* source_buffer_view,
    iree_hal_buffer_view_t** out_buffer_view);

// Produces the logical contents of a newly allocated buffer view. |contents|
// is valid only for the duration of the call.
typedef iree_status_t(IREE_API_PTR* iree_hal_buffer_view_generator_fn_t)(
    void* user_data, iree_byte_span_t contents);

// Allocates a host-visible buffer view and produces its initial contents with
// |generator|. After successful allocation and mapping the generator is called
// exactly once with the logical view byte span; allocator padding is not
// exposed.
//
// IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
// IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
// IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED, and
// IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE are added to
// |buffer_params|. Non-coherent host writes are flushed before returning.
// No device queue operations or staging transfers are performed.
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_generate(
    iree_hal_allocator_t* allocator, iree_hal_buffer_params_t buffer_params,
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type,
    iree_hal_buffer_view_generator_fn_t generator, void* user_data,
    iree_hal_buffer_view_t** out_buffer_view);

//===----------------------------------------------------------------------===//
// Buffer view parsing and printing
//===----------------------------------------------------------------------===//

// Parses a serialized set of buffer elements in the canonical tensor format
// (the same as produced by iree_hal_buffer_view_format). The underlying buffer
// is allocated with |allocator| and |buffer_params| as by
// iree_hal_buffer_view_generate.
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_parse(
    iree_string_view_t value, iree_hal_allocator_t* allocator,
    iree_hal_buffer_params_t buffer_params,
    iree_hal_buffer_view_t** out_buffer_view);

// TODO(#5413): enum for printing mode (include shape, precision).

// Converts the already host-accessible logical |contents| of |buffer_view| into
// a fully-specified string-form format like `2x2xi16=[1 2][3 4]`.
// |contents| must have exactly iree_hal_buffer_view_byte_length(buffer_view)
// bytes.
//
// |max_element_count| can be used to limit the total number of elements printed
// when the count may be large. Elided elements will be replaced with `...`.
//
// |buffer_capacity| defines the size of |buffer| in bytes and
// |out_buffer_length| will return the string length in characters. Returns
// IREE_STATUS_OUT_OF_RANGE if the buffer capacity is insufficient to hold the
// formatted elements and |out_buffer_length| will contain the required size.
//
// Follows the standard API string formatting rules. See iree/base/api.h.
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_format_contents(
    const iree_hal_buffer_view_t* buffer_view, iree_const_byte_span_t contents,
    iree_host_size_t max_element_count, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length);

// Maps and converts buffer view elements into a fully-specified string-form
// format like `2x2xi16=[1 2][3 4]`.
//
// The logical buffer view contents must be ready for host access and the
// underlying buffer must support scoped read mappings. Non-coherent memory is
// invalidated before it is read. Use iree_hal_buffer_view_format_contents when
// the contents were downloaded from device-only memory.
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_format(
    const iree_hal_buffer_view_t* buffer_view,
    iree_host_size_t max_element_count, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length);

// Maps and prints buffer view elements into a fully-specified string-form
// format like `2x2xi16=[1 2][3 4]`.
//
// |max_element_count| can be used to limit the total number of elements printed
// when the count may be large. Elided elements will be replaced with `...`.
//
// |host_allocator| will be used for any transient allocations required while
// printing.
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_fprint(
    FILE* file, const iree_hal_buffer_view_t* buffer_view,
    iree_host_size_t max_element_count, iree_allocator_t host_allocator);

// Maps and appends to |builder| a buffer view with contents without a trailing
// newline.
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_append_to_builder(
    const iree_hal_buffer_view_t* buffer_view,
    iree_host_size_t max_element_count, iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_BUFFER_VIEW_UTIL_H_
