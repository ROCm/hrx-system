// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_VALUE_IO_H_
#define IREE_TOOLING_VALUE_IO_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Scalar Values
//===----------------------------------------------------------------------===//

// Kind of scalar value parsed from a tooling value specification.
typedef uint8_t iree_tooling_value_kind_t;
enum iree_tooling_value_kind_e {
  // No scalar value is present.
  IREE_TOOLING_VALUE_KIND_NONE = 0,
  // Signed 32-bit integer value.
  IREE_TOOLING_VALUE_KIND_I32 = 1,
  // Unsigned 32-bit integer value.
  IREE_TOOLING_VALUE_KIND_U32 = 2,
  // Signed 64-bit integer value.
  IREE_TOOLING_VALUE_KIND_I64 = 3,
  // Unsigned 64-bit integer value.
  IREE_TOOLING_VALUE_KIND_U64 = 4,
  // IEEE 754 binary32 value.
  IREE_TOOLING_VALUE_KIND_F32 = 5,
  // IEEE 754 binary64 value.
  IREE_TOOLING_VALUE_KIND_F64 = 6,
  // Raw 32-bit ABI word specified as a bare hexadecimal literal.
  IREE_TOOLING_VALUE_KIND_RAW_U32 = 7,
};

// Parsed scalar value.
typedef struct iree_tooling_value_t {
  // The scalar kind active in |storage|.
  iree_tooling_value_kind_t kind;
  // Type-specific scalar storage.
  union {
    // Signed 32-bit integer payload.
    int32_t i32;
    // Unsigned 32-bit integer payload.
    uint32_t u32;
    // Signed 64-bit integer payload.
    int64_t i64;
    // Unsigned 64-bit integer payload.
    uint64_t u64;
    // IEEE 754 binary32 payload.
    float f32;
    // IEEE 754 binary64 payload.
    double f64;
  } storage;
} iree_tooling_value_t;

// Parses |literal| as an explicit scalar |kind|.
iree_status_t iree_tooling_value_parse(iree_tooling_value_kind_t kind,
                                       iree_string_view_t literal,
                                       iree_tooling_value_t* out_value);

// Parses a strict scalar specification.
//
// Supported forms are:
//   i32=...
//   u32=...
//   i64=...
//   u64=...
//   f32=...
//   f64=...
//   0x...      raw uint32 ABI word
//
// Bare decimal literals are rejected so raw ABI inputs cannot accidentally
// depend on an implicit signedness or width.
iree_status_t iree_tooling_value_spec_parse(iree_string_view_t spec,
                                            iree_tooling_value_t* out_value);

// Returns the number of 32-bit ABI words required to encode |kind|.
iree_host_size_t iree_tooling_value_abi_word_count(
    iree_tooling_value_kind_t kind);

// Writes |value| into an ordered 32-bit ABI word stream.
//
// Floating-point values are bit-preserved in the host ABI representation.
iree_status_t iree_tooling_value_write_abi_words(
    const iree_tooling_value_t* value, iree_host_size_t word_capacity,
    uint32_t* out_words, iree_host_size_t* out_word_count);

//===----------------------------------------------------------------------===//
// Buffer Specifications
//===----------------------------------------------------------------------===//

// Reusable state for parsing value I/O specifications.
typedef struct iree_tooling_value_io_context_t iree_tooling_value_io_context_t;

// Allocates a value I/O context.
iree_status_t iree_tooling_value_io_context_allocate(
    iree_allocator_t host_allocator,
    iree_tooling_value_io_context_t** out_context);

// Frees a value I/O context and closes any streams it owns.
void iree_tooling_value_io_context_free(
    iree_tooling_value_io_context_t* context);

// Parses a single HAL buffer view specification.
//
// Supported forms match the function tooling tensor grammar:
//   2x2xf32=1,2,3,4
//   2x2xf32=@file.bin
//   2x2xf32=+file.bin
//   @file.npy
//   +file.npy
//
// A single specification always materializes a single buffer view. Splat forms
// that expand one specification into multiple values are intentionally handled
// above this layer by callers that own an aggregate argument convention.
iree_status_t iree_tooling_buffer_view_spec_parse(
    iree_tooling_value_io_context_t* context, iree_string_view_t spec,
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    iree_hal_buffer_view_t** out_buffer_view);

// Parses a storage-buffer specification beginning with `&`.
//
// The shaped tensor metadata is used only to size and optionally initialize the
// returned buffer. The caller owns the returned buffer.
iree_status_t iree_tooling_storage_buffer_spec_parse(
    iree_tooling_value_io_context_t* context, iree_string_view_t spec,
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    iree_hal_buffer_t** out_buffer);

// Kind of buffer binding parsed from a tooling buffer specification.
typedef uint8_t iree_tooling_buffer_binding_kind_t;
enum iree_tooling_buffer_binding_kind_e {
  // No buffer binding is present.
  IREE_TOOLING_BUFFER_BINDING_KIND_NONE = 0,
  // Binding came from a `&...` storage-buffer specification.
  IREE_TOOLING_BUFFER_BINDING_KIND_STORAGE_BUFFER = 1,
  // Binding came from a shaped HAL buffer-view specification.
  IREE_TOOLING_BUFFER_BINDING_KIND_BUFFER_VIEW = 2,
};

// Retained HAL binding materialized from a tooling buffer specification.
typedef struct iree_tooling_buffer_binding_t {
  // The buffer binding kind active in this record.
  iree_tooling_buffer_binding_kind_t kind;
  // Retained storage buffer backing the binding.
  iree_hal_buffer_t* buffer;
  // Optional retained buffer view preserving shaped metadata.
  iree_hal_buffer_view_t* buffer_view;
  // Logical byte offset to pass for the binding.
  iree_device_size_t byte_offset;
  // Logical byte length to pass for the binding.
  iree_device_size_t byte_length;
} iree_tooling_buffer_binding_t;

// Releases any retained objects in |binding| and resets it to empty.
void iree_tooling_buffer_binding_deinitialize(
    iree_tooling_buffer_binding_t* binding);

// Parses a single HAL buffer binding specification.
iree_status_t iree_tooling_buffer_binding_spec_parse(
    iree_tooling_value_io_context_t* context, iree_string_view_t spec,
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    iree_tooling_buffer_binding_t* out_binding);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_VALUE_IO_H_
