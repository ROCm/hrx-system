// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_METAL_EXECUTABLE_FORMAT_H_
#define IREE_HAL_DRIVERS_METAL_EXECUTABLE_FORMAT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Metal executable bundle format
//===----------------------------------------------------------------------===//

// Metal executable bundles use the following little-endian wire format:
//
//   header:
//     u32 magic                 // "MTLE"
//     u32 version               // IREE_HAL_METAL_EXECUTABLE_FORMAT_VERSION
//     u32 library_count
//     u32 pipeline_count
//
//   library[library_count]:
//     u32 source_offset         // Relative to the beginning of the bundle.
//     u32 source_length         // Zero when source is absent.
//     u32 source_version        // UINT32_MAX selects the runtime default.
//     u32 metallib_offset       // Relative to the beginning of the bundle.
//     u32 metallib_length       // Zero when a metallib is absent.
//
//   pipeline[pipeline_count]:
//     u32 library_ordinal
//     u32 entry_point_offset    // Relative to the beginning of the bundle.
//     u32 entry_point_length
//     u32 max_threads_per_threadgroup  // Zero selects the Metal default.
//     u32 threadgroup_size_x
//     u32 threadgroup_size_y
//     u32 threadgroup_size_z
//     u32 flags
//     u32 constant_count
//     u32 binding_count
//     u64 binding_read_only_bits
//
//   payload:
//     MSL source, metallib images, and entry-point names
//
// Payload spans must begin after both tables and remain within the bundle.
// Text spans must be non-empty and contain no embedded NUL bytes; they need not
// be NUL-terminated. Binary and text spans may overlap. When both MSL source
// and a metallib are present, the metallib is used. An absent source has a zero
// offset and length and must use the default source-version sentinel. An absent
// metallib has a zero offset and length.

// Little-endian value whose bytes spell "MTLE".
#define IREE_HAL_METAL_EXECUTABLE_FORMAT_MAGIC UINT32_C(0x454C544D)

enum {
  // Version of the Metal executable bundle format decoded by this runtime.
  IREE_HAL_METAL_EXECUTABLE_FORMAT_VERSION = 1,
  // Encoded byte length of the bundle header.
  IREE_HAL_METAL_EXECUTABLE_FORMAT_HEADER_SIZE = 16,
  // Encoded byte length of each library record.
  IREE_HAL_METAL_EXECUTABLE_FORMAT_LIBRARY_SIZE = 20,
  // Encoded byte length of each pipeline record.
  IREE_HAL_METAL_EXECUTABLE_FORMAT_PIPELINE_SIZE = 48,
  // Maximum number of buffer bindings supported by the Metal HAL ABI.
  IREE_HAL_METAL_MAX_DESCRIPTOR_SET_BINDING_COUNT = 16,
  // Maximum number of 32-bit push constants supported by the Metal HAL ABI.
  IREE_HAL_METAL_MAX_PUSH_CONSTANT_COUNT = 64,
};

// Selects the latest MSL language version supported by the runtime.
#define IREE_HAL_METAL_EXECUTABLE_SOURCE_VERSION_DEFAULT UINT32_MAX

// Pipeline behavior encoded in each pipeline record.
typedef uint32_t iree_hal_metal_executable_pipeline_flags_t;
enum iree_hal_metal_executable_pipeline_flag_bits_e {
  // The threadgroup size is always a multiple of the execution width.
  IREE_HAL_METAL_EXECUTABLE_PIPELINE_FLAG_THREADGROUP_SIZE_ALIGNED = 1u << 0,
};

// Validated view of a Metal executable bundle.
//
// All storage is borrowed from the input bundle and must remain live while the
// view is used.
typedef struct iree_hal_metal_executable_format_t {
  // Complete encoded bundle bytes.
  iree_const_byte_span_t data;
  // Number of Metal library records in the bundle.
  uint32_t library_count;
  // Number of compute pipeline records in the bundle.
  uint32_t pipeline_count;
  // Byte offset of the first pipeline record.
  iree_host_size_t pipeline_table_offset;
  // Byte offset immediately following the pipeline table.
  iree_host_size_t payload_offset;
} iree_hal_metal_executable_format_t;

// Decoded source and binary data for one Metal library.
//
// Storage is borrowed from the parsed bundle. Empty spans indicate that the
// corresponding representation is absent.
typedef struct iree_hal_metal_executable_library_t {
  // Optional MSL source text.
  iree_string_view_t source;
  // MTLLanguageVersion value or the default-version sentinel.
  uint32_t source_version;
  // Optional precompiled metallib image.
  iree_const_byte_span_t metallib;
} iree_hal_metal_executable_library_t;

// Decoded metadata for one Metal compute pipeline.
//
// String storage is borrowed from the parsed bundle.
typedef struct iree_hal_metal_executable_pipeline_t {
  // Ordinal of the Metal library containing the exported function.
  uint32_t library_ordinal;
  // Metal function name and stable HAL function name.
  iree_string_view_t entry_point;
  // Optional maximum total threads per threadgroup.
  uint32_t max_threads_per_threadgroup;
  // Required static threadgroup dimensions.
  uint32_t threadgroup_size[3];
  // Pipeline behavior flags.
  iree_hal_metal_executable_pipeline_flags_t flags;
  // Number of 32-bit dispatch constants.
  uint16_t constant_count;
  // Number of densely numbered buffer bindings.
  uint16_t binding_count;
  // One bit per binding indicating that the binding is read-only.
  uint64_t binding_read_only_bits;
} iree_hal_metal_executable_pipeline_t;

// Parses and validates |data| as a Metal executable bundle.
iree_status_t iree_hal_metal_executable_format_parse(
    iree_const_byte_span_t data,
    iree_hal_metal_executable_format_t* out_format);

// Decodes library |ordinal| from a validated executable bundle.
iree_status_t iree_hal_metal_executable_format_read_library(
    const iree_hal_metal_executable_format_t* format, iree_host_size_t ordinal,
    iree_hal_metal_executable_library_t* out_library);

// Decodes pipeline |ordinal| from a validated executable bundle.
iree_status_t iree_hal_metal_executable_format_read_pipeline(
    const iree_hal_metal_executable_format_t* format, iree_host_size_t ordinal,
    iree_hal_metal_executable_pipeline_t* out_pipeline);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_METAL_EXECUTABLE_FORMAT_H_
