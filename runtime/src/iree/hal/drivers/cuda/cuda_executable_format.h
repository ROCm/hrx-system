// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_CUDA_EXECUTABLE_FORMAT_H_
#define IREE_HAL_DRIVERS_CUDA_CUDA_EXECUTABLE_FORMAT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// CUDA executable bundle format
//===----------------------------------------------------------------------===//

// CUDA executable bundles use the following little-endian wire format:
//
//   header:
//     u32 magic                 // "CUDA"
//     u32 version               // IREE_HAL_CUDA_EXECUTABLE_FORMAT_VERSION
//     u32 module_count
//     u32 export_count
//
//   module[module_count]:
//     u32 ptx_image_offset      // Relative to the beginning of the bundle.
//     u32 ptx_image_length      // Excludes the required trailing NUL byte.
//
//   export[export_count]:
//     u32 module_ordinal
//     u32 kernel_name_offset    // Relative to the beginning of the bundle.
//     u32 kernel_name_length    // Excludes the required trailing NUL byte.
//     u32 block_size_x
//     u32 block_size_y
//     u32 block_size_z
//     u32 block_shared_memory_size
//     u32 constant_count
//     u32 binding_count
//
//   payload:
//     NUL-terminated kernel names and PTX source images
//
// Payload spans must begin after both tables and remain within the bundle. PTX
// images and kernel names must be non-empty, contain no embedded NUL bytes, and
// have a NUL byte immediately following their declared span. Payload spans may
// overlap, allowing exports to share names and modules to share PTX storage.

// Little-endian value whose bytes spell "CUDA".
#define IREE_HAL_CUDA_EXECUTABLE_FORMAT_MAGIC UINT32_C(0x41445543)

enum {
  // Version of the CUDA executable bundle format decoded by this runtime.
  IREE_HAL_CUDA_EXECUTABLE_FORMAT_VERSION = 1,
  // Encoded byte length of the bundle header.
  IREE_HAL_CUDA_EXECUTABLE_FORMAT_HEADER_SIZE = 16,
  // Encoded byte length of each module record.
  IREE_HAL_CUDA_EXECUTABLE_FORMAT_MODULE_SIZE = 8,
  // Encoded byte length of each export record.
  IREE_HAL_CUDA_EXECUTABLE_FORMAT_EXPORT_SIZE = 36,
};

// Validated view of a CUDA executable bundle.
//
// All storage is borrowed from the input bundle and must remain live while the
// view is used.
typedef struct iree_hal_cuda_executable_format_t {
  // Complete encoded bundle bytes.
  iree_const_byte_span_t data;
  // Number of PTX module records in the bundle.
  uint32_t module_count;
  // Number of export records in the bundle.
  uint32_t export_count;
  // Byte offset of the first export record.
  iree_host_size_t export_table_offset;
  // Byte offset immediately following the export table.
  iree_host_size_t payload_offset;
} iree_hal_cuda_executable_format_t;

// Decoded PTX image for one CUDA executable module.
//
// String storage is borrowed from the parsed bundle and has a trailing NUL byte
// that is not included in the string view.
typedef struct iree_hal_cuda_executable_module_t {
  // PTX source image passed to cuModuleLoadDataEx.
  iree_string_view_t ptx_image;
} iree_hal_cuda_executable_module_t;

// Decoded metadata for one CUDA executable export.
//
// String storage is borrowed from the parsed bundle and has a trailing NUL byte
// that is not included in the string view.
typedef struct iree_hal_cuda_executable_export_t {
  // Ordinal of the PTX module containing the exported kernel.
  uint32_t module_ordinal;
  // CUDA kernel symbol and stable HAL function name.
  iree_string_view_t kernel_name;
  // Required CUDA block dimensions.
  uint32_t block_size[3];
  // Dynamic shared memory size in bytes.
  uint32_t block_shared_memory_size;
  // Number of 32-bit dispatch constants.
  uint32_t constant_count;
  // Number of buffer bindings.
  uint32_t binding_count;
} iree_hal_cuda_executable_export_t;

// Parses and validates |data| as a CUDA executable bundle.
iree_status_t iree_hal_cuda_executable_format_parse(
    iree_const_byte_span_t data, iree_hal_cuda_executable_format_t* out_format);

// Decodes module |ordinal| from a validated executable bundle.
iree_status_t iree_hal_cuda_executable_format_read_module(
    const iree_hal_cuda_executable_format_t* format, iree_host_size_t ordinal,
    iree_hal_cuda_executable_module_t* out_module);

// Decodes export |ordinal| from a validated executable bundle.
iree_status_t iree_hal_cuda_executable_format_read_export(
    const iree_hal_cuda_executable_format_t* format, iree_host_size_t ordinal,
    iree_hal_cuda_executable_export_t* out_export);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_CUDA_CUDA_EXECUTABLE_FORMAT_H_
