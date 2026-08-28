// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_WEBGPU_EXECUTABLE_FORMAT_H_
#define IREE_HAL_DRIVERS_WEBGPU_WEBGPU_EXECUTABLE_FORMAT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// WebGPU executable bundle format
//===----------------------------------------------------------------------===//

// WebGPU executable bundles use the following little-endian wire format:
//
//   header:
//     u32 magic                 // "WGPU"
//     u32 version               // IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_VERSION
//     u32 export_count
//
//   export[export_count]:
//     u32 wgsl_source_offset    // Relative to the beginning of the bundle.
//     u32 wgsl_source_length
//     u32 entry_point_offset    // Relative to the beginning of the bundle.
//     u32 entry_point_length
//     u32 workgroup_size_x
//     u32 workgroup_size_y
//     u32 workgroup_size_z
//     u32 binding_count
//
//   payload:
//     entry-point names and WGSL source bytes
//
// Payload spans must begin after the export table and remain within the bundle.
// They need not be NUL-terminated and may overlap, allowing several exports to
// reference one WGSL source module. Workgroup sizes must match the
// corresponding WGSL entry-point declarations. WebGPU has no push constants, so
// constants are deliberately absent from the format and must be lowered to
// buffer bindings.

// Little-endian value whose bytes spell "WGPU".
#define IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_MAGIC UINT32_C(0x55504757)

enum {
  // Version of the WebGPU executable bundle format decoded by this runtime.
  IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_VERSION = 1,
  // Encoded byte length of the bundle header.
  IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_HEADER_SIZE = 12,
  // Encoded byte length of each export record.
  IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_EXPORT_SIZE = 32,
};

// Validated view of a WebGPU executable bundle.
//
// All storage is borrowed from the input bundle and must remain live while the
// view is used.
typedef struct iree_hal_webgpu_executable_format_t {
  // Complete encoded bundle bytes.
  iree_const_byte_span_t data;
  // Number of export records in the bundle.
  uint32_t export_count;
  // Byte offset immediately following the export table.
  iree_host_size_t payload_offset;
} iree_hal_webgpu_executable_format_t;

// Decoded metadata and source for one WebGPU executable export.
//
// String storage is borrowed from the parsed bundle.
typedef struct iree_hal_webgpu_executable_export_t {
  // WGSL source containing the exported entry point.
  iree_string_view_t wgsl_source;
  // WGSL entry-point name and stable HAL function name.
  iree_string_view_t entry_point;
  // Static workgroup size declared by the entry point.
  uint32_t workgroup_size[3];
  // Number of densely numbered WebGPU resource bindings.
  uint16_t binding_count;
} iree_hal_webgpu_executable_export_t;

// Parses and validates |data| as a WebGPU executable bundle.
iree_status_t iree_hal_webgpu_executable_format_parse(
    iree_const_byte_span_t data,
    iree_hal_webgpu_executable_format_t* out_format);

// Decodes export |ordinal| from a validated executable bundle.
iree_status_t iree_hal_webgpu_executable_format_read_export(
    const iree_hal_webgpu_executable_format_t* format, iree_host_size_t ordinal,
    iree_hal_webgpu_executable_export_t* out_export);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_WEBGPU_EXECUTABLE_FORMAT_H_
