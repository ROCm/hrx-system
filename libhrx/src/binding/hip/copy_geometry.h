// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_HIP_COPY_GEOMETRY_H_
#define HRX_BINDING_HIP_COPY_GEOMETRY_H_

#include "binding/hip/api.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Byte geometry of one endpoint, independent of its address and memory class.
typedef struct iree_hip_copy_layout_t {
  // Byte offset from the allocation pointer to the first copied byte.
  iree_host_size_t offset;
  // Byte span from the first copied byte through the final copied row.
  iree_host_size_t span;
  // Byte distance between successive rows.
  iree_host_size_t row_pitch;
  // Byte distance between successive depth slices.
  iree_host_size_t slice_pitch;
} iree_hip_copy_layout_t;

// Checked geometry carried from HIP parameter preparation to copy recording.
typedef struct iree_hip_copy_geometry_t {
  // Region dimensions with width in bytes, including resolved array elements.
  hipExtent extent;
  // Number of copied bytes, excluding row and slice padding.
  iree_host_size_t length;
  // Source offsets and strides relative to the supplied source pointer.
  iree_hip_copy_layout_t source;
  // Destination offsets and strides relative to the supplied target pointer.
  iree_hip_copy_layout_t target;
} iree_hip_copy_geometry_t;

// Calculates checked byte geometry for a nonempty copy. The caller resolves
// array descriptors and validates the public API's pitch/origin bounds first.
// Zero ysize includes the y origin in the effective slice height. Returns false
// on user-supplied arithmetic overflow; no pointers are accessed or retained.
// The result is only usable on success.
bool iree_hip_copy_geometry_calculate(const hipMemcpy3DParms* params,
                                      iree_hip_copy_geometry_t* out_geometry);

// Returns the packed layout used by staging for already-validated geometry.
static inline iree_hip_copy_layout_t iree_hip_copy_geometry_compact_layout(
    const iree_hip_copy_geometry_t* geometry) {
  return (iree_hip_copy_layout_t){
      .offset = 0,
      .span = geometry->length,
      .row_pitch = geometry->extent.width,
      .slice_pitch = geometry->extent.width * geometry->extent.height,
  };
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_HIP_COPY_GEOMETRY_H_
