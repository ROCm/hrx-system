// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/copy_geometry.h"

static bool iree_hip_copy_layout_calculate(hipPitchedPtr pointer, size_t x,
                                           size_t y, size_t z, hipExtent extent,
                                           iree_hip_copy_layout_t* out_layout) {
  iree_host_size_t rows_per_slice = pointer.ysize;
  iree_host_size_t slice_pitch = 0;
  iree_host_size_t offset = 0;
  iree_host_size_t row_offset = 0;
  iree_host_size_t span = 0;
  iree_host_size_t leading_rows = 0;
  iree_host_size_t end = 0;
  if (IREE_UNLIKELY(
          (rows_per_slice == 0 &&
           !iree_host_size_checked_add(y, extent.height, &rows_per_slice)) ||
          !iree_host_size_checked_mul(pointer.pitch, rows_per_slice,
                                      &slice_pitch) ||
          !iree_host_size_checked_mul(z, slice_pitch, &offset) ||
          !iree_host_size_checked_mul(y, pointer.pitch, &row_offset) ||
          !iree_host_size_checked_add(offset, row_offset, &offset) ||
          !iree_host_size_checked_add(offset, x, &offset) ||
          !iree_host_size_checked_mul(extent.depth - 1, slice_pitch, &span) ||
          !iree_host_size_checked_mul(extent.height - 1, pointer.pitch,
                                      &leading_rows) ||
          !iree_host_size_checked_add(span, leading_rows, &span) ||
          !iree_host_size_checked_add(span, extent.width, &span) ||
          !iree_host_size_checked_add(offset, span, &end))) {
    return false;
  }
  *out_layout = (iree_hip_copy_layout_t){
      .offset = offset,
      .span = span,
      .row_pitch = pointer.pitch,
      .slice_pitch = slice_pitch,
  };
  return true;
}

bool iree_hip_copy_geometry_calculate(const hipMemcpy3DParms* params,
                                      iree_hip_copy_geometry_t* out_geometry) {
  out_geometry->extent = (hipExtent){
      params->extent.width, params->extent.height, params->extent.depth};
  iree_host_size_t slice_length = 0;
  return iree_host_size_checked_mul(params->extent.width, params->extent.height,
                                    &slice_length) &&
         iree_host_size_checked_mul(slice_length, params->extent.depth,
                                    &out_geometry->length) &&
         iree_hip_copy_layout_calculate(
             params->srcPtr, params->srcPos.x, params->srcPos.y,
             params->srcPos.z, out_geometry->extent, &out_geometry->source) &&
         iree_hip_copy_layout_calculate(
             params->dstPtr, params->dstPos.x, params->dstPos.y,
             params->dstPos.z, out_geometry->extent, &out_geometry->target);
}
