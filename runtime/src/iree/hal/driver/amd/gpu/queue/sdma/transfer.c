// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/sdma/transfer.h"

#include "iree/base/internal/math.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/commands.h"

iree_hal_amd_gpu_dma_transfer_packet_t
iree_hal_amd_gpu_sdma_transfer_emit_region(
    amdf_queue_format_features_t features, uint64_t source, uint64_t target,
    const iree_hal_transfer_region_t* region, uint64_t offset,
    uint32_t word_capacity, uint32_t* words) {
  if (!(features & AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT) ||
      word_capacity < 13 || offset % region->row_length) {
    return (iree_hal_amd_gpu_dma_transfer_packet_t){0};
  }
  uint64_t source_offset, target_offset;
  iree_hal_transfer_region_t part = iree_hal_transfer_region_slice(
      region, offset, UINT32_MAX, &source_offset, &target_offset);
  source += source_offset;
  target += target_offset;
  // A single row per slice is an ordinary rectangle over the slice strides.
  if (part.row_count == 1 && part.slice_count > 1) {
    part.row_count = part.slice_count;
    part.slice_count = 1;
    part.source_strides.row_stride = part.source_strides.slice_stride;
    part.target_strides.row_stride = part.target_strides.slice_stride;
  }
  if (part.row_count == 1 || !part.source_strides.row_stride ||
      !part.target_strides.row_stride ||
      ((part.source_strides.row_stride | part.target_strides.row_stride) & 3)) {
    return (iree_hal_amd_gpu_dma_transfer_packet_t){0};
  }
  const bool wide =
      (features & AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_WIDE) != 0;
  const uint32_t maximum_xy = 1u << (wide ? 16 : 14);
  const uint32_t maximum_z =
      1u << (wide ? 14
             : (features &
                AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_EXTENDED_Z)
                 ? 13
                 : 11);
  const uint32_t maximum_pitch = 1u << (wide ? 16 : 19);
  const uint64_t maximum_slice_pitch = UINT64_C(1) << (wide ? 32 : 28);
  const uint64_t alignment = part.row_length | part.source_strides.row_stride |
                             part.target_strides.row_stride | (source & 3) |
                             (target & 3) | 16;
  uint32_t element = iree_math_count_trailing_zeros_u64(alignment);
  if ((part.row_length >> element) > maximum_xy) {
    return (iree_hal_amd_gpu_dma_transfer_packet_t){0};
  }
  const bool transpose =
      (part.source_strides.row_stride >> element) > maximum_pitch ||
      (part.target_strides.row_stride >> element) > maximum_pitch;
  if (transpose) {
    // The Z pitch is wider than Y's. Each original row becomes one slice;
    // separate original slices retain their existing logical cursor boundary.
    if ((part.source_strides.row_stride >> element) > maximum_slice_pitch ||
        (part.target_strides.row_stride >> element) > maximum_slice_pitch) {
      return (iree_hal_amd_gpu_dma_transfer_packet_t){0};
    }
    part.slice_count = iree_min(part.row_count, maximum_z);
    part.row_count = 1;
    part.source_strides.slice_stride = part.source_strides.row_stride;
    part.target_strides.slice_stride = part.target_strides.row_stride;
    part.source_strides.row_stride = part.target_strides.row_stride =
        iree_align_uint64(part.row_length, 4);
  } else {
    if (part.row_count > maximum_xy) {
      part.row_count = maximum_xy;
      part.slice_count = 1;
    }
    if (part.slice_count > 1) {
      const uint64_t slice_alignment =
          part.source_strides.slice_stride | part.target_strides.slice_stride;
      const uint32_t slice_element =
          iree_math_count_trailing_zeros_u64(alignment | slice_alignment);
      if ((slice_alignment & 3) || !part.source_strides.slice_stride ||
          !part.target_strides.slice_stride ||
          (part.row_length >> slice_element) > maximum_xy ||
          (part.source_strides.row_stride >> slice_element) > maximum_pitch ||
          (part.target_strides.row_stride >> slice_element) > maximum_pitch ||
          (part.source_strides.slice_stride >> slice_element) >
              maximum_slice_pitch ||
          (part.target_strides.slice_stride >> slice_element) >
              maximum_slice_pitch) {
        part.slice_count = 1;
      } else {
        element = slice_element;
        part.slice_count = iree_min(part.slice_count, maximum_z);
      }
    }
  }
  const uint32_t pitch_shift = wide ? 16 : 13;
  words[0] = 1u | (4u << 8) | (element << 29);
  words[1] = (uint32_t)source & ~3u;
  words[2] = (uint32_t)(source >> 32);
  words[3] = (uint32_t)(source & 3) >> element;
  words[4] = (uint32_t)((part.source_strides.row_stride >> element) - 1)
             << pitch_shift;
  words[5] = part.slice_count > 1
                 ? (uint32_t)((part.source_strides.slice_stride >> element) - 1)
                 : 0;
  words[6] = (uint32_t)target & ~3u;
  words[7] = (uint32_t)(target >> 32);
  words[8] = (uint32_t)(target & 3) >> element;
  words[9] = (uint32_t)((part.target_strides.row_stride >> element) - 1)
             << pitch_shift;
  words[10] =
      part.slice_count > 1
          ? (uint32_t)((part.target_strides.slice_stride >> element) - 1)
          : 0;
  words[11] = (uint32_t)((part.row_length >> element) - 1) |
              ((uint32_t)(part.row_count - 1) << 16);
  words[12] = (uint32_t)(part.slice_count - 1) |
              ((features & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE)
                   ? (3u << 18) | (3u << 26)
                   : 0);
  return (iree_hal_amd_gpu_dma_transfer_packet_t){
      .word_count = 13,
      .length = (uint32_t)iree_hal_transfer_region_length(&part),
  };
}

iree_hal_amd_gpu_dma_transfer_encoding_t
iree_hal_amd_gpu_sdma_transfer_encoding(amdf_queue_format_features_t features) {
  return (iree_hal_amd_gpu_dma_transfer_encoding_t){
      .features = features,
      .max_packet_length = IREE_HAL_AMD_GPU_SDMA_MAX_PACKET_LENGTH,
      .max_packet_words = 7,
      .copy = iree_hal_amd_gpu_sdma_emit_copy,
      .fill = iree_hal_amd_gpu_sdma_emit_fill,
      .copy_region = (features & AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT)
                         ? iree_hal_amd_gpu_sdma_transfer_emit_region
                         : NULL,
  };
}
