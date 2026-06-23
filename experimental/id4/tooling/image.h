// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_TOOLING_IMAGE_H_
#define EXPERIMENTAL_ID4_TOOLING_IMAGE_H_

#include "experimental/id4/pipeline/plan.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Normalization applied before clamping and quantizing F32 RGB pixels.
typedef enum id4_tooling_image_normalization_e {
  // Invalid normalization sentinel.
  ID4_TOOLING_IMAGE_NORMALIZATION_INVALID = 0,
  // Input pixels are already in [0, 1].
  ID4_TOOLING_IMAGE_NORMALIZATION_ZERO_TO_ONE = 1,
  // Input pixels are in [-1, 1] and are transformed with (x + 1) / 2.
  ID4_TOOLING_IMAGE_NORMALIZATION_MINUS_ONE_TO_ONE = 2,
} id4_tooling_image_normalization_t;

// Options for writing a WHCB F32 RGB tensor as a binary PPM image.
typedef struct id4_tooling_write_f32_rgb_ppm_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Output file path to write.
  iree_string_view_t path;
  // WHCB tensor shape. Rank must be 4, C must be 3, and B must be 1.
  id4_pipeline_tensor_shape_t shape;
  // Raw F32 tensor bytes in WHCB dense order.
  iree_const_byte_span_t pixels;
  // Pixel normalization applied before clamping to [0, 1].
  id4_tooling_image_normalization_t normalization;
  // Host allocator used for transient path storage.
  iree_allocator_t host_allocator;
} id4_tooling_write_f32_rgb_ppm_options_t;

// Writes a WHCB F32 RGB tensor as a binary PPM image.
iree_status_t id4_tooling_write_f32_rgb_ppm(
    const id4_tooling_write_f32_rgb_ppm_options_t* options);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_TOOLING_IMAGE_H_
