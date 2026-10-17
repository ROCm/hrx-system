// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_IMAGE_OUTPUT_H_
#define LOOM_SERVE_SERVICE_IMAGE_OUTPUT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Checks representable F32 RGB storage and PNG dimensions/chunk limits before
// generation and returns the exact encoded byte length without allocating.
// Failure leaves *out_length zero. Model-specific geometry remains a source
// preparation contract, independent of these transport representation limits.
iree_status_t loom_serve_image_rgb_f32_png_size(uint32_t width, uint32_t height,
                                                iree_host_size_t* out_length);

// Encodes completed little-endian F32 CHW RGB in [-1,1] as an RGB8 PNG.
// The exact shape and every sample are checked before publishing output.
// Conversion rounds (value/2+0.5)*255 using the caller's FP rounding mode.
// PNG uses filter zero and stored DEFLATE blocks: one output allocation, no
// converted image scratch or external codec dependency. Nonfinite/out-of-range
// device output fails with DATA_LOSS instead of silently clamping it.
// Success transfers output storage to the caller, freed with host_allocator;
// failure leaves *out_png empty. Input storage is only borrowed during the
// call.
iree_status_t loom_serve_image_encode_rgb_f32_png(
    uint32_t width, uint32_t height, iree_const_byte_span_t rgb,
    iree_allocator_t host_allocator, iree_byte_span_t* out_png);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_IMAGE_OUTPUT_H_
