// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Gfx1150 64K_D_X NV12 addressing for the bounded media ownership witness.
// This test checks the returned swizzle and XOR before launching this recipe.
#include <loomcxx/kernel.h>
#include <loomcxx/scalar.h>

static unsigned luma_offset(unsigned x, unsigned y) {
  return ((((x >> 0)) & 1u) << 0) | ((((x >> 1)) & 1u) << 1) |
         ((((y >> 0)) & 1u) << 2) | ((((x >> 2)) & 1u) << 3) |
         ((((y >> 1)) & 1u) << 4) | ((((y >> 2)) & 1u) << 5) |
         ((((x >> 3)) & 1u) << 6) | ((((y >> 3)) & 1u) << 7) |
         ((((y >> 4) ^ (x >> 5)) & 1u) << 8) |
         ((((x >> 4) ^ (y >> 5)) & 1u) << 9) | ((((y >> 5)) & 1u) << 10) |
         ((((x >> 5)) & 1u) << 11) | ((((y >> 6)) & 1u) << 12) |
         ((((x >> 6)) & 1u) << 13) | ((((y >> 7)) & 1u) << 14) |
         ((((x >> 7)) & 1u) << 15);
}

static unsigned chroma_offset(unsigned x, unsigned y) {
  return ((((x >> 0)) & 1u) << 1) | ((((y >> 0)) & 1u) << 2) |
         ((((x >> 1)) & 1u) << 3) | ((((y >> 1)) & 1u) << 4) |
         ((((x >> 2)) & 1u) << 5) | ((((y >> 2)) & 1u) << 6) |
         ((((x >> 3)) & 1u) << 7) | ((((y >> 3) ^ (x >> 5)) & 1u) << 8) |
         ((((x >> 4) ^ (y >> 4)) & 1u) << 9) | ((((y >> 4)) & 1u) << 10) |
         ((((x >> 5)) & 1u) << 11) | ((((y >> 5)) & 1u) << 12) |
         ((((x >> 6)) & 1u) << 13) | ((((y >> 6)) & 1u) << 14) |
         ((((x >> 7)) & 1u) << 15);
}

static unsigned sample(const unsigned char* planes, unsigned x, unsigned y,
                       unsigned uv_offset, unsigned y_xor, unsigned uv_xor) {
  unsigned luma = planes[luma_offset(x, y) ^ y_xor];
  unsigned chroma = uv_offset + (chroma_offset(x / 2, y / 2) ^ uv_xor);
  unsigned u = planes[chroma];
  unsigned v = planes[chroma + 1];
  return luma | (u << 8) | (v << 16);
}
static unsigned quantize(float value) {
  if (value < 0.0f) {
    value = 0.0f;
  }
  if (value > 255.0f) {
    value = 255.0f;
  }
  return unsigned(loom::scalar::roundevenf(value));
}
[[loom::kernel, loom::workgroup_size(64, 1, 1),
  loom::workgroup_count(1024, 1, 1)]]
void media_consume(const unsigned char* planes, unsigned* output,
                   unsigned crop_x, unsigned crop_y, unsigned generation,
                   unsigned uv_offset, unsigned y_xor, unsigned uv_xor) {
  unsigned pixel = unsigned(loom::kernel::workgroup::id.x) * 64 +
                   unsigned(loom::kernel::workitem::id.x);
  output[pixel] =
      sample(planes, pixel % 256, pixel / 256, uv_offset, y_xor, uv_xor) |
      (generation << 24);
  if (pixel < 4096) {
    unsigned packed =
        sample(planes, crop_x + (pixel % 64) * 2, crop_y + (pixel / 64) * 2,
               uv_offset, y_xor, uv_xor);
    float luma = float(packed & 255);
    float u = float((packed >> 8) & 255) - 128.0f;
    float v = float((packed >> 16) & 255) - 128.0f;
    unsigned r = quantize(luma + 1.402f * v);
    unsigned g = quantize(luma - 0.344136f * u - 0.714136f * v);
    unsigned b = quantize(luma + 1.772f * u);
    output[65536 + pixel] = r | (g << 8) | (b << 16) | 0xff000000u;
  }
}
