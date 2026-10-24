// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native NV12/P010 addressing for media ownership witnesses. Gfx1150 uses its
// qualified 64K_D_X swizzle while Linux VCN images use explicit linear pitches.
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

static unsigned chroma16_offset(unsigned x, unsigned y) {
  return ((((x >> 0)) & 1u) << 2) | ((((y >> 0)) & 1u) << 3) |
         ((((x >> 1)) & 1u) << 4) | ((((y >> 1)) & 1u) << 5) |
         ((((x >> 2)) & 1u) << 6) | ((((y >> 2)) & 1u) << 7) |
         ((((y >> 3) ^ (x >> 4)) & 1u) << 8) |
         ((((x >> 3) ^ (y >> 4)) & 1u) << 9) | ((((y >> 4)) & 1u) << 10) |
         ((((x >> 4)) & 1u) << 11) | ((((y >> 5)) & 1u) << 12) |
         ((((x >> 5)) & 1u) << 13) | ((((y >> 6)) & 1u) << 14) |
         ((((x >> 6)) & 1u) << 15);
}

struct Components {
  // Stored luma sample, retaining P010's six low zero bits.
  unsigned y;
  // Stored blue-difference chroma sample.
  unsigned u;
  // Stored red-difference chroma sample.
  unsigned v;
};

static Components sample(const unsigned char* planes, unsigned x, unsigned y,
                         unsigned width, unsigned p010, unsigned uv_offset,
                         unsigned y_xor, unsigned uv_xor, unsigned linear,
                         unsigned y_pitch, unsigned uv_pitch) {
  Components result = {};
  if (linear) {
    if (p010) {
      result.y = *reinterpret_cast<const unsigned short*>(planes + y * y_pitch +
                                                          x * 2);
      unsigned uv = *reinterpret_cast<const unsigned*>(
          planes + uv_offset + (y / 2) * uv_pitch + (x / 2) * 4);
      result.u = uv & 65535;
      result.v = uv >> 16;
    } else {
      result.y = planes[y * y_pitch + x];
      unsigned uv = *reinterpret_cast<const unsigned short*>(
          planes + uv_offset + (y / 2) * uv_pitch + (x / 2) * 2);
      result.u = uv & 255;
      result.v = uv >> 8;
    }
    return result;
  }
  if (p010) {
    unsigned y_block = (y / 128) * ((width + 255) / 256) + x / 256;
    unsigned uv_block = (y / 256) * ((width + 255) / 256) + x / 256;
    unsigned y_address = y_block * 65536 + (chroma_offset(x, y) ^ y_xor);
    unsigned uv_address =
        uv_offset + uv_block * 65536 + (chroma16_offset(x / 2, y / 2) ^ uv_xor);
    result.y = *reinterpret_cast<const unsigned short*>(planes + y_address);
    unsigned uv = *reinterpret_cast<const unsigned*>(planes + uv_address);
    result.u = uv & 65535;
    result.v = uv >> 16;
  } else {
    unsigned y_block = (y / 256) * ((width + 255) / 256) + x / 256;
    unsigned uv_block = (y / 256) * ((width + 511) / 512) + x / 512;
    unsigned y_address = y_block * 65536 + (luma_offset(x, y) ^ y_xor);
    unsigned uv_address =
        uv_offset + uv_block * 65536 + (chroma_offset(x / 2, y / 2) ^ uv_xor);
    result.y = planes[y_address];
    unsigned uv = *reinterpret_cast<const unsigned short*>(planes + uv_address);
    result.u = uv & 255;
    result.v = uv >> 8;
  }
  return result;
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
  loom::workgroup_count(2048, 1, 1)]]
void media_consume(const unsigned char* planes, unsigned* output,
                   unsigned crop_x, unsigned crop_y, unsigned generation,
                   unsigned uv_offset, unsigned y_xor, unsigned uv_xor,
                   unsigned width, unsigned height, unsigned p010,
                   unsigned linear, unsigned y_pitch, unsigned uv_pitch) {
  unsigned pixel = unsigned(loom::kernel::workgroup::id.x) * 64 +
                   unsigned(loom::kernel::workitem::id.x);
  if (pixel < width * height) {
    Components values =
        sample(planes, pixel % width, pixel / width, width, p010, uv_offset,
               y_xor, uv_xor, linear, y_pitch, uv_pitch);
    output[pixel * 2] = values.y | (values.u << 16);
    output[pixel * 2 + 1] = values.v | (generation << 16);
  }
  if (pixel < 4096) {
    Components values = sample(
        planes, crop_x + (pixel % 64) * 2, crop_y + (pixel / 64) * 2, width,
        p010, uv_offset, y_xor, uv_xor, linear, y_pitch, uv_pitch);
    float scale = p010 ? 1.0f / 256.0f : 1.0f;
    float luma = float(values.y) * scale;
    float u = float(values.u) * scale - 128.0f;
    float v = float(values.v) * scale - 128.0f;
    unsigned r = quantize(luma + 1.402f * v);
    unsigned g = quantize(luma - 0.344136f * u - 0.714136f * v);
    unsigned b = quantize(luma + 1.772f * u);
    output[width * height * 2 + pixel] = r | (g << 8) | (b << 16) | 0xff000000u;
  }
}

[[loom::kernel, loom::workgroup_size(64, 1, 1),
  loom::workgroup_count(2048, 1, 1)]]
void media_produce(unsigned char* planes, unsigned generation,
                   unsigned uv_offset, unsigned y_xor, unsigned uv_xor,
                   unsigned width, unsigned height, unsigned p010,
                   unsigned linear, unsigned y_pitch, unsigned uv_pitch) {
  unsigned pixel = unsigned(loom::kernel::workgroup::id.x) * 64 +
                   unsigned(loom::kernel::workitem::id.x);
  if (pixel >= width * height) {
    return;
  }
  unsigned x = pixel % width;
  unsigned y = pixel / width;
  unsigned value = 16 + x / 3 + y / 5 + generation * 4;
  if (y < 32) {
    value = 40 + generation * 12;
  }
  if (x >= 32 + generation * 8 && x < 64 + generation * 8 && y >= 96 &&
      y < 128) {
    value = 224;
  }
  if (linear && p010) {
    *reinterpret_cast<unsigned short*>(planes + y * y_pitch + x * 2) =
        static_cast<unsigned short>((value * 4 + (x + y + generation) % 4)
                                    << 6);
  } else if (linear) {
    planes[y * y_pitch + x] = static_cast<unsigned char>(value);
  } else if (p010) {
    unsigned y_block = (y / 128) * ((width + 255) / 256) + x / 256;
    unsigned y_address = y_block * 65536 + (chroma_offset(x, y) ^ y_xor);
    *reinterpret_cast<unsigned short*>(planes + y_address) =
        static_cast<unsigned short>((value * 4 + (x + y + generation) % 4)
                                    << 6);
  } else {
    unsigned y_block = (y / 256) * ((width + 255) / 256) + x / 256;
    unsigned y_address = y_block * 65536 + (luma_offset(x, y) ^ y_xor);
    planes[y_address] = static_cast<unsigned char>(value);
  }
  if ((x & 1u) != 0 || (y & 1u) != 0) {
    return;
  }
  unsigned chroma_x = x / 2;
  unsigned chroma_y = y / 2;
  unsigned u = 96 + x / 8 + generation;
  unsigned v = 160 - y / 8 - generation;
  if (linear && p010) {
    u = (u * 4 + (chroma_x + generation) % 4) << 6;
    v = (v * 4 + (chroma_y + generation) % 4) << 6;
    *reinterpret_cast<unsigned*>(planes + uv_offset + chroma_y * uv_pitch +
                                 chroma_x * 4) = u | (v << 16);
  } else if (linear) {
    *reinterpret_cast<unsigned short*>(planes + uv_offset +
                                       chroma_y * uv_pitch + chroma_x * 2) =
        static_cast<unsigned short>(u | (v << 8));
  } else if (p010) {
    unsigned uv_block = (y / 256) * ((width + 255) / 256) + x / 256;
    unsigned uv_address = uv_offset + uv_block * 65536 +
                          (chroma16_offset(chroma_x, chroma_y) ^ uv_xor);
    u = (u * 4 + (chroma_x + generation) % 4) << 6;
    v = (v * 4 + (chroma_y + generation) % 4) << 6;
    *reinterpret_cast<unsigned*>(planes + uv_address) = u | (v << 16);
  } else {
    unsigned uv_block = (y / 256) * ((width + 511) / 512) + x / 512;
    unsigned uv_address = uv_offset + uv_block * 65536 +
                          (chroma_offset(chroma_x, chroma_y) ^ uv_xor);
    *reinterpret_cast<unsigned short*>(planes + uv_address) =
        static_cast<unsigned short>(u | (v << 8));
  }
}
