// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/image/output.h"

#include <math.h>
#include <string.h>

// PNG's reflected IEEE CRC-32 polynomial, indexed one nibble at a time.
static uint32_t image_crc32(const uint8_t* data, iree_host_size_t length) {
  static const uint32_t table[16] = {
      0x00000000, 0x1db71064, 0x3b6e20c8, 0x26d930ac, 0x76dc4190, 0x6b6b51f4,
      0x4db26158, 0x5005713c, 0xedb88320, 0xf00f9344, 0xd6d6a3e8, 0xcb61b38c,
      0x9b64c2b0, 0x86d3d2d4, 0xa00ae278, 0xbdbdf21c};
  uint32_t crc = UINT32_MAX;
  for (iree_host_size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    crc = (crc >> 4) ^ table[crc & 15];
    crc = (crc >> 4) ^ table[crc & 15];
  }
  return ~crc;
}

static void image_store_be32(uint8_t* target, uint32_t value) {
  target[0] = (uint8_t)(value >> 24);
  target[1] = (uint8_t)(value >> 16);
  target[2] = (uint8_t)(value >> 8);
  target[3] = (uint8_t)value;
}

iree_status_t loom_serve_image_encode_rgb_f32_png(
    uint32_t width, uint32_t height, iree_const_byte_span_t rgb,
    iree_byte_span_t* out_png, iree_allocator_t host_allocator) {
  *out_png = iree_byte_span_empty();
  const uint64_t pixels = (uint64_t)width * height;
  if (!width || !height || width > INT32_MAX || height > INT32_MAX ||
      pixels > IREE_HOST_SIZE_MAX / 12 || rgb.data_length != pixels * 12) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected exact positive F32 CHW RGB shape");
  }
  const uint64_t row_length = (uint64_t)width * 3 + 1;
  const uint64_t raw_length = row_length * height;
  const uint64_t blocks = (raw_length + 65534) / 65535;
  const uint64_t idat_length = raw_length + 5 * blocks + 6;
  if (idat_length > INT32_MAX || idat_length + 57 > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "image exceeds one PNG data chunk");
  }
  uint8_t* png = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, (iree_host_size_t)idat_length + 57, (void**)&png));
  memcpy(png, "\x89PNG\r\n\x1a\n", 8);
  image_store_be32(png + 8, 13);
  memcpy(png + 12, "IHDR", 4);
  image_store_be32(png + 16, width);
  image_store_be32(png + 20, height);
  png[24] = 8;
  png[25] = 2;
  memset(png + 26, 0, 3);
  image_store_be32(png + 29, image_crc32(png + 12, 17));
  image_store_be32(png + 33, (uint32_t)idat_length);
  memcpy(png + 37, "IDAT", 4);
  png[41] = 0x78;
  png[42] = 0x01;
  for (uint64_t block = 0; block < blocks; ++block) {
    uint8_t* header = png + 43 + block * (65535 + 5);
    const uint16_t length =
        (uint16_t)iree_min(65535, raw_length - block * 65535);
    header[0] = block + 1 == blocks ? 1 : 0;
    iree_unaligned_store_le_u16(header + 1, length);
    iree_unaligned_store_le_u16(header + 3, (uint16_t)~length);
  }
  uint32_t adler_first = 1;
  uint32_t adler_second = 0;
  iree_status_t status = iree_ok_status();
  for (uint64_t i = 0; i < raw_length && iree_status_is_ok(status); ++i) {
    const uint64_t column = i % row_length;
    uint8_t value = 0;
    if (column) {
      const uint64_t channel = (column - 1) % 3;
      const uint64_t pixel = (i / row_length) * width + (column - 1) / 3;
      const float sample =
          iree_unaligned_load_le_f32(rgb.data + (channel * pixels + pixel) * 4);
      if (!isfinite(sample) || sample < -1.0f || sample > 1.0f) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "nonfinite or out-of-range final RGB");
      } else {
        value = (uint8_t)nearbyintf((sample / 2.0f + 0.5f) * 255.0f);
      }
    }
    // Five-byte stored-block headers interrupt the logical filtered stream.
    png[43 + i + 5 * (i / 65535 + 1)] = value;
    adler_first = (adler_first + value) % 65521;
    adler_second = (adler_second + adler_first) % 65521;
  }
  if (iree_status_is_ok(status)) {
    image_store_be32(png + 41 + idat_length - 4,
                     (adler_second << 16) | adler_first);
    image_store_be32(png + 41 + idat_length,
                     image_crc32(png + 37, (iree_host_size_t)idat_length + 4));
    uint8_t* end = png + 45 + idat_length;
    image_store_be32(end, 0);
    memcpy(end + 4, "IEND", 4);
    image_store_be32(end + 8, image_crc32(end + 4, 4));
    *out_png = iree_make_byte_span(png, (iree_host_size_t)idat_length + 57);
  } else {
    iree_allocator_free(host_allocator, png);
  }
  return status;
}
