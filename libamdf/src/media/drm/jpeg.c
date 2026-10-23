// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/drm/jpeg.h"

static void amdf_media_drm_jpeg_register(uint32_t** cursor, uint32_t address,
                                         uint32_t value) {
  *(*cursor)++ = address;
  *(*cursor)++ = value;
}

uint32_t amdf_media_drm_jpeg_emit(uint32_t* words, uint64_t compressed_address,
                                  uint32_t compressed_byte_length,
                                  const amdf_media_drm_frame_t* output) {
  uint32_t* cursor = words;
  // PACKETJ type 3/condition 3 waits for the masked hardware register value.
  // This per-picture decoder reset is the ordinary JPEG engine protocol, not
  // a DRM device/context reset or a device-fault recovery operation.
  amdf_media_drm_jpeg_register(&cursor, 0x402f, 1);
  amdf_media_drm_jpeg_register(&cursor, 0x408e, 0x01400200);
  amdf_media_drm_jpeg_register(&cursor, 0x408f, 1u << 16);
  amdf_media_drm_jpeg_register(&cursor, 0x3300402f, 1u << 16);
  amdf_media_drm_jpeg_register(&cursor, 0x402f, 0);
  amdf_media_drm_jpeg_register(&cursor, 0x408f, 0);
  amdf_media_drm_jpeg_register(&cursor, 0x3300402f, 1u << 16);
  // Complete JPEG bytes include marker/table headers and zero padding to 128B.
  amdf_media_drm_jpeg_register(&cursor, 0x40e1,
                               (uint32_t)(compressed_address >> 32));
  amdf_media_drm_jpeg_register(&cursor, 0x40e0, (uint32_t)compressed_address);
  amdf_media_drm_jpeg_register(&cursor, 0x4001, 0);
  amdf_media_drm_jpeg_register(&cursor, 0x4004, 0xfffffff0u);
  amdf_media_drm_jpeg_register(&cursor, 0x4002, compressed_byte_length / 4);
  amdf_media_drm_jpeg_register(&cursor, 0x401f, output->row_pitches[0] / 16);
  amdf_media_drm_jpeg_register(&cursor, 0x4020, output->row_pitches[1] / 16);
  // Linear luma/chroma addressing, without compression metadata.
  amdf_media_drm_jpeg_register(&cursor, 0x4027, 0);
  amdf_media_drm_jpeg_register(&cursor, 0x4024, 0);
  amdf_media_drm_jpeg_register(&cursor, 0x4025, 0);
  amdf_media_drm_jpeg_register(&cursor, 0x40e3,
                               (uint32_t)(output->addresses[0] >> 32));
  amdf_media_drm_jpeg_register(&cursor, 0x40e2, (uint32_t)output->addresses[0]);
  const uint32_t offsets[3] = {
      0, (uint32_t)(output->addresses[1] - output->addresses[0]), 0};
  for (uint32_t i = 0; i < 3; ++i) {
    amdf_media_drm_jpeg_register(&cursor, 0x402c, i);
    amdf_media_drm_jpeg_register(&cursor, 0x402d, offsets[i]);
  }
  amdf_media_drm_jpeg_register(&cursor, 0x400f, 0);
  amdf_media_drm_jpeg_register(&cursor, 0x401e, 0);
  amdf_media_drm_jpeg_register(&cursor, 0x401c, 0x14c7);
  amdf_media_drm_jpeg_register(&cursor, 0x400a, 0xfffffffeu);
  amdf_media_drm_jpeg_register(&cursor, 0x4000, 6);
  amdf_media_drm_jpeg_register(&cursor, 0x408f, compressed_byte_length / 4);
  amdf_media_drm_jpeg_register(&cursor, 0x408e, 0x01400200);
  amdf_media_drm_jpeg_register(&cursor, 0x33004003, UINT32_MAX);
  amdf_media_drm_jpeg_register(&cursor, 0x408f, UINT32_MAX);
  amdf_media_drm_jpeg_register(&cursor, 0x3300401d, 1);
  amdf_media_drm_jpeg_register(&cursor, 0x4000, 4);
  while ((cursor - words) % 16 != 0) {
    amdf_media_drm_jpeg_register(&cursor, 0x60000000, 0);
  }
  return (uint32_t)(cursor - words) * 4;
}
