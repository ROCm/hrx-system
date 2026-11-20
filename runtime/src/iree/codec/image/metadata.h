// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEC_IMAGE_METADATA_H_
#define IREE_CODEC_IMAGE_METADATA_H_

#include "iree/base/api.h"

static inline uint16_t iree_codec_image_load_le_u16(const uint8_t* data) {
  return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

static inline uint32_t iree_codec_image_load_le_u32(const uint8_t* data) {
  return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
         ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

static inline uint16_t iree_codec_image_load_be_u16(const uint8_t* data) {
  return ((uint16_t)data[0] << 8) | (uint16_t)data[1];
}

static inline uint32_t iree_codec_image_load_be_u32(const uint8_t* data) {
  return ((uint32_t)data[0] << 24) | ((uint32_t)data[1] << 16) |
         ((uint32_t)data[2] << 8) | (uint32_t)data[3];
}

// Reads optional TIFF/EXIF orientation; absent or malformed optional metadata
// has the default orientation. Codec APIs validate the actual pixel stream.
uint8_t iree_codec_image_exif_orientation(iree_const_byte_span_t data);

// Consumes one length-delimited container chunk after its fixed-size header.
// Returns false for a truncated header or payload without advancing offset.
bool iree_codec_image_chunk(iree_const_byte_span_t data,
                            iree_host_size_t* offset,
                            iree_host_size_t header_length, uint32_t length,
                            iree_const_byte_span_t* out_chunk);

#endif  // IREE_CODEC_IMAGE_METADATA_H_
