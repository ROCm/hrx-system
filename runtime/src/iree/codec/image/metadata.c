// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/codec/image/metadata.h"

#include <string.h>

static bool iree_codec_image_tiff_read_u16(iree_const_byte_span_t data,
                                           bool little_endian,
                                           iree_host_size_t offset,
                                           uint16_t* out_value) {
  if (offset > data.data_length || data.data_length - offset < 2) {
    return false;
  }
  *out_value = little_endian ? iree_codec_image_load_le_u16(data.data + offset)
                             : iree_codec_image_load_be_u16(data.data + offset);
  return true;
}

static bool iree_codec_image_tiff_read_u32(iree_const_byte_span_t data,
                                           bool little_endian,
                                           iree_host_size_t offset,
                                           uint32_t* out_value) {
  if (offset > data.data_length || data.data_length - offset < 4) {
    return false;
  }
  *out_value = little_endian ? iree_codec_image_load_le_u32(data.data + offset)
                             : iree_codec_image_load_be_u32(data.data + offset);
  return true;
}

// Malformed optional metadata has the same effect as absent metadata. Codec
// validation still owns the image byte stream itself.
uint8_t iree_codec_image_exif_orientation(iree_const_byte_span_t data) {
  static const uint8_t exif_prefix[] = {'E', 'x', 'i', 'f', 0, 0};
  if (data.data_length >= sizeof(exif_prefix) &&
      memcmp(data.data, exif_prefix, sizeof(exif_prefix)) == 0) {
    data.data += sizeof(exif_prefix);
    data.data_length -= sizeof(exif_prefix);
  }
  if (data.data_length < 8) {
    return 1;
  }
  const bool little_endian = data.data[0] == 'I' && data.data[1] == 'I';
  if (!little_endian && !(data.data[0] == 'M' && data.data[1] == 'M')) {
    return 1;
  }
  uint16_t magic = 0;
  uint32_t directory_offset = 0;
  if (!iree_codec_image_tiff_read_u16(data, little_endian, 2, &magic) ||
      magic != 42 ||
      !iree_codec_image_tiff_read_u32(data, little_endian, 4,
                                      &directory_offset)) {
    return 1;
  }
  uint16_t entry_count = 0;
  if (!iree_codec_image_tiff_read_u16(data, little_endian, directory_offset,
                                      &entry_count)) {
    return 1;
  }
  iree_host_size_t entry_offset = (iree_host_size_t)directory_offset + 2;
  for (uint16_t i = 0; i < entry_count; ++i) {
    if (entry_offset > data.data_length ||
        data.data_length - entry_offset < 12) {
      return 1;
    }
    uint16_t tag = 0;
    uint16_t type = 0;
    uint32_t count = 0;
    if (!iree_codec_image_tiff_read_u16(data, little_endian, entry_offset,
                                        &tag) ||
        !iree_codec_image_tiff_read_u16(data, little_endian, entry_offset + 2,
                                        &type) ||
        !iree_codec_image_tiff_read_u32(data, little_endian, entry_offset + 4,
                                        &count)) {
      return 1;
    }
    if (tag == 0x0112 && type == 3 && count == 1) {
      uint16_t orientation = 0;
      if (iree_codec_image_tiff_read_u16(data, little_endian, entry_offset + 8,
                                         &orientation) &&
          orientation >= 1 && orientation <= 8) {
        return (uint8_t)orientation;
      }
      return 1;
    }
    entry_offset += 12;
  }
  return 1;
}

bool iree_codec_image_chunk(iree_const_byte_span_t data,
                            iree_host_size_t* offset,
                            iree_host_size_t header_length, uint32_t length,
                            iree_const_byte_span_t* out_chunk) {
  if (*offset > data.data_length ||
      data.data_length - *offset < header_length ||
      length > data.data_length - *offset - header_length) {
    return false;
  }
  *out_chunk =
      iree_make_const_byte_span(data.data + *offset + header_length, length);
  *offset += header_length + length;
  return true;
}
