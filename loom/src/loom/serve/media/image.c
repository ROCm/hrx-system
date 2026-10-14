// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/media/image.h"

#include <setjmp.h>
#include <string.h>

#include "gif_lib.h"
#include "iree/base/internal/base64.h"
#include "iree/base/internal/json.h"
#include "jpeglib.h"
#include "png.h"
#include "webp/decode.h"

typedef struct loom_serve_media_compressed_image_t {
  iree_allocator_t host_allocator;
  uint8_t* allocation;
  iree_const_byte_span_t data;
  loom_serve_media_image_format_t format;
} loom_serve_media_compressed_image_t;

static void loom_serve_media_compressed_image_deinitialize(
    loom_serve_media_compressed_image_t* image) {
  iree_allocator_free(image->host_allocator, image->allocation);
  memset(image, 0, sizeof(*image));
}

static iree_status_t loom_serve_media_image_data_url_decode(
    iree_string_view_t escaped_data_url, iree_host_size_t maximum_length,
    iree_allocator_t host_allocator,
    loom_serve_media_compressed_image_t* out_image) {
  memset(out_image, 0, sizeof(*out_image));
  out_image->host_allocator = host_allocator;

  iree_host_size_t data_url_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_json_unescape_string(escaped_data_url, 0, NULL, &data_url_length));
  const iree_host_size_t maximum_encoded_length =
      iree_base64_encoded_size(maximum_length);
  iree_host_size_t maximum_data_url_length = 0;
  if (maximum_encoded_length == IREE_HOST_SIZE_MAX ||
      !iree_host_size_checked_add(maximum_encoded_length, 32,
                                  &maximum_data_url_length) ||
      data_url_length > maximum_data_url_length) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "image data URL exceeds the compressed byte bound");
  }
  if (!data_url_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image data URL is empty");
  }

  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, data_url_length,
                                             (void**)&out_image->allocation));
  iree_status_t status = iree_ok_status();
  iree_host_size_t unescaped_length = 0;
  status = iree_json_unescape_string(escaped_data_url, data_url_length,
                                     (char*)out_image->allocation,
                                     &unescaped_length);
  iree_string_view_t encoded = iree_make_string_view(
      (const char*)out_image->allocation, unescaped_length);
  if (iree_status_is_ok(status)) {
    if (iree_string_view_consume_prefix(&encoded,
                                        IREE_SV("data:image/png;base64,"))) {
      out_image->format = LOOM_SERVE_MEDIA_IMAGE_FORMAT_PNG;
    } else if (iree_string_view_consume_prefix(
                   &encoded, IREE_SV("data:image/jpeg;base64,")) ||
               iree_string_view_consume_prefix(
                   &encoded, IREE_SV("data:image/jpg;base64,"))) {
      out_image->format = LOOM_SERVE_MEDIA_IMAGE_FORMAT_JPEG;
    } else if (iree_string_view_consume_prefix(
                   &encoded, IREE_SV("data:image/gif;base64,"))) {
      out_image->format = LOOM_SERVE_MEDIA_IMAGE_FORMAT_GIF;
    } else if (iree_string_view_consume_prefix(
                   &encoded, IREE_SV("data:image/webp;base64,"))) {
      out_image->format = LOOM_SERVE_MEDIA_IMAGE_FORMAT_WEBP;
    } else {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "image_url must be a base64 PNG, JPEG, GIF or WebP data URL");
    }
  }
  iree_host_size_t decoded_length = 0;
  if (iree_status_is_ok(status)) {
    decoded_length = iree_base64_decoded_size(encoded);
    if (!decoded_length || decoded_length == IREE_HOST_SIZE_MAX ||
        decoded_length > maximum_length) {
      status = iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "compressed image must contain between 1 and %" PRIhsz " bytes",
          maximum_length);
    }
  }
  if (iree_status_is_ok(status)) {
    // Decoding toward the beginning is safe: each complete input quartet is
    // read before its three output bytes are written, and the data-URL prefix
    // leaves the write cursor strictly behind unread input.
    status = iree_base64_decode(
        encoded, iree_make_byte_span(out_image->allocation, decoded_length),
        &decoded_length);
  }
  if (iree_status_is_ok(status)) {
    out_image->data =
        iree_make_const_byte_span(out_image->allocation, decoded_length);
  } else {
    loom_serve_media_compressed_image_deinitialize(out_image);
  }
  return status;
}

static inline uint16_t loom_serve_media_load_le_u16(const uint8_t* data) {
  return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

static inline uint32_t loom_serve_media_load_le_u24(const uint8_t* data) {
  return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
         ((uint32_t)data[2] << 16);
}

static inline uint32_t loom_serve_media_load_le_u32(const uint8_t* data) {
  return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
         ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

static inline uint16_t loom_serve_media_load_be_u16(const uint8_t* data) {
  return ((uint16_t)data[0] << 8) | (uint16_t)data[1];
}

static inline uint32_t loom_serve_media_load_be_u32(const uint8_t* data) {
  return ((uint32_t)data[0] << 24) | ((uint32_t)data[1] << 16) |
         ((uint32_t)data[2] << 8) | (uint32_t)data[3];
}

static bool loom_serve_media_tiff_read_u16(iree_const_byte_span_t data,
                                           bool little_endian,
                                           iree_host_size_t offset,
                                           uint16_t* out_value) {
  if (offset > data.data_length || data.data_length - offset < 2) {
    return false;
  }
  *out_value = little_endian ? loom_serve_media_load_le_u16(data.data + offset)
                             : loom_serve_media_load_be_u16(data.data + offset);
  return true;
}

static bool loom_serve_media_tiff_read_u32(iree_const_byte_span_t data,
                                           bool little_endian,
                                           iree_host_size_t offset,
                                           uint32_t* out_value) {
  if (offset > data.data_length || data.data_length - offset < 4) {
    return false;
  }
  *out_value = little_endian ? loom_serve_media_load_le_u32(data.data + offset)
                             : loom_serve_media_load_be_u32(data.data + offset);
  return true;
}

// Malformed optional metadata has the same effect as absent metadata. Codec
// validation still owns the image byte stream itself.
static uint8_t loom_serve_media_exif_orientation(iree_const_byte_span_t data) {
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
  if (!loom_serve_media_tiff_read_u16(data, little_endian, 2, &magic) ||
      magic != 42 ||
      !loom_serve_media_tiff_read_u32(data, little_endian, 4,
                                      &directory_offset)) {
    return 1;
  }
  uint16_t entry_count = 0;
  if (!loom_serve_media_tiff_read_u16(data, little_endian, directory_offset,
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
    if (!loom_serve_media_tiff_read_u16(data, little_endian, entry_offset,
                                        &tag) ||
        !loom_serve_media_tiff_read_u16(data, little_endian, entry_offset + 2,
                                        &type) ||
        !loom_serve_media_tiff_read_u32(data, little_endian, entry_offset + 4,
                                        &count)) {
      return 1;
    }
    if (tag == 0x0112 && type == 3 && count == 1) {
      uint16_t orientation = 0;
      if (loom_serve_media_tiff_read_u16(data, little_endian, entry_offset + 8,
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

static bool loom_serve_media_chunk(iree_const_byte_span_t data,
                                   iree_host_size_t* offset,
                                   iree_host_size_t header_length,
                                   uint32_t length,
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

static iree_status_t loom_serve_media_png_inspect(
    iree_const_byte_span_t data, loom_serve_media_image_info_t* out_info) {
  static const uint8_t signature[] = {0x89, 'P',  'N',  'G',
                                      0x0d, 0x0a, 0x1a, 0x0a};
  if (data.data_length < sizeof(signature) ||
      memcmp(data.data, signature, sizeof(signature)) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image/png data has no PNG signature");
  }
  png_image image = {.version = PNG_IMAGE_VERSION};
  if (!png_image_begin_read_from_memory(&image, data.data, data.data_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid PNG: %s",
                            image.message);
  }
  out_info->width = image.width;
  out_info->height = image.height;
  png_image_free(&image);

  iree_host_size_t offset = sizeof(signature);
  while (offset < data.data_length) {
    if (data.data_length - offset < 12) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "truncated PNG chunk");
    }
    const uint32_t length = loom_serve_media_load_be_u32(data.data + offset);
    const uint8_t* type = data.data + offset + 4;
    iree_const_byte_span_t chunk;
    if (!loom_serve_media_chunk(data, &offset, 8, length, &chunk) ||
        data.data_length - offset < 4) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "truncated PNG chunk payload");
    }
    offset += 4;  // CRC; libpng validates it during realization.
    if (memcmp(type, "acTL", 4) == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "animated PNG inputs are not supported");
    }
    if (memcmp(type, "eXIf", 4) == 0) {
      out_info->orientation = loom_serve_media_exif_orientation(chunk);
    }
    if (memcmp(type, "IEND", 4) == 0) {
      break;
    }
  }
  return iree_ok_status();
}

typedef struct loom_serve_media_jpeg_error_t {
  struct jpeg_error_mgr base;
  jmp_buf jump;
  char message[JMSG_LENGTH_MAX];
} loom_serve_media_jpeg_error_t;

static void loom_serve_media_jpeg_error_exit(j_common_ptr common) {
  loom_serve_media_jpeg_error_t* error =
      (loom_serve_media_jpeg_error_t*)common->err;
  (*common->err->format_message)(common, error->message);
  longjmp(error->jump, 1);
}

static iree_status_t loom_serve_media_jpeg_inspect(
    iree_const_byte_span_t data, loom_serve_media_image_info_t* out_info) {
  if (data.data_length < 2 || data.data[0] != 0xff || data.data[1] != 0xd8) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image/jpeg data has no JPEG signature");
  }
  struct jpeg_decompress_struct decoder;
  loom_serve_media_jpeg_error_t error = {0};
  decoder.err = jpeg_std_error(&error.base);
  error.base.error_exit = loom_serve_media_jpeg_error_exit;
  volatile bool created = false;
  if (setjmp(error.jump)) {
    if (created) {
      jpeg_destroy_decompress(&decoder);
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid JPEG: %s",
                            error.message);
  }
  jpeg_create_decompress(&decoder);
  created = true;
  jpeg_save_markers(&decoder, JPEG_APP0 + 1, 0xffff);
  jpeg_mem_src(&decoder, data.data, data.data_length);
  jpeg_read_header(&decoder, TRUE);
  out_info->width = decoder.image_width;
  out_info->height = decoder.image_height;
  for (jpeg_saved_marker_ptr marker = decoder.marker_list; marker;
       marker = marker->next) {
    if (marker->marker == JPEG_APP0 + 1) {
      out_info->orientation = loom_serve_media_exif_orientation(
          iree_make_const_byte_span(marker->data, marker->data_length));
      if (out_info->orientation != 1) {
        break;
      }
    }
  }
  jpeg_destroy_decompress(&decoder);
  return iree_ok_status();
}

static bool loom_serve_media_gif_skip_subblocks(iree_const_byte_span_t data,
                                                iree_host_size_t* offset) {
  while (*offset < data.data_length) {
    const uint8_t length = data.data[(*offset)++];
    if (!length) {
      return true;
    }
    if (length > data.data_length - *offset) {
      return false;
    }
    *offset += length;
  }
  return false;
}

static iree_status_t loom_serve_media_gif_inspect(
    iree_const_byte_span_t data, loom_serve_media_image_info_t* out_info) {
  if (data.data_length < 13 || (memcmp(data.data, "GIF87a", 6) != 0 &&
                                memcmp(data.data, "GIF89a", 6) != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image/gif data has no GIF signature");
  }
  out_info->width = loom_serve_media_load_le_u16(data.data + 6);
  out_info->height = loom_serve_media_load_le_u16(data.data + 8);
  iree_host_size_t offset = 13;
  if (data.data[10] & 0x80) {
    const iree_host_size_t color_bytes = 3u << ((data.data[10] & 0x07) + 1u);
    if (color_bytes > data.data_length - offset) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "truncated GIF color table");
    }
    offset += color_bytes;
  }
  uint32_t image_count = 0;
  bool complete = false;
  while (offset < data.data_length && !complete) {
    const uint8_t marker = data.data[offset++];
    switch (marker) {
      case 0x21:  // Extension.
        if (offset >= data.data_length) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "truncated GIF extension");
        }
        ++offset;
        if (!loom_serve_media_gif_skip_subblocks(data, &offset)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "truncated GIF extension data");
        }
        break;
      case 0x2c: {  // Image descriptor.
        ++image_count;
        if (image_count > 1) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "animated GIF inputs are not supported");
        }
        if (data.data_length - offset < 9) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "truncated GIF image descriptor");
        }
        const uint8_t flags = data.data[offset + 8];
        offset += 9;
        if (flags & 0x80) {
          const iree_host_size_t color_bytes = 3u << ((flags & 0x07) + 1u);
          if (color_bytes > data.data_length - offset) {
            return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "truncated GIF local color table");
          }
          offset += color_bytes;
        }
        if (offset >= data.data_length) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "truncated GIF image data");
        }
        ++offset;  // LZW minimum code size.
        if (!loom_serve_media_gif_skip_subblocks(data, &offset)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "truncated GIF image data");
        }
        break;
      }
      case 0x3b:
        complete = true;
        break;
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid GIF block marker 0x%02x", marker);
    }
  }
  if (!complete || image_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "GIF must contain exactly one complete image");
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_media_webp_inspect(
    iree_const_byte_span_t data, loom_serve_media_image_info_t* out_info) {
  if (data.data_length < 12 || memcmp(data.data, "RIFF", 4) != 0 ||
      memcmp(data.data + 8, "WEBP", 4) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image/webp data has no WebP signature");
  }
  WebPBitstreamFeatures features;
  const VP8StatusCode feature_status =
      WebPGetFeatures(data.data, data.data_length, &features);
  if (feature_status != VP8_STATUS_OK) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid WebP feature header (%d)",
                            (int)feature_status);
  }
  if (features.has_animation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "animated WebP inputs are not supported");
  }
  out_info->width = (uint32_t)features.width;
  out_info->height = (uint32_t)features.height;

  iree_host_size_t offset = 12;
  while (data.data_length - offset >= 8) {
    const uint8_t* type = data.data + offset;
    const uint32_t length =
        loom_serve_media_load_le_u32(data.data + offset + 4);
    iree_const_byte_span_t chunk;
    if (!loom_serve_media_chunk(data, &offset, 8, length, &chunk)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "truncated WebP chunk");
    }
    if (length & 1) {
      if (offset == data.data_length) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "truncated WebP chunk padding");
      }
      ++offset;
    }
    if (memcmp(type, "ANIM", 4) == 0 || memcmp(type, "ANMF", 4) == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "animated WebP inputs are not supported");
    }
    if (memcmp(type, "EXIF", 4) == 0) {
      out_info->orientation = loom_serve_media_exif_orientation(chunk);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_media_image_inspect_compressed(
    const loom_serve_media_compressed_image_t* compressed,
    const loom_serve_media_image_limits_t* limits,
    loom_serve_media_image_info_t* out_info) {
  memset(out_info, 0, sizeof(*out_info));
  out_info->format = compressed->format;
  out_info->orientation = 1;
  out_info->compressed_length = compressed->data.data_length;
  loom_serve_media_digest_compute(compressed->data, &out_info->digest);
  iree_status_t status = iree_ok_status();
  switch (compressed->format) {
    case LOOM_SERVE_MEDIA_IMAGE_FORMAT_PNG:
      status = loom_serve_media_png_inspect(compressed->data, out_info);
      break;
    case LOOM_SERVE_MEDIA_IMAGE_FORMAT_JPEG:
      status = loom_serve_media_jpeg_inspect(compressed->data, out_info);
      break;
    case LOOM_SERVE_MEDIA_IMAGE_FORMAT_GIF:
      status = loom_serve_media_gif_inspect(compressed->data, out_info);
      break;
    case LOOM_SERVE_MEDIA_IMAGE_FORMAT_WEBP:
      status = loom_serve_media_webp_inspect(compressed->data, out_info);
      break;
  }
  uint64_t pixel_count = 0;
  if (iree_status_is_ok(status)) {
    pixel_count = (uint64_t)out_info->width * (uint64_t)out_info->height;
    if (!out_info->width || !out_info->height ||
        out_info->width > limits->width || out_info->height > limits->height ||
        pixel_count > limits->pixels) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "image geometry %" PRIu32 "x%" PRIu32
                                " exceeds the configured bound",
                                out_info->width, out_info->height);
    }
  }
  if (iree_status_is_ok(status)) {
    const bool swaps_axes = out_info->orientation >= 5;
    out_info->display_width = swaps_axes ? out_info->height : out_info->width;
    out_info->display_height = swaps_axes ? out_info->width : out_info->height;
  } else {
    memset(out_info, 0, sizeof(*out_info));
  }
  return status;
}

iree_status_t loom_serve_media_image_inspect(
    iree_string_view_t escaped_data_url,
    const loom_serve_media_image_limits_t* limits,
    iree_allocator_t host_allocator, loom_serve_media_image_info_t* out_info) {
  memset(out_info, 0, sizeof(*out_info));
  loom_serve_media_compressed_image_t compressed;
  IREE_RETURN_IF_ERROR(loom_serve_media_image_data_url_decode(
      escaped_data_url, limits->compressed_bytes, host_allocator, &compressed));
  iree_status_t status =
      loom_serve_media_image_inspect_compressed(&compressed, limits, out_info);
  loom_serve_media_compressed_image_deinitialize(&compressed);
  return status;
}

static iree_status_t loom_serve_media_png_decode(
    iree_const_byte_span_t data, const loom_serve_media_image_info_t* info,
    iree_byte_span_t pixels) {
  png_image image = {.version = PNG_IMAGE_VERSION};
  if (!png_image_begin_read_from_memory(&image, data.data, data.data_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid PNG: %s",
                            image.message);
  }
  image.format = PNG_FORMAT_RGB;
  if (image.width != info->width || image.height != info->height ||
      PNG_IMAGE_SIZE(image) != pixels.data_length) {
    png_image_free(&image);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PNG geometry changed after inspection");
  }
  if (!png_image_finish_read(&image, NULL, pixels.data, 0, NULL)) {
    const iree_status_t status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "invalid PNG pixels: %s", image.message);
    png_image_free(&image);
    return status;
  }
  png_image_free(&image);
  return iree_ok_status();
}

static iree_status_t loom_serve_media_jpeg_decode(
    iree_const_byte_span_t data, const loom_serve_media_image_info_t* info,
    iree_byte_span_t pixels) {
  struct jpeg_decompress_struct decoder;
  loom_serve_media_jpeg_error_t error = {0};
  decoder.err = jpeg_std_error(&error.base);
  error.base.error_exit = loom_serve_media_jpeg_error_exit;
  volatile bool created = false;
  if (setjmp(error.jump)) {
    if (created) {
      jpeg_destroy_decompress(&decoder);
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid JPEG pixels: %s", error.message);
  }
  jpeg_create_decompress(&decoder);
  created = true;
  jpeg_mem_src(&decoder, data.data, data.data_length);
  jpeg_read_header(&decoder, TRUE);
  decoder.out_color_space = JCS_RGB;
  jpeg_start_decompress(&decoder);
  if (decoder.output_width != info->width ||
      decoder.output_height != info->height || decoder.output_components != 3) {
    jpeg_destroy_decompress(&decoder);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "JPEG geometry changed after inspection");
  }
  const iree_host_size_t row_length = (iree_host_size_t)info->width * 3;
  while (decoder.output_scanline < decoder.output_height) {
    JSAMPROW row =
        pixels.data + (iree_host_size_t)decoder.output_scanline * row_length;
    if (jpeg_read_scanlines(&decoder, &row, 1) != 1) {
      jpeg_destroy_decompress(&decoder);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "JPEG scanline decode stopped early");
    }
  }
  jpeg_finish_decompress(&decoder);
  jpeg_destroy_decompress(&decoder);
  return iree_ok_status();
}

typedef struct loom_serve_media_gif_reader_t {
  iree_const_byte_span_t data;
  iree_host_size_t offset;
} loom_serve_media_gif_reader_t;

static int loom_serve_media_gif_read(GifFileType* file, GifByteType* output,
                                     int requested_length) {
  loom_serve_media_gif_reader_t* reader = file->UserData;
  const iree_host_size_t length =
      iree_min((iree_host_size_t)requested_length,
               reader->data.data_length - reader->offset);
  memcpy(output, reader->data.data + reader->offset, length);
  reader->offset += length;
  return (int)length;
}

static iree_status_t loom_serve_media_gif_decode(
    iree_const_byte_span_t data, const loom_serve_media_image_info_t* info,
    iree_byte_span_t pixels) {
  loom_serve_media_gif_reader_t reader = {.data = data};
  int error = 0;
  GifFileType* gif = DGifOpen(&reader, loom_serve_media_gif_read, &error);
  if (!gif) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid GIF: %s",
                            GifErrorString(error));
  }
  iree_status_t status = iree_ok_status();
  if (DGifSlurp(gif) == GIF_ERROR) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid GIF pixels: %s",
                         GifErrorString(gif->Error));
  }
  if (iree_status_is_ok(status) &&
      (gif->ImageCount != 1 || gif->SWidth != (GifWord)info->width ||
       gif->SHeight != (GifWord)info->height)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "GIF geometry changed after inspection");
  }
  if (iree_status_is_ok(status)) {
    uint8_t background[3] = {0, 0, 0};
    if (gif->SColorMap && gif->SBackGroundColor >= 0 &&
        gif->SBackGroundColor < gif->SColorMap->ColorCount) {
      const GifColorType color = gif->SColorMap->Colors[gif->SBackGroundColor];
      background[0] = color.Red;
      background[1] = color.Green;
      background[2] = color.Blue;
    }
    for (iree_host_size_t i = 0; i < pixels.data_length; i += 3) {
      memcpy(pixels.data + i, background, sizeof(background));
    }
    const SavedImage* frame = &gif->SavedImages[0];
    const GifImageDesc* descriptor = &frame->ImageDesc;
    const uint32_t left = (uint32_t)descriptor->Left;
    const uint32_t top = (uint32_t)descriptor->Top;
    const uint32_t width = (uint32_t)descriptor->Width;
    const uint32_t height = (uint32_t)descriptor->Height;
    if (left > info->width || top > info->height ||
        width > info->width - left || height > info->height - top) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "GIF frame exceeds its logical canvas");
    }
    const ColorMapObject* colors =
        descriptor->ColorMap ? descriptor->ColorMap : gif->SColorMap;
    if (iree_status_is_ok(status) && !colors) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "GIF frame has no color table");
    }
    static const uint8_t interlace_starts[] = {0, 4, 2, 1};
    static const uint8_t interlace_steps[] = {8, 8, 4, 2};
    uint32_t source_row = 0;
    const uint32_t pass_count = descriptor->Interlace ? 4 : 1;
    for (uint32_t pass = 0; pass < pass_count && iree_status_is_ok(status);
         ++pass) {
      const uint32_t first_row =
          descriptor->Interlace ? interlace_starts[pass] : 0;
      const uint32_t row_step =
          descriptor->Interlace ? interlace_steps[pass] : 1;
      for (uint32_t row = first_row; row < height && iree_status_is_ok(status);
           row += row_step) {
        for (uint32_t column = 0; column < width; ++column) {
          const GifByteType index =
              frame->RasterBits[(iree_host_size_t)source_row * width + column];
          if (index >= colors->ColorCount) {
            status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                      "GIF palette index is out of range");
            break;
          }
          const GifColorType color = colors->Colors[index];
          uint8_t* pixel =
              pixels.data +
              ((iree_host_size_t)(top + row) * info->width + left + column) * 3;
          pixel[0] = color.Red;
          pixel[1] = color.Green;
          pixel[2] = color.Blue;
        }
        ++source_row;
      }
    }
  }
  const int close_result = DGifCloseFile(gif, &error);
  if (iree_status_is_ok(status) && close_result == GIF_ERROR) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "GIF close failed: %s", GifErrorString(error));
  }
  return status;
}

static iree_status_t loom_serve_media_webp_decode(
    iree_const_byte_span_t data, const loom_serve_media_image_info_t* info,
    iree_byte_span_t pixels) {
  const int stride = (int)(info->width * 3u);
  if (!WebPDecodeRGBInto(data.data, data.data_length, pixels.data,
                         pixels.data_length, stride)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid WebP pixel payload");
  }
  return iree_ok_status();
}

static bool loom_serve_media_image_info_equal(
    const loom_serve_media_image_info_t* lhs,
    const loom_serve_media_image_info_t* rhs) {
  return lhs->format == rhs->format && lhs->width == rhs->width &&
         lhs->height == rhs->height &&
         lhs->display_width == rhs->display_width &&
         lhs->display_height == rhs->display_height &&
         lhs->orientation == rhs->orientation &&
         lhs->compressed_length == rhs->compressed_length &&
         loom_serve_media_digest_equal(&lhs->digest, &rhs->digest);
}

static iree_status_t loom_serve_media_image_pixel_length(
    const loom_serve_media_image_info_t* info,
    iree_host_size_t* out_pixel_length) {
  *out_pixel_length = 0;
  iree_host_size_t pixel_count = 0;
  if (!iree_host_size_checked_mul((iree_host_size_t)info->width,
                                  (iree_host_size_t)info->height,
                                  &pixel_count) ||
      !iree_host_size_checked_mul(pixel_count, 3, out_pixel_length)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "image pixel storage size overflow");
  }
  return iree_ok_status();
}

iree_status_t loom_serve_media_image_decode_into(
    iree_string_view_t escaped_data_url,
    const loom_serve_media_image_limits_t* limits,
    const loom_serve_media_image_info_t* expected_info, iree_byte_span_t pixels,
    iree_allocator_t host_allocator) {
  loom_serve_media_compressed_image_t compressed;
  IREE_RETURN_IF_ERROR(loom_serve_media_image_data_url_decode(
      escaped_data_url, limits->compressed_bytes, host_allocator, &compressed));
  loom_serve_media_image_info_t observed_info;
  iree_status_t status = loom_serve_media_image_inspect_compressed(
      &compressed, limits, &observed_info);
  if (iree_status_is_ok(status) &&
      !loom_serve_media_image_info_equal(expected_info, &observed_info)) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "image bytes changed after admission inspection");
  }
  iree_host_size_t pixel_length = 0;
  if (iree_status_is_ok(status)) {
    status = loom_serve_media_image_pixel_length(&observed_info, &pixel_length);
  }
  if (iree_status_is_ok(status) && pixels.data_length != pixel_length) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "RGB8 destination has %zu bytes but image requires %zu",
        pixels.data_length, pixel_length);
  }
  if (iree_status_is_ok(status)) {
    switch (observed_info.format) {
      case LOOM_SERVE_MEDIA_IMAGE_FORMAT_PNG:
        status = loom_serve_media_png_decode(compressed.data, &observed_info,
                                             pixels);
        break;
      case LOOM_SERVE_MEDIA_IMAGE_FORMAT_JPEG:
        status = loom_serve_media_jpeg_decode(compressed.data, &observed_info,
                                              pixels);
        break;
      case LOOM_SERVE_MEDIA_IMAGE_FORMAT_GIF:
        status = loom_serve_media_gif_decode(compressed.data, &observed_info,
                                             pixels);
        break;
      case LOOM_SERVE_MEDIA_IMAGE_FORMAT_WEBP:
        status = loom_serve_media_webp_decode(compressed.data, &observed_info,
                                              pixels);
        break;
    }
  }
  loom_serve_media_compressed_image_deinitialize(&compressed);
  return status;
}

iree_status_t loom_serve_media_image_decode(
    iree_string_view_t escaped_data_url,
    const loom_serve_media_image_limits_t* limits,
    const loom_serve_media_image_info_t* expected_info,
    iree_allocator_t host_allocator, loom_serve_media_image_t* out_image) {
  memset(out_image, 0, sizeof(*out_image));
  out_image->host_allocator = host_allocator;
  iree_host_size_t pixel_length = 0;
  iree_status_t status =
      loom_serve_media_image_pixel_length(expected_info, &pixel_length);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, pixel_length,
                                   (void**)&out_image->pixels.data);
    out_image->pixels.data_length = pixel_length;
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_media_image_decode_into(
        escaped_data_url, limits, expected_info, out_image->pixels,
        host_allocator);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_media_image_deinitialize(out_image);
  } else {
    out_image->info = *expected_info;
  }
  return status;
}

void loom_serve_media_image_deinitialize(loom_serve_media_image_t* image) {
  iree_allocator_free(image->host_allocator, image->pixels.data);
  memset(image, 0, sizeof(*image));
}
