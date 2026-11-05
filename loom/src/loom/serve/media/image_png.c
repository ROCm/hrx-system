// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <setjmp.h>
#include <string.h>

#include "loom/serve/media/image_codec.h"
#include "loom/serve/media/image_metadata.h"
#include "png.h"

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
  out_info->pixel_format = (image.format & PNG_FORMAT_FLAG_ALPHA)
                               ? LOOM_SERVE_MEDIA_IMAGE_PIXEL_FORMAT_RGBA8
                               : LOOM_SERVE_MEDIA_IMAGE_PIXEL_FORMAT_RGB8;
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

typedef struct loom_serve_media_png_reader_t {
  // Immutable compressed stream borrowed through png_read_end.
  iree_const_byte_span_t data;
  // Next unread byte in data.
  iree_host_size_t offset;
} loom_serve_media_png_reader_t;

typedef struct loom_serve_media_png_error_t {
  // Diagnostic supplied by libpng before unwinding.
  char message[128];
} loom_serve_media_png_error_t;

static void loom_serve_media_png_fail(png_structp png,
                                      png_const_charp message) {
  loom_serve_media_png_error_t* error = png_get_error_ptr(png);
  const size_t length = iree_min(strlen(message), sizeof(error->message) - 1);
  memcpy(error->message, message, length);
  error->message[length] = 0;
  png_longjmp(png, 1);
}

static void loom_serve_media_png_read(png_structp png, png_bytep output,
                                      png_size_t length) {
  loom_serve_media_png_reader_t* reader = png_get_io_ptr(png);
  if (length > reader->data.data_length - reader->offset) {
    png_error(png, "truncated PNG input");
  }
  memcpy(output, reader->data.data + reader->offset, length);
  reader->offset += length;
}

static iree_status_t loom_serve_media_png_decode(
    iree_const_byte_span_t data, const loom_serve_media_image_info_t* info,
    iree_byte_span_t pixels) {
  loom_serve_media_png_error_t error = {0};
  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, &error,
                                           loom_serve_media_png_fail,
                                           loom_serve_media_png_fail);
  if (!png) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "cannot allocate PNG decoder");
  }
  png_infop header = png_create_info_struct(png);
  if (!header) {
    png_destroy_read_struct(&png, NULL, NULL);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "cannot allocate PNG header");
  }
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_read_struct(&png, &header, NULL);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid PNG pixel stream: %s", error.message);
  }
  loom_serve_media_png_reader_t reader = {.data = data};
  png_set_read_fn(png, &reader, loom_serve_media_png_read);
  png_read_info(png, header);
  if (png_get_image_width(png, header) != info->width ||
      png_get_image_height(png, header) != info->height) {
    png_error(png, "PNG geometry changed after inspection");
  }
  const int color = png_get_color_type(png, header);
  const int depth = png_get_bit_depth(png, header);
  if (color == PNG_COLOR_TYPE_PALETTE) {
    png_set_palette_to_rgb(png);
  }
  if (color == PNG_COLOR_TYPE_GRAY && depth < 8) {
    png_set_expand_gray_1_2_4_to_8(png);
  }
  if (png_get_valid(png, header, PNG_INFO_tRNS)) {
    png_set_tRNS_to_alpha(png);
  }
  if (depth == 16) {
    png_set_strip_16(png);
  }
  if (color == PNG_COLOR_TYPE_GRAY || color == PNG_COLOR_TYPE_GRAY_ALPHA) {
    png_set_gray_to_rgb(png);
  }
  const int passes = png_set_interlace_handling(png);
  png_read_update_info(png, header);
  const iree_host_size_t row_length =
      (iree_host_size_t)info->width * info->pixel_format;
  if (png_get_rowbytes(png, header) != row_length ||
      png_get_channels(png, header) != info->pixel_format) {
    png_error(png, "PNG channel layout changed after inspection");
  }
  // Interlaced passes merge into previously written samples in the same plane.
  if (passes > 1) {
    memset(pixels.data, 0, pixels.data_length);
  }
  for (int pass = 0; pass < passes; ++pass) {
    for (uint32_t row = 0; row < info->height; ++row) {
      png_read_row(png, pixels.data + row * row_length, NULL);
    }
  }
  png_read_end(png, header);
  png_destroy_read_struct(&png, &header, NULL);
  return iree_ok_status();
}

static void loom_serve_media_png_inspect_batch(
    iree_host_size_t count, loom_serve_media_image_inspection_t* images) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    images[i].info = (loom_serve_media_image_info_t){
        .format = LOOM_SERVE_MEDIA_IMAGE_FORMAT_PNG,
        .pixel_format = LOOM_SERVE_MEDIA_IMAGE_PIXEL_FORMAT_RGB8,
        .orientation = 1,
    };
    images[i].status =
        loom_serve_media_png_inspect(images[i].data, &images[i].info);
    if (!iree_status_is_ok(images[i].status)) {
      memset(&images[i].info, 0, sizeof(images[i].info));
    }
  }
}

static void loom_serve_media_png_decode_batch(
    iree_host_size_t count, loom_serve_media_image_decoding_t* images) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    images[i].status = loom_serve_media_png_decode(
        images[i].data, images[i].info, images[i].pixels);
  }
}

const loom_serve_media_image_codec_t loom_serve_media_image_codec_png = {
    .format = LOOM_SERVE_MEDIA_IMAGE_FORMAT_PNG,
    .mime_type = IREE_SVL("image/png"),
    .inspect = loom_serve_media_png_inspect_batch,
    .decode = loom_serve_media_png_decode_batch,
};
