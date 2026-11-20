// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <setjmp.h>
#include <stdio.h>
#include <string.h>

#include "iree/codec/image/codec.h"
#include "iree/codec/image/metadata.h"
#include "jpeglib.h"

typedef struct iree_codec_image_jpeg_error_t {
  // libjpeg callback table; first so the callback can recover this owner.
  struct jpeg_error_mgr base;
  // Recovery target for the current libjpeg operation.
  jmp_buf jump;
  // Diagnostic formatted by libjpeg before unwinding.
  char message[JMSG_LENGTH_MAX];
} iree_codec_image_jpeg_error_t;

static void iree_codec_image_jpeg_fail(j_common_ptr common) {
  iree_codec_image_jpeg_error_t* error =
      (iree_codec_image_jpeg_error_t*)common->err;
  (*common->err->format_message)(common, error->message);
  longjmp(error->jump, 1);
}

static void iree_codec_image_jpeg_emit_message(j_common_ptr common,
                                               int message_level) {
  if (message_level < 0) {
    // A decoder warning means the input was malformed even when libjpeg can
    // synthesize partial pixels. Report it through the codec status instead of
    // letting libjpeg write an unsolicited process-level diagnostic.
    iree_codec_image_jpeg_fail(common);
  }
}

static void iree_codec_image_jpeg_error_initialize(
    struct jpeg_decompress_struct* decoder,
    iree_codec_image_jpeg_error_t* error) {
  decoder->err = jpeg_std_error(&error->base);
  error->base.error_exit = iree_codec_image_jpeg_fail;
  error->base.emit_message = iree_codec_image_jpeg_emit_message;
}

static iree_status_t iree_codec_image_jpeg_inspect(
    iree_const_byte_span_t data, iree_codec_image_info_t* out_info) {
  if (data.data_length < 2 || data.data[0] != 0xff || data.data[1] != 0xd8) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image/jpeg data has no JPEG signature");
  }
  struct jpeg_decompress_struct decoder;
  iree_codec_image_jpeg_error_t error = {0};
  iree_codec_image_jpeg_error_initialize(&decoder, &error);
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
      out_info->orientation = iree_codec_image_exif_orientation(
          iree_make_const_byte_span(marker->data, marker->data_length));
      if (out_info->orientation != 1) {
        break;
      }
    }
  }
  jpeg_destroy_decompress(&decoder);
  return iree_ok_status();
}

static iree_status_t iree_codec_image_jpeg_decode(
    iree_const_byte_span_t data, const iree_codec_image_info_t* info,
    iree_byte_span_t pixels) {
  struct jpeg_decompress_struct decoder;
  iree_codec_image_jpeg_error_t error = {0};
  iree_codec_image_jpeg_error_initialize(&decoder, &error);
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

static void iree_codec_image_jpeg_inspect_batch(
    iree_host_size_t count, iree_codec_image_inspection_t* images) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    images[i].info = (iree_codec_image_info_t){
        .format = IREE_CODEC_IMAGE_FORMAT_JPEG,
        .pixel_format = IREE_CODEC_IMAGE_PIXEL_FORMAT_RGB8,
        .orientation = 1,
    };
    images[i].status =
        iree_codec_image_jpeg_inspect(images[i].data, &images[i].info);
    if (!iree_status_is_ok(images[i].status)) {
      memset(&images[i].info, 0, sizeof(images[i].info));
    }
  }
}

static void iree_codec_image_jpeg_decode_batch(
    iree_host_size_t count, iree_codec_image_decoding_t* images) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    images[i].status = iree_codec_image_jpeg_decode(
        images[i].data, images[i].info, images[i].pixels);
  }
}

const iree_codec_image_codec_t iree_codec_image_codec_jpeg = {
    .format = IREE_CODEC_IMAGE_FORMAT_JPEG,
    .mime_type = IREE_SVL("image/jpeg"),
    .inspect = iree_codec_image_jpeg_inspect_batch,
    .decode = iree_codec_image_jpeg_decode_batch,
};
