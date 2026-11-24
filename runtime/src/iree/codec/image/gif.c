// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "gif_lib.h"
#include "iree/codec/image/codec.h"
#include "iree/codec/image/metadata.h"

static bool iree_codec_image_gif_skip_subblocks(iree_const_byte_span_t data,
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

static iree_status_t iree_codec_image_gif_inspect(
    iree_const_byte_span_t data, iree_codec_image_info_t* out_info) {
  if (data.data_length < 13 || (memcmp(data.data, "GIF87a", 6) != 0 &&
                                memcmp(data.data, "GIF89a", 6) != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image/gif data has no GIF signature");
  }
  out_info->width = iree_codec_image_load_le_u16(data.data + 6);
  out_info->height = iree_codec_image_load_le_u16(data.data + 8);
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
        if (data.data[offset] == GRAPHICS_EXT_FUNC_CODE &&
            data.data_length - offset >= 7 && data.data[offset + 1] == 4 &&
            (data.data[offset + 2] & 1)) {
          out_info->pixel_format = IREE_CODEC_IMAGE_PIXEL_FORMAT_RGBA8;
        }
        ++offset;
        if (!iree_codec_image_gif_skip_subblocks(data, &offset)) {
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
        const uint32_t left = iree_codec_image_load_le_u16(data.data + offset);
        const uint32_t top =
            iree_codec_image_load_le_u16(data.data + offset + 2);
        const uint32_t width =
            iree_codec_image_load_le_u16(data.data + offset + 4);
        const uint32_t height =
            iree_codec_image_load_le_u16(data.data + offset + 6);
        if (!width || !height || left > out_info->width ||
            top > out_info->height || width > out_info->width - left ||
            height > out_info->height - top) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "GIF frame exceeds its logical canvas");
        }
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
        if (!iree_codec_image_gif_skip_subblocks(data, &offset)) {
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

typedef struct iree_codec_image_gif_reader_t {
  // Immutable compressed stream borrowed through DGifCloseFile.
  iree_const_byte_span_t data;
  // Next unread byte in data.
  iree_host_size_t offset;
} iree_codec_image_gif_reader_t;

static int iree_codec_image_gif_read(GifFileType* file, GifByteType* output,
                                     int requested_length) {
  iree_codec_image_gif_reader_t* reader = file->UserData;
  const iree_host_size_t length =
      iree_min((iree_host_size_t)requested_length,
               reader->data.data_length - reader->offset);
  memcpy(output, reader->data.data + reader->offset, length);
  reader->offset += length;
  return (int)length;
}

static iree_status_t iree_codec_image_gif_decode(
    iree_const_byte_span_t data, const iree_codec_image_info_t* info,
    iree_byte_span_t pixels) {
  iree_codec_image_gif_reader_t reader = {.data = data};
  int error = 0;
  GifFileType* gif = DGifOpen(&reader, iree_codec_image_gif_read, &error);
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
    const uint32_t channels = info->pixel_format;
    uint8_t background[4] = {0, 0, 0, 0};
    if (gif->SColorMap && gif->SBackGroundColor >= 0 &&
        gif->SBackGroundColor < gif->SColorMap->ColorCount) {
      const GifColorType color = gif->SColorMap->Colors[gif->SBackGroundColor];
      background[0] = color.Red;
      background[1] = color.Green;
      background[2] = color.Blue;
    }
    for (iree_host_size_t i = 0; i < pixels.data_length; i += channels) {
      memcpy(pixels.data + i, background, channels);
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
    GraphicsControlBlock control;
    const int transparent = DGifSavedExtensionToGCB(gif, 0, &control) == GIF_OK
                                ? control.TransparentColor
                                : NO_TRANSPARENT_COLOR;
    // DGifSlurp already places interlaced passes in final row order.
    for (uint32_t row = 0; row < height && iree_status_is_ok(status); ++row) {
      for (uint32_t column = 0; column < width; ++column) {
        const GifByteType index =
            frame->RasterBits[(iree_host_size_t)row * width + column];
        if (index >= colors->ColorCount) {
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "GIF palette index is out of range");
          break;
        }
        const GifColorType color = colors->Colors[index];
        uint8_t* pixel =
            pixels.data +
            ((iree_host_size_t)(top + row) * info->width + left + column) *
                channels;
        pixel[0] = color.Red;
        pixel[1] = color.Green;
        pixel[2] = color.Blue;
        if (channels == 4) {
          pixel[3] = index == transparent ? 0 : 255;
        }
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

static void iree_codec_image_gif_inspect_batch(
    iree_host_size_t count, iree_codec_image_inspection_t* images) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    images[i].info = (iree_codec_image_info_t){
        .format = IREE_CODEC_IMAGE_FORMAT_GIF,
        .pixel_format = IREE_CODEC_IMAGE_PIXEL_FORMAT_RGB8,
        .orientation = 1,
    };
    images[i].status =
        iree_codec_image_gif_inspect(images[i].data, &images[i].info);
    if (!iree_status_is_ok(images[i].status)) {
      memset(&images[i].info, 0, sizeof(images[i].info));
    }
  }
}

static void iree_codec_image_gif_decode_batch(
    iree_host_size_t count, iree_codec_image_decoding_t* images) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    images[i].status = iree_codec_image_gif_decode(
        images[i].data, images[i].info, images[i].pixels);
  }
}

const iree_codec_image_codec_t iree_codec_image_codec_gif = {
    .format = IREE_CODEC_IMAGE_FORMAT_GIF,
    .mime_type = IREE_SVL("image/gif"),
    .inspect = iree_codec_image_gif_inspect_batch,
    .decode = iree_codec_image_gif_decode_batch,
};
