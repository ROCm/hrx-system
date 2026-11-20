// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <limits.h>
#include <string.h>

#include "iree/codec/image/codec.h"
#include "iree/codec/image/metadata.h"
#include "webp/decode.h"

static iree_status_t iree_codec_image_webp_inspect(
    iree_const_byte_span_t data, iree_codec_image_info_t* out_info) {
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
  out_info->pixel_format = features.has_alpha
                               ? IREE_CODEC_IMAGE_PIXEL_FORMAT_RGBA8
                               : IREE_CODEC_IMAGE_PIXEL_FORMAT_RGB8;

  iree_host_size_t offset = 12;
  while (data.data_length - offset >= 8) {
    const uint8_t* type = data.data + offset;
    const uint32_t length =
        iree_codec_image_load_le_u32(data.data + offset + 4);
    iree_const_byte_span_t chunk;
    if (!iree_codec_image_chunk(data, &offset, 8, length, &chunk)) {
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
      out_info->orientation = iree_codec_image_exif_orientation(chunk);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_codec_image_webp_decode(
    iree_const_byte_span_t data, const iree_codec_image_info_t* info,
    iree_byte_span_t pixels) {
  const uint64_t stride = (uint64_t)info->width * info->pixel_format;
  if (stride > INT_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "WebP row stride exceeds decoder range");
  }
  uint8_t* decoded =
      info->pixel_format == IREE_CODEC_IMAGE_PIXEL_FORMAT_RGBA8
          ? WebPDecodeRGBAInto(data.data, data.data_length, pixels.data,
                               pixels.data_length, (int)stride)
          : WebPDecodeRGBInto(data.data, data.data_length, pixels.data,
                              pixels.data_length, (int)stride);
  if (!decoded) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid WebP pixel payload");
  }
  return iree_ok_status();
}

static void iree_codec_image_webp_inspect_batch(
    iree_host_size_t count, iree_codec_image_inspection_t* images) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    images[i].info = (iree_codec_image_info_t){
        .format = IREE_CODEC_IMAGE_FORMAT_WEBP,
        .pixel_format = IREE_CODEC_IMAGE_PIXEL_FORMAT_RGB8,
        .orientation = 1,
    };
    images[i].status =
        iree_codec_image_webp_inspect(images[i].data, &images[i].info);
    if (!iree_status_is_ok(images[i].status)) {
      memset(&images[i].info, 0, sizeof(images[i].info));
    }
  }
}

static void iree_codec_image_webp_decode_batch(
    iree_host_size_t count, iree_codec_image_decoding_t* images) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    images[i].status = iree_codec_image_webp_decode(
        images[i].data, images[i].info, images[i].pixels);
  }
}

const iree_codec_image_codec_t iree_codec_image_codec_webp = {
    .format = IREE_CODEC_IMAGE_FORMAT_WEBP,
    .mime_type = IREE_SVL("image/webp"),
    .inspect = iree_codec_image_webp_inspect_batch,
    .decode = iree_codec_image_webp_decode_batch,
};
