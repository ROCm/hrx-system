// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/serve/media/image_codec.h"

// PAM's defined visual tuples share one packed raster. Custom tuples have no
// RGB interpretation. Samples retain their transfer function and straight
// alpha.
typedef struct loom_serve_media_pam_header_t {
  // Encoded image width in pixels.
  uint32_t width;
  // Encoded image height in pixels.
  uint32_t height;
  // Number of samples per encoded pixel.
  uint32_t depth;
  // Inclusive sample bound, in [1, 65535].
  uint32_t maximum;
  // Raster borrowed from the immutable encoded image.
  iree_const_byte_span_t raster;
} loom_serve_media_pam_header_t;

static iree_status_t loom_serve_media_pam_decimal(iree_string_view_t value,
                                                  uint32_t* out_value) {
  uint32_t number = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < value.size && iree_status_is_ok(status);
       ++i) {
    const uint32_t digit = (uint32_t)(uint8_t)value.data[i] - '0';
    if (digit > 9 || number > (UINT32_MAX - digit) / 10) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "PAM field requires an unsigned decimal integer");
    } else {
      number = number * 10 + digit;
    }
  }
  if (iree_status_is_ok(status) && !number) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "PAM dimensions, depth and maximum must be positive");
  }
  *out_value = number;
  return status;
}

static iree_status_t loom_serve_media_pam_header(
    iree_const_byte_span_t data, loom_serve_media_pam_header_t* out_header) {
  memset(out_header, 0, sizeof(*out_header));
  iree_string_view_t remaining =
      iree_make_string_view((const char*)data.data, data.data_length);
  if (!iree_string_view_consume_prefix(&remaining, IREE_SV("P7\n"))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "image/x-portable-arbitrarymap data has no PAM signature");
  }
  static const iree_string_view_t field_names[] = {
      IREE_SVL("WIDTH"), IREE_SVL("HEIGHT"), IREE_SVL("DEPTH"),
      IREE_SVL("MAXVAL")};
  uint32_t* fields[] = {&out_header->width, &out_header->height,
                        &out_header->depth, &out_header->maximum};
  uint32_t seen = 0;
  iree_string_view_t tuple = iree_string_view_empty();
  bool ended = false;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && !ended) {
    iree_string_view_t line;
    if (iree_string_view_split(remaining, '\n', &line, &remaining) < 0) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "PAM header has no complete ENDHDR line");
      break;
    }
    line = iree_string_view_trim(line);
    if (!line.size || line.data[0] == '#') {
      continue;
    }
    const iree_host_size_t separator =
        iree_string_view_find_first_of(line, IREE_SV(" \t\r\v\f"), 0);
    const iree_string_view_t name = iree_string_view_substr(line, 0, separator);
    const iree_string_view_t value = iree_string_view_trim(
        iree_string_view_substr(line, name.size, IREE_STRING_VIEW_NPOS));
    if (iree_string_view_equal(name, IREE_SV("ENDHDR"))) {
      ended = true;
      if (value.size) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "PAM ENDHDR must have no value");
      }
    } else if (iree_string_view_equal(name, IREE_SV("TUPLTYPE"))) {
      if (tuple.size || !value.size) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "PAM requires one visual tuple type");
      } else {
        tuple = value;
      }
    } else {
      iree_host_size_t field = 0;
      while (field < IREE_ARRAYSIZE(field_names) &&
             !iree_string_view_equal(name, field_names[field])) {
        ++field;
      }
      if (field == IREE_ARRAYSIZE(field_names) || (seen & (1u << field))) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unknown or repeated PAM header field");
      } else {
        seen |= 1u << field;
        status = loom_serve_media_pam_decimal(value, fields[field]);
      }
    }
  }
  if (!iree_status_is_ok(status)) {
    return status;
  }
  if (seen != 15 || !tuple.size || out_header->maximum > 65535) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PAM requires geometry, a visual tuple and maximum <= 65535");
  }
  const bool alpha = iree_string_view_consume_suffix(&tuple, IREE_SV("_ALPHA"));
  uint32_t channels = 0;
  if (iree_string_view_equal(tuple, IREE_SV("RGB"))) {
    channels = 3;
  } else if (iree_string_view_equal(tuple, IREE_SV("GRAYSCALE"))) {
    channels = 1;
  } else if (iree_string_view_equal(tuple, IREE_SV("BLACKANDWHITE")) &&
             out_header->maximum == 1) {
    channels = 1;
  }
  if (!channels || out_header->depth != channels + (uint32_t)alpha) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported PAM visual tuple or inconsistent depth");
  }
  out_header->raster =
      iree_make_const_byte_span(remaining.data, remaining.size);
  return iree_ok_status();
}

static void loom_serve_media_pam_inspect_batch(
    iree_host_size_t count, loom_serve_media_image_inspection_t* images) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    memset(&images[i].info, 0, sizeof(images[i].info));
    loom_serve_media_pam_header_t header;
    images[i].status = loom_serve_media_pam_header(images[i].data, &header);
    if (iree_status_is_ok(images[i].status)) {
      images[i].info = (loom_serve_media_image_info_t){
          .format = LOOM_SERVE_MEDIA_IMAGE_FORMAT_PAM,
          .pixel_format = (header.depth & 1)
                              ? LOOM_SERVE_MEDIA_IMAGE_PIXEL_FORMAT_RGB8
                              : LOOM_SERVE_MEDIA_IMAGE_PIXEL_FORMAT_RGBA8,
          .width = header.width,
          .height = header.height,
          .orientation = 1,
      };
    }
  }
}

static iree_status_t loom_serve_media_pam_decode(
    const loom_serve_media_image_decoding_t* image) {
  // Like the other codecs, realization opens the immutable encoded header;
  // the facade's bounds already qualify the caller's exact pixel allocation.
  loom_serve_media_pam_header_t header;
  IREE_RETURN_IF_ERROR(loom_serve_media_pam_header(image->data, &header));
  const iree_host_size_t pixel_count =
      image->pixels.data_length / image->info->pixel_format;
  const iree_host_size_t sample_byte_count = header.maximum < 256 ? 1 : 2;
  iree_host_size_t raster_bytes = 0;
  if (!iree_host_size_checked_mul(pixel_count, header.depth * sample_byte_count,
                                  &raster_bytes) ||
      header.raster.data_length != raster_bytes) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PAM requires exactly one complete image raster");
  }
  if (header.maximum == 255 && header.depth >= 3) {
    memcpy(image->pixels.data, header.raster.data, raster_bytes);
    return iree_ok_status();
  }
  const uint8_t* source = header.raster.data;
  uint8_t* target = image->pixels.data;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < pixel_count && iree_status_is_ok(status);
       ++i) {
    uint8_t samples[4] = {0, 0, 0, 255};
    for (uint32_t channel = 0;
         channel < header.depth && iree_status_is_ok(status); ++channel) {
      uint32_t value = *source++;
      if (sample_byte_count == 2) {
        value = (value << 8) | *source++;
      }
      if (value > header.maximum) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "PAM sample exceeds its declared maximum");
      } else {
        samples[channel] =
            (uint8_t)((value * 255 + header.maximum / 2) / header.maximum);
      }
    }
    if (iree_status_is_ok(status)) {
      if (header.depth <= 2) {
        *target++ = samples[0];
        *target++ = samples[0];
        *target++ = samples[0];
        if (header.depth == 2) {
          *target++ = samples[1];
        }
      } else {
        memcpy(target, samples, header.depth);
        target += header.depth;
      }
    }
  }
  return status;
}

static void loom_serve_media_pam_decode_batch(
    iree_host_size_t count, loom_serve_media_image_decoding_t* images) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    images[i].status = loom_serve_media_pam_decode(&images[i]);
  }
}

const loom_serve_media_image_codec_t loom_serve_media_image_codec_pam = {
    .format = LOOM_SERVE_MEDIA_IMAGE_FORMAT_PAM,
    .mime_type = IREE_SVL("image/x-portable-arbitrarymap"),
    .inspect = loom_serve_media_pam_inspect_batch,
    .decode = loom_serve_media_pam_decode_batch,
};
