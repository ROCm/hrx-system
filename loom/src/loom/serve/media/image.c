// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/media/image.h"

#include <string.h>

#include "iree/base/internal/base64.h"
#include "iree/base/internal/json.h"
#include "loom/serve/media/image_codec.h"

typedef struct loom_serve_media_compressed_image_t {
  // Allocator owning the unescaped compressed payload.
  iree_allocator_t host_allocator;
  // One allocation shared by transport unescaping and base64 decoding.
  uint8_t* allocation;
  // Retained transport allocation size in bytes.
  iree_host_size_t allocation_size;
  // Decoded compressed bytes borrowed from allocation.
  iree_const_byte_span_t data;
  // Format declared by the transport MIME type.
  loom_serve_media_image_format_t format;
} loom_serve_media_compressed_image_t;

static void loom_serve_media_compressed_image_deinitialize(
    loom_serve_media_compressed_image_t* image) {
  iree_allocator_free(image->host_allocator, image->allocation);
  memset(image, 0, sizeof(*image));
}

static iree_status_t loom_serve_media_image_source_decode(
    loom_serve_media_image_source_t source, iree_host_size_t maximum_length,
    iree_allocator_t host_allocator,
    loom_serve_media_compressed_image_t* out_image) {
  memset(out_image, 0, sizeof(*out_image));
  out_image->host_allocator = host_allocator;
  out_image->format = source.format;

  iree_host_size_t transport_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_json_unescape_string(source.value, 0, NULL, &transport_length));
  const iree_host_size_t maximum_encoded_length =
      iree_base64_encoded_size(maximum_length);
  iree_host_size_t maximum_transport_length = 0;
  if (maximum_encoded_length == IREE_HOST_SIZE_MAX ||
      !iree_host_size_checked_add(
          maximum_encoded_length,
          source.format
              ? 0
              : sizeof("data:image/x-portable-arbitrarymap;base64,") - 1,
          &maximum_transport_length) ||
      transport_length > maximum_transport_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "encoded image exceeds the compressed byte bound");
  }
  if (!transport_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "encoded image is empty");
  }

  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, transport_length,
                                             (void**)&out_image->allocation));
  out_image->allocation_size = transport_length;
  iree_status_t status = iree_ok_status();
  iree_host_size_t unescaped_length = 0;
  status = iree_json_unescape_string(source.value, transport_length,
                                     (char*)out_image->allocation,
                                     &unescaped_length);
  iree_string_view_t encoded = iree_make_string_view(
      (const char*)out_image->allocation, unescaped_length);
  if (iree_status_is_ok(status) && !source.format) {
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
    } else if (iree_string_view_consume_prefix(
                   &encoded,
                   IREE_SV("data:image/x-portable-arbitrarymap;base64,"))) {
      out_image->format = LOOM_SERVE_MEDIA_IMAGE_FORMAT_PAM;
    } else {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "image_url must be a base64 PNG, JPEG, GIF, WebP, or PAM data URL");
    }
  }
  iree_host_size_t decoded_length = 0;
  if (iree_status_is_ok(status)) {
    decoded_length = iree_base64_decoded_size(encoded);
    if (!decoded_length || decoded_length == IREE_HOST_SIZE_MAX ||
        decoded_length > maximum_length) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "compressed image must contain between 1 and %" PRIhsz " bytes",
          maximum_length);
    }
  }
  if (iree_status_is_ok(status)) {
    // Decoding toward the beginning is safe: each complete input quartet is
    // read before its three output bytes are written, including when the input
    // has no data-URL prefix.
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

static iree_status_t loom_serve_media_image_inspect_compressed(
    const loom_serve_media_compressed_image_t* compressed,
    const loom_serve_media_image_limits_t* limits,
    loom_serve_media_image_info_t* out_info) {
  memset(out_info, 0, sizeof(*out_info));
  const loom_serve_media_image_codec_t* codec =
      loom_serve_media_image_codec(compressed->format);
  if (!codec) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "image codec is disabled in this build");
  }
  loom_serve_media_image_inspection_t inspection = {
      .data = compressed->data,
  };
  codec->inspect(1, &inspection);
  iree_status_t status = inspection.status;
  *out_info = inspection.info;
  uint64_t pixel_count = 0;
  if (iree_status_is_ok(status)) {
    pixel_count = (uint64_t)out_info->width * (uint64_t)out_info->height;
    if (!out_info->width || !out_info->height ||
        out_info->width > limits->width || out_info->height > limits->height ||
        pixel_count > limits->pixels) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "image geometry %" PRIu32 "x%" PRIu32
                                " exceeds the configured bound",
                                out_info->width, out_info->height);
    }
  }
  if (iree_status_is_ok(status)) {
    if (pixel_count > IREE_HOST_SIZE_MAX / out_info->pixel_format) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "image pixel storage size overflow");
    }
  }
  if (iree_status_is_ok(status)) {
    const bool swaps_axes = out_info->orientation >= 5;
    out_info->display_width = swaps_axes ? out_info->height : out_info->width;
    out_info->display_height = swaps_axes ? out_info->width : out_info->height;
    out_info->compressed_length = compressed->data.data_length;
    out_info->pixel_length =
        (iree_host_size_t)pixel_count * out_info->pixel_format;
    loom_serve_media_digest_compute(compressed->data, &out_info->digest);
  } else {
    memset(out_info, 0, sizeof(*out_info));
  }
  return status;
}

iree_status_t loom_serve_media_image_inspect(
    loom_serve_media_image_source_t source,
    const loom_serve_media_image_limits_t* limits,
    iree_allocator_t host_allocator, loom_serve_media_image_info_t* out_info) {
  memset(out_info, 0, sizeof(*out_info));
  loom_serve_media_compressed_image_t compressed;
  IREE_RETURN_IF_ERROR(loom_serve_media_image_source_decode(
      source, limits->compressed_bytes, host_allocator, &compressed));
  iree_status_t status =
      loom_serve_media_image_inspect_compressed(&compressed, limits, out_info);
  loom_serve_media_compressed_image_deinitialize(&compressed);
  return status;
}

iree_status_t loom_serve_media_image_input_initialize(
    loom_serve_media_image_source_t source,
    const loom_serve_media_image_limits_t* limits,
    iree_allocator_t host_allocator,
    loom_serve_media_image_input_t* out_input) {
  memset(out_input, 0, sizeof(*out_input));
  loom_serve_media_compressed_image_t compressed;
  IREE_RETURN_IF_ERROR(loom_serve_media_image_source_decode(
      source, limits->compressed_bytes, host_allocator, &compressed));
  iree_status_t status = loom_serve_media_image_inspect_compressed(
      &compressed, limits, &out_input->info);
  if (iree_status_is_ok(status)) {
    out_input->encoded = compressed.data;
    out_input->host_allocator = host_allocator;
    out_input->allocation = compressed.allocation;
    out_input->allocation_size = compressed.allocation_size;
  } else {
    loom_serve_media_compressed_image_deinitialize(&compressed);
  }
  return status;
}

void loom_serve_media_image_input_deinitialize(
    loom_serve_media_image_input_t* input) {
  iree_allocator_free(input->host_allocator, input->allocation);
  memset(input, 0, sizeof(*input));
}

iree_status_t loom_serve_media_image_decode_batch(
    iree_host_size_t count, const loom_serve_media_image_input_t* const* inputs,
    const iree_byte_span_t* pixels, iree_allocator_t host_allocator,
    iree_status_t* out_statuses) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    out_statuses[i] = iree_ok_status();
  }
  if (!count) {
    return iree_ok_status();
  }
  iree_host_size_t scratch_bytes = 0;
  if (!iree_host_size_checked_mul(
          count, sizeof(loom_serve_media_image_decoding_t), &scratch_bytes)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "image batch scratch size overflow");
  }
  loom_serve_media_image_decoding_t* decoding = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, scratch_bytes, (void**)&decoding));
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (pixels[i].data_length != inputs[i]->info.pixel_length) {
      out_statuses[i] = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pixel destination has %zu bytes but image requires %zu",
          pixels[i].data_length, inputs[i]->info.pixel_length);
    }
  }
  const loom_serve_media_image_codec_t* const* codecs = NULL;
  const iree_host_size_t codec_count = loom_serve_media_image_codecs(&codecs);
  for (iree_host_size_t c = 0; c < codec_count; ++c) {
    iree_host_size_t selected_count = 0;
    for (iree_host_size_t i = 0; i < count; ++i) {
      if (inputs[i]->info.format == codecs[c]->format &&
          iree_status_is_ok(out_statuses[i])) {
        decoding[selected_count++] =
            (loom_serve_media_image_decoding_t){.data = inputs[i]->encoded,
                                                .info = &inputs[i]->info,
                                                .pixels = pixels[i]};
      }
    }
    if (!selected_count) {
      continue;
    }
    codecs[c]->decode(selected_count, decoding);
    iree_host_size_t selected_index = 0;
    for (iree_host_size_t i = 0; i < count; ++i) {
      if (inputs[i]->info.format == codecs[c]->format &&
          iree_status_is_ok(out_statuses[i])) {
        out_statuses[i] = decoding[selected_index++].status;
      }
    }
  }
  iree_allocator_free(host_allocator, decoding);
  return iree_ok_status();
}
