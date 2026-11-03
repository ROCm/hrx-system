// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "iree/base/internal/base64.h"
#include "loom/serve/media/image.h"
#include "loom/serve/media/image_codec.h"
#include "loom/serve/media/image_request.h"

static void Exercise(iree_string_view_t input) {
  const loom_serve_media_image_limits_t limits = {
      .compressed_bytes = 65536,
      .width = 128,
      .height = 128,
      .pixels = 16384,
  };
  loom_serve_media_image_input_t image;
  iree_status_t status = loom_serve_media_image_input_initialize(
      loom_serve_media_image_source_from_data_url(input), &limits,
      iree_allocator_system(), &image);
  if (iree_status_is_ok(status)) {
    const loom_serve_media_image_input_t* images[] = {&image, &image};
    const size_t size = image.info.pixel_length;
    std::vector<uint8_t> pixels(size * 2);
    const iree_byte_span_t destinations[] = {
        iree_make_byte_span(pixels.data(), size),
        iree_make_byte_span(pixels.data() + size, size)};
    iree_status_t results[2];
    status = loom_serve_media_image_decode_batch(
        2, images, destinations, results, iree_allocator_system());
    for (auto result : results) {
      iree_status_free(result);
    }
    if (iree_status_is_ok(status)) {
      // The same inspected input reaches the source-packet boundary with
      // distinct metadata alignment and projection extents in one cohort.
      const uint8_t metadata[] = {1, 2, 3};
      const loom_serve_media_image_request_t requests[] = {
          {1, &image, {}, 1},
          {1, &image, iree_make_const_byte_span(metadata, sizeof(metadata)), 3},
      };
      iree_vm_buffer_t* packets[2];
      status = loom_serve_media_image_request_create_batch(
          2, requests, iree_allocator_system(), packets, results);
      for (size_t i = 0; i < 2; ++i) {
        iree_vm_buffer_release(packets[i]);
        iree_status_free(results[i]);
      }
    }
  }
  loom_serve_media_image_input_deinitialize(&image);
  // Malformed input is the expected result of mutation at this API boundary.
  iree_status_free(status);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size > 65536) {
    return 0;
  }
  // Text corpus entries end in a line separator outside the data URL.
  if (size && data[size - 1] == '\n') {
    --size;
  }
  Exercise(iree_make_string_view(reinterpret_cast<const char*>(data), size));
  const loom_serve_media_image_codec_t* const* codecs = nullptr;
  const iree_host_size_t count = loom_serve_media_image_codecs(&codecs);
  if (!size || !count) {
    return 0;
  }
  const auto* codec = codecs[data[0] % count];
  std::string encoded(iree_base64_encoded_size(size - 1), '\0');
  iree_host_size_t written = 0;
  iree_status_t status = iree_base64_encode(
      iree_make_const_byte_span(data + 1, size - 1),
      iree_make_mutable_string_view(encoded.data(), encoded.size()), &written);
  if (iree_status_is_ok(status)) {
    encoded.resize(written);
    const std::string url =
        "data:" + std::string(codec->mime_type.data, codec->mime_type.size) +
        ";base64," + encoded;
    Exercise(iree_make_string_view(url.data(), url.size()));
  }
  iree_status_free(status);
  return 0;
}
