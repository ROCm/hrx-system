// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native prompt-to-image CLI using one retained Krea model.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/loom_serve/models/krea2/model.h"
#include "iree/base/internal/json.h"
#include "iree/base/internal/math.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"

IREE_FLAG(string, model, "experimental/loom_serve/models/krea2",
          "Live source catalog directory.");
IREE_FLAG(string, checkpoint, "",
          "Official Krea 2 Turbo checkpoint directory.");
IREE_FLAG(string, adapter, "", "Optional softwatercolor safetensors file.");
IREE_FLAG(string, prompt, "", "Image prompt; the model supplies its template.");
IREE_FLAG(string, seed, "0", "Unsigned decimal 64-bit native noise seed.");
IREE_FLAG(int32_t, height, 384, "Output pixel height, divisible by 16.");
IREE_FLAG(int32_t, width, 384, "Output pixel width, divisible by 16.");
IREE_FLAG(int32_t, text_tokens, 512, "Retained text extent, divisible by 16.");
IREE_FLAG(float, strength, 1.0f, "Adapter strength; zero is base identity.");
IREE_FLAG(string, output, "", "Output PPM image path.");

static iree_status_t krea2_write_image(iree_const_byte_span_t rgb,
                                       iree_allocator_t allocator) {
  char header[64];
  const iree_host_size_t header_length = (iree_host_size_t)snprintf(
      header, sizeof(header), "P6\n%d %d\n255\n", FLAG_width, FLAG_height);
  const iree_host_size_t pixels = rgb.data_length / 12;
  uint8_t* image = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, header_length + pixels * 3, (void**)&image));
  memcpy(image, header, header_length);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < pixels * 3 && iree_status_is_ok(status);
       ++i) {
    const float value = iree_unaligned_load_le_f32(rgb.data + i * 4);
    if (!isfinite(value) || value < -1.0f || value > 1.0f) {
      status =
          iree_make_status(IREE_STATUS_DATA_LOSS,
                           "nonfinite or out-of-range final RGB at %zu", i);
    } else {
      const iree_host_size_t channel = i / pixels;
      const iree_host_size_t pixel = i % pixels;
      image[header_length + pixel * 3 + channel] =
          (uint8_t)nearbyintf((value / 2.0f + 0.5f) * 255.0f);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_file_contents_write(
        iree_make_cstring_view(FLAG_output),
        iree_make_const_byte_span(image, header_length + pixels * 3),
        allocator);
  }
  iree_allocator_free(allocator, image);
  return status;
}

static iree_status_t krea2_run(iree_allocator_t allocator) {
  if (!FLAG_output[0] || !FLAG_checkpoint[0]) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--checkpoint and --output are required");
  }
  if ((!FLAG_adapter[0] && FLAG_strength != 1.0f) || !isfinite(FLAG_strength)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "finite --strength requires --adapter");
  }
  uint64_t seed = 0;
  IREE_RETURN_IF_ERROR(
      iree_json_parse_uint64(iree_make_cstring_view(FLAG_seed), &seed));
  const loom_serve_krea2_model_options_t options = {
      .source_directory = iree_make_cstring_view(FLAG_model),
      .checkpoint_directory = iree_make_cstring_view(FLAG_checkpoint),
      .adapter_path = iree_make_cstring_view(FLAG_adapter),
      .height = (uint32_t)FLAG_height,
      .width = (uint32_t)FLAG_width,
      .text_tokens = (uint32_t)FLAG_text_tokens,
  };
  loom_serve_krea2_model_t* model = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_krea2_model_create(&options, &model, allocator));
  iree_const_byte_span_t rgb = iree_const_byte_span_empty();
  iree_status_t status = loom_serve_krea2_model_generate(
      model, iree_make_cstring_view(FLAG_prompt), seed, FLAG_strength, &rgb);
  if (iree_status_is_ok(status)) {
    status = krea2_write_image(rgb, allocator);
  }
  return iree_status_join(status, loom_serve_krea2_model_destroy(model));
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  iree_status_t status = krea2_run(iree_allocator_system());
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  printf("Wrote %s\n", FLAG_output);
  return EXIT_SUCCESS;
}
