// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Full-checkpoint retained ownership witness. No replacement model or kernels.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "experimental/loom_serve/image/model.h"
#include "iree/base/internal/path.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"

IREE_FLAG(string, model, "", "Source catalog.");
IREE_FLAG(string, checkpoint, "", "Official checkpoint directory.");
IREE_FLAG(string, adapter, "", "Optional model-specific adapter asset.");
IREE_FLAG(string, output, "",
          "Existing directory for completed F32 RGB files.");

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (!FLAG_model[0] || !FLAG_output[0] || !FLAG_checkpoint[0]) {
    fprintf(stderr, "model, checkpoint and output are required.\n");
    return EXIT_FAILURE;
  }
  const iree_allocator_t allocator = iree_allocator_system();
  const loom_serve_image_model_options_t options = {
      .source_directory = iree_make_cstring_view(FLAG_model),
      .checkpoint_directory = iree_make_cstring_view(FLAG_checkpoint),
      .adapter_path = iree_make_cstring_view(FLAG_adapter),
      .height = 384,
      .width = 384,
      .text_tokens = 512,
  };
  loom_serve_image_model_t* model = NULL;
  iree_status_t status =
      loom_serve_image_model_create(&options, &model, allocator);
  const char* prompts[] = {
      "a red fox in the snow",
      "red"
      " red red red red red red red red red red red red red red red red"
      " red red red red red red red red red red red red red red red red"
      " red red red red red red red red red red red red red red red red"
      " red red red red red red red red red red red red red red red red",
  };
  const uint32_t cases[] = {0, 1, 0};
  const iree_host_size_t case_count =
      IREE_ARRAYSIZE(cases) * (FLAG_adapter[0] ? 2 : 1);
  for (iree_host_size_t i = 0; i < case_count && iree_status_is_ok(status);
       ++i) {
    iree_const_byte_span_t rgb = iree_const_byte_span_empty();
    iree_status_t rejected = loom_serve_image_model_generate(
        model, IREE_SV("rejected"), 0, NAN, &rgb);
    if (iree_status_code(rejected) != IREE_STATUS_INVALID_ARGUMENT ||
        rgb.data_length) {
      status = iree_status_join(
          iree_make_status(IREE_STATUS_DATA_LOSS,
                           "invalid request returned RGB or wrong status"),
          rejected);
    } else {
      // Expected failure is observed at the qualification boundary.
      iree_status_fprint(stderr, rejected);
      iree_status_free(rejected);
      const uint32_t prompt = cases[i % IREE_ARRAYSIZE(cases)];
      const float strength =
          FLAG_adapter[0] && i < IREE_ARRAYSIZE(cases) ? 0.0f : 1.0f;
      status = loom_serve_image_model_generate(
          model, iree_make_cstring_view(prompts[prompt]), prompt ? 42 : 0,
          strength, &rgb);
    }
    char name[32];
    snprintf(name, sizeof(name), "image-%zu.f32", i);
    char* path = NULL;
    if (iree_status_is_ok(status)) {
      status =
          iree_file_path_join(iree_make_cstring_view(FLAG_output),
                              iree_make_cstring_view(name), allocator, &path);
    }
    if (iree_status_is_ok(status)) {
      status = iree_io_file_contents_write(iree_make_cstring_view(path), rgb,
                                           allocator);
    }
    iree_allocator_free(allocator, path);
  }
  status = iree_status_join(status, loom_serve_image_model_destroy(model));
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  printf(
      "PASS: retained model produced %zu short/long/short images after "
      "rejected requests.\n",
      case_count);
  return EXIT_SUCCESS;
}
