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
#include "experimental/loom_serve/runtime/device_flags.h"
#include "experimental/loom_serve/runtime/host_flags.h"
#include "iree/base/internal/path.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"

IREE_FLAG(string, model, "", "Source catalog.");
IREE_FLAG(string, checkpoint, "", "Official checkpoint directory.");
IREE_FLAG(string, adapter, "", "Optional model-specific adapter asset.");
IREE_FLAG(string, output, "",
          "Existing directory for completed F32 RGB files.");
IREE_FLAG(bool, reload_weights, false,
          "Deactivate after each image and reactivate on the next request.");

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
  loom_serve_device_group_t* group = NULL;
  loom_serve_device_t* device = NULL;
  loom_serve_host_t* host = NULL;
  iree_status_t status = loom_serve_host_create_from_flags(&host, allocator);
  if (iree_status_is_ok(status)) {
    status = loom_serve_device_group_create_from_flags(host, &group, allocator);
  }
  if (iree_status_is_ok(status)) {
    device = loom_serve_device_group_at(group, 0);
    status = loom_serve_image_model_create(device, &options, &model, allocator);
  }
  if (iree_status_is_ok(status) &&
      loom_serve_device_memory_statistics(device).workspace.bytes_committed) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "model registration committed private scratch");
  }
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
    loom_serve_image_prepared_t prepared;
    iree_status_t rejected = loom_serve_image_model_generate(
        model, IREE_SV("rejected"), 0, NAN, &prepared, &rgb);
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
          strength, &prepared, &rgb);
    }
    char name[32];
    snprintf(name, sizeof(name), "image-%zu.f32", i);
    char* path = NULL;
    if (iree_status_is_ok(status)) {
      printf("{\"event\":\"image_prepared\",\"stage\":%" PRIu64
             ",\"tag\":%" PRId64 ",\"input_bytes\":%" PRIu64 "}\n",
             prepared.stage, prepared.tag, prepared.input_bytes);
      status =
          iree_file_path_join(iree_make_cstring_view(FLAG_output),
                              iree_make_cstring_view(name), allocator, &path);
    }
    if (iree_status_is_ok(status)) {
      status = iree_io_file_contents_write(iree_make_cstring_view(path), rgb,
                                           allocator);
    }
    if (iree_status_is_ok(status)) {
      const iree_hal_pool_stats_t workspace =
          loom_serve_device_memory_statistics(device).workspace;
      if (workspace.reservation_count || workspace.bytes_reserved ||
          workspace.reserve_count != i + 1 ||
          workspace.release_count != i + 1) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "completed image retained private scratch");
      }
    }
    // Reclaim after each short/long/short group. The following group must
    // regrow backing without disturbing the model, weights or output buffers.
    if (iree_status_is_ok(status) && (i + 1) % IREE_ARRAYSIZE(cases) == 0) {
      status = loom_serve_device_trim(
          device,
          loom_serve_device_memory_statistics(device).retained.committed_bytes);
      if (iree_status_is_ok(status)) {
        const iree_hal_pool_stats_t workspace =
            loom_serve_device_memory_statistics(device).workspace;
        printf("{\"event\":\"workspace_trimmed\",\"committed_bytes\":%" PRIu64
               ",\"reuse_count\":%" PRIu64 "}\n",
               (uint64_t)workspace.bytes_committed, workspace.reuse_count);
        if (workspace.bytes_committed || !workspace.reuse_count) {
          status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                    "private scratch did not reuse and trim");
        }
      }
    }
    if (iree_status_is_ok(status) && FLAG_reload_weights) {
      status = loom_serve_image_model_deactivate(model);
      if (iree_status_is_ok(status) &&
          loom_serve_image_model_weight_statistics(model).committed_bytes) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "image deactivation retained weight backing");
      }
    }
    iree_allocator_free(allocator, path);
  }
  status = iree_status_join(status, loom_serve_image_model_destroy(model));
  status = iree_status_join(status, loom_serve_device_group_destroy(group));
  loom_serve_host_release(host);
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
