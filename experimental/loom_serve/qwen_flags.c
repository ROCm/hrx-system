// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/qwen_flags.h"

#include "iree/base/tooling/flags.h"

IREE_FLAG(string, model, "experimental/loom_serve/models/qwen38",
          "Portable model source directory; JIT compiled for the live device.");
IREE_FLAG(int32_t, prefill_capacity, 512, "Isolated prefill specialization.");
IREE_FLAG(int32_t, context_capacity, 16384,
          "Retained tokens per resident row.");
IREE_FLAG(int32_t, pool_capacity, 0,
          "Shared target/draft KV token capacity (multiple of 64); requires "
          "packed epochs. Zero selects the dense comparison layout.");
IREE_FLAG_LIST(
    string, epoch,
    "Packed JIT shape as tokens:spans, e.g. 128:8; repeat to cache shapes.");
IREE_FLAG(bool, mtp, false, "JIT MTP proposal, catch-up, and verifier stages.");
IREE_FLAG(string, weights, "", "Canonical Qwen3.8-27B UD-Q5_K_XL GGUF path.");
IREE_FLAG(string, tokenizer, "", "Hugging Face tokenizer.json path.");
IREE_FLAG(string, kernel_sanitizer, "none",
          "Device checks: none or access|value|operation|race.");
IREE_FLAG(string, kernel_sanitizer_reporting, "default",
          "Device assertion reporting: default, trap, or report-only.");

static iree_status_t qwen_sanitizer_from_flags(
    loomc_sanitizer_options_t* out_options) {
  *out_options = (loomc_sanitizer_options_t){
      .type = LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
      .structure_size = sizeof(*out_options),
  };
  iree_string_view_t remaining = iree_make_cstring_view(FLAG_kernel_sanitizer);
  if (iree_string_view_ends_with(remaining, IREE_SV("|"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel sanitizer list has an empty check");
  }
  if (!iree_string_view_equal(remaining, IREE_SV("none"))) {
    do {
      iree_string_view_t check;
      iree_string_view_split(remaining, '|', &check, &remaining);
      if (iree_string_view_equal(check, IREE_SV("access"))) {
        out_options->checks |= LOOMC_SANITIZER_CHECK_ACCESS;
      } else if (iree_string_view_equal(check, IREE_SV("value"))) {
        out_options->checks |= LOOMC_SANITIZER_CHECK_VALUE;
      } else if (iree_string_view_equal(check, IREE_SV("operation"))) {
        out_options->checks |= LOOMC_SANITIZER_CHECK_OPERATION;
      } else if (iree_string_view_equal(check, IREE_SV("race"))) {
        out_options->checks |= LOOMC_SANITIZER_CHECK_RACE;
      } else {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown kernel sanitizer '%.*s'",
                                (int)check.size, check.data);
      }
    } while (!iree_string_view_is_empty(remaining));
  }
  const iree_string_view_t reporting =
      iree_make_cstring_view(FLAG_kernel_sanitizer_reporting);
  if (iree_string_view_equal(reporting, IREE_SV("default"))) {
    out_options->reporting_mode = LOOMC_SANITIZER_REPORTING_MODE_DEFAULT;
  } else if (iree_string_view_equal(reporting, IREE_SV("trap"))) {
    out_options->reporting_mode = LOOMC_SANITIZER_REPORTING_MODE_TRAP;
  } else if (iree_string_view_equal(reporting, IREE_SV("report-only"))) {
    out_options->reporting_mode = LOOMC_SANITIZER_REPORTING_MODE_REPORT_ONLY;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown kernel sanitizer reporting mode");
  }
  return iree_ok_status();
}

iree_host_size_t loom_serve_qwen_shape_count_from_flags(void) {
  return FLAG_epoch_list().count;
}

bool loom_serve_qwen_mtp_from_flags(void) { return FLAG_mtp; }

iree_status_t loom_serve_qwen_model_create_from_flags(
    iree_host_size_t row_count, iree_allocator_t host_allocator,
    loom_serve_qwen_model_t** out_model) {
  *out_model = NULL;
  if (!FLAG_model[0] || !FLAG_weights[0] || !FLAG_tokenizer[0] ||
      FLAG_prefill_capacity < 1 || FLAG_context_capacity < 1 ||
      FLAG_pool_capacity < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "provide model sources, weights, tokenizer and positive capacities");
  }
  const iree_flag_string_list_t epochs = FLAG_epoch_list();
  loomc_sanitizer_options_t sanitizer;
  IREE_RETURN_IF_ERROR(qwen_sanitizer_from_flags(&sanitizer));
  loom_serve_qwen_shape_t* shapes = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, epochs.count, sizeof(*shapes), (void**)&shapes));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < epochs.count && iree_status_is_ok(status);
       ++i) {
    iree_string_view_t tokens, spans;
    iree_string_view_split(epochs.values[i], ':', &tokens, &spans);
    uint32_t token_count = 0, span_count = 0;
    if (!iree_string_view_atoi_uint32(tokens, &token_count) ||
        !iree_string_view_atoi_uint32(spans, &span_count)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "epoch must be tokens:spans, got '%.*s'",
                           (int)epochs.values[i].size, epochs.values[i].data);
    }
    shapes[i] = (loom_serve_qwen_shape_t){token_count, span_count};
  }
  const loom_serve_qwen_options_t options = {
      .source_directory = iree_make_cstring_view(FLAG_model),
      .prefill_capacity = (iree_host_size_t)FLAG_prefill_capacity,
      .context_capacity = (iree_host_size_t)FLAG_context_capacity,
      .pool_capacity = (iree_host_size_t)FLAG_pool_capacity,
      .epoch_count = epochs.count,
      .epoch_shapes = shapes,
      .enable_mtp = FLAG_mtp,
      .kernel_sanitizer = sanitizer,
      .weights_path = iree_make_cstring_view(FLAG_weights),
      .tokenizer_path = iree_make_cstring_view(FLAG_tokenizer),
      .row_count = row_count,
  };
  if (iree_status_is_ok(status)) {
    status = loom_serve_qwen_model_create(&options, host_allocator, out_model);
  }
  iree_allocator_free(host_allocator, shapes);
  return status;
}
