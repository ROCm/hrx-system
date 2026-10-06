// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/flags.h"

#include <string.h>

#include "experimental/loom_serve/text/schedule.h"
#include "iree/base/tooling/flags.h"

IREE_FLAG(string, model, "",
          "Portable model source directory; JIT compiled for the live device.");
IREE_FLAG(int32_t, prefill_capacity, 512,
          "Maximum automatic packed shape and isolated prefill capacity.");
IREE_FLAG(int32_t, context_capacity, 16384,
          "Retained tokens per resident row.");
IREE_FLAG(int32_t, pool_capacity, -1,
          "Shared target/draft KV token capacity (whole source-defined pages); "
          "requires "
          "packed epochs. -1 uses the entry point default (65536 for the "
          "packed server, zero for tools). Zero selects dense comparison.");
IREE_FLAG_LIST(
    string, epoch,
    "Packed JIT shape as tokens:spans; repeated flags replace the automatic "
    "catalog bounded by prefill_capacity and resident rows.");
IREE_FLAG(string, pool_backing, "elastic",
          "Pooled state backing: elastic demand-commits physical slabs; fixed "
          "backs all storage at startup and supports device sanitization.");
IREE_FLAG(int64_t, slab_bytes, 2097152,
          "Elastic physical slab size; zero selects allocator recommendation.");
IREE_FLAG(bool, mtp, false, "JIT MTP proposal, catch-up, and verifier stages.");
IREE_FLAG(string, weights, "", "Parameter file selected by the model source.");
IREE_FLAG(string, tokenizer, "", "Hugging Face tokenizer.json path.");
IREE_FLAG(string, kernel_sanitizer, "none",
          "Device checks: none or access|value|operation|race.");
IREE_FLAG(string, kernel_sanitizer_reporting, "default",
          "Device assertion reporting: default, trap, or report-only.");

static iree_status_t text_sanitizer_from_flags(
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

iree_host_size_t loom_serve_text_explicit_shape_count_from_flags(void) {
  return FLAG_epoch_list().count;
}

bool loom_serve_text_mtp_from_flags(void) { return FLAG_mtp; }

iree_status_t loom_serve_text_model_create_from_flags(
    const loom_serve_text_flag_defaults_t* defaults,
    loom_serve_text_model_t** out_model, iree_allocator_t host_allocator) {
  *out_model = NULL;
  if (!FLAG_model[0] || !FLAG_weights[0] || !FLAG_tokenizer[0] ||
      FLAG_prefill_capacity < 1 || FLAG_prefill_capacity > 512 ||
      FLAG_context_capacity < 1 || FLAG_pool_capacity < -1 ||
      defaults->row_count < 1 ||
      defaults->row_count > LOOM_SERVE_TEXT_ROW_CAPACITY) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "provide model sources, weights, tokenizer, 1-16 rows, positive "
        "context and prefill capacity in [1, 512]");
  }
  const iree_flag_string_list_t epochs = FLAG_epoch_list();
  loomc_sanitizer_options_t sanitizer;
  IREE_RETURN_IF_ERROR(text_sanitizer_from_flags(&sanitizer));
  const iree_host_size_t pool_capacity =
      FLAG_pool_capacity < 0 ? defaults->pool_capacity
                             : (iree_host_size_t)FLAG_pool_capacity;
  if ((strcmp(FLAG_pool_backing, "elastic") &&
       strcmp(FLAG_pool_backing, "fixed")) ||
      FLAG_slab_bytes < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pool_backing must be elastic or fixed, and "
                            "slab_bytes must be nonnegative");
  }
  loom_serve_packing_shape_t automatic[LOOM_SERVE_TEXT_DEFAULT_SHAPE_CAPACITY];
  iree_host_size_t automatic_count = 0;
  if (!epochs.count &&
      (defaults->automatic_shapes || pool_capacity || FLAG_mtp)) {
    automatic_count = loom_serve_text_default_shapes(
        defaults->row_count, (iree_host_size_t)FLAG_prefill_capacity,
        automatic);
  }
  loom_serve_packing_shape_t* shapes = NULL;
  if (epochs.count) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        host_allocator, epochs.count, sizeof(*shapes), (void**)&shapes));
  }
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
    shapes[i] = (loom_serve_packing_shape_t){token_count, span_count};
  }
  const loom_serve_text_options_t options = {
      .source_directory = iree_make_cstring_view(FLAG_model),
      .prefill_capacity = (iree_host_size_t)FLAG_prefill_capacity,
      .context_capacity = (iree_host_size_t)FLAG_context_capacity,
      .pool_capacity = pool_capacity,
      .backing = pool_capacity && !strcmp(FLAG_pool_backing, "elastic")
                     ? LOOM_SERVE_TEXT_BACKING_ELASTIC
                     : LOOM_SERVE_TEXT_BACKING_FIXED,
      .slab_size = (iree_device_size_t)FLAG_slab_bytes,
      .epoch_count = epochs.count ? epochs.count : automatic_count,
      .epoch_shapes = epochs.count ? shapes : automatic,
      .enable_mtp = FLAG_mtp,
      .kernel_sanitizer = sanitizer,
      .weights_path = iree_make_cstring_view(FLAG_weights),
      .tokenizer_path = iree_make_cstring_view(FLAG_tokenizer),
      .row_count = defaults->row_count,
  };
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_model_create(&options, host_allocator, out_model);
  }
  iree_allocator_free(host_allocator, shapes);
  return status;
}
