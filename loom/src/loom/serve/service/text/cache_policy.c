// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/cache_policy.h"

uint64_t loom_serve_text_cache_default_target(uint64_t memory_limit) {
  const uint64_t target = UINT64_C(2) * 1024 * 1024 * 1024;
  return memory_limit ? iree_min(target, memory_limit / 16) : target;
}

static iree_host_size_t loom_serve_text_cache_endpoints(
    iree_host_size_t explicit_capacity, iree_host_size_t session_capacity,
    iree_host_size_t prefix_capacity) {
  return explicit_capacity + prefix_capacity +
         (session_capacity ? session_capacity + 1 : 0);
}

iree_status_t loom_serve_text_cache_select(
    const loom_serve_text_cache_options_t* options,
    iree_host_size_t explicit_capacity,
    loom_serve_text_cache_evaluate_fn_t evaluate, void* user_data,
    loom_serve_text_cache_selection_t* out_selection) {
  const bool automatic_sessions =
      options->session_capacity == LOOM_SERVE_TEXT_CACHE_AUTOMATIC;
  const bool automatic_prefixes =
      options->prefix_capacity == LOOM_SERVE_TEXT_CACHE_AUTOMATIC;
  *out_selection = (loom_serve_text_cache_selection_t){
      .session_capacity = automatic_sessions ? 0 : options->session_capacity,
      .prefix_capacity = automatic_prefixes ? 0 : options->prefix_capacity,
      .target_bytes = options->target_bytes};
  out_selection->checkpoint_capacity = loom_serve_text_cache_endpoints(
      explicit_capacity, out_selection->session_capacity,
      out_selection->prefix_capacity);
  iree_host_size_t evaluated_capacity = out_selection->checkpoint_capacity;
  loom_serve_text_cache_cost_t baseline = {0};
  IREE_RETURN_IF_ERROR(evaluate(user_data, evaluated_capacity, &baseline));
  if (!options->target_bytes) {
    return iree_ok_status();
  }

  iree_status_t status = iree_ok_status();
  if (automatic_prefixes) {
    const iree_host_size_t maximum =
        iree_min(2, INT32_MAX - out_selection->checkpoint_capacity);
    for (iree_host_size_t count = maximum; count && iree_status_is_ok(status);
         --count) {
      evaluated_capacity = out_selection->checkpoint_capacity + count;
      loom_serve_text_cache_cost_t cost = {0};
      status = evaluate(user_data, evaluated_capacity, &cost);
      const uint64_t additional =
          cost.storage_bytes -
          iree_min(cost.storage_bytes, baseline.storage_bytes);
      if (iree_status_is_ok(status) && additional <= options->target_bytes) {
        out_selection->prefix_capacity = count;
        out_selection->checkpoint_capacity = evaluated_capacity;
        out_selection->additional_bytes = additional;
        break;
      }
    }
  }
  if (automatic_sessions && iree_status_is_ok(status)) {
    const iree_host_size_t remaining =
        INT32_MAX - out_selection->checkpoint_capacity;
    const iree_host_size_t maximum = remaining ? remaining - 1 : 0;
    iree_host_size_t lower = 0, upper = maximum, candidate = 1;
    bool refining = false;
    while (lower < upper && iree_status_is_ok(status)) {
      evaluated_capacity =
          explicit_capacity + out_selection->prefix_capacity + candidate + 1;
      loom_serve_text_cache_cost_t cost = {0};
      status = evaluate(user_data, evaluated_capacity, &cost);
      if (iree_status_is_ok(status)) {
        const uint64_t additional =
            cost.storage_bytes -
            iree_min(cost.storage_bytes, baseline.storage_bytes);
        if (additional <= options->target_bytes) {
          lower = candidate;
          out_selection->session_capacity = candidate;
          out_selection->checkpoint_capacity = evaluated_capacity;
          out_selection->additional_bytes = additional;
        } else {
          upper = candidate - 1;
          refining = true;
        }
        candidate = refining ? lower + (upper - lower + 1) / 2
                             : iree_min(upper, candidate * 2);
      }
    }
  }
  if (iree_status_is_ok(status) &&
      evaluated_capacity != out_selection->checkpoint_capacity) {
    loom_serve_text_cache_cost_t cost = {0};
    status = evaluate(user_data, out_selection->checkpoint_capacity, &cost);
  }
  return status;
}
