// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/text/statistics.h"

#include <string.h>

iree_status_t loom_service_text_sample_statistics(
    const loom_service_text_statistics_t* statistics,
    loom_runtime_telemetry_sample_sink_t sink) {
  static const loom_runtime_telemetry_metric_t fields[] = {
      {"prefill_rows", "requests", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"decode_rows", "requests", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"backpressured_rows", "requests", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"session_capacity", "sessions", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"pool_capacity", "tokens", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"pool_resident", "tokens", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"pool_reserved", "tokens", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"session_cache_endpoints", "sessions", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"prefix_cache_endpoints", "prefixes", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"checkpoint_host_bytes", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"elastic_state_committed_bytes", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"elastic_state_reserved_bytes", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"completed_epochs", "epochs", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"traversals", "traversals", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"prefill_tokens", "tokens", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"decode_tokens", "tokens", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"model_duration", "nanoseconds", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"draft_proposed_tokens", "tokens", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"draft_accepted_inputs", "tokens", LOOM_RUNTIME_TELEMETRY_COUNTER},
  };
  uint64_t values[] = {
      statistics->prefill_rows,
      statistics->decode_rows,
      statistics->backpressured_rows,
      statistics->session_capacity,
      statistics->pool.capacity,
      statistics->pool.resident,
      statistics->pool.reserved,
      statistics->session_cache.endpoints,
      statistics->prefix_cache.endpoints,
      statistics->checkpoint_host_bytes,
      statistics->state_memory.committed_bytes,
      statistics->state_memory.reserved_bytes,
      statistics->completed_epochs,
      statistics->traversals,
      statistics->prefill_tokens,
      statistics->decode_tokens,
      (uint64_t)statistics->model_duration,
      statistics->draft.proposed_tokens,
      statistics->draft.accepted_inputs,
  };
  static_assert(IREE_ARRAYSIZE(fields) == IREE_ARRAYSIZE(values),
                "one value per text field");
  static const loom_runtime_telemetry_topic_t topic = {
      "text", 1, IREE_ARRAYSIZE(fields), fields};
  if (statistics->draft.proposed_tokens == UINT64_MAX) {
    // Numeric samples have no null value. Keep the accepted count as the
    // final metric in the schema without the unavailable proposal counter.
    loom_runtime_telemetry_metric_t legacy_fields[IREE_ARRAYSIZE(fields) - 1];
    const iree_host_size_t last = IREE_ARRAYSIZE(legacy_fields) - 1;
    memcpy(legacy_fields, fields, last * sizeof(fields[0]));
    legacy_fields[last] = fields[last + 1];
    values[last] = values[last + 1];
    const loom_runtime_telemetry_topic_t legacy_topic = {
        "text", 2, IREE_ARRAYSIZE(legacy_fields), legacy_fields};
    return sink.emit(sink.self, &legacy_topic, values);
  }
  return sink.emit(sink.self, &topic, values);
}
