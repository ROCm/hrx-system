// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/text/statistics.h"

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
  const uint64_t values[] = {
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
  const loom_runtime_telemetry_topic_t topic = {"text", 1,
                                                IREE_ARRAYSIZE(fields), fields};
  return sink.emit(sink.self, &topic, values);
}
