// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/telemetry/pool.h"

iree_status_t loom_runtime_telemetry_sample_pool(
    const char* name, const iree_hal_pool_stats_t* statistics,
    loom_runtime_telemetry_sample_sink_t sink) {
  static const loom_runtime_telemetry_metric_t fields[] = {
      {"reserved_bytes", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"committed_bytes", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"free_bytes", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"quarantined_bytes", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"abandoned_bytes", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"budget_limit", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"reservation_count", "reservations", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"slab_count", "slabs", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"reserve_count", "reservations", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"release_count", "reservations", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"reuse_count", "reservations", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"reuse_miss_count", "reservations", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"fresh_count", "reservations", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"exhausted_count", "reservations", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"over_budget_count", "reservations", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"wait_count", "reservations", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"quarantine_eviction_count", "reservations",
       LOOM_RUNTIME_TELEMETRY_COUNTER},
  };
  const uint64_t values[] = {
      statistics->bytes_reserved,
      statistics->bytes_committed,
      statistics->bytes_free,
      statistics->bytes_quarantined,
      statistics->bytes_abandoned,
      statistics->budget_limit,
      statistics->reservation_count,
      statistics->slab_count,
      statistics->reserve_count,
      statistics->release_count,
      statistics->reuse_count,
      statistics->reuse_miss_count,
      statistics->fresh_count,
      statistics->exhausted_count,
      statistics->over_budget_count,
      statistics->wait_count,
      statistics->quarantine_eviction_count,
  };
  static_assert(IREE_ARRAYSIZE(fields) == IREE_ARRAYSIZE(values),
                "one value per pool field");
  const loom_runtime_telemetry_topic_t topic = {name, 1, IREE_ARRAYSIZE(fields),
                                                fields};
  return sink.emit(sink.self, &topic, values);
}
