// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/telemetry.h"

#include <string.h>

iree_status_t loom_runtime_storage_sample_pool(
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

static iree_status_t loom_runtime_storage_shared_memory_fields(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_runtime_storage_shared_memory_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_bool("shared", true),
      loom_runtime_telemetry_field_uint64("retained_committed_bytes",
                                          value.retained_committed_bytes),
      loom_runtime_telemetry_field_uint64("workspace_committed_bytes",
                                          value.workspace_committed_bytes),
      loom_runtime_telemetry_field_uint64("workspace_live_bytes",
                                          value.workspace_live_bytes),
      loom_runtime_telemetry_field_uint64("workspace_slab_count",
                                          value.workspace_slab_count),
      loom_runtime_telemetry_field_uint64("workspace_reuse_count",
                                          value.workspace_reuse_count),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static const loom_runtime_telemetry_event_t shared_pool_memory_event = {
    "shared_pool_memory", loom_runtime_storage_shared_memory_fields};

void loom_runtime_storage_record_shared_memory(
    loom_runtime_telemetry_stream_t* stream, iree_time_t time_ns,
    const loom_runtime_storage_shared_memory_record_t* memory) {
  loom_runtime_telemetry_record_t record = {.time_ns = time_ns,
                                            .event = &shared_pool_memory_event};
  static_assert(sizeof(*memory) <= sizeof(record.data), "record capacity");
  memcpy(record.data.words, memory, sizeof(*memory));
  loom_runtime_telemetry_stream_append(stream, &record);
}
