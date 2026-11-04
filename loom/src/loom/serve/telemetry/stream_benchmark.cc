// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "benchmark/benchmark.h"
#include "loom/runtime/telemetry/stream.h"
#include "loom/serve/telemetry/record.h"

namespace {

static loom_runtime_telemetry_record_t Snapshot() {
  loom_runtime_telemetry_record_t record = {
      .event = &LOOM_SERVICE_TELEMETRY_SNAPSHOT,
  };
  const loom_service_telemetry_snapshot_record_t snapshot = {
      .counters = {1000, 990, 2, 3, 1, 16384, 4096, 2048},
      .queued = 4,
      .active = 16,
      .phase = "executing",
  };
  std::memcpy(record.data.words, &snapshot, sizeof(snapshot));
  return record;
}

static uint64_t& BatchCount(loom_runtime_telemetry_record_t& record) {
  constexpr size_t offset =
      offsetof(loom_service_telemetry_snapshot_record_t, counters) +
      offsetof(loom_service_telemetry_counters_t, batches);
  return record.data.words[offset / sizeof(uint64_t)];
}

static loom_runtime_telemetry_stream_t* CreatePopulatedStream(size_t capacity) {
  loom_runtime_telemetry_stream_t* stream = nullptr;
  IREE_CHECK_OK(loom_runtime_telemetry_stream_allocate(
      capacity, iree_allocator_system(), &stream));
  auto record = Snapshot();
  for (size_t i = 0; i < capacity; ++i) {
    record.time_ns = iree_time_now();
    ++BatchCount(record);
    loom_runtime_telemetry_stream_append(stream, &record);
  }
  return stream;
}

// One timed iteration captures and publishes one native snapshot. There is no
// consumer, formatting or sink; a full ring is overwritten on every iteration.
static void BM_PublishSnapshot(benchmark::State& state) {
  auto* stream = CreatePopulatedStream(state.range(0));
  auto record = Snapshot();
  BatchCount(record) += state.range(0);
  for (auto _ : state) {
    record.time_ns = iree_time_now();
    ++BatchCount(record);
    loom_runtime_telemetry_stream_append(stream, &record);
  }
  benchmark::DoNotOptimize(loom_runtime_telemetry_stream_head(stream));
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * sizeof(record));
  loom_runtime_telemetry_stream_free(stream);
}
BENCHMARK(BM_PublishSnapshot)->Arg(128)->Arg(256);

// One timed iteration copies and formats the maximum 64-record HTTP history
// batch. This measures consumer work, not publication, TCP or sink latency.
static void BM_CopyFormatBatch64(benchmark::State& state) {
  auto* stream = CreatePopulatedStream(state.range(0));
  const uint64_t start = loom_runtime_telemetry_stream_head(stream) - 64;
  iree_string_builder_t output;
  iree_string_builder_initialize(iree_allocator_system(), &output);
  auto consume = [&]() {
    iree_string_builder_reset(&output);
    uint64_t cursor = start;
    for (size_t i = 0; i < 64; ++i) {
      loom_runtime_telemetry_record_t record;
      const auto result =
          loom_runtime_telemetry_stream_read(stream, &cursor, &record);
      if (result != LOOM_RUNTIME_TELEMETRY_READ_RECORD) {
        state.SkipWithError("fixed published snapshot batch is unavailable");
        return;
      }
      IREE_CHECK_OK(
          loom_serve_telemetry_record_format(cursor - 1, &record, &output));
    }
  };
  // Materialize the exact reusable JSON capacity before steady-state timing.
  consume();
  for (auto _ : state) {
    consume();
    benchmark::DoNotOptimize(iree_string_builder_view(&output));
  }
  state.SetItemsProcessed(state.iterations() * 64);
  state.SetBytesProcessed(state.iterations() *
                          iree_string_builder_size(&output));
  iree_string_builder_deinitialize(&output);
  loom_runtime_telemetry_stream_free(stream);
}
BENCHMARK(BM_CopyFormatBatch64)->Arg(128)->Arg(256);

static iree_status_t AccumulateFields(
    void* self, iree_host_size_t count,
    const loom_runtime_telemetry_field_t* fields) {
  auto* total = static_cast<uint64_t*>(self);
  for (iree_host_size_t i = 0; i < count; ++i) {
    switch (fields[i].kind) {
      case LOOM_RUNTIME_TELEMETRY_FIELD_NULL:
        break;
      case LOOM_RUNTIME_TELEMETRY_FIELD_UINT64:
        *total += fields[i].value.uint64_value;
        break;
      case LOOM_RUNTIME_TELEMETRY_FIELD_INT64:
        *total += static_cast<uint64_t>(fields[i].value.int64_value);
        break;
      case LOOM_RUNTIME_TELEMETRY_FIELD_BOOL:
        *total += fields[i].value.bool_value;
        break;
      case LOOM_RUNTIME_TELEMETRY_FIELD_STRING:
        *total += fields[i].value.string_value.size;
        break;
      case LOOM_RUNTIME_TELEMETRY_FIELD_OBJECT:
        IREE_CHECK_OK(AccumulateFields(self, fields[i].value.object_value.count,
                                       fields[i].value.object_value.fields));
        break;
    }
  }
  return iree_ok_status();
}

// The same bounded snapshot batch, consumed natively without a JSON builder.
static void BM_CopyReadFieldsBatch64(benchmark::State& state) {
  auto* stream = CreatePopulatedStream(state.range(0));
  const uint64_t start = loom_runtime_telemetry_stream_head(stream) - 64;
  uint64_t total = 0;
  for (auto _ : state) {
    uint64_t cursor = start;
    for (size_t i = 0; i < 64; ++i) {
      loom_runtime_telemetry_record_t record;
      if (loom_runtime_telemetry_stream_read(stream, &cursor, &record) !=
          LOOM_RUNTIME_TELEMETRY_READ_RECORD) {
        state.SkipWithError("fixed published snapshot batch is unavailable");
        break;
      }
      IREE_CHECK_OK(
          record.event->read_fields(&record, {&total, AccumulateFields}));
    }
    benchmark::DoNotOptimize(total);
  }
  state.SetItemsProcessed(state.iterations() * 64);
  loom_runtime_telemetry_stream_free(stream);
}
BENCHMARK(BM_CopyReadFieldsBatch64)->Arg(128)->Arg(256);

}  // namespace
