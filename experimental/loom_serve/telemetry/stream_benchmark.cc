// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "benchmark/benchmark.h"
#include "experimental/loom_serve/telemetry/stream.h"

namespace {

static loom_serve_telemetry_record_t Snapshot() {
  loom_serve_telemetry_record_t record = {};
  record.event = &LOOM_SERVE_TELEMETRY_SNAPSHOT;
  record.data.snapshot.counters = {1000, 990, 2, 3, 1, 16384, 4096, 2048};
  record.data.snapshot.queued = 4;
  record.data.snapshot.active = 16;
  std::memcpy(record.data.snapshot.phase, "executing", 10);
  return record;
}

static loom_serve_telemetry_stream_t* CreatePopulatedStream(size_t capacity) {
  loom_serve_telemetry_stream_t* stream = nullptr;
  IREE_CHECK_OK(loom_serve_telemetry_stream_create(capacity, &stream,
                                                   iree_allocator_system()));
  auto record = Snapshot();
  for (size_t i = 0; i < capacity; ++i) {
    record.time_ns = iree_time_now();
    ++record.data.snapshot.counters.batches;
    loom_serve_telemetry_stream_append(stream, &record);
  }
  return stream;
}

// One timed iteration captures and publishes one native snapshot. There is no
// consumer, formatting or sink; a full ring is overwritten on every iteration.
static void BM_PublishSnapshot(benchmark::State& state) {
  auto* stream = CreatePopulatedStream(state.range(0));
  auto record = Snapshot();
  record.data.snapshot.counters.batches += state.range(0);
  for (auto _ : state) {
    record.time_ns = iree_time_now();
    ++record.data.snapshot.counters.batches;
    loom_serve_telemetry_stream_append(stream, &record);
  }
  benchmark::DoNotOptimize(loom_serve_telemetry_stream_head(stream));
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * sizeof(record));
  loom_serve_telemetry_stream_destroy(stream);
}
BENCHMARK(BM_PublishSnapshot)->Arg(128)->Arg(256);

// One timed iteration copies and formats the maximum 64-record HTTP history
// batch. This measures consumer work, not publication, TCP or sink latency.
static void BM_CopyFormatBatch64(benchmark::State& state) {
  auto* stream = CreatePopulatedStream(state.range(0));
  const uint64_t start = loom_serve_telemetry_stream_head(stream) - 64;
  iree_string_builder_t output;
  iree_string_builder_initialize(iree_allocator_system(), &output);
  auto consume = [&]() {
    iree_string_builder_reset(&output);
    uint64_t cursor = start;
    for (size_t i = 0; i < 64; ++i) {
      loom_serve_telemetry_record_t record;
      const auto result =
          loom_serve_telemetry_stream_read(stream, &cursor, &record);
      if (result != LOOM_SERVE_TELEMETRY_READ_RECORD) {
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
  loom_serve_telemetry_stream_destroy(stream);
}
BENCHMARK(BM_CopyFormatBatch64)->Arg(128)->Arg(256);

}  // namespace
