// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks for LEB128 varint encode/decode.
//
// Measures:
//   - Single-value encode/decode at different byte widths (1, 2, 5, 10).
//   - Batch encode/decode with a realistic value distribution.
//   - Signed vs unsigned overhead (zigzag cost).
//   - Length computation throughput.

#include <cstdint>
#include <cstring>

#include "benchmark/benchmark.h"
#include "loom/format/bytecode/varint.h"

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

constexpr int kBatchSize = 4096;

// Fills an array with values matching a realistic bytecode distribution:
// ~70% 1-byte (string IDs, type IDs, small counts), ~20% 2-byte (value refs),
// ~8% 3-4 byte (op counts, section sizes), ~2% 5+ byte (offsets, lengths).
static void FillRealisticValues(uint64_t* values, int count) {
  uint32_t state = 0xDEADBEEF;
  for (int i = 0; i < count; ++i) {
    state = state * 1103515245 + 12345;
    uint32_t magnitude = (state >> 16) & 0xFF;
    if (magnitude < 180) {
      values[i] = (state >> 8) & 0x7F;
    } else if (magnitude < 230) {
      values[i] = 128 + ((state >> 4) & 0x3F7F);
    } else if (magnitude < 250) {
      values[i] = 0x4000 + ((state >> 2) & 0xFFFFF);
    } else {
      values[i] = (uint64_t)(state) << 16 | (state >> 8);
    }
  }
}

//===----------------------------------------------------------------------===//
// Unsigned encode: single value at different byte widths
//===----------------------------------------------------------------------===//

static void BM_UvarintEncode_1Byte(benchmark::State& state) {
  uint8_t raw[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t buffer = iree_make_byte_span(raw, sizeof(raw));
  iree_host_size_t length = 0;
  for (auto _ : state) {
    IREE_CHECK_OK(loom_uvarint_encode(0x7F, buffer, &length));
    benchmark::DoNotOptimize(raw[0]);
  }
}
BENCHMARK(BM_UvarintEncode_1Byte);

static void BM_UvarintEncode_2Byte(benchmark::State& state) {
  uint8_t raw[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t buffer = iree_make_byte_span(raw, sizeof(raw));
  iree_host_size_t length = 0;
  for (auto _ : state) {
    IREE_CHECK_OK(loom_uvarint_encode(0x3FFF, buffer, &length));
    benchmark::DoNotOptimize(raw[0]);
  }
}
BENCHMARK(BM_UvarintEncode_2Byte);

static void BM_UvarintEncode_5Byte(benchmark::State& state) {
  uint8_t raw[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t buffer = iree_make_byte_span(raw, sizeof(raw));
  iree_host_size_t length = 0;
  for (auto _ : state) {
    IREE_CHECK_OK(loom_uvarint_encode(UINT32_MAX, buffer, &length));
    benchmark::DoNotOptimize(raw[0]);
  }
}
BENCHMARK(BM_UvarintEncode_5Byte);

static void BM_UvarintEncode_10Byte(benchmark::State& state) {
  uint8_t raw[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t buffer = iree_make_byte_span(raw, sizeof(raw));
  iree_host_size_t length = 0;
  for (auto _ : state) {
    IREE_CHECK_OK(loom_uvarint_encode(UINT64_MAX, buffer, &length));
    benchmark::DoNotOptimize(raw[0]);
  }
}
BENCHMARK(BM_UvarintEncode_10Byte);

//===----------------------------------------------------------------------===//
// Unsigned decode: single value at different byte widths
//===----------------------------------------------------------------------===//

static void BM_UvarintDecode_1Byte(benchmark::State& state) {
  uint8_t raw[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t buffer = iree_make_byte_span(raw, sizeof(raw));
  iree_host_size_t encoded_length = 0;
  IREE_CHECK_OK(loom_uvarint_encode(0x7F, buffer, &encoded_length));
  uint64_t decoded = 0;
  for (auto _ : state) {
    loom_bytecode_cursor_t cursor;
    loom_bytecode_cursor_initialize(raw, encoded_length, &cursor);
    IREE_CHECK_OK(loom_uvarint_decode(&cursor, &decoded));
    benchmark::DoNotOptimize(decoded);
  }
}
BENCHMARK(BM_UvarintDecode_1Byte);

static void BM_UvarintDecode_2Byte(benchmark::State& state) {
  uint8_t raw[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t buffer = iree_make_byte_span(raw, sizeof(raw));
  iree_host_size_t encoded_length = 0;
  IREE_CHECK_OK(loom_uvarint_encode(0x3FFF, buffer, &encoded_length));
  uint64_t decoded = 0;
  for (auto _ : state) {
    loom_bytecode_cursor_t cursor;
    loom_bytecode_cursor_initialize(raw, encoded_length, &cursor);
    IREE_CHECK_OK(loom_uvarint_decode(&cursor, &decoded));
    benchmark::DoNotOptimize(decoded);
  }
}
BENCHMARK(BM_UvarintDecode_2Byte);

static void BM_UvarintDecode_5Byte(benchmark::State& state) {
  uint8_t raw[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t buffer = iree_make_byte_span(raw, sizeof(raw));
  iree_host_size_t encoded_length = 0;
  IREE_CHECK_OK(loom_uvarint_encode(UINT32_MAX, buffer, &encoded_length));
  uint64_t decoded = 0;
  for (auto _ : state) {
    loom_bytecode_cursor_t cursor;
    loom_bytecode_cursor_initialize(raw, encoded_length, &cursor);
    IREE_CHECK_OK(loom_uvarint_decode(&cursor, &decoded));
    benchmark::DoNotOptimize(decoded);
  }
}
BENCHMARK(BM_UvarintDecode_5Byte);

static void BM_UvarintDecode_10Byte(benchmark::State& state) {
  uint8_t raw[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t buffer = iree_make_byte_span(raw, sizeof(raw));
  iree_host_size_t encoded_length = 0;
  IREE_CHECK_OK(loom_uvarint_encode(UINT64_MAX, buffer, &encoded_length));
  uint64_t decoded = 0;
  for (auto _ : state) {
    loom_bytecode_cursor_t cursor;
    loom_bytecode_cursor_initialize(raw, encoded_length, &cursor);
    IREE_CHECK_OK(loom_uvarint_decode(&cursor, &decoded));
    benchmark::DoNotOptimize(decoded);
  }
}
BENCHMARK(BM_UvarintDecode_10Byte);

//===----------------------------------------------------------------------===//
// Batch encode: realistic value distribution
//===----------------------------------------------------------------------===//

static void BM_UvarintEncode_Batch(benchmark::State& state) {
  uint64_t values[kBatchSize];
  FillRealisticValues(values, kBatchSize);
  uint8_t raw[kBatchSize * LOOM_VARINT_MAX_LENGTH];

  for (auto _ : state) {
    iree_host_size_t total = 0;
    for (int i = 0; i < kBatchSize; ++i) {
      iree_byte_span_t remaining =
          iree_make_byte_span(raw + total, sizeof(raw) - total);
      iree_host_size_t length = 0;
      IREE_CHECK_OK(loom_uvarint_encode(values[i], remaining, &length));
      total += length;
    }
    benchmark::DoNotOptimize(raw[0]);
  }
  state.SetItemsProcessed(state.iterations() * kBatchSize);
}
BENCHMARK(BM_UvarintEncode_Batch);

//===----------------------------------------------------------------------===//
// Batch decode: realistic value distribution
//===----------------------------------------------------------------------===//

static void BM_UvarintDecode_Batch(benchmark::State& state) {
  uint64_t values[kBatchSize];
  FillRealisticValues(values, kBatchSize);

  // Pre-encode into a contiguous buffer.
  uint8_t raw[kBatchSize * LOOM_VARINT_MAX_LENGTH];
  iree_host_size_t total = 0;
  for (int i = 0; i < kBatchSize; ++i) {
    iree_byte_span_t remaining =
        iree_make_byte_span(raw + total, sizeof(raw) - total);
    iree_host_size_t length = 0;
    IREE_CHECK_OK(loom_uvarint_encode(values[i], remaining, &length));
    total += length;
  }

  uint64_t decoded = 0;
  for (auto _ : state) {
    loom_bytecode_cursor_t cursor;
    loom_bytecode_cursor_initialize(raw, total, &cursor);
    for (int i = 0; i < kBatchSize; ++i) {
      IREE_CHECK_OK(loom_uvarint_decode(&cursor, &decoded));
    }
    benchmark::DoNotOptimize(decoded);
  }
  state.SetItemsProcessed(state.iterations() * kBatchSize);
}
BENCHMARK(BM_UvarintDecode_Batch);

//===----------------------------------------------------------------------===//
// Signed varint: zigzag overhead comparison
//===----------------------------------------------------------------------===//

static void BM_SvarintEncode_1Byte(benchmark::State& state) {
  uint8_t raw[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t buffer = iree_make_byte_span(raw, sizeof(raw));
  iree_host_size_t length = 0;
  for (auto _ : state) {
    IREE_CHECK_OK(loom_svarint_encode(-1, buffer, &length));
    benchmark::DoNotOptimize(raw[0]);
  }
}
BENCHMARK(BM_SvarintEncode_1Byte);

static void BM_SvarintDecode_1Byte(benchmark::State& state) {
  uint8_t raw[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t buffer = iree_make_byte_span(raw, sizeof(raw));
  iree_host_size_t encoded_length = 0;
  IREE_CHECK_OK(loom_svarint_encode(-1, buffer, &encoded_length));
  int64_t decoded = 0;
  for (auto _ : state) {
    loom_bytecode_cursor_t cursor;
    loom_bytecode_cursor_initialize(raw, encoded_length, &cursor);
    IREE_CHECK_OK(loom_svarint_decode(&cursor, &decoded));
    benchmark::DoNotOptimize(decoded);
  }
}
BENCHMARK(BM_SvarintDecode_1Byte);

static void BM_SvarintEncode_10Byte(benchmark::State& state) {
  uint8_t raw[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t buffer = iree_make_byte_span(raw, sizeof(raw));
  iree_host_size_t length = 0;
  for (auto _ : state) {
    IREE_CHECK_OK(loom_svarint_encode(INT64_MIN, buffer, &length));
    benchmark::DoNotOptimize(raw[0]);
  }
}
BENCHMARK(BM_SvarintEncode_10Byte);

static void BM_SvarintDecode_10Byte(benchmark::State& state) {
  uint8_t raw[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t buffer = iree_make_byte_span(raw, sizeof(raw));
  iree_host_size_t encoded_length = 0;
  IREE_CHECK_OK(loom_svarint_encode(INT64_MIN, buffer, &encoded_length));
  int64_t decoded = 0;
  for (auto _ : state) {
    loom_bytecode_cursor_t cursor;
    loom_bytecode_cursor_initialize(raw, encoded_length, &cursor);
    IREE_CHECK_OK(loom_svarint_decode(&cursor, &decoded));
    benchmark::DoNotOptimize(decoded);
  }
}
BENCHMARK(BM_SvarintDecode_10Byte);

//===----------------------------------------------------------------------===//
// Length computation throughput
//===----------------------------------------------------------------------===//

static void BM_UvarintLength_Batch(benchmark::State& state) {
  uint64_t values[kBatchSize];
  FillRealisticValues(values, kBatchSize);

  for (auto _ : state) {
    iree_host_size_t total = 0;
    for (int i = 0; i < kBatchSize; ++i) {
      total += loom_uvarint_length(values[i]);
    }
    benchmark::DoNotOptimize(total);
  }
  state.SetItemsProcessed(state.iterations() * kBatchSize);
}
BENCHMARK(BM_UvarintLength_Batch);

}  // namespace
