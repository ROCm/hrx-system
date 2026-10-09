// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/telemetry/stream.h"

#include <atomic>
#include <cstring>
#include <thread>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TelemetryStreamTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(loom_serve_telemetry_stream_create(4, &stream_,
                                                      iree_allocator_system()));
  }
  void TearDown() override { loom_serve_telemetry_stream_destroy(stream_); }

  // Every word depends on position; a mixed-generation copy fails comparison.
  static loom_serve_telemetry_record_t Record(uint64_t position) {
    loom_serve_telemetry_record_t record;
    for (size_t i = 0; i < sizeof(record) / sizeof(uint64_t); ++i) {
      const uint64_t word = position ^ (UINT64_C(0x9e3779b97f4a7c15) * (i + 1));
      memcpy(reinterpret_cast<uint8_t*>(&record) + i * sizeof(word), &word,
             sizeof(word));
    }
    return record;
  }
  void Append(uint64_t position) {
    const auto record = Record(position);
    loom_serve_telemetry_stream_append(stream_, &record);
  }

  // Producer storage shared by independent test-owned reader cursors.
  loom_serve_telemetry_stream_t* stream_ = nullptr;
};

TEST_F(TelemetryStreamTest, IndependentReadersReportTheirOwnGaps) {
  uint64_t fast = 0, slow = 0;
  loom_serve_telemetry_record_t output;
  EXPECT_EQ(loom_serve_telemetry_stream_read(stream_, &fast, &output),
            LOOM_SERVE_TELEMETRY_READ_EMPTY);
  for (uint64_t i = 0; i < 17; ++i) {
    Append(i);
    ASSERT_EQ(loom_serve_telemetry_stream_read(stream_, &fast, &output),
              LOOM_SERVE_TELEMETRY_READ_RECORD);
    const auto expected = Record(i);
    EXPECT_EQ(memcmp(&output, &expected, sizeof(output)), 0);
    EXPECT_EQ(fast, i + 1);
  }
  EXPECT_EQ(loom_serve_telemetry_stream_head(stream_), 17u);
  EXPECT_EQ(loom_serve_telemetry_stream_capacity(stream_), 4u);
  ASSERT_EQ(loom_serve_telemetry_stream_read(stream_, &slow, &output),
            LOOM_SERVE_TELEMETRY_READ_GAP);
  EXPECT_EQ(slow, 13u);
  uint64_t joining = 0;
  ASSERT_EQ(loom_serve_telemetry_stream_read(stream_, &joining, &output),
            LOOM_SERVE_TELEMETRY_READ_GAP);
  EXPECT_EQ(joining, slow);
  for (; slow < 17;) {
    const auto expected = Record(slow);
    ASSERT_EQ(loom_serve_telemetry_stream_read(stream_, &slow, &output),
              LOOM_SERVE_TELEMETRY_READ_RECORD);
    EXPECT_EQ(memcmp(&output, &expected, sizeof(output)), 0);
  }
  // Another reader neither consumes nor acknowledges these retained records.
  EXPECT_EQ(joining, 13u);
  EXPECT_EQ(loom_serve_telemetry_stream_read(stream_, &joining, &output),
            LOOM_SERVE_TELEMETRY_READ_RECORD);
  EXPECT_EQ(joining, 14u);
  EXPECT_EQ(fast, 17u);
}

TEST_F(TelemetryStreamTest, OverwriteDuringCopyNeverReturnsTornRecord) {
  constexpr uint64_t kRecords = 50000;
  Append(0);
  std::atomic<bool> readers_started[2] = {};
  std::atomic<bool> done{false};
  std::atomic<bool> torn{false};
  uint64_t totals[2] = {}, gaps[2] = {};
  auto consume = [&](size_t index) {
    uint64_t cursor = 0;
    readers_started[index].store(true);
    while (!done.load() || cursor < loom_serve_telemetry_stream_head(stream_)) {
      const uint64_t position = cursor;
      loom_serve_telemetry_record_t output;
      switch (loom_serve_telemetry_stream_read(stream_, &cursor, &output)) {
        case LOOM_SERVE_TELEMETRY_READ_RECORD: {
          const auto expected = Record(position);
          if (memcmp(&output, &expected, sizeof(output))) {
            torn.store(true);
          }
          ++totals[index];
          break;
        }
        case LOOM_SERVE_TELEMETRY_READ_GAP:
          gaps[index] += cursor - position;
          break;
        case LOOM_SERVE_TELEMETRY_READ_EMPTY:
        case LOOM_SERVE_TELEMETRY_READ_BUSY:
          std::this_thread::yield();
          break;
      }
    }
  };
  std::thread first(consume, 0), second(consume, 1);
  while (!readers_started[0].load() || !readers_started[1].load()) {
    std::this_thread::yield();
  }
  for (uint64_t i = 1; i < kRecords; ++i) {
    Append(i);
  }
  done.store(true);
  first.join();
  second.join();
  EXPECT_FALSE(torn.load());
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_GT(totals[i], 0u);
    EXPECT_EQ(totals[i] + gaps[i], kRecords);
  }
}

TEST_F(TelemetryStreamTest, IncarnationsAreDistinctAndDiagnosticsAreOwned) {
  loom_serve_telemetry_stream_t* other = nullptr;
  IREE_ASSERT_OK(
      loom_serve_telemetry_stream_create(1, &other, iree_allocator_system()));
  EXPECT_NE(loom_serve_telemetry_stream_identity(stream_).ordinal,
            loom_serve_telemetry_stream_identity(other).ordinal);
  loom_serve_telemetry_stream_destroy(other);
  const std::string message(200, 'x');
  iree_status_t status =
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", message.c_str());
  loom_serve_telemetry_diagnostic_t diagnostic;
  loom_serve_telemetry_diagnostic_capture(status, &diagnostic);
  iree_status_free(status);
  EXPECT_EQ(diagnostic.code, IREE_STATUS_INVALID_ARGUMENT);
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS)
  EXPECT_EQ(diagnostic.length, sizeof(diagnostic.message));
  EXPECT_EQ(diagnostic.truncated, 1u);
  EXPECT_EQ(std::string(diagnostic.message, diagnostic.length),
            message.substr(0, diagnostic.length));
#endif
  loom_serve_telemetry_diagnostic_capture(iree_ok_status(), &diagnostic);
  EXPECT_EQ(diagnostic.code, 0u);
  EXPECT_EQ(diagnostic.length, 0u);
  EXPECT_EQ(diagnostic.truncated, 0u);
}

TEST(TelemetryStream, RejectsInvalidCapacity) {
  for (iree_host_size_t capacity : {0u, 3u}) {
    loom_serve_telemetry_stream_t* stream = nullptr;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_telemetry_stream_create(
                              capacity, &stream, iree_allocator_system()));
    EXPECT_EQ(stream, nullptr);
  }
}

}  // namespace
