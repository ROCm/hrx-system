// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/telemetry/pool.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(TelemetryPoolTest, NativeConsumerReceivesTypedOwnedValues) {
  // The independent atomic observations need not form a coherent total.
  iree_hal_pool_stats_t statistics = {
      .bytes_reserved = 1,
      .bytes_free = 3,
      .bytes_quarantined = 4,
      .bytes_committed = 2,
      .bytes_abandoned = 5,
      .budget_limit = 6,
      .reservation_count = 7,
      .slab_count = 8,
      .reserve_count = 9,
      .release_count = 10,
      .reuse_count = 11,
      .reuse_miss_count = 12,
      .fresh_count = 13,
      .exhausted_count = 14,
      .over_budget_count = 15,
      .wait_count = 16,
      .quarantine_eviction_count = UINT64_MAX,
  };
  std::vector<uint64_t> captured;
  IREE_ASSERT_OK(loom_runtime_telemetry_sample_pool(
      "storage", &statistics,
      {&captured, [](void* self, const loom_runtime_telemetry_topic_t* topic,
                     const uint64_t* values) {
         static const char* const names[] = {
             "reserved_bytes",
             "committed_bytes",
             "free_bytes",
             "quarantined_bytes",
             "abandoned_bytes",
             "budget_limit",
             "reservation_count",
             "slab_count",
             "reserve_count",
             "release_count",
             "reuse_count",
             "reuse_miss_count",
             "fresh_count",
             "exhausted_count",
             "over_budget_count",
             "wait_count",
             "quarantine_eviction_count",
         };
         EXPECT_STREQ(topic->name, "storage");
         EXPECT_EQ(topic->version, 1u);
         EXPECT_EQ(topic->count, IREE_ARRAYSIZE(names));
         for (iree_host_size_t i = 0;
              i < topic->count && i < IREE_ARRAYSIZE(names); ++i) {
           EXPECT_STREQ(topic->fields[i].name, names[i]);
           EXPECT_STREQ(topic->fields[i].unit, i < 6    ? "bytes"
                                               : i == 7 ? "slabs"
                                                        : "reservations");
           EXPECT_EQ(topic->fields[i].kind,
                     i < 8 ? LOOM_RUNTIME_TELEMETRY_GAUGE
                           : LOOM_RUNTIME_TELEMETRY_COUNTER);
         }
         auto* captured = static_cast<std::vector<uint64_t>*>(self);
         EXPECT_TRUE(captured->empty());
         captured->assign(values, values + topic->count);
         return iree_ok_status();
       }}));
  statistics = {};
  ASSERT_EQ(captured.size(), 17u);
  for (size_t i = 0; i < captured.size(); ++i) {
    EXPECT_EQ(captured[i], i == 16 ? UINT64_MAX : i + 1);
  }
}

TEST(TelemetryPoolTest, ConsumerFailureRetainsOriginalStatus) {
  const iree_hal_pool_stats_t statistics = {};
  iree_status_t status = loom_runtime_telemetry_sample_pool(
      "workspace", &statistics,
      {nullptr, [](void*, const loom_runtime_telemetry_topic_t* topic,
                   const uint64_t* values) {
         EXPECT_STREQ(topic->name, "workspace");
         for (iree_host_size_t i = 0; i < topic->count; ++i) {
           EXPECT_EQ(values[i], 0u);
         }
         return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                 "observation output is full");
       }});
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_RESOURCE_EXHAUSTED);
  const iree_string_view_t message = iree_status_message(status);
  EXPECT_EQ(std::string(message.data, message.size),
            "observation output is full");
  iree_status_free(status);
}

}  // namespace
