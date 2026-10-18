// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/tracing.h"

#include <array>
#include <atomic>
#include <string>
#include <thread>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(MemoryTraceTest, EqualNamesShareAnIdentityBeyondCallerLifetime) {
  std::string name = "memory-trace-test-transient-name";
  iree_hal_memory_trace_t first = {};
  IREE_ASSERT_OK(iree_hal_memory_trace_initialize(
      iree_make_string_view(name.data(), name.size()), "unused", &first));
  name.assign(name.size(), '?');
  iree_hal_memory_trace_deinitialize(&first);
  iree_hal_memory_trace_t second = {};
  IREE_ASSERT_OK(iree_hal_memory_trace_initialize(
      iree_string_view_empty(), "memory-trace-test-transient-name", &second));
  EXPECT_EQ(first.memory_id, second.memory_id);
  if (IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING) {
    ASSERT_NE(second.memory_id, nullptr);
    EXPECT_STREQ(second.memory_id, "memory-trace-test-transient-name");
  } else {
    EXPECT_EQ(second.memory_id, nullptr);
  }
}

TEST(MemoryTraceTest, DomainsDistinguishPlacementAndShareProviders) {
  iree_hal_memory_trace_t first = {}, same = {}, other = {}, unspecified = {};
  IREE_ASSERT_OK(
      iree_hal_memory_trace_initialize_domain("libamdf-numa", 0, &first));
  IREE_ASSERT_OK(
      iree_hal_memory_trace_initialize_domain("libamdf-numa", 0, &same));
  IREE_ASSERT_OK(
      iree_hal_memory_trace_initialize_domain("libamdf-numa", 1, &other));
  IREE_ASSERT_OK(iree_hal_memory_trace_initialize_domain(
      "libamdf-system", UINT32_MAX, &unspecified));
  EXPECT_EQ(first.memory_id, same.memory_id);
  if (IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING) {
    EXPECT_NE(first.memory_id, other.memory_id);
    EXPECT_STREQ(first.memory_id, "libamdf-numa-0");
    EXPECT_STREQ(other.memory_id, "libamdf-numa-1");
    EXPECT_STREQ(unspecified.memory_id, "libamdf-system");
  }
}

TEST(MemoryTraceTest, ConcurrentConstructionSharesOnePublishedIdentity) {
  std::atomic<bool> start{false};
  std::array<iree_hal_memory_trace_t, 8> traces = {};
  std::array<std::thread, 8> threads;
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread([&, i] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (uint32_t ordinal = 0; ordinal < 32; ++ordinal) {
        IREE_ASSERT_OK(iree_hal_memory_trace_initialize_domain(
            "memory-trace-concurrent", ordinal, &traces[i]));
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }
  for (const auto& trace : traces) {
    EXPECT_EQ(traces[0].memory_id, trace.memory_id);
  }
  if (IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING) {
    EXPECT_STREQ(traces[0].memory_id, "memory-trace-concurrent-31");
  }
}

TEST(MemoryTraceTest, LogicalPoolStreamsAreOptional) {
  iree_hal_memory_trace_t trace = {};
  IREE_ASSERT_OK(iree_hal_memory_trace_initialize_pool(
      IREE_SV("memory-trace-logical-pool"), "unused", &trace));
  const bool enabled = IREE_HAL_MEMORY_TRACE_ENABLE_POOL_STREAMS
                           ? (IREE_TRACING_FEATURES &
                              IREE_TRACING_FEATURE_ALLOCATION_TRACKING) != 0
                           : false;
  EXPECT_EQ(trace.memory_id != nullptr, enabled);
  if (enabled) {
    EXPECT_STREQ(trace.memory_id, "memory-trace-logical-pool");
  } else {
    EXPECT_EQ(trace.memory_id, nullptr);
  }
}

}  // namespace
