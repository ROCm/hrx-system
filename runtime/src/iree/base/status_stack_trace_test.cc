// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <atomic>
#include <string>
#include <thread>
#include <utility>

#include "iree/base/api.h"
#include "iree/base/status_cc.h"
#include "iree/base/threading/thread.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_STACK_TRACE) != 0 && \
    defined(IREE_HAVE_LIBBACKTRACE)

extern "C" IREE_ATTRIBUTE_NOINLINE iree_status_t
iree_status_stack_trace_test_capture_cold();

extern "C" IREE_ATTRIBUTE_NOINLINE iree_status_t
iree_status_stack_trace_test_capture_scoped(bool sibling);

namespace iree {
namespace {

IREE_ATTRIBUTE_NOINLINE iree_status_t CaptureWarmStatus() {
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "concurrent symbolization test");
}

TEST(StatusStackTraceTest, ClonesColdNestedScopesOnSmallWorkerStack) {
  // Debug information depth must not consume the reporting thread's stack.
  // Keep the scope fixture in its own compilation unit so other tests cannot
  // populate its function cache before the worker reaches it.
  std::array<Status, 2> results;
  iree_thread_create_params_t params = {.stack_size = 32 * 1024};
  iree_thread_t* thread = nullptr;
  IREE_ASSERT_OK(iree_thread_create(
      +[](void* user_data) -> int {
        auto& results = *static_cast<std::array<Status, 2>*>(user_data);
        for (size_t i = 0; i < results.size(); ++i) {
          Status original(iree_status_stack_trace_test_capture_scoped(i != 0));
          results[i] = Status(iree_status_clone(original.get()));
        }
        return 0;
      },
      &results, params, iree_allocator_system(), &thread));
  iree_thread_release(thread);

  for (size_t i = 0; i < results.size(); ++i) {
    SCOPED_TRACE(i);
    const Status& status = results[i];
    EXPECT_EQ(status.code(), StatusCode::kDataLoss);
    const std::string diagnostic = status.ToString();
    EXPECT_THAT(
        diagnostic,
        ::testing::AllOf(
            ::testing::HasSubstr("nested scope diagnostic"),
            ::testing::HasSubstr("preserved annotation"),
            ::testing::HasSubstr("(status_stack_trace_test_scopes.cc:"),
            ::testing::HasSubstr(i ? "CaptureSibling" : "CaptureDeepScope"),
            ::testing::Not(::testing::HasSubstr(i ? "CaptureDeepScope"
                                                  : "CaptureSibling"))));
    const size_t inner = diagnostic.find("InlineInner");
    const size_t outer = diagnostic.find("InlineOuter");
    ASSERT_NE(inner, std::string::npos) << diagnostic;
    ASSERT_NE(outer, std::string::npos) << diagnostic;
    EXPECT_LT(inner, outer) << diagnostic;
  }
}

TEST(StatusStackTraceTest, ConcurrentlySymbolizesColdCompilationUnit) {
  Status warm_status(CaptureWarmStatus());
  ASSERT_NE(warm_status.ToString().find("CaptureWarmStatus"),
            std::string::npos);

  constexpr size_t kThreadCount = 2;
  std::array<Status, kThreadCount> statuses = {
      Status(iree_status_stack_trace_test_capture_cold()),
      Status(iree_status_stack_trace_test_capture_cold()),
  };

  std::atomic<size_t> ready_count{0};
  std::atomic<bool> start{false};
  std::atomic<bool> all_symbolized{true};
  std::array<std::thread, kThreadCount> threads;
  for (size_t i = 0; i < kThreadCount; ++i) {
    threads[i] = std::thread([&, i] {
      ready_count.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      Status status(std::move(statuses[i]));
      if (status.ToString().find("iree_status_stack_trace_test_capture_cold") ==
          std::string::npos) {
        all_symbolized.store(false, std::memory_order_relaxed);
      }
    });
  }

  while (ready_count.load(std::memory_order_acquire) != kThreadCount) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (std::thread& thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(all_symbolized.load(std::memory_order_relaxed));
}

}  // namespace
}  // namespace iree

#endif  // stack traces with libbacktrace
