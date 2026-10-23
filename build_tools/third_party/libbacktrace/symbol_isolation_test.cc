// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <cstdlib>
#include <string>

#include "iree/base/api.h"
#include "iree/base/status_cc.h"
#include "iree/testing/gtest.h"

// An embedding application can supply the original public and implementation
// symbols. Neither may interpose IREE's pinned stack-trace implementation.
static std::atomic<int> application_create_count{0};
static std::atomic<int> application_sort_count{0};

struct backtrace_state;
extern "C" struct backtrace_state* backtrace_create_state(
    const char*, int, void (*)(void*, const char*, int), void*) {
  application_create_count.fetch_add(1, std::memory_order_relaxed);
  return nullptr;
}

extern "C" void backtrace_qsort(void* base, size_t count, size_t size,
                                int (*compare)(const void*, const void*)) {
  application_sort_count.fetch_add(1, std::memory_order_relaxed);
  std::qsort(base, count, size, compare);
}

namespace iree {
namespace {

IREE_ATTRIBUTE_NOINLINE iree_status_t CaptureIsolatedStatus() {
  return iree_make_status(IREE_STATUS_INTERNAL, "private stack trace");
}

TEST(LibbacktraceTest, ApplicationSymbolsCannotInterposePrivateImplementation) {
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_STACK_TRACE) == 0
  GTEST_SKIP() << "Status stack traces are disabled in this configuration.";
#endif
  Status status(CaptureIsolatedStatus());
  std::string message = status.ToString();
  EXPECT_NE(message.find("CaptureIsolatedStatus"), std::string::npos)
      << message;
  EXPECT_EQ(application_create_count.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(application_sort_count.load(std::memory_order_relaxed), 0);
}

}  // namespace
}  // namespace iree
