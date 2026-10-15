// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/base/tracing.h"
#include "iree/testing/gtest.h"

namespace {

TEST(ConsoleTracingTest, DynamicNameBoundaries) {
  for (size_t length : {0, 1, 63, 64, 65, 1024}) {
    SCOPED_TRACE(length);
    std::string name(length, 'x');
    ::testing::internal::CaptureStderr();
    IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(zone, name.data(), name.size());
    // The caller may reuse the name storage before ending the zone.
    name.assign(length, 'y');
    IREE_TRACE_ZONE_END(zone);
    std::string output = ::testing::internal::GetCapturedStderr();
    EXPECT_THAT(output, ::testing::HasSubstr("► " + std::string(length, 'x') +
                                             " (console_test.cc:"));
#if IREE_TRACING_CONSOLE_TIMING
    const size_t retained_length = length < 64 ? length : 64;
    EXPECT_THAT(output,
                ::testing::HasSubstr("◄ " + std::string(retained_length, 'x') +
                                     " t = "));
#else
    EXPECT_THAT(output, ::testing::Not(::testing::HasSubstr("◄")));
#endif  // IREE_TRACING_CONSOLE_TIMING
  }
}

TEST(ConsoleTracingTest, NestedNames) {
  ::testing::internal::CaptureStderr();
  IREE_TRACE_ZONE_BEGIN_NAMED(outer, "outer");
  IREE_TRACE_ZONE_BEGIN_NAMED(inner, "inner");
  IREE_TRACE_ZONE_END(inner);
  IREE_TRACE_ZONE_END(outer);
  std::string output = ::testing::internal::GetCapturedStderr();
  EXPECT_THAT(output, ::testing::HasSubstr("► outer (console_test.cc:"));
  EXPECT_THAT(output, ::testing::HasSubstr(" ► inner (console_test.cc:"));
#if IREE_TRACING_CONSOLE_TIMING
  EXPECT_THAT(output, ::testing::HasSubstr(" ◄ inner t = "));
  EXPECT_THAT(output, ::testing::HasSubstr("◄ outer t = "));
  EXPECT_LT(output.find("◄ inner"), output.find("◄ outer"));
#else
  EXPECT_THAT(output, ::testing::Not(::testing::HasSubstr("◄")));
#endif  // IREE_TRACING_CONSOLE_TIMING
}

TEST(ConsoleTracingTest, MessagesAndAllocations) {
  std::string allocation(128, 'x');
  ::testing::internal::CaptureStderr();
  IREE_TRACE_MESSAGE(INFO, "embedded message");
  IREE_TRACE_MESSAGE_DYNAMIC(INFO, allocation.data(), 8);
  IREE_TRACE_ALLOC(allocation.data(), allocation.size());
  IREE_TRACE_FREE(allocation.data());
  IREE_TRACE_APP_EXIT(0);
  std::string output = ::testing::internal::GetCapturedStderr();
  EXPECT_THAT(output, ::testing::HasSubstr("i embedded message\n"));
  EXPECT_THAT(output, ::testing::HasSubstr("i xxxxxxxx\n"));
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
  EXPECT_THAT(output, ::testing::HasSubstr("heap alloc "));
  EXPECT_THAT(output, ::testing::HasSubstr("heap free "));
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING
}

}  // namespace

int main(int argc, char** argv) {
  // Model an embedding application that does not initialize runtime tracing.
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
