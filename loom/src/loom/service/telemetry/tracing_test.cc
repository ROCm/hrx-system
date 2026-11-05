// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/telemetry/tracing.h"

#include <string>

#include "iree/testing/gtest.h"

namespace {

TEST(ServingTracingTest, DeploymentNamesLabelEachOperation) {
  const struct {
    // Elapsed operation selected by its serving owner.
    loom_service_trace_section_t section;
    // Category and display-label prefix for this operation.
    const char* label;
  } cases[] = {
      {LOOM_SERVICE_TRACE_SECTION_PREPARE, "model-prepare"},
      {LOOM_SERVICE_TRACE_SECTION_WARMUP, "model-warmup"},
      {LOOM_SERVICE_TRACE_SECTION_DEACTIVATE, "model-deactivate"},
      {LOOM_SERVICE_TRACE_SECTION_RETIRE, "model-retire"},
      {LOOM_SERVICE_TRACE_SECTION_EXECUTE, "model-execute"},
  };
  ::testing::internal::CaptureStderr();
  loom_service_trace_initialize();
  loom_service_trace_initialize();
  const std::string categories = ::testing::internal::GetCapturedStderr();
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.label);
    const std::string category = std::to_string(test_case.section);
    EXPECT_THAT(categories,
                ::testing::HasSubstr("section category=" + category + " " +
                                     test_case.label + "\n"));
    ::testing::internal::CaptureStderr();
    const iree_section_id_t first =
        loom_service_trace_section_enter(test_case.section, IREE_SV("scout"));
    char name[] = {'1', '0', '0', '%'};
    const iree_section_id_t second = loom_service_trace_section_enter(
        test_case.section, iree_make_string_view(name, sizeof(name)));
    name[0] = 'x';
    IREE_TRACE_SECTION_LEAVE(first);
    IREE_TRACE_SECTION_LEAVE(second);
    const std::string events = ::testing::internal::GetCapturedStderr();
    EXPECT_NE(first, 0u);
    EXPECT_NE(second, 0u);
    EXPECT_NE(first, second);
    EXPECT_THAT(events, ::testing::HasSubstr(
                            "ns " + std::string(test_case.label) + ":scout\n"));
    EXPECT_THAT(events, ::testing::HasSubstr(
                            "ns " + std::string(test_case.label) + ":100%\n"));
    EXPECT_THAT(events, ::testing::HasSubstr(
                            "section " + std::to_string(first) + " leave t="));
    EXPECT_THAT(events, ::testing::HasSubstr(
                            "section " + std::to_string(second) + " leave t="));
  }
}

}  // namespace
