// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/configuration.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(RealtimeConfigurationTest, ExecutionWeightIsExplicitPositivePolicy) {
  for (const uint32_t weight : {0u, 1u, 8u, UINT32_MAX}) {
    std::string document =
        R"({"name":"assistant","text":"text","recognition":"recognition",)"
        R"("speech":"speech","voice":"voice")";
    if (weight) {
      document += ",\"execution_weight\":" + std::to_string(weight);
    }
    document += "}";
    loom_serve_realtime_configuration_t configuration;
    IREE_ASSERT_OK(loom_serve_realtime_configuration_initialize(
        iree_make_string_view(document.data(), document.size()),
        iree_allocator_system(), &configuration));
    EXPECT_EQ(configuration.profile.execution_weight, weight ? weight : 1);
    loom_serve_realtime_configuration_deinitialize(&configuration);
  }
}

TEST(RealtimeConfigurationTest, RejectsInvalidExecutionWeight) {
  for (const char* value :
       {"0", "-1", "1.5", "4294967296", "null", "true", "\"8\""}) {
    SCOPED_TRACE(value);
    const std::string document =
        std::string(
            R"({"name":"assistant","text":"text","recognition":"recognition",)"
            R"("speech":"speech","voice":"voice","execution_weight":)") +
        value + "}";
    loom_serve_realtime_configuration_t configuration;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_realtime_configuration_initialize(
            iree_make_string_view(document.data(), document.size()),
            iree_allocator_system(), &configuration));
    loom_serve_realtime_configuration_deinitialize(&configuration);
  }
}

}  // namespace
