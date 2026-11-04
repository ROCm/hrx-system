// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/telemetry/sample.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(TelemetrySampleTest, ExactIntegersAndFieldSemantics) {
  const loom_runtime_telemetry_metric_t fields[] = {
      {"completed", "requests", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"resident", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE},
  };
  const loom_runtime_telemetry_topic_t topic = {"work", 1, 2, fields};
  const uint64_t values[] = {UINT64_MAX, 0};
  iree_string_builder_t output;
  iree_string_builder_initialize(iree_allocator_system(), &output);
  IREE_EXPECT_OK(
      loom_serve_telemetry_sample_write_json(&topic, values, &output));
  const auto view = iree_string_builder_view(&output);
  EXPECT_EQ(
      std::string(view.data, view.size),
      "{\"topic\":\"work\",\"version\":1,\"fields\":["
      "{\"name\":\"completed\",\"unit\":\"requests\",\"kind\":\"counter\","
      "\"value\":\"18446744073709551615\"},"
      "{\"name\":\"resident\",\"unit\":\"bytes\",\"kind\":\"gauge\","
      "\"value\":\"0\"}]}");
  iree_string_builder_deinitialize(&output);
}

TEST(TelemetrySampleTest, FixedOutputCapacityFailsExplicitly) {
  char storage[16];
  iree_string_builder_t output;
  iree_string_builder_initialize_with_storage(storage, sizeof(storage),
                                              &output);
  const loom_runtime_telemetry_topic_t topic = {"empty", 1, 0, nullptr};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_telemetry_sample_write_json(&topic, nullptr, &output));
  iree_string_builder_deinitialize(&output);
}

}  // namespace
