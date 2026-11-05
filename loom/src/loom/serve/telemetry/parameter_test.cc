// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/runtime/parameter/telemetry.h"
#include "loom/serve/telemetry/record.h"

namespace {

TEST(ParameterTelemetry, JsonConsumesOwnedNativeActivity) {
  loom_runtime_telemetry_stream_t* stream = nullptr;
  IREE_ASSERT_OK(loom_runtime_telemetry_stream_allocate(
      2, iree_allocator_system(), &stream));
  loom_runtime_parameter_activity_t activity = {3, 1, 100, 40, 5};
  loom_runtime_parameter_record_activity(stream, 17, &activity);
  activity = {};
  uint64_t cursor = 0;
  loom_runtime_telemetry_record_t record;
  ASSERT_EQ(loom_runtime_telemetry_stream_read(stream, &cursor, &record),
            LOOM_RUNTIME_TELEMETRY_READ_RECORD);
  loom_runtime_telemetry_stream_free(stream);
  iree_string_builder_t output;
  iree_string_builder_initialize(iree_allocator_system(), &output);
  IREE_ASSERT_OK(loom_serve_telemetry_record_format(0, &record, &output));
  const auto text = iree_string_builder_view(&output);
  EXPECT_EQ(std::string(text.data, text.size),
            "{\"event\":\"parameter_activity\",\"sequence\":0,\"time_ns\":17,"
            "\"request\":0,\"activations\":3,\"deactivations\":1,"
            "\"read_bytes\":100,\"prepared_bytes\":40,\"groups\":5}\n");
  iree_string_builder_deinitialize(&output);
}

}  // namespace
