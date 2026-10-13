// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/telemetry.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(WeightTelemetry, RecordOwnsActivityAcrossProducerRetirement) {
  loom_serve_telemetry_stream_t* stream = nullptr;
  IREE_ASSERT_OK(
      loom_serve_telemetry_stream_create(2, &stream, iree_allocator_system()));
  loom_serve_weight_activity_t activity = {3, 1, 100, 40, 5};
  loom_serve_telemetry_record_weights(stream, 17, &activity);
  activity = {};
  uint64_t cursor = 0;
  loom_serve_telemetry_record_t record;
  ASSERT_EQ(loom_serve_telemetry_stream_read(stream, &cursor, &record),
            LOOM_SERVE_TELEMETRY_READ_RECORD);
  loom_serve_telemetry_stream_destroy(stream);
  iree_string_builder_t output;
  iree_string_builder_initialize(iree_allocator_system(), &output);
  IREE_ASSERT_OK(loom_serve_telemetry_record_format(0, &record, &output));
  const auto text = iree_string_builder_view(&output);
  EXPECT_EQ(std::string(text.data, text.size),
            "{\"event\":\"weight_activity\",\"sequence\":0,\"time_ns\":17,"
            "\"request\":0,\"activations\":3,\"deactivations\":1,"
            "\"read_bytes\":100,\"prepared_bytes\":40,\"groups\":5}\n");
  iree_string_builder_deinitialize(&output);
}

}  // namespace
