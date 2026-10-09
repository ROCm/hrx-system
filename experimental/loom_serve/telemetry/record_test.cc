// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/telemetry/record.h"

#include <string>

#include "experimental/loom_serve/telemetry/stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(TelemetryRecord, PublicationDoesNotRenderOrBorrowProducerPayload) {
  static int render_count;
  render_count = 0;
  static const loom_serve_telemetry_event_t event = {
      "epoch", [](const loom_serve_telemetry_record_t* record,
                  iree_string_builder_t* output) {
        ++render_count;
        return iree_string_builder_append_format(
            output, ",\"useful_tokens\":%" PRIu64, record->data.words[0]);
      }};
  loom_serve_telemetry_stream_t* stream = nullptr;
  IREE_ASSERT_OK(
      loom_serve_telemetry_stream_create(2, &stream, iree_allocator_system()));
  loom_serve_telemetry_record_t record = {
      .time_ns = 19, .request = 5, .event = &event};
  record.data.words[0] = 37;
  loom_serve_telemetry_stream_append(stream, &record);
  record.data.words[0] = 0;
  EXPECT_EQ(render_count, 0);
  uint64_t cursor = 0;
  loom_serve_telemetry_record_t copy;
  ASSERT_EQ(loom_serve_telemetry_stream_read(stream, &cursor, &copy),
            LOOM_SERVE_TELEMETRY_READ_RECORD);
  loom_serve_telemetry_stream_destroy(stream);
  iree_string_builder_t output;
  iree_string_builder_initialize(iree_allocator_system(), &output);
  IREE_ASSERT_OK(loom_serve_telemetry_record_format(0, &copy, &output));
  EXPECT_EQ(render_count, 1);
  const auto text = iree_string_builder_view(&output);
  EXPECT_EQ(std::string(text.data, text.size),
            "{\"event\":\"epoch\",\"sequence\":0,\"time_ns\":19,\"request\":5,"
            "\"useful_tokens\":37}\n");
  iree_string_builder_deinitialize(&output);
}

}  // namespace
