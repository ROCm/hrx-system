// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/telemetry/record.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/telemetry/stream.h"

namespace {

TEST(TelemetryRecord, DiagnosticsOutliveTheirBorrowedStatus) {
  const std::string message(200, 'x');
  iree_status_t status =
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", message.c_str());
  loom_serve_telemetry_diagnostic_t diagnostic;
  loom_serve_telemetry_diagnostic_capture(status, &diagnostic);
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS)
  const auto original_message = iree_status_message(status);
  EXPECT_EQ(std::string(original_message.data, original_message.size), message);
#endif
  iree_status_free(status);
  EXPECT_EQ(diagnostic.code, IREE_STATUS_INVALID_ARGUMENT);
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS)
  EXPECT_EQ(diagnostic.length, sizeof(diagnostic.message));
  EXPECT_EQ(diagnostic.truncated, 1u);
  EXPECT_EQ(std::string(diagnostic.message, diagnostic.length),
            message.substr(0, diagnostic.length));
#endif
  loom_serve_telemetry_diagnostic_capture(iree_ok_status(), &diagnostic);
  EXPECT_EQ(diagnostic.code, 0u);
  EXPECT_EQ(diagnostic.length, 0u);
  EXPECT_EQ(diagnostic.truncated, 0u);
}

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
      .time_ns = 19,
      .request = 5,
      .event = &event,
  };
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
