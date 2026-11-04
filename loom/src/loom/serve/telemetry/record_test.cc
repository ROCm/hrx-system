// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/telemetry/record.h"

#include <cstring>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/runtime/telemetry/field.h"
#include "loom/runtime/telemetry/stream.h"

namespace {

TEST(TelemetryRecord, FirstOutputIdentifiesTheRequestWithoutContent) {
  loom_runtime_telemetry_record_t record = {
      .time_ns = 71,
      .request = 5,
      .event = &LOOM_SERVICE_TELEMETRY_FIRST_OUTPUT,
  };
  const loom_service_telemetry_first_output_record_t first_output = {37};
  std::memcpy(record.data.words, &first_output, sizeof(first_output));
  iree_string_builder_t output;
  iree_string_builder_initialize(iree_allocator_system(), &output);
  IREE_ASSERT_OK(loom_serve_telemetry_record_format(2, &record, &output));
  const auto text = iree_string_builder_view(&output);
  EXPECT_EQ(
      std::string(text.data, text.size),
      "{\"event\":\"first_output\",\"sequence\":2,\"time_ns\":71,\"request\":5,"
      "\"duration_ns\":37}\n");
  iree_string_builder_deinitialize(&output);
}

TEST(TelemetryRecord, DiagnosticsOutliveTheirBorrowedStatus) {
  const std::string message(200, 'x');
  iree_status_t status =
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", message.c_str());
  loom_service_telemetry_diagnostic_t diagnostic;
  loom_service_telemetry_diagnostic_capture(status, &diagnostic);
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
  loom_service_telemetry_diagnostic_capture(iree_ok_status(), &diagnostic);
  EXPECT_EQ(diagnostic.code, 0u);
  EXPECT_EQ(diagnostic.length, 0u);
  EXPECT_EQ(diagnostic.truncated, 0u);
}

TEST(TelemetryRecord, PublicationDoesNotRenderOrBorrowProducerPayload) {
  static int render_count;
  render_count = 0;
  static const loom_runtime_telemetry_event_t event = {
      "epoch", [](const loom_runtime_telemetry_record_t* record,
                  loom_runtime_telemetry_field_sink_t sink) {
        ++render_count;
        const auto field = loom_runtime_telemetry_field_uint64(
            "useful_tokens", record->data.words[0]);
        return sink.emit(sink.self, 1, &field);
      }};
  loom_runtime_telemetry_stream_t* stream = nullptr;
  IREE_ASSERT_OK(loom_runtime_telemetry_stream_allocate(
      2, iree_allocator_system(), &stream));
  loom_runtime_telemetry_record_t record = {
      .time_ns = 19,
      .request = 5,
      .event = &event,
  };
  record.data.words[0] = 37;
  loom_runtime_telemetry_stream_append(stream, &record);
  record.data.words[0] = 0;
  EXPECT_EQ(render_count, 0);
  uint64_t cursor = 0;
  loom_runtime_telemetry_record_t copy;
  ASSERT_EQ(loom_runtime_telemetry_stream_read(stream, &cursor, &copy),
            LOOM_RUNTIME_TELEMETRY_READ_RECORD);
  loom_runtime_telemetry_stream_free(stream);
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
