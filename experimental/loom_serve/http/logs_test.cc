// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/http/logs.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class LogsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_string_builder_initialize(iree_allocator_system(), &output_);
    IREE_ASSERT_OK(loom_serve_telemetry_stream_create(128, &stream_,
                                                      iree_allocator_system()));
    const auto identity = loom_serve_telemetry_stream_identity(stream_);
    prefix_ = std::to_string(identity.created_ns) + ":" +
              std::to_string(identity.ordinal) + ":";
  }
  void TearDown() override {
    loom_serve_telemetry_stream_destroy(stream_);
    iree_string_builder_deinitialize(&output_);
  }
  iree_status_t Poll(const std::string& cursor = {}) {
    return loom_serve_http_logs(stream_, &counters_, IREE_SV("fixture"),
                                iree_make_cstring_view(cursor.c_str()),
                                &output_);
  }
  std::string Output() {
    const auto view = iree_string_builder_view(&output_);
    return std::string(view.data, view.size);
  }

  // Real bounded publisher used by the HTTP serializer.
  loom_serve_telemetry_stream_t* stream_ = nullptr;
  // Owner-only cumulative state, deliberately independent of event history.
  loom_serve_telemetry_counters_t counters_ = {};
  // Consumer-owned copied HTTP bytes.
  iree_string_builder_t output_;
  // Current stream incarnation in the HTTP cursor representation.
  std::string prefix_;
};

TEST_F(LogsTest, BoundsPollWorkAndPreservesIndependentTotals) {
  const loom_serve_telemetry_record_t record = {
      .time_ns = 12, .request = 7, .event = LOOM_SERVE_TELEMETRY_REQUEST_BEGIN};
  for (int i = 0; i < 80; ++i) {
    loom_serve_telemetry_stream_append(stream_, &record);
  }
  counters_.requests = 999;
  IREE_ASSERT_OK(Poll());
  EXPECT_NE(Output().find(R"("sequence":63,"time_ns":12,"request":7)"),
            std::string::npos);
  EXPECT_EQ(Output().find("\"sequence\":64,"), std::string::npos);
  EXPECT_NE(Output().find("\"value\":\"" + prefix_ + "64\",\"next\":64"),
            std::string::npos);
  EXPECT_NE(Output().find(R"("values":{"requests":999)"), std::string::npos);
  IREE_ASSERT_OK(Poll(prefix_ + "64"));
  EXPECT_NE(Output().find("\"sequence\":79,"), std::string::npos);
  EXPECT_EQ(Output().find("\"sequence\":63,"), std::string::npos);
  IREE_ASSERT_OK(Poll(prefix_ + "80"));
  EXPECT_EQ(Output().find("\"sequence\":"), std::string::npos);
  EXPECT_NE(Output().find(R"("values":{"requests":999)"), std::string::npos);
}

TEST_F(LogsTest, RejectsMalformedFutureAndDifferentLifetimeCursors) {
  for (const char* cursor : {"1", "1:2", "1:2:", "1:2:3:4", "-1:2:3", "1:2:+3",
                             "1:2: 3", "1:2:18446744073709551616"}) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Poll(cursor));
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Poll(prefix_ + "1"));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, Poll("0:0:0"));
}

TEST_F(LogsTest, CopiesEscapedErrorBeforeStatusRetires) {
  loom_serve_telemetry_record_t record = {
      .time_ns = 42,
      .request = 9,
      .event = LOOM_SERVE_TELEMETRY_REQUEST_END,
      .data = {.end = {.outcome = LOOM_SERVE_TELEMETRY_REJECTED,
                       .duration_ns = 17,
                       .input_tokens = 3}}};
  iree_status_t status =
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "bad \"value\"\n");
  loom_serve_telemetry_diagnostic_capture(status, &record.data.end.diagnostic);
  iree_status_free(status);
  loom_serve_telemetry_stream_append(stream_, &record);
  IREE_ASSERT_OK(Poll());
  EXPECT_NE(
      Output().find(
          R"({"event":"request_end","sequence":0,"time_ns":42,"request":9,"outcome":"rejected","duration_ns":17,"input_tokens":3,"status":3,"message_truncated":false,"message":)"),
      std::string::npos);
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS)
  EXPECT_NE(Output().find(R"("message":"bad \"value\"\n"})"),
            std::string::npos);
#endif
}

TEST_F(LogsTest, DiagnosticOnlyScopeDoesNotInventCounters) {
  const loom_serve_telemetry_record_t record = {
      .time_ns = 10, .event = LOOM_SERVE_TELEMETRY_STREAM_BEGIN};
  loom_serve_telemetry_stream_append(stream_, &record);
  IREE_ASSERT_OK(loom_serve_http_logs(stream_, nullptr,
                                      iree_string_view_empty(),
                                      iree_string_view_empty(), &output_));
  EXPECT_NE(Output().find(R"("scope":"server","counters_available":false)"),
            std::string::npos);
  EXPECT_NE(Output().find("{\"event\":\"stream_begin\",\"sequence\":0,\"time_"
                          "ns\":10,\"request\":0}\n"),
            std::string::npos);
  EXPECT_EQ(Output().find("\"counters\":"), std::string::npos);
  EXPECT_EQ(Output().find(R"("event":"counters")"), std::string::npos);
  EXPECT_NE(Output().find("\"value\":\"" + prefix_ + "1\",\"next\":1"),
            std::string::npos);
}

}  // namespace
