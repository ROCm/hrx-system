// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/device_event.h"

#include <algorithm>
#include <cstring>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class DeviceEventTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(loom_testbench_device_event_capture_initialize(
        2, iree_allocator_system(), &capture_));
  }

  void TearDown() override {
    loom_testbench_device_event_snapshot_deinitialize(&snapshot_);
    loom_testbench_device_event_capture_deinitialize(&capture_);
  }

  void Publish(iree_hal_device_event_type_t type, const void* payload,
               iree_host_size_t payload_length,
               const iree_hal_device_event_site_t* site = nullptr) {
    iree_hal_device_event_t event = iree_hal_device_event_default();
    event.type = type;
    event.severity = IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR;
    event.source.driver_id = IREE_SV("test");
    event.site = site;
    event.payload = iree_make_const_byte_span(payload, payload_length);
    iree_hal_device_event_sink_publish(
        loom_testbench_device_event_capture_sink(&capture_), &event);
  }

  void SnapshotAndReset() {
    IREE_ASSERT_OK(loom_testbench_device_event_capture_snapshot(
        &capture_, iree_allocator_system(), &snapshot_));
    loom_testbench_device_event_capture_reset(&capture_);
    // Replace the recorder's storage while the snapshot remains live.
    const iree_hal_device_asan_report_t report = {};
    Publish(IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT, &report, sizeof(report));
  }

  std::string Dump() {
    ::testing::internal::CaptureStderr();
    loom_testbench_device_event_capture_dump(&capture_);
    return ::testing::internal::GetCapturedStderr();
  }

  // Recorder owning the callback payload beyond the producer's lifetime.
  loom_testbench_device_event_capture_t capture_ = {};
  // Immutable copy retained across recorder reuse.
  loom_testbench_device_event_snapshot_t snapshot_ = {};
};

TEST_F(DeviceEventTest, DriverFailureRetainsMessageAcrossSnapshotAndReset) {
  const std::string expected = "native event wait lost the device";
  std::string message = expected;
  iree_hal_device_driver_failure_event_t failure = {
      .record_length = sizeof(failure),
      .status_code = IREE_STATUS_DATA_LOSS,
      .backend_result_code = 0x12345678,
      .message = iree_make_string_view(message.data(), message.size()),
  };
  Publish(IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE, &failure, sizeof(failure));
  std::fill(message.begin(), message.end(), 'X');
  SnapshotAndReset();
  ASSERT_EQ(snapshot_.events.count, 1u);
  iree_hal_device_driver_failure_event_t retained = {};
  std::memcpy(&retained, snapshot_.events.records[0].event.payload.data,
              sizeof(retained));
  EXPECT_EQ(retained.status_code, IREE_STATUS_DATA_LOSS);
  EXPECT_EQ(retained.backend_result_code, 0x12345678u);
  EXPECT_EQ(std::string(retained.message.data, retained.message.size),
            expected);
}

TEST_F(DeviceEventTest, PrintfRetainsTextAndArgumentsAcrossSnapshotAndReset) {
  const std::string expected_text = "device diagnostic";
  const std::string expected_arguments("\x00\x12\x00\x34", 4);
  std::string text = expected_text;
  std::string arguments = expected_arguments;
  iree_hal_device_printf_event_t payload = {
      .record_length = sizeof(payload),
      .text = iree_make_string_view(text.data(), text.size()),
      .arguments =
          iree_make_const_byte_span(arguments.data(), arguments.size()),
  };
  Publish(IREE_HAL_DEVICE_EVENT_TYPE_PRINTF, &payload, sizeof(payload));
  std::fill(text.begin(), text.end(), 'X');
  std::fill(arguments.begin(), arguments.end(), 'X');
  SnapshotAndReset();
  ASSERT_EQ(snapshot_.events.count, 1u);
  iree_hal_device_printf_event_t retained = {};
  std::memcpy(&retained, snapshot_.events.records[0].event.payload.data,
              sizeof(retained));
  EXPECT_EQ(std::string(retained.text.data, retained.text.size), expected_text);
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(retained.arguments.data),
                        retained.arguments.data_length),
            expected_arguments);
}

TEST_F(DeviceEventTest,
       DumpsOriginalDriverFailureAndSourceAfterProducerReturns) {
  {
    std::string message = "native event wait lost the device";
    std::string source_file = "fault.loom";
    std::string function = "kernel_entry";
    iree_hal_device_driver_failure_event_t failure = {
        .record_length = sizeof(failure),
        .status_code = IREE_STATUS_DATA_LOSS,
        .backend_result_code = 0x12345678,
        .message = iree_make_string_view(message.data(), message.size()),
    };
    iree_hal_device_event_site_t site = iree_hal_device_event_site_default();
    site.source_file =
        iree_make_string_view(source_file.data(), source_file.size());
    site.function_name =
        iree_make_string_view(function.data(), function.size());
    site.start_line = 17;
    site.start_column = 3;
    Publish(IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE, &failure,
            sizeof(failure), &site);
  }
  const std::string output = Dump();
  EXPECT_NE(output.find("type=driver_failure"), std::string::npos);
  EXPECT_NE(output.find("status=DATA_LOSS"), std::string::npos);
  EXPECT_NE(output.find("backend_result=0x0000000012345678"),
            std::string::npos);
  EXPECT_NE(output.find("native event wait lost the device"),
            std::string::npos);
  EXPECT_NE(output.find("fault.loom:17:3"), std::string::npos);
  EXPECT_NE(output.find("function=kernel_entry"), std::string::npos);
}

TEST_F(DeviceEventTest, OversizedBorrowedFieldsAreReportedAsDropped) {
  const std::string text(LOOM_TESTBENCH_DEVICE_EVENT_STRING_CAPACITY + 1, 'X');
  const std::string arguments(LOOM_TESTBENCH_DEVICE_EVENT_PAYLOAD_CAPACITY + 1,
                              'X');
  iree_hal_device_driver_failure_event_t failure = {
      .record_length = sizeof(failure),
      .message = iree_make_string_view(text.data(), text.size()),
  };
  Publish(IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE, &failure, sizeof(failure));
  iree_hal_device_printf_event_t printf_event = {
      .record_length = sizeof(printf_event),
      .text = iree_make_string_view(text.data(), text.size()),
  };
  Publish(IREE_HAL_DEVICE_EVENT_TYPE_PRINTF, &printf_event,
          sizeof(printf_event));
  printf_event.text = iree_string_view_empty();
  printf_event.arguments =
      iree_make_const_byte_span(arguments.data(), arguments.size());
  Publish(IREE_HAL_DEVICE_EVENT_TYPE_PRINTF, &printf_event,
          sizeof(printf_event));
  EXPECT_EQ(Dump(),
            "device event capture: dropped 3 events exceeding capture "
            "capacity\n");
  SnapshotAndReset();
  EXPECT_EQ(snapshot_.events.count, 0u);
  EXPECT_EQ(snapshot_.events.dropped_count, 3u);
  loom_testbench_device_event_capture_reset(&capture_);
  EXPECT_EQ(Dump(), "");
}

TEST_F(DeviceEventTest, CaptureOverflowPreservesReportsAndCountsDroppedEvents) {
  iree_hal_device_driver_failure_event_t failure = {
      .record_length = sizeof(failure),
      .status_code = IREE_STATUS_DATA_LOSS,
      .message = IREE_SV("retained failure"),
  };
  for (int i = 0; i < 3; ++i) {
    Publish(IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE, &failure,
            sizeof(failure));
  }
  const std::string output = Dump();
  EXPECT_NE(output.find("retained failure"), std::string::npos);
  EXPECT_NE(output.find("dropped 1 events"), std::string::npos);
  SnapshotAndReset();
  EXPECT_EQ(snapshot_.events.count, 2u);
  EXPECT_EQ(snapshot_.events.dropped_count, 1u);
}

}  // namespace
}  // namespace loom
