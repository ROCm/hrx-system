// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "hrx_internal.h"
#include "iree/base/api.h"
#include "iree/hal/profile_schema.h"
#include "iree/hal/utils/profile_file.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

class RuntimeProfileGpuTest : public testing::Test {
 protected:
  void SetUp() override {
    if (const char* value = std::getenv("HRX_PROFILE_FILE")) {
      old_file_ = value;
    }
    if (const char* value = std::getenv("HRX_PROFILE_MODE")) {
      old_mode_ = value;
    }
    ASSERT_TRUE(profile_file_);
    ASSERT_EQ(setenv("HRX_PROFILE_FILE", profile_file_.path().c_str(), 1), 0);
    ASSERT_EQ(setenv("HRX_PROFILE_MODE", "queue", 1), 0);
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_gpu_initialize(0)));
    initialized_ = true;
    hrx_device_t device = nullptr;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_gpu_device_get(0, &device)));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_create(device, 0, &stream_)));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_event_create(device, HRX_EVENT_FLAG_DISABLE_TIMING, &event_)));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_buffer_allocate(stream_, 64, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                            HRX_BUFFER_USAGE_DEFAULT, &buffer_)));
  }

  void TearDown() override {
    if (stream_) {
      IREE_EXPECT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));
    }
    hrx_buffer_release(buffer_);
    hrx_event_release(event_);
    hrx_stream_release(stream_);
    if (initialized_) {
      IREE_EXPECT_OK(hrx_status_to_iree(hrx_gpu_shutdown()));
    }
    EXPECT_EQ(old_file_ ? setenv("HRX_PROFILE_FILE", old_file_->c_str(), 1)
                        : unsetenv("HRX_PROFILE_FILE"),
              0);
    EXPECT_EQ(old_mode_ ? setenv("HRX_PROFILE_MODE", old_mode_->c_str(), 1)
                        : unsetenv("HRX_PROFILE_MODE"),
              0);
    if (!HasFailure()) {
      CheckCapture();
    }
  }

  void CheckCaptureProgress() {
    // The file sink buffers stdio writes. Large groups must reach the file
    // before shutdown, but the visible suffix may still be a partial record.
    std::ifstream input(profile_file_.path(), std::ios::binary | std::ios::ate);
    ASSERT_TRUE(input.good());
    const std::streamoff length = input.tellg();
    ASSERT_GT(length, capture_length_);
    capture_length_ = length;
  }

  void CheckCapture() {
    std::ifstream input(profile_file_.path(), std::ios::binary);
    ASSERT_TRUE(input.good());
    const std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
    auto contents = iree_make_const_byte_span(bytes.data(), bytes.size());
    iree_hal_profile_file_header_t header;
    iree_host_size_t offset = 0;
    IREE_ASSERT_OK(
        iree_hal_profile_file_parse_header(contents, &header, &offset));
    EXPECT_EQ(header.file_length, bytes.size());
    uint32_t executions = 0;
    uint32_t session_ends = 0;
    while (offset < contents.data_length) {
      iree_hal_profile_file_record_t record;
      IREE_ASSERT_OK(iree_hal_profile_file_parse_record(contents, offset,
                                                        &record, &offset));
      EXPECT_EQ(record.header.dropped_record_count, 0u);
      if (record.header.record_type ==
          IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END) {
        EXPECT_EQ(record.header.session_status_code, IREE_STATUS_OK);
        ++session_ends;
      }
      if (!iree_string_view_equal(record.content_type,
                                  IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
        continue;
      }
      for (iree_host_size_t position = 0;
           position < record.payload.data_length;) {
        iree_hal_profile_queue_event_t event;
        ASSERT_GE(record.payload.data_length - position, sizeof(event));
        std::memcpy(&event, record.payload.data + position, sizeof(event));
        ASSERT_GE(event.record_length, sizeof(event));
        ASSERT_LE(event.record_length, record.payload.data_length - position);
        if (event.type == IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE) {
          ++executions;
        }
        position += event.record_length;
      }
    }
    EXPECT_GT(session_ends, 0u);
    EXPECT_EQ(executions, executions_);
  }

  // Original environment restored after the one runtime lifetime.
  std::optional<std::string> old_file_;
  // Original profiling mode, independent of the output path.
  std::optional<std::string> old_mode_;
  // Capture retained in the test's artifact directory.
  iree::testing::TempFilePath profile_file_{"hrx_queue", ".ireeprof"};
  // Successful initialization requires matching shutdown.
  bool initialized_ = false;
  // Stream owning the tested execution sequence.
  hrx_stream_t stream_ = nullptr;
  // Event reused for explicit event-completion boundaries.
  hrx_event_t event_ = nullptr;
  // Small allocation reused for every fill.
  hrx_buffer_t buffer_ = nullptr;
  // Successfully synchronized one-fill executions.
  uint32_t executions_ = 0;
  // File length at the preceding synchronization group's checkpoint.
  std::streamoff capture_length_ = 0;
};

TEST_F(RuntimeProfileGpuTest, SynchronizationDrainsLongRunningQueueCapture) {
  // Exceeds the native producer's bounded ring without accumulating in-flight
  // work. Each boundary must drain output while the session remains active.
  for (uint32_t i = 0; i < 5000; ++i) {
    const uint8_t pattern = (uint8_t)i;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_fill_buffer(
        stream_, buffer_, 0, 64, &pattern, sizeof(pattern))));
    switch ((i / 1000) % 3) {
      case 0:
        IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));
        break;
      case 1:
        IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_flush(stream_)));
        IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_wait(stream_)));
        break;
      case 2:
        IREE_ASSERT_OK(hrx_status_to_iree(hrx_event_record(event_, stream_)));
        IREE_ASSERT_OK(hrx_status_to_iree(hrx_event_synchronize(event_)));
        break;
    }
    ++executions_;
    if (executions_ % 1000 == 0) {
      CheckCaptureProgress();
    }
  }
}

}  // namespace
