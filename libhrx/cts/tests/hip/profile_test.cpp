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

#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/base/api.h"
#include "iree/hal/profile_schema.h"
#include "iree/hal/utils/profile_file.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "libhrx/cts/core/amdgpu_executable_test_data.hpp"

namespace {

class HipProfileTest : public testing::Test {
 protected:
  template <typename Function, typename... Args>
  hipError_t Call(const char* name, Args... args) {
    auto function = library_.Resolve<Function>(name);
    if (!function) {
      ADD_FAILURE() << library_.error();
      return hipErrorNotInitialized;
    }
    return function(args...);
  }

  void SetUp() override {
    if (const char* value = std::getenv("HRX_PROFILE_FILE")) {
      old_file_ = value;
    }
    if (const char* value = std::getenv("HRX_PROFILE_MODE")) {
      old_mode_ = value;
    }
    ASSERT_TRUE(profile_file_);
    ASSERT_EQ(setenv("HRX_PROFILE_FILE", profile_file_.path().c_str(), 1), 0);
    ASSERT_EQ(setenv("HRX_PROFILE_MODE", "dispatch", 1), 0);
    ASSERT_TRUE(library_.Open()) << library_.error();
    ASSERT_EQ(hipSuccess, Call<decltype(&hipInit)>("hipInit", 0));
    initialized_ = true;
  }

  void TearDown() override {
    if (initialized_) {
      EXPECT_EQ(hipSuccess,
                Call<decltype(&hipDeviceSynchronize)>("hipDeviceSynchronize"));
      if (graph_executable_) {
        EXPECT_EQ(hipSuccess, Call<decltype(&hipGraphExecDestroy)>(
                                  "hipGraphExecDestroy", graph_executable_));
      }
      if (graph_) {
        EXPECT_EQ(hipSuccess,
                  Call<decltype(&hipGraphDestroy)>("hipGraphDestroy", graph_));
      }
      if (event_) {
        EXPECT_EQ(hipSuccess,
                  Call<decltype(&hipEventDestroy)>("hipEventDestroy", event_));
      }
      if (stream_) {
        EXPECT_EQ(hipSuccess, Call<decltype(&hipStreamDestroy)>(
                                  "hipStreamDestroy", stream_));
      }
      if (output_) {
        EXPECT_EQ(hipSuccess, Call<decltype(&hipFree)>("hipFree", output_));
      }
      if (module_) {
        EXPECT_EQ(hipSuccess,
                  Call<decltype(&hipModuleUnload)>("hipModuleUnload", module_));
      }
      EXPECT_EQ(hipSuccess, Call<hipError_t (*)()>("hipHALDeinit"));
    }
    if (library_.is_open()) {
      EXPECT_TRUE(library_.Close()) << library_.error();
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
    // The file sink buffers stdio writes. Check progress here and parse every
    // record after shutdown has committed the final buffered suffix.
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
    ASSERT_EQ(header.file_length, bytes.size());
    uint32_t direct_count = 0, recorded_count = 0, ended_count = 0;
    while (offset < contents.data_length) {
      iree_hal_profile_file_record_t record;
      IREE_ASSERT_OK(iree_hal_profile_file_parse_record(contents, offset,
                                                        &record, &offset));
      EXPECT_EQ(record.header.dropped_record_count, 0u);
      if (record.header.record_type ==
          IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END) {
        EXPECT_EQ(record.header.session_status_code, IREE_STATUS_OK);
        ++ended_count;
      }
      if (!iree_string_view_equal(
              record.content_type,
              IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS)) {
        continue;
      }
      for (iree_host_size_t position = 0;
           position < record.payload.data_length;) {
        iree_hal_profile_dispatch_event_t event;
        ASSERT_GE(record.payload.data_length - position, sizeof(event));
        std::memcpy(&event, record.payload.data + position, sizeof(event));
        ASSERT_GE(event.record_length, sizeof(event));
        ASSERT_LE(event.record_length, record.payload.data_length - position);
        EXPECT_GT(event.start_tick, 0u);
        EXPECT_GE(event.end_tick, event.start_tick);
        EXPECT_NE(event.executable_id, 0u);
        if (event.command_buffer_id) {
          ++recorded_count;
        } else {
          ++direct_count;
        }
        position += event.record_length;
      }
    }
    EXPECT_GT(ended_count, 0u);
    EXPECT_EQ(direct_count, direct_count_);
    EXPECT_EQ(recorded_count, 2u);
  }

  // Original environment restored after the single runtime lifetime.
  std::optional<std::string> old_file_;
  // Original capture mode, independent of the output path.
  std::optional<std::string> old_mode_;
  // Unique capture path owned by the outer test's artifact directory.
  iree::testing::TempFilePath profile_file_{"hip_dispatch", ".ireeprof"};
  // Exact built HIP artifact, closed only after terminal runtime teardown.
  hrx::hip::testing::HipDso library_;
  // A successful initialization requires a matching terminal deinitialization.
  bool initialized_ = false;
  // Module retained until all direct and recorded invocations have completed.
  hipModule_t module_ = nullptr;
  // Output allocation retained through the final device join.
  void* output_ = nullptr;
  // Explicit stream used for ordinary dispatch and graph capture.
  hipStream_t stream_ = nullptr;
  // Reusable event covering dispatch completion without timestamp work.
  hipEvent_t event_ = nullptr;
  // Captured graph retained until its executable is released.
  hipGraph_t graph_ = nullptr;
  // Instantiated graph retained through its final execution.
  hipGraphExec_t graph_executable_ = nullptr;
  // Direct dispatches whose completion has been synchronized by the test.
  uint32_t direct_count_ = 0;
  // File length at the preceding synchronization group's checkpoint.
  std::streamoff capture_length_ = 0;
};

TEST_F(HipProfileTest, DispatchModeCapturesDirectAndRecordedExecution) {
  int device = 0;
  ASSERT_EQ(hipSuccess, Call<decltype(&hipGetDevice)>("hipGetDevice", &device));
  hipDeviceProp_t properties = {};
  ASSERT_EQ(hipSuccess,
            Call<decltype(&hipGetDevicePropertiesR0600)>(
                "hipGetDevicePropertiesR0600", &properties, device));
  const auto image =
      hrx_cts::FindAmdgpuExecutableTestImage(properties.gcnArchName);
  ASSERT_NE(image.file, nullptr) << properties.gcnArchName;
  ASSERT_EQ(hipSuccess, Call<decltype(&hipModuleLoadData)>(
                            "hipModuleLoadData", &module_, image.file->data));
  hipFunction_t function = nullptr;
  ASSERT_EQ(hipSuccess, Call<decltype(&hipModuleGetFunction)>(
                            "hipModuleGetFunction", &function, module_,
                            "hrx_store_output"));
  ASSERT_EQ(hipSuccess, Call<decltype(&hipMalloc)>("hipMalloc", &output_, 4));
  ASSERT_EQ(hipSuccess,
            Call<decltype(&hipStreamCreate)>("hipStreamCreate", &stream_));
  ASSERT_EQ(hipSuccess,
            Call<decltype(&hipEventCreateWithFlags)>(
                "hipEventCreateWithFlags", &event_, hipEventDisableTiming));
  uint32_t expected = 0xC0FFEE;
  void* arguments[] = {&output_, &expected};
  ASSERT_EQ(hipSuccess, Call<decltype(&hipModuleLaunchKernel)>(
                            "hipModuleLaunchKernel", function, 1, 1, 1, 1, 1, 1,
                            0, stream_, arguments, nullptr));
  ASSERT_EQ(hipSuccess, Call<decltype(&hipStreamSynchronize)>(
                            "hipStreamSynchronize", stream_));
  ++direct_count_;
  uint32_t actual = 0;
  ASSERT_EQ(hipSuccess,
            Call<decltype(&hipMemcpy)>("hipMemcpy", &actual, output_,
                                       sizeof(actual), hipMemcpyDeviceToHost));
  ASSERT_EQ(actual, expected);
  expected = 0xD15CA7C4;
  ASSERT_EQ(hipSuccess,
            Call<decltype(&hipStreamBeginCapture)>(
                "hipStreamBeginCapture", stream_, hipStreamCaptureModeGlobal));
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(hipSuccess, Call<decltype(&hipModuleLaunchKernel)>(
                              "hipModuleLaunchKernel", function, 1, 1, 1, 1, 1,
                              1, 0, stream_, arguments, nullptr));
  }
  ASSERT_EQ(hipSuccess, Call<decltype(&hipStreamEndCapture)>(
                            "hipStreamEndCapture", stream_, &graph_));
  ASSERT_EQ(hipSuccess, Call<decltype(&hipGraphInstantiate)>(
                            "hipGraphInstantiate", &graph_executable_, graph_,
                            nullptr, nullptr, 0));
  ASSERT_EQ(hipSuccess, Call<decltype(&hipGraphLaunch)>(
                            "hipGraphLaunch", graph_executable_, stream_));
  ASSERT_EQ(hipSuccess, Call<decltype(&hipStreamSynchronize)>(
                            "hipStreamSynchronize", stream_));
  actual = 0;
  ASSERT_EQ(hipSuccess,
            Call<decltype(&hipMemcpy)>("hipMemcpy", &actual, output_,
                                       sizeof(actual), hipMemcpyDeviceToHost));
  EXPECT_EQ(actual, expected);

  // Cross the producer's ring capacity in one runtime lifetime. Each group uses
  // only its named host synchronization boundary and must publish its records
  // before shutdown, including when the synchronization is event-scoped.
  {
    std::ifstream input(profile_file_.path(), std::ios::binary | std::ios::ate);
    ASSERT_TRUE(input.good());
    capture_length_ = input.tellg();
    ASSERT_GE(capture_length_, 0);
  }
  for (uint32_t i = 0; i < 5000; ++i) {
    ASSERT_EQ(hipSuccess, Call<decltype(&hipModuleLaunchKernel)>(
                              "hipModuleLaunchKernel", function, 1, 1, 1, 1, 1,
                              1, 0, stream_, arguments, nullptr));
    switch (i / 1000) {
      case 0:
        ASSERT_EQ(hipSuccess, Call<decltype(&hipStreamSynchronize)>(
                                  "hipStreamSynchronize", stream_));
        break;
      case 1:
        ASSERT_EQ(hipSuccess, Call<decltype(&hipDeviceSynchronize)>(
                                  "hipDeviceSynchronize"));
        break;
      case 2:
        ASSERT_EQ(hipSuccess,
                  Call<decltype(&hipCtxSynchronize)>("hipCtxSynchronize"));
        break;
      case 3:
        ASSERT_EQ(hipSuccess, Call<decltype(&hipEventRecord)>("hipEventRecord",
                                                              event_, stream_));
        ASSERT_EQ(hipSuccess, Call<decltype(&hipEventSynchronize)>(
                                  "hipEventSynchronize", event_));
        break;
      case 4:
        ASSERT_EQ(hipSuccess, Call<decltype(&hipStreamSynchronize)>(
                                  "hipStreamSynchronize", nullptr));
        break;
    }
    ++direct_count_;
    if ((i + 1) % 1000 == 0) {
      CheckCaptureProgress();
    }
  }
}

}  // namespace
