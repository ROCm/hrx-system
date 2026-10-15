// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <atomic>
#include <cstring>
#include <memory>

#include "iree/hal/api.h"
#include "iree/hal/drivers/amd/gpu/cts/feedback_images.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu::cts {
namespace {

class GpuFeedbackTest : public GpuNativeTest {
 protected:
  void SetUp() override {
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    if (!(context_->device_info.features &
          AMDF_GPU_DEVICE_FEATURE_NATIVE_EVENTS)) {
      GTEST_SKIP() << "Native shader notification is unavailable";
    }
    const uint32_t gfx_major = context_->target_info.gfx_ip.major;
    context_->event_sink = {Publish, this};
    Create(IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_FEEDBACK);
    if (HasFatalFailure()) {
      return;
    }
    const auto* files = iree_hal_amd_gpu_feedback_images_create();
    const char* selector = gfx_major == 11 ? "gfx11-generic" : "gfx12-generic";
    for (size_t i = 0; i < iree_hal_amd_gpu_feedback_images_size(); ++i) {
      if (std::strstr(files[i].name, selector)) {
        artifact_ = iree_make_const_byte_span(files[i].data, files[i].size);
      }
    }
    ASSERT_NE(artifact_.data, nullptr);
  }
  void TearDown() override {
    for (auto* recording : recordings_) {
      iree_hal_command_buffer_release(recording);
    }
    for (auto* semaphore : done_) {
      iree_hal_semaphore_release(semaphore);
    }
    GpuNativeTest::TearDown();
  }
  static void Publish(void* user_data, const iree_hal_device_event_t* event) {
    auto& self = *static_cast<GpuFeedbackTest*>(user_data);
    if (event->type != IREE_HAL_DEVICE_EVENT_TYPE_UBSAN_REPORT) {
      iree_hal_device_event_sink_publish(iree_hal_device_event_sink_stderr(),
                                         event);
      self.errors_.fetch_add(1);
      return;
    }
    EXPECT_NE(event->source.executable_id, 0u);
    EXPECT_NE(event->site, nullptr);
    if (event->site) {
      EXPECT_TRUE(iree_string_view_equal(event->site->operation_name,
                                         IREE_SV("kernel.assert")));
    }
    const auto* report =
        reinterpret_cast<const iree_hal_device_ubsan_report_t*>(
            event->payload.data);
    EXPECT_EQ(report->check_kind, IREE_HAL_DEVICE_UBSAN_CHECK_KIND_ASSERTION);
    EXPECT_EQ(report->site_id, 0u);
    EXPECT_EQ(report->operand0, 0u);
    self.reports_.fetch_add(1);
  }
  void Record(uint32_t count, iree_hal_command_buffer_t** out_recording) {
    const auto* family = iree_hal_queue_family(queues_[0]);
    const auto* target =
        &iree_hal_device_spec_executables(iree_hal_device_spec(device_))
             ->targets[0];
    iree_hal_executable_load_params_t load;
    iree_hal_executable_load_params_initialize(&load);
    load.executable_data = artifact_;
    iree_hal_executable_t* executable = nullptr;
    IREE_ASSERT_OK(
        iree_hal_executable_load(family, target, &load, &executable));
    std::unique_ptr<iree_hal_executable_t,
                    decltype(&iree_hal_executable_release)>
        executable_owner(executable, iree_hal_executable_release);
    iree_hal_executable_function_t function;
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executable, IREE_SV("report_assertion"), &function));
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        family, 0, IREE_HAL_COMMAND_CATEGORY_DISPATCH, 0, out_recording));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(*out_recording));
    auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
    config.workgroup_size[0] = config.workgroup_size[1] =
        config.workgroup_size[2] = 1;
    const uint32_t value = 0;
    for (uint32_t i = 0; i < count; ++i) {
      IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
          *out_recording, executable, function, config,
          iree_make_const_byte_span(&value, sizeof(value)),
          iree_hal_buffer_ref_list_empty(), 0));
    }
    IREE_ASSERT_OK(iree_hal_command_buffer_end(*out_recording));
  }
  // Real Loom artifact compiled for this native device family.
  iree_const_byte_span_t artifact_ = {};
  // Public reports observed on the completion worker.
  std::atomic<uint64_t> reports_ = 0;
  // Driver failures, including dropped or malformed packets.
  std::atomic<uint64_t> errors_ = 0;
  // Owning recordings, also reclaimed after an assertion aborts the test body.
  std::array<iree_hal_command_buffer_t*, 2> recordings_ = {};
  // Owning completion semaphores, released before the fixture's device.
  std::array<iree_hal_semaphore_t*, 2> done_ = {};
};

TEST_F(GpuFeedbackTest, SharedChannelSurvivesConcurrentQueuesAndCircularReuse) {
  // More than twice the channel's physical capacity over bounded batches.
  constexpr uint32_t kDispatches = 128;
  constexpr uint64_t kReplays = 160;
  for (size_t i = 0; i < recordings_.size(); ++i) {
    Record(kDispatches, &recordings_[i]);
    CreateSemaphore(&done_[i]);
  }
  ASSERT_FALSE(HasFatalFailure());
  for (uint64_t value = 1; value <= kReplays; ++value) {
    for (size_t i = 0; i < recordings_.size(); ++i) {
      IREE_ASSERT_OK(iree_hal_queue_execute(
          queues_[i], iree_hal_semaphore_list_empty(), {1, &done_[i], &value},
          recordings_[i], iree_hal_buffer_binding_table_empty(), 0));
    }
    if (value == kReplays) {
      for (auto*& recording : recordings_) {
        iree_hal_command_buffer_release(recording);
        recording = nullptr;
      }
    }
    for (auto* semaphore : done_) {
      IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, value,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
    }
    ASSERT_EQ(reports_.load(), value * recordings_.size() * kDispatches);
    ASSERT_EQ(errors_.load(), 0u);
  }
}

}  // namespace
}  // namespace iree::hal::amd::gpu::cts
