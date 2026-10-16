// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <atomic>
#include <cstring>
#include <semaphore>
#include <tuple>

#include "iree/hal/api.h"
#include "iree/hal/drivers/amd/gpu/cts/asan_report_images.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu::cts {
namespace {

class GpuAsanReportTest
    : public GpuNativeTest,
      public ::testing::WithParamInterface<std::tuple<bool, bool>> {
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
    options_.asan.memory.shadow_slab_size = 1024 * 1024;
    options_.asan.report_policy =
        std::get<0>(GetParam())
            ? IREE_HAL_AMD_GPU_SANITIZER_REPORT_POLICY_FAIL_DEVICE
            : IREE_HAL_AMD_GPU_SANITIZER_REPORT_POLICY_REPORT_ONLY;
    Create(IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_ASAN);
    if (HasFatalFailure()) {
      return;
    }
    const auto* family = iree_hal_queue_family(queues_[0]);
    const iree_hal_pool_family_access_t access = {
        .family = family,
        .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
        .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = 1,
        .families = &access,
        .host =
            {
                .access = IREE_HAL_MEMORY_ACCESS_READ,
                .modes = IREE_HAL_MAPPING_MODE_SCOPED,
            },
    };
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &options,
                                             iree_allocator_system(), &pool_));
    iree_hal_memory_scope_t program_scope;
    IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
        group_, {IREE_HAL_MEMORY_SITE_PROGRAM, family}, &program_scope));
    IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
        pool_, program_scope, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
        &slot_));
    const auto* files = iree_hal_amd_gpu_asan_report_images_create();
    const char* selector = gfx_major == 11 ? "gfx11-generic" : "gfx12-generic";
    iree_hal_executable_load_params_t load;
    iree_hal_executable_load_params_initialize(&load);
    for (size_t i = 0; i < iree_hal_amd_gpu_asan_report_images_size(); ++i) {
      if (std::strstr(files[i].name, selector)) {
        load.executable_data =
            iree_make_const_byte_span(files[i].data, files[i].size);
      }
    }
    ASSERT_NE(load.executable_data.data, nullptr);
    const auto* target =
        &iree_hal_device_spec_executables(iree_hal_device_spec(device_))
             ->targets[0];
    IREE_ASSERT_OK(
        iree_hal_executable_load(family, target, &load, &executable_));
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executable_, IREE_SV("checked_write"), &function_));
    CreateSemaphore(&done_);
  }

  void TearDown() override {
    iree_hal_command_buffer_release(recording_);
    iree_hal_executable_release(executable_);
    for (auto* buffer : buffers_) {
      iree_hal_buffer_release(buffer);
    }
    if (observing_release_) {
      // A failure signal does not prove the queue released its buffer capture.
      // Pools must outlive every buffer, including those still retiring.
      released_.acquire();
    }
    iree_hal_pool_release(pool_);
    iree_hal_semaphore_release(done_);
    GpuNativeTest::TearDown();
    EXPECT_EQ(errors_.load(), 0u);
  }

  static void Publish(void* user_data, const iree_hal_device_event_t* event) {
    auto& self = *static_cast<GpuAsanReportTest*>(user_data);
    if (event->type == IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE &&
        self.options_.asan.report_policy ==
            IREE_HAL_AMD_GPU_SANITIZER_REPORT_POLICY_FAIL_DEVICE) {
      const auto* failure =
          reinterpret_cast<const iree_hal_device_driver_failure_event_t*>(
              event->payload.data);
      EXPECT_EQ(failure->status_code, IREE_STATUS_ABORTED);
      self.queue_failures_.fetch_add(1);
      return;
    }
    if (event->type != IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT) {
      iree_hal_device_event_sink_publish(iree_hal_device_event_sink_stderr(),
                                         event);
      self.errors_.fetch_add(1);
      return;
    }
    const auto* report = reinterpret_cast<const iree_hal_device_asan_report_t*>(
        event->payload.data);
    EXPECT_EQ(report->access_kind, IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE);
    EXPECT_EQ(report->access_length, 4u);
    EXPECT_EQ(report->fault_address, self.expected_fault_.load());
    EXPECT_NE(report->shadow_address, 0u);
    EXPECT_EQ(report->shadow_value, 4u);
    EXPECT_NE(report->source_dispatch_ptr, 0u);
    EXPECT_NE(event->source.executable_id, 0u);
    self.reports_.fetch_add(1);
  }

  void Submit(iree_hal_buffer_t* buffer, uint64_t value) {
    auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
    config.workgroup_size[0] = config.workgroup_size[1] =
        config.workgroup_size[2] = 1;
    const auto binding =
        iree_hal_make_buffer_ref(buffer, 0, IREE_HAL_WHOLE_BUFFER);
    if (std::get<1>(GetParam())) {
      IREE_ASSERT_OK(iree_hal_command_buffer_create(
          iree_hal_queue_family(queues_[0]),
          IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
          IREE_HAL_COMMAND_CATEGORY_DISPATCH, 0, &recording_));
      IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording_));
      IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
          recording_, executable_, function_, config,
          iree_const_byte_span_empty(), {1, &binding}, 0));
      const iree_hal_barrier_t release = {
          .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
          .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
          .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
      };
      IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording_, &release));
      IREE_ASSERT_OK(iree_hal_command_buffer_end(recording_));
      IREE_ASSERT_OK(iree_hal_queue_execute(
          queues_[0], {}, {1, &done_, &value}, recording_,
          iree_hal_buffer_binding_table_empty(), 0));
      iree_hal_command_buffer_release(recording_);
      recording_ = nullptr;
    } else {
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          queues_[0], {}, {1, &done_, &value}, executable_, function_, config,
          iree_const_byte_span_empty(), {1, &binding}, nullptr, 0));
    }
  }

  // Shared allocation contract enabling owned-allocation ASAN advice.
  iree_hal_pool_t* pool_ = nullptr;
  // Exact program address slot used to verify the reported address.
  iree_hal_buffer_native_binding_slot_t slot_ = {};
  // Actual compiler output with the established ASAN and feedback globals.
  iree_hal_executable_t* executable_ = nullptr;
  // Kernel that checks and writes the second i32 in its buffer argument.
  iree_hal_executable_function_t function_ = {};
  // Optional recording, retained by the queue after public release.
  iree_hal_command_buffer_t* recording_ = nullptr;
  // First a valid eight-byte allocation, then a four-byte allocation.
  std::array<iree_hal_buffer_t*, 2> buffers_ = {};
  // True when the faulting buffer owns a final-release callback.
  bool observing_release_ = false;
  // Final backing release, independently of the public failure signal.
  std::binary_semaphore released_{0};
  // Public success or fail-device signal for the current dispatch.
  iree_hal_semaphore_t* done_ = nullptr;
  // Expected address set before publishing the faulting dispatch.
  std::atomic<uint64_t> expected_fault_ = 0;
  // Sanitizer reports consumed before completion becomes observable.
  std::atomic<uint32_t> reports_ = 0;
  // Failed-operation diagnostics accompanying fail-device policy.
  std::atomic<uint32_t> queue_failures_ = 0;
  // Unexpected native or channel failures.
  std::atomic<uint32_t> errors_ = 0;
};

TEST_P(GpuAsanReportTest, ValidAccessThenRedzoneReportPreservesRetirement) {
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(8, 4),
      iree_infinite_timeout(), &buffers_[0]));
  Submit(buffers_[0], 1);
  ASSERT_FALSE(HasFatalFailure());
  IREE_ASSERT_OK(iree_hal_semaphore_wait(done_, 1, iree_infinite_timeout(), 0));
  EXPECT_EQ(reports_.load(), 0u);
  uint32_t actual = 0;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(buffers_[0], 4, &actual, sizeof(actual)));
  EXPECT_EQ(actual, 7u);
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(4, 4),
      iree_infinite_timeout(), &buffers_[1]));
  iree_hal_buffer_t* observed = nullptr;
  IREE_ASSERT_OK(iree_hal_subspan_buffer_create_with_callback(
      buffers_[1], iree_hal_buffer_byte_offset(buffers_[1]), 4,
      {+[](void* user_data, iree_hal_buffer_t* buffer) {
         static_cast<std::binary_semaphore*>(user_data)->release();
       },
       &released_},
      iree_allocator_system(), &observed));
  iree_hal_buffer_release(buffers_[1]);
  buffers_[1] = observed;
  observing_release_ = true;
  // The access stays inside owned backing; its logical allocation is too short.
  expected_fault_.store(
      iree_hal_buffer_native_binding(buffers_[1], slot_).device_address + 4);
  Submit(buffers_[1], 2);
  ASSERT_FALSE(HasFatalFailure());
  iree_hal_executable_release(executable_);
  executable_ = nullptr;
  iree_hal_buffer_release(buffers_[1]);
  buffers_[1] = nullptr;
  const Status status(
      iree_hal_semaphore_wait(done_, 2, iree_infinite_timeout(), 0));
  EXPECT_EQ(status.code(),
            std::get<0>(GetParam()) ? StatusCode::kAborted : StatusCode::kOk);
  EXPECT_EQ(reports_.load(), 1u);
  if (std::get<0>(GetParam())) {
    EXPECT_GE(queue_failures_.load(), 1u);
  }
  // The fixture waits for actual capture, native memory, and worker
  // reclamation.
}

INSTANTIATE_TEST_SUITE_P(PolicyAndRecording, GpuAsanReportTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

}  // namespace
}  // namespace iree::hal::amd::gpu::cts
