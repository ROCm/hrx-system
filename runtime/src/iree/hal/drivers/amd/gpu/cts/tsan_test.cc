// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <atomic>
#include <cstring>

#include "iree/hal/api.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/cts/tsan_images.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu::cts {
namespace {

class GpuTsanTest : public GpuNativeTest,
                    public ::testing::WithParamInterface<uint32_t> {
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
    context_->event_sink = {Publish, this};
    const uint32_t gfx_major = context_->target_info.gfx_ip.major;
    const auto* files = iree_hal_amd_gpu_tsan_images_create();
    const char* selector = gfx_major == 11 ? "gfx11-generic" : "gfx12-generic";
    for (size_t i = 0; i < iree_hal_amd_gpu_tsan_images_size(); ++i) {
      if (std::strstr(files[i].name, selector)) {
        artifact_ = iree_make_const_byte_span(files[i].data, files[i].size);
      }
    }
    ASSERT_NE(artifact_.data, nullptr);
    options_.tsan.memory.workgroup_local_memory_size = 256;
    options_.tsan.memory.workgroup_capacity = 4;
    options_.tsan.memory.shadow_slot_count = GetParam();
  }

  void TearDown() override {
    iree_hal_command_buffer_release(recording_);
    iree_hal_executable_release(executable_);
    iree_hal_buffer_release(counts_);
    iree_hal_pool_release(pool_);
    for (auto* semaphore : done_) {
      iree_hal_semaphore_release(semaphore);
    }
    GpuNativeTest::TearDown();
  }

  void Load(iree_hal_amd_gpu_sanitizer_report_policy_t policy) {
    options_.tsan.report_policy = policy;
    Create(IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_TSAN);
    if (HasFatalFailure()) {
      return;
    }
    const auto* family = iree_hal_queue_family(queues_[0]);
    const auto* target =
        &iree_hal_device_spec_executables(iree_hal_device_spec(device_))
             ->targets[0];
    iree_hal_executable_load_params_t load;
    iree_hal_executable_load_params_initialize(&load);
    load.executable_data = artifact_;
    IREE_ASSERT_OK(
        iree_hal_executable_load(family, target, &load, &executable_));
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executable_, IREE_SV("report_race"), &race_));
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executable_, IREE_SV("safe_write"), &safe_));
    for (auto*& semaphore : done_) {
      CreateSemaphore(&semaphore);
    }
  }

  static void Publish(void* user_data, const iree_hal_device_event_t* event) {
    auto& self = *static_cast<GpuTsanTest*>(user_data);
    if (event->type == IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE &&
        self.options_.tsan.report_policy ==
            IREE_HAL_AMD_GPU_SANITIZER_REPORT_POLICY_FAIL_DEVICE) {
      const auto* failure =
          reinterpret_cast<const iree_hal_device_driver_failure_event_t*>(
              event->payload.data);
      EXPECT_EQ(failure->status_code, IREE_STATUS_DATA_LOSS);
      self.queue_failures_.fetch_add(1);
      return;
    }
    if (event->type != IREE_HAL_DEVICE_EVENT_TYPE_TSAN_REPORT) {
      iree_hal_device_event_sink_publish(iree_hal_device_event_sink_stderr(),
                                         event);
      self.errors_.fetch_add(1);
      return;
    }
    const auto* report = reinterpret_cast<const iree_hal_device_tsan_report_t*>(
        event->payload.data);
    EXPECT_EQ(report->check_kind, IREE_HAL_DEVICE_TSAN_CHECK_KIND_DATA_RACE);
    EXPECT_EQ(report->memory_space,
              IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_WORKGROUP);
    EXPECT_EQ(report->current_access_kind,
              IREE_HAL_DEVICE_TSAN_ACCESS_KIND_WRITE);
    EXPECT_EQ(report->prior_access_kind,
              IREE_HAL_DEVICE_TSAN_ACCESS_KIND_WRITE);
    EXPECT_EQ(report->access_length, 4u);
    EXPECT_NE(report->shadow_address, 0u);
    EXPECT_NE(report->source_dispatch_ptr, 0u);
    EXPECT_NE(event->source.executable_id, 0u);
    uint64_t identity = 0;
    self.executable_id_.compare_exchange_strong(identity,
                                                event->source.executable_id);
    EXPECT_EQ(self.executable_id_.load(), event->source.executable_id);
    self.last_shadow_.store(report->shadow_address);
    self.reports_.fetch_add(1);
  }

  static iree_hal_dispatch_config_t Config(uint32_t workgroups = 1) {
    auto config = iree_hal_make_static_dispatch_config(workgroups, 1, 1);
    config.workgroup_size[0] = 2;
    config.workgroup_size[1] = config.workgroup_size[2] = 1;
    return config;
  }

  void Dispatch(uint32_t queue, iree_hal_executable_function_t function,
                uint64_t signal) {
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        queues_[queue], iree_hal_semaphore_list_empty(),
        {1, &done_[queue], &signal}, executable_, function, Config(),
        iree_const_byte_span_empty(), iree_hal_buffer_ref_list_empty(), nullptr,
        0));
  }

  void Record(uint32_t pairs) {
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        iree_hal_queue_family(queues_[0]), 0,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH, 0, &recording_));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording_));
    for (uint32_t i = 0; i < pairs; ++i) {
      for (auto function : {safe_, race_}) {
        IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
            recording_, executable_, function, Config(3),
            iree_const_byte_span_empty(), iree_hal_buffer_ref_list_empty(), 0));
      }
    }
    IREE_ASSERT_OK(iree_hal_command_buffer_end(recording_));
  }

  // Real compiler output for the native device family.
  iree_const_byte_span_t artifact_ = {};
  // One executable owns all queue variants and the shared source identity.
  iree_hal_executable_t* executable_ = nullptr;
  // Race-producing entry point.
  iree_hal_executable_function_t race_ = {};
  // Disjoint-access entry point, which must not report a race.
  iree_hal_executable_function_t safe_ = {};
  // One immutable recording submitted concurrently on both queues.
  iree_hal_command_buffer_t* recording_ = nullptr;
  // Optional indirect dispatch tuple, retained by recordings and submissions.
  iree_hal_buffer_t* counts_ = nullptr;
  // Explicit backing contract for the indirect tuple.
  iree_hal_pool_t* pool_ = nullptr;
  // Public completion boundaries for each queue.
  std::array<iree_hal_semaphore_t*, 2> done_ = {};
  // Reports delivered by the sole completion consumer.
  std::atomic<uint64_t> reports_ = 0;
  // Unexpected channel or device failures.
  std::atomic<uint64_t> errors_ = 0;
  // Failed-operation diagnostics accompanying the requested fail-device policy.
  std::atomic<uint64_t> queue_failures_ = 0;
  // Shared executable identity retained after the public handle is released.
  std::atomic<uint64_t> executable_id_ = 0;
  // Most recent native shadow address, inspected only after completion.
  std::atomic<uint64_t> last_shadow_ = 0;
};

TEST_P(GpuTsanTest, SharedRecordingSelectsQueueShadowAndJoinsSlotReuse) {
  Load(IREE_HAL_AMD_GPU_SANITIZER_REPORT_POLICY_REPORT_ONLY);
  ASSERT_FALSE(HasFatalFailure());
  std::array<uint64_t, 2> shadows = {};
  for (uint32_t queue = 0; queue < 2; ++queue) {
    Dispatch(queue, race_, 1);
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        done_[queue], 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    shadows[queue] = last_shadow_.load();
  }
  ASSERT_NE(shadows[0], shadows[1]);
  ASSERT_EQ(reports_.load(), 2u);
  constexpr uint32_t kPairs = 32;
  constexpr uint64_t kReplays = 24;
  Record(kPairs);
  ASSERT_FALSE(HasFatalFailure());
  for (uint64_t replay = 0; replay < kReplays; ++replay) {
    uint64_t value = replay + 2;
    for (uint32_t queue = 0; queue < 2; ++queue) {
      IREE_ASSERT_OK(iree_hal_queue_execute(
          queues_[queue], iree_hal_semaphore_list_empty(),
          {1, &done_[queue], &value}, recording_,
          iree_hal_buffer_binding_table_empty(), 0));
    }
    if (replay + 1 == kReplays) {
      iree_hal_command_buffer_release(recording_);
      recording_ = nullptr;
      iree_hal_executable_release(executable_);
      executable_ = nullptr;
    }
    for (auto* semaphore : done_) {
      IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, value,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
    }
    ASSERT_EQ(reports_.load(), 2 + (replay + 1) * 2 * kPairs * 3);
    ASSERT_EQ(errors_.load(), 0u);
  }
}

TEST_P(GpuTsanTest, ReportFailurePropagatesWithoutAbandoningNativeRetirement) {
  Load(IREE_HAL_AMD_GPU_SANITIZER_REPORT_POLICY_FAIL_DEVICE);
  ASSERT_FALSE(HasFatalFailure());
  Dispatch(0, race_, 1);
  ASSERT_FALSE(HasFatalFailure());
  iree_hal_executable_release(executable_);
  executable_ = nullptr;
  EXPECT_EQ(Status(iree_hal_semaphore_wait(done_[0], 1, iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE))
                .code(),
            StatusCode::kDataLoss);
  EXPECT_EQ(reports_.load(), 1u);
  EXPECT_EQ(errors_.load(), 0u);
  EXPECT_GE(queue_failures_.load(), 1u);
  // Fixture teardown waits for actual device and worker reclamation even when
  // the public signal failed before native completion.
}

TEST_P(GpuTsanTest, IndirectDispatchRelocatesPacketAndPreparationProgram) {
  Load(IREE_HAL_AMD_GPU_SANITIZER_REPORT_POLICY_REPORT_ONLY);
  ASSERT_FALSE(HasFatalFailure());
  const auto* family = iree_hal_queue_family(queues_[0]);
  const iree_hal_pool_family_access_t access = {
      .family = family,
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS,
      .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
  };
  const iree_hal_pool_scope_t scope = {
      .family_count = 1,
      .families = &access,
      .host =
          {
              .access = IREE_HAL_MEMORY_ACCESS_WRITE,
              .modes = IREE_HAL_MAPPING_MODE_SCOPED,
          },
  };
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &options,
                                           iree_allocator_system(), &pool_));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(12, 4),
      iree_infinite_timeout(), &counts_));
  const std::array<uint32_t, 3> counts = {3, 1, 1};
  IREE_ASSERT_OK(
      iree_hal_buffer_map_write(counts_, 0, counts.data(), sizeof(counts)));
  auto config = Config();
  config.workgroup_count_ref = iree_hal_make_buffer_ref(counts_, 0, 12);
  uint64_t value = 0;
  uint64_t expected_reports = 0;
  for (auto flags : {IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS,
                     IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS}) {
    SCOPED_TRACE(flags);
    ++value;
    for (uint32_t queue = 0; queue < 2; ++queue) {
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          queues_[queue], iree_hal_semaphore_list_empty(),
          {1, &done_[queue], &value}, executable_, race_, config,
          iree_const_byte_span_empty(), iree_hal_buffer_ref_list_empty(),
          nullptr, flags));
    }
    for (auto* semaphore : done_) {
      IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, value,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
    }
    expected_reports += 6;
    ASSERT_EQ(reports_.load(), expected_reports);
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        family, 0, IREE_HAL_COMMAND_CATEGORY_DISPATCH, 0, &recording_));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording_));
    const iree_hal_barrier_t acquire = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_HOST,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
        .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording_, &acquire));
    for (uint32_t i = 0; i < 17; ++i) {
      for (auto function : {safe_, race_}) {
        IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
            recording_, executable_, function, config,
            iree_const_byte_span_empty(), iree_hal_buffer_ref_list_empty(),
            flags));
      }
    }
    IREE_ASSERT_OK(iree_hal_command_buffer_end(recording_));
    ++value;
    for (uint32_t queue = 0; queue < 2; ++queue) {
      IREE_ASSERT_OK(iree_hal_queue_execute(
          queues_[queue], iree_hal_semaphore_list_empty(),
          {1, &done_[queue], &value}, recording_,
          iree_hal_buffer_binding_table_empty(), 0));
    }
    iree_hal_command_buffer_release(recording_);
    recording_ = nullptr;
    for (auto* semaphore : done_) {
      IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, value,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
    }
    expected_reports += 2 * 17 * 3;
    ASSERT_EQ(reports_.load(), expected_reports);
    ASSERT_EQ(errors_.load(), 0u);
  }
}

INSTANTIATE_TEST_SUITE_P(ShadowWindow, GpuTsanTest,
                         ::testing::Values(1u, 2u, 16u));

}  // namespace
}  // namespace iree::hal::amd::gpu::cts
