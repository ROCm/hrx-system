// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <atomic>
#include <cstring>
#include <string>
#include <tuple>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/profile_test_util.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/cts/tsan_images.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu::cts {
namespace {

// Both fixture kernels fit in one 16-byte application-memory chunk: a 32-byte
// record header, a 16-byte-aligned state bitmap, and 16 bytes of history per
// application byte. Keep the expected layout independent of reported geometry.
constexpr uint32_t kWorkgroupShadowStride = 32 + 16 + 16 * 16;

class GpuTsanTest : public GpuNativeTest,
                    public ::testing::WithParamInterface<
                        std::tuple<amdf_queue_command_type_t, uint32_t>> {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return std::get<0>(GetParam());
  }

  void SetUp() override {
    IREE_ASSERT_OK(memory_gate_.CreateFabric(&fabric_, UINT64_MAX,
                                             AMDF_NATIVE_LIFETIME_PROCESS));
    memory_gate_.RequireCodeUpload();
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    if (!(context_->device_info.features &
          AMDF_GPU_DEVICE_FEATURE_NATIVE_EVENTS)) {
      GTEST_SKIP() << "Native shader notification is unavailable";
    }
    context_->event_sink = {Publish, this};
    const auto gfx = context_->target_info.gfx_ip;
    const auto* files = iree_hal_amd_gpu_tsan_images_create();
    const char* selector = gfx.major == 9
                               ? (gfx.minor == 4 ? "gfx9-4-generic" : "gfx90a")
                           : gfx.major == 11 ? "gfx11-generic"
                                             : "gfx12-generic";
    for (size_t i = 0; i < iree_hal_amd_gpu_tsan_images_size(); ++i) {
      if (std::strstr(files[i].name, selector)) {
        artifact_ = iree_make_const_byte_span(files[i].data, files[i].size);
      }
    }
    ASSERT_NE(artifact_.data, nullptr);
    options_.tsan.memory.shadow_size = std::get<1>(GetParam());
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
      EXPECT_EQ(failure->status_code, self.expected_failure_);
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
    if (report->check_kind == IREE_HAL_DEVICE_TSAN_CHECK_KIND_SHADOW_CAPACITY) {
      const auto& capacity = report->detail.shadow_capacity;
      EXPECT_EQ(capacity.shadow_size, self.options_.tsan.memory.shadow_size);
      EXPECT_EQ(capacity.workgroup_local_memory_size, 8u);
      EXPECT_EQ(capacity.shadow_cell_size, 16u);
      EXPECT_EQ(capacity.shadow_history_count, 2u);
      EXPECT_EQ(capacity.workgroup_shadow_stride, kWorkgroupShadowStride);
      EXPECT_EQ(capacity.workgroup_ordinal,
                capacity.shadow_size / kWorkgroupShadowStride);
      EXPECT_EQ(capacity.required_shadow_end,
                (capacity.workgroup_ordinal + 1) * kWorkgroupShadowStride);
      self.capacity_reports_.fetch_add(1);
      return;
    }
    ASSERT_EQ(report->check_kind, IREE_HAL_DEVICE_TSAN_CHECK_KIND_DATA_RACE);
    const auto& data_race = report->detail.data_race;
    EXPECT_EQ(data_race.memory_space,
              IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_WORKGROUP);
    EXPECT_EQ(data_race.current_access_kind,
              IREE_HAL_DEVICE_TSAN_ACCESS_KIND_WRITE);
    EXPECT_EQ(data_race.prior_access_kind,
              IREE_HAL_DEVICE_TSAN_ACCESS_KIND_WRITE);
    EXPECT_EQ(data_race.access_length, 4u);
    EXPECT_NE(data_race.shadow_address, 0u);
    EXPECT_NE(report->source_dispatch_ptr, 0u);
    EXPECT_NE(event->source.executable_id, 0u);
    uint64_t identity = 0;
    self.executable_id_.compare_exchange_strong(identity,
                                                event->source.executable_id);
    EXPECT_EQ(self.executable_id_.load(), event->source.executable_id);
    self.last_shadow_.store(data_race.shadow_address);
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
  // Force configuration patching through staging before real shader execution.
  NativeMemoryGate memory_gate_;
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
  // Capacity failures reported before the excess workgroup executes.
  std::atomic<uint64_t> capacity_reports_ = 0;
  // Unexpected channel or device failures.
  std::atomic<uint64_t> errors_ = 0;
  // Failed-operation diagnostics accompanying the requested fail-device policy.
  std::atomic<uint64_t> queue_failures_ = 0;
  // Expected terminal status for the current fail-device policy witness.
  iree_status_code_t expected_failure_ = IREE_STATUS_DATA_LOSS;
  // Shared executable identity retained after the public handle is released.
  std::atomic<uint64_t> executable_id_ = 0;
  // Most recent native shadow address, inspected only after completion.
  std::atomic<uint64_t> last_shadow_ = 0;
};

TEST_P(GpuTsanTest, SharedRecordingSelectsQueueShadowAndJoinsArenaReuse) {
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
    ASSERT_EQ(capacity_reports_.load(), 0u);
    ASSERT_EQ(errors_.load(), 0u);
  }
}

TEST_P(GpuTsanTest, ProfileSharesOriginalImageAcrossQueuePlacements) {
  ASSERT_NO_FATAL_FAILURE(
      Load(IREE_HAL_AMD_GPU_SANITIZER_REPORT_POLICY_REPORT_ONLY));
  iree::hal::cts::TestProfileSink sink;
  iree::hal::cts::TestProfileSinkInitialize(&sink);
  iree::hal::cts::DeviceProfilingScope profile(device_);
  IREE_ASSERT_OK(
      profile.Begin(IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA,
                    iree::hal::cts::TestProfileSinkAsBase(&sink)));
  uint64_t one = 1;
  for (uint32_t queue = 0; queue < 2; ++queue) {
    ASSERT_NO_FATAL_FAILURE(Dispatch(queue, safe_, one));
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(done_[queue], one, iree_infinite_timeout(), 0));
  }
  iree_hal_executable_release(executable_);
  executable_ = nullptr;
  IREE_ASSERT_OK(profile.End());
  ASSERT_EQ(sink.code_objects.size(), 1u);
  ASSERT_EQ(sink.code_object_loads.size(), 2u);
  const auto& artifact = sink.code_objects[0];
  ASSERT_EQ(artifact.data.size(), artifact_.data_length);
  EXPECT_EQ(
      std::memcmp(artifact.data.data(), artifact_.data, artifact.data.size()),
      0);
  EXPECT_NE(sink.code_object_loads[0].load_delta,
            sink.code_object_loads[1].load_delta);
  for (const auto& load : sink.code_object_loads) {
    EXPECT_EQ(load.executable_id, artifact.record.executable_id);
    EXPECT_EQ(load.code_object_id, artifact.record.code_object_id);
    EXPECT_EQ(load.physical_device_ordinal, 0u);
    EXPECT_GT(load.load_size, 0u);
  }
  EXPECT_EQ(reports_.load(), 0u);
  EXPECT_EQ(errors_.load(), 0u);
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
  EXPECT_EQ(capacity_reports_.load(), 0u);
  EXPECT_EQ(errors_.load(), 0u);
  EXPECT_GE(queue_failures_.load(), 1u);
  // Fixture teardown waits for actual device and worker reclamation even when
  // the public signal failed before native completion.
}

TEST_P(GpuTsanTest, CapacityFailurePreservesStatusAndNativeRetirement) {
  expected_failure_ = IREE_STATUS_RESOURCE_EXHAUSTED;
  Load(IREE_HAL_AMD_GPU_SANITIZER_REPORT_POLICY_FAIL_DEVICE);
  ASSERT_FALSE(HasFatalFailure());
  // Only the first excess workgroup reports capacity; valid preceding
  // workgroups have disjoint data.
  const uint32_t workgroups =
      options_.tsan.memory.shadow_size / kWorkgroupShadowStride + 1;
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      queues_[0], iree_hal_semaphore_list_empty(), {1, &done_[0], &value},
      executable_, safe_, Config(workgroups), iree_const_byte_span_empty(),
      iree_hal_buffer_ref_list_empty(), nullptr, 0));
  iree_hal_executable_release(executable_);
  executable_ = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_semaphore_wait(done_[0], value, iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_EQ(capacity_reports_.load(), 1u);
  EXPECT_EQ(reports_.load(), 0u);
  EXPECT_EQ(errors_.load(), 0u);
  EXPECT_GE(queue_failures_.load(), 1u);
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
    ASSERT_EQ(capacity_reports_.load(), 0u);
    ASSERT_EQ(errors_.load(), 0u);
  }
}

// Each arena admits the three-workgroup replay cases while exercising
// different capacity boundaries.
INSTANTIATE_TEST_SUITE_P(
    ShadowBytes, GpuTsanTest,
    ::testing::Combine(::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                         AMDF_QUEUE_COMMAND_TYPE_GPU_AQL),
                       ::testing::Values(4 * kWorkgroupShadowStride,
                                         8 * kWorkgroupShadowStride,
                                         64 * kWorkgroupShadowStride)),
    [](const ::testing::TestParamInfo<
        std::tuple<amdf_queue_command_type_t, uint32_t>>& info) {
      return std::string(std::get<0>(info.param) ==
                                 AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
                             ? "Aql"
                             : "Pm4") +
             std::to_string(std::get<1>(info.param));
    });

}  // namespace
}  // namespace iree::hal::amd::gpu::cts
