// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <thread>
#include <tuple>
#include <vector>

#include "iree/base/internal/math.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/profile_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/driver/amd/gpu/cts/test_device.h"
#include "iree/hal/driver/amd/gpu/cts/testdata.h"
#include "iree/hal/driver/amd/gpu/executable.h"
#include "iree/hal/driver/amd/gpu/image/hip_images.h"
#include "iree/hal/driver/amd/gpu/image/hsaco.h"
#include "iree/hal/driver/amd/gpu/image/hsaco_test_util.h"
#include "iree/hal/driver/amd/gpu/queue/aql/command_buffer.h"
#include "iree/hal/driver/amd/gpu/queue/aql/publication.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/perf_counters.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/thread_trace.h"
#include "iree/hal/driver/amd/gpu/target/identity.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {
using iree::hal::amd::gpu::cts::GpuNativeTest;
using iree::hal::cts::Ref;

class GpuNativeProfileTestBase : public GpuNativeTest {
 protected:
  virtual iree_hal_device_profiling_data_families_t Families() const {
    return IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS;
  }

  virtual uint32_t ArgumentCapacity() const { return 32; }

  // Adjusts the session request beyond its data families.
  virtual void ConfigureProfiling(
      iree_hal_device_profiling_options_t* options) const {}

  // A family this device or queue protocol does not implement skips the test
  // instead of failing it.
  virtual bool SkipsUnimplementedFamilies() const { return false; }

  // Primary ring of each queue. Small by default, so capacity paths run.
  virtual uint32_t RingByteLength() const { return 4096; }

  void SetUp() override {
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    options_.queue.retirement_capacity = 4;
    options_.queue.argument_capacity = ArgumentCapacity();
    options_.queue.ring_byte_length = RingByteLength();
    Create();
    if (HasFatalFailure()) {
      return;
    }
    iree::hal::cts::TestProfileSinkInitialize(&sink_);
    const iree_hal_pool_family_access_t access = {
        .family = iree_hal_queue_family(queues_[0]),
        .usage = IREE_HAL_BUFFER_USAGE_STORAGE |
                 IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS,
        .interfaces =
            (UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS) |
            (UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE),
    };
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, {1, &access, {}}, &options,
                                             iree_allocator_system(),
                                             pool_.out()));
    iree_hal_device_profiling_options_t profile_options = {
        .data_families = Families(),
        .sink = iree::hal::cts::TestProfileSinkAsBase(&sink_),
    };
    ConfigureProfiling(&profile_options);
    iree_status_t begin_status =
        iree_hal_device_profiling_begin(device_, &profile_options);
    if (SkipsUnimplementedFamilies() &&
        iree_status_is_unimplemented(begin_status)) {
      iree_status_ignore(begin_status);
      GTEST_SKIP() << "device does not implement the requested data families";
    }
    IREE_ASSERT_OK(begin_status);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_FAILED_PRECONDITION,
        iree_hal_device_profiling_begin(device_, &profile_options));
  }

  void TearDown() override {
    if (device_) {
      IREE_EXPECT_OK(iree_hal_device_profiling_end(device_));
    }
    pool_.reset();
    GpuNativeTest::TearDown();
  }

  void Wait(iree_hal_semaphore_t* semaphore, uint64_t value) {
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(semaphore, value, iree_infinite_timeout(), 0));
  }

  void End(uint32_t queue_count = 2) {
    IREE_ASSERT_OK(iree_hal_device_profiling_end(device_));
    EXPECT_EQ(sink_.begin_count, 1);
    EXPECT_EQ(sink_.end_count, 1);
    EXPECT_EQ(sink_.queue_metadata_count, queue_count);
    EXPECT_FALSE(sink_.write_after_end);
    for (const auto& event : sink_.queue_events) {
      EXPECT_EQ(event.record_length, sizeof(event));
      EXPECT_EQ(event.physical_device_ordinal, 0u);
      EXPECT_LT(event.queue_ordinal, queue_count);
      EXPECT_EQ(event.stream_id, event.queue_ordinal + 1u);
      EXPECT_NE(event.event_id, 0u);
      EXPECT_NE(event.submission_id, 0u);
      if (event.ready_host_time_ns) {
        EXPECT_GE(event.ready_host_time_ns, event.host_time_ns);
      }
    }
  }

  std::vector<iree_hal_profile_queue_event_t> Events(
      iree_hal_profile_queue_event_type_t type) {
    std::vector<iree_hal_profile_queue_event_t> result;
    for (const auto& event : sink_.queue_events) {
      if (event.type == type) {
        result.push_back(event);
      }
    }
    return result;
  }

  // gfx IP version of the device under test, from its executable target.
  iree_hal_amd_gpu_gfxip_version_t TargetVersion() {
    const auto* targets =
        iree_hal_device_spec_executables(iree_hal_device_spec(device_));
    iree_hal_amd_gpu_target_identity_t identity = {};
    IREE_EXPECT_OK(iree_hal_amd_gpu_target_identity_parse_artifact_key(
        targets->targets[0].target_key, &identity));
    return identity.version;
  }

  // Whether the device's trace and counter tables ran on its exact target.
  // That holds on gfx1151 alone; elsewhere captures carry the unverified flag.
  bool TraceTableVerified() {
    iree_hal_amd_gpu_pm4_thread_trace_profile_t profile;
    IREE_EXPECT_OK(iree_hal_amd_gpu_pm4_thread_trace_profile_initialize(
        TargetVersion(), &profile));
    return profile.verified;
  }
  bool CounterTableVerified() {
    iree_hal_amd_gpu_pm4_perf_counter_profile_t profile;
    IREE_EXPECT_OK(iree_hal_amd_gpu_pm4_perf_counter_profile_initialize(
        TargetVersion(), &profile));
    return profile.verified;
  }

  void Load(Ref<iree_hal_executable_t>& executable,
            iree_hal_executable_function_t& function) {
    const auto* targets =
        iree_hal_device_spec_executables(iree_hal_device_spec(device_));
    iree_hal_executable_load_params_t params;
    iree_hal_executable_load_params_initialize(&params);
    IREE_ASSERT_OK(iree::hal::amd::gpu::cts::LookupCodeObject(
        iree_hal_amd_gpu_hip_images_create(),
        iree_hal_amd_gpu_hip_images_size(), targets->targets[0].target_key,
        &params.executable_data));
    loaded_image_ = params.executable_data;
    IREE_ASSERT_OK(iree_hal_executable_load(iree_hal_queue_family(queues_[0]),
                                            &targets->targets[0], &params,
                                            executable.out()));
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executable, IREE_SV("image_accumulate"), &function));
    sink_.expected_function_ordinal = (uint32_t)function.value;
  }

  void Record(iree_hal_executable_t* executable,
              iree_hal_executable_function_t function,
              iree_hal_command_buffer_mode_t mode, uint32_t dispatch_count,
              bool timestamps, uint64_t fixed_address,
              iree_hal_command_buffer_t** out_recording) {
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        iree_hal_queue_family(queues_[0]), mode,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH |
            (timestamps ? IREE_HAL_COMMAND_CATEGORY_TRANSFER : 0),
        fixed_address ? 0 : 1, out_recording));
    auto* recording = *out_recording;
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
    const iree_hal_barrier_t acquire = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
        .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
    };
    const iree_hal_memory_barrier_t memory = {
        .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
        .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                        IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
    };
    const iree_hal_barrier_t dependency = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
        .memory_barrier_count = 1,
        .memory_barriers = &memory,
    };
    const iree_hal_barrier_t release = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &acquire));
    if (timestamps) {
      IREE_ASSERT_OK(iree_hal_command_buffer_timestamp(
          recording, 0, IREE_HAL_EXECUTION_STAGE_DISPATCH,
          iree_hal_make_indirect_buffer_ref(0, 8, 8), 0));
    }
    auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
    config.workgroup_size[0] = 32;
    config.workgroup_size[1] = config.workgroup_size[2] = 1;
    auto reference = iree_hal_make_indirect_buffer_ref(0, 0, 4);
    for (uint32_t i = 0; i < dispatch_count; ++i) {
      if (i) {
        IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &dependency));
      }
      struct {
        // Caller-owned storage retained through all recorded native readers.
        uint64_t output;
        // Scalar increment copied by recording.
        uint32_t increment;
      } arguments = {fixed_address, i + 1};
      IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
          recording, executable, function, config,
          fixed_address ? iree_make_const_byte_span(&arguments, 12)
                        : iree_make_const_byte_span(&arguments.increment, 4),
          {fixed_address ? 0u : 1u, &reference},
          fixed_address ? IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS : 0));
    }
    if (timestamps) {
      IREE_ASSERT_OK(iree_hal_command_buffer_timestamp(
          recording, IREE_HAL_EXECUTION_STAGE_DISPATCH, 0,
          iree_hal_make_indirect_buffer_ref(0, 16, 8), 0));
    }
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
  }

  void ImportFile(iree_byte_span_t contents, iree_hal_memory_access_t access,
                  iree_hal_file_t** out_file) {
    iree_io_file_handle_t* handle = nullptr;
    IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE, contents, {},
        iree_allocator_system(), &handle));
    iree_status_t status = iree_hal_file_import(
        device_, nullptr, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, access, handle, 0,
        out_file);
    iree_io_file_handle_release(handle);
    IREE_ASSERT_OK(status);
  }

  // Borrowed embedded artifact passed to the most recent native executable
  // load.
  iree_const_byte_span_t loaded_image_ = {};
  // Explicit storage contract for dispatches, transfers, and mapping sources.
  Ref<iree_hal_pool_t> pool_;
  // Session sink remains live through idle end and device teardown.
  iree::hal::cts::TestProfileSink sink_;
};

class GpuNativeProfileTest
    : public GpuNativeProfileTestBase,
      public ::testing::WithParamInterface<amdf_queue_command_type_t> {
 protected:
  amdf_queue_command_type_t command_type() const override { return GetParam(); }
};

class GpuNativeDispatchProfileTest
    : public GpuNativeProfileTestBase,
      public ::testing::WithParamInterface<amdf_queue_command_type_t> {
 protected:
  amdf_queue_command_type_t command_type() const override { return GetParam(); }

  iree_hal_device_profiling_data_families_t Families() const override {
    return IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
           IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
           IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA;
  }
};

class GpuNativeArgumentProfileTest
    : public GpuNativeProfileTestBase,
      public ::testing::WithParamInterface<
          std::tuple<amdf_queue_command_type_t, bool>> {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return std::get<0>(GetParam());
  }

  // Several fresh records share each traversal before the small ring wraps.
  uint32_t ArgumentCapacity() const override { return 1024; }

  iree_hal_device_profiling_data_families_t Families() const override {
    return IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
           (std::get<1>(GetParam())
                ? IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS
                : 0);
  }
};

TEST_P(GpuNativeArgumentProfileTest,
       DependentDispatchesPreserveArgumentsAcrossReuse) {
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(4, 4),
      iree_infinite_timeout(), output.out()));
  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(Load(executable, function));
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  auto* timeline = done.get();
  uint64_t next = 1;
  uint32_t zero = 0, observed = 0, expected = 0;
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {1, &timeline, &next},
                                     output, 0, 4, &zero, 4, nullptr, 0));
  auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  auto reference = iree_hal_make_buffer_ref(output, 0, 4);
  const iree_hal_memory_barrier_t read = {
      .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ,
  };
  const iree_hal_memory_barrier_t write = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
  };
  const iree_hal_barrier_t acquire = {
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .memory_barrier_count = 1,
      .memory_barriers = &read,
  };
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .memory_barrier_count = 1,
      .memory_barriers = &write,
  };
  const iree_hal_barrier_list_t before = {1, &acquire};
  const iree_hal_barrier_list_t after = {1, &release};
  const iree_hal_queue_barriers_t device_local = {&before, &after};
  constexpr uint32_t kRounds = 4;
  constexpr uint32_t kDispatches = 32;
  for (uint32_t round = 0; round < kRounds; ++round) {
    for (uint32_t i = 0; i < kDispatches; ++i) {
      uint32_t increment = round * kDispatches + i + 1;
      expected += increment;
      uint64_t previous = next++;
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          queues_[0], {1, &timeline, &previous}, {1, &timeline, &next},
          executable, function, config,
          iree_make_const_byte_span(&increment, sizeof(increment)),
          {1, &reference}, &device_local, 0));
    }
    // Host visibility belongs to the explicit download, not each interior
    // dispatch. Every next round reuses retired argument and sample storage.
    uint64_t previous = next++;
    IREE_ASSERT_OK(iree_hal_queue_download(
        queues_[0], {1, &timeline, &previous}, {1, &timeline, &next}, output, 0,
        &observed, sizeof(observed), nullptr));
    ASSERT_NO_FATAL_FAILURE(Wait(done, next));
    ASSERT_EQ(observed, expected);
  }
  executable.reset();
  ASSERT_NO_FATAL_FAILURE(End());
  EXPECT_EQ(Events(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH).size(),
            kRounds * kDispatches);
  if (std::get<1>(GetParam())) {
    ASSERT_EQ(sink_.dispatch_events.size(), kRounds * kDispatches);
    uint64_t previous_end = 0;
    for (const auto& event : sink_.dispatch_events) {
      EXPECT_GT(event.start_tick, 0u);
      EXPECT_GE(event.start_tick, previous_end);
      EXPECT_GE(event.end_tick, event.start_tick);
      EXPECT_EQ(event.command_buffer_id, 0u);
      previous_end = event.end_tick;
    }
  } else {
    EXPECT_TRUE(sink_.dispatch_events.empty());
  }
}

TEST_P(GpuNativeArgumentProfileTest,
       RecordedCohortsRetainArgumentsAcrossBarriersAndReplays) {
  constexpr uint32_t kWidth = 8;
  constexpr uint32_t kReplays = 8;
  constexpr std::array<uint32_t, 2> kStages = {3, 9};
  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(Load(executable, function));
  auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  const iree_hal_memory_barrier_t memory = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
  };
  const iree_hal_barrier_t dependency = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .memory_barrier_count = 1,
      .memory_barriers = &memory,
  };
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  // Each cohort updates disjoint words, then the next cohort consumes them.
  // The longer recording crosses the native ring's packet capacity; replay
  // alternates distinct argument images through both queues' small arenas.
  std::array<Ref<iree_hal_command_buffer_t>, kStages.size()> recordings;
  std::array<std::vector<uint32_t>, kStages.size()> command_indices;
  std::array<uint32_t, kWidth> expected = {};
  for (size_t i = 0; i < recordings.size(); ++i) {
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        iree_hal_queue_family(queues_[0]),
        IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH, 1, recordings[i].out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(recordings[i]));
    for (uint32_t stage = 0; stage < kStages[i]; ++stage) {
      if (stage) {
        IREE_ASSERT_OK(
            iree_hal_command_buffer_barrier(recordings[i], &dependency));
      }
      for (uint32_t lane = 0; lane < kWidth; ++lane) {
        uint32_t increment = (stage + 1) * 16 + lane + 1;
        expected[lane] += increment * (kReplays / kStages.size());
        auto reference = iree_hal_make_indirect_buffer_ref(0, lane * 4, 4);
        IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
            recordings[i], executable, function, config,
            iree_make_const_byte_span(&increment, sizeof(increment)),
            {1, &reference}, 0));
        increment = UINT32_MAX;
        command_indices[i].push_back(stage * (kWidth + 1) + lane);
      }
    }
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recordings[i], &release));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(recordings[i]));
  }
  std::array<Ref<iree_hal_buffer_t>, 2> outputs;
  std::array<Ref<iree_hal_semaphore_t>, 2> completions;
  uint32_t zero = 0;
  uint64_t one = 1;
  for (size_t q = 0; q < queues_.size(); ++q) {
    CreateSemaphore(completions[q].out());
    auto* timeline = completions[q].get();
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, iree_hal_make_pool_reservation_request(sizeof(expected), 4),
        iree_infinite_timeout(), outputs[q].out()));
    IREE_ASSERT_OK(iree_hal_queue_fill(queues_[q], {}, {1, &timeline, &one},
                                       outputs[q], 0, sizeof(expected), &zero,
                                       sizeof(zero), nullptr, 0));
  }
  for (uint64_t replay = 0; replay < kReplays; ++replay) {
    uint64_t previous = replay + 1, next = previous + 1;
    for (size_t q = 0; q < queues_.size(); ++q) {
      auto* timeline = completions[q].get();
      iree_hal_buffer_binding_t binding = {
          .buffer = outputs[q],
          .length = sizeof(expected),
      };
      IREE_ASSERT_OK(iree_hal_queue_execute(
          queues_[q], {1, &timeline, &previous}, {1, &timeline, &next},
          recordings[replay % recordings.size()], {1, &binding}, 0));
    }
  }
  for (size_t q = 0; q < queues_.size(); ++q) {
    auto* timeline = completions[q].get();
    uint64_t previous = kReplays + 1, next = previous + 1;
    std::array<uint32_t, kWidth> observed = {};
    IREE_ASSERT_OK(iree_hal_queue_download(
        queues_[q], {1, &timeline, &previous}, {1, &timeline, &next},
        outputs[q], 0, observed.data(), sizeof(observed), nullptr));
    ASSERT_NO_FATAL_FAILURE(Wait(timeline, next));
    EXPECT_EQ(observed, expected);
    for (size_t replay = 0; replay < kReplays; ++replay) {
      const auto& indices = command_indices[replay % recordings.size()];
      sink_.expected_dispatch_command_indices.insert(
          sink_.expected_dispatch_command_indices.end(), indices.begin(),
          indices.end());
    }
  }
  sink_.expected_dispatch_flags =
      IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER;
  ASSERT_NO_FATAL_FAILURE(End());
  EXPECT_EQ(sink_.dispatch_events.size(),
            std::get<1>(GetParam())
                ? sink_.expected_dispatch_command_indices.size()
                : 0u);
  for (const auto& event : sink_.dispatch_events) {
    EXPECT_GT(event.start_tick, 0u);
    EXPECT_GE(event.end_tick, event.start_tick);
  }
}

INSTANTIATE_TEST_SUITE_P(
    NativeQueues, GpuNativeArgumentProfileTest,
    ::testing::Combine(::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                         AMDF_QUEUE_COMMAND_TYPE_GPU_AQL),
                       ::testing::Bool()));

TEST_P(GpuNativeDispatchProfileTest,
       AcquiredQueueRetainsRecordedSamplesAfterFinalRelease) {
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(24, 8),
      iree_infinite_timeout(), output.out()));
  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(Load(executable, function));
  Ref<iree_hal_command_buffer_t> recording;
  ASSERT_NO_FATAL_FAILURE(
      Record(executable, function,
             IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA, 1,
             /*timestamps=*/true, 0, recording.out()));
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  auto* timeline = done.get();
  uint64_t value = 1;
  uint32_t zero = 0;
  std::array<uint64_t, 3> observed = {};
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {1, &timeline, &value},
                                     output, 0, 24, &zero, 4, nullptr, 0));
  iree_hal_buffer_binding_t binding = {.buffer = output, .length = 24};
  for (uint32_t i = 0; i < 2; ++i) {
    Ref<iree_hal_queue_t> acquired;
    IREE_ASSERT_OK(AcquireQueue(acquired.out()));
    uint64_t wait = value++;
    IREE_ASSERT_OK(iree_hal_queue_execute(acquired, {1, &timeline, &wait},
                                          {1, &timeline, &value}, recording,
                                          {1, &binding}, 0));
    wait = value++;
    IREE_ASSERT_OK(iree_hal_queue_download(
        queues_[0], {1, &timeline, &wait}, {1, &timeline, &value}, output, 0,
        observed.data(), sizeof(observed), nullptr));
    acquired.reset();
    ASSERT_NO_FATAL_FAILURE(Wait(done, value));
    EXPECT_EQ(observed[0], i + 1);
    EXPECT_LE(observed[1], observed[2]);
  }
  recording.reset();
  executable.reset();
  sink_.expected_dispatch_flags =
      IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER;
  sink_.expected_dispatch_command_indices = {2, 2};
  ASSERT_NO_FATAL_FAILURE(End(3));
  ASSERT_EQ(sink_.dispatch_events.size(), 2u);
  for (const auto& event : sink_.dispatch_events) {
    EXPECT_GT(event.start_tick, 0u);
    EXPECT_GE(event.end_tick, event.start_tick);
    EXPECT_NE(event.command_buffer_id, 0u);
  }
  auto executions = Events(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE);
  ASSERT_EQ(executions.size(), 2u);
  EXPECT_EQ(executions[0].queue_ordinal, 2u);
  EXPECT_EQ(executions[1].queue_ordinal, 2u);
  EXPECT_EQ(executions[0].submission_id, 1u);
  EXPECT_EQ(executions[1].submission_id, 2u);
}

TEST_P(GpuNativeDispatchProfileTest,
       FixedArgumentsPreserveInvocationSpecificDispatchSamples) {
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(4, 4),
      iree_infinite_timeout(), output.out()));
  iree_hal_memory_scope_t scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      group_, {IREE_HAL_MEMORY_SITE_PROGRAM, iree_hal_queue_family(queues_[0])},
      &scope));
  iree_hal_buffer_native_binding_slot_t slot;
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      pool_, scope, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS, &slot));
  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(Load(executable, function));
  Ref<iree_hal_command_buffer_t> recording;
  ASSERT_NO_FATAL_FAILURE(
      Record(executable, function,
             IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA, 3, false,
             iree_hal_buffer_native_binding(output, slot).device_address,
             recording.out()));
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  auto* timeline = done.get();
  uint64_t value = 1;
  uint32_t zero = 0, observed = 0;
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {1, &timeline, &value},
                                     output, 0, 4, &zero, 4, nullptr, 0));
  for (uint32_t i = 0; i < 2; ++i) {
    uint64_t wait = value++;
    IREE_ASSERT_OK(iree_hal_queue_execute(queues_[i], {1, &timeline, &wait},
                                          {1, &timeline, &value}, recording, {},
                                          0));
  }
  recording.reset();
  executable.reset();
  uint64_t wait = value++;
  IREE_ASSERT_OK(iree_hal_queue_download(queues_[0], {1, &timeline, &wait},
                                         {1, &timeline, &value}, output, 0,
                                         &observed, sizeof(observed), nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(done, value));
  EXPECT_EQ(observed, 12u);
  sink_.expected_dispatch_flags =
      IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER;
  sink_.expected_dispatch_command_indices = {1, 3, 5, 1, 3, 5};
  ASSERT_NO_FATAL_FAILURE(End());
  ASSERT_EQ(sink_.dispatch_events.size(), 6u);
  for (const auto& event : sink_.dispatch_events) {
    EXPECT_GT(event.start_tick, 0u);
    EXPECT_GE(event.end_tick, event.start_tick);
    EXPECT_NE(event.command_buffer_id, 0u);
  }
  EXPECT_EQ(Events(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE).size(), 2u);
}

TEST_P(GpuNativeProfileTest, ExistingAcquiredQueueParticipatesInCapture) {
  ASSERT_NO_FATAL_FAILURE(End());
  Ref<iree_hal_queue_t> acquired;
  IREE_ASSERT_OK(AcquireQueue(acquired.out()));
  iree::hal::cts::TestProfileSink sink;
  iree::hal::cts::TestProfileSinkInitialize(&sink);
  iree::hal::cts::DeviceProfilingScope capture(device_);
  const iree_hal_device_profiling_options_t options = {
      .data_families = Families(),
      .sink = iree::hal::cts::TestProfileSinkAsBase(&sink),
  };
  IREE_ASSERT_OK(capture.Begin(&options));
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  auto* timeline = done.get();
  uint64_t value = 1;
  IREE_ASSERT_OK(
      iree_hal_queue_barrier(acquired, {}, {1, &timeline, &value}, nullptr, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(done, value));
  IREE_ASSERT_OK(capture.End());
  // Capture detachment must cover acquired queues while they remain usable.
  ++value;
  IREE_ASSERT_OK(
      iree_hal_queue_barrier(acquired, {}, {1, &timeline, &value}, nullptr, 0));
  acquired.reset();
  EXPECT_EQ(sink.begin_count, 1);
  EXPECT_EQ(sink.end_count, 1);
  EXPECT_EQ(sink.queue_metadata_count, 3);
  ASSERT_EQ(sink.queue_events.size(), 1u);
  EXPECT_EQ(sink.queue_events[0].queue_ordinal, 2u);
  EXPECT_FALSE(sink.write_after_end);
}

TEST_P(GpuNativeDispatchProfileTest, DirectSamplesSurviveExecutableRelease) {
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(4, 4),
      iree_infinite_timeout(), output.out()));
  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(Load(executable, function));
  uint64_t launch_address = 0;
  if (command_type() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL) {
    const iree_hal_amd_gpu_aql_function_t* native_function = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_gpu_executable_lookup_aql_for_queue(
        executable, function, 0, &native_function));
    launch_address = native_function->kernel_object;
  } else {
    const iree_hal_amd_gpu_pm4_function_t* native_function = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_gpu_executable_lookup_pm4(executable, function,
                                                          &native_function));
    launch_address = (uint64_t{native_function->launch.program[0]} << 8) |
                     (uint64_t{native_function->launch.program[1]} << 40);
  }
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  auto* timeline = done.get();
  uint64_t previous = 0, next = 1;
  uint32_t zero = 0, increment = 1, observed = 0;
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {1, &timeline, &next},
                                     output, 0, 4, &zero, 4, nullptr, 0));
  auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  auto reference = iree_hal_make_buffer_ref(output, 0, 4);
  for (size_t i = 0; i < 16; ++i) {
    previous = next++;
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        queues_[i % 2], {1, &timeline, &previous}, {1, &timeline, &next},
        executable, function, config,
        iree_make_const_byte_span(&increment, sizeof(increment)),
        {1, &reference}, nullptr, 0));
  }
  previous = next++;
  IREE_ASSERT_OK(iree_hal_queue_download(queues_[0], {1, &timeline, &previous},
                                         {1, &timeline, &next}, output, 0,
                                         &observed, sizeof(observed), nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(done, next));
  EXPECT_EQ(observed, 16u);
  executable.reset();
  ASSERT_NO_FATAL_FAILURE(End());
  ASSERT_EQ(sink_.dispatch_events.size(), 16u);
  ASSERT_EQ(sink_.executable_ids.size(), 1u);
  ASSERT_EQ(sink_.code_objects.size(), 1u);
  ASSERT_EQ(sink_.code_object_loads.size(), 1u);
  const auto& artifact = sink_.code_objects[0];
  const auto& load = sink_.code_object_loads[0];
  EXPECT_EQ(artifact.record.executable_id, sink_.executable_ids[0]);
  EXPECT_EQ(load.executable_id, artifact.record.executable_id);
  EXPECT_EQ(load.code_object_id, artifact.record.code_object_id);
  EXPECT_EQ(load.physical_device_ordinal, 0u);
  ASSERT_EQ(artifact.data.size(), loaded_image_.data_length);
  EXPECT_EQ(std::memcmp(artifact.data.data(), loaded_image_.data,
                        artifact.data.size()),
            0);
  // Offline attribution uses only the emitted image and placement. It must
  // recover the native launch address even after executable release: the
  // descriptor for AQL and the instruction entry for PM4.
  auto* source = iree::hal::amd::gpu::testing::ByteSequence::Create(
      iree_make_const_byte_span(artifact.data.data(), artifact.data.size()), 0);
  iree_hal_amd_gpu_image_t image = {};
  iree_status_t status = iree_hal_amd_gpu_hsaco_image_initialize(
      source, iree_allocator_system(), &image);
  iree_byte_sequence_release(source);
  IREE_ASSERT_OK(status);
  EXPECT_EQ(load.load_size, image.byte_length);
  const auto& kernel = image.kernels[function.value];
  const uint64_t launch_offset =
      command_type() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
          ? kernel.descriptor_offset
          : kernel.entry_offset;
  EXPECT_EQ(static_cast<uint64_t>(load.load_delta) + image.virtual_base +
                launch_offset,
            launch_address);
  iree_hal_amd_gpu_image_deinitialize(&image);
  for (const auto& event : sink_.dispatch_events) {
    EXPECT_GT(event.start_tick, 0u);
    EXPECT_GE(event.end_tick, event.start_tick);
    EXPECT_EQ(event.command_buffer_id, 0u);
  }
  // Flush emits one queue stream at a time. Each FIFO has eight independent
  // invocation samples even though the small argument ring repeatedly wraps.
  for (size_t i = 0; i < sink_.dispatch_events.size(); ++i) {
    if (i % 8) {
      EXPECT_GE(sink_.dispatch_events[i].start_tick,
                sink_.dispatch_events[i - 1].end_tick);
    }
  }
}

TEST_P(GpuNativeDispatchProfileTest,
       CaptureFiltersPreserveDirectAndResidentWork) {
  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(Load(executable, function));
  std::array<Ref<iree_hal_command_buffer_t>, 2> recordings;
  for (auto& recording : recordings) {
    ASSERT_NO_FATAL_FAILURE(
        Record(executable, function,
               IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA, 2, false,
               0, recording.out()));
  }
  Ref<iree_hal_command_buffer_t> unprofiled;
  ASSERT_NO_FATAL_FAILURE(
      Record(executable, function, 0, 2, false, 0, unprofiled.out()));
  const uint64_t first_id = iree_hal_command_buffer_profile_id(recordings[0]);
  const uint64_t second_id = iree_hal_command_buffer_profile_id(recordings[1]);
  ASSERT_NE(first_id, second_id);
  ASSERT_NE(first_id, 0u);
  ASSERT_NE(second_id, 0u);
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(4, 4),
      iree_infinite_timeout(), output.out()));
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  auto* timeline = done.get();
  uint64_t value = 0;
  auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  auto reference = iree_hal_make_buffer_ref(output, 0, 4);
  iree_hal_buffer_binding_t binding = {.buffer = output, .length = 4};
  constexpr uint32_t kPattern =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN;
  constexpr uint32_t kCommandBuffer =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_BUFFER_ID;
  constexpr uint32_t kCommand =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_INDEX;
  constexpr uint32_t kDevice =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_PHYSICAL_DEVICE_ORDINAL;
  constexpr uint32_t kQueue =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_QUEUE_ORDINAL;
  struct Case {
    // Existing public filter, applied to one complete session.
    iree_hal_profile_capture_filter_t filter;
    // Selected direct dispatches from the two queues.
    uint32_t direct_count;
    // Selected command indices in queue-stream emission order.
    std::vector<uint32_t> command_indices;
  };
  const Case cases[] = {
      {{}, 2, {1, 3, 1, 3}},
      {{.flags = kQueue, .queue_ordinal = 1}, 1, {1, 3}},
      {{.flags = kDevice, .physical_device_ordinal = 1}, 0, {}},
      {{
           .flags = kPattern,
           .executable_function_pattern = IREE_SV("image_acc*"),
       },
       2,
       {1, 3, 1, 3}},
      {{.flags = kPattern, .executable_function_pattern = IREE_SV("absent_*")},
       0,
       {}},
      {{.flags = kCommandBuffer, .command_buffer_id = first_id}, 0, {1, 3}},
      {{.flags = kCommand, .command_index = 3}, 0, {3, 3}},
      {{
           .flags = kPattern | kCommandBuffer | kCommand | kDevice | kQueue,
           .executable_function_pattern = IREE_SV("image_acc*"),
           .command_buffer_id = second_id,
           .command_index = 3,
           .physical_device_ordinal = 0,
           .queue_ordinal = 1,
       },
       0,
       {3}},
  };
  ASSERT_NO_FATAL_FAILURE(End());
  for (const auto& test : cases) {
    SCOPED_TRACE(test.filter.flags);
    iree::hal::cts::TestProfileSink sink;
    iree::hal::cts::TestProfileSinkInitialize(&sink);
    sink.expected_function_ordinal = (uint32_t)function.value;
    iree::hal::cts::DeviceProfilingScope capture(device_);
    const iree_hal_device_profiling_options_t options = {
        .data_families = Families(),
        .sink = iree::hal::cts::TestProfileSinkAsBase(&sink),
        .capture_filter = test.filter,
    };
    IREE_ASSERT_OK(capture.Begin(&options));
    uint32_t zero = 0;
    uint64_t previous = value++;
    IREE_ASSERT_OK(iree_hal_queue_fill(
        queues_[0], {previous ? 1u : 0u, &timeline, &previous},
        {1, &timeline, &value}, output, 0, 4, &zero, 4, nullptr, 0));
    const uint32_t increment = 1;
    for (size_t i = 0; i < queues_.size(); ++i) {
      previous = value++;
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          queues_[i], {1, &timeline, &previous}, {1, &timeline, &value},
          executable, function, config,
          iree_make_const_byte_span(&increment, sizeof(increment)),
          {1, &reference}, nullptr, 0));
    }
    ASSERT_NO_FATAL_FAILURE(Wait(done, value));
    IREE_ASSERT_OK(iree_hal_device_profiling_flush(device_));
    EXPECT_EQ(sink.dispatch_events.size(), test.direct_count);
    sink.expected_dispatch_flags =
        IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER;
    sink.expected_dispatch_command_indices.assign(test.direct_count,
                                                  UINT32_MAX);
    sink.expected_dispatch_command_indices.insert(
        sink.expected_dispatch_command_indices.end(),
        test.command_indices.begin(), test.command_indices.end());
    for (size_t i = 0; i < recordings.size(); ++i) {
      previous = value++;
      IREE_ASSERT_OK(iree_hal_queue_execute(
          queues_[i], {1, &timeline, &previous}, {1, &timeline, &value},
          recordings[i], {1, &binding}, 0));
    }
    const bool excludes_unprofiled = (test.filter.flags & kCommandBuffer) != 0;
    if (excludes_unprofiled) {
      // Exclusion is known from invocation identity, so this recording needs
      // neither retained profile metadata nor native sample storage.
      previous = value++;
      IREE_ASSERT_OK(iree_hal_queue_execute(
          queues_[0], {1, &timeline, &previous}, {1, &timeline, &value},
          unprofiled, {1, &binding}, 0));
    }
    uint32_t observed = 0;
    previous = value++;
    IREE_ASSERT_OK(iree_hal_queue_download(
        queues_[0], {1, &timeline, &previous}, {1, &timeline, &value}, output,
        0, &observed, sizeof(observed), nullptr));
    ASSERT_NO_FATAL_FAILURE(Wait(done, value));
    EXPECT_EQ(observed, excludes_unprofiled ? 11u : 8u);
    IREE_ASSERT_OK(capture.End());
    EXPECT_EQ(sink.dispatch_events.size(),
              test.direct_count + test.command_indices.size());
    EXPECT_EQ(sink.queue_events.size(), excludes_unprofiled ? 7u : 6u);
    EXPECT_EQ(sink.begin_count, 1);
    EXPECT_EQ(sink.end_count, 1);
    for (const auto& event : sink.dispatch_events) {
      EXPECT_GT(event.start_tick, 0u);
      EXPECT_GE(event.end_tick, event.start_tick);
    }
  }
}

TEST_P(GpuNativeDispatchProfileTest,
       ResidentTransferExitJoinsBeforePublishingSamples) {
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(8192, 4),
      iree_infinite_timeout(), output.out()));
  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(Load(executable, function));
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  auto* timeline = done.get();
  uint64_t value = 1;
  uint32_t zero = 0;
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {1, &timeline, &value},
                                     output, 0, 8192, &zero, 4, nullptr, 0));

  Ref<iree_hal_command_buffer_t> recording;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      iree_hal_queue_family(queues_[0]),
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH | IREE_HAL_COMMAND_CATEGORY_TRANSFER,
      0, recording.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
  const iree_hal_barrier_t acquire = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &acquire));
  auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  auto reference = iree_hal_make_buffer_ref(output, 0, 4);
  uint32_t increment = 1;
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      recording, executable, function, config,
      iree_make_const_byte_span(&increment, sizeof(increment)), {1, &reference},
      0));
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      recording, iree_hal_make_buffer_ref(output, 0, 4096),
      iree_hal_make_buffer_ref(output, 4096, 4096), 0));
  // Program exit owns the final asynchronous copy join, including when the
  // invocation also publishes dispatch samples through its continuation.
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
  for (size_t i = 0; i < 8; ++i) {
    uint64_t previous = value++;
    IREE_ASSERT_OK(iree_hal_queue_execute(queues_[0], {1, &timeline, &previous},
                                          {1, &timeline, &value}, recording, {},
                                          0));
  }
  std::array<uint32_t, 1024> observed;
  uint64_t previous = value++;
  IREE_ASSERT_OK(iree_hal_queue_download(
      queues_[0], {1, &timeline, &previous}, {1, &timeline, &value}, output,
      4096, observed.data(), sizeof(observed), nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(done, value));
  EXPECT_EQ(observed[0], 8u);
  EXPECT_TRUE(std::all_of(observed.begin() + 1, observed.end(),
                          [](uint32_t word) { return word == 0; }));
  sink_.expected_dispatch_flags =
      IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER;
  sink_.expected_dispatch_command_indices.assign(8, 1);
  ASSERT_NO_FATAL_FAILURE(End());
  ASSERT_EQ(sink_.dispatch_events.size(), 8u);
  for (const auto& event : sink_.dispatch_events) {
    EXPECT_GT(event.start_tick, 0u);
    EXPECT_GE(event.end_tick, event.start_tick);
  }
}

TEST_P(GpuNativeDispatchProfileTest,
       RecordedReplaysKeepInvocationPrivateSamples) {
  constexpr std::array<uint32_t, 2> kDispatchCounts = {16, 64};
  constexpr uint32_t kReplayCount = 16;
  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(Load(executable, function));
  // Alternate bounded and multi-block recordings while both queues retain
  // invocation samples. The small argument rings force reuse between them.
  std::array<Ref<iree_hal_command_buffer_t>, kDispatchCounts.size()> recordings;
  std::array<uint64_t, kDispatchCounts.size()> command_ids;
  uint32_t expected = 0;
  uint32_t dispatches_per_queue = 0;
  for (size_t i = 0; i < recordings.size(); ++i) {
    ASSERT_NO_FATAL_FAILURE(
        Record(executable, function,
               IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA,
               kDispatchCounts[i], false, 0, recordings[i].out()));
    command_ids[i] = iree_hal_command_buffer_profile_id(recordings[i]);
    const uint32_t count = kDispatchCounts[i];
    const uint32_t replays = kReplayCount / kDispatchCounts.size();
    expected += replays * count * (count + 1) / 2;
    dispatches_per_queue += replays * count;
  }
  std::array<Ref<iree_hal_buffer_t>, 2> outputs;
  std::array<Ref<iree_hal_semaphore_t>, 2> done;
  std::array<uint64_t, 2> values = {1, 1};
  std::array<uint32_t, 2> observed = {};
  uint32_t zero = 0;
  for (size_t q = 0; q < 2; ++q) {
    CreateSemaphore(done[q].out());
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, iree_hal_make_pool_reservation_request(4, 4),
        iree_infinite_timeout(), outputs[q].out()));
    auto* timeline = done[q].get();
    IREE_ASSERT_OK(iree_hal_queue_fill(queues_[q], {},
                                       {1, &timeline, &values[q]}, outputs[q],
                                       0, 4, &zero, 4, nullptr, 0));
  }
  for (size_t i = 0; i < kReplayCount; ++i) {
    for (size_t q = 0; q < 2; ++q) {
      auto* timeline = done[q].get();
      uint64_t previous = values[q]++;
      iree_hal_buffer_binding_t binding = {.buffer = outputs[q], .length = 4};
      IREE_ASSERT_OK(iree_hal_queue_execute(
          queues_[q], {1, &timeline, &previous}, {1, &timeline, &values[q]},
          recordings[i % recordings.size()], {1, &binding}, 0));
    }
  }
  for (size_t q = 0; q < 2; ++q) {
    auto* timeline = done[q].get();
    uint64_t previous = values[q]++;
    IREE_ASSERT_OK(iree_hal_queue_download(
        queues_[q], {1, &timeline, &previous}, {1, &timeline, &values[q]},
        outputs[q], 0, &observed[q], sizeof(observed[q]), nullptr));
    ASSERT_NO_FATAL_FAILURE(Wait(done[q], values[q]));
    EXPECT_EQ(observed[q], expected);
  }
  for (auto& recording : recordings) {
    recording.reset();
  }
  executable.reset();
  sink_.expected_dispatch_flags =
      IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER;
  for (size_t i = 0; i < outputs.size() * kReplayCount; ++i) {
    for (uint32_t j = 0; j < kDispatchCounts[i % kDispatchCounts.size()]; ++j) {
      sink_.expected_dispatch_command_indices.push_back(j * 2 + 1);
    }
  }
  ASSERT_NO_FATAL_FAILURE(End());
  ASSERT_EQ(sink_.dispatch_events.size(),
            outputs.size() * dispatches_per_queue);
  EXPECT_EQ(sink_.command_buffer_ids.size(), command_ids.size());
  for (uint64_t command_id : command_ids) {
    EXPECT_EQ(std::count(sink_.command_buffer_ids.begin(),
                         sink_.command_buffer_ids.end(), command_id),
              1);
  }
  EXPECT_EQ(sink_.executable_ids.size(), 1u);
  size_t index = 0;
  for (size_t q = 0; q < outputs.size(); ++q) {
    uint64_t previous_end = 0;
    for (size_t i = 0; i < kReplayCount; ++i) {
      const size_t program = i % kDispatchCounts.size();
      for (uint32_t j = 0; j < kDispatchCounts[program]; ++j, ++index) {
        const auto& event = sink_.dispatch_events[index];
        EXPECT_EQ(event.command_buffer_id, command_ids[program]);
        EXPECT_GT(event.start_tick, 0u);
        EXPECT_GE(event.end_tick, event.start_tick);
        // Native firmware may stamp adjacent end/begin boundaries at one tick.
        EXPECT_GE(event.start_tick, previous_end);
        previous_end = event.end_tick;
        if (j) {
          EXPECT_EQ(event.submission_id,
                    sink_.dispatch_events[index - 1].submission_id);
        }
      }
    }
  }
}

TEST_P(GpuNativeProfileTest, ProfileCapableRecordingRunsWithoutDeviceCapture) {
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(4, 4),
      iree_infinite_timeout(), output.out()));
  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(Load(executable, function));
  Ref<iree_hal_command_buffer_t> recording;
  ASSERT_NO_FATAL_FAILURE(
      Record(executable, function,
             IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA, 2, false, 0,
             recording.out()));
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  auto* timeline = done.get();
  uint64_t one = 1, two = 2, three = 3;
  uint32_t zero = 0, observed = 0;
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {1, &timeline, &one},
                                     output, 0, 4, &zero, 4, nullptr, 0));
  iree_hal_buffer_binding_t binding = {.buffer = output, .length = 4};
  IREE_ASSERT_OK(iree_hal_queue_execute(queues_[0], {1, &timeline, &one},
                                        {1, &timeline, &two}, recording,
                                        {1, &binding}, 0));
  IREE_ASSERT_OK(iree_hal_queue_download(queues_[0], {1, &timeline, &two},
                                         {1, &timeline, &three}, output, 0,
                                         &observed, sizeof(observed), nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(done, three));
  EXPECT_EQ(observed, 3u);
  ASSERT_NO_FATAL_FAILURE(End());
  EXPECT_TRUE(sink_.dispatch_events.empty());
}

TEST_P(GpuNativeDispatchProfileTest,
       UsedQueuesRestartCaptureAndReuseInstrumentedRecordings) {
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(4, 4),
      iree_infinite_timeout(), output.out()));
  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(Load(executable, function));
  ASSERT_NO_FATAL_FAILURE(End());
  iree_hal_memory_scope_t scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      group_, {IREE_HAL_MEMORY_SITE_PROGRAM, iree_hal_queue_family(queues_[0])},
      &scope));
  iree_hal_buffer_native_binding_slot_t slot;
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      pool_, scope, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS, &slot));
  const uint64_t address =
      iree_hal_buffer_native_binding(output, slot).device_address;
  for (bool fixed : {false, true}) {
    SCOPED_TRACE(fixed);
    Ref<iree_hal_command_buffer_t> recording;
    ASSERT_NO_FATAL_FAILURE(
        Record(executable, function,
               IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA, 2, false,
               fixed ? address : 0, recording.out()));
    const uint8_t* template_bytes = nullptr;
    std::vector<uint8_t> saved_template;
    if (fixed && command_type() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL) {
      const auto* program =
          iree_hal_amd_gpu_aql_command_buffer_program(recording);
      if (program->constants.page) {
        ASSERT_NE(program->host_template, nullptr);
        ASSERT_EQ(program->host_template->fixup_count, 0u);
        template_bytes =
            reinterpret_cast<const uint8_t*>(program->host_template + 1);
        saved_template.assign(
            template_bytes,
            template_bytes + program->host_template->byte_length);
        iree_hal_amd_gpu_aql_command_invocation_t invocation;
        IREE_ASSERT_OK(iree_hal_amd_gpu_aql_command_invocation_initialize(
            recording, {}, nullptr, 0, &invocation));
        EXPECT_EQ(iree_hal_amd_gpu_aql_execute_argument_length(&invocation),
                  0u);
        iree_hal_amd_gpu_aql_command_invocation_deinitialize(&invocation);
      }
    }

    Ref<iree_hal_semaphore_t> done;
    CreateSemaphore(done.out());
    auto* timeline = done.get();
    uint64_t value = 0;
    const uint32_t zero = 0;
    auto run = [&]() {
      uint64_t previous = value++;
      IREE_ASSERT_OK(iree_hal_queue_fill(
          queues_[0], {previous ? 1u : 0u, &timeline, &previous},
          {1, &timeline, &value}, output, 0, 4, &zero, 4, nullptr, 0));
      previous = value++;
      iree_hal_buffer_binding_t binding = {.buffer = output, .length = 4};
      IREE_ASSERT_OK(iree_hal_queue_execute(
          queues_[0], {1, &timeline, &previous}, {1, &timeline, &value},
          recording, {fixed ? 0u : 1u, &binding}, 0));
      previous = value++;
      uint32_t observed = 0;
      IREE_ASSERT_OK(iree_hal_queue_download(
          queues_[0], {1, &timeline, &previous}, {1, &timeline, &value}, output,
          0, &observed, sizeof(observed), nullptr));
      ASSERT_NO_FATAL_FAILURE(Wait(done, value));
      EXPECT_EQ(observed, 3u);
    };
    for (uint32_t i = 0; i < 4; ++i) {
      SCOPED_TRACE(i);
      // Reuse the same resident template before, during, and after capture.
      ASSERT_NO_FATAL_FAILURE(run());
      iree::hal::cts::TestProfileSink sink;
      iree::hal::cts::TestProfileSinkInitialize(&sink);
      sink.expected_function_ordinal = (uint32_t)function.value;
      sink.expected_dispatch_flags =
          IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER;
      sink.expected_dispatch_command_indices = {1, 3};
      iree::hal::cts::DeviceProfilingScope capture(device_);
      IREE_ASSERT_OK(capture.Begin(
          Families(), iree::hal::cts::TestProfileSinkAsBase(&sink)));
      ASSERT_NO_FATAL_FAILURE(run());
      IREE_ASSERT_OK(capture.End());
      ASSERT_EQ(sink.dispatch_events.size(), 2u);
      for (const auto& event : sink.dispatch_events) {
        EXPECT_GT(event.start_tick, 0u);
        EXPECT_GE(event.end_tick, event.start_tick);
      }
      ASSERT_NO_FATAL_FAILURE(run());
      EXPECT_FALSE(sink.write_after_end);
      EXPECT_EQ(sink.dispatch_events.size(), 2u);
      if (template_bytes) {
        EXPECT_EQ(std::memcmp(template_bytes, saved_template.data(),
                              saved_template.size()),
                  0);
      }
    }
  }
}

TEST_P(GpuNativeDispatchProfileTest,
       IndirectGeometryAndPacketPreparationRemainExact) {
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(4096, 8),
      iree_infinite_timeout(), buffer.out()));
  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(Load(executable, function));
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  auto* timeline = done.get();
  uint64_t value = 0;
  const uint32_t zero = 0, marker = 0xC132869F;
  const std::array<uint32_t, 3> counts = {2, 1, 1};
  const iree_hal_barrier_t acquire = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS |
                           IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
  };
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  for (bool packet : {false, true}) {
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executable,
        packet ? IREE_SV("image_dispatch_packet") : IREE_SV("image_geometry"),
        &function));
    sink_.expected_function_ordinal = (uint32_t)function.value;
    for (bool recorded : {false, true}) {
      for (auto flags : {IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS,
                         IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS}) {
        SCOPED_TRACE(::testing::Message()
                     << packet << "/" << recorded << "/" << flags);
        uint64_t previous = value++;
        IREE_ASSERT_OK(iree_hal_queue_fill(
            queues_[0], {previous ? 1u : 0u, &timeline, &previous},
            {1, &timeline, &value}, buffer, 0, 4096, &zero, 4, nullptr, 0));
        previous = value++;
        IREE_ASSERT_OK(iree_hal_queue_upload(
            queues_[0], {1, &timeline, &previous}, {1, &timeline, &value},
            counts.data(), buffer, 2048, sizeof(counts), nullptr));
        auto config = iree_hal_make_static_dispatch_config(0, 0, 0);
        config.workgroup_size[0] = 32;
        config.workgroup_size[1] = config.workgroup_size[2] = 1;
        const auto constants =
            iree_make_const_byte_span(&marker, sizeof(marker));
        previous = value++;
        Ref<iree_hal_command_buffer_t> recording;
        if (recorded) {
          IREE_ASSERT_OK(iree_hal_command_buffer_create(
              iree_hal_queue_family(queues_[0]),
              IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA,
              IREE_HAL_COMMAND_CATEGORY_DISPATCH, 1, recording.out()));
          IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
          IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &acquire));
          config.workgroup_count_ref =
              iree_hal_make_indirect_buffer_ref(0, 2048, 12);
          auto reference = iree_hal_make_indirect_buffer_ref(0, 0, 2048);
          IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
              recording, executable, function, config, constants,
              {1, &reference}, flags));
          IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
          IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
          iree_hal_buffer_binding_t binding = {
              .buffer = buffer,
              .length = 4096,
          };
          IREE_ASSERT_OK(iree_hal_queue_execute(
              queues_[0], {1, &timeline, &previous}, {1, &timeline, &value},
              recording, {1, &binding}, 0));
        } else {
          config.workgroup_count_ref =
              iree_hal_make_buffer_ref(buffer, 2048, 12);
          auto reference = iree_hal_make_buffer_ref(buffer, 0, 2048);
          IREE_ASSERT_OK(iree_hal_queue_dispatch(
              queues_[0], {1, &timeline, &previous}, {1, &timeline, &value},
              executable, function, config, constants, {1, &reference}, nullptr,
              flags));
        }
        std::array<uint32_t, 512> observed = {};
        previous = value++;
        IREE_ASSERT_OK(iree_hal_queue_download(
            queues_[0], {1, &timeline, &previous}, {1, &timeline, &value},
            buffer, 0, observed.data(), sizeof(observed), nullptr));
        ASSERT_NO_FATAL_FAILURE(Wait(done, value));
        if (packet) {
          EXPECT_EQ(observed[3], 64u);
          EXPECT_EQ(observed[4], 1u);
          EXPECT_EQ(observed[5], 1u);
          EXPECT_EQ(observed[18], marker);
        } else {
          // Native indirect launches expose all three dimensions; unused
          // dimensions have size/count one, as in the ordinary launch path.
          const std::array<uint32_t, 8> expected = {2, 1, 1,      32,
                                                    1, 1, marker, 3};
          for (size_t group = 0; group < 2; ++group) {
            for (size_t field = 0; field < expected.size(); ++field) {
              EXPECT_EQ(observed[group * 8 + field], expected[field])
                  << "group " << group << " field " << field;
            }
          }
        }
        sink_.expected_dispatch_flags =
            IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_INDIRECT_PARAMETERS |
            (recorded ? IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER
                      : 0);
        sink_.expected_dispatch_command_indices =
            recorded ? std::vector<uint32_t>{1} : std::vector<uint32_t>{};
        // Device-produced counts are deliberately unknown to host metadata.
        // The indirect flag distinguishes them from static zero-work launches.
        std::fill(std::begin(sink_.expected_workgroup_count),
                  std::end(sink_.expected_workgroup_count), 0);
        IREE_ASSERT_OK(iree_hal_device_profiling_flush(device_));
        ASSERT_EQ(sink_.dispatch_events.size(), 1u);
        EXPECT_GT(sink_.dispatch_events[0].start_tick, 0u);
        EXPECT_GE(sink_.dispatch_events[0].end_tick,
                  sink_.dispatch_events[0].start_tick);
        sink_.dispatch_events.clear();
      }
    }
  }
  ASSERT_NO_FATAL_FAILURE(End());
}

TEST_P(GpuNativeDispatchProfileTest,
       UnexecutedDispatchesProduceNoDeviceSamples) {
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(4, 4),
      iree_infinite_timeout(), output.out()));
  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(Load(executable, function));
  Ref<iree_hal_semaphore_t> gate, done;
  CreateSemaphore(gate.out());
  CreateSemaphore(done.out());
  auto* gate_handle = gate.get();
  auto* done_handle = done.get();
  uint64_t one = 1, two = 2;
  uint32_t increment = 1;
  auto config = iree_hal_make_static_dispatch_config(0, 1, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  auto reference = iree_hal_make_buffer_ref(output, 0, 4);
  auto constants = iree_make_const_byte_span(&increment, sizeof(increment));
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      queues_[0], {}, {1, &done_handle, &one}, executable, function, config,
      constants, {1, &reference}, nullptr, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(done, one));
  Ref<iree_hal_command_buffer_t> recording;
  ASSERT_NO_FATAL_FAILURE(
      Record(executable, function, 0, 2, false, 0, recording.out()));
  iree_hal_buffer_binding_t binding = {.buffer = output, .length = 4};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_queue_execute(queues_[0], {}, {1, &done_handle, &two}, recording,
                             {1, &binding}, 0));
  config.workgroup_count[0] = 1;
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      queues_[0], {1, &gate_handle, &one}, {1, &done_handle, &two}, executable,
      function, config, constants, {1, &reference}, nullptr, 0));
  iree_hal_semaphore_fail(gate, iree_status_from_code(IREE_STATUS_CANCELLED));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_hal_semaphore_wait(done, two, iree_infinite_timeout(), 0));
  ASSERT_NO_FATAL_FAILURE(End());
  EXPECT_TRUE(sink_.dispatch_events.empty());
}

TEST_P(GpuNativeProfileTest, ConsumerBeforeProducerRecordsObservedReadiness) {
  Ref<iree_hal_semaphore_t> producer, consumer;
  CreateSemaphore(producer.out());
  CreateSemaphore(consumer.out());
  auto* producer_handle = producer.get();
  auto* consumer_handle = consumer.get();
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[1], {1, &producer_handle, &one},
                                        {1, &consumer_handle, &one}, nullptr,
                                        0));
  IREE_ASSERT_OK(iree_hal_device_profiling_flush(device_));
  EXPECT_TRUE(sink_.queue_events.empty());
  const iree_time_t release_time = iree_time_now();
  IREE_ASSERT_OK(iree_hal_queue_barrier(
      queues_[0], {}, {1, &producer_handle, &one}, nullptr, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(consumer, one));
  ASSERT_NO_FATAL_FAILURE(End());
  ASSERT_EQ(sink_.queue_events.size(), 2u);
  const auto& deferred = sink_.queue_events[1];
  EXPECT_EQ(deferred.queue_ordinal, 1u);
  EXPECT_EQ(deferred.wait_count, 1u);
  EXPECT_EQ(deferred.signal_count, 1u);
  EXPECT_EQ(deferred.dependency_strategy,
            IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER);
  EXPECT_GE(deferred.ready_host_time_ns, release_time);
}

TEST_P(GpuNativeProfileTest, FailedDependencyRecordsNoReadiness) {
  Ref<iree_hal_semaphore_t> gate, done;
  CreateSemaphore(gate.out());
  CreateSemaphore(done.out());
  auto* gate_handle = gate.get();
  auto* done_handle = done.get();
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[0], {1, &gate_handle, &one},
                                        {1, &done_handle, &one}, nullptr, 0));
  iree_hal_semaphore_fail(gate, iree_status_from_code(IREE_STATUS_CANCELLED));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_hal_semaphore_wait(done, one, iree_infinite_timeout(), 0));
  ASSERT_NO_FATAL_FAILURE(End());
  ASSERT_EQ(sink_.queue_events.size(), 1u);
  EXPECT_EQ(sink_.queue_events[0].ready_host_time_ns, 0);
}

TEST_P(GpuNativeProfileTest,
       DirectAndReusableRecordedWorkHaveSeparateIdentities) {
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(4, 4),
      iree_infinite_timeout(), output.out()));
  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(Load(executable, function));
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  auto* timeline = done.get();
  uint64_t one = 1, two = 2, three = 3, four = 4, five = 5;
  uint32_t zero = 0, increment = 1, observed = 0;
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {1, &timeline, &one},
                                     output, 0, 4, &zero, 4, nullptr, 0));
  auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  auto reference = iree_hal_make_buffer_ref(output, 0, 4);
  auto constants = iree_make_const_byte_span(&increment, sizeof(increment));
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      queues_[1], {1, &timeline, &one}, {1, &timeline, &two}, executable,
      function, config, constants, {1, &reference}, nullptr, 0));
  Ref<iree_hal_command_buffer_t> recording;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      iree_hal_queue_family(queues_[0]), 0, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      1, recording.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
  const iree_hal_barrier_t acquire = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &acquire));
  auto indirect = iree_hal_make_indirect_buffer_ref(0, 0, 4);
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      recording, executable, function, config, constants, {1, &indirect}, 0));
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
  const uint64_t recording_id = iree_hal_command_buffer_profile_id(recording);
  iree_hal_buffer_binding_t binding = {
      .buffer = output,
      .offset = 0,
      .length = 4,
  };
  IREE_ASSERT_OK(iree_hal_queue_execute(queues_[0], {1, &timeline, &two},
                                        {1, &timeline, &three}, recording,
                                        {1, &binding}, 0));
  IREE_ASSERT_OK(iree_hal_queue_execute(queues_[1], {1, &timeline, &three},
                                        {1, &timeline, &four}, recording,
                                        {1, &binding}, 0));
  IREE_ASSERT_OK(iree_hal_queue_download(queues_[0], {1, &timeline, &four},
                                         {1, &timeline, &five}, output, 0,
                                         &observed, sizeof(observed), nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(done, five));
  EXPECT_EQ(observed, 3u);
  ASSERT_NO_FATAL_FAILURE(End());
  ASSERT_EQ(sink_.queue_events.size(), 5u);
  ASSERT_EQ(Events(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH).size(), 1u);
  const auto executions = Events(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE);
  ASSERT_EQ(executions.size(), 2u);
  for (const auto& event : executions) {
    EXPECT_EQ(event.command_buffer_id, recording_id);
    // Count the encoded payload operations; invocation fixup and clone work
    // belongs to native transport and contributes no additional operation.
    EXPECT_EQ(event.operation_count, 3u);  // Acquire, dispatch, and release.
  }
}

TEST_P(GpuNativeProfileTest,
       QueuedAllocationFileAndHostEffectsExcludeTransportEvents) {
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  auto* timeline = done.get();
  std::array<uint64_t, 7> values = {1, 2, 3, 4, 5, 6, 7};
  auto point = [&](size_t i) {
    return iree_hal_semaphore_list_t{1, &timeline, &values[i]};
  };
  auto request = iree_hal_make_pool_reservation_request(256, 4);
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[0], {}, point(0), pool_, 1,
                                       &request, buffer.out()));
  std::array<uint32_t, 64> source, target = {};
  source.fill(0x12345678);
  Ref<iree_hal_file_t> source_file, target_file;
  ASSERT_NO_FATAL_FAILURE(
      ImportFile(iree_make_byte_span(source.data(), sizeof(source)),
                 IREE_HAL_MEMORY_ACCESS_READ, source_file.out()));
  ASSERT_NO_FATAL_FAILURE(
      ImportFile(iree_make_byte_span(target.data(), sizeof(target)),
                 IREE_HAL_MEMORY_ACCESS_WRITE, target_file.out()));
  IREE_ASSERT_OK(iree_hal_queue_read_linear(queues_[1], point(0), point(1),
                                            source_file, 0, buffer, 0,
                                            sizeof(source), nullptr, 0));
  std::array<iree_hal_transfer_operation_t, 2> transfers = {};
  uint32_t pattern = 0xABCDEF00, update = 0x13579BDF;
  transfers[0].type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
  transfers[0].fill = {buffer, 0, 4, &pattern, 4};
  transfers[1].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE;
  transfers[1].update = {
      .source_buffer = &update,
      .target_buffer = buffer,
      .target_offset = 4,
      .length = 4,
  };
  IREE_ASSERT_OK(iree_hal_queue_transfer(queues_[0], point(1), point(2),
                                         transfers.size(), transfers.data(),
                                         nullptr));
  IREE_ASSERT_OK(iree_hal_queue_write_linear(queues_[1], point(2), point(3),
                                             buffer, 0, target_file, 0,
                                             sizeof(target), nullptr, 0));
  bool called = false;
  const iree_hal_host_call_t call = {
      .fn =
          [](void* data, const uint64_t[4], iree_hal_host_call_context_t*) {
            *static_cast<bool*>(data) = true;
            return iree_ok_status();
          },
      .user_data = &called,
  };
  const uint64_t arguments[4] = {};
  IREE_ASSERT_OK(iree_hal_queue_host_call(queues_[0], point(3), point(4), call,
                                          arguments, 0));
  auto* buffer_handle = buffer.get();
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[1], point(4), point(5), 1,
                                         &buffer_handle));
  ASSERT_NO_FATAL_FAILURE(Wait(done, values[5]));
  EXPECT_TRUE(called);
  source[0] = pattern;
  source[1] = update;
  EXPECT_EQ(source, target);
  ASSERT_NO_FATAL_FAILURE(End());
  ASSERT_EQ(sink_.queue_events.size(), 6u);
  for (auto type : {IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA,
                    IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ,
                    IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_TRANSFER,
                    IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE,
                    IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL,
                    IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA}) {
    ASSERT_EQ(Events(type).size(), 1u);
  }
  const auto batch = Events(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_TRANSFER)[0];
  EXPECT_EQ(batch.operation_count, 2u);
  EXPECT_EQ(batch.payload_length, 8u);
  const auto read = Events(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ)[0];
  EXPECT_EQ(read.operation_count, 1u);
  EXPECT_EQ(read.payload_length, sizeof(source));
  const auto write = Events(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE)[0];
  EXPECT_EQ(write.operation_count, 1u);
  EXPECT_EQ(write.payload_length, sizeof(target));
  EXPECT_EQ(Events(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL)[0].signal_count,
            1u);
}

TEST_P(GpuNativeProfileTest, TimestampIsOnePublicEventWithoutInternalDispatch) {
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(8, 8),
      iree_infinite_timeout(), output.out()));
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  auto* timeline = done.get();
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_timestamp(queues_[0], {}, {1, &timeline, &one},
                                          output, 0, nullptr, 0));
  output.reset();
  ASSERT_NO_FATAL_FAILURE(Wait(done, one));
  ASSERT_NO_FATAL_FAILURE(End());
  ASSERT_EQ(sink_.queue_events.size(), 1u);
  const auto& event = sink_.queue_events.front();
  EXPECT_EQ(event.type, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_TIMESTAMP);
  EXPECT_EQ(event.operation_count, 1u);
  EXPECT_EQ(event.wait_count, 0u);
  EXPECT_EQ(event.signal_count, 1u);
  EXPECT_TRUE(sink_.dispatch_events.empty());
}

TEST_P(GpuNativeProfileTest, MappingBatchIsOnePublicEvent) {
  const iree_hal_pool_family_access_t access = {
      .family = iree_hal_queue_family(queues_[0]),
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
      .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
  };
  const iree_hal_pool_scope_t scope = {1, &access, {}};
  iree_hal_pool_t* source_pool = pool_.get();
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, scope, {1, &source_pool}, iree_allocator_system(),
      &capabilities));
  const auto source = iree_hal_pool_query_mapping_requirements(pool_);
  const uint64_t unit =
      iree_max(source.minimum_granularity,
               iree_max(capabilities.mapping_granularity,
                        capabilities.reservation_granularity));
  const iree_hal_virtual_buffer_options_t options = {
      .byte_length = 2 * unit,
      .mapping_granularity = unit,
  };
  Ref<iree_hal_buffer_t> reservation, backing;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
      group_, scope, {1, &source_pool}, &options, iree_allocator_system(),
      reservation.out()));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(2 * unit, 0),
      iree_infinite_timeout(), backing.out()));
  iree_hal_memory_scope_t program_scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      group_, {IREE_HAL_MEMORY_SITE_PROGRAM, access.family}, &program_scope));
  std::array<iree_hal_memory_mapping_t, 2> mappings;
  for (size_t i = 0; i < mappings.size(); ++i) {
    mappings[i] = {
        .target = reservation,
        .scope = program_scope,
        .target_range = {i * unit, unit},
        .source = backing,
        .source_offset = i * unit,
        .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
        .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
    };
  }
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  auto* timeline = done.get();
  uint64_t one = 1, two = 2;
  IREE_ASSERT_OK(iree_hal_queue_map(queues_[0], {}, {1, &timeline, &one},
                                    mappings.size(), mappings.data(), 0));
  for (auto& mapping : mappings) {
    mapping.previous_source = mapping.source;
    mapping.previous_source_offset = mapping.source_offset;
    mapping.source = nullptr;
    mapping.source_offset = 0;
    mapping.access = 0;
    mapping.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND;
  }
  IREE_ASSERT_OK(iree_hal_queue_map(queues_[1], {1, &timeline, &one},
                                    {1, &timeline, &two}, mappings.size(),
                                    mappings.data(), 0));
  ASSERT_NO_FATAL_FAILURE(Wait(done, two));
  ASSERT_NO_FATAL_FAILURE(End());
  ASSERT_EQ(sink_.queue_events.size(), 2u);
  for (const auto& event : sink_.queue_events) {
    EXPECT_EQ(event.type, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_MAP);
    EXPECT_EQ(event.operation_count, 2u);
    EXPECT_EQ(event.signal_count, 1u);
  }
}

TEST_P(GpuNativeProfileTest, EmptyOperationsPreservePublicIdentities) {
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(4, 4),
      iree_infinite_timeout(), buffer.out()));
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  auto* timeline = done.get();
  uint64_t one = 1, two = 2;
  const uint32_t pattern = 0x12345678;
  std::array<iree_hal_transfer_operation_t, 3> transfers = {};
  for (size_t i = 0; i < transfers.size(); ++i) {
    transfers[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
    transfers[i].fill = {buffer, 0, i == 1 ? 4u : 0u, &pattern, 4};
  }
  IREE_ASSERT_OK(iree_hal_queue_transfer(queues_[0], {}, {1, &timeline, &one},
                                         transfers.size(), transfers.data(),
                                         nullptr));
  Ref<iree_hal_command_buffer_t> recording;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      iree_hal_queue_family(queues_[0]), 0, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      0, recording.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
  IREE_ASSERT_OK(iree_hal_queue_execute(queues_[0], {1, &timeline, &one},
                                        {1, &timeline, &two}, recording, {},
                                        0));
  ASSERT_NO_FATAL_FAILURE(Wait(done, two));
  ASSERT_NO_FATAL_FAILURE(End());
  ASSERT_EQ(sink_.queue_events.size(), 2u);
  const auto fills = Events(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL);
  ASSERT_EQ(fills.size(), 1u);
  EXPECT_EQ(fills[0].operation_count, 1u);
  EXPECT_EQ(fills[0].payload_length, 4u);
  const auto executions = Events(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE);
  ASSERT_EQ(executions.size(), 1u);
  EXPECT_EQ(executions[0].operation_count, 0u);
  EXPECT_EQ(executions[0].command_buffer_id,
            iree_hal_command_buffer_profile_id(recording));
}

TEST_P(GpuNativeProfileTest, ChunkedRecordingHasOnePublicEvent) {
  constexpr uint32_t kDispatchCount = 64;
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(4, 4),
      iree_infinite_timeout(), output.out()));
  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(Load(executable, function));
  Ref<iree_hal_command_buffer_t> recording;
  ASSERT_NO_FATAL_FAILURE(Record(executable, function, 0, kDispatchCount, false,
                                 0, recording.out()));
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  auto* timeline = done.get();
  uint64_t one = 1, two = 2, three = 3;
  uint32_t zero = 0, observed = 0;
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {1, &timeline, &one},
                                     output, 0, 4, &zero, 4, nullptr, 0));
  iree_hal_buffer_binding_t binding = {.buffer = output, .length = 4};
  IREE_ASSERT_OK(iree_hal_queue_execute(queues_[0], {1, &timeline, &one},
                                        {1, &timeline, &two}, recording,
                                        {1, &binding}, 0));
  IREE_ASSERT_OK(iree_hal_queue_download(queues_[0], {1, &timeline, &two},
                                         {1, &timeline, &three}, output, 0,
                                         &observed, sizeof(observed), nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(done, three));
  EXPECT_EQ(observed, kDispatchCount * (kDispatchCount + 1) / 2);
  ASSERT_NO_FATAL_FAILURE(End());
  ASSERT_EQ(sink_.queue_events.size(), 3u);
  const auto executions = Events(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE);
  ASSERT_EQ(executions.size(), 1u);
  EXPECT_EQ(executions[0].operation_count, 2 * kDispatchCount + 1);
}

TEST_P(GpuNativeProfileTest, ConcurrentSubmittersPreserveEveryPublicIdentity) {
  constexpr size_t kThreads = 4, kOperations = 128;
  std::array<Ref<iree_hal_semaphore_t>, kThreads> completions;
  std::array<std::thread, kThreads> threads;
  for (auto& completion : completions) {
    CreateSemaphore(completion.out());
  }
  for (size_t t = 0; t < kThreads; ++t) {
    threads[t] = std::thread([&, t] {
      auto* timeline = completions[t].get();
      for (uint64_t i = 0; i < kOperations; ++i) {
        uint64_t value = i + 1;
        IREE_ASSERT_OK(iree_hal_queue_barrier(
            queues_[t % queues_.size()], {i ? 1u : 0u, &timeline, &i},
            {1, &timeline, &value}, nullptr, 0));
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (auto& completion : completions) {
    ASSERT_NO_FATAL_FAILURE(Wait(completion, kOperations));
  }
  ASSERT_NO_FATAL_FAILURE(End());
  ASSERT_EQ(sink_.queue_events.size(), kThreads * kOperations);
  for (uint32_t queue = 0; queue < queues_.size(); ++queue) {
    std::vector<uint64_t> submissions, events;
    for (const auto& event : sink_.queue_events) {
      if (event.queue_ordinal == queue) {
        submissions.push_back(event.submission_id);
        events.push_back(event.event_id);
      }
    }
    std::sort(submissions.begin(), submissions.end());
    std::sort(events.begin(), events.end());
    ASSERT_EQ(submissions.size(), kThreads * kOperations / queues_.size());
    for (size_t i = 0; i < submissions.size(); ++i) {
      EXPECT_EQ(submissions[i], i + 1);
      EXPECT_EQ(events[i], i + 1);
    }
  }
}

// Thread traces on the native PM4 queue. Needs a GPU generation with a
// qualified register table; every other device skips.
class GpuNativeTraceProfileTest : public GpuNativeProfileTestBase {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return AMDF_QUEUE_COMMAND_TYPE_GPU_PM4;
  }

  iree_hal_device_profiling_data_families_t Families() const override {
    return IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
           IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
           IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |
           IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES;
  }

  void ConfigureProfiling(
      iree_hal_device_profiling_options_t* options) const override {
    options->executable_traces.buffer_byte_length = 128 * 1024;
    // Budget for a second trace that the held bracket then has to refuse.
    options->executable_traces.max_trace_count = 2;
  }

  bool SkipsUnimplementedFamilies() const override { return true; }

  // Every traced engine reports how it was captured, and each trace chunk
  // agrees with its capture record.
  void ExpectCaptures(bool command_buffer_scope, uint32_t dispatch_count) {
    ASSERT_FALSE(sink_.executable_trace_captures.empty());
    uint64_t written = 0;
    for (const auto& capture : sink_.executable_trace_captures) {
      EXPECT_EQ(capture.trace_id, 1u);
      EXPECT_EQ(capture.physical_device_ordinal, 0u);
      EXPECT_NE(capture.shader_engine_mask & (1u << capture.shader_engine), 0u);
      EXPECT_EQ(capture.buffer_byte_length, 128u * 1024);
      EXPECT_EQ(capture.dispatch_count, dispatch_count);
      EXPECT_EQ(capture.trace_limit, 2u);
      EXPECT_LE(capture.written_byte_length, capture.buffer_byte_length);
      // The stop sequence ran in the ring and the finish completed.
      EXPECT_TRUE(capture.flags &
                  IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_FINISHED);
      EXPECT_FALSE(capture.flags &
                   IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_UNCOLLECTED);
      // A bracket that never ran on this exact target says so.
      EXPECT_EQ(
          (capture.flags &
           IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_UNVERIFIED_TARGET) !=
              0,
          !TraceTableVerified());
      EXPECT_EQ(
          (capture.flags &
           IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_COMMAND_BUFFER_SCOPE) !=
              0,
          command_buffer_scope);
      written += capture.written_byte_length;
    }
    EXPECT_EQ(sink_.executable_trace_captures.size(),
              (size_t)iree_math_count_ones_u32(
                  sink_.executable_trace_captures[0].shader_engine_mask));
    // The traced SIMD saw the dispatch on at least one engine.
    EXPECT_GT(written, 0u);
    for (const auto& trace : sink_.executable_traces) {
      const auto capture = std::find_if(
          sink_.executable_trace_captures.begin(),
          sink_.executable_trace_captures.end(), [&](const auto& candidate) {
            return candidate.shader_engine == trace.record.shader_engine;
          });
      ASSERT_NE(capture, sink_.executable_trace_captures.end());
      EXPECT_EQ(trace.record.data_length, capture->written_byte_length);
      EXPECT_EQ(trace.record.format,
                IREE_HAL_PROFILE_EXECUTABLE_TRACE_FORMAT_AMDGPU_ATT);
      EXPECT_EQ(trace.record.trace_id, 1u);
    }
  }
};

TEST_F(GpuNativeTraceProfileTest, DirectDispatchIsBracketedAndNamesItsEvent) {
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(4, 4),
      iree_infinite_timeout(), output.out()));
  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(Load(executable, function));
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  auto* timeline = done.get();
  uint64_t previous = 0, next = 1;
  uint32_t zero = 0, increment = 1, observed = 0;
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {1, &timeline, &next},
                                     output, 0, 4, &zero, 4, nullptr, 0));
  auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  auto reference = iree_hal_make_buffer_ref(output, 0, 4);
  // The first dispatch takes the bracket; the rest have budget but run
  // untraced, because the bracket is held until the final flush.
  for (size_t i = 0; i < 4; ++i) {
    previous = next++;
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        queues_[0], {1, &timeline, &previous}, {1, &timeline, &next},
        executable, function, config,
        iree_make_const_byte_span(&increment, sizeof(increment)),
        {1, &reference}, nullptr, 0));
  }
  previous = next++;
  IREE_ASSERT_OK(iree_hal_queue_download(queues_[0], {1, &timeline, &previous},
                                         {1, &timeline, &next}, output, 0,
                                         &observed, sizeof(observed), nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(done, next));
  // Tracing changed nothing the kernels computed.
  EXPECT_EQ(observed, 4u);
  executable.reset();
  ASSERT_NO_FATAL_FAILURE(End());
  ASSERT_EQ(sink_.dispatch_events.size(), 4u);
  ASSERT_NO_FATAL_FAILURE(ExpectCaptures(/*command_buffer_scope=*/false, 1));
  ASSERT_FALSE(sink_.executable_traces.empty());
  for (const auto& trace : sink_.executable_traces) {
    EXPECT_EQ(trace.record.flags,
              IREE_HAL_PROFILE_EXECUTABLE_TRACE_FLAG_DISPATCH_EVENT);
    EXPECT_EQ(trace.record.dispatch_event_id,
              sink_.dispatch_events[0].event_id);
    EXPECT_EQ(trace.record.submission_id,
              sink_.dispatch_events[0].submission_id);
    EXPECT_EQ(trace.record.executable_id,
              sink_.dispatch_events[0].executable_id);
    EXPECT_EQ(trace.record.command_buffer_id, 0u);
  }
  // Skipped candidates are counted against the trace that held the bracket.
  EXPECT_EQ(sink_.executable_trace_captures[0].skipped_count, 3u);
}

TEST_F(GpuNativeTraceProfileTest, RecordedExecutionIsBracketedWhole) {
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(24, 8),
      iree_infinite_timeout(), output.out()));
  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(Load(executable, function));
  Ref<iree_hal_command_buffer_t> recording;
  ASSERT_NO_FATAL_FAILURE(
      Record(executable, function,
             IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA, 3,
             /*timestamps=*/true, 0, recording.out()));
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  auto* timeline = done.get();
  uint64_t value = 1;
  uint32_t zero = 0;
  std::array<uint64_t, 3> observed = {};
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {1, &timeline, &value},
                                     output, 0, 24, &zero, 4, nullptr, 0));
  iree_hal_buffer_binding_t binding = {.buffer = output, .length = 24};
  uint64_t wait = value++;
  IREE_ASSERT_OK(iree_hal_queue_execute(queues_[0], {1, &timeline, &wait},
                                        {1, &timeline, &value}, recording,
                                        {1, &binding}, 0));
  wait = value++;
  IREE_ASSERT_OK(iree_hal_queue_download(
      queues_[0], {1, &timeline, &wait}, {1, &timeline, &value}, output, 0,
      observed.data(), sizeof(observed), nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(done, value));
  // Three dispatches adding 1, 2 and 3, bracketed by the recording's clocks.
  EXPECT_EQ(observed[0] & 0xffffffffu, 6u);
  EXPECT_LE(observed[1], observed[2]);
  recording.reset();
  executable.reset();
  sink_.expected_dispatch_flags =
      IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER;
  sink_.expected_dispatch_command_indices = {2, 4, 6};
  ASSERT_NO_FATAL_FAILURE(End());
  ASSERT_EQ(sink_.dispatch_events.size(), 3u);
  ASSERT_NO_FATAL_FAILURE(ExpectCaptures(/*command_buffer_scope=*/true, 3));
  ASSERT_FALSE(sink_.executable_traces.empty());
  for (const auto& trace : sink_.executable_traces) {
    // No single dispatch or command owns a whole-recording trace.
    EXPECT_EQ(trace.record.flags, IREE_HAL_PROFILE_EXECUTABLE_TRACE_FLAG_NONE);
    EXPECT_EQ(trace.record.dispatch_event_id, 0u);
    EXPECT_EQ(trace.record.command_buffer_id,
              sink_.dispatch_events[0].command_buffer_id);
    EXPECT_EQ(trace.record.submission_id,
              sink_.dispatch_events[0].submission_id);
    EXPECT_EQ(trace.record.command_index, UINT32_MAX);
  }
}

// Hardware counters on the native PM4 queue. Needs a GPU generation with a
// qualified register table; every other device skips.
class GpuNativeCounterProfileTest : public GpuNativeProfileTestBase {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return AMDF_QUEUE_COMMAND_TYPE_GPU_PM4;
  }

  iree_hal_device_profiling_data_families_t Families() const override {
    return IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
           IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
           IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |
           IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES;
  }

  void ConfigureProfiling(
      iree_hal_device_profiling_options_t* options) const override {
    // The default engine set plus the per-workgroup-processor set, bounded
    // below what the tests dispatch.
    options->counter_set_count = 1;
    options->counter_sets = &selection_;
    options->counter_samples.max_sample_count = 3;
  }

  bool SkipsUnimplementedFamilies() const override { return true; }

  // A direct dispatch carries its start, two sample and stop sequences in the
  // ring itself: about 1,500 DWORDs with both sets.
  uint32_t RingByteLength() const override { return 64 * 1024; }

  // Sum of a named counter's values in one sample.
  uint64_t Value(const iree::hal::cts::TestProfileSink::CounterSample& sample,
                 const char* name) {
    for (const auto& counter : sink_.counters) {
      if (counter.name != name) {
        continue;
      }
      uint64_t sum = 0;
      for (uint32_t i = 0; i < counter.record.sample_value_count; ++i) {
        sum += sample.values[counter.record.sample_value_offset + i];
      }
      return sum;
    }
    ADD_FAILURE() << "no counter named " << name;
    return 0;
  }

  // Every sample counts exactly one 32-workitem workgroup of the test kernel:
  // one wave, whatever its lane width.
  void ExpectOneWorkgroupPerSample(size_t sample_count) {
    ASSERT_EQ(sink_.counters.size(), 9u);
    ASSERT_EQ(sink_.counter_samples.size(), sample_count);
    for (const auto& sample : sink_.counter_samples) {
      EXPECT_EQ(sample.record.scope,
                IREE_HAL_PROFILE_COUNTER_SAMPLE_SCOPE_DISPATCH);
      EXPECT_TRUE(sample.record.flags &
                  IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DISPATCH_EVENT);
      EXPECT_EQ(Value(sample, "SQG_WAVES"), 1u);
      EXPECT_EQ(Value(sample, "SQG_WAVES_32") + Value(sample, "SQG_WAVES_64"),
                1u);
      EXPECT_EQ(Value(sample, "SQG_ITEMS"), 32u);
      EXPECT_GT(Value(sample, "SQG_BUSY_CYCLES"), 0u);
      // The per-workgroup-processor counters agree with the engine ones.
      EXPECT_EQ(Value(sample, "SQ_WAVES"), 1u);
      EXPECT_GT(Value(sample, "SQ_INSTS_ALL"), 0u);
      EXPECT_EQ(
          Value(sample, "SQ_INSTS_ALL") >=
              Value(sample, "SQ_INSTS_VALU") + Value(sample, "SQ_INSTS_SALU"),
          true);
      const auto event = std::find_if(
          sink_.dispatch_events.begin(), sink_.dispatch_events.end(),
          [&](const auto& candidate) {
            return candidate.event_id == sample.record.dispatch_event_id;
          });
      ASSERT_NE(event, sink_.dispatch_events.end());
      EXPECT_EQ(sample.record.start_tick, event->start_tick);
      EXPECT_EQ(sample.record.end_tick, event->end_tick);
    }
    ASSERT_FALSE(sink_.counter_captures.empty());
    const auto& capture = sink_.counter_captures.back();
    EXPECT_TRUE(capture.flags &
                IREE_HAL_PROFILE_COUNTER_CAPTURE_FLAG_EXCLUSIVE_WINDOW);
    EXPECT_TRUE(capture.flags &
                IREE_HAL_PROFILE_COUNTER_CAPTURE_FLAG_PROCESS_SCOPE);
    // Sequences that never ran on this exact target say so.
    EXPECT_EQ((capture.flags &
               IREE_HAL_PROFILE_COUNTER_CAPTURE_FLAG_UNVERIFIED_TARGET) != 0,
              !CounterTableVerified());
    EXPECT_EQ(capture.sample_limit, 3u);
    EXPECT_EQ(capture.sampled_dispatch_count, sample_count);
    EXPECT_EQ(capture.invalid_sample_count, 0u);
    EXPECT_EQ(capture.dropped_sample_count, 0u);
    // Nobody else was collecting, and each stop restored the defaults the
    // next start then found.
    EXPECT_EQ(capture.foreign_state_count, 0u);
    EXPECT_GT(capture.instance_count, 0u);
  }

  // Requested names: both named sets.
  const iree_string_view_t names_[2] = {IREE_SVL("sqg"),
                                        IREE_SVL("instructions")};
  const iree_hal_profile_counter_set_selection_t selection_ = {
      .counter_name_count = 2,
      .counter_names = names_,
  };
};

TEST_F(GpuNativeCounterProfileTest, DirectDispatchesAreSampledUpToTheBound) {
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(4, 4),
      iree_infinite_timeout(), output.out()));
  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(Load(executable, function));
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  auto* timeline = done.get();
  uint64_t previous = 0, next = 1;
  uint32_t zero = 0, increment = 1, observed = 0;
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {1, &timeline, &next},
                                     output, 0, 4, &zero, 4, nullptr, 0));
  auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  auto reference = iree_hal_make_buffer_ref(output, 0, 4);
  // Four dispatches against a bound of three.
  for (size_t i = 0; i < 4; ++i) {
    previous = next++;
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        queues_[0], {1, &timeline, &previous}, {1, &timeline, &next},
        executable, function, config,
        iree_make_const_byte_span(&increment, sizeof(increment)),
        {1, &reference}, nullptr, 0));
  }
  previous = next++;
  IREE_ASSERT_OK(iree_hal_queue_download(queues_[0], {1, &timeline, &previous},
                                         {1, &timeline, &next}, output, 0,
                                         &observed, sizeof(observed), nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(done, next));
  // Sampling changed nothing the kernels computed.
  EXPECT_EQ(observed, 4u);
  executable.reset();
  ASSERT_NO_FATAL_FAILURE(End());
  ASSERT_EQ(sink_.dispatch_events.size(), 4u);
  ASSERT_NO_FATAL_FAILURE(ExpectOneWorkgroupPerSample(3));
  EXPECT_TRUE(sink_.counter_captures.back().flags &
              IREE_HAL_PROFILE_COUNTER_CAPTURE_FLAG_LIMIT_REACHED);
  EXPECT_EQ(sink_.counter_captures.back().sampled_operation_count, 3u);
  for (const auto& sample : sink_.counter_samples) {
    EXPECT_FALSE(sample.record.flags &
                 IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_COMMAND_OPERATION);
  }
}

TEST_F(GpuNativeCounterProfileTest, RecordedDispatchesAreSampledOneByOne) {
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(24, 8),
      iree_infinite_timeout(), output.out()));
  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(Load(executable, function));
  // Recorded while the session samples counters, so every dispatch carries
  // its own sample sequences.
  Ref<iree_hal_command_buffer_t> recording;
  ASSERT_NO_FATAL_FAILURE(
      Record(executable, function,
             IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA, 3,
             /*timestamps=*/true, 0, recording.out()));
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  auto* timeline = done.get();
  uint64_t value = 1;
  uint32_t zero = 0;
  std::array<uint64_t, 3> observed = {};
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {1, &timeline, &value},
                                     output, 0, 24, &zero, 4, nullptr, 0));
  iree_hal_buffer_binding_t binding = {.buffer = output, .length = 24};
  // Two executions: the first uses the whole bound, the second runs with its
  // sample sequences disabled and must compute the same thing.
  for (size_t i = 0; i < 2; ++i) {
    uint64_t wait = value++;
    IREE_ASSERT_OK(iree_hal_queue_execute(queues_[0], {1, &timeline, &wait},
                                          {1, &timeline, &value}, recording,
                                          {1, &binding}, 0));
  }
  uint64_t wait = value++;
  IREE_ASSERT_OK(iree_hal_queue_download(
      queues_[0], {1, &timeline, &wait}, {1, &timeline, &value}, output, 0,
      observed.data(), sizeof(observed), nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(done, value));
  // Three dispatches adding 1, 2 and 3, twice.
  EXPECT_EQ(observed[0] & 0xffffffffu, 12u);
  recording.reset();
  executable.reset();
  sink_.expected_dispatch_flags =
      IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER;
  sink_.expected_dispatch_command_indices = {2, 4, 6, 2, 4, 6};
  ASSERT_NO_FATAL_FAILURE(End());
  ASSERT_EQ(sink_.dispatch_events.size(), 6u);
  ASSERT_NO_FATAL_FAILURE(ExpectOneWorkgroupPerSample(3));
  EXPECT_EQ(sink_.counter_captures.back().sampled_operation_count, 1u);
  EXPECT_EQ(sink_.counter_captures.back().skipped_operation_count, 0u);
  const std::array<uint32_t, 3> indices = {2, 4, 6};
  for (size_t i = 0; i < 3; ++i) {
    const auto& record = sink_.counter_samples[i].record;
    EXPECT_TRUE(record.flags &
                IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_COMMAND_OPERATION);
    EXPECT_EQ(record.command_index, indices[i]);
    EXPECT_EQ(record.command_buffer_id,
              sink_.dispatch_events[0].command_buffer_id);
    EXPECT_EQ(record.dispatch_event_id, sink_.dispatch_events[i].event_id);
  }
}

// The same session on a ring too small for a direct dispatch's counter
// sequences.
class GpuNativeCounterSmallRingProfileTest
    : public GpuNativeCounterProfileTest {
 protected:
  uint32_t RingByteLength() const override { return 4096; }
};

TEST_F(GpuNativeCounterSmallRingProfileTest,
       DirectDispatchesThatCannotFitRunUnsampled) {
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(4, 4),
      iree_infinite_timeout(), output.out()));
  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(Load(executable, function));
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  auto* timeline = done.get();
  uint64_t previous = 0, next = 1;
  uint32_t zero = 0, increment = 1, observed = 0;
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {1, &timeline, &next},
                                     output, 0, 4, &zero, 4, nullptr, 0));
  auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  auto reference = iree_hal_make_buffer_ref(output, 0, 4);
  for (size_t i = 0; i < 2; ++i) {
    previous = next++;
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        queues_[0], {1, &timeline, &previous}, {1, &timeline, &next},
        executable, function, config,
        iree_make_const_byte_span(&increment, sizeof(increment)),
        {1, &reference}, nullptr, 0));
  }
  previous = next++;
  IREE_ASSERT_OK(iree_hal_queue_download(queues_[0], {1, &timeline, &previous},
                                         {1, &timeline, &next}, output, 0,
                                         &observed, sizeof(observed), nullptr));
  // The dispatches complete instead of waiting for ring capacity that a
  // 1,024-DWORD ring can never provide.
  ASSERT_NO_FATAL_FAILURE(Wait(done, next));
  EXPECT_EQ(observed, 2u);
  executable.reset();
  ASSERT_NO_FATAL_FAILURE(End());
  EXPECT_EQ(sink_.dispatch_events.size(), 2u);
  EXPECT_TRUE(sink_.counter_samples.empty());
  // The counters are still described, and the loss is reported.
  EXPECT_EQ(sink_.counters.size(), 9u);
  ASSERT_FALSE(sink_.counter_captures.empty());
  EXPECT_EQ(sink_.counter_captures.back().sampled_dispatch_count, 0u);
  EXPECT_EQ(sink_.counter_captures.back().skipped_operation_count, 2u);
}

INSTANTIATE_TEST_SUITE_P(
    Protocol, GpuNativeDispatchProfileTest,
    ::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                      AMDF_QUEUE_COMMAND_TYPE_GPU_AQL),
    [](const ::testing::TestParamInfo<amdf_queue_command_type_t>& info) {
      return info.param == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL ? "Aql" : "Pm4";
    });

INSTANTIATE_TEST_SUITE_P(
    Protocol, GpuNativeProfileTest,
    ::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                      AMDF_QUEUE_COMMAND_TYPE_GPU_AQL),
    [](const ::testing::TestParamInfo<amdf_queue_command_type_t>& info) {
      return info.param == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL ? "Aql" : "Pm4";
    });

}  // namespace
