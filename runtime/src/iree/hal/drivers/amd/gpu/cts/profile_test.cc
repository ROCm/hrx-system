// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <thread>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/profile_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/image/hip_images.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {
using iree::hal::amd::gpu::cts::GpuNativeTest;
using iree::hal::cts::Ref;

class GpuNativeProfileTest : public GpuNativeTest {
 protected:
  void SetUp() override {
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    gfx_major_ = context_->target_info.gfx_ip.major;
    options_.pm4.retirement_capacity = 4;
    options_.pm4.argument_capacity = 32;
    Create();
    if (HasFatalFailure()) {
      return;
    }
    iree::hal::cts::TestProfileSinkInitialize(&sink_);
    const iree_hal_pool_family_access_t access = {
        .family = iree_hal_queue_family(queues_[0]),
        .usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        .interfaces =
            (UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS) |
            (UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE),
    };
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, {1, &access, {}}, &options,
                                             iree_allocator_system(),
                                             pool_.out()));
    const iree_hal_device_profiling_options_t profile_options = {
        .data_families = IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
        .sink = iree::hal::cts::TestProfileSinkAsBase(&sink_),
    };
    IREE_ASSERT_OK(iree_hal_device_profiling_begin(device_, &profile_options));
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

  void End() {
    IREE_ASSERT_OK(iree_hal_device_profiling_end(device_));
    EXPECT_EQ(sink_.begin_count, 1);
    EXPECT_EQ(sink_.end_count, 1);
    EXPECT_EQ(sink_.queue_metadata_count, 2);
    EXPECT_FALSE(sink_.write_after_end);
    for (const auto& event : sink_.queue_events) {
      EXPECT_EQ(event.record_length, sizeof(event));
      EXPECT_EQ(event.physical_device_ordinal, 0u);
      EXPECT_LT(event.queue_ordinal, 2u);
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

  void Load(Ref<iree_hal_executable_t>& executable,
            iree_hal_executable_function_t& function) {
    const auto* files = iree_hal_amd_gpu_hip_images_create();
    iree_hal_executable_load_params_t params;
    iree_hal_executable_load_params_initialize(&params);
    for (size_t i = 0; i < iree_hal_amd_gpu_hip_images_size(); ++i) {
      if (std::strstr(files[i].name,
                      gfx_major_ == 11 ? "gfx11_generic" : "gfx12_generic")) {
        params.executable_data =
            iree_make_const_byte_span(files[i].data, files[i].size);
      }
    }
    ASSERT_NE(params.executable_data.data, nullptr);
    const auto* targets =
        iree_hal_device_spec_executables(iree_hal_device_spec(device_));
    IREE_ASSERT_OK(iree_hal_executable_load(iree_hal_queue_family(queues_[0]),
                                            &targets->targets[0], &params,
                                            executable.out()));
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executable, IREE_SV("image_accumulate"), &function));
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

  // Native image selector captured before ownership transfers into the device.
  uint32_t gfx_major_ = 0;
  // Explicit storage contract for dispatches, transfers, and mapping sources.
  Ref<iree_hal_pool_t> pool_;
  // Session sink remains live through idle end and device teardown.
  iree::hal::cts::TestProfileSink sink_;
};

TEST_F(GpuNativeProfileTest, ConsumerBeforeProducerRecordsObservedReadiness) {
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

TEST_F(GpuNativeProfileTest, FailedDependencyRecordsNoReadiness) {
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

TEST_F(GpuNativeProfileTest,
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

TEST_F(GpuNativeProfileTest,
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
  IREE_ASSERT_OK(iree_hal_queue_read(queues_[1], point(0), point(1),
                                     source_file, 0, buffer, 0, sizeof(source),
                                     nullptr, 0));
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
  IREE_ASSERT_OK(iree_hal_queue_write(queues_[1], point(2), point(3), buffer, 0,
                                      target_file, 0, sizeof(target), nullptr,
                                      0));
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
  EXPECT_EQ(Events(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL)[0].signal_count,
            1u);
}

TEST_F(GpuNativeProfileTest, MappingBatchIsOnePublicEvent) {
  const iree_hal_pool_family_access_t access = {
      .family = iree_hal_queue_family(queues_[0]),
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
      .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
  };
  const iree_hal_pool_scope_t scope = {1, &access, {}};
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, scope, iree_allocator_system(), &capabilities));
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
      group_, scope, &options, iree_allocator_system(), reservation.out()));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(2 * unit, unit),
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

TEST_F(GpuNativeProfileTest, ConcurrentSubmittersPreserveEveryPublicIdentity) {
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

}  // namespace
