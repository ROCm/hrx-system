// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <vector>

#include "iree/hal/cts/util/profile_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/task/registration/driver_module.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/maintenance.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf_pool.h"

namespace iree::hal::cts {
namespace {

class HeterogeneousSlabPoolTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    iree_hal_executable_target_selection_result_t selection;
    IREE_ASSERT_OK(SelectExecutableTarget(
        iree_hal_device_queue_family(device_, 0), &selection));
    if (selection.outcome ==
        IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
      GTEST_SKIP() << "native executable target is unavailable";
    }
    ASSERT_EQ(selection.outcome,
              IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);

    IREE_ASSERT_OK(
        iree_hal_driver_registry_allocate(iree_allocator_system(), &registry_));
    IREE_ASSERT_OK(iree_hal_task_driver_module_register(registry_));
    IREE_ASSERT_OK(iree_hal_driver_registry_try_create(
        registry_, IREE_SV("task"), iree_allocator_system(), &host_driver_));
    const std::array<iree_hal_driver_t*, 2> drivers = {host_driver_, driver_};
    for (size_t i = 0; i < devices_.size(); ++i) {
      IREE_ASSERT_OK(contexts_[i].Initialize(iree_allocator_system()));
      IREE_ASSERT_OK(iree_hal_driver_create_default_device(
          drivers[i], contexts_[i].params(), iree_allocator_system(),
          &devices_[i]));
    }
    iree_hal_device_group_builder_t builder;
    iree_hal_device_group_builder_initialize(&builder,
                                             contexts_[0].frontier_tracker());
    iree_status_t status = iree_ok_status();
    for (auto* device : devices_) {
      if (iree_status_is_ok(status)) {
        status = iree_hal_device_group_builder_add_device(&builder, device);
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_builder_finalize(
          &builder, iree_allocator_system(), &group_);
    }
    iree_hal_device_group_builder_deinitialize(&builder);
    IREE_ASSERT_OK(status);
    for (size_t i = 0; i < devices_.size(); ++i) {
      queues_[i] = iree_hal_device_queue(devices_[i], 0, 0);
      ASSERT_NE(queues_[i], nullptr);
    }
    IREE_ASSERT_OK(SelectBackendExecutableTarget(
        devices_[1], iree_hal_queue_family(queues_[1]), GetParam(),
        &selection));
    ASSERT_EQ(selection.outcome,
              IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
    IREE_ASSERT_OK(LoadExecutable(
        iree_hal_queue_family(queues_[1]), selection.target,
        IREE_HAL_EXECUTABLE_LOAD_FLAG_NONE,
        executable_data(
            IREE_SV("command_buffer_dispatch_constants_bindings_test.bin")),
        executable_.out()));
    for (size_t i = 0; i < families_.size(); ++i) {
      families_[i].family = iree_hal_queue_family(queues_[i]);
      families_[i].usage =
          IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE;
    }
  }

  void TearDown() override {
    executable_.reset();
    for (auto* device : devices_) {
      iree_hal_device_release(device);
    }
    iree_hal_device_group_release(group_);
    for (auto& context : contexts_) {
      context.Deinitialize();
    }
    iree_hal_driver_release(host_driver_);
    iree_hal_driver_registry_free(registry_);
    CtsTestBase::TearDown();
  }

  // Joins queued child/cache returns, including work they enqueue. A pending
  // frontier callback can still enqueue native source retirement afterward.
  void JoinMaintenance(iree_hal_pool_t* source) {
    iree_hal_memory_maintenance_call(
        source->maintenance,
        [](void* user_data) {
          auto* owner = static_cast<iree_hal_memory_maintenance_t*>(user_data);
          while (iree_hal_memory_maintenance_run_one(owner)) {
          }
        },
        source->maintenance);
  }

  iree_status_t CreateSource(iree_hal_pool_t** out_source) {
    const iree_hal_pool_scope_t scope = {
        families_.size(), families_.data(), {}};
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    return iree_hal_slab_pool_create(group_, scope, &options,
                                     iree_allocator_system(), out_source);
  }

  iree_hal_pool_reservation_request_t MakeRequest(iree_hal_pool_t* pool) {
    iree_hal_pool_capabilities_t capabilities;
    iree_hal_pool_query_capabilities(pool, &capabilities);
    iree_hal_pool_reservation_request_t request = {};
    request.params.min_alignment =
        iree_min(4096, capabilities.max_allocation_alignment);
    request.allocation_size = 256;
    return request;
  }

  // Private driver registry for the ordinary Task creation path.
  iree_hal_driver_registry_t* registry_ = nullptr;
  // Owned Task driver, independent of the fixture's native GPU driver.
  iree_hal_driver_t* host_driver_ = nullptr;
  // Existing progress services, retained until all grouped devices retire.
  std::array<DeviceCreateContext, 2> contexts_;
  // Owned Task and GPU devices, in that order.
  std::array<iree_hal_device_t*, 2> devices_ = {};
  // Borrowed provisioned queues for the two devices.
  std::array<iree_hal_queue_t*, 2> queues_ = {};
  // Owned sealed group; no device is introduced after construction.
  iree_hal_device_group_t* group_ = nullptr;
  // Actual GPU kernel consuming and producing interior shared ranges.
  Ref<iree_hal_executable_t> executable_;
  // Simultaneous CPU/GPU execution without public mapping grants.
  std::array<iree_hal_pool_family_access_t, 2> families_ = {};
};

TEST_P(HeterogeneousSlabPoolTest, BothQueuesPublishSharedNativeStorage) {
  const iree_hal_pool_scope_t scope = {families_.size(), families_.data(), {}};
  iree_hal_slab_pool_options_t source_options;
  iree_hal_slab_pool_options_initialize(&source_options);
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group_, scope, &source_options, iree_allocator_system(), source.out()));
  EXPECT_EQ(iree_hal_pool_query_host_access(source).access,
            IREE_HAL_MEMORY_ACCESS_NONE);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.bytes_committed, 0u);
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(source, &capabilities);
  iree_hal_slab_cache_options_t cache_options;
  iree_hal_slab_cache_options_initialize(&cache_options);
  cache_options.slab.allocation_size = 65536;
  cache_options.slab.params.min_alignment =
      iree_min(4096, capabilities.max_allocation_alignment);
  Ref<iree_hal_pool_t> cache;
  IREE_ASSERT_OK(iree_hal_slab_cache_create(
      source, &cache_options, iree_allocator_system(), cache.out()));
  std::array<Ref<iree_hal_pool_t>, 2> children;
  iree_hal_tlsf_pool_options_t tlsf_options = {};
  tlsf_options.tlsf_options.range_length = 4096;
  tlsf_options.tlsf_options.frontier_capacity = 4;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create(
      cache, &tlsf_options, iree_allocator_system(), children[0].out()));
  iree_hal_fixed_block_pool_options_t block_options = {};
  block_options.block_size = 256;
  block_options.blocks_per_slab = 16;
  block_options.frontier_capacity = 4;
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create(
      cache, &block_options, iree_allocator_system(), children[1].out()));

  const iree_hal_buffer_backing_facts_t* first_backing = nullptr;
  for (auto& child : children) {
    for (size_t allocating_device = 0; allocating_device < devices_.size();
         ++allocating_device) {
      SCOPED_TRACE(allocating_device);
      SemaphoreList gate(devices_[0], {0}, {1});
      SemaphoreList allocated(devices_[allocating_device], {0}, {1});
      iree_hal_pool_reservation_request_t request = {};
      request.allocation_size = 96;
      Ref<iree_hal_buffer_t> buffer;
      IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[allocating_device], gate,
                                           allocated, child, 1, &request,
                                           buffer.out()));
      Ref<iree_hal_buffer_t> subspan;
      IREE_ASSERT_OK(iree_hal_buffer_subspan(
          buffer, 16, 64, iree_allocator_system(), subspan.out()));
      const auto* native_table = subspan.get()->memory.bindings;
      std::array<uint32_t, 24> input;
      for (size_t i = 0; i < input.size(); ++i) {
        input[i] = 100 + i;
      }
      SemaphoreList uploaded(devices_[0], {0}, {1});
      SemaphoreList transferred(devices_[1], {0}, {1});
      const uint32_t fill_pattern = 0xAABBCCDD;
      const uint32_t update_values[] = {0x11223344, 0x55667788};
      const uint32_t upload_values[] = {0x10203040, 0x50607080};
      std::array<uint32_t, 2> downloaded_values = {};
      iree_hal_transfer_operation_t operations[5] = {};
      operations[0].type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
      operations[0].fill = {buffer,
                            0,
                            8,
                            &fill_pattern,
                            sizeof(fill_pattern),
                            IREE_HAL_FILL_FLAG_NONE};
      operations[1].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE;
      operations[1].update = {update_values,
                              0,
                              buffer,
                              8,
                              sizeof(update_values),
                              IREE_HAL_UPDATE_FLAG_NONE};
      operations[2].type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY;
      operations[2].copy = {subspan, 0, subspan,
                            56,      8, IREE_HAL_COPY_FLAG_NONE};
      operations[3].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
      operations[3].upload = {upload_values, buffer, 88, sizeof(upload_values)};
      operations[4].type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
      operations[4].download = {subspan, 24, downloaded_values.data(),
                                sizeof(downloaded_values)};

      // Capture every transfer form while the allocation is still blocked.
      // The unordered siblings touch disjoint write ranges. On capture failure,
      // unblock and join the last accepted operation before reporting it.
      iree_hal_semaphore_list_t last_submitted = allocated;
      iree_status_t status =
          iree_hal_queue_upload(queues_[0], allocated, uploaded, input.data(),
                                buffer, 0, sizeof(input));
      if (iree_status_is_ok(status)) {
        last_submitted = uploaded;
        status =
            iree_hal_queue_transfer(queues_[1], uploaded, transferred,
                                    IREE_ARRAYSIZE(operations), operations);
        if (iree_status_is_ok(status)) {
          last_submitted = transferred;
        }
      }
      EXPECT_FALSE(iree_hal_semaphore_list_poll(transferred));
      status = iree_status_join(status,
                                iree_hal_semaphore_list_signal(gate, nullptr));
      status = iree_status_join(
          status,
          iree_hal_semaphore_list_wait(last_submitted, iree_infinite_timeout(),
                                       IREE_ASYNC_WAIT_FLAG_NONE));
      IREE_ASSERT_OK(status);
      EXPECT_EQ(downloaded_values,
                (std::array<uint32_t, 2>{input[10], input[11]}));
      const iree_hal_buffer_ref_t refs[] = {
          iree_hal_make_buffer_ref(subspan, 8, 16),
          iree_hal_make_buffer_ref(subspan, 40, 16),
      };
      const uint32_t constants[] = {3, 7};
      SemaphoreList executed(devices_[1], {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          queues_[1], transferred, executed, executable_,
          iree_hal_executable_function_from_index(0),
          iree_hal_make_static_dispatch_config(1, 1, 1),
          iree_make_const_byte_span(constants, sizeof(constants)),
          {IREE_ARRAYSIZE(refs), refs}, IREE_HAL_DISPATCH_FLAG_NONE));
      // Native atomic validation reads the producer's trailing capability
      // cells, which must survive a Task-owned transient just like addresses.
      iree_hal_atomic_store_params_t store = {};
      store.width = IREE_HAL_ATOMIC_WIDTH_32;
      store.value = 0x1234ABCD;
      SemaphoreList stored(devices_[1], {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_atomic_store(queues_[1], executed, stored,
                                                 buffer, 84, store));
      std::array<uint32_t, 24> output = {};
      SemaphoreList downloaded(devices_[0], {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_download(queues_[0], stored, downloaded,
                                             buffer, 0, output.data(),
                                             sizeof(output)));
      IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
          downloaded, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
      auto expected = input;
      expected[0] = expected[1] = fill_pattern;
      expected[2] = update_values[0];
      expected[3] = update_values[1];
      for (size_t i = 0; i < 4; ++i) {
        expected[14 + i] = input[6 + i] * 3 + 7;
      }
      expected[18] = input[4];
      expected[19] = input[5];
      expected[21] = store.value;
      expected[22] = upload_values[0];
      expected[23] = upload_values[1];
      EXPECT_EQ(output, expected);
      EXPECT_EQ(subspan.get()->memory.bindings, native_table);
      const auto memory = iree_hal_buffer_memory_view(subspan);
      ASSERT_NE(memory.backing, nullptr);
      if (!first_backing) {
        first_backing = memory.backing;
      }
      EXPECT_EQ(memory.backing, first_backing);
      iree_hal_buffer_mapping_t mapping = {};
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_PERMISSION_DENIED,
          iree_hal_buffer_map_range(subspan, IREE_HAL_MAPPING_MODE_SCOPED,
                                    IREE_HAL_MEMORY_ACCESS_READ,
                                    IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 64,
                                    &mapping));
      subspan.reset();
      const size_t freeing_device = 1 - allocating_device;
      SemaphoreList deallocated(devices_[freeing_device], {0}, {1});
      auto* raw_buffer = buffer.get();
      IREE_ASSERT_OK(iree_hal_queue_dealloca(
          queues_[freeing_device], downloaded, deallocated, 1, &raw_buffer));
      IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
          deallocated, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
      buffer.reset();
      JoinMaintenance(source);
      iree_hal_pool_query_stats(child, &stats);
      EXPECT_EQ(stats.bytes_committed, 0u);
    }
  }
  iree_hal_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.reserve_count, 1u);
  iree_hal_slab_cache_stats_t cache_stats;
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache, &cache_stats));
  EXPECT_EQ(cache_stats.hit_count, 3u);
  EXPECT_EQ(cache_stats.ready_count, 1u);
  iree_hal_pool_trim(cache, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  JoinMaintenance(source);
  iree_hal_pool_query_stats(cache, &stats);
  EXPECT_EQ(stats.bytes_committed, 0u);
  iree_hal_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.bytes_reserved, 0u);
}

TEST_P(HeterogeneousSlabPoolTest, MixedBatchDeallocaBeforeCommit) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(source.out()));
  const auto request = MakeRequest(source);
  for (size_t freeing_device = 0; freeing_device < devices_.size();
       ++freeing_device) {
    SCOPED_TRACE(freeing_device);
    SemaphoreList gate(devices_[0], {0}, {1});
    SemaphoreList allocated(devices_[0], {0, 0}, {1, 1});
    SemaphoreList deallocated(devices_[freeing_device], {0}, {1});
    std::array<Ref<iree_hal_buffer_t>, 2> buffers;
    std::array<iree_hal_buffer_t*, 2> roots = {};
    iree_status_t status = iree_ok_status();
    size_t submitted_count = 0;
    while (submitted_count < buffers.size() && iree_status_is_ok(status)) {
      const iree_hal_semaphore_list_t signal = {
          1, &allocated.semaphores[submitted_count],
          &allocated.payload_values[submitted_count]};
      status =
          iree_hal_queue_alloca(queues_[submitted_count], gate, signal, source,
                                1, &request, buffers[submitted_count].out());
      if (iree_status_is_ok(status)) {
        roots[submitted_count] = buffers[submitted_count].get();
        ++submitted_count;
      }
    }
    bool release_submitted = false;
    if (iree_status_is_ok(status)) {
      status = iree_hal_queue_dealloca(queues_[freeing_device], allocated,
                                       deallocated, roots.size(), roots.data());
      release_submitted = iree_status_is_ok(status);
    }
    EXPECT_FALSE(iree_hal_semaphore_list_poll(deallocated));
    // Open the gate and join accepted work even when submission fails, so a
    // failing assertion cannot strand a device behind an unsignaled dependency.
    status =
        iree_status_join(status, iree_hal_semaphore_list_signal(gate, nullptr));
    const iree_hal_semaphore_list_t accepted = {
        submitted_count, allocated.semaphores.data(),
        allocated.payload_values.data()};
    status = iree_status_join(
        status, iree_hal_semaphore_list_wait(accepted, iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
    if (release_submitted) {
      status = iree_status_join(
          status,
          iree_hal_semaphore_list_wait(deallocated, iree_infinite_timeout(),
                                       IREE_ASYNC_WAIT_FLAG_NONE));
    }
    IREE_ASSERT_OK(status);
    // Completion returns allocation epochs. Native backing retires
    // asynchronously after the caller also releases its views.
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(source, &stats);
    EXPECT_EQ(stats.reservation_count, 0u);
    EXPECT_EQ(stats.bytes_reserved, 0u);
    EXPECT_EQ(stats.release_count, stats.reserve_count);
    for (auto& buffer : buffers) {
      buffer.reset();
    }
  }
}

TEST_P(HeterogeneousSlabPoolTest, RejectedMixedDeallocaPreservesAllEpochs) {
  std::array<Ref<iree_hal_pool_t>, 2> sources;
  for (auto& source : sources) {
    IREE_ASSERT_OK(CreateSource(source.out()));
  }
  for (size_t freeing_device = 0; freeing_device < devices_.size();
       ++freeing_device) {
    SCOPED_TRACE(freeing_device);
    std::array<Ref<iree_hal_buffer_t>, 3> buffers;
    for (size_t i = 0; i < buffers.size(); ++i) {
      auto* source = sources[i / 2].get();
      const auto request = MakeRequest(source);
      SemaphoreList allocated(devices_[i % 2], {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_alloca(
          queues_[i % 2], iree_hal_semaphore_list_empty(), allocated, source, 1,
          &request, buffers[i].out()));
      IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
          allocated, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    }
    SemaphoreList rejected(devices_[freeing_device], {0}, {1});
    // Both wrapper kinds are marked before the final entry rejects the batch.
    iree_hal_buffer_t* duplicate[] = {buffers[0], buffers[1], buffers[0]};
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_FAILED_PRECONDITION,
        iree_hal_queue_dealloca(queues_[freeing_device],
                                iree_hal_semaphore_list_empty(), rejected,
                                IREE_ARRAYSIZE(duplicate), duplicate));
    iree_hal_buffer_t* mixed_pools[] = {buffers[0], buffers[1], buffers[2]};
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_queue_dealloca(queues_[freeing_device],
                                iree_hal_semaphore_list_empty(), rejected,
                                IREE_ARRAYSIZE(mixed_pools), mixed_pools));
    EXPECT_FALSE(iree_hal_semaphore_list_poll(rejected));
    for (size_t i = 0; i < buffers.size(); ++i) {
      const uint32_t pattern = 0xAA001100 + i;
      SemaphoreList filled(devices_[1], {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_fill(
          queues_[1], iree_hal_semaphore_list_empty(), filled, buffers[i], 0,
          256, &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
      std::array<uint32_t, 64> output = {};
      SemaphoreList downloaded(devices_[0], {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_download(queues_[0], filled, downloaded,
                                             buffers[i], 0, output.data(),
                                             sizeof(output)));
      IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
          downloaded, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
      for (auto value : output) {
        EXPECT_EQ(value, pattern);
      }
    }
    iree_hal_buffer_t* same_pool[] = {buffers[0], buffers[1]};
    SemaphoreList released(devices_[freeing_device], {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_dealloca(
        queues_[freeing_device], iree_hal_semaphore_list_empty(), released,
        IREE_ARRAYSIZE(same_pool), same_pool));
    auto* other_pool = buffers[2].get();
    SemaphoreList all_released(devices_[freeing_device], {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[freeing_device], released,
                                           all_released, 1, &other_pool));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        all_released, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    for (auto& source : sources) {
      iree_hal_pool_stats_t stats;
      iree_hal_pool_query_stats(source, &stats);
      EXPECT_EQ(stats.reservation_count, 0u);
      EXPECT_EQ(stats.bytes_reserved, 0u);
      EXPECT_EQ(stats.release_count, stats.reserve_count);
    }
    for (auto& buffer : buffers) {
      buffer.reset();
    }
  }
}

TEST_P(HeterogeneousSlabPoolTest, AllocationIdentitySpansDeviceCaptures) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(source.out()));
  const std::array<iree_hal_pool_reservation_request_t, 2> requests = {
      MakeRequest(source), MakeRequest(source)};
  const auto data_families = IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
                             IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS;
  // Device-local capture numbers are deliberately different.
  TestProfileSink prior_sink;
  TestProfileSinkInitialize(&prior_sink);
  DeviceProfilingScope gpu_profiling(devices_[1]);
  IREE_ASSERT_OK(
      gpu_profiling.Begin(data_families, TestProfileSinkAsBase(&prior_sink)));
  IREE_ASSERT_OK(gpu_profiling.End());
  for (size_t buffer_count : {1, 2}) {
    SCOPED_TRACE(buffer_count);
    std::array<TestProfileSink, 2> sinks;
    for (auto& sink : sinks) {
      TestProfileSinkInitialize(&sink);
    }
    DeviceProfilingScope cpu_profiling(devices_[0]);
    DeviceProfilingScope gpu_capture(devices_[1]);
    IREE_ASSERT_OK(
        cpu_profiling.Begin(data_families, TestProfileSinkAsBase(&sinks[0])));
    IREE_ASSERT_OK(
        gpu_capture.Begin(data_families, TestProfileSinkAsBase(&sinks[1])));
    std::array<std::array<uint64_t, 2>, 2> identities = {};
    for (size_t allocating_device = 0; allocating_device < devices_.size();
         ++allocating_device) {
      const size_t freeing_device = 1 - allocating_device;
      std::array<iree_hal_buffer_t*, 2> roots = {};
      SemaphoreList allocated(devices_[allocating_device], {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_alloca(
          queues_[allocating_device], iree_hal_semaphore_list_empty(),
          allocated, source, buffer_count, requests.data(), roots.data()));
      std::array<Ref<iree_hal_buffer_t>, 2> buffers;
      for (size_t i = 0; i < buffer_count; ++i) {
        buffers[i].reset(roots[i]);
        identities[allocating_device][i] =
            iree_hal_buffer_allocation_profile(buffers[i]).id;
        EXPECT_NE(identities[allocating_device][i], 0u);
      }
      SemaphoreList released(devices_[freeing_device], {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[freeing_device], allocated,
                                             released, buffer_count,
                                             roots.data()));
      IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
          released, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    }
    IREE_ASSERT_OK(cpu_profiling.End());
    IREE_ASSERT_OK(gpu_capture.End());
    EXPECT_NE(identities[0][0], identities[1][0]);
    if (buffer_count == 2) {
      EXPECT_NE(identities[0][0], identities[0][1]);
      EXPECT_NE(identities[1][0], identities[1][1]);
    }
    EXPECT_NE(sinks[0].session_id, sinks[1].session_id);
    for (size_t device = 0; device < devices_.size(); ++device) {
      SCOPED_TRACE(device);
      size_t allocation_count = 0;
      size_t release_count = 0;
      for (const auto& event : sinks[device].queue_events) {
        if (event.type == IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA) {
          EXPECT_EQ(event.allocation_id,
                    buffer_count == 1 ? identities[device][0] : 0);
          EXPECT_EQ(event.operation_count, buffer_count);
          ++allocation_count;
        } else if (event.type == IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA) {
          EXPECT_EQ(event.allocation_id,
                    buffer_count == 1 ? identities[1 - device][0] : 0);
          EXPECT_EQ(event.operation_count, buffer_count);
          ++release_count;
        }
      }
      EXPECT_EQ(allocation_count, 1u);
      EXPECT_EQ(release_count, 1u);
      std::vector<uint64_t> allocation_ids;
      std::vector<uint64_t> release_ids;
      for (const auto& event : sinks[device].memory_events) {
        if (event.type == IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA) {
          allocation_ids.push_back(event.allocation_id);
          EXPECT_EQ(event.length, requests[0].allocation_size);
        } else if (event.type ==
                   IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA) {
          release_ids.push_back(event.allocation_id);
          EXPECT_EQ(event.length, requests[0].allocation_size);
        }
      }
      EXPECT_EQ(allocation_ids, std::vector<uint64_t>(
                                    identities[device].begin(),
                                    identities[device].begin() + buffer_count));
      EXPECT_EQ(
          release_ids,
          std::vector<uint64_t>(identities[1 - device].begin(),
                                identities[1 - device].begin() + buffer_count));
      EXPECT_FALSE(sinks[device].write_after_end);
    }
  }
}

TEST_P(HeterogeneousSlabPoolTest, UnprofiledAllocationRetainsItsIdentity) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(source.out()));
  const auto request = MakeRequest(source);
  for (size_t allocating_device = 0; allocating_device < devices_.size();
       ++allocating_device) {
    SCOPED_TRACE(allocating_device);
    const size_t freeing_device = 1 - allocating_device;
    Ref<iree_hal_buffer_t> buffer;
    SemaphoreList allocated(devices_[allocating_device], {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_alloca(
        queues_[allocating_device], iree_hal_semaphore_list_empty(), allocated,
        source, 1, &request, buffer.out()));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        allocated, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    const uint64_t identity = iree_hal_buffer_allocation_profile(buffer).id;
    EXPECT_NE(identity, 0u);
    TestProfileSink sink;
    TestProfileSinkInitialize(&sink);
    DeviceProfilingScope profiling(devices_[freeing_device]);
    IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS,
                                   TestProfileSinkAsBase(&sink)));
    auto* root = buffer.get();
    SemaphoreList released(devices_[freeing_device], {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[freeing_device], allocated,
                                           released, 1, &root));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        released, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    IREE_ASSERT_OK(profiling.End());
    size_t release_count = 0;
    for (const auto& event : sink.memory_events) {
      if (event.type == IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA) {
        EXPECT_EQ(event.allocation_id, identity);
        ++release_count;
      }
    }
    EXPECT_EQ(release_count, 1u);
  }
}

TEST_P(HeterogeneousSlabPoolTest, DispatchRequiresItsOwnFamilyGrants) {
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  const iree_hal_pool_scope_t allowed_scope = {
      families_.size(), families_.data(), {}};
  Ref<iree_hal_pool_t> allowed_source;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, allowed_scope, &options,
                                           iree_allocator_system(),
                                           allowed_source.out()));
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(allowed_source, &capabilities);
  iree_hal_buffer_params_t params = {};
  params.min_alignment = iree_min(4096, capabilities.max_allocation_alignment);
  Ref<iree_hal_buffer_t> allowed_buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(allowed_source, params, 32,
                                               iree_infinite_timeout(),
                                               allowed_buffer.out()));

  // A CPU grant never authorizes GPU execution, whether the GPU is excluded
  // entirely or admitted with a narrower usage mask.
  auto restricted_families = families_;
  restricted_families[0].usage |=
      IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS;
  restricted_families[1].usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  for (iree_host_size_t family_count : {1, 2}) {
    SCOPED_TRACE(family_count);
    const iree_hal_pool_scope_t restricted_scope = {
        family_count, restricted_families.data(), {}};
    Ref<iree_hal_pool_t> restricted_source;
    IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, restricted_scope, &options,
                                             iree_allocator_system(),
                                             restricted_source.out()));
    iree_hal_pool_query_capabilities(restricted_source, &capabilities);
    params.min_alignment =
        iree_min(4096, capabilities.max_allocation_alignment);
    Ref<iree_hal_buffer_t> restricted_buffer;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(restricted_source, params, 32,
                                                 iree_infinite_timeout(),
                                                 restricted_buffer.out()));
    EXPECT_TRUE(iree_all_bits_set(
        iree_hal_buffer_allowed_usage(restricted_buffer),
        IREE_HAL_BUFFER_USAGE_STORAGE |
            IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS));
    const uint32_t constants[] = {3, 7};
    for (auto flags : {IREE_HAL_DISPATCH_FLAG_NONE,
                       IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS,
                       IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS}) {
      SCOPED_TRACE(flags);
      const bool indirect = iree_hal_dispatch_uses_indirect_parameters(flags);
      auto* binding_buffer =
          indirect ? allowed_buffer.get() : restricted_buffer.get();
      const iree_hal_buffer_ref_t refs[] = {
          iree_hal_make_buffer_ref(binding_buffer, 0, 16),
          iree_hal_make_buffer_ref(binding_buffer, 16, 16),
      };
      auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
      if (indirect) {
        config.workgroup_count_ref =
            iree_hal_make_buffer_ref(restricted_buffer, 0, sizeof(uint32_t[3]));
      }
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_PERMISSION_DENIED,
          iree_hal_queue_dispatch(
              queues_[1], iree_hal_semaphore_list_empty(),
              iree_hal_semaphore_list_empty(), executable_,
              iree_hal_executable_function_from_index(0), config,
              iree_make_const_byte_span(constants, sizeof(constants)),
              {IREE_ARRAYSIZE(refs), refs}, flags));
    }
  }
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(HeterogeneousSlabPoolTest);

}  // namespace
}  // namespace iree::hal::cts
