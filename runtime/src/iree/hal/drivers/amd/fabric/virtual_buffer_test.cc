// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/memory_backend_test_base.h"

namespace iree::hal::amd::fabric {

TEST_F(MemoryBackendTest, RejectsUnavailableNativeNpuReservations) {
  const iree_hal_pool_family_access_t access = {
      .family = families_[0],
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
      .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
  };
  const iree_hal_pool_scope_t scope = {1, &access, {}};
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, scope, allocator_, &capabilities));
  EXPECT_EQ(capabilities.max_byte_length, 0u);
  const iree_hal_virtual_buffer_options_t options = {.byte_length = 4096};
  iree_hal_buffer_t* buffer = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_virtual_buffer_create(group_, scope, &options,
                                                       allocator_, &buffer));
  EXPECT_EQ(buffer, nullptr);
  EXPECT_EQ(native_.reservation_count, 0);
}

TEST_F(MemoryBackendTest, QueriesReservationGeometryBeforeAcquiringAddresses) {
  native_.mapping_enabled = true;
  native_.reservation_granularities = {4096, 65536};
  std::array<iree_hal_pool_family_access_t, 2> accesses = {};
  for (size_t i = 0; i < accesses.size(); ++i) {
    binding_descriptions_[i] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                AMDF_MEMORY_ADDRESS_GPU};
    native_families_[i].host_usage = 0;
    accesses[i].family = families_[i];
    accesses[i].usage = IREE_HAL_BUFFER_USAGE_STORAGE;
  }
  iree_hal_pool_scope_t scope = {accesses.size(), accesses.data(), {}};
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, scope, allocator_, &capabilities));
  EXPECT_EQ(capabilities.reservation_granularity, 65536u);
  EXPECT_EQ(capabilities.mapping_granularity, 65536u);
  EXPECT_EQ(capabilities.max_byte_length, UINT64_C(1) << 40);
  EXPECT_EQ(capabilities.max_alignment, UINT64_C(1) << 30);
  EXPECT_EQ(native_.reservation_count, 0);
  EXPECT_EQ(native_.create_count, 0);

  iree_hal_virtual_buffer_options_t options = {.byte_length = 4096};
  iree_hal_buffer_t* reservation = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_virtual_buffer_create(
                            group_, scope, &options, allocator_, &reservation));
  EXPECT_EQ(reservation, nullptr);
  EXPECT_EQ(native_.reservation_count, 0);
  options.byte_length = capabilities.reservation_granularity;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(group_, scope, &options,
                                                allocator_, &reservation));
  iree_hal_virtual_buffer_info_t info;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_info(reservation, &info));
  EXPECT_EQ(info.mapping_granularity, capabilities.mapping_granularity);
  EXPECT_EQ(native_.reservation_count, 1);
  EXPECT_EQ(native_.create_count, 0);
  iree_hal_buffer_release(reservation);

  // Adding host access makes this device-only native reservation inadmissible.
  // A capability query reports absence without manufacturing an error status.
  scope.host.access = IREE_HAL_MEMORY_ACCESS_READ;
  scope.host.modes = IREE_HAL_MAPPING_MODE_PERSISTENT;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, scope, allocator_, &capabilities));
  EXPECT_EQ(capabilities.max_byte_length, 0u);
  EXPECT_EQ(capabilities.reservation_granularity, 0u);
  EXPECT_EQ(capabilities.mapping_granularity, 0u);
  EXPECT_EQ(capabilities.max_alignment, 0u);
  EXPECT_EQ(native_.reservation_count, 0);
}

class ReservationBackendTest : public MemoryBackendTest,
                               public ::testing::WithParamInterface<bool> {};

TEST_P(ReservationBackendTest, MapsExactConsumersThroughNativeProjections) {
  native_.mapping_enabled = true;
  native_.alias_projections = GetParam();
  native_.native_prefix = 4096;
  for (uint32_t i = 0; i < 2; ++i) {
    binding_descriptions_[i] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                AMDF_MEMORY_ADDRESS_GPU};
    native_.address_kinds[i] = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
    native_families_[i].host_usage = 0;
  }
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool,
                 UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE));
  iree_hal_buffer_t* backing = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(32768, 4096),
      iree_infinite_timeout(), &backing));
  iree_hal_buffer_t* view = nullptr;
  IREE_ASSERT_OK(
      iree_hal_buffer_subspan(backing, 8192, 16384, allocator_, &view));
  std::array<iree_hal_pool_family_access_t, 2> families = {};
  for (uint32_t i = 0; i < 2; ++i) {
    families[i].family = families_[1 - i];
    families[i].usage = IREE_HAL_BUFFER_USAGE_STORAGE;
  }
  const iree_hal_pool_scope_t scope = {families.size(), families.data(), {}};
  const iree_hal_virtual_buffer_options_t options = {
      .byte_length = 16384,
  };
  std::array<iree_hal_buffer_t*, 2> reservations = {};
  for (auto& reservation : reservations) {
    IREE_ASSERT_OK(iree_hal_virtual_buffer_create(group_, scope, &options,
                                                  allocator_, &reservation));
  }
  const auto first = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[0]);
  const auto second = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[1]);
  uint32_t first_projection, second_projection;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_resolve_projection(
      reservations[0], first, &first_projection));
  IREE_ASSERT_OK(iree_hal_virtual_buffer_resolve_projection(
      reservations[0], second, &second_projection));
  EXPECT_EQ(first_projection == second_projection, native_.alias_projections);

  iree_hal_semaphore_t* gate = nullptr;
  iree_hal_semaphore_t* done = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      cpu_device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, 0, &gate));
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      cpu_device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, 0, &done));
  uint64_t value = 1;
  const iree_hal_semaphore_list_t waits = {1, &gate, &value};
  const iree_hal_semaphore_list_t signals = {1, &done, &value};
  std::array<iree_hal_memory_mapping_t, 2> mappings = {};
  for (uint32_t i = 0; i < 2; ++i) {
    mappings[i] = {
        .target = reservations[i],
        .scope = i ? second : first,
        .target_range = {4096, 8192},
        .source = view,
        .source_offset = 4096,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
    };
  }
  auto* queue = iree_hal_device_queue(cpu_device_, 0, 0);
  IREE_ASSERT_OK(iree_hal_queue_map(queue, waits, signals, mappings.size(),
                                    mappings.data(), 0));
  EXPECT_EQ(native_.update_count, 0);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(done, value, iree_infinite_timeout(), 0));
  EXPECT_EQ(native_.update_count, 2);
  ASSERT_EQ(native_.last_updates.size(), 1u);
  // The slab prefix participates in native alignment, but libamdf consumes
  // logical source coordinates. Nested HAL view offsets are included once.
  EXPECT_EQ(native_.last_updates[0].source_byte_offset, 12288u);
  EXPECT_EQ(native_.last_updates[0].target_byte_offset, 4096u);
  EXPECT_EQ(native_.last_updates[0].byte_length, 8192u);

  // ANY expands distinct projections while deduplicating native aliases.
  mappings[0].scope = {first.domain, IREE_HAL_MEMORY_SCOPE_ANY};
  ++value;
  IREE_ASSERT_OK(iree_hal_queue_map(queue, {}, signals, 1, mappings.data(), 0));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(done, value, iree_infinite_timeout(), 0));
  EXPECT_EQ(native_.update_count, native_.alias_projections ? 3 : 4);
  mappings[0].scope = first;

  // Distinct consumers of one page table still produce a single native vector.
  mappings[1].target = reservations[0];
  mappings[0].target_range = {0, 4096};
  mappings[1].target_range = {8192, 4096};
  const int previous_updates = native_.update_count;
  ++value;
  IREE_ASSERT_OK(iree_hal_queue_map(queue, {}, signals, mappings.size(),
                                    mappings.data(), 0));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(done, value, iree_infinite_timeout(), 0));
  EXPECT_EQ(native_.update_count - previous_updates,
            native_.alias_projections ? 1 : 2);
  EXPECT_EQ(native_.last_updates.size(), native_.alias_projections ? 2u : 1u);
  mappings[0].target_range = mappings[1].target_range = {4096, 8192};

  // Conflicts are checked by shared projection, even across exact consumers.
  mappings[1].target = reservations[0];
  if (!native_.alias_projections) {
    mappings[1].scope = first;
  }
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_queue_map(queue, {}, {}, mappings.size(), mappings.data(), 0));
  for (uint32_t i = 0; i < 2; ++i) {
    mappings[i].target = reservations[i];
    mappings[i].source = nullptr;
    mappings[i].source_offset = 0;
    mappings[i].access = 0;
  }
  ++value;
  IREE_ASSERT_OK(iree_hal_queue_map(queue, {}, signals, mappings.size(),
                                    mappings.data(), 0));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(done, value, iree_infinite_timeout(), 0));
  EXPECT_EQ(native_.last_updates[0].memory, nullptr);
  EXPECT_EQ(native_.last_updates[0].access, 0u);
  for (auto* reservation : reservations) {
    iree_hal_buffer_release(reservation);
  }
  EXPECT_EQ(native_.reservation_count, 0);
  iree_hal_semaphore_release(gate);
  iree_hal_semaphore_release(done);
  iree_hal_buffer_release(view);
  iree_hal_buffer_release(backing);
  iree_hal_pool_release(pool);
}

INSTANTIATE_TEST_SUITE_P(ProjectionIdentity, ReservationBackendTest,
                         ::testing::Values(false, true));

TEST_F(MemoryBackendTest, NativeMappingFailureReachesQueueCompletion) {
  iree_hal_buffer_t* reservation = nullptr;
  IREE_ASSERT_OK(CreateReservation(&reservation));
  iree_hal_semaphore_t* done = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      cpu_device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, 0, &done));
  uint64_t value = 1;
  const iree_hal_semaphore_list_t signals = {1, &done, &value};
  const iree_hal_memory_mapping_t mapping = {
      .target = reservation,
      .scope = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]),
      .target_range = {0, 4096},
  };
  native_.update_failure = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  IREE_ASSERT_OK(iree_hal_queue_map(iree_hal_device_queue(cpu_device_, 0, 0),
                                    {}, signals, 1, &mapping, 0));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_semaphore_wait(done, value, iree_infinite_timeout(), 0));
  EXPECT_EQ(native_.update_count, 1);
  EXPECT_EQ(native_.reservation_count, 1);
  iree_hal_buffer_release(reservation);
  EXPECT_EQ(native_.reservation_count, 0);
  iree_hal_semaphore_release(done);
}

class ConcurrentMemoryBackendTest : public MemoryBackendTest {
 protected:
  void SetUp() override {
    cpu_worker_count_ = 2;
    MemoryBackendTest::SetUp();
  }
};

TEST_F(ConcurrentMemoryBackendTest,
       SlowNativeMappingLeavesOtherWorkersAvailable) {
  iree_hal_buffer_t* reservation = nullptr;
  IREE_ASSERT_OK(CreateReservation(&reservation));
  iree_hal_semaphore_t* done = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      cpu_device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, 0, &done));
  struct NativeWork {
    // Protects the native entry and release predicates.
    std::mutex mutex;
    // Publishes native entry and independent CPU work completion.
    std::condition_variable condition;
    // True once the actual native update callback is blocked.
    bool entered = false;
    // Set only by the independent CPU queue operation.
    bool released = false;
  } work;
  native_.before_update = [&] {
    std::unique_lock<std::mutex> lock(work.mutex);
    work.entered = true;
    work.condition.notify_all();
    work.condition.wait(lock, [&] { return work.released; });
  };
  uint64_t value = 1;
  const iree_hal_semaphore_list_t signals = {1, &done, &value};
  const iree_hal_memory_mapping_t mapping = {
      .target = reservation,
      .scope = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]),
      .target_range = {0, 4096},
  };
  auto* queue = iree_hal_device_queue(cpu_device_, 0, 0);
  IREE_ASSERT_OK(iree_hal_queue_map(queue, {}, signals, 1, &mapping, 0));
  {
    std::unique_lock<std::mutex> lock(work.mutex);
    work.condition.wait(lock, [&] { return work.entered; });
  }
  const uint64_t args[4] = {};
  iree_status_t status = iree_hal_queue_host_call(
      queue, {}, {},
      iree_hal_make_host_call(
          +[](void* user_data, const uint64_t*, iree_hal_host_call_context_t*) {
            auto* work = static_cast<NativeWork*>(user_data);
            std::lock_guard<std::mutex> lock(work->mutex);
            work->released = true;
            work->condition.notify_all();
            return iree_ok_status();
          },
          &work),
      args, 0);
  if (!iree_status_is_ok(status)) {
    std::lock_guard<std::mutex> lock(work.mutex);
    work.released = true;
    work.condition.notify_all();
  }
  IREE_ASSERT_OK(status);
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(done, value, iree_infinite_timeout(), 0));
  EXPECT_EQ(native_.update_count, 1);
  native_.before_update = {};
  iree_hal_buffer_release(reservation);
  iree_hal_semaphore_release(done);
}

TEST_F(MemoryBackendTest, NativeMappingFailureFailsFrontierWithoutSignals) {
  iree_hal_buffer_t* reservation = nullptr;
  IREE_ASSERT_OK(CreateReservation(&reservation));
  alignas(iree_async_frontier_t)
      uint8_t frontier_storage[sizeof(iree_async_frontier_t) +
                               sizeof(iree_async_frontier_entry_t)] = {};
  auto* frontier = reinterpret_cast<iree_async_frontier_t*>(frontier_storage);
  iree_async_frontier_initialize(frontier, 1);
  frontier->entries[0] = {
      iree_hal_device_topology_info(cpu_device_)->frontier.base_axis, 1};
  struct Completion {
    // Terminal status transferred from the queue's frontier failure.
    iree_status_t status = iree_ok_status();
    // Publishes status after the callback's last non-atomic access.
    std::atomic<bool> ready{false};
  } completion;
  iree_async_frontier_waiter_t waiter = {};
  IREE_ASSERT_OK(iree_async_frontier_tracker_wait(
      tracker_, frontier,
      +[](void* user_data, iree_status_t status) {
        auto* completion = static_cast<Completion*>(user_data);
        completion->status = status;
        completion->ready.store(true, std::memory_order_release);
      },
      &completion, &waiter));
  const iree_hal_memory_mapping_t mapping = {
      .target = reservation,
      .scope = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]),
      .target_range = {0, 4096},
  };
  native_.update_failure = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  IREE_ASSERT_OK(iree_hal_queue_map(iree_hal_device_queue(cpu_device_, 0, 0),
                                    {}, {}, 1, &mapping, 0));
  while (!completion.ready.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, completion.status);
  EXPECT_EQ(native_.update_count, 1);
  iree_hal_buffer_release(reservation);
}

TEST_F(MemoryBackendTest, FinalBufferReleaseOwnsNativeDestruction) {
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(CreateReservation(&buffer));
  iree_hal_buffer_t* view = nullptr;
  IREE_ASSERT_OK(
      iree_hal_buffer_subspan(buffer, 4096, 4096, allocator_, &view));
  native_.destroy_failure = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  iree_hal_buffer_release(buffer);
  EXPECT_EQ(native_.reservation_count, 1);
  iree_hal_virtual_buffer_info_t info;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_info(view, &info));
  EXPECT_EQ(info.mapping_byte_offset, 4096u);
  EXPECT_TRUE(native_failures_.empty());
  iree_hal_buffer_release(view);
  EXPECT_EQ(native_.reservation_count, 0);
  EXPECT_EQ(native_failures_, (std::vector<uint64_t>{native_.destroy_failure}));
}

TEST_F(MemoryBackendTest, CheckedRemovalDoesNotReleaseAddresses) {
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(CreateReservation(&buffer));
  const auto scope = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[0]);
  native_.detach_failure = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_virtual_buffer_detach(buffer, scope));
  EXPECT_EQ(native_.reservation_count, 1);
  native_.detach_failure = AMDF_STATUS_OK;
  IREE_EXPECT_OK(iree_hal_virtual_buffer_detach(buffer, scope));
  EXPECT_EQ(native_.reservation_count, 1);
  EXPECT_EQ(native_.detached_accesses, (std::vector<uint32_t>{0, 0}));
  iree_hal_buffer_release(buffer);
  EXPECT_EQ(native_.reservation_count, 0);
}

TEST_F(MemoryBackendTest,
       RejectsUnavailableReservationConsumersBeforeAllocation) {
  iree_hal_pool_family_access_t family = {
      .family = families_[0],
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
  };
  const iree_hal_pool_scope_t scope = {1, &family, {}};
  const iree_hal_virtual_buffer_options_t options = {
      .byte_length = 16384,
  };
  iree_hal_buffer_t* reservation = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_virtual_buffer_create(
                            group_, scope, &options, allocator_, &reservation));
  EXPECT_EQ(reservation, nullptr);
  EXPECT_EQ(native_.reservation_count, 0);
  EXPECT_EQ(native_.create_count, 0);
  iree_hal_pool_t* pool = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool,
                 UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE));
  EXPECT_EQ(pool, nullptr);
}

}  // namespace iree::hal::amd::fabric
