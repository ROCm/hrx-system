// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <utility>

#include "loom/serve/runtime/parameter/residency.h"
#include "loom/serve/runtime/storage/memory_test_fixture.h"

namespace {

using loom_serve::testing::memory::DedicatedNativeWorkerMemoryTest;
using loom_serve::testing::memory::DeviceMemoryTest;
using loom_serve::testing::memory::MemoryTest;

TEST_F(MemoryTest, GeometryRespectsFiniteBackingCapacity) {
  iree_hal_pool_family_access_t family = {
      .family = iree_hal_queue_family(loom_serve_device_transfer_queue(device)),
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
      .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
  };
  const iree_hal_pool_scope_t scope = {.family_count = 1, .families = &family};
  iree_hal_pool_t* source = nullptr;
  iree_hal_slab_pool_options_t source_options;
  iree_hal_slab_pool_options_initialize(&source_options);
  IREE_ASSERT_OK(iree_hal_slab_pool_create(loom_serve_device_group(device),
                                           scope, &source_options,
                                           iree_allocator_system(), &source));
  family.interfaces = 0;
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      loom_serve_device_group(device), scope, iree_allocator_system(),
      &capabilities));
  loom_serve_memory_geometry_t geometry;
  IREE_ASSERT_OK(loom_serve_memory_pool_query_geometry(&capabilities, source,
                                                       kSlabSize, &geometry));
  const iree_hal_fixed_block_pool_options_t options = {
      .block_size = geometry.slab_size,
      .blocks_per_slab = 1,
      .alignment = geometry.granularity,
  };
  iree_hal_pool_t* bounded_source = nullptr;
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create(
      source, &options, iree_allocator_system(), &bounded_source));
  iree_hal_pool_release(source);
  IREE_ASSERT_OK(loom_serve_memory_pool_query_geometry(
      &capabilities, bounded_source, kSlabSize, &geometry));
  EXPECT_EQ(geometry.slab_size, kSlabSize);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_serve_memory_pool_query_geometry(&capabilities, bounded_source,
                                            2 * kSlabSize, &geometry));
  EXPECT_EQ(geometry.slab_size, 0u);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_memory_pool_query_geometry(&capabilities, bounded_source,
                                            kSlabSize - 1, &geometry));
  IREE_ASSERT_OK(loom_serve_memory_pool_query_geometry(
      &capabilities, bounded_source, 0, &geometry));
  EXPECT_GT(geometry.slab_size, 0u);
  EXPECT_LE(geometry.slab_size, kSlabSize);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(bounded_source, &stats);
  EXPECT_EQ(stats.bytes_committed, 0u);
  iree_hal_pool_release(bounded_source);
}

TEST_F(MemoryTest, DetachedSlabsServeTwoModelsThenReturnToOriginalAddresses) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), 3 * kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), 2 * kSlabSize, 256,
      &statistics[2], &third));
  auto* const original = loom_serve_virtual_buffer_handle(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 3 * kSlabSize));
  RoundTrip(original, 2 * kSlabSize, 0x12345678);
  // Warm swaps must not ask the native provider for another slab. Keep a
  // failure armed through both swaps, then consume it after an explicit trim.
  native.FailNextAllocation(IREE_ALLOCATOR_COMMAND_CALLOC,
                            /*native_only=*/true);
  loom_serve_virtual_buffer_begin_trim(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  auto backing = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(statistics[0].committed_bytes, 0u);
  EXPECT_EQ(backing.committed_bytes, 3 * kSlabSize);
  EXPECT_EQ(backing.cached_bytes, 3 * kSlabSize);

  const loom_serve_memory_range_t ranges[] = {{second, 0, kSlabSize},
                                              {third, 0, 2 * kSlabSize}};
  bool admitted = false;
  IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(pool, IREE_ARRAYSIZE(ranges),
                                                   ranges, &admitted));
  ASSERT_TRUE(admitted);
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x87654321);
  RoundTrip(loom_serve_virtual_buffer_handle(third), kSlabSize, 0xabcdef01);
  CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x87654321);
  backing = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(backing.committed_bytes, 3 * kSlabSize);
  EXPECT_EQ(backing.cached_bytes, 0u);

  loom_serve_virtual_buffer_begin_trim(second);
  loom_serve_virtual_buffer_begin_trim(third);
  loom_serve_virtual_buffer_t* detached[] = {second, third};
  loom_serve_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_serve_memory_pool_trim_async(
      pool, IREE_ARRAYSIZE(detached), detached, nullptr, 0, &operation));
  loom_serve_memory_operation_await(operation);
  IREE_ASSERT_OK(loom_serve_memory_operation_query_status(operation));
  loom_serve_memory_operation_free(operation);
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 3 * kSlabSize));
  EXPECT_EQ(loom_serve_virtual_buffer_handle(first), original);
  RoundTrip(original, 2 * kSlabSize, 0x10203040);
  backing = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(backing.committed_bytes, 3 * kSlabSize);
  EXPECT_EQ(backing.cached_bytes, 0u);

  // Explicit physical trim cannot destroy the active model's mappings.
  loom_serve_memory_pool_trim(pool, 0);
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  CheckRetained(original, 2 * kSlabSize, 0x10203040);
  loom_serve_virtual_buffer_begin_trim(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  loom_serve_memory_pool_trim(pool, 0);
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
}

TEST_F(MemoryTest, WeightTailReusesStateClassAcrossTwoCompleteSwaps) {
  const auto bulk = loom_serve_device_weight_geometry(device);
  const auto state = loom_serve_device_state_geometry(device);
  const uint64_t bytes = 2 * bulk.slab_size + state.slab_size;
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, bulk, bytes, 256,
                                                    &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, state, state.slab_size, 256, &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, bulk, 2 * bulk.slab_size, 256, &statistics[2], &third));
  auto* const identity = loom_serve_virtual_buffer_handle(first);
  const auto missing = loom_serve_virtual_buffer_missing_demand(first);
  EXPECT_EQ(missing.cached_bytes[0], 2 * bulk.slab_size);
  EXPECT_EQ(missing.cached_bytes[1], state.slab_size);
  EXPECT_EQ(missing.uncached_bytes, 0u);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, bytes));
  native.FailNextAllocation(IREE_ALLOCATOR_COMMAND_CALLOC, true);
  for (uint32_t cycle = 0; cycle < 2; ++cycle) {
    RoundTrip(identity, 2 * bulk.slab_size, 100 + cycle);
    loom_serve_virtual_buffer_begin_trim(first);
    IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
    const loom_serve_memory_range_t ranges[] = {{second, 0, state.slab_size},
                                                {third, 0, 2 * bulk.slab_size}};
    bool admitted = false;
    IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(
        pool, IREE_ARRAYSIZE(ranges), ranges, &admitted));
    ASSERT_TRUE(admitted);
    RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 200 + cycle);
    RoundTrip(loom_serve_virtual_buffer_handle(third), bulk.slab_size,
              300 + cycle);
    loom_serve_virtual_buffer_begin_trim(second);
    loom_serve_virtual_buffer_begin_trim(third);
    IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(second));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(third));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, bytes));
    EXPECT_EQ(loom_serve_virtual_buffer_handle(first), identity);
    EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, bytes);
    const auto backed = loom_serve_virtual_buffer_missing_demand(first);
    EXPECT_EQ(backed.cached_bytes[0], 0u);
    EXPECT_EQ(backed.cached_bytes[1], 0u);
    EXPECT_EQ(backed.uncached_bytes, 0u);
  }
  // The failure remains armed through both warm cycles and is consumed only
  // after explicitly disposing the reusable native slabs.
  loom_serve_virtual_buffer_begin_trim(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  loom_serve_memory_pool_trim(pool, 0);
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_virtual_buffer_commit(first, 0, bytes));
}

TEST_F(MemoryTest, PhysicalPeakSurvivesCachedSlabDisposal) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_state_geometry(device), kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  loom_serve_virtual_buffer_begin_trim(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
  // Neither reservation ever owns more than one slab's worth of bytes, but
  // the native source owns both the live state class and the idle weight class.
  auto memory = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(memory.committed_bytes, 2 * kSlabSize);
  EXPECT_EQ(memory.cached_bytes, kSlabSize);
  EXPECT_EQ(memory.peak_bytes, 2 * kSlabSize);
  loom_serve_memory_pool_trim(pool, 0);
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  memory = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(memory.committed_bytes, kSlabSize);
  EXPECT_EQ(memory.cached_bytes, 0u);
  EXPECT_EQ(memory.peak_bytes, 2 * kSlabSize);
  CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
}

TEST_F(DedicatedNativeWorkerMemoryTest,
       HeldGrowthAndUnrelatedReturnDoNotCreateOtherClassCredit) {
  const auto bulk = loom_serve_device_weight_geometry(device);
  const auto state = loom_serve_device_state_geometry(device);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, bulk, kSlabSize, 256,
                                                    &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, bulk, kSlabSize, 256,
                                                    &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, state, 2 * kSlabSize, 256, &statistics[2], &third));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  loom_serve_memory_operation_t* operation = nullptr;
  {
    loom_serve::testing::ScopedNativeHold hold(native,
                                               IREE_ALLOCATOR_COMMAND_CALLOC);
    const loom_serve_memory_range_t range = {second, 0, kSlabSize};
    loom_serve_memory_admission_t commit_admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &range, nullptr, 0, &commit_admission, &operation));
    ASSERT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    hold.WaitEntered();
    // No consumers remain on first. Its checked detach returns a regular
    // slab while the independent cold acquisition still owns pending capacity.
    IREE_ASSERT_OK(loom_serve_virtual_buffer_free(first));
    first = nullptr;
    loom_serve_memory_admission_t admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
        pool, loom_serve_virtual_buffer_missing_demand(third), &admission));
    EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_DEFERRED);
    EXPECT_EQ(statistics[2].pending_bytes, 0u);
  }
  loom_serve_memory_operation_await(operation);
  IREE_ASSERT_OK(loom_serve_memory_operation_query_status(operation));
  loom_serve_memory_operation_free(operation);
  loom_serve_memory_admission_t admission;
  IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
      pool, loom_serve_virtual_buffer_missing_demand(third), &admission));
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(third, 0, 2 * kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x11223344);
  RoundTrip(loom_serve_virtual_buffer_handle(third), kSlabSize, 0x55667788);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            3 * kSlabSize);
}

TEST_F(DedicatedNativeWorkerMemoryTest,
       IncompatibleIdleSlabsStayChargedUntilNativeDisposal) {
  const auto bulk = loom_serve_device_weight_geometry(device);
  const auto state = loom_serve_device_state_geometry(device);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, bulk, 3 * kSlabSize, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, state, 2 * kSlabSize, 256, &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 3 * kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
  loom_serve_virtual_buffer_begin_trim(first);
  loom_serve_virtual_buffer_keep(first, 0, kSlabSize);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  {
    loom_serve::testing::ScopedNativeHold hold(native,
                                               IREE_ALLOCATOR_COMMAND_FREE);
    loom_serve_memory_admission_t admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
        pool, loom_serve_virtual_buffer_missing_demand(second), &admission));
    EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_DEFERRED);
    hold.WaitEntered();
    const auto memory = loom_serve_memory_pool_statistics(pool);
    EXPECT_EQ(memory.committed_bytes, 3 * kSlabSize);
    EXPECT_EQ(memory.releasing_bytes, 2 * kSlabSize);
    EXPECT_EQ(memory.cached_bytes, 0u);
    CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
  }
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, 2 * kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(second), kSlabSize, 0x87654321);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            3 * kSlabSize);
}

TEST_F(DedicatedNativeWorkerMemoryTest,
       FailedAcquisitionPreservesAnAdmittedStreamingCohort) {
  const auto bulk = loom_serve_device_weight_geometry(device);
  const auto state = loom_serve_device_state_geometry(device);
  const uint64_t failed_bytes = 2 * state.slab_size;
  const uint64_t cohort_bytes = 6 * state.slab_size;
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, state, failed_bytes, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, bulk, bulk.slab_size, 256, &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, state, cohort_bytes, 256, &statistics[2], &third));
  std::array<loom_serve_memory_operation_t*, 4> operations = {};
  native.FailNextAllocation(IREE_ALLOCATOR_COMMAND_CALLOC, true);
  {
    loom_serve::testing::ScopedNativeHold hold(native,
                                               IREE_ALLOCATOR_COMMAND_CALLOC);
    const loom_serve_memory_range_t failed = {first, 0, failed_bytes};
    loom_serve_memory_admission_t commit_admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &failed, nullptr, 0, &commit_admission, &operations[0]));
    ASSERT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    hold.WaitEntered();
    const loom_serve_memory_range_t growth = {second, 0, bulk.slab_size};
    IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &growth, nullptr, 0, &commit_admission, &operations[1]));
    ASSERT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);

    // A complete cohort is admitted before its separate read windows commit
    // backing. Both unrelated operations still own accepted capacity here.
    loom_serve_memory_admission_t admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
        pool, loom_serve_virtual_buffer_missing_demand(third), &admission));
    ASSERT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    hold.ReleaseAndArm(IREE_ALLOCATOR_COMMAND_CALLOC);
    hold.WaitEntered();
    loom_serve_memory_operation_await(operations[0]);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        loom_serve_memory_operation_query_status(operations[0]));
    loom_serve_memory_operation_free(operations[0]);
    operations[0] = nullptr;
    EXPECT_FALSE(loom_serve_memory_operation_is_complete(operations[1]));
    IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool));

    // Publishing the unrelated failure must neither revoke the cohort's
    // admission nor grant its still-unreconciled native charge to a retry.
    const loom_serve_memory_range_t windows[] = {
        {third, 0, state.slab_size},
        {third, state.slab_size, cohort_bytes - state.slab_size},
    };
    for (size_t i = 0; i < IREE_ARRAYSIZE(windows); ++i) {
      IREE_EXPECT_OK(loom_serve_memory_pool_commit_async(
          pool, 1, windows + i, nullptr, 0, &commit_admission,
          &operations[i + 2]));
      EXPECT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    }
    IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
        pool, loom_serve_virtual_buffer_missing_demand(first), &admission));
    EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_DEFERRED);
  }
  for (auto* operation : operations) {
    if (!operation) {
      continue;
    }
    loom_serve_memory_operation_await(operation);
    IREE_EXPECT_OK(loom_serve_memory_operation_query_status(operation));
    loom_serve_memory_operation_free(operation);
  }
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            bulk.slab_size + cohort_bytes);
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
  RoundTrip(loom_serve_virtual_buffer_handle(third), state.slab_size,
            0x87654321);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, failed_bytes));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x11223344);
  CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
  CheckRetained(loom_serve_virtual_buffer_handle(third), state.slab_size,
                0x87654321);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            3 * kSlabSize);
}

TEST_F(DedicatedNativeWorkerMemoryTest,
       ExplicitTrimSettlesBeforeStreamingCohortAdmission) {
  const auto bulk = loom_serve_device_weight_geometry(device);
  const auto state = loom_serve_device_state_geometry(device);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, bulk, bulk.slab_size, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, state, state.slab_size, 256, &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, bulk, 2 * bulk.slab_size, 256, &statistics[2], &third));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, bulk.slab_size));
  loom_serve_virtual_buffer_begin_trim(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  loom_serve_memory_operation_t* operation = nullptr;
  {
    loom_serve::testing::ScopedNativeHold hold(native,
                                               IREE_ALLOCATOR_COMMAND_CALLOC);
    const loom_serve_memory_range_t growth = {second, 0, state.slab_size};
    loom_serve_memory_admission_t commit_admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &growth, nullptr, 0, &commit_admission, &operation));
    ASSERT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    hold.WaitEntered();
    loom_serve_memory_pool_trim(pool, 0);
    // The cohort would fit, but its promise must not depend on cached backing
    // that an already-requested trim could destroy between streamed commits.
    loom_serve_memory_admission_t admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
        pool, loom_serve_virtual_buffer_missing_demand(third), &admission));
    EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_DEFERRED);
  }
  loom_serve_memory_operation_await(operation);
  IREE_ASSERT_OK(loom_serve_memory_operation_query_status(operation));
  loom_serve_memory_operation_free(operation);
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            state.slab_size);
  loom_serve_memory_admission_t admission;
  IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
      pool, loom_serve_virtual_buffer_missing_demand(third), &admission));
  ASSERT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(third, 0, bulk.slab_size));
  IREE_ASSERT_OK(
      loom_serve_virtual_buffer_commit(third, bulk.slab_size, bulk.slab_size));
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x11223344);
  RoundTrip(loom_serve_virtual_buffer_handle(third), bulk.slab_size,
            0x55667788);
}

TEST_F(MemoryTest, MixedSlabsAndShortTailsShareExactNativeBudget) {
  const auto bulk = loom_serve_device_weight_geometry(device);
  const auto state = loom_serve_device_state_geometry(device);
  const uint64_t tail_bytes = state.slab_size;
  const uint64_t first_bytes = bulk.slab_size + tail_bytes;
  const uint64_t committed_bytes = first_bytes + state.slab_size;
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking(committed_bytes));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, bulk, first_bytes - 1, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, state, 2 * state.slab_size, 256, &statistics[1], &second));
  EXPECT_EQ(statistics[0].reserved_bytes, first_bytes);
  EXPECT_EQ(loom_serve_virtual_buffer_slab_count(first), 2u);

  // Duplicate and overlapping logical ranges charge each physical extent once.
  const loom_serve_memory_range_t ranges[] = {
      {first, bulk.slab_size - 4, 8},
      {second, 0, 4},
      {first, bulk.slab_size, tail_bytes - 1},
  };
  auto footprint =
      loom_serve_memory_ranges_footprint(IREE_ARRAYSIZE(ranges), ranges);
  EXPECT_EQ(footprint.total_bytes, committed_bytes);
  EXPECT_EQ(footprint.missing_bytes, committed_bytes);
  deferred_backing->defer = true;
  loom_serve_memory_admission_t commit_admission;
  loom_serve_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
      pool, IREE_ARRAYSIZE(ranges), ranges, nullptr, 0, &commit_admission,
      &operation));
  ASSERT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
  deferred_backing->WaitForAcquisition();
  EXPECT_EQ(statistics[0].pending_bytes, first_bytes);
  EXPECT_EQ(statistics[1].pending_bytes, state.slab_size);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).pending_bytes,
            committed_bytes);
  footprint =
      loom_serve_memory_ranges_footprint(IREE_ARRAYSIZE(ranges), ranges);
  EXPECT_EQ(footprint.total_bytes, committed_bytes);
  EXPECT_EQ(footprint.missing_bytes, 0u);
  deferred_backing->ResumeAcquisitions();
  loom_serve_memory_operation_await(operation);
  IREE_ASSERT_OK(loom_serve_memory_operation_query_status(operation));
  loom_serve_memory_operation_free(operation);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            committed_bytes);
  EXPECT_EQ(statistics[0].committed_bytes, first_bytes);
  EXPECT_EQ(statistics[1].committed_bytes, state.slab_size);
  RoundTrip(loom_serve_virtual_buffer_handle(first), bulk.slab_size,
            0x12345678);
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x87654321);

  const loom_serve_memory_range_t extra = {second, state.slab_size, 4};
  IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
      pool, 1, &extra, nullptr, 0, &commit_admission, &operation));
  EXPECT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_CAPACITY);
  EXPECT_EQ(operation, nullptr);

  // Return a full bulk slab and a state slab in one batch, retaining the tail.
  // Native return, not the logical trim, makes their capacity available again.
  deferred_backing->DeferReturns();
  loom_serve_virtual_buffer_begin_trim(first);
  loom_serve_virtual_buffer_keep(first, bulk.slab_size, tail_bytes);
  loom_serve_virtual_buffer_begin_trim(second);
  loom_serve_virtual_buffer_t* buffers[] = {first, second};
  IREE_ASSERT_OK(loom_serve_memory_pool_trim_async(
      pool, IREE_ARRAYSIZE(buffers), buffers, nullptr, 0, &operation));
  loom_serve_memory_operation_await(operation);
  IREE_ASSERT_OK(loom_serve_memory_operation_query_status(operation));
  loom_serve_memory_operation_free(operation);
  EXPECT_EQ(statistics[0].committed_bytes, tail_bytes);
  EXPECT_EQ(statistics[1].committed_bytes, 0u);
  loom_serve_memory_pool_trim(pool, 0);
  const auto retiring = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(retiring.committed_bytes, committed_bytes);
  EXPECT_EQ(retiring.releasing_bytes, bulk.slab_size + state.slab_size);
  EXPECT_EQ(deferred_backing->PendingReturnCount(), 2u);
  IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
      pool, 1, &extra, nullptr, 0, &commit_admission, &operation));
  EXPECT_EQ(operation, nullptr);
  CheckRetained(loom_serve_virtual_buffer_handle(first), bulk.slab_size,
                0x12345678);
  // Retirement can finish before a caller handles the denied commitment. Its
  // original result must still request another admission turn: a fresh pending
  // query no longer distinguishes progress from permanent capacity exhaustion.
  deferred_backing->CompleteReturns();
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  EXPECT_FALSE(loom_serve_memory_pool_is_pending(pool));
  EXPECT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_DEFERRED);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            tail_bytes);
  bool admitted = false;
  IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(pool, 1, &extra, &admitted));
  ASSERT_TRUE(admitted);
  RoundTrip(loom_serve_virtual_buffer_handle(second), state.slab_size,
            0xABCDEF01);
}

TEST_F(MemoryTest, PendingReleaseRetainsBudgetAndLeavesKeptSlabsUsable) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking());
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), 3 * kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), 2 * kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 3 * kSlabSize));
  ASSERT_NO_FATAL_FAILURE(
      RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0xA1B2C3D4));
  std::atomic<uint32_t> wakes = 0;
  loom_serve_memory_pool_set_observer(
      pool,
      [](void* user_data) {
        static_cast<std::atomic<uint32_t>*>(user_data)->fetch_add(1);
      },
      &wakes);
  loom_serve_memory_operation_t* operation = nullptr;
  iree_hal_semaphore_t* ready = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready));
  {
    // Queue completion returns ownership to the source pool. Keep its real
    // parent tokens live until explicitly releasing them below, independently
    // of metadata destruction or the threads chosen by a native backend.
    deferred_backing->DeferReturns();
    loom_serve_virtual_buffer_begin_trim(first);
    loom_serve_virtual_buffer_keep(first, 0, kSlabSize);
    IREE_ASSERT_OK(loom_serve_memory_pool_trim_async(pool, 1, &first, ready, 1,
                                                     &operation));
    ASSERT_NE(operation, nullptr);
    loom_serve_memory_operation_await(operation);
    loom_serve_memory_pool_trim(pool, 0);
    EXPECT_EQ(deferred_backing->PendingReturnCount(), 2u);
    EXPECT_TRUE(loom_serve_memory_operation_is_complete(operation));
    const auto memory = loom_serve_memory_pool_statistics(pool);
    EXPECT_EQ(memory.committed_bytes, 3 * kSlabSize);
    EXPECT_EQ(memory.releasing_bytes, 2 * kSlabSize);
    EXPECT_EQ(statistics[0].releasing_bytes, 0u);
    loom_serve_memory_admission_t admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
        pool, loom_serve_virtual_buffer_missing_demand(second), &admission));
    EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_DEFERRED);
    ASSERT_NO_FATAL_FAILURE(
        CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0xA1B2C3D4));
    EXPECT_GE(wakes.load(), 1u);
    uint64_t value = 0;
    IREE_EXPECT_OK(iree_hal_semaphore_query(ready, &value));
    EXPECT_EQ(value, 1u);
    deferred_backing->CompleteReturns();
  }
  IREE_ASSERT_OK(iree_hal_semaphore_wait(ready, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_TRUE(loom_serve_memory_operation_is_complete(operation));
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  const auto retired = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(retired.pending_bytes, 0u);
  EXPECT_EQ(retired.releasing_bytes, 0u);
  EXPECT_EQ(retired.committed_bytes, kSlabSize);
  EXPECT_EQ(retired.released_bytes, 2 * kSlabSize);
  EXPECT_GE(wakes.load(), 1u);
  EXPECT_EQ(statistics[0].releasing_bytes, 0u);
  EXPECT_EQ(statistics[0].committed_bytes, kSlabSize);
  EXPECT_EQ(statistics[0].released_bytes, 2 * kSlabSize);
  loom_serve_memory_operation_free(operation);
  loom_serve_memory_pool_set_observer(pool, nullptr, nullptr);
  operation = nullptr;
  {
    loom_serve::testing::ScopedNativeHold gate(native,
                                               IREE_ALLOCATOR_COMMAND_CALLOC);
    IREE_ASSERT_OK(loom_serve_memory_pool_trim_async(pool, 1, &first, ready, 2,
                                                     &operation));
    EXPECT_EQ(operation, nullptr);
    uint64_t value = 0;
    IREE_EXPECT_OK(iree_hal_semaphore_query(ready, &value));
    EXPECT_EQ(value, 2u);
  }
  iree_hal_semaphore_release(ready);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, 2 * kSlabSize));
  ASSERT_NO_FATAL_FAILURE(
      RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x55667788));
}

TEST_F(DedicatedNativeWorkerMemoryTest,
       PoolObserverDetachesFromAlreadyAcceptedNativeWork) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[0], &first));
  std::atomic<uint32_t> wakes = 0;
  loom_serve_memory_pool_set_observer(
      pool,
      [](void* user_data) {
        static_cast<std::atomic<uint32_t>*>(user_data)->fetch_add(1);
      },
      &wakes);
  loom_serve_memory_operation_t* operation = nullptr;
  {
    loom_serve::testing::ScopedNativeHold gate(native,
                                               IREE_ALLOCATOR_COMMAND_CALLOC);
    const loom_serve_memory_range_t range = {first, 0, kSlabSize};
    loom_serve_memory_admission_t commit_admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &range, nullptr, 0, &commit_admission, &operation));
    ASSERT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    gate.WaitEntered();
    loom_serve_memory_pool_set_observer(pool, nullptr, nullptr);
  }
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  EXPECT_EQ(wakes.load(), 0u);
  loom_serve_memory_operation_free(operation);
}

TEST_F(DeviceMemoryTest, SparseGrowthTrimRegrowthAndSharedAdmission) {
  const auto before = loom_serve_memory_pool_statistics(pool);
  const auto* dispatch =
      iree_hal_queue_family(loom_serve_device_dispatch_queue(device));
  const auto* transfer =
      iree_hal_queue_family(loom_serve_device_transfer_queue(device));
  const iree_hal_pool_family_access_t families[] = {
      {
          .family = dispatch,
          .usage = IREE_HAL_BUFFER_USAGE_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS,
      },
      {.family = transfer, .usage = IREE_HAL_BUFFER_USAGE_TRANSFER},
  };
  const iree_hal_pool_scope_t scope = {
      .family_count = dispatch == transfer ? 1u : 2u,
      .families = families,
  };
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      loom_serve_device_group(device), scope, iree_allocator_system(),
      &capabilities));
  if (capabilities.max_byte_length < kHighOffset + kSlabSize) {
    GTEST_SKIP() << "Native address interval limit is "
                 << capabilities.max_byte_length
                 << " bytes; this case requires offsets above four GiB";
  }
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kHighOffset + kSlabSize,
      256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), 2 * kSlabSize, 256,
      &statistics[1], &second));
  auto* const identity = loom_serve_virtual_buffer_handle(first);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_device_group_free(group));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, 0u);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, kHighOffset, 256));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 512));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 256, 256));
  RoundTrip(identity, kHighOffset, 0x12345678);
  RoundTrip(identity, 0, 0xABCDEF01);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            2 * kSlabSize);

  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, 256));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_virtual_buffer_commit(second, kSlabSize, 256));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            3 * kSlabSize);
  loom_serve_virtual_buffer_begin_trim(first);
  loom_serve_virtual_buffer_keep(first, kHighOffset, 256);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).released_bytes,
            before.released_bytes + kSlabSize);
  EXPECT_EQ(loom_serve_virtual_buffer_handle(first), identity);
  CheckRetained(identity, kHighOffset, 0x12345678);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, kSlabSize, 256));
  RoundTrip(loom_serve_virtual_buffer_handle(second), kSlabSize, 0xDEADBEEF);
  loom_serve_virtual_buffer_begin_trim(second);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 512));
  EXPECT_EQ(loom_serve_virtual_buffer_handle(first), identity);
  RoundTrip(identity, 0, 0xCAFEBABE);
  CheckRetained(identity, kHighOffset, 0x12345678);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).peak_bytes, 3 * kSlabSize);
  EXPECT_EQ(statistics[0].committed_bytes, 2 * kSlabSize);
  EXPECT_EQ(statistics[1].committed_bytes, 0u);
  EXPECT_EQ(statistics[0].released_bytes, kSlabSize);
  EXPECT_EQ(statistics[1].released_bytes, 2 * kSlabSize);
  EXPECT_EQ(statistics[0].peak_bytes, 2 * kSlabSize);
  EXPECT_EQ(statistics[1].peak_bytes, 2 * kSlabSize);
}

TEST_F(MemoryTest, CohortAdmissionCountsUniqueSlabsAndLeavesDenialUntouched) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), 3 * kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), 2 * kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 256));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
  const loom_serve_memory_range_t ranges[] = {
      {first, 0, 256},
      {first, kSlabSize, 256},
      {first, kSlabSize + 128, 256},
      {second, kSlabSize - 128, 256},
  };
  // Forecast all touched slots, not just missing backing. Repeated queries
  // neither accumulate marks nor affect the commitment/trim transaction.
  auto footprint =
      loom_serve_memory_ranges_footprint(IREE_ARRAYSIZE(ranges), ranges);
  EXPECT_EQ(footprint.total_bytes, 4 * kSlabSize);
  EXPECT_EQ(footprint.missing_bytes, 3 * kSlabSize);
  footprint = loom_serve_memory_ranges_footprint(3, ranges);
  EXPECT_EQ(footprint.total_bytes, 2 * kSlabSize);
  EXPECT_EQ(footprint.missing_bytes, kSlabSize);
  footprint = loom_serve_memory_ranges_footprint(0, nullptr);
  EXPECT_EQ(footprint.total_bytes, 0u);
  EXPECT_EQ(footprint.missing_bytes, 0u);
  bool admitted = true;
  IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(pool, IREE_ARRAYSIZE(ranges),
                                                   ranges, &admitted));
  EXPECT_FALSE(admitted);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, kSlabSize);
  EXPECT_EQ(statistics[1].committed_bytes, 0u);
  footprint =
      loom_serve_memory_ranges_footprint(IREE_ARRAYSIZE(ranges), ranges);
  EXPECT_EQ(footprint.total_bytes, 4 * kSlabSize);
  EXPECT_EQ(footprint.missing_bytes, 3 * kSlabSize);
  CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
  IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(pool, 3, ranges, &admitted));
  EXPECT_TRUE(admitted);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            2 * kSlabSize);
  footprint = loom_serve_memory_ranges_footprint(3, ranges);
  EXPECT_EQ(footprint.total_bytes, 2 * kSlabSize);
  EXPECT_EQ(footprint.missing_bytes, 0u);
  RoundTrip(loom_serve_virtual_buffer_handle(first), kSlabSize, 0x87654321);
  IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(pool, 3, ranges, &admitted));
  EXPECT_TRUE(admitted);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            2 * kSlabSize);
  CheckRetained(loom_serve_virtual_buffer_handle(first), kSlabSize, 0x87654321);
  const loom_serve_memory_range_t last = {second, 0, 256};
  IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(pool, 1, &last, &admitted));
  EXPECT_TRUE(admitted);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            3 * kSlabSize);
}

TEST_F(MemoryTest, RecommittedSlabsPublishEveryWordOfTheNewGeneration) {
  constexpr size_t kGenerationCount = 4;
  constexpr uint64_t kOrigin = 2 * kSlabSize;
  constexpr size_t kWordsPerSlab = kSlabSize / sizeof(uint32_t);
  const std::array<loom_serve_virtual_buffer_t**, 2> destinations = {&first,
                                                                     &second};
  for (size_t i = 0; i < destinations.size(); ++i) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
        pool, loom_serve_device_weight_geometry(device),
        kOrigin + 4 * kSlabSize, 256, &statistics[i], destinations[i]));
  }
  const std::array<iree_hal_buffer_t*, 2> handles = {
      loom_serve_virtual_buffer_handle(first),
      loom_serve_virtual_buffer_handle(second)};
  // Prepare every generation before the first mapping turnover. Comparing or
  // filling a large host image between remap and device access would perturb
  // the publication boundary this case exercises.
  std::array<std::vector<uint32_t>, kGenerationCount> expected;
  for (size_t generation = 0; generation < kGenerationCount; ++generation) {
    expected[generation].resize(2 * kWordsPerSlab);
    for (size_t word = 0; word < expected[generation].size(); ++word) {
      expected[generation][word] =
          static_cast<uint32_t>((generation + 1) << 28) |
          static_cast<uint32_t>(word);
    }
  }
  std::vector<uint32_t> observed(2 * kWordsPerSlab);
  std::vector<uint32_t> captured(2 * kWordsPerSlab);
  uint64_t completion = 0;
  for (size_t generation = 0; generation < kGenerationCount; ++generation) {
    SCOPED_TRACE(generation);
    const uint64_t offset = kOrigin + (generation % 2) * 2 * kSlabSize;
    const std::array<loom_serve_memory_range_t, 2> ranges = {{
        {first, offset, kSlabSize},
        {second, offset, kSlabSize},
    }};
    std::array<iree_hal_transfer_operation_t, 2> uploads = {};
    std::array<iree_hal_transfer_operation_t, 2> downloads = {};
    for (size_t i = 0; i < handles.size(); ++i) {
      uploads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
      uploads[i].upload = {expected[generation].data() + i * kWordsPerSlab,
                           handles[i], offset, kSlabSize};
      downloads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
      downloads[i].download = {handles[i], offset,
                               observed.data() + i * kWordsPerSlab, kSlabSize};
    }
    bool admitted = false;
    IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(pool, ranges.size(),
                                                     ranges.data(), &admitted));
    ASSERT_TRUE(admitted);
    IREE_ASSERT_OK(loom_serve_execution_transfer(execution, uploads.size(),
                                                 uploads.data(), &completion));
    IREE_ASSERT_OK(loom_serve_execution_transfer(
        execution, downloads.size(), downloads.data(), &completion));
    IREE_ASSERT_OK(loom_serve_execution_wait(execution, completion));
    ASSERT_EQ(observed, expected[generation]);
    if (generation) {
      ASSERT_EQ(captured, expected[generation - 1]);
    }
    // Capture and immediately relinquish both physical slabs, as in a cold
    // session suspension. The next iteration reuses the retained VA roots.
    for (size_t i = 0; i < handles.size(); ++i) {
      downloads[i].download.target = captured.data() + i * kWordsPerSlab;
    }
    IREE_ASSERT_OK(loom_serve_execution_transfer(
        execution, downloads.size(), downloads.data(), &completion));
    IREE_ASSERT_OK(loom_serve_execution_wait(execution, completion));
    for (auto* buffer : {first, second}) {
      loom_serve_virtual_buffer_begin_trim(buffer);
      IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(buffer));
    }
  }
  ASSERT_EQ(captured, expected.back());
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            2 * kSlabSize);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).cached_bytes,
            2 * kSlabSize);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).peak_bytes, 2 * kSlabSize);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).released_bytes,
            kGenerationCount * 2 * kSlabSize);
}

TEST_F(MemoryTest, FailedReclamationRetainsSharedFailureAndRemainingOwnership) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 256));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, 256));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x87654321);
  loom_serve_memory_admission_t admission;
  IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
      pool, {.cached_bytes = {4 * kSlabSize}}, &admission));
  EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_CAPACITY);
  IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool));

  // The dependency returns reusable backing before reporting a subsequent
  // failure. The requested capacity would now fit, but cannot be admitted.
  loom_serve_memory_pool_set_reclaimer(
      pool, {first, [](void* self, uint64_t target_bytes) -> iree_status_t {
               auto* buffer = static_cast<loom_serve_virtual_buffer_t*>(self);
               loom_serve_virtual_buffer_begin_trim(buffer);
               IREE_RETURN_IF_ERROR(loom_serve_virtual_buffer_trim(buffer));
               return iree_make_status(IREE_STATUS_UNAVAILABLE,
                                       "reclamation failed after release");
             }});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      loom_serve_memory_pool_prepare(pool, {.cached_bytes = {2 * kSlabSize}},
                                     &admission));
  EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_CAPACITY);
  EXPECT_EQ(statistics[0].committed_bytes, 0u);
  EXPECT_EQ(statistics[1].committed_bytes, kSlabSize);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).reserved_bytes,
            2 * kSlabSize);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            2 * kSlabSize);
  loom_serve_memory_pool_set_reclaimer(pool, {});
  for (int i = 0; i < 2; ++i) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                          loom_serve_memory_pool_query_status(pool));
  }
  // No new device work follows shared failure. TearDown drains accepted work
  // and releases the remaining ownership through the ordinary native path.
}

TEST_F(DeviceMemoryTest, StateAndWorkspaceReuseOnePhysicalSlab) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[0], &first));
  auto* const state = loom_serve_virtual_buffer_handle(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  RoundTrip(state, 0, 0x12345678);
  loom_serve_virtual_buffer_begin_trim(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).cached_bytes, kSlabSize);

  const iree_hal_pool_reservation_request_t request = {
      .allocation_size = kSlabSize / 2,
      .min_alignment = 256,
  };
  IREE_ASSERT_OK(
      loom_serve_execution_alloca(execution, 1, &request, &workspace));
  RoundTrip(workspace, 0, 0xABCDEF01);
  const auto active = loom_serve_device_memory_statistics(device);
  EXPECT_EQ(active.retained.committed_bytes, kSlabSize);
  EXPECT_EQ(active.retained.cached_bytes, 0u);
  EXPECT_EQ(active.workspace.bytes_committed, kSlabSize);

  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_dealloca(execution, workspace, &completion));
  iree_hal_buffer_release(workspace);
  workspace = nullptr;
  IREE_ASSERT_OK(loom_serve_execution_drain(execution));
  auto* shared_workspace = loom_serve_device_workspace(device);
  iree_device_size_t retained = 0;
  IREE_ASSERT_OK(loom_serve_workspace_trim(shared_workspace, &retained));
  EXPECT_EQ(retained, 0u);
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  IREE_ASSERT_OK(loom_serve_workspace_advance(shared_workspace));
  const auto detached = loom_serve_workspace_memory_usage(shared_workspace);
  EXPECT_GT(detached.reserved_bytes, 0u);
  EXPECT_EQ(detached.committed_bytes, 0u);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).cached_bytes, kSlabSize);

  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  EXPECT_EQ(loom_serve_virtual_buffer_handle(first), state);
  RoundTrip(state, 0, 0x87654321);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, kSlabSize);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).cached_bytes, 0u);
}

TEST_F(DeviceMemoryTest, AdmissionIncludesWorkspaceSlabGranularity) {
  const uint64_t state_bytes = 5 * kSlabSize / 2;
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_state_geometry(device), state_bytes, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, state_bytes));
  loom_serve_residency_t* residency = nullptr;
  IREE_ASSERT_OK(loom_serve_residency_allocate(
      loom_serve_device_residency_cache(device), execution, 0, nullptr,
      iree_allocator_system(), &residency));
  std::unique_ptr<loom_serve_residency_t, decltype(&loom_serve_residency_free)>
      claim(residency, loom_serve_residency_free);
  IREE_ASSERT_OK(loom_serve_residency_prepare_working_set(residency, 256, 0));
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residency, &admission));
  // Half a slab remains, but even this small invocation needs one whole
  // workspace slab. Reject it before any queue allocation is accepted.
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
  if (admission == LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
    loom_serve_residency_release(residency);
  }
  EXPECT_EQ(loom_serve_execution_frontier(execution).values[0], 0u);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_free(first));
  first = nullptr;
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  IREE_ASSERT_OK(loom_serve_residency_acquire_wait(residency));
  const auto request = iree_hal_make_pool_reservation_request(256, 256);
  IREE_ASSERT_OK(
      loom_serve_execution_alloca(execution, 1, &request, &workspace));
  RoundTrip(workspace, 0, 0x12345678);
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_dealloca(execution, workspace, &completion));
  iree_hal_buffer_release(workspace);
  workspace = nullptr;
  IREE_ASSERT_OK(loom_serve_execution_drain(execution));
  loom_serve_residency_release(residency);
  IREE_ASSERT_OK(loom_serve_test_device_trim_and_drain(device, 0));
}

TEST_F(DeviceMemoryTest, DeviceTrimPreservesLiveWorkspaceAndMutableState) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 256));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0xABCDEF01);
  IREE_ASSERT_OK(
      loom_serve_execution_reserve_workspace(execution, kSlabSize, 256));
  const iree_hal_pool_reservation_request_t request = {
      .allocation_size = kSlabSize,
      .min_alignment = 256,
  };
  for (int iteration = 0; iteration < 2; ++iteration) {
    IREE_ASSERT_OK(
        loom_serve_execution_alloca(execution, 1, &request, &workspace));
    RoundTrip(workspace, 0, 0x12345678);
    const auto before = loom_serve_device_memory_statistics(device);
    EXPECT_EQ(before.retained.committed_bytes, 2 * kSlabSize);
    EXPECT_EQ(before.workspace.bytes_committed, kSlabSize);
    EXPECT_EQ(before.workspace.reservation_count, 1u);
    IREE_ASSERT_OK(loom_serve_test_device_trim_and_drain(device, 0));
    const auto live = loom_serve_device_memory_statistics(device);
    EXPECT_EQ(live.retained.committed_bytes, 2 * kSlabSize);
    EXPECT_EQ(live.workspace.bytes_committed, kSlabSize);
    EXPECT_EQ(live.workspace.reservation_count, 1u);
    // A real consumer still reaches the allocation after an unreachable trim.
    CheckRetained(workspace, 0, 0x12345678);
    RoundTrip(workspace, 0, 0x87654321);
    CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0xABCDEF01);
    uint64_t completion = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_dealloca(execution, workspace, &completion));
    iree_hal_buffer_release(workspace);
    workspace = nullptr;
    IREE_ASSERT_OK(loom_serve_execution_drain(execution));
    IREE_ASSERT_OK(
        loom_serve_test_device_trim_and_drain(device, 2 * kSlabSize));
    EXPECT_EQ(
        loom_serve_device_memory_statistics(device).workspace.bytes_committed,
        kSlabSize);
    IREE_ASSERT_OK(loom_serve_test_device_trim_and_drain(device, kSlabSize));
    const auto idle = loom_serve_device_memory_statistics(device);
    EXPECT_EQ(idle.retained.committed_bytes, kSlabSize);
    EXPECT_EQ(idle.workspace.bytes_committed, 0u);
    EXPECT_EQ(idle.workspace.reservation_count, 0u);
    CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0xABCDEF01);
  }
}

TEST_F(DedicatedNativeWorkerMemoryTest,
       NativeAllocationPublicationDoesNotDoubleChargeAdmission) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), 2 * kSlabSize, 256,
      &statistics[1], &second));
  std::array<loom_serve_memory_operation_t*, 2> operations = {};
  {
    // Select this acquisition before holding its publication. Earlier native
    // operations may still free metadata after their backing has retired.
    loom_serve::testing::ScopedNativeHold gate(native,
                                               IREE_ALLOCATOR_COMMAND_CALLOC);
    const loom_serve_memory_range_t first_range = {first, 0, kSlabSize};
    loom_serve_memory_admission_t commit_admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &first_range, nullptr, 0, &commit_admission, &operations[0]));
    ASSERT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    gate.WaitEntered();
    EXPECT_EQ(loom_serve_memory_pool_statistics(pool).peak_bytes, 0u);
    // The source pool frees its acquisition capture after accounting native
    // backing and before publishing completion to the queue. Transfer the hold
    // atomically so that this worker cannot pass the publication boundary.
    gate.ReleaseAndArm(IREE_ALLOCATOR_COMMAND_FREE);
    gate.WaitEntered();
    EXPECT_FALSE(loom_serve_memory_operation_is_complete(operations[0]));
    const auto pending = loom_serve_memory_pool_statistics(pool);
    EXPECT_EQ(pending.peak_bytes, kSlabSize);
    EXPECT_EQ(pending.pending_bytes, kSlabSize);
    EXPECT_EQ(pending.committed_bytes, 0u);
    EXPECT_EQ(pending.releasing_bytes, 0u);
    loom_serve_memory_admission_t admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
        pool, {.cached_bytes = {2 * kSlabSize}}, &admission));
    EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    const loom_serve_memory_range_t second_range = {second, 0, 2 * kSlabSize};
    IREE_EXPECT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &second_range, nullptr, 0, &commit_admission, &operations[1]));
    EXPECT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    EXPECT_NE(operations[1], nullptr);
    EXPECT_EQ(loom_serve_memory_pool_statistics(pool).pending_bytes,
              3 * kSlabSize);
  }
  for (auto* operation : operations) {
    loom_serve_memory_operation_await(operation);
    IREE_EXPECT_OK(loom_serve_memory_operation_query_status(operation));
    loom_serve_memory_operation_free(operation);
  }
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  const auto completed = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(completed.pending_bytes, 0u);
  EXPECT_EQ(completed.committed_bytes, 3 * kSlabSize);
  EXPECT_EQ(completed.releasing_bytes, 0u);
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
  RoundTrip(loom_serve_virtual_buffer_handle(second), kSlabSize, 0x87654321);
}

TEST_F(MemoryTest, PendingCommitChargesBudgetAndLeavesPublishedStorageUsable) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking());
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), 3 * kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
  iree_hal_semaphore_t* ready = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready));
  loom_serve_memory_operation_t* operation = nullptr;
  {
    deferred_backing->defer = true;
    const loom_serve_memory_range_t ranges[] = {
        {first, kSlabSize, 2 * kSlabSize},
        {first, kSlabSize + 256, kSlabSize},
    };
    loom_serve_memory_admission_t commit_admission;
    IREE_EXPECT_OK(loom_serve_memory_pool_commit_async(
        pool, IREE_ARRAYSIZE(ranges), ranges, ready, 1, &commit_admission,
        &operation));
    EXPECT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    deferred_backing->WaitForAcquisition();
    EXPECT_FALSE(loom_serve_memory_operation_is_complete(operation));
    const auto pending = loom_serve_memory_pool_statistics(pool);
    EXPECT_EQ(pending.committed_bytes, kSlabSize);
    EXPECT_EQ(pending.pending_bytes, 2 * kSlabSize);
    EXPECT_EQ(statistics[0].pending_bytes, 2 * kSlabSize);
    const loom_serve_memory_range_t other = {second, 0, kSlabSize};
    bool admitted = false;
    IREE_EXPECT_OK(
        loom_serve_memory_pool_try_commit(pool, 1, &other, &admitted));
    EXPECT_FALSE(admitted);
    // Native work already backed by a different range remains executable while
    // commitment is held. No source-file readiness or queue ordering is mocked.
    CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
    deferred_backing->ResumeAcquisitions();
  }
  IREE_ASSERT_OK(iree_hal_semaphore_wait(ready, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_TRUE(loom_serve_memory_operation_is_complete(operation));
  IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool));
  const auto completed = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(completed.committed_bytes, 3 * kSlabSize);
  EXPECT_EQ(completed.pending_bytes, 0u);
  RoundTrip(loom_serve_virtual_buffer_handle(first), 2 * kSlabSize, 0x87654321);
  loom_serve_memory_operation_free(operation);
  operation = nullptr;
  // Already-backed requests signal readiness without another acquisition and
  // consume no additional budget.
  {
    deferred_backing->defer = true;
    const loom_serve_memory_range_t range = {first, kSlabSize, kSlabSize};
    loom_serve_memory_admission_t commit_admission;
    IREE_EXPECT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &range, ready, 2, &commit_admission, &operation));
    EXPECT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    EXPECT_EQ(operation, nullptr);
    uint64_t value = 0;
    IREE_EXPECT_OK(iree_hal_semaphore_query(ready, &value));
    EXPECT_EQ(value, 2u);
    EXPECT_EQ(deferred_backing->pending.count, 0u);
    deferred_backing->ResumeAcquisitions();
  }
  iree_hal_semaphore_release(ready);
}

TEST_F(MemoryTest, FailedReadinessDoesNotReleaseNativeOperationOwnership) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking());
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), 2 * kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
  iree_hal_semaphore_t* ready = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready));
  loom_serve_memory_operation_t* operation = nullptr;
  {
    deferred_backing->defer = true;
    const loom_serve_memory_range_t range = {first, 0, kSlabSize};
    loom_serve_memory_admission_t commit_admission;
    IREE_EXPECT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &range, ready, 1, &commit_admission, &operation));
    EXPECT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    deferred_backing->WaitForAcquisition();
    // A failed timeline can be observed before an accepted native producer has
    // even started. It is not permission to destroy its reservation or job.
    iree_hal_semaphore_fail(
        ready, iree_make_status(IREE_STATUS_CANCELLED, "cancelled timeline"));
    uint64_t value = 0;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                          iree_hal_semaphore_query(ready, &value));
    EXPECT_FALSE(loom_serve_memory_operation_is_complete(operation));
    EXPECT_EQ(loom_serve_memory_pool_statistics(pool).pending_bytes, kSlabSize);
    CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
    deferred_backing->ResumeAcquisitions();
  }
  loom_serve_memory_operation_await(operation);
  EXPECT_TRUE(loom_serve_memory_operation_is_complete(operation));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        loom_serve_memory_operation_query_status(operation));
  IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool));
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        iree_hal_semaphore_query(ready, &value));
  const auto memory = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(memory.committed_bytes, 2 * kSlabSize);
  EXPECT_EQ(memory.pending_bytes, 0u);
  loom_serve_memory_operation_free(operation);
  iree_hal_semaphore_release(ready);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, kSlabSize, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(first), kSlabSize, 0x87654321);
  CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
}

TEST_F(MemoryTest, DisjointMappingsFollowTheirReservationNotTheSharedPool) {
  mapping_queue.Initialize(loom_serve_device_transfer_queue(device));
  ASSERT_NO_FATAL_FAILURE(UseBacking(false, 4 * kSlabSize, &mapping_queue));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), 3 * kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[1], &second));
  // Captures borrow private semaphores from the operations kept alive below.
  auto& mapping_waits = mapping_queue.mapping_waits;
  auto& mapping_signals = mapping_queue.mapping_signals;
  mapping_queue.observe_mappings = true;
  auto release_gate = [](iree_hal_semaphore_t* semaphore) {
    IREE_EXPECT_OK(iree_hal_semaphore_signal(semaphore, 1, nullptr));
    iree_hal_semaphore_release(semaphore);
  };
  const std::pair<bool, bool> transitions[] = {
      {false, false}, {false, true}, {true, false}};
  for (auto [release_first, release_next] : transitions) {
    if (release_first || release_next) {
      IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
    }
    iree_hal_semaphore_t* semaphore = nullptr;
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
    std::unique_ptr<iree_hal_semaphore_t, decltype(release_gate)> gate(
        semaphore, release_gate);
    mapping_queue.gate_next_mapping = semaphore;
    mapping_waits.clear();
    mapping_signals.clear();
    loom_serve_memory_operation_t* operations[3] = {};
    loom_serve_memory_admission_t admission;
    if (release_first) {
      loom_serve_virtual_buffer_begin_trim(first);
      IREE_ASSERT_OK(loom_serve_memory_pool_trim_async(pool, 1, &first, nullptr,
                                                       0, &operations[0]));
    } else {
      const loom_serve_memory_range_t range = {first, kSlabSize, kSlabSize};
      IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
          pool, 1, &range, nullptr, 0, &admission, &operations[0]));
      ASSERT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    }
    if (release_next) {
      loom_serve_virtual_buffer_begin_trim(first);
      loom_serve_virtual_buffer_keep(first, kSlabSize, kSlabSize);
      IREE_ASSERT_OK(loom_serve_memory_pool_trim_async(pool, 1, &first, nullptr,
                                                       0, &operations[1]));
    } else {
      const loom_serve_memory_range_t next = {first, 2 * kSlabSize, kSlabSize};
      IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
          pool, 1, &next, nullptr, 0, &admission, &operations[1]));
      ASSERT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    }
    ASSERT_EQ(mapping_waits.size(), 2u);
    EXPECT_NE(std::find(mapping_waits[1].begin(), mapping_waits[1].end(),
                        mapping_signals[0]),
              mapping_waits[1].end());
    // The mapping owner emits dependencies even if physical allocation is
    // still pending. No scheduling luck or native mapping race is required.
    EXPECT_FALSE(loom_serve_memory_operation_is_complete(operations[0]));
    const loom_serve_memory_range_t independent = {second, 0, kSlabSize};
    IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &independent, nullptr, 0, &admission, &operations[2]));
    ASSERT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    ASSERT_EQ(mapping_waits.size(), 3u);
    EXPECT_EQ(mapping_waits[2].size(), 1u);  // Only its own allocation.
    loom_serve_memory_operation_await(operations[2]);
    IREE_ASSERT_OK(loom_serve_memory_operation_query_status(operations[2]));
    RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
    gate.reset();
    for (auto* operation : operations) {
      loom_serve_memory_operation_await(operation);
      IREE_EXPECT_OK(loom_serve_memory_operation_query_status(operation));
      loom_serve_memory_operation_free(operation);
    }
    if (!release_first) {
      RoundTrip(loom_serve_virtual_buffer_handle(first), kSlabSize, 0xabcdef01);
    }
    if (!release_next) {
      RoundTrip(loom_serve_virtual_buffer_handle(first), 2 * kSlabSize,
                0x87654321);
    }
    loom_serve_virtual_buffer_begin_trim(first);
    IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
    loom_serve_virtual_buffer_begin_trim(second);
    IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(second));
  }
  mapping_queue.observe_mappings = false;
}

TEST_F(MemoryTest, ArenaRangesShareMappingOrderButNotPhysicalOwnership) {
  mapping_queue.Initialize(loom_serve_device_transfer_queue(device));
  ASSERT_NO_FATAL_FAILURE(UseBacking(false, 3 * kSlabSize, &mapping_queue));
  const auto geometry = loom_serve_device_weight_geometry(device);
  IREE_ASSERT_OK(
      loom_serve_memory_arena_allocate(pool, geometry, 2 * kSlabSize, &arena));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate_in_arena(
      arena, kSlabSize, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate_in_arena(
      arena, kSlabSize, 256, &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, geometry, kSlabSize, 256, &statistics[2], &third));
  auto release_gate = [](iree_hal_semaphore_t* semaphore) {
    IREE_EXPECT_OK(iree_hal_semaphore_signal(semaphore, 1, nullptr));
    iree_hal_semaphore_release(semaphore);
  };
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  std::unique_ptr<iree_hal_semaphore_t, decltype(release_gate)> gate(
      semaphore, release_gate);
  mapping_queue.gate_next_mapping = semaphore;
  mapping_queue.observe_mappings = true;
  loom_serve_memory_operation_t* operations[3] = {};
  loom_serve_virtual_buffer_t* buffers[] = {first, second, third};
  for (size_t i = 0; i < 3; ++i) {
    const loom_serve_memory_range_t range = {buffers[i], 0, kSlabSize};
    loom_serve_memory_admission_t admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &range, nullptr, 0, &admission, &operations[i]));
    ASSERT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
  }
  const auto& waits = mapping_queue.mapping_waits;
  ASSERT_EQ(waits.size(), 3u);
  EXPECT_NE(std::find(waits[1].begin(), waits[1].end(),
                      mapping_queue.mapping_signals[0]),
            waits[1].end());
  EXPECT_EQ(waits[2].size(), 1u);
  loom_serve_memory_operation_await(operations[2]);
  IREE_ASSERT_OK(loom_serve_memory_operation_query_status(operations[2]));
  EXPECT_FALSE(loom_serve_memory_operation_is_complete(operations[0]));
  EXPECT_FALSE(loom_serve_memory_operation_is_complete(operations[1]));
  gate.reset();
  for (auto* operation : operations) {
    loom_serve_memory_operation_await(operation);
    IREE_EXPECT_OK(loom_serve_memory_operation_query_status(operation));
    loom_serve_memory_operation_free(operation);
  }
  mapping_queue.observe_mappings = false;
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
  loom_serve_virtual_buffer_begin_trim(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_free(first));
  first = nullptr;
  EXPECT_EQ(statistics[0].committed_bytes, 0u);
  EXPECT_EQ(statistics[1].committed_bytes, kSlabSize);
  CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
}

TEST_F(MemoryTest, TrimmedArenaRangeRecyclesWhilePeerMappingIsPending) {
  mapping_queue.Initialize(loom_serve_device_transfer_queue(device));
  ASSERT_NO_FATAL_FAILURE(UseBacking(false, 2 * kSlabSize, &mapping_queue));
  const auto geometry = loom_serve_device_weight_geometry(device);
  IREE_ASSERT_OK(
      loom_serve_memory_arena_allocate(pool, geometry, 2 * kSlabSize, &arena));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate_in_arena(
      arena, kSlabSize, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate_in_arena(
      arena, kSlabSize, 256, &statistics[1], &second));
  const auto original =
      iree_hal_buffer_virtual_view(loom_serve_virtual_buffer_handle(first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  loom_serve_virtual_buffer_begin_trim(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  EXPECT_EQ(statistics[0].committed_bytes, 0u);

  auto release_gate = [](iree_hal_semaphore_t* semaphore) {
    IREE_EXPECT_OK(iree_hal_semaphore_signal(semaphore, 1, nullptr));
    iree_hal_semaphore_release(semaphore);
  };
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  std::unique_ptr<iree_hal_semaphore_t, decltype(release_gate)> gate(
      semaphore, release_gate);
  mapping_queue.gate_next_mapping = semaphore;
  const loom_serve_memory_range_t range = {second, 0, kSlabSize};
  loom_serve_memory_admission_t admission;
  loom_serve_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
      pool, 1, &range, nullptr, 0, &admission, &operation));
  ASSERT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
  EXPECT_FALSE(loom_serve_memory_operation_is_complete(operation));

  // Returning an already trimmed lease must not join the shared target's
  // newer frontier. That mapping belongs to the still-live peer alone.
  IREE_ASSERT_OK(loom_serve_virtual_buffer_free(first));
  first = nullptr;
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate_in_arena(
      arena, kSlabSize, 256, &statistics[0], &first));
  const auto replacement =
      iree_hal_buffer_virtual_view(loom_serve_virtual_buffer_handle(first));
  EXPECT_EQ(replacement.root, original.root);
  EXPECT_EQ(replacement.byte_offset, original.byte_offset);
  EXPECT_FALSE(loom_serve_memory_operation_is_complete(operation));

  gate.reset();
  loom_serve_memory_operation_await(operation);
  IREE_ASSERT_OK(loom_serve_memory_operation_query_status(operation));
  loom_serve_memory_operation_free(operation);
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x87654321);
  CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
}

TEST_F(MemoryTest, FailedArenaEditRequiresOwnerDetachBeforeRangeRelease) {
  mapping_queue.Initialize(loom_serve_device_transfer_queue(device));
  ASSERT_NO_FATAL_FAILURE(UseBacking(false, 3 * kSlabSize, &mapping_queue));
  const auto geometry = loom_serve_device_weight_geometry(device);
  IREE_ASSERT_OK(
      loom_serve_memory_arena_allocate(pool, geometry, 3 * kSlabSize, &arena));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate_in_arena(
      arena, kSlabSize, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate_in_arena(
      arena, kSlabSize, 256, &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
  mapping_queue.fail_next_map = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  EXPECT_EQ(statistics[0].committed_bytes, kSlabSize);
  EXPECT_EQ(statistics[1].committed_bytes, kSlabSize);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_virtual_buffer_free(first));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_virtual_buffer_free(second));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_virtual_buffer_commit(second, 0, kSlabSize));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_virtual_buffer_allocate_in_arena(
                            arena, kSlabSize, 256, &statistics[2], &third));
  EXPECT_EQ(third, nullptr);
  // This injected failure rejected the edit before native mutation. The
  // peer's original content proves neither failed child cleanup detached it;
  // executing after an actual partial native failure is not a valid contract.
  CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
  IREE_ASSERT_OK(loom_serve_execution_drain(execution));
  IREE_ASSERT_OK(loom_serve_memory_arena_detach(arena));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_free(first));
  first = nullptr;
  IREE_ASSERT_OK(loom_serve_virtual_buffer_free(second));
  second = nullptr;
  EXPECT_EQ(statistics[0].committed_bytes, 0u);
  EXPECT_EQ(statistics[1].committed_bytes, 0u);
}

TEST_F(MemoryTest, NativeAllocationFailurePreservesWarmWorkAndAllowsRetry) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);

  native.FailNextAllocation(IREE_ALLOCATOR_COMMAND_CALLOC,
                            /*native_only=*/true);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_virtual_buffer_commit(second, 0, kSlabSize));
  IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool));
  EXPECT_EQ(statistics[1].pending_bytes, 0u);
  EXPECT_EQ(statistics[1].committed_bytes, 0u);
  CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);

  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x87654321);
  CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
}

TEST_F(MemoryTest, FailedMappingKeepsSourcesUntilCheckedDetach) {
  constexpr uint64_t kMappingBytes = 2 * kSlabSize;
  mapping_queue.Initialize(loom_serve_device_transfer_queue(device));
  ASSERT_NO_FATAL_FAILURE(
      UseDeferredBacking(kMappingBytes + kSlabSize, &mapping_queue));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kMappingBytes, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);

  // Mapping rejection cannot retire the allocation already accepted by the
  // native queue. Keep it pending until the failed mapping has been observed.
  deferred_backing->defer = true;
  mapping_queue.fail_next_map = true;
  const loom_serve_memory_range_t range = {first, 0, kMappingBytes};
  loom_serve_memory_admission_t commit_admission;
  loom_serve_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
      pool, 1, &range, nullptr, 0, &commit_admission, &operation));
  ASSERT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
  deferred_backing->WaitForAcquisition();
  EXPECT_FALSE(mapping_queue.fail_next_map);
  EXPECT_FALSE(loom_serve_memory_operation_is_complete(operation));
  deferred_backing->ResumeAcquisitions();
  loom_serve_memory_operation_await(operation);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_memory_operation_query_status(operation));
  loom_serve_memory_operation_free(operation);
  IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool));
  EXPECT_EQ(statistics[0].pending_bytes, 0u);
  EXPECT_EQ(statistics[0].committed_bytes, kMappingBytes);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);

  IREE_ASSERT_OK(loom_serve_virtual_buffer_free(first));
  first = nullptr;
  loom_serve_memory_pool_trim(pool, 0);
  IREE_EXPECT_OK(loom_serve_memory_pool_await(pool));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, kSlabSize);
  CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
}

}  // namespace
