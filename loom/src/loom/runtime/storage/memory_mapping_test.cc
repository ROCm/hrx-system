// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/memory_test_fixture.h"

namespace {

using CompletedOperation =
    std::unique_ptr<loom_runtime_memory_operation_t,
                    decltype(&loom_runtime_memory_operation_free)>;

CompletedOperation AwaitOperation(loom_runtime_memory_operation_t* operation) {
  loom_runtime_memory_operation_await(operation);
  return CompletedOperation(operation, loom_runtime_memory_operation_free);
}

class MemoryMappingTest : public loom_runtime::testing::memory::MemoryTest {
 protected:
  void SetUpPool() override {
    mapping_queue.Initialize(loom_runtime_device_transfer_queue(device));
    UseBacking(false, 3 * kSlabSize, &mapping_queue);
    mapping_queue.observe_mappings = true;
  }
};

TEST_F(MemoryMappingTest, PendingMappingDefersOnlyIntersectingCommitments) {
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.mapping_size;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, 4 * page, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(pool, geometry, page, 256,
                                                      &statistics[1], &second));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, page));
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);

  for (bool unmap : {false, true}) {
    SCOPED_TRACE(unmap ? "pending unmap" : "pending bind");
    const auto retire = [](loom_runtime_memory_operation_t* operation) {
      loom_runtime_memory_operation_await(operation);
      loom_runtime_memory_operation_free(operation);
    };
    // Destroy the gate before joining an accepted operation on assertion exit.
    std::unique_ptr<loom_runtime_memory_operation_t, decltype(retire)> pending(
        nullptr, retire);
    const auto release_gate = [](iree_hal_semaphore_t* semaphore) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(semaphore, 1, nullptr));
      iree_hal_semaphore_release(semaphore);
    };
    iree_hal_semaphore_t* semaphore = nullptr;
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        loom_runtime_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
    std::unique_ptr<iree_hal_semaphore_t, decltype(release_gate)> gate(
        semaphore, release_gate);
    mapping_queue.ClearMappings();
    mapping_queue.gate_next_mapping = semaphore;
    const loom_runtime_memory_range_t changing = {first, 2 * page, page};
    loom_runtime_memory_admission_t admission;
    loom_runtime_memory_operation_t* operation = nullptr;
    if (unmap) {
      IREE_ASSERT_OK(loom_runtime_memory_pool_trim_async(
          pool, 1, &changing, nullptr, 0, &operation));
    } else {
      IREE_ASSERT_OK(loom_runtime_memory_pool_commit_async(
          pool, 1, &changing, nullptr, 0, &admission, &operation));
      EXPECT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
    }
    pending.reset(operation);
    mapping_queue.WaitForMappings(1);
    EXPECT_FALSE(loom_runtime_memory_operation_is_complete(pending.get()));

    // Ready data in the same reservation neither waits nor submits a mapping.
    const loom_runtime_memory_range_t ready = {first, 0, page};
    IREE_ASSERT_OK(loom_runtime_memory_pool_commit_async(
        pool, 1, &ready, nullptr, 0, &admission, &operation));
    EXPECT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
    EXPECT_EQ(operation, nullptr);
    CheckRetained(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);
    EXPECT_FALSE(loom_runtime_memory_operation_is_complete(pending.get()));

    // A partially selected prefix spans both descriptors and buffers. Overlap
    // must cancel it before pressure, allocation, or native submission.
    const loom_runtime_memory_range_t ranges[] = {{second, 0, page},
                                                  {first, page, 3 * page}};
    const auto before = loom_runtime_memory_pool_statistics(pool);
    loom_runtime_memory_commit_t prepared = {};
    IREE_ASSERT_OK(loom_runtime_memory_pool_prepare_commit(
        pool, IREE_ARRAYSIZE(ranges), ranges, &admission, &prepared));
    EXPECT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_DEFERRED);
    EXPECT_EQ(prepared.ranges, nullptr);
    EXPECT_EQ(prepared.mapping_count, 0u);
    IREE_ASSERT_OK(loom_runtime_memory_pool_commit_async(
        pool, IREE_ARRAYSIZE(ranges), ranges, nullptr, 0, &admission,
        &operation));
    EXPECT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_DEFERRED);
    EXPECT_EQ(operation, nullptr);
    const auto after = loom_runtime_memory_pool_statistics(pool);
    EXPECT_EQ(after.pending_bytes, before.pending_bytes);
    EXPECT_EQ(after.committed_bytes, before.committed_bytes);
    EXPECT_EQ(after.cached_bytes, before.cached_bytes);
    EXPECT_EQ(statistics[1].pending_bytes, 0u);
    EXPECT_EQ(statistics[1].committed_bytes, 0u);

    gate.reset();
    loom_runtime_memory_operation_await(pending.get());
    IREE_ASSERT_OK(loom_runtime_memory_operation_query_status(pending.get()));
    pending.reset();
    IREE_ASSERT_OK(loom_runtime_memory_pool_prepare_commit(
        pool, IREE_ARRAYSIZE(ranges), ranges, &admission, &prepared));
    ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
    EXPECT_EQ(prepared.mapping_count, unmap ? 4u : 3u);
    IREE_ASSERT_OK(loom_runtime_memory_pool_commit_prepared_async(
        pool, &prepared, {}, nullptr, 0, &admission, &operation));
    ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
    const auto completed = AwaitOperation(operation);
    IREE_ASSERT_OK(loom_runtime_memory_operation_query_status(operation));
    RoundTrip(loom_runtime_virtual_buffer_handle(first), page, 0x87654321);
    RoundTrip(loom_runtime_virtual_buffer_handle(second), 0, 0xabcdef01);
    CheckRetained(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);

    // Leave the changing page bound for the next iteration's unmap.
    const loom_runtime_memory_range_t releases[] = {
        {first, page, page}, {first, 3 * page, page}, {second, 0, page}};
    IREE_ASSERT_OK(loom_runtime_memory_pool_trim_async(
        pool, IREE_ARRAYSIZE(releases), releases, nullptr, 0, &operation));
    const auto trimmed = AwaitOperation(operation);
    IREE_ASSERT_OK(loom_runtime_memory_operation_query_status(operation));
  }
}

enum class MappingSize { kNative, kThreeNativePages, kParent };

class MemoryGranularityTest
    : public loom_runtime::testing::memory::MemoryTest,
      public ::testing::WithParamInterface<MappingSize> {
 protected:
  uint64_t mapping_size() const {
    const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
    switch (GetParam()) {
      case MappingSize::kNative:
        return geometry.minimum_size;
      case MappingSize::kThreeNativePages:
        return 3 * geometry.minimum_size;
      case MappingSize::kParent:
        return kSlabSize;
    }
    IREE_CHECK_UNREACHABLE("unknown mapping size");
  }

  void SetUpPool() override {
    UseBacking(false, 3 * kSlabSize, nullptr, false, mapping_size());
  }
};

TEST_P(MemoryGranularityTest, MixedMappingSizesPreserveNativePacking) {
  const auto native = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = native.minimum_size;
  RecordProperty("native_page_bytes", static_cast<int>(page));
  RecordProperty("mapping_bytes", static_cast<int>(mapping_size()));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, native, 3 * page, native.granularity, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, 3 * page));
  // Workspace uses native pages even when state has larger replacement units.
  // Three small mappings must occupy one physical parent, not three parents.
  ASSERT_EQ(loom_runtime_memory_pool_retained_bytes(pool), kSlabSize);
  ASSERT_EQ(loom_runtime_memory_pool_statistics(pool).cached_bytes,
            kSlabSize - 3 * page);
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 2 * page, 0x87654321);

  auto coarse = native;
  coarse.mapping_size = mapping_size();
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, coarse, coarse.mapping_size + page, native.granularity,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(second, 0, page));
  IREE_ASSERT_OK(
      loom_runtime_virtual_buffer_commit(second, coarse.mapping_size, page));
  EXPECT_EQ(statistics[1].committed_bytes, coarse.mapping_size + page);
  RoundTrip(loom_runtime_virtual_buffer_handle(second), 0, 0xaabbccdd);
  RoundTrip(loom_runtime_virtual_buffer_handle(second), coarse.mapping_size,
            0x11223344);

  // Returning the large mapping preserves its short tail and native neighbors.
  const loom_runtime_memory_range_t release = {second, 0, coarse.mapping_size};
  loom_runtime_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_runtime_memory_pool_trim_async(pool, 1, &release, nullptr,
                                                     0, &operation));
  const auto completed = AwaitOperation(operation);
  IREE_ASSERT_OK(loom_runtime_memory_operation_query_status(operation));
  // Dedicated extents return to the source after their unmap completes.
  IREE_ASSERT_OK(loom_runtime_memory_pool_await(pool));
  CheckRetained(loom_runtime_virtual_buffer_handle(second), coarse.mapping_size,
                0x11223344);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), 2 * page,
                0x87654321);

  // A new small mapping reuses native slots without acquiring another parent.
  const uint64_t retained = loom_runtime_memory_pool_retained_bytes(pool);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, native, page, native.granularity, &statistics[2], &third));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(third, 0, page));
  EXPECT_EQ(loom_runtime_memory_pool_retained_bytes(pool), retained);
  RoundTrip(loom_runtime_virtual_buffer_handle(third), 0, 0x55667788);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(second, 0, page));
  RoundTrip(loom_runtime_virtual_buffer_handle(second), 0, 0xabcdef01);
  CheckRetained(loom_runtime_virtual_buffer_handle(third), 0, 0x55667788);
  CheckRetained(loom_runtime_virtual_buffer_handle(second), coarse.mapping_size,
                0x11223344);
}

INSTANTIATE_TEST_SUITE_P(MappingSizes, MemoryGranularityTest,
                         ::testing::Values(MappingSize::kNative,
                                           MappingSize::kThreeNativePages,
                                           MappingSize::kParent));

TEST_F(MemoryMappingTest, ColdGrowthPacksAlignedRunsBeforeExistingHoles) {
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.minimum_size;
  const uint64_t parent = geometry.backing_size;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, page, geometry.granularity, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, page));
  const auto initial = mapping_queue.WaitForMappings(1)[0].updates[0];
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);
  mapping_queue.ClearMappings();

  // The accepted cohort needs exactly one new parent. Starting there keeps
  // its full prefix aligned; the final page still reuses the older hole.
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, parent + page, geometry.granularity, &statistics[1],
      &second));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(second, 0, parent + page));
  const auto mappings = mapping_queue.WaitForMappings(1)[0].updates;
  ASSERT_EQ(mappings.size(), 2u);
  EXPECT_EQ(mappings[0].target_range.byte_offset, 0u);
  EXPECT_EQ(mappings[0].target_range.byte_length, parent);
  EXPECT_EQ(mappings[0].source_offset, 0u);
  EXPECT_NE(mappings[0].source, initial.source);
  EXPECT_EQ(mappings[1].target_range.byte_offset, parent);
  EXPECT_EQ(mappings[1].target_range.byte_length, page);
  EXPECT_EQ(mappings[1].source, initial.source);
  EXPECT_EQ(mappings[1].source_offset, page);
  EXPECT_EQ(loom_runtime_memory_pool_retained_bytes(pool), 2 * parent);
  EXPECT_EQ(loom_runtime_memory_pool_statistics(pool).cached_bytes,
            parent - 2 * page);
  RoundTrip(loom_runtime_virtual_buffer_handle(second), 0, 0xaabbccdd);
  RoundTrip(loom_runtime_virtual_buffer_handle(second), parent, 0x87654321);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);

  IREE_ASSERT_OK(Trim(second));
  loom_runtime_memory_pool_trim(pool, 0);
  IREE_ASSERT_OK(loom_runtime_memory_pool_await(pool));
  EXPECT_EQ(loom_runtime_memory_pool_statistics(pool).committed_bytes, parent);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);
}

TEST_F(MemoryMappingTest, PackedClaimStopsAtLivePagesInsideOneParent) {
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.mapping_size;
  const uint64_t parent = geometry.backing_size;
  ASSERT_GE(parent, 8 * page);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, parent, geometry.granularity, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, parent));
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 3 * page, 0x12345678);
  IREE_ASSERT_OK(Trim(first, page, 2 * page));
  IREE_ASSERT_OK(Trim(first, 4 * page, 3 * page));
  mapping_queue.ClearMappings();

  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, 5 * page, geometry.granularity, &statistics[1], &second));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(second, 0, 5 * page));
  const auto mappings = mapping_queue.WaitForMappings(1)[0].updates;
  ASSERT_EQ(mappings.size(), 2u);
  EXPECT_EQ(mappings[0].source, mappings[1].source);
  EXPECT_EQ(mappings[0].target_range.byte_offset, 0u);
  EXPECT_EQ(mappings[0].target_range.byte_length, 2 * page);
  EXPECT_EQ(mappings[0].source_offset, page);
  EXPECT_EQ(mappings[1].target_range.byte_offset, 2 * page);
  EXPECT_EQ(mappings[1].target_range.byte_length, 3 * page);
  EXPECT_EQ(mappings[1].source_offset, 4 * page);
  RoundTrip(loom_runtime_virtual_buffer_handle(second), 2 * page, 0xaabbccdd);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), 3 * page,
                0x12345678);
}

TEST_F(MemoryMappingTest, ContiguousRunKeepsIndependentPageOwnership) {
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.mapping_size;
  ASSERT_GE(geometry.backing_size, 4 * page);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, 4 * page, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, 2 * page, 256, &statistics[1], &second));
  const loom_runtime_memory_range_t range = {first, 0, 4 * page};
  loom_runtime_memory_operation_t* operation = nullptr;
  loom_runtime_memory_admission_t admission;
  IREE_ASSERT_OK(loom_runtime_memory_pool_commit_async(
      pool, 1, &range, nullptr, 0, &admission, &operation));
  ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
  const auto bound = AwaitOperation(operation);
  IREE_ASSERT_OK(loom_runtime_memory_operation_query_status(operation));
  auto observations = mapping_queue.WaitForMappings(1);
  ASSERT_EQ(observations[0].updates.size(), 1u);
  const auto original = observations[0].updates[0];
  EXPECT_EQ(original.operation, IREE_HAL_MEMORY_MAPPING_OPERATION_BIND);
  EXPECT_EQ(original.target_range.byte_offset, 0u);
  EXPECT_EQ(original.target_range.byte_length, 4 * page);
  ASSERT_EQ(loom_runtime_memory_operation_mapping_count(operation), 1u);
  iree_hal_buffer_binding_t changed;
  EXPECT_EQ(loom_runtime_memory_operation_mapping(operation, 0, &changed),
            IREE_HAL_MEMORY_MAPPING_OPERATION_BIND);
  EXPECT_EQ(changed.buffer, original.target);
  EXPECT_EQ(changed.offset, original.target_range.byte_offset);
  EXPECT_EQ(changed.length, original.target_range.byte_length);
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 0, 0x11223344);
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 2 * page, 0x55667788);

  // Releasing alternate pages splits the unmap without widening ownership.
  const loom_runtime_memory_range_t releases[] = {{first, page, page},
                                                  {first, 3 * page, page}};
  // Repeating an already-backed commitment has no physical changes.
  IREE_ASSERT_OK(loom_runtime_memory_pool_commit_async(
      pool, 1, &range, nullptr, 0, &admission, &operation));
  ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
  EXPECT_EQ(operation, nullptr);
  EXPECT_EQ(loom_runtime_memory_operation_mapping_count(operation), 0u);
  IREE_ASSERT_OK(loom_runtime_memory_pool_trim_async(
      pool, IREE_ARRAYSIZE(releases), releases, nullptr, 0, &operation));
  const auto completed = AwaitOperation(operation);
  IREE_ASSERT_OK(loom_runtime_memory_operation_query_status(operation));
  observations = mapping_queue.WaitForMappings(2);
  ASSERT_EQ(observations[1].updates.size(), 2u);
  ASSERT_EQ(loom_runtime_memory_operation_mapping_count(operation), 2u);
  for (size_t i = 0; i < 2; ++i) {
    const auto& update = observations[1].updates[i];
    EXPECT_EQ(update.operation, IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND);
    EXPECT_EQ(update.target_range.byte_offset, (2 * i + 1) * page);
    EXPECT_EQ(update.target_range.byte_length, page);
    EXPECT_EQ(update.previous_source, original.source);
    EXPECT_EQ(update.previous_source_offset,
              original.source_offset + (2 * i + 1) * page);
    EXPECT_EQ(loom_runtime_memory_operation_mapping(operation, i, &changed),
              IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND);
    EXPECT_EQ(changed.buffer, update.target);
    EXPECT_EQ(changed.offset, update.target_range.byte_offset);
    EXPECT_EQ(changed.length, update.target_range.byte_length);
  }
  EXPECT_EQ(statistics[0].committed_bytes, 2 * page);

  // A contiguous destination reusing physical holes still needs two runs.
  const auto retained = loom_runtime_memory_pool_retained_bytes(pool);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(second, 0, 2 * page));
  observations = mapping_queue.WaitForMappings(3);
  ASSERT_EQ(observations[2].updates.size(), 2u);
  for (size_t i = 0; i < 2; ++i) {
    const auto& update = observations[2].updates[i];
    EXPECT_EQ(update.target_range.byte_offset, i * page);
    EXPECT_EQ(update.target_range.byte_length, page);
    EXPECT_EQ(update.source, original.source);
    EXPECT_EQ(update.source_offset,
              original.source_offset + (2 * i + 1) * page);
    RoundTrip(loom_runtime_virtual_buffer_handle(second), i * page,
              0xaabbccdd + i);
  }
  EXPECT_EQ(loom_runtime_memory_pool_retained_bytes(pool), retained);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), 0, 0x11223344);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), 2 * page,
                0x55667788);
}

TEST_F(MemoryMappingTest, RunsStopAtSourceParentsAndIndependentReservations) {
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.mapping_size;
  const uint64_t parent = geometry.backing_size;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, parent + page, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(pool, geometry, page, 256,
                                                      &statistics[1], &second));
  const loom_runtime_memory_range_t ranges[] = {{first, 0, parent + page},
                                                {second, 0, page}};
  bool admitted = false;
  IREE_ASSERT_OK(loom_runtime_memory_pool_try_commit(
      pool, IREE_ARRAYSIZE(ranges), ranges, &admitted));
  ASSERT_TRUE(admitted);
  const auto observations = mapping_queue.WaitForMappings(1);
  ASSERT_EQ(observations[0].updates.size(), 3u);
  const auto& updates = observations[0].updates;
  EXPECT_EQ(updates[0].target, loom_runtime_virtual_buffer_handle(first));
  EXPECT_EQ(updates[0].target_range.byte_length, parent);
  EXPECT_EQ(updates[1].target, updates[0].target);
  EXPECT_EQ(updates[1].target_range.byte_offset, parent);
  EXPECT_EQ(updates[1].target_range.byte_length, page);
  EXPECT_NE(updates[0].source, updates[1].source);
  EXPECT_EQ(updates[2].source, updates[1].source);
  EXPECT_EQ(updates[2].source_offset, updates[1].source_offset + page);
  EXPECT_EQ(updates[2].target, loom_runtime_virtual_buffer_handle(second));
  EXPECT_EQ(updates[2].target_range.byte_length, page);
  RoundTrip(loom_runtime_virtual_buffer_handle(first), parent, 0x12345678);
  RoundTrip(loom_runtime_virtual_buffer_handle(second), 0, 0x87654321);

  IREE_ASSERT_OK(Trim(first));
  const auto trimmed = mapping_queue.WaitForMappings(2);
  ASSERT_EQ(trimmed[1].updates.size(), 2u);
  EXPECT_EQ(trimmed[1].updates[0].target_range.byte_length, parent);
  EXPECT_EQ(trimmed[1].updates[1].target_range.byte_length, page);
  EXPECT_EQ(statistics[0].committed_bytes, 0u);
  EXPECT_EQ(statistics[1].committed_bytes, page);
  CheckRetained(loom_runtime_virtual_buffer_handle(second), 0, 0x87654321);
}

TEST_F(MemoryMappingTest, SourceIoKeepsCallerOrderAndOverlappingRanges) {
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.mapping_size;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, 4 * page, 256, &statistics[0], &first));
  const loom_runtime_memory_range_t ranges[] = {{first, 2 * page, 2 * page},
                                                {first, 0, 2 * page},
                                                {first, 2 * page, page}};
  loom_runtime_memory_operation_t* operation = nullptr;
  loom_runtime_memory_admission_t admission;
  IREE_ASSERT_OK(loom_runtime_memory_pool_commit_async(
      pool, IREE_ARRAYSIZE(ranges), ranges, nullptr, 0, &admission,
      &operation));
  ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
  const auto completed = AwaitOperation(operation);
  IREE_ASSERT_OK(loom_runtime_memory_operation_query_status(operation));
  for (size_t i = 0; i < 4; ++i) {
    const auto input = loom_runtime_memory_operation_backing(operation, i);
    const auto published =
        loom_runtime_virtual_buffer_backing(first, (i + 2) % 4);
    EXPECT_EQ(input.buffer, published.buffer);
    EXPECT_EQ(input.offset, published.offset);
    EXPECT_EQ(input.length, published.length);
  }
  const auto observations = mapping_queue.WaitForMappings(1);
  ASSERT_EQ(observations[0].updates.size(), 2u);
  EXPECT_EQ(observations[0].updates[0].target_range.byte_offset, 2 * page);
  EXPECT_EQ(observations[0].updates[1].target_range.byte_offset, 0u);
  for (const auto& update : observations[0].updates) {
    EXPECT_EQ(update.target_range.byte_length, 2 * page);
    RoundTrip(loom_runtime_virtual_buffer_handle(first),
              update.target_range.byte_offset, 0x12345678);
  }
  EXPECT_EQ(statistics[0].committed_bytes, 4 * page);
}

TEST_F(MemoryMappingTest, GrowthRebindsRunsWithoutCopyingOrFillingHoles) {
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.mapping_size;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, 5 * page, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, 2 * page));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 3 * page, 2 * page));
  RoundTrip(loom_runtime_virtual_buffer_handle(first), page, 0x12345678);
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 4 * page, 0x87654321);
  const auto retained = loom_runtime_memory_pool_retained_bytes(pool);
  auto* previous = loom_runtime_virtual_buffer_handle(first);
  iree_hal_buffer_retain(previous);
  const std::unique_ptr<iree_hal_buffer_t, decltype(&iree_hal_buffer_release)>
      previous_owner(previous, iree_hal_buffer_release);
  loom_runtime_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_grow_async(
      first, 9 * page, nullptr, 0, &operation));
  const auto completed = AwaitOperation(operation);
  IREE_ASSERT_OK(loom_runtime_memory_operation_query_status(operation));
  // Rebinding the same storage into a larger address reservation preserves
  // logical backing; its retired old target is not a placement change.
  EXPECT_EQ(loom_runtime_memory_operation_mapping_count(operation), 0u);
  const auto observations = mapping_queue.WaitForMappings(4);
  const auto& attached = observations[2].updates;
  const auto& detached = observations[3].updates;
  ASSERT_EQ(attached.size(), 2u);
  ASSERT_EQ(detached.size(), 2u);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(attached[i].target, loom_runtime_virtual_buffer_handle(first));
    EXPECT_EQ(attached[i].target_range.byte_offset, i * 3 * page);
    EXPECT_EQ(attached[i].target_range.byte_length, 2 * page);
    EXPECT_EQ(detached[i].target, previous);
    EXPECT_EQ(detached[i].target_range.byte_offset,
              attached[i].target_range.byte_offset);
    EXPECT_EQ(detached[i].target_range.byte_length, 2 * page);
    EXPECT_EQ(detached[i].previous_source, attached[i].source);
    EXPECT_EQ(detached[i].previous_source_offset, attached[i].source_offset);
  }
  EXPECT_EQ(statistics[0].committed_bytes, 4 * page);
  EXPECT_EQ(loom_runtime_memory_pool_retained_bytes(pool), retained);
  EXPECT_FALSE(loom_runtime_virtual_buffer_is_backed(first, 2 * page, page));
  CheckRetained(loom_runtime_virtual_buffer_handle(first), page, 0x12345678);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), 4 * page,
                0x87654321);
}

TEST_F(MemoryMappingTest, DeferredSourcesPublishExactTailBeforeAdjacentState) {
  ASSERT_NO_FATAL_FAILURE(
      UseDeferredBacking(3 * kSlabSize, &mapping_queue, true));
  const auto bulk = loom_runtime_pool_weight_geometry(storage_owner);
  const auto state = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t tail = state.mapping_size;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, bulk, bulk.mapping_size + tail, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, state, 2 * tail, 256, &statistics[1], &second));
  const loom_runtime_memory_range_t ranges[] = {
      {first, 0, bulk.mapping_size + tail}, {second, 0, 2 * tail}};
  deferred_backing->defer = true;
  loom_runtime_memory_operation_t* operation = nullptr;
  loom_runtime_memory_admission_t admission;
  IREE_ASSERT_OK(loom_runtime_memory_pool_commit_async(
      pool, IREE_ARRAYSIZE(ranges), ranges, nullptr, 0, &admission,
      &operation));
  ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
  deferred_backing->WaitForAcquisition();
  EXPECT_FALSE(loom_runtime_memory_operation_is_complete(operation));
  deferred_backing->ResumeAcquisitions();
  const auto completed = AwaitOperation(operation);
  IREE_ASSERT_OK(loom_runtime_memory_operation_query_status(operation));
  const auto observations = mapping_queue.WaitForMappings(1);
  ASSERT_EQ(observations[0].updates.size(), 3u);
  const auto& updates = observations[0].updates;
  EXPECT_EQ(updates[0].target_range.byte_length, bulk.mapping_size);
  EXPECT_EQ(updates[1].target_range.byte_offset, bulk.mapping_size);
  EXPECT_EQ(updates[1].target_range.byte_length, tail);
  EXPECT_EQ(updates[2].target_range.byte_length, 2 * tail);
  EXPECT_EQ(updates[1].source, updates[2].source);
  EXPECT_EQ(updates[1].source_offset + tail, updates[2].source_offset);
  RoundTrip(loom_runtime_virtual_buffer_handle(first), bulk.mapping_size,
            0x12345678);
  RoundTrip(loom_runtime_virtual_buffer_handle(second), tail, 0x87654321);
}

TEST_F(MemoryMappingTest, FailedRunDetachRetainsEveryPageForCheckedCleanup) {
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.mapping_size;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, 3 * page, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(pool, geometry, page, 256,
                                                      &statistics[1], &second));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, 3 * page));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(second, 0, page));
  RoundTrip(loom_runtime_virtual_buffer_handle(second), 0, 0x12345678);

  mapping_queue.fail_next_map = true;
  loom_runtime_memory_operation_t* operation = nullptr;
  const loom_runtime_memory_range_t release = {first, 0, 3 * page};
  IREE_ASSERT_OK(loom_runtime_memory_pool_trim_async(pool, 1, &release, nullptr,
                                                     0, &operation));
  const auto completed = AwaitOperation(operation);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_runtime_memory_operation_query_status(operation));
  const auto observations = mapping_queue.WaitForMappings(3);
  ASSERT_EQ(observations[2].updates.size(), 1u);
  EXPECT_EQ(observations[2].updates[0].target_range.byte_length, 3 * page);
  EXPECT_EQ(statistics[0].committed_bytes, 3 * page);
  EXPECT_EQ(statistics[0].releasing_bytes, 0u);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_free(first));
  first = nullptr;
  EXPECT_EQ(statistics[0].committed_bytes, 0u);
  EXPECT_EQ(statistics[1].committed_bytes, page);
  CheckRetained(loom_runtime_virtual_buffer_handle(second), 0, 0x12345678);
}

TEST_F(MemoryMappingTest, WholeAndPackedClaimsShareDistinctParentClasses) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking(2 * kSlabSize, &mapping_queue,
                                             /*distinct_classes=*/true));
  const auto bulk = loom_runtime_pool_weight_geometry(storage_owner);
  const auto packed = loom_runtime_pool_state_geometry(storage_owner);
  ASSERT_LT(packed.backing_size, bulk.mapping_size);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, bulk, 2 * kSlabSize, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, bulk, kSlabSize, 256, &statistics[1], &second));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, packed, kSlabSize / 2, 256, &statistics[2], &third));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, 2 * kSlabSize));

  IREE_ASSERT_OK(Trim(first));
  iree_hal_pool_stats_t before;
  iree_hal_pool_query_stats(deferred_backing, &before);

  // Whole-parent claims must be assigned before sharing the other parent's
  // pages. Admission and capture agree without another native allocation.
  const loom_runtime_memory_range_t ranges[] = {{third, 0, kSlabSize / 2},
                                                {second, 0, kSlabSize}};
  bool admitted = false;
  IREE_ASSERT_OK(loom_runtime_memory_pool_try_commit(
      pool, IREE_ARRAYSIZE(ranges), ranges, &admitted));
  ASSERT_TRUE(admitted);
  const auto observations = mapping_queue.WaitForMappings(3);
  ASSERT_EQ(observations[2].updates.size(), 2u);
  iree_hal_pool_stats_t after;
  iree_hal_pool_query_stats(deferred_backing, &after);
  EXPECT_EQ(after.reserve_count, before.reserve_count);
  EXPECT_EQ(loom_runtime_memory_pool_statistics(pool).committed_bytes,
            2 * kSlabSize);
  EXPECT_EQ(loom_runtime_memory_pool_statistics(pool).cached_bytes,
            kSlabSize / 2);
  RoundTrip(loom_runtime_virtual_buffer_handle(second), 0, 0x12345678);
  RoundTrip(loom_runtime_virtual_buffer_handle(third), 0, 0x87654321);

  IREE_ASSERT_OK(Trim(second));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, kSlabSize));
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 0, 0xabcdef01);
  CheckRetained(loom_runtime_virtual_buffer_handle(third), 0, 0x87654321);
  iree_hal_pool_query_stats(deferred_backing, &after);
  EXPECT_EQ(after.reserve_count, before.reserve_count);
}

}  // namespace
