// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/buffer_range.h"

#include <algorithm>
#include <array>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/maintenance_thread.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class BufferRangeTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), allocator_, &proactor_));
    IREE_ASSERT_OK(iree_async_notification_create(
        proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification_));
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), allocator_, &tracker_));
    for (uint8_t i = 0; i < 2; ++i) {
      IREE_ASSERT_OK(iree_async_frontier_tracker_register_axis(
          tracker_, Axis(i), nullptr));
    }
    IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(/*min_alignment=*/0,
                                                     allocator_, &provider_));
    iree_hal_fixed_block_pool_options_t options =
        {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment sequencing
             // spans intervening work.
    options.block_size = 4096;
    options.blocks_per_slab = 1;
    options.frontier_capacity = 2;
    IREE_ASSERT_OK(iree_hal_memory_maintenance_thread_create({}, allocator_,
                                                             &maintenance_));
    iree_hal_pool_t* native_pool = nullptr;
    IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
        {}, provider_, notification_, tracker_, maintenance_, allocator_,
        &native_pool));
    pools_.push_back(native_pool);
    iree_hal_pool_reservation_request_t backing_request;
    IREE_ASSERT_OK(iree_hal_fixed_block_pool_query_backing_request(
        native_pool, &options, &backing_request));
    iree_hal_buffer_t* backing_buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        native_pool, backing_request.params, backing_request.allocation_size,
        iree_infinite_timeout(), &backing_buffer));
    options.blocks_per_slab = 0;
    iree_status_t status = iree_hal_fixed_block_pool_create_from_buffer(
        backing_buffer, 0, IREE_HAL_WHOLE_BUFFER, &options, allocator_,
        &source_pool_);
    iree_hal_buffer_release(backing_buffer);
    IREE_ASSERT_OK(status);
    pools_.push_back(source_pool_);
    params_.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
    params_.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params_.usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  }

  void TearDown() override {
    for (auto* buffer : buffers_) {
      iree_hal_buffer_release(buffer);
    }
    for (auto i = pools_.rbegin(); i != pools_.rend(); ++i) {
      iree_hal_pool_release(*i);
    }
    iree_hal_memory_maintenance_release(maintenance_);
    iree_hal_slab_provider_release(provider_);
    iree_async_frontier_tracker_release(tracker_);
    iree_async_notification_release(notification_);
    iree_async_proactor_release(proactor_);
  }

  static iree_async_axis_t Axis(uint8_t index) {
    return iree_async_axis_make_queue(1, 0, 0, index, 0);
  }

  iree_status_t Allocate(iree_hal_pool_t* pool, iree_device_size_t size,
                         iree_hal_buffer_t** out_buffer) {
    iree_status_t status = iree_hal_pool_allocate_buffer(
        pool, params_, size, iree_immediate_timeout(), out_buffer);
    if (iree_status_is_ok(status)) {
      buffers_.push_back(*out_buffer);
    }
    return status;
  }

  iree_status_t CreateChild(iree_hal_buffer_t* buffer,
                            iree_device_size_t offset,
                            iree_device_size_t length,
                            uint8_t frontier_capacity,
                            iree_hal_pool_t** out_pool) {
    iree_status_t status = iree_ok_status();
    if (GetParam()) {
      iree_hal_tlsf_pool_options_t options = {};
      options.tlsf_options.frontier_capacity = frontier_capacity;
      status = iree_hal_tlsf_pool_create_from_buffer(
          buffer, offset, length, &options, allocator_, out_pool);
    } else {
      iree_hal_fixed_block_pool_options_t options =
          {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment
               // conversion differs from list initialization.
      options.block_size = 256;
      options.frontier_capacity = frontier_capacity;
      status = iree_hal_fixed_block_pool_create_from_buffer(
          buffer, offset, length, &options, allocator_, out_pool);
    }
    if (iree_status_is_ok(status)) {
      pools_.push_back(*out_pool);
    }
    return status;
  }

  // Prepares real host storage whose 256-byte maintenance coordinates have a
  // 64-byte residue in the native address space. Captured facts belong to this
  // native storage producer and remain alive until all views and pools retire.
  iree_status_t CreateMaintenanceBuffer(iree_hal_buffer_t** out_buffer) {
    maintenance_storage_.fill(0x11);
    maintenance_backing_.notification = notification_;
    maintenance_backing_.tracker = tracker_;
    maintenance_backing_.maintenance = maintenance_;
    maintenance_backing_.allocation_alignment = 64;
    maintenance_backing_.maintenance_alignment = 256;
    const iree_hal_buffer_placement_t placement = {
        nullptr, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE};
    IREE_RETURN_IF_ERROR(iree_hal_heap_buffer_wrap(
        placement, IREE_HAL_CPU_SLAB_PROVIDER_MEMORY_TYPE,
        IREE_HAL_MEMORY_ACCESS_ALL, params_.usage, 4096,
        iree_make_byte_span(maintenance_storage_.data() + 64, 4096), {},
        allocator_, out_buffer));
    (*out_buffer)->memory.backing = &maintenance_backing_;
    buffers_.push_back(*out_buffer);
    return iree_ok_status();
  }

  // Allocator for metadata and real CPU slab backing.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Progress owner shared by the native source and child pools.
  iree_async_proactor_t* proactor_ = nullptr;
  // Capacity notification retained by the pools.
  iree_async_notification_t* notification_ = nullptr;
  // Completion tracker with two registered production queue axes.
  iree_async_frontier_tracker_t* tracker_ = nullptr;
  // Captured owner for native allocation and retirement.
  iree_hal_memory_maintenance_t* maintenance_ = nullptr;
  // Native CPU storage provider.
  iree_hal_slab_provider_t* provider_ = nullptr;
  // One native block used as the source for each test.
  iree_hal_pool_t* source_pool_ = nullptr;
  // Pools in construction order, released children first.
  std::vector<iree_hal_pool_t*> pools_;
  // Owned buffer references released before pool teardown.
  std::vector<iree_hal_buffer_t*> buffers_;
  // Common permissions used by native and child allocations.
  iree_hal_buffer_params_t params_ = {};
  // Imported native storage with explicit padding before byte zero.
  alignas(256) std::array<uint8_t, 4096 + 64> maintenance_storage_;
  // Immutable native facts borrowed by the imported buffer and descendants.
  iree_hal_buffer_backing_facts_t maintenance_backing_ = {};
};

TEST_P(BufferRangeTest, InheritsExactHistoryThroughSplitsAndRollback) {
  alignas(iree_async_frontier_entry_t)
      uint8_t history_storage[sizeof(iree_async_frontier_t) +
                              2 * sizeof(iree_async_frontier_entry_t)] = {};
  auto* history = reinterpret_cast<iree_async_frontier_t*>(history_storage);
  iree_async_frontier_initialize(history, 2);
  history->entries[0] = {Axis(0), 7};
  history->entries[1] = {Axis(1), 11};
  const iree_hal_pool_reservation_request_t source_request = {params_, 4096};
  iree_hal_pool_reservation_t source_reservation;
  iree_hal_pool_acquire_info_t source_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      source_pool_, 1, &source_request, nullptr,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &source_reservation, &source_info,
      &result));
  ASSERT_EQ(IREE_HAL_POOL_ACQUIRE_OK_FRESH, result);
  iree_hal_pool_release_reservations(source_pool_, 1, &source_reservation,
                                     history);
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      source_pool_, 1, &source_request, history,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &source_reservation, &source_info,
      &result));
  ASSERT_EQ(IREE_HAL_POOL_ACQUIRE_OK, result);
  iree_hal_buffer_t* source = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
      source_pool_, 1, &source_request, &source_reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, &source));
  buffers_.push_back(source);
  ASSERT_EQ(IREE_ASYNC_FRONTIER_EQUAL,
            iree_async_frontier_compare(
                history, iree_hal_buffer_memory_view(source).reuse_frontier));

  iree_hal_pool_t* child = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        CreateChild(source, 128, 1024, 1, &child));
  EXPECT_EQ(nullptr, child);
  IREE_ASSERT_OK(CreateChild(source, 128, 1024, 2, &child));
  const iree_hal_pool_reservation_request_t request = {params_, 256};
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      child, 1, &request, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      &reservation, &info, &result));
  EXPECT_EQ(IREE_HAL_POOL_ACQUIRE_EXHAUSTED, result);
  EXPECT_EQ(0u, info.flags & IREE_HAL_POOL_ACQUIRE_FLAG_GROWTH_REQUIRED);

  // Both the first range and the untouched remainder inherit the source axes.
  std::array<iree_hal_pool_reservation_t, 2> held;
  for (auto& element : held) {
    IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
        child, 1, &request, nullptr,
        IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER, &element, &info,
        &result));
    ASSERT_EQ(IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT, result);
    ASSERT_NE(nullptr, info.reuse_frontier);
    EXPECT_EQ(IREE_ASYNC_FRONTIER_EQUAL,
              iree_async_frontier_compare(history, info.reuse_frontier));
  }
  iree_hal_pool_release_reservations(child, held.size(), held.data(), history);

  // Exhausting a transaction and an immediate synchronous wait both restore
  // the exact prerequisite; neither can turn unconsumed storage into fresh
  // bytes.
  std::array<iree_hal_pool_reservation_request_t, 5> requests;
  requests.fill(request);
  std::array<iree_hal_pool_reservation_t, 5> reservations = {};
  std::array<iree_hal_pool_acquire_info_t, 5> infos;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      child, requests.size(), requests.data(), nullptr,
      IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER, reservations.data(),
      infos.data(), &result));
  EXPECT_EQ(IREE_HAL_POOL_ACQUIRE_EXHAUSTED, result);
  iree_hal_buffer_t* pending = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED,
                        Allocate(child, 256, &pending));
  EXPECT_EQ(nullptr, pending);
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      child, 1, &request, nullptr,
      IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER, &reservation, &info,
      &result));
  ASSERT_EQ(IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT, result);
  EXPECT_EQ(IREE_ASYNC_FRONTIER_EQUAL,
            iree_async_frontier_compare(history, info.reuse_frontier));
  iree_hal_pool_release_reservations(child, 1, &reservation, history);
  iree_async_frontier_tracker_advance(tracker_, Axis(0), 7);
  iree_async_frontier_tracker_advance(tracker_, Axis(1), 11);
  IREE_ASSERT_OK(Allocate(child, 256, &pending));
  uint32_t value = 0x12345678;
  IREE_ASSERT_OK(iree_hal_buffer_map_write(pending, 0, &value, sizeof(value)));
  uint32_t actual = 0;
  IREE_ASSERT_OK(iree_hal_buffer_map_read(pending, 0, &actual, sizeof(actual)));
  EXPECT_EQ(value, actual);
}

TEST_P(BufferRangeTest, EnforcesVisibleRangeAndPermissions) {
  iree_hal_buffer_t* source = nullptr;
  IREE_ASSERT_OK(Allocate(source_pool_, 4096, &source));
  iree_hal_pool_t* child = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        CreateChild(source, 4000, 256, 2, &child));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      CreateChild(source, 4096, IREE_HAL_WHOLE_BUFFER, 2, &child));
  IREE_ASSERT_OK(CreateChild(source, 128, 512, 2, &child));
  auto unsupported = params_;
  unsupported.usage |= IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_pool_allocate_buffer(child, unsupported, 64,
                                    iree_immediate_timeout(), &buffer));
  EXPECT_EQ(nullptr, buffer);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(child, &stats);
  EXPECT_EQ(0u, stats.reservation_count);
  IREE_ASSERT_OK(Allocate(child, 64, &buffer));
  EXPECT_EQ(128u, iree_hal_buffer_memory_view(buffer).offset);
}

TEST_P(BufferRangeTest, AlignsNativeCoordinatesWithinVisibleRange) {
  iree_hal_buffer_t* source = nullptr;
  IREE_ASSERT_OK(Allocate(source_pool_, 4096, &source));
  std::array<uint8_t, 4096> expected = {};
  expected.fill(0x11);
  IREE_ASSERT_OK(
      iree_hal_buffer_map_write(source, 0, expected.data(), expected.size()));
  iree_hal_buffer_t* view = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(source, 3, 128, allocator_, &view));
  buffers_.push_back(view);

  iree_hal_pool_t* child = nullptr;
  if (GetParam()) {
    iree_hal_tlsf_pool_options_t options = {};
    options.tlsf_options.alignment = 24;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          iree_hal_tlsf_pool_create_from_buffer(
                              view, 2, 64, &options, allocator_, &child));
    options.tlsf_options.alignment =
        iree_hal_buffer_memory_view(source).backing->allocation_alignment * 2;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          iree_hal_tlsf_pool_create_from_buffer(
                              view, 2, 64, &options, allocator_, &child));
    options.tlsf_options.alignment = 16;
    IREE_ASSERT_OK(iree_hal_tlsf_pool_create_from_buffer(view, 2, 64, &options,
                                                         allocator_, &child));
  } else {
    iree_hal_fixed_block_pool_options_t options =
        {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment sequencing
             // spans intervening work.
    options.block_size = 13;
    options.alignment = 24;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          iree_hal_fixed_block_pool_create_from_buffer(
                              view, 2, 64, &options, allocator_, &child));
    options.alignment =
        iree_hal_buffer_memory_view(source).backing->allocation_alignment * 2;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          iree_hal_fixed_block_pool_create_from_buffer(
                              view, 2, 64, &options, allocator_, &child));
    options.alignment = 16;
    IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
        view, 2, 64, &options, allocator_, &child));
  }
  pools_.push_back(child);
  params_.min_alignment = 16;
  for (iree_device_size_t offset : {16, 32, 48}) {
    iree_hal_buffer_t* allocation = nullptr;
    IREE_ASSERT_OK(Allocate(child, 13, &allocation));
    EXPECT_EQ(offset, iree_hal_buffer_memory_view(allocation).offset);
    std::array<uint8_t, 13> payload;
    payload.fill(static_cast<uint8_t>(offset));
    iree_byte_span_t native_span;
    IREE_ASSERT_OK(iree_hal_buffer_native_host_span(
        allocation, 0, IREE_HAL_WHOLE_BUFFER, &native_span));
    ASSERT_EQ(payload.size(), native_span.data_length);
    std::copy(payload.begin(), payload.end(), native_span.data);
    std::copy(payload.begin(), payload.end(), expected.begin() + offset);
  }
  iree_hal_buffer_t* exhausted = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED,
                        Allocate(child, 13, &exhausted));
  std::array<uint8_t, 4096> actual;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(source, 0, actual.data(), actual.size()));
  EXPECT_EQ(expected, actual);
}

TEST_P(BufferRangeTest, MaintenanceGranulesPreserveFiniteNativeAlignment) {
  iree_hal_buffer_t* source = nullptr;
  IREE_ASSERT_OK(CreateMaintenanceBuffer(&source));
  iree_hal_buffer_t* view = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(source, 3, 4093, allocator_, &view));
  buffers_.push_back(view);
  iree_hal_pool_t* child = nullptr;
  if (GetParam()) {
    iree_hal_tlsf_pool_options_t options = {};
    options.tlsf_options.alignment = 64;
    IREE_ASSERT_OK(iree_hal_tlsf_pool_create_from_buffer(
        view, 2, 1024, &options, allocator_, &child));
  } else {
    iree_hal_fixed_block_pool_options_t options = {.block_size = 13,
                                                   .alignment = 64};
    IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
        view, 2, 1024, &options, allocator_, &child));
  }
  pools_.push_back(child);
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(child, &capabilities);
  EXPECT_EQ(capabilities.max_allocation_alignment, 64u);
  EXPECT_EQ(capabilities.maintenance_alignment, 256u);
  params_.min_alignment = 128;
  iree_hal_buffer_t* invalid = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Allocate(child, 13, &invalid));
  EXPECT_EQ(invalid, nullptr);
  params_.min_alignment = 64;
  std::array<uint8_t, 4096> expected;
  expected.fill(0x11);
  for (iree_device_size_t offset : {256, 512, 768}) {
    iree_hal_buffer_t* allocation = nullptr;
    IREE_ASSERT_OK(Allocate(child, 13, &allocation));
    EXPECT_EQ(iree_hal_buffer_memory_view(allocation).offset, offset);
    iree_byte_span_t native;
    IREE_ASSERT_OK(iree_hal_buffer_native_host_span(
        allocation, 0, IREE_HAL_WHOLE_BUFFER, &native));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(native.data) % 256, 64u);
    std::fill_n(native.data, native.data_length, 0xD5);
    std::fill_n(expected.data() + offset, 13, 0xD5);
  }
  std::array<uint8_t, 4096> actual;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(source, 0, actual.data(), actual.size()));
  EXPECT_EQ(actual, expected);
}

TEST_P(BufferRangeTest, MaintenanceGranulesSurviveGrowableChildren) {
  iree_hal_buffer_t* source = nullptr;
  IREE_ASSERT_OK(CreateMaintenanceBuffer(&source));
  iree_hal_fixed_block_pool_options_t parent_options = {.block_size = 1024};
  iree_hal_pool_t* parent = nullptr;
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
      source, 0, IREE_HAL_WHOLE_BUFFER, &parent_options, allocator_, &parent));
  pools_.push_back(parent);
  iree_hal_pool_t* child = nullptr;
  iree_hal_pool_reservation_request_t backing_request;
  if (GetParam()) {
    iree_hal_tlsf_pool_options_t options = {};
    options.tlsf_options.range_length = 512;
    IREE_ASSERT_OK(iree_hal_tlsf_pool_query_backing_request(parent, &options,
                                                            &backing_request));
    IREE_ASSERT_OK(
        iree_hal_tlsf_pool_create(parent, &options, allocator_, &child));
  } else {
    iree_hal_fixed_block_pool_options_t options = {.block_size = 13,
                                                   .blocks_per_slab = 2};
    IREE_ASSERT_OK(iree_hal_fixed_block_pool_query_backing_request(
        parent, &options, &backing_request));
    IREE_ASSERT_OK(
        iree_hal_fixed_block_pool_create(parent, &options, allocator_, &child));
  }
  pools_.push_back(child);
  EXPECT_EQ(backing_request.params.min_alignment, 64u);
  EXPECT_EQ(backing_request.allocation_size, 512u);
  params_.min_alignment = 128;
  iree_hal_buffer_t* invalid = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Allocate(child, 13, &invalid));
  params_.min_alignment = 64;
  std::array<uint8_t, 4096> expected;
  expected.fill(0x11);
  for (int i = 0; i < 6; ++i) {
    iree_hal_buffer_t* allocation = nullptr;
    IREE_ASSERT_OK(Allocate(child, 13, &allocation));
    const auto offset = iree_hal_buffer_memory_view(allocation).offset;
    EXPECT_EQ(offset % 256, 0u);
    iree_byte_span_t native;
    IREE_ASSERT_OK(iree_hal_buffer_native_host_span(
        allocation, 0, IREE_HAL_WHOLE_BUFFER, &native));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(native.data) % 256, 64u);
    const uint8_t value = 0xD0 + i;
    std::fill_n(native.data, native.data_length, value);
    std::fill_n(expected.data() + offset, 13, value);
  }
  if (GetParam()) {
    // A larger allocation takes TLSF's dedicated reservation path through
    // the same backing pool and must preserve both coordinate guarantees.
    iree_hal_buffer_t* allocation = nullptr;
    IREE_ASSERT_OK(Allocate(child, 577, &allocation));
    const auto offset = iree_hal_buffer_memory_view(allocation).offset;
    EXPECT_EQ(offset % 256, 0u);
    std::array<uint8_t, 577> payload;
    payload.fill(0xE7);
    IREE_ASSERT_OK(iree_hal_buffer_map_write(allocation, 0, payload.data(),
                                             payload.size()));
    std::copy(payload.begin(), payload.end(), expected.begin() + offset);
  }
  std::array<uint8_t, 4096> actual;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(source, 0, actual.data(), actual.size()));
  EXPECT_EQ(actual, expected);
}

INSTANTIATE_TEST_SUITE_P(Strategies, BufferRangeTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "TLSF" : "FixedBlock";
                         });

}  // namespace
