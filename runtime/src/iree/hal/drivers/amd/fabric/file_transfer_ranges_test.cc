// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <tuple>
#include <vector>

#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/fabric/file_transfer.h"
#include "iree/hal/drivers/amd/fabric/memory_backend_test_base.h"
#include "iree/hal/drivers/amd/fabric/virtual_buffer.h"
#include "iree/hal/utils/fd_file.h"
#include "iree/io/file_contents.h"
#include "iree/testing/temp_file.h"

namespace {
using iree::hal::amd::fabric::FakeNative;
using iree::hal::amd::fabric::MemoryBackendTest;
using iree::hal::cts::Ref;

class FileTransferRangesTest
    : public MemoryBackendTest,
      public ::testing::WithParamInterface<std::tuple<uint32_t, bool>> {};

TEST_P(FileTransferRangesTest, PreservesEveryPhysicalRouteAndCoordinate) {
  static_assert(sizeof(iree_hal_amd_fabric_file_range_t) <= 64);
  RecordProperty("source_record_bytes",
                 sizeof(iree_hal_amd_fabric_file_range_t));
  constexpr uint64_t kUnit = 4096;
  constexpr uint64_t kBufferOffset = 128;
  constexpr uint64_t kFileOffset = 64;
  constexpr uint64_t kLength = 3 * kUnit - 256;
  const auto [routes, write] = GetParam();
  // Each physical source is private, CPU-accessible, or file-capable. CPU
  // accessibility alone does not permit native file I/O through that view.
  const std::array<uint32_t, 3> source_routes = {routes % 3, (routes / 3) % 3,
                                                 routes / 9};
  native_.required_flags = 0;
  native_.mapping_enabled = true;
  native_.api.memory_query_pair_info =
      +[](const amdf_memory_site_t* producer,
          const amdf_memory_site_t* consumer,
          amdf_memory_pair_info_t* out_info) -> amdf_status_t {
    const auto* device =
        producer->kind == AMDF_MEMORY_SITE_KIND_DEVICE ? producer : consumer;
    const auto* host = device == producer ? consumer : producer;
    EXPECT_EQ(host->kind, AMDF_MEMORY_SITE_KIND_FILE_IO);
    const auto* memory =
        reinterpret_cast<FakeNative::Memory*>(device->value.device.memory);
    EXPECT_EQ(reinterpret_cast<const void*>(host->value.host.mapping), memory);
    EXPECT_EQ(device->value.device.access_ordinal, 0u);
    EXPECT_EQ(device->value.device.queue_family_ordinal,
              memory->owner->queue_family_ordinals[0]);
    out_info->flags = AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE;
    out_info->release.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
    out_info->acquire.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
    return AMDF_STATUS_OK;
  };
  for (uint32_t i = 0; i < families_.size(); ++i) {
    binding_descriptions_[i] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                AMDF_MEMORY_ADDRESS_GPU};
    native_.address_kinds[i] = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
    native_families_[i].host_usage = 0;
  }
  std::array<Ref<iree_hal_pool_t>, 3> pools;
  for (uint32_t i = 0; i < pools.size(); ++i) {
    native_.host_mappable = i != 0;
    native_.host_visibility_guaranteed = i != 0;
    native_.host_mapping_features =
        i == 2 ? AMDF_HOST_MAPPING_FEATURE_FILE_IO : 0;
    IREE_ASSERT_OK(CreatePool(
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
        pools[i].out(), UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
        IREE_HAL_MEMORY_ACCESS_NONE));
  }
  std::array<Ref<iree_hal_buffer_t>, 3> sources;
  std::array<uint8_t*, 3> source_contents;
  std::array<iree_hal_memory_mapping_t, 3> mappings = {};
  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(CreateReservation(target.out()));
  const auto scope = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[0]);
  for (uint32_t i = 0; i < sources.size(); ++i) {
    native_.host_mapping_features =
        source_routes[i] == 2 ? AMDF_HOST_MAPPING_FEATURE_FILE_IO : 0;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pools[source_routes[i]],
        iree_hal_make_pool_reservation_request(2 * kUnit, kUnit),
        iree_infinite_timeout(), sources[i].out()));
    const auto memory = iree_hal_buffer_memory_view(sources[i]);
    const auto* facts =
        reinterpret_cast<const iree_hal_amd_fabric_memory_binding_facts_t*>(
            memory.bindings + memory.contract->binding_layout.binding_count);
    auto* native = reinterpret_cast<FakeNative::Memory*>(facts->memory->handle);
    source_contents[i] = static_cast<uint8_t*>(native->data);
    std::memset(source_contents[i], 0xCD, 2 * kUnit);
    EXPECT_EQ(facts->memory->mapping != nullptr, source_routes[i] != 0);
    mappings[i] = {
        .target = target,
        .scope = scope,
        .target_range = {i * kUnit, kUnit},
        .source = sources[i],
        .source_offset = kUnit,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
    };
  }
  Ref<iree_hal_semaphore_t> timeline;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      cpu_device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, 0, timeline.out()));
  auto* semaphore = timeline.get();
  uint64_t mapped = 1;
  IREE_ASSERT_OK(iree_hal_queue_map(iree_hal_device_queue(cpu_device_, 0, 0),
                                    {}, {1, &semaphore, &mapped},
                                    mappings.size(), mappings.data(), 0));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(timeline, mapped, iree_infinite_timeout(), 0));

  std::vector<uint8_t> expected(kLength + 2 * kFileOffset, 0xCD);
  for (uint64_t i = 0; i < kLength; ++i) {
    expected[kFileOffset + i] = uint8_t(i * 7 + 3);
    if (write) {
      const uint64_t offset = kBufferOffset + i;
      source_contents[offset / kUnit][kUnit + offset % kUnit] =
          expected[kFileOffset + i];
    }
  }
  const std::vector<uint8_t> initial =
      write ? std::vector<uint8_t>(expected.size(), 0xCD) : expected;
  iree::testing::TempFilePath file_path("iree_file_transfer_ranges");
  IREE_ASSERT_OK(iree_io_file_contents_write(
      file_path.path_view(),
      iree_make_const_byte_span(initial.data(), initial.size()), allocator_));
  iree_io_file_handle_t* handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
          IREE_IO_FILE_MODE_ASYNC,
      file_path.path_view(), allocator_, &handle));
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_platform(
      iree_async_proactor_options_default(), allocator_, &proactor));
  Ref<iree_hal_file_t> file;
  iree_status_t status = iree_hal_fd_file_from_handle(
      IREE_HAL_MEMORY_ACCESS_ALL, handle, proactor, allocator_, file.out());
  iree_io_file_handle_release(handle);
  IREE_ASSERT_OK(status);

  iree_hal_amd_fabric_file_transfer_t transfer = {};
  IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_initialize(
      file, write ? IREE_HAL_MEMORY_ACCESS_WRITE : IREE_HAL_MEMORY_ACCESS_READ,
      kFileOffset, target, kBufferOffset, kLength, allocator_, &transfer));
  target.reset();
  file.reset();
  iree_hal_memory_effects_t before = {}, after = {};
  IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_prepare_ranges(
      &transfer, &native_.api, families_[0],
      native_families_[0].queue_family_ordinal, &before, &after));
  EXPECT_EQ(before.bits, 0u);
  EXPECT_EQ(after.bits, 0u);
  ASSERT_EQ(transfer.sources.count, sources.size());
  uint32_t host_count = 0;
  for (uint32_t i = 0; i < sources.size(); ++i) {
    const uint64_t start = std::max(i * kUnit, kBufferOffset);
    const uint64_t end = std::min((i + 1) * kUnit, kBufferOffset + kLength);
    const bool host = source_routes[i] == 2;
    host_count += host;
    const auto& range = transfer.sources.ranges[i];
    const auto* memory =
        reinterpret_cast<const FakeNative::Memory*>(range.storage->handle);
    EXPECT_EQ(memory->data, source_contents[i]);
    EXPECT_EQ(range.offset, start - kBufferOffset);
    EXPECT_EQ(range.source_offset, kUnit + start % kUnit);
    EXPECT_EQ(range.contents.data_length, end - start);
    EXPECT_EQ(range.contents.data,
              host ? source_contents[i] + range.source_offset : nullptr);
    EXPECT_EQ(range.access_ordinal, 0u);
    EXPECT_EQ(range.region, nullptr);
    EXPECT_EQ(range.effects.bits, 0u);
    transfer.async.request.offset = range.offset;
    EXPECT_EQ(iree_hal_amd_fabric_file_transfer_current_range(&transfer),
              &range);
    if (host) {
      IREE_ASSERT_OK(
          iree_hal_amd_fabric_file_transfer_prepare_range(&transfer, &range));
      EXPECT_EQ(transfer.async.request.span.length, end - start);
    }
    // Resolution and request preparation neither copy nor mutate payloads.
    for (uint64_t j = 0; j < 2 * kUnit; ++j) {
      const uint64_t position = i * kUnit + j;
      const uint8_t expected_byte =
          write && j >= kUnit && position - kUnit >= kBufferOffset &&
                  position - kUnit < kBufferOffset + kLength
              ? expected[kFileOffset + position - kUnit - kBufferOffset]
              : uint8_t{0xCD};
      EXPECT_EQ(source_contents[i][j], expected_byte) << i << ':' << j;
    }
  }
  EXPECT_EQ(transfer.sources.host_count, host_count);
  iree_hal_amd_fabric_file_transfer_deinitialize(&transfer);
  iree_async_proactor_release(proactor);
  iree_io_file_contents_t* actual = nullptr;
  IREE_ASSERT_OK(
      iree_io_file_contents_read(file_path.path_view(), allocator_, &actual));
  ASSERT_EQ(actual->const_buffer.data_length, expected.size());
  EXPECT_EQ(
      std::memcmp(actual->const_buffer.data, initial.data(), initial.size()),
      0);
  iree_io_file_contents_free(actual);
  EXPECT_TRUE(file_path.Remove());
}

INSTANTIATE_TEST_SUITE_P(Routes, FileTransferRangesTest,
                         ::testing::Combine(::testing::Range(0u, 27u),
                                            ::testing::Bool()));

}  // namespace
