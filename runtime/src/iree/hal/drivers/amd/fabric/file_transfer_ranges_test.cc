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
      public ::testing::WithParamInterface<std::tuple<uint32_t, bool, bool>> {};

TEST_P(FileTransferRangesTest, PreservesEveryPhysicalRouteAndCoordinate) {
  static_assert(sizeof(iree_hal_amd_fabric_file_source_t) <= 64);
  RecordProperty("source_record_bytes",
                 sizeof(iree_hal_amd_fabric_file_source_t));
  constexpr uint64_t kUnit = 4096;
  constexpr uint64_t kBufferOffset = 128;
  constexpr uint64_t kFileOffset = 64;
  constexpr uint64_t kLength = 3 * kUnit - 256;
  const auto [routes, write, direct] = GetParam();
  // Each physical source is private, CPU-only, pinnable for file I/O, or
  // buffered-only. The default route stages unpinnable views even when they
  // support native buffered access.
  const std::array<uint32_t, 3> source_routes = {routes % 4, (routes / 4) % 4,
                                                 routes / 16};
  constexpr std::array<amdf_host_mapping_features_t, 4> kFeatures = {
      0, 0, AMDF_HOST_MAPPING_FEATURE_FILE_IO,
      AMDF_HOST_MAPPING_FEATURE_BUFFERED_FILE_IO};
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
  std::array<Ref<iree_hal_pool_t>, 4> pools;
  for (uint32_t i = 0; i < pools.size(); ++i) {
    native_.host_mappable = i != 0;
    native_.host_visibility_guaranteed = i != 0;
    native_.host_mapping_features = kFeatures[i];
    IREE_ASSERT_OK(CreatePool(
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
        pools[i].out(), UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
        IREE_HAL_MEMORY_ACCESS_NONE));
  }
  std::array<Ref<iree_hal_buffer_t>, 3> sources;
  std::array<uint8_t*, 3> source_contents;
  for (uint32_t i = 0; i < sources.size(); ++i) {
    native_.host_mapping_features = kFeatures[source_routes[i]];
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
  }

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

  // The owner already has the physical ranges. Each read preserves that
  // identity rather than making the file path reconstruct installed aliases.
  for (uint32_t i = 0; i < sources.size(); ++i) {
    const uint64_t start = std::max(i * kUnit, kBufferOffset);
    const uint64_t end = std::min((i + 1) * kUnit, kBufferOffset + kLength);
    const bool host = source_routes[i] == 2;
    iree_hal_amd_fabric_file_transfer_t transfer = {};
    IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_initialize(
        file,
        write ? IREE_HAL_MEMORY_ACCESS_WRITE : IREE_HAL_MEMORY_ACCESS_READ,
        kFileOffset + start - kBufferOffset, sources[i], kUnit + start % kUnit,
        end - start, allocator_, &transfer));
    // Route qualification is independent of the temporary filesystem's mode.
    if (direct) {
      transfer.async.file->flags |= IREE_ASYNC_FILE_FLAG_DIRECT;
    }
    iree_hal_memory_effects_t before = {}, after = {};
    IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_prepare_source(
        &transfer, &native_.api, families_[0],
        native_families_[0].queue_family_ordinal, &before, &after));
    EXPECT_EQ(before.bits, 0u);
    EXPECT_EQ(after.bits, 0u);
    const auto& range = transfer.source;
    const auto* memory =
        reinterpret_cast<const FakeNative::Memory*>(range.storage->handle);
    EXPECT_EQ(memory->data, source_contents[i]);
    EXPECT_EQ(range.source_offset, kUnit + start % kUnit);
    EXPECT_EQ(range.contents.data_length, end - start);
    EXPECT_EQ(range.contents.data,
              host ? source_contents[i] + range.source_offset : nullptr);
    EXPECT_EQ(range.access_ordinal, 0u);
    EXPECT_EQ(range.region, nullptr);
    EXPECT_EQ(range.effects.bits, 0u);
    transfer.async.request.offset = 0;
    if (host && !direct) {
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
    iree_hal_amd_fabric_file_transfer_deinitialize(&transfer);
  }
  file.reset();
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

TEST_F(MemoryBackendTest, ExplicitSourcesPreserveIoCoordinatesAcrossLayouts) {
  constexpr uint64_t kUnit = 4096;
  native_.required_flags = 0;
  native_.mapping_enabled = true;
  native_.host_mappable = true;
  native_.host_visibility_guaranteed = true;
  native_.host_mapping_features = AMDF_HOST_MAPPING_FEATURE_FILE_IO;
  native_.api.memory_query_pair_info =
      +[](const amdf_memory_site_t*, const amdf_memory_site_t*,
          amdf_memory_pair_info_t* info) -> amdf_status_t {
    info->flags = AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE;
    info->release.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
    info->acquire.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
    return AMDF_STATUS_OK;
  };
  for (size_t i = 0; i < families_.size(); ++i) {
    binding_descriptions_[i] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                AMDF_MEMORY_ADDRESS_GPU};
    native_.address_kinds[i] = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
    native_families_[i].host_usage = 0;
  }
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(
      CreatePool(IREE_HAL_BUFFER_USAGE_TRANSFER, pool.out(),
                 UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
                 IREE_HAL_MEMORY_ACCESS_NONE));
  std::array<Ref<iree_hal_buffer_t>, 2> sources;
  std::array<uint8_t*, 2> contents;
  for (size_t i = 0; i < sources.size(); ++i) {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool, iree_hal_make_pool_reservation_request(4 * kUnit, kUnit),
        iree_infinite_timeout(), sources[i].out()));
    const auto memory = iree_hal_buffer_memory_view(sources[i]);
    const auto* facts =
        reinterpret_cast<const iree_hal_amd_fabric_memory_binding_facts_t*>(
            memory.bindings + memory.contract->binding_layout.binding_count);
    contents[i] = static_cast<uint8_t*>(
        reinterpret_cast<FakeNative::Memory*>(facts->memory->handle)->data);
  }
  iree::testing::TempFilePath path("iree_mapping_file_round_trip");
  std::vector<uint8_t> expected(4 * kUnit);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = uint8_t(i * 7 + 3);
  }
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(),
      iree_make_const_byte_span(expected.data(), expected.size()), allocator_));
  iree_io_file_handle_t* handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
          IREE_IO_FILE_MODE_ASYNC,
      path.path_view(), allocator_, &handle));
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_platform(
      iree_async_proactor_options_default(), allocator_, &proactor));
  Ref<iree_hal_file_t> file;
  iree_status_t status = iree_hal_fd_file_from_handle(
      IREE_HAL_MEMORY_ACCESS_ALL, handle, proactor, allocator_, file.out());
  iree_io_file_handle_release(handle);
  IREE_ASSERT_OK(status);

  for (uint32_t epoch = 0; epoch < 3; ++epoch) {
    // File bytes follow the owner's current physical layout, including a
    // split middle source and a later return to one contiguous source.
    for (auto* bytes : contents) {
      std::memset(bytes, 0xCD, 4 * kUnit);
    }
    for (const auto access :
         {IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_MEMORY_ACCESS_WRITE}) {
      if (access == IREE_HAL_MEMORY_ACCESS_WRITE) {
        for (size_t i = 0; i < expected.size(); ++i) {
          expected[i] ^= uint8_t(0x5A + epoch);
          const bool middle = i >= kUnit && i < 3 * kUnit;
          contents[epoch < 2 && middle ? 1 : 0][i] = expected[i];
        }
      }
      const uint32_t range_count = epoch < 2 ? 3 : 1;
      for (uint32_t i = 0; i < range_count; ++i) {
        const uint64_t offset = range_count == 1 ? 0
                                : i == 2         ? 3 * kUnit
                                                 : i * kUnit;
        const uint64_t length = range_count == 1 ? 4 * kUnit
                                : i == 1         ? 2 * kUnit
                                                 : kUnit;
        const size_t owner = range_count > 1 && i == 1 ? 1 : 0;
        iree_hal_amd_fabric_file_transfer_t transfer = {};
        IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_initialize(
            file, access, offset, sources[owner], offset, length, allocator_,
            &transfer));
        iree_hal_memory_effects_t before = {}, after = {};
        IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_prepare_source(
            &transfer, &native_.api, families_[0],
            native_families_[0].queue_family_ordinal, &before, &after));
        EXPECT_NE(transfer.source.contents.data, nullptr);
        EXPECT_EQ(before.bits, 0u);
        EXPECT_EQ(after.bits, 0u);
        while (iree_hal_amd_fabric_file_transfer_bytes_transferred(&transfer) <
               transfer.length) {
          const auto* range = &transfer.source;
          IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_prepare_range(
              &transfer, range));
          struct Completion {
            // Set by the proactor callback before polling returns.
            bool complete = false;
            // Owned completion result, consumed before submitting another
            // range.
            iree_status_t status = iree_ok_status();
          } completion;
          IREE_ASSERT_OK(iree_hal_amd_fabric_file_request_submit(
              &transfer.async.request,
              {+[](void* user_data, iree_status_t status) {
                 auto* completion = static_cast<Completion*>(user_data);
                 completion->status = status;
                 completion->complete = true;
               },
               &completion}));
          while (!completion.complete) {
            IREE_ASSERT_OK(iree_async_proactor_poll(
                proactor, iree_infinite_timeout(), nullptr));
          }
          if (!iree_status_is_ok(completion.status)) {
            iree_hal_amd_fabric_file_transfer_deinitialize(&transfer);
          }
          IREE_ASSERT_OK(completion.status);
        }
        IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_flush(&transfer));
        iree_hal_amd_fabric_file_transfer_deinitialize(&transfer);
      }
      for (size_t source = 0; source < contents.size(); ++source) {
        for (size_t i = 0; i < expected.size(); ++i) {
          const bool middle = i >= kUnit && i < 3 * kUnit;
          const size_t owner = epoch < 2 && middle ? 1 : 0;
          EXPECT_EQ(contents[source][i], source == owner ? expected[i] : 0xCD);
        }
      }
    }
  }
  file.reset();
  iree_async_proactor_release(proactor);
  iree_io_file_contents_t* actual = nullptr;
  IREE_ASSERT_OK(
      iree_io_file_contents_read(path.path_view(), allocator_, &actual));
  ASSERT_EQ(actual->const_buffer.data_length, expected.size());
  EXPECT_EQ(
      std::memcmp(actual->const_buffer.data, expected.data(), expected.size()),
      0);
  iree_io_file_contents_free(actual);
  EXPECT_TRUE(path.Remove());
}

INSTANTIATE_TEST_SUITE_P(Routes, FileTransferRangesTest,
                         ::testing::Combine(::testing::Range(0u, 64u),
                                            ::testing::Bool(),
                                            ::testing::Bool()));

}  // namespace
