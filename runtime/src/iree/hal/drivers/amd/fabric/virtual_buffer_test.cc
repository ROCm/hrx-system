// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/virtual_buffer.h"

#include <algorithm>
#include <future>

#include "iree/hal/drivers/amd/fabric/file_transfer.h"
#include "iree/hal/drivers/amd/fabric/memory_backend_test_base.h"
#include "iree/hal/utils/fd_file.h"
#include "iree/hal/utils/memory_file.h"
#include "iree/testing/temp_file.h"

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
  options.mapping_granularity = 4 * capabilities.mapping_granularity;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(group_, scope, &options,
                                                allocator_, &reservation));
  iree_hal_virtual_buffer_info_t info;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_info(reservation, &info));
  EXPECT_EQ(info.mapping_granularity, options.mapping_granularity);
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

TEST_F(MemoryBackendTest, SharedTranslationsDoNotGrantSourceConsumerAccess) {
  native_.mapping_enabled = true;
  native_.alias_projections = true;
  native_.required_flags = 0;
  std::array<iree_hal_pool_family_access_t, 2> families = {};
  for (uint32_t i = 0; i < families.size(); ++i) {
    binding_descriptions_[i] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                AMDF_MEMORY_ADDRESS_GPU};
    native_.address_kinds[i] = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
    native_families_[i].host_usage = 0;
    families[i] = {
        .family = families_[i],
        .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
        .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
    };
  }
  iree_hal_pool_t* pool = nullptr;
  const iree_hal_pool_scope_t source_scope = {1, families.data(), {}};
  IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, source_scope, &pool_options_,
                                           allocator_, &pool));
  iree_hal_buffer_t* backing = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(4096, 4096),
      iree_infinite_timeout(), &backing));
  for (auto& family : families) {
    family.interfaces = 0;
  }
  const iree_hal_pool_scope_t target_scope = {
      families.size(), families.data(), {}};
  const iree_hal_virtual_buffer_options_t options = {.byte_length = 4096};
  iree_hal_buffer_t* target = nullptr;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(group_, target_scope, &options,
                                                allocator_, &target));
  const auto first = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[0]);
  const auto second = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[1]);
  const iree_hal_memory_mapping_t mapping = {
      .target = target,
      .scope = first,
      .target_range = {0, 4096},
      .source = backing,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
  };
  iree_hal_semaphore_t* done = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      cpu_device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, 0, &done));
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_map(iree_hal_device_queue(cpu_device_, 0, 0),
                                    {}, {1, &done, &value}, 1, &mapping, 0));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(done, value, iree_infinite_timeout(), 0));
  iree_hal_amd_fabric_virtual_mapping_t source;
  const auto view = iree_hal_buffer_virtual_view(target);
  IREE_ASSERT_OK(iree_hal_amd_fabric_virtual_buffer_query_source(
      view, first.id, 0, 4096, &source));
  EXPECT_EQ(source.source->access_count, 1u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_hal_amd_fabric_virtual_buffer_query_source(
                            view, second.id, 0, 4096, &source));
  iree_hal_semaphore_release(done);
  iree_hal_buffer_release(target);
  iree_hal_buffer_release(backing);
  iree_hal_pool_release(pool);
}

TEST_F(MemoryBackendTest, FileTransfersQualifySourcesAfterDeviceOnlyPrefix) {
  native_.mapping_enabled = true;
  native_.host_mappable = false;
  native_.required_flags = 0;
  for (uint32_t i = 0; i < families_.size(); ++i) {
    binding_descriptions_[i] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                AMDF_MEMORY_ADDRESS_GPU};
    native_.address_kinds[i] = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
    native_families_[i].host_usage = 0;
  }
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      CreatePool(IREE_HAL_BUFFER_USAGE_TRANSFER, &pool,
                 UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
                 IREE_HAL_MEMORY_ACCESS_NONE));
  iree_hal_buffer_t* backing = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(16384, 4096),
      iree_infinite_timeout(), &backing));
  iree_hal_buffer_t* target = nullptr;
  IREE_ASSERT_OK(CreateReservation(&target));
  iree_hal_semaphore_t* done = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      cpu_device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, 0, &done));
  const auto scope = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[0]);
  iree_hal_memory_mapping_t mappings[2] = {
      {
          .target = target,
          .scope = scope,
          .target_range = {0, 8192},
          .source = backing,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_REPLACE,
      },
      {.target = target, .scope = scope, .target_range = {8192, 8192}},
  };

  iree::testing::TempFilePath path("iree_fabric_file_sources");
  iree_io_file_handle_t* handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_create(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
          IREE_IO_FILE_MODE_ASYNC,
      path.path_view(), 16384, allocator_, &handle));
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_get(proactor_pool_, 0, &proactor));
  std::array<iree_hal_file_t*, 2> files = {};
  iree_status_t status = iree_hal_fd_file_from_handle(
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, handle,
      proactor, allocator_, &files[0]);
  iree_io_file_handle_release(handle);
  IREE_ASSERT_OK(status);
  std::vector<uint8_t> contents(16384);
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(contents.data(), contents.size()), {}, allocator_,
      &handle));
  status = iree_hal_memory_file_wrap(
      nullptr, IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      handle, IREE_HAL_MEMORY_FILE_FLAG_NONE, allocator_, &files[1]);
  iree_io_file_handle_release(handle);
  IREE_ASSERT_OK(status);

  uint64_t value = 0;
  for (const auto access :
       {IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_MEMORY_ACCESS_WRITE}) {
    for (const auto suffix_access :
         {IREE_HAL_MEMORY_ACCESS_NONE, IREE_HAL_MEMORY_ACCESS_READ,
          IREE_HAL_MEMORY_ACCESS_ALL}) {
      // Read-only mappings allow writing the file but deny reading into memory.
      const auto expected = !suffix_access ? IREE_STATUS_FAILED_PRECONDITION
                            : access == IREE_HAL_MEMORY_ACCESS_READ &&
                                    suffix_access == IREE_HAL_MEMORY_ACCESS_READ
                                ? IREE_STATUS_PERMISSION_DENIED
                                : IREE_STATUS_OK;
      SCOPED_TRACE(::testing::Message()
                   << "file access=" << access << " expected=" << expected);
      const bool mapped = suffix_access != IREE_HAL_MEMORY_ACCESS_NONE;
      mappings[1].source = mapped ? backing : nullptr;
      mappings[1].source_offset = mapped ? 8192 : 0;
      mappings[1].access = suffix_access;
      mappings[1].operation = mapped ? IREE_HAL_MEMORY_MAPPING_OPERATION_REPLACE
                                     : IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND;
      ++value;
      IREE_ASSERT_OK(iree_hal_queue_map(
          iree_hal_device_queue(cpu_device_, 0, 0), {}, {1, &done, &value},
          IREE_ARRAYSIZE(mappings), mappings, 0));
      IREE_ASSERT_OK(
          iree_hal_semaphore_wait(done, value, iree_infinite_timeout(), 0));
      for (auto* file : files) {
        SCOPED_TRACE(iree_hal_file_async_handle(file) ? "platform" : "memory");
        iree_hal_amd_fabric_file_transfer_t transfer;
        IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_initialize(
            file, access, 0, target, 0, 16384, allocator_, &transfer));
        iree_hal_memory_effects_t before, after;
        IREE_EXPECT_STATUS_IS(
            expected,
            iree_hal_amd_fabric_file_transfer_prepare_ranges(
                &transfer, &native_.api, families_[0],
                native_families_[0].queue_family_ordinal, &before, &after));
        EXPECT_EQ(transfer.sources.host_count, 0u);
        if (expected == IREE_STATUS_OK && iree_hal_file_async_handle(file)) {
          EXPECT_GE(transfer.sources.count, 1u);
          for (size_t i = 0; i < transfer.sources.count; ++i) {
            EXPECT_NE(transfer.sources.ranges[i].storage, nullptr);
            EXPECT_EQ(transfer.sources.ranges[i].contents.data, nullptr);
          }
        } else {
          EXPECT_EQ(transfer.sources.count, 0u);
        }
        EXPECT_EQ(native_.map_count, 0);
        iree_hal_amd_fabric_file_transfer_deinitialize(&transfer);
      }
    }
  }
  for (auto* file : files) {
    iree_hal_file_release(file);
  }
  EXPECT_TRUE(path.Remove());
  iree_hal_semaphore_release(done);
  iree_hal_buffer_release(target);
  iree_hal_buffer_release(backing);
  iree_hal_pool_release(pool);
}

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
  iree_hal_amd_fabric_virtual_mapping_t source;
  for (uint32_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(iree_hal_amd_fabric_virtual_buffer_query_source(
        iree_hal_buffer_virtual_view(reservations[i]), i ? second.id : first.id,
        5000, 8000, &source));
    EXPECT_EQ(source.native.source_byte_offset, 12288u + 904u);
    EXPECT_EQ(source.native.byte_length, 8192u - 904u);
    EXPECT_EQ(source.native.memory_access_ordinal, i);
  }

  // ANY expands distinct projections while deduplicating native aliases.
  mappings[0].scope = {first.domain, IREE_HAL_MEMORY_SCOPE_ANY};
  ++value;
  IREE_ASSERT_OK(iree_hal_queue_map(queue, {}, signals, 1, mappings.data(), 0));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(done, value, iree_infinite_timeout(), 0));
  EXPECT_EQ(native_.update_count, native_.alias_projections ? 3 : 4);
  IREE_ASSERT_OK(iree_hal_amd_fabric_virtual_buffer_query_source(
      iree_hal_buffer_virtual_view(reservations[0]), second.id, 4096, 8192,
      &source));
  EXPECT_EQ(source.native.memory_access_ordinal, 1u);
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
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_amd_fabric_virtual_buffer_query_source(
                            iree_hal_buffer_virtual_view(reservations[0]),
                            first.id, 4096, 1, &source));
  IREE_ASSERT_OK(iree_hal_amd_fabric_virtual_buffer_query_source(
      iree_hal_buffer_virtual_view(reservations[0]), first.id, 0, 16384,
      &source));
  EXPECT_EQ(source.native.byte_length, 4096u);
  // Exercise source-index splits, holes, and coalescing through real queue_map
  // admission. A page assignment oracle is independent of the run
  // representation.
  const uint32_t changes[][3] = {
      {0, 4, 0},          {1, 2, 4},          {0, 1, UINT32_MAX},
      {2, 2, 5},          {1, 3, UINT32_MAX}, {0, 4, 0},
      {1, 1, UINT32_MAX}, {1, 1, 1},          {0, 4, UINT32_MAX},
  };
  std::array<uint32_t, 4> pages;
  pages.fill(UINT32_MAX);
  for (const auto& change : changes) {
    const bool bind = change[2] != UINT32_MAX;
    const iree_hal_memory_mapping_t mapping = {
        .target = reservations[0],
        .scope = first,
        .target_range = {change[0] * 4096u, change[1] * 4096u},
        .source = bind ? backing : nullptr,
        .source_offset = bind ? change[2] * 4096u : 0,
        .access = static_cast<iree_hal_memory_access_t>(
            bind ? IREE_HAL_MEMORY_ACCESS_ALL : 0),
        .operation = bind ? IREE_HAL_MEMORY_MAPPING_OPERATION_REPLACE
                          : IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND,
    };
    ++value;
    IREE_ASSERT_OK(iree_hal_queue_map(queue, {}, signals, 1, &mapping, 0));
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(done, value, iree_infinite_timeout(), 0));
    for (uint32_t p = 0; p < change[1]; ++p) {
      pages[change[0] + p] = bind ? change[2] + p : UINT32_MAX;
    }
    for (uint32_t p = 0; p < pages.size(); ++p) {
      auto result = iree_hal_amd_fabric_virtual_buffer_query_source(
          iree_hal_buffer_virtual_view(reservations[0]), first.id,
          p * 4096u + 123, (4 - p) * 4096u - 123, &source);
      if (pages[p] == UINT32_MAX) {
        IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, result);
      } else {
        IREE_ASSERT_OK(result);
        EXPECT_EQ(source.native.source_byte_offset, pages[p] * 4096u + 123);
        uint32_t end = p + 1;
        while (end < pages.size() && pages[end] == pages[p] + end - p) {
          ++end;
        }
        EXPECT_EQ(source.native.byte_length, (end - p) * 4096u - 123);
      }
    }
  }
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
  iree_hal_amd_fabric_virtual_mapping_t source;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_amd_fabric_virtual_buffer_query_source(
                            iree_hal_buffer_virtual_view(reservation),
                            mapping.scope.id, 0, 1, &source));
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

enum class NativeBatchKind { kAcquire, kRelease, kRollback };

class MappingPriorityTest
    : public MemoryBackendTest,
      public ::testing::WithParamInterface<NativeBatchKind> {};

TEST_P(MappingPriorityTest, MappingOvertakesRemainingNativeSteps) {
  native_.mapping_enabled = true;
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
      pool, iree_hal_make_pool_reservation_request(8192, 4096),
      iree_infinite_timeout(), &backing));
  iree_hal_buffer_t* reservation = nullptr;
  IREE_ASSERT_OK(CreateReservation(&reservation));
  const auto scope = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[0]);
  const iree_hal_memory_mapping_t mappings[] = {
      {
          .target = reservation,
          .scope = scope,
          .target_range = {0, 4096},
          .source = backing,
          .source_offset = 0,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
      },
      {
          .target = reservation,
          .scope = scope,
          .target_range = {4096, 4096},
          .source = backing,
          .source_offset = 4096,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
      },
  };
  iree_hal_memory_mapping_batch_t* batch = nullptr;
  IREE_ASSERT_OK(iree_hal_memory_mapping_batch_create(
      iree_hal_device_queue(cpu_device_, 0, 0), IREE_ARRAYSIZE(mappings),
      mappings, 0, allocator_, &batch));
  IREE_ASSERT_OK(iree_hal_memory_mapping_batch_resolve(batch));

  // Private backing and pool slabs share the same native owner. The test gates
  // real native entry points underneath that owner, not its scheduler.
  iree_hal_amd_fabric_memory_options_t options = {.scope = native_.scope()};
  options.profile.roles = AMDF_MEMORY_PROFILE_ROLE_CREATE;
  options.profile.allocation.byte_length_granularity = 4096;
  options.profile.allocation.minimum_alignment = 4096;
  options.profile.allocation.maximum_byte_length = UINT64_C(1) << 30;
  options.required_flags = native_.required_flags;
  std::array<iree_hal_amd_fabric_memory_request_t, 100> requests;
  std::array<iree_hal_amd_fabric_memory_t, 100> memories = {};
  for (auto& request : requests) {
    request = {&options, 4096, 4096};
  }
  std::promise<iree::Status> acquired;
  iree_hal_amd_fabric_memory_acquire_request_t acquire = {
      .operation =
          {
              .api = &native_.api,
              .event_sink = iree_hal_device_event_sink_discard(),
          },
      .count = requests.size(),
      .requests = requests.data(),
      .memories = memories.data(),
      .completion =
          {
              .fn =
                  [](void* user_data, iree_status_t status) {
                    static_cast<std::promise<iree::Status>*>(user_data)
                        ->set_value(iree::Status(std::move(status)));
                  },
              .user_data = &acquired,
          },
  };
  std::promise<void> released;
  iree_hal_amd_fabric_memory_release_request_t release = {
      .operation =
          {
              .api = &native_.api,
              .event_sink = iree_hal_device_event_sink_discard(),
          },
      .count = memories.size(),
      .memories = memories.data(),
      .completion =
          {
              .fn =
                  [](void* user_data) {
                    static_cast<std::promise<void>*>(user_data)->set_value();
                  },
              .user_data = &released,
          },
  };
  if (GetParam() == NativeBatchKind::kRelease) {
    iree_hal_amd_fabric_allocation_queue_acquire_memory(allocation_queue_,
                                                        &acquire);
    IREE_ASSERT_OK(acquired.get_future().get());
  }

  std::vector<char> sequence;
  std::promise<void> entered;
  std::promise<void> resume;
  auto resumed = resume.get_future().share();
  uint32_t acquisition_count = 0, release_count = 0;
  native_.before_create = [&] {
    sequence.push_back('A');
    if (++acquisition_count == 1 && GetParam() == NativeBatchKind::kAcquire) {
      entered.set_value();
      resumed.wait();
    }
    if (acquisition_count == 3 && GetParam() == NativeBatchKind::kRollback) {
      native_.create_failure =
          amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    }
  };
  native_.before_destroy = [&] {
    sequence.push_back('R');
    if (++release_count == 1 && GetParam() != NativeBatchKind::kAcquire) {
      entered.set_value();
      resumed.wait();
    }
  };
  native_.before_update = [&] { sequence.push_back('M'); };
  if (GetParam() == NativeBatchKind::kRelease) {
    iree_hal_amd_fabric_allocation_queue_release_memory(allocation_queue_,
                                                        &release);
  } else {
    iree_hal_amd_fabric_allocation_queue_acquire_memory(allocation_queue_,
                                                        &acquire);
  }
  entered.get_future().get();
  std::promise<iree::Status> mapped;
  iree_hal_amd_fabric_memory_map_request_t map = {
      .batch = batch,
      .completion =
          {
              .fn =
                  [](void* user_data, iree_status_t status) {
                    static_cast<std::promise<iree::Status>*>(user_data)
                        ->set_value(iree::Status(std::move(status)));
                  },
              .user_data = &mapped,
          },
  };
  iree_hal_amd_fabric_allocation_queue_map_memory(allocation_queue_, &map);
  resume.set_value();
  IREE_EXPECT_OK(mapped.get_future().get());
  if (GetParam() == NativeBatchKind::kRelease) {
    released.get_future().get();
  } else if (GetParam() == NativeBatchKind::kRollback) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                          acquired.get_future().get());
    EXPECT_EQ(sequence, (std::vector<char>{'A', 'A', 'A', 'R', 'M', 'R'}));
    for (const auto& memory : memories) {
      EXPECT_EQ(memory.handle, nullptr);
    }
  } else {
    IREE_EXPECT_OK(acquired.get_future().get());
  }
  const auto mapping_position =
      std::find(sequence.begin(), sequence.end(), 'M');
  EXPECT_EQ(mapping_position - sequence.begin(),
            GetParam() == NativeBatchKind::kRollback ? 4 : 1);
  EXPECT_EQ(native_.update_count, 1);
  EXPECT_EQ(native_.last_updates.size(), 2u);

  native_.before_create = {};
  native_.before_destroy = {};
  native_.before_update = {};
  if (GetParam() == NativeBatchKind::kAcquire) {
    iree_hal_amd_fabric_allocation_queue_release_memory(allocation_queue_,
                                                        &release);
    released.get_future().get();
  }
  iree_hal_memory_mapping_batch_release(batch);
  IREE_EXPECT_OK(iree_hal_virtual_buffer_detach(reservation, scope));
  iree_hal_buffer_release(reservation);
  iree_hal_buffer_release(backing);
  iree_hal_pool_release(pool);
}

INSTANTIATE_TEST_SUITE_P(NativeBatches, MappingPriorityTest,
                         ::testing::Values(NativeBatchKind::kAcquire,
                                           NativeBatchKind::kRelease,
                                           NativeBatchKind::kRollback));

}  // namespace iree::hal::amd::fabric
