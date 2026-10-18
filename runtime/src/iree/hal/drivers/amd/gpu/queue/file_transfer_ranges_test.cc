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
#include "iree/hal/drivers/amd/gpu/queue/file_transfer.h"
#include "iree/hal/utils/fd_file.h"
#include "iree/io/file_contents.h"
#include "iree/testing/temp_file.h"

namespace {
using iree::hal::amd::fabric::FakeNative;
using iree::hal::amd::fabric::MemoryBackendTest;
using iree::hal::cts::Ref;

// Models only native execution. The physical pools, virtual mappings, file
// proactor, source resolution, transfer pipeline, and semaphores are real.
struct MappedQueue : iree_hal_queue_t {
  // Exact byte ranges handed to the device copy executor.
  std::vector<iree_hal_buffer_range_t> copies;
  // Aggregate producer boundaries submitted before touching any source.
  uint32_t before_count = 0;
  // Aggregate consumer boundaries submitted after every source retires.
  uint32_t after_count = 0;

  static iree_status_t Submit(iree_hal_queue_t* base,
                              const iree_hal_transfer_operation_t* operation,
                              iree_hal_semaphore_list_t signals,
                              const void* before, const void* after) {
    auto* queue = static_cast<MappedQueue*>(base);
    queue->before_count += before != nullptr;
    queue->after_count += after != nullptr;
    if (operation) {
      const bool upload =
          operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
      EXPECT_TRUE(upload ||
                  operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD);
      iree_hal_buffer_t* buffer = upload ? operation->upload.target_buffer
                                         : operation->download.source_buffer;
      const uint64_t offset = upload ? operation->upload.target_offset
                                     : operation->download.source_offset;
      const uint64_t length =
          upload ? operation->upload.length : operation->download.length;
      queue->copies.push_back({offset, length});
      iree_status_t status = iree_ok_status();
      for (uint64_t position = 0;
           position < length && iree_status_is_ok(status);) {
        iree_hal_amd_fabric_virtual_mapping_t mapping;
        status = iree_hal_amd_fabric_virtual_buffer_query_source(
            iree_hal_buffer_virtual_view(buffer),
            queue->queue_family->memory.queue_scope_id, offset + position,
            length - position, &mapping);
        if (iree_status_is_ok(status)) {
          auto* source =
              reinterpret_cast<FakeNative::Memory*>(mapping.native.memory);
          auto* data = static_cast<uint8_t*>(source->data) +
                       mapping.native.source_byte_offset;
          if (upload) {
            std::memcpy(data,
                        static_cast<const uint8_t*>(operation->upload.source) +
                            position,
                        mapping.native.byte_length);
          } else {
            std::memcpy(
                static_cast<uint8_t*>(operation->download.target) + position,
                data, mapping.native.byte_length);
          }
          position += mapping.native.byte_length;
        }
      }
      IREE_RETURN_IF_ERROR(status);
    }
    return iree_hal_semaphore_list_signal(signals, nullptr);
  }
};

class FileTransferRangesTest
    : public MemoryBackendTest,
      public ::testing::WithParamInterface<std::tuple<uint32_t, bool>> {};

TEST_P(FileTransferRangesTest, CopiesOnlySourcesWithoutFileViews) {
  static_assert(sizeof(iree_hal_amd_fabric_file_range_t) <= 64);
  RecordProperty("source_record_bytes",
                 sizeof(iree_hal_amd_fabric_file_range_t));
  constexpr uint64_t kUnit = 4096;
  constexpr uint64_t kBufferOffset = 128;
  constexpr uint64_t kFileOffset = 64;
  constexpr uint64_t kLength = 3 * kUnit - 256;
  const auto [host_mask, write] = GetParam();
  native_.required_flags = 0;
  native_.mapping_enabled = true;
  native_.api.memory_query_pair_info =
      +[](const amdf_memory_site_t* producer,
          const amdf_memory_site_t* consumer,
          amdf_memory_pair_info_t* out_info) -> amdf_status_t {
    const auto* device =
        producer->kind == AMDF_MEMORY_SITE_KIND_DEVICE ? producer : consumer;
    const auto* host = device == producer ? consumer : producer;
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
  std::array<Ref<iree_hal_pool_t>, 2> pools;
  for (uint32_t i = 0; i < pools.size(); ++i) {
    native_.host_mappable = i != 0;
    native_.host_visibility_guaranteed = i != 0;
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
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pools[(host_mask >> i) & 1],
        iree_hal_make_pool_reservation_request(2 * kUnit, kUnit),
        iree_infinite_timeout(), sources[i].out()));
    const auto memory = iree_hal_buffer_memory_view(sources[i]);
    const auto* facts =
        reinterpret_cast<const iree_hal_amd_fabric_memory_binding_facts_t*>(
            memory.bindings + memory.contract->binding_layout.binding_count);
    auto* native = reinterpret_cast<FakeNative::Memory*>(facts->memory->handle);
    source_contents[i] = static_cast<uint8_t*>(native->data);
    std::memset(source_contents[i], 0xCD, 2 * kUnit);
    EXPECT_EQ(facts->memory->mapping != nullptr, ((host_mask >> i) & 1) != 0);
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
  uint64_t mapped = 1, ready = 2, completed = 3;
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

  iree_async_semaphore_wait_context_t waits;
  iree_async_semaphore_wait_context_initialize(&waits);
  iree_hal_amd_gpu_capture_pool_t captures;
  iree_hal_amd_gpu_capture_pool_initialize(4096, allocator_, &captures);
  iree_hal_queue_family_t family = *families_[0];
  family.device = cpu_device_;
  MappedQueue queue;
  queue.queue_family = &family;
  iree_hal_amd_gpu_context_t context = {.api = &native_.api};
  const iree_hal_amd_gpu_file_executor_t executor = {
      .queue = &queue,
      .context = &context,
      .captures = &captures,
      .waits = &waits,
      .native_family_ordinal = native_families_[0].queue_family_ordinal,
      .barrier_size = sizeof(uint64_t),
      .add_effects = +[](iree_hal_queue_t*, iree_hal_memory_effects_t effects,
                         void*) { EXPECT_EQ(effects.bits, 0u); },
      .submit = MappedQueue::Submit,
  };
  const uint64_t before = 17, after = 29;
  IREE_ASSERT_OK(iree_hal_amd_gpu_file_transfer_submit(
      &executor, {1, &semaphore, &ready}, {1, &semaphore, &completed}, file,
      write ? IREE_HAL_MEMORY_ACCESS_WRITE : IREE_HAL_MEMORY_ACCESS_READ,
      kFileOffset, target, kBufferOffset, kLength, &before, &after));
  target.reset();
  file.reset();
  EXPECT_TRUE(queue.copies.empty());
  EXPECT_EQ(queue.before_count, 0u);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(timeline, ready, nullptr));
  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(timeline, &value));
  while (value < completed) {
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
    IREE_ASSERT_OK(iree_hal_semaphore_query(timeline, &value));
  }
  iree_hal_amd_gpu_capture_pool_deinitialize(&captures);
  iree_async_semaphore_wait_context_deinitialize(&waits);
  iree_async_proactor_end_polling(proactor);
  iree_async_proactor_release(proactor);
  EXPECT_EQ(queue.before_count, 1u);
  EXPECT_EQ(queue.after_count, 1u);
  size_t copy_index = 0;
  for (uint32_t i = 0; i < sources.size(); ++i) {
    const uint64_t start = std::max(i * kUnit, kBufferOffset);
    const uint64_t end = std::min((i + 1) * kUnit, kBufferOffset + kLength);
    if (!(host_mask & (1u << i))) {
      ASSERT_LT(copy_index, queue.copies.size());
      EXPECT_EQ(queue.copies[copy_index].byte_offset, start);
      EXPECT_EQ(queue.copies[copy_index++].byte_length, end - start);
    }
    for (uint64_t j = 0; j < 2 * kUnit; ++j) {
      const uint64_t position = i * kUnit + j;
      const uint8_t expected_byte =
          j >= kUnit && position - kUnit >= kBufferOffset &&
                  position - kUnit < kBufferOffset + kLength
              ? expected[kFileOffset + position - kUnit - kBufferOffset]
              : uint8_t{0xCD};
      EXPECT_EQ(source_contents[i][j], expected_byte) << i << ':' << j;
    }
  }
  EXPECT_EQ(copy_index, queue.copies.size());
  iree_io_file_contents_t* actual = nullptr;
  IREE_ASSERT_OK(
      iree_io_file_contents_read(file_path.path_view(), allocator_, &actual));
  ASSERT_EQ(actual->const_buffer.data_length, expected.size());
  EXPECT_EQ(
      std::memcmp(actual->const_buffer.data, expected.data(), expected.size()),
      0);
  iree_io_file_contents_free(actual);
  EXPECT_TRUE(file_path.Remove());
}

INSTANTIATE_TEST_SUITE_P(Routes, FileTransferRangesTest,
                         ::testing::Combine(::testing::Range(0u, 8u),
                                            ::testing::Bool()));

}  // namespace
