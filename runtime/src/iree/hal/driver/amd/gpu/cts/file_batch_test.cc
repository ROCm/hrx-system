// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <vector>

#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/driver/amd/gpu/cts/test_device.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {
using iree::hal::cts::Ref;

static iree_status_t TransferFiles(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_file_binding_table_t files,
    iree_hal_buffer_binding_table_t buffers,
    const iree_hal_file_transfer_list_t* transfers,
    iree_hal_memory_access_t access,
    const iree_hal_queue_barriers_t* barriers) {
  return access == IREE_HAL_MEMORY_ACCESS_READ
             ? iree_hal_queue_read(queue, waits, signals, files, buffers,
                                   transfers, barriers, 0)
             : iree_hal_queue_write(queue, waits, signals, buffers, files,
                                    transfers, barriers, 0);
}

enum class Descriptions { kHost, kIndirectHost, kIndirectPrivate };
enum class FileStorage { kNative, kHost, kDevice };

struct HostStorage {
  ~HostStorage() { iree_allocator_free_aligned(iree_allocator_system(), data); }
  // Borrowed by memory-file handles through the accepted queue completion.
  uint8_t* data = nullptr;
  // Full imported extent, including untouched allocation-granularity padding.
  iree_host_size_t length = 0;
};

class GpuNativeFileBatchTest : public iree::hal::amd::gpu::cts::GpuNativeTest {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return AMDF_QUEUE_COMMAND_TYPE_UNKNOWN;
  }

  void Import(const iree::testing::TempFilePath& path,
              const std::vector<uint8_t>& contents,
              iree_hal_file_t** out_file) {
    IREE_ASSERT_OK(iree_io_file_contents_write(
        path.path_view(),
        iree_make_const_byte_span(contents.data(), contents.size()),
        iree_allocator_system()));
    iree_io_file_handle_t* handle = nullptr;
    IREE_ASSERT_OK(iree_io_file_handle_open(
        IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
            IREE_IO_FILE_MODE_ASYNC | IREE_IO_FILE_MODE_SHARE_READ |
            IREE_IO_FILE_MODE_SHARE_WRITE,
        path.path_view(), iree_allocator_system(), &handle));
    iree_status_t status = iree_hal_file_import(
        device_, nullptr, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        IREE_HAL_MEMORY_ACCESS_ALL, handle, 0, out_file);
    iree_io_file_handle_release(handle);
    IREE_ASSERT_OK(status);
  }

  void Allocate(iree_hal_pool_t* pool, size_t length,
                iree_hal_buffer_t** out_buffer) {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool, iree_hal_make_pool_reservation_request(length, 64),
        iree_infinite_timeout(), out_buffer));
  }

  void ImportMemory(iree_hal_pool_t* pool, const std::vector<uint8_t>& contents,
                    iree_hal_memory_access_t access, HostStorage& storage,
                    iree_hal_file_t** out_file) {
    iree_hal_pool_import_requirements_t requirements = {
        .min_byte_length = 1,
        .base_alignment = 64,
        .byte_length_granularity = 1,
    };
    if (pool) {
      ASSERT_TRUE(iree_hal_pool_query_import_requirements(
          pool,
          iree_hal_make_external_storage_class(
              IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
          &requirements));
    }
    uint64_t length = 0;
    ASSERT_TRUE(iree_checked_add_u64(
        iree_max(contents.size(), requirements.min_byte_length),
        requirements.byte_length_granularity - 1, &length));
    length -= length % requirements.byte_length_granularity;
    ASSERT_LE(length, IREE_HOST_SIZE_MAX);
    storage.length = static_cast<iree_host_size_t>(length);
    IREE_ASSERT_OK(iree_allocator_malloc_aligned(
        iree_allocator_system(), storage.length,
        iree_max(iree_host_size_t{64}, requirements.base_alignment), 0,
        reinterpret_cast<void**>(&storage.data)));
    std::memset(storage.data, 0xCD, storage.length);
    std::memcpy(storage.data, contents.data(), contents.size());
    iree_io_file_handle_t* handle = nullptr;
    IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
        access == IREE_HAL_MEMORY_ACCESS_READ ? IREE_IO_FILE_ACCESS_READ
                                              : IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(storage.data, storage.length), {},
        iree_allocator_system(), &handle));
    iree_status_t status =
        iree_hal_file_import(device_, pool, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                             access, handle, 0, out_file);
    iree_io_file_handle_release(handle);
    IREE_ASSERT_OK(status);
    if (pool) {
      ASSERT_NE(iree_hal_file_storage_buffer(*out_file), nullptr);
      EXPECT_TRUE(iree_all_bits_set(
          iree_hal_buffer_memory_type(iree_hal_file_storage_buffer(*out_file)),
          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
    }
  }

  void RoundTrip(iree_hal_queue_t* io_queue, iree_hal_pool_t* payload_pool,
                 iree_hal_pool_t* metadata_pool, iree_hal_pool_t* import_pool,
                 Descriptions descriptions, FileStorage storage) {
    SCOPED_TRACE(static_cast<int>(descriptions));
    SCOPED_TRACE(static_cast<int>(storage));
    const auto transfer = TransferFiles;
    // Cross both native request-group and private metadata-page boundaries.
    constexpr size_t kCount = 65;
    constexpr size_t kCapacity = kCount + 3;
    constexpr size_t kRowLength = 90;
    constexpr size_t kFileOrigin = 13;
    constexpr size_t kFileStride = 192;
    constexpr size_t kBufferOrigin = 17;
    constexpr size_t kBufferStride = 128;
    constexpr size_t kFileLength = kFileOrigin + kCount * kFileStride + 7;
    constexpr size_t kBufferLength = kBufferOrigin + kCount * kBufferStride + 5;
    constexpr size_t kMetadataLength = 64 + kCapacity * 64;
    std::vector<uint8_t> source(kFileLength);
    std::vector<uint8_t> output(kFileLength, 0xCD);
    std::vector<uint8_t> expected_output = output;
    std::vector<uint8_t> expected_buffer(kBufferLength, 0xCD);
    std::vector<uint8_t> observed(kBufferLength, 0xFF);
    for (size_t i = 0; i < source.size(); ++i) {
      source[i] = static_cast<uint8_t>((i * 17) ^ (i >> 8));
    }
    iree::testing::TempFilePath source_path("iree_file_batch_source");
    iree::testing::TempFilePath output_path("iree_file_batch_output");
    HostStorage source_storage, output_storage;
    Ref<iree_hal_file_t> source_file, output_file;
    if (storage == FileStorage::kNative) {
      ASSERT_NO_FATAL_FAILURE(Import(source_path, source, source_file.out()));
      ASSERT_NO_FATAL_FAILURE(Import(output_path, output, output_file.out()));
    } else {
      auto* pool = storage == FileStorage::kDevice ? import_pool : nullptr;
      ASSERT_NO_FATAL_FAILURE(ImportMemory(pool, source,
                                           IREE_HAL_MEMORY_ACCESS_READ,
                                           source_storage, source_file.out()));
      ASSERT_NO_FATAL_FAILURE(ImportMemory(pool, output,
                                           IREE_HAL_MEMORY_ACCESS_WRITE,
                                           output_storage, output_file.out()));
    }
    Ref<iree_hal_buffer_t> payload, consumer, metadata, independent;
    std::array<uint32_t, 4> independent_values = {};
    ASSERT_NO_FATAL_FAILURE(
        Allocate(payload_pool, kBufferLength, payload.out()));
    ASSERT_NO_FATAL_FAILURE(
        Allocate(payload_pool, kBufferLength, consumer.out()));
    ASSERT_NO_FATAL_FAILURE(
        Allocate(payload_pool, sizeof(independent_values), independent.out()));
    ASSERT_NO_FATAL_FAILURE(
        Allocate(metadata_pool, kMetadataLength, metadata.out()));

    // The metadata starts invalid. The accepted producer supplies the live
    // count and records only after all downstream work has been enqueued.
    std::array<uint64_t, kMetadataLength / sizeof(uint64_t)> words;
    words.fill(UINT64_MAX);
    std::array<iree_hal_file_transfer_t, kCount> host = {};
    for (size_t i = 0; i < kCount; ++i) {
      const size_t file_offset = ((i * 37) % kCount) * kFileStride;
      const size_t buffer_offset = i * kBufferStride;
      auto* record = &words[8 + i * 8];
      record[0] = file_offset;
      record[1] = buffer_offset;
      record[2] = kRowLength;
      host[i] = {
          .file_offset = file_offset,
          .buffer_offset = buffer_offset,
          .row_length = kRowLength,
          .row_count = 1,
          .slice_count = 1,
      };
      std::memcpy(expected_buffer.data() + kBufferOrigin + buffer_offset,
                  source.data() + kFileOrigin + file_offset, kRowLength);
      std::memcpy(expected_output.data() + kFileOrigin + file_offset,
                  source.data() + kFileOrigin + file_offset, kRowLength);
    }
    words[0] = kCount;

    Ref<iree_hal_semaphore_t> initialized, gate, requests_ready, rows_ready;
    Ref<iree_hal_semaphore_t> consumed, written, completed, independent_done;
    ASSERT_NO_FATAL_FAILURE(CreateSemaphore(initialized.out()));
    ASSERT_NO_FATAL_FAILURE(CreateSemaphore(gate.out()));
    ASSERT_NO_FATAL_FAILURE(CreateSemaphore(requests_ready.out()));
    ASSERT_NO_FATAL_FAILURE(CreateSemaphore(rows_ready.out()));
    ASSERT_NO_FATAL_FAILURE(CreateSemaphore(consumed.out()));
    ASSERT_NO_FATAL_FAILURE(CreateSemaphore(written.out()));
    ASSERT_NO_FATAL_FAILURE(CreateSemaphore(completed.out()));
    ASSERT_NO_FATAL_FAILURE(CreateSemaphore(independent_done.out()));
    auto* initialized_handle = initialized.get();
    auto* requests_handle = requests_ready.get();
    auto* rows_handle = rows_ready.get();
    auto* consumed_handle = consumed.get();
    auto* written_handle = written.get();
    auto* completed_handle = completed.get();
    const uint8_t padding = 0xCD;
    const uint8_t invalid = 0xFF;
    uint64_t one = 1;
    const iree_hal_fill_region_t payload_region = {kBufferLength, 1, 1};
    const iree_hal_fill_region_t metadata_region = {kMetadataLength, 1, 1};
    iree_hal_transfer_operation_t initialization[2] = {};
    initialization[0].type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
    initialization[0].fill = {payload, 0, &payload_region, &padding, 1};
    initialization[1].type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
    initialization[1].fill = {metadata, 0, &metadata_region, &invalid, 1};
    IREE_ASSERT_OK(iree_hal_queue_transfer(queues_[0], {},
                                           {1, &initialized_handle, &one}, 2,
                                           initialization, nullptr));
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(initialized, one, iree_infinite_timeout(), 0));
    iree_hal_semaphore_t* producer_waits[] = {initialized, gate};
    uint64_t producer_values[] = {one, one};
    IREE_ASSERT_OK(iree_hal_queue_upload_linear(
        queues_[0], {2, producer_waits, producer_values},
        {1, &requests_handle, &one}, words.data(), metadata, 0, sizeof(words),
        nullptr));

    iree_hal_file_binding_t input_binding = {
        source_file,
        kFileOrigin,
        kCount * kFileStride,
    };
    iree_hal_file_binding_t output_binding = {
        output_file,
        kFileOrigin,
        kCount * kFileStride,
    };
    iree_hal_buffer_binding_t payload_binding = {
        payload,
        kBufferOrigin,
        kCount * kBufferStride,
    };
    iree_hal_buffer_binding_t consumer_binding = {
        consumer,
        kBufferOrigin,
        kCount * kBufferStride,
    };
    iree_hal_file_transfer_list_t list = {};
    if (descriptions == Descriptions::kHost) {
      list.host = {host.size(), host.data()};
    } else {
      list.type = IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT;
      list.indirect.records = {metadata, 64, kCapacity * 64};
      list.indirect.count_buffer = metadata;
    }
    // Do not ASSERT after accepting gated work: release the gate and join its
    // accepted tail even if a later admission fails.
    iree_status_t status =
        transfer(io_queue, {1, &requests_handle, &one}, {1, &rows_handle, &one},
                 {1, &input_binding}, {1, &payload_binding}, &list,
                 IREE_HAL_MEMORY_ACCESS_READ, nullptr);
    iree_hal_semaphore_t* tail = requests_handle;
    if (iree_status_is_ok(status)) {
      tail = rows_handle;
      status = iree_hal_queue_copy_linear(
          queues_[0], {1, &rows_handle, &one}, {1, &consumed_handle, &one},
          payload, 0, consumer, 0, kBufferLength, nullptr, 0);
    }
    if (iree_status_is_ok(status)) {
      tail = consumed_handle;
      status = transfer(io_queue, {1, &consumed_handle, &one},
                        {1, &written_handle, &one}, {1, &output_binding},
                        {1, &consumer_binding}, &list,
                        IREE_HAL_MEMORY_ACCESS_WRITE, nullptr);
    }
    if (iree_status_is_ok(status)) {
      tail = written_handle;
      status = iree_hal_queue_download_linear(
          queues_[0], {1, &written_handle, &one}, {1, &completed_handle, &one},
          consumer, 0, observed.data(), observed.size(), nullptr);
    }
    if (iree_status_is_ok(status)) {
      tail = completed_handle;
    }
    // An accepted file operation waiting on its producer cannot occupy the
    // native FIFO ahead of unrelated work, including work on this exact queue.
    auto* independent_handle = independent_done.get();
    uint64_t independent_value = 0, next = 1;
    const uint32_t pattern = 0x23A56BCD;
    if (iree_status_is_ok(status)) {
      status = iree_hal_queue_fill_linear(
          io_queue, {}, {1, &independent_handle, &next}, independent, 0,
          sizeof(independent_values), &pattern, sizeof(pattern), nullptr, 0);
    }
    if (iree_status_is_ok(status)) {
      independent_value = next++;
      status = iree_hal_queue_download_linear(
          io_queue, {1, &independent_handle, &independent_value},
          {1, &independent_handle, &next}, independent, 0,
          independent_values.data(), sizeof(independent_values), nullptr);
      if (iree_status_is_ok(status)) {
        independent_value = next;
      }
    }
    if (independent_value) {
      status = iree_status_join(
          status, iree_hal_semaphore_wait(independent_done, independent_value,
                                          iree_infinite_timeout(), 0));
    }
    if (iree_status_is_ok(status)) {
      for (auto value : independent_values) {
        EXPECT_EQ(value, pattern);
      }
    }
    // Captures own the tables and objects. The caller owns the pools and the
    // upload/download host spans until the accepted tail retires.
    input_binding = {};
    output_binding = {};
    payload_binding = {};
    consumer_binding = {};
    list = {};
    host.fill({});
    source_file.reset();
    output_file.reset();
    payload.reset();
    consumer.reset();
    metadata.reset();
    IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, one, nullptr));
    status = iree_status_join(
        status, iree_hal_semaphore_wait(tail, one, iree_infinite_timeout(), 0));
    IREE_ASSERT_OK(status);
    EXPECT_EQ(observed, expected_buffer);
    if (storage == FileStorage::kNative) {
      iree_io_file_contents_t* actual = nullptr;
      IREE_ASSERT_OK(iree_io_file_contents_read(
          output_path.path_view(), iree_allocator_system(), &actual));
      EXPECT_EQ(actual->buffer.data_length, expected_output.size());
      if (actual->buffer.data_length == expected_output.size()) {
        EXPECT_EQ(std::memcmp(actual->buffer.data, expected_output.data(),
                              expected_output.size()),
                  0);
      }
      iree_io_file_contents_free(actual);
      EXPECT_TRUE(source_path.Remove());
      EXPECT_TRUE(output_path.Remove());
    } else {
      EXPECT_EQ(std::memcmp(output_storage.data, expected_output.data(),
                            expected_output.size()),
                0);
      for (size_t i = expected_output.size(); i < output_storage.length; ++i) {
        ASSERT_EQ(output_storage.data[i], 0xCD) << i;
      }
    }
  }
};

TEST_F(GpuNativeFileBatchTest, HostNumaRequestsFeedNativeFiles) {
  ASSERT_NO_FATAL_FAILURE(Create());
  const auto* topology = iree_hal_device_group_topology(group_);
  const uint32_t node_id = iree_numa_node_for_current_thread();
  uint32_t host_node = IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID;
  for (iree_host_size_t i = 0; i < iree_hal_topology_node_count(topology);
       ++i) {
    const auto* node = iree_hal_topology_node_at(topology, i);
    if (node->kind == IREE_HAL_TOPOLOGY_NODE_KIND_HOST_NUMA &&
        node->local_ordinal == node_id) {
      host_node = node->ordinal;
      break;
    }
  }
  ASSERT_NE(host_node, IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID);
  const iree_hal_pool_family_access_t family = {
      .family = iree_hal_queue_family(queues_[0]),
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
  };
  iree_hal_pool_scope_t scope = {1, &family, {}};
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  Ref<iree_hal_pool_t> payload_pool, metadata_pool;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group_, scope, &options, iree_allocator_system(), payload_pool.out()));
  scope.host = {
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .modes = IREE_HAL_MAPPING_MODE_SCOPED,
      .properties = IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED,
  };
  options.placement = {IREE_HAL_POOL_PLACEMENT_REQUIRED, host_node};
  auto status = iree_hal_slab_pool_create(
      group_, scope, &options, iree_allocator_system(), metadata_pool.out());
  if (iree_status_is_unavailable(status)) {
    iree_status_free(status);
    GTEST_SKIP() << "native host NUMA backing is unavailable";
  }
  IREE_ASSERT_OK(status);
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(metadata_pool, &capabilities);
  EXPECT_EQ(capabilities.placement.mode, IREE_HAL_POOL_PLACEMENT_REQUIRED);
  EXPECT_EQ(capabilities.placement.node, host_node);
  EXPECT_TRUE(iree_all_bits_set(
      capabilities.memory_type,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  EXPECT_TRUE(iree_all_bits_set(
      iree_hal_pool_query_host_access(metadata_pool).properties,
      IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED));
  ASSERT_NO_FATAL_FAILURE(RoundTrip(queues_[1], payload_pool, metadata_pool,
                                    nullptr, Descriptions::kIndirectHost,
                                    FileStorage::kNative));
}

TEST_F(GpuNativeFileBatchTest, ProducerAndConsumerShareOrForkNativeQueues) {
  ASSERT_NO_FATAL_FAILURE(Create());
  Ref<iree_hal_queue_t> dma;
  const auto* families =
      iree_hal_device_spec_queues(iree_hal_device_spec(device_));
  for (uint32_t i = 0; i < families->family_count && !dma; ++i) {
    const auto roles = families->families[i].role_flags;
    if ((roles & IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER) &&
        !(roles & IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH)) {
      iree_hal_queue_params_t params;
      iree_hal_queue_params_initialize(&params);
      IREE_ASSERT_OK(iree_hal_queue_acquire(
          iree_hal_device_queue_family(device_, i), &params, dma.out()));
    }
  }
  std::vector<iree_hal_pool_family_access_t> accesses = {{
      .family = iree_hal_queue_family(queues_[0]),
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
  }};
  if (dma) {
    accesses.push_back({
        .family = iree_hal_queue_family(dma),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
    });
  }
  iree_hal_pool_scope_t scope = {accesses.size(), accesses.data(), {}};
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  Ref<iree_hal_pool_t> private_pool, host_pool;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group_, scope, &options, iree_allocator_system(), private_pool.out()));
  scope.host = {
      .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      .modes = IREE_HAL_MAPPING_MODE_SCOPED,
  };
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group_, scope, &options, iree_allocator_system(), host_pool.out()));
  const iree_hal_pool_import_class_t import_class = {
      .storage_class = iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
  };
  options.owned.mode = IREE_HAL_SLAB_POOL_IMPORT_ONLY;
  options.imports = {1, &import_class};
  Ref<iree_hal_pool_t> import_pool;
  auto import_status = iree_hal_slab_pool_create(
      group_, scope, &options, iree_allocator_system(), import_pool.out());
  if (iree_status_is_unavailable(import_status) ||
      iree_status_is_unimplemented(import_status)) {
    // Some native platforms only support staged host memory files.
    iree_status_free(import_status);
  } else {
    IREE_ASSERT_OK(import_status);
  }
  std::vector<iree_hal_queue_t*> io_queues = {queues_[0], queues_[1]};
  if (dma) {
    io_queues.push_back(dma);
  }
  RecordProperty("sdma_queue", dma ? 1 : 0);
  RecordProperty("imported_memory_file", import_pool ? 1 : 0);
  for (size_t i = 0; i < io_queues.size(); ++i) {
    SCOPED_TRACE(i);
    for (auto descriptions : {Descriptions::kHost, Descriptions::kIndirectHost,
                              Descriptions::kIndirectPrivate}) {
      for (auto storage :
           {FileStorage::kNative, FileStorage::kHost, FileStorage::kDevice}) {
        if (storage == FileStorage::kDevice && !import_pool) {
          continue;
        }
        ASSERT_NO_FATAL_FAILURE(RoundTrip(
            io_queues[i], private_pool,
            descriptions == Descriptions::kIndirectPrivate ? private_pool.get()
                                                           : host_pool.get(),
            import_pool, descriptions, storage));
      }
    }
  }
}

}  // namespace
