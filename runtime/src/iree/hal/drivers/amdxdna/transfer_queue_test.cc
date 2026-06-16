// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/transfer_queue.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/allocator.h"
#include "iree/hal/buffer.h"
#include "iree/hal/utils/memory_file.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_async_proactor_t* TestProactor() {
  static iree_async_proactor_t* proactor = nullptr;
  if (!proactor) {
    IREE_CHECK_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor));
    atexit([] {
      iree_async_proactor_release(proactor);
      proactor = nullptr;
    });
  }
  return proactor;
}

class TransferQueueTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(/*total_block_size=*/8 * 1024,
                                     iree_allocator_system(), &block_pool_);
    IREE_ASSERT_OK(iree_hal_amdxdna_transfer_queue_create(
        &block_pool_, iree_allocator_system(), &queue_));
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(
        iree_make_cstring_view("test-heap"), iree_allocator_system(),
        iree_allocator_system(), &heap_allocator_));
  }

  void TearDown() override {
    iree_hal_allocator_release(heap_allocator_);
    heap_allocator_ = nullptr;
    iree_hal_amdxdna_transfer_queue_destroy(queue_);
    queue_ = nullptr;
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_hal_semaphore_list_t MakeList(std::vector<iree_async_semaphore_t*>* sems,
                                     std::vector<uint64_t>* values) {
    return iree_hal_semaphore_list_t{
        sems->size(),
        reinterpret_cast<iree_hal_semaphore_t**>(sems->data()),
        values->data(),
    };
  }

  iree_async_semaphore_t* MakeSem(uint64_t initial_value) {
    iree_async_semaphore_t* sem = nullptr;
    IREE_CHECK_OK(iree_async_semaphore_create(
        TestProactor(), initial_value,
        IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY, iree_allocator_system(),
        &sem));
    return sem;
  }

  void WaitForValue(iree_async_semaphore_t* sem, uint64_t value) {
    IREE_ASSERT_OK(iree_async_semaphore_multi_wait(
        IREE_ASYNC_WAIT_MODE_ALL, &sem, &value, 1, iree_make_timeout_ms(5000),
        IREE_ASYNC_WAIT_FLAG_NONE, iree_allocator_system()));
  }

  iree_status_code_t WaitForFailure(iree_async_semaphore_t* sem,
                                    uint64_t value) {
    iree_status_code_t code = IREE_STATUS_OK;
    for (int i = 0; i < 100; ++i) {
      code = iree_async_semaphore_query_status(sem);
      if (code != IREE_STATUS_OK) return code;
      iree_status_t status = iree_async_semaphore_multi_wait(
          IREE_ASYNC_WAIT_MODE_ALL, &sem, &value, 1, iree_make_timeout_ms(50),
          IREE_ASYNC_WAIT_FLAG_NONE, iree_allocator_system());
      iree_status_ignore(status);
    }
    return iree_async_semaphore_query_status(sem);
  }

  iree_hal_buffer_t* CreateBuffer(iree_device_size_t length) {
    iree_hal_buffer_params_t params = {};
    params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                  IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                  IREE_HAL_MEMORY_TYPE_HOST_CACHED;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_MAPPING |
                   IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
    iree_hal_buffer_t* buffer = nullptr;
    IREE_CHECK_OK(iree_hal_allocator_allocate_buffer(heap_allocator_, params,
                                                     length, &buffer));
    return buffer;
  }

  iree_hal_file_t* WrapMemoryFile(std::vector<uint8_t>* contents,
                                  iree_hal_memory_access_t access) {
    iree_io_file_handle_t* handle = nullptr;
    IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(contents->data(), contents->size()),
        iree_io_file_handle_release_callback_null(), iree_allocator_system(),
        &handle));
    iree_hal_file_t* file = nullptr;
    IREE_CHECK_OK(iree_hal_memory_file_wrap(
        /*device_allocator=*/nullptr, IREE_HAL_QUEUE_AFFINITY_ANY, access,
        handle, iree_allocator_system(), &file));
    iree_io_file_handle_release(handle);
    return file;
  }

  iree_arena_block_pool_t block_pool_;
  iree_hal_amdxdna_transfer_queue_t* queue_ = nullptr;
  iree_hal_allocator_t* heap_allocator_ = nullptr;
};

TEST_F(TransferQueueTest, ChunkedReadThenWrite) {
  const iree_device_size_t length = 1024 * 1024 + 17;
  const iree_device_size_t source_offset = 31;
  const iree_device_size_t target_offset = length + 32;
  std::vector<uint8_t> file_storage(target_offset + length);
  for (iree_host_size_t i = 0; i < file_storage.size(); ++i) {
    file_storage[i] = static_cast<uint8_t>((i * 13 + 7) & 0xFF);
  }
  iree_hal_file_t* file =
      WrapMemoryFile(&file_storage, IREE_HAL_MEMORY_ACCESS_ALL);
  iree_hal_buffer_t* buffer = CreateBuffer(length);

  iree_async_semaphore_t* sem = MakeSem(0);
  std::vector<iree_async_semaphore_t*> read_signal_sems = {sem};
  std::vector<uint64_t> read_signal_values = {1};
  std::vector<iree_async_semaphore_t*> write_wait_sems = {sem};
  std::vector<uint64_t> write_wait_values = {1};
  std::vector<iree_async_semaphore_t*> write_signal_sems = {sem};
  std::vector<uint64_t> write_signal_values = {2};

  IREE_ASSERT_OK(iree_hal_amdxdna_transfer_queue_read(
      queue_, iree_hal_semaphore_list_empty(),
      MakeList(&read_signal_sems, &read_signal_values), file, source_offset,
      buffer, /*target_offset=*/0, length));
  IREE_ASSERT_OK(iree_hal_amdxdna_transfer_queue_write(
      queue_, MakeList(&write_wait_sems, &write_wait_values),
      MakeList(&write_signal_sems, &write_signal_values), buffer,
      /*source_offset=*/0, file, target_offset, length));
  WaitForValue(sem, 2);

  EXPECT_EQ(std::memcmp(file_storage.data() + source_offset,
                        file_storage.data() + target_offset, length),
            0);

  iree_async_semaphore_release(sem);
  iree_hal_buffer_release(buffer);
  iree_hal_file_release(file);
}

TEST_F(TransferQueueTest, WorkerFailureFailsSignalList) {
  std::vector<uint8_t> file_storage(64, 0xA5);
  iree_hal_file_t* file =
      WrapMemoryFile(&file_storage, IREE_HAL_MEMORY_ACCESS_READ);
  iree_hal_buffer_t* buffer = CreateBuffer(16);

  iree_async_semaphore_t* signal_sem = MakeSem(0);
  std::vector<iree_async_semaphore_t*> signal_sems = {signal_sem};
  std::vector<uint64_t> signal_values = {1};

  IREE_ASSERT_OK(iree_hal_amdxdna_transfer_queue_read(
      queue_, iree_hal_semaphore_list_empty(),
      MakeList(&signal_sems, &signal_values), file, /*source_offset=*/0, buffer,
      /*target_offset=*/0, /*length=*/64));

  EXPECT_NE(WaitForFailure(signal_sem, 1), IREE_STATUS_OK);
  EXPECT_EQ(iree_async_semaphore_query(signal_sem), 0u);

  iree_async_semaphore_release(signal_sem);
  iree_hal_buffer_release(buffer);
  iree_hal_file_release(file);
}

TEST_F(TransferQueueTest, UnresolvedWaitDoesNotBlockReadyTransfer) {
  std::vector<uint8_t> file_storage(128);
  for (iree_host_size_t i = 0; i < file_storage.size(); ++i) {
    file_storage[i] = static_cast<uint8_t>(i);
  }
  iree_hal_file_t* file =
      WrapMemoryFile(&file_storage, IREE_HAL_MEMORY_ACCESS_READ);
  iree_hal_buffer_t* blocked_buffer = CreateBuffer(16);
  iree_hal_buffer_t* ready_buffer = CreateBuffer(16);

  iree_async_semaphore_t* wait_sem = MakeSem(0);
  iree_async_semaphore_t* blocked_signal = MakeSem(0);
  iree_async_semaphore_t* ready_signal = MakeSem(0);
  std::vector<iree_async_semaphore_t*> wait_sems = {wait_sem};
  std::vector<uint64_t> wait_values = {1};
  std::vector<iree_async_semaphore_t*> blocked_signal_sems = {blocked_signal};
  std::vector<uint64_t> blocked_signal_values = {1};
  std::vector<iree_async_semaphore_t*> ready_signal_sems = {ready_signal};
  std::vector<uint64_t> ready_signal_values = {1};

  IREE_ASSERT_OK(iree_hal_amdxdna_transfer_queue_read(
      queue_, MakeList(&wait_sems, &wait_values),
      MakeList(&blocked_signal_sems, &blocked_signal_values), file,
      /*source_offset=*/0, blocked_buffer, /*target_offset=*/0, 16));
  IREE_ASSERT_OK(iree_hal_amdxdna_transfer_queue_read(
      queue_, iree_hal_semaphore_list_empty(),
      MakeList(&ready_signal_sems, &ready_signal_values), file,
      /*source_offset=*/32, ready_buffer, /*target_offset=*/0, 16));

  WaitForValue(ready_signal, 1);
  std::vector<uint8_t> ready_contents(16);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      ready_buffer, 0, ready_contents.data(), ready_contents.size()));
  EXPECT_EQ(std::memcmp(ready_contents.data(), file_storage.data() + 32,
                        ready_contents.size()),
            0);
  EXPECT_EQ(iree_async_semaphore_query(blocked_signal), 0u);

  IREE_ASSERT_OK(iree_async_semaphore_signal(wait_sem, 1, nullptr));
  WaitForValue(blocked_signal, 1);

  iree_async_semaphore_release(ready_signal);
  iree_async_semaphore_release(blocked_signal);
  iree_async_semaphore_release(wait_sem);
  iree_hal_buffer_release(ready_buffer);
  iree_hal_buffer_release(blocked_buffer);
  iree_hal_file_release(file);
}

}  // namespace
