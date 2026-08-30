// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/parameter_index_provider.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/task/registration/driver_module.h"
#include "iree/io/file_handle.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_provider.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

template <typename T, void (*Release)(T*)>
class Ref {
 public:
  Ref() = default;
  Ref(const Ref&) = delete;
  Ref& operator=(const Ref&) = delete;

  ~Ref() { reset(); }

  T* get() const { return value_; }

  T** out() {
    reset();
    return &value_;
  }

  void reset(T* value = nullptr) {
    if (value_) Release(value_);
    value_ = value;
  }

 private:
  T* value_ = nullptr;
};

static iree_hal_device_group_t* CreateTaskDeviceGroup() {
  Ref<iree_async_proactor_pool_t, iree_async_proactor_pool_release>
      proactor_pool;
  IREE_CHECK_OK(iree_async_proactor_pool_create(
      /*node_count=*/1, /*node_ids=*/nullptr,
      iree_async_proactor_pool_options_default(), iree_allocator_system(),
      proactor_pool.out()));

  iree_hal_driver_registry_t* registry = nullptr;
  IREE_CHECK_OK(
      iree_hal_driver_registry_allocate(iree_allocator_system(), &registry));
  IREE_CHECK_OK(iree_hal_task_driver_module_register(registry));

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool.get();

  iree_hal_device_t* device = nullptr;
  iree_status_t status =
      iree_hal_create_device(registry, IREE_SV("task"), &create_params,
                             iree_allocator_system(), &device);
  iree_hal_driver_registry_free(registry);
  IREE_CHECK_OK(status);

  Ref<iree_async_frontier_tracker_t, iree_async_frontier_tracker_release>
      frontier_tracker;
  IREE_CHECK_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      frontier_tracker.out()));

  iree_hal_device_group_t* device_group = nullptr;
  IREE_CHECK_OK(iree_hal_device_group_create_from_device(
      device, frontier_tracker.get(), iree_allocator_system(), &device_group));
  iree_hal_device_release(device);
  return device_group;
}

static iree_status_t AddSplatEntry(iree_io_parameter_index_t* index,
                                   iree_string_view_t key, uint64_t length,
                                   uint8_t pattern) {
  iree_io_parameter_index_entry_t entry;
  std::memset(&entry, 0, sizeof(entry));
  entry.key = key;
  entry.length = length;
  entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT;
  entry.storage.splat.pattern_length = 1;
  entry.storage.splat.pattern[0] = pattern;
  return iree_io_parameter_index_add(index, &entry);
}

static iree_status_t AddFileEntry(iree_io_parameter_index_t* index,
                                  iree_string_view_t key, uint64_t offset,
                                  uint64_t length,
                                  iree_io_file_handle_t* handle) {
  iree_io_parameter_index_entry_t entry;
  std::memset(&entry, 0, sizeof(entry));
  entry.key = key;
  entry.length = length;
  entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE;
  entry.storage.file.handle = handle;
  entry.storage.file.offset = offset;
  return iree_io_parameter_index_add(index, &entry);
}

typedef struct ParameterRequest {
  // Parameter key.
  iree_string_view_t key;
  // Parameter and buffer span.
  iree_io_parameter_span_t span;
} ParameterRequest;

static iree_status_t EnumerateParameterRequest(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  const ParameterRequest* requests =
      static_cast<const ParameterRequest*>(user_data);
  *out_key = requests[i].key;
  *out_span = requests[i].span;
  return iree_ok_status();
}

static iree_hal_buffer_t* AllocateTransferBuffer(
    iree_hal_device_t* device, iree_device_size_t byte_length) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_MAPPING |
                 IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
                 IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_CHECK_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), params, byte_length, &buffer));
  return buffer;
}

static void WriteBufferBytes(iree_hal_buffer_t* buffer,
                             iree_const_byte_span_t source) {
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
                                           0, source.data_length, &mapping));
  iree_byte_span_t span;
  IREE_ASSERT_OK(iree_hal_buffer_mapping_subspan(
      &mapping, IREE_HAL_MEMORY_ACCESS_WRITE, 0, source.data_length, &span));
  std::memcpy(span.data, source.data, source.data_length);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
}

static void ExpectBufferBytes(iree_hal_buffer_t* buffer,
                              iree_device_size_t byte_length,
                              uint8_t expected_value) {
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_READ, 0,
                                           byte_length, &mapping));
  iree_byte_span_t span;
  IREE_ASSERT_OK(iree_hal_buffer_mapping_subspan(
      &mapping, IREE_HAL_MEMORY_ACCESS_READ, 0, byte_length, &span));
  for (iree_host_size_t i = 0; i < byte_length; ++i) {
    EXPECT_EQ(span.data[i], expected_value);
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
}

static void ExpectBufferBytesEqual(iree_hal_buffer_t* buffer,
                                   iree_const_byte_span_t expected) {
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_READ, 0,
                                           expected.data_length, &mapping));
  iree_byte_span_t span;
  IREE_ASSERT_OK(iree_hal_buffer_mapping_subspan(
      &mapping, IREE_HAL_MEMORY_ACCESS_READ, 0, expected.data_length, &span));
  EXPECT_EQ(span.data_length, expected.data_length);
  EXPECT_EQ(0, std::memcmp(span.data, expected.data, expected.data_length));
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
}

class ParameterIndexProviderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { device_group_ = CreateTaskDeviceGroup(); }

  static void TearDownTestSuite() {
    iree_hal_device_group_release(device_group_);
    device_group_ = nullptr;
  }

  static iree_hal_device_t* device() {
    return iree_hal_device_group_device_at(device_group_, 0);
  }

 private:
  static iree_hal_device_group_t* device_group_;
};

iree_hal_device_group_t* ParameterIndexProviderTest::device_group_ = nullptr;

TEST_F(ParameterIndexProviderTest, GatherBatchPreservesGroupSignals) {
  iree_hal_device_t* device = this->device();

  Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), index.out()));
  IREE_ASSERT_OK(AddSplatEntry(index.get(), IREE_SV("first"), 4, 0xAB));
  IREE_ASSERT_OK(AddSplatEntry(index.get(), IREE_SV("second"), 4, 0xCD));

  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
      IREE_SV("model"), index.get(),
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      iree_allocator_system(), provider.out()));

  Ref<iree_hal_buffer_t, iree_hal_buffer_release> first_buffer;
  first_buffer.reset(AllocateTransferBuffer(device, 4));
  Ref<iree_hal_buffer_t, iree_hal_buffer_release> second_buffer;
  second_buffer.reset(AllocateTransferBuffer(device, 4));

  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> first_signal;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, first_signal.out()));
  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> second_signal;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, second_signal.out()));

  ParameterRequest first_request = {
      /*.key=*/IREE_SV("first"),
      /*.span=*/
      {/*.parameter_offset=*/0, /*.buffer_offset=*/0,
       /*.length=*/4},
  };
  ParameterRequest second_request = {
      /*.key=*/IREE_SV("second"),
      /*.span=*/
      {/*.parameter_offset=*/0, /*.buffer_offset=*/0,
       /*.length=*/4},
  };

  uint64_t first_signal_value = 1;
  uint64_t second_signal_value = 1;
  iree_hal_semaphore_t* first_signal_ptr = first_signal.get();
  iree_hal_semaphore_t* second_signal_ptr = second_signal.get();
  iree_io_parameter_gather_t gathers[2] = {
      {
          /*.source_scope=*/IREE_SV("model"),
          /*.target_buffer=*/first_buffer.get(),
          /*.count=*/1,
          /*.enumerator=*/
          {
              /*.fn=*/EnumerateParameterRequest,
              /*.user_data=*/&first_request,
          },
          /*.wait_semaphore_list=*/iree_hal_semaphore_list_empty(),
          /*.signal_semaphore_list=*/
          {
              /*.count=*/1,
              /*.semaphores=*/&first_signal_ptr,
              /*.payload_values=*/&first_signal_value,
          },
      },
      {
          /*.source_scope=*/IREE_SV("model"),
          /*.target_buffer=*/second_buffer.get(),
          /*.count=*/1,
          /*.enumerator=*/
          {
              /*.fn=*/EnumerateParameterRequest,
              /*.user_data=*/&second_request,
          },
          /*.wait_semaphore_list=*/iree_hal_semaphore_list_empty(),
          /*.signal_semaphore_list=*/
          {
              /*.count=*/1,
              /*.semaphores=*/&second_signal_ptr,
              /*.payload_values=*/&second_signal_value,
          },
      },
  };

  IREE_ASSERT_OK(iree_io_parameter_provider_gather_batch(
      provider.get(), device, IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_ARRAYSIZE(gathers), gathers));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(first_signal.get(), first_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      second_signal.get(), second_signal_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));

  ExpectBufferBytes(first_buffer.get(), 4, 0xAB);
  ExpectBufferBytes(second_buffer.get(), 4, 0xCD);
}

TEST_F(ParameterIndexProviderTest, GatherBatchReadsAdjacentFileSpans) {
  iree_hal_device_t* device = this->device();

  uint8_t source_data[12] = {
      0x10, 0x11, 0x12, 0x13, 0x20, 0x21, 0x22, 0x23, 0x30, 0x31, 0x32, 0x33,
  };
  Ref<iree_io_file_handle_t, iree_io_file_handle_release> source_handle;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ,
      iree_make_byte_span(source_data, sizeof(source_data)),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      source_handle.out()));

  Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), index.out()));
  IREE_ASSERT_OK(
      AddFileEntry(index.get(), IREE_SV("first"), 0, 4, source_handle.get()));
  IREE_ASSERT_OK(
      AddFileEntry(index.get(), IREE_SV("second"), 4, 4, source_handle.get()));
  IREE_ASSERT_OK(
      AddFileEntry(index.get(), IREE_SV("third"), 8, 4, source_handle.get()));

  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
      IREE_SV("model"), index.get(),
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      iree_allocator_system(), provider.out()));

  Ref<iree_hal_buffer_t, iree_hal_buffer_release> target_buffer;
  target_buffer.reset(AllocateTransferBuffer(device, sizeof(source_data)));

  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> signal;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, signal.out()));

  ParameterRequest requests[3] = {
      {
          /*.key=*/IREE_SV("first"),
          /*.span=*/
          {/*.parameter_offset=*/0, /*.buffer_offset=*/0, /*.length=*/4},
      },
      {
          /*.key=*/IREE_SV("second"),
          /*.span=*/
          {/*.parameter_offset=*/0, /*.buffer_offset=*/4, /*.length=*/4},
      },
      {
          /*.key=*/IREE_SV("third"),
          /*.span=*/
          {/*.parameter_offset=*/0, /*.buffer_offset=*/8, /*.length=*/4},
      },
  };

  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_ptr = signal.get();
  iree_io_parameter_gather_t gather = {
      /*.source_scope=*/IREE_SV("model"),
      /*.target_buffer=*/target_buffer.get(),
      /*.count=*/IREE_ARRAYSIZE(requests),
      /*.enumerator=*/
      {
          /*.fn=*/EnumerateParameterRequest,
          /*.user_data=*/requests,
      },
      /*.wait_semaphore_list=*/iree_hal_semaphore_list_empty(),
      /*.signal_semaphore_list=*/
      {
          /*.count=*/1,
          /*.semaphores=*/&signal_ptr,
          /*.payload_values=*/&signal_value,
      },
  };

  IREE_ASSERT_OK(iree_io_parameter_provider_gather_batch(
      provider.get(), device, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*gather_count=*/1, &gather));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(signal.get(), signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  ExpectBufferBytesEqual(
      target_buffer.get(),
      iree_make_const_byte_span(source_data, sizeof(source_data)));
}

TEST_F(ParameterIndexProviderTest, ScatterBatchPreservesGroupSemaphores) {
  iree_hal_device_t* device = this->device();

  uint8_t target_data[8] = {0};
  Ref<iree_io_file_handle_t, iree_io_file_handle_release> target_handle;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(target_data, sizeof(target_data)),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      target_handle.out()));

  Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), index.out()));
  IREE_ASSERT_OK(
      AddFileEntry(index.get(), IREE_SV("first"), 0, 4, target_handle.get()));
  IREE_ASSERT_OK(
      AddFileEntry(index.get(), IREE_SV("second"), 4, 4, target_handle.get()));

  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
      IREE_SV("model"), index.get(),
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      iree_allocator_system(), provider.out()));

  Ref<iree_hal_buffer_t, iree_hal_buffer_release> first_buffer;
  first_buffer.reset(AllocateTransferBuffer(device, 4));
  Ref<iree_hal_buffer_t, iree_hal_buffer_release> second_buffer;
  second_buffer.reset(AllocateTransferBuffer(device, 4));

  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> first_wait;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, first_wait.out()));
  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> second_wait;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, second_wait.out()));
  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> first_signal;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, first_signal.out()));
  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> second_signal;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, second_signal.out()));

  ParameterRequest first_request = {
      /*.key=*/IREE_SV("first"),
      /*.span=*/
      {/*.parameter_offset=*/0, /*.buffer_offset=*/0, /*.length=*/4},
  };
  ParameterRequest second_request = {
      /*.key=*/IREE_SV("second"),
      /*.span=*/
      {/*.parameter_offset=*/0, /*.buffer_offset=*/0, /*.length=*/4},
  };

  uint64_t first_wait_value = 1;
  uint64_t second_wait_value = 1;
  uint64_t first_signal_value = 1;
  uint64_t second_signal_value = 1;
  iree_hal_semaphore_t* first_wait_ptr = first_wait.get();
  iree_hal_semaphore_t* second_wait_ptr = second_wait.get();
  iree_hal_semaphore_t* first_signal_ptr = first_signal.get();
  iree_hal_semaphore_t* second_signal_ptr = second_signal.get();
  iree_io_parameter_scatter_t scatters[2] = {
      {
          /*.target_scope=*/IREE_SV("model"),
          /*.source_buffer=*/first_buffer.get(),
          /*.count=*/1,
          /*.enumerator=*/
          {
              /*.fn=*/EnumerateParameterRequest,
              /*.user_data=*/&first_request,
          },
          /*.wait_semaphore_list=*/
          {
              /*.count=*/1,
              /*.semaphores=*/&first_wait_ptr,
              /*.payload_values=*/&first_wait_value,
          },
          /*.signal_semaphore_list=*/
          {
              /*.count=*/1,
              /*.semaphores=*/&first_signal_ptr,
              /*.payload_values=*/&first_signal_value,
          },
      },
      {
          /*.target_scope=*/IREE_SV("model"),
          /*.source_buffer=*/second_buffer.get(),
          /*.count=*/1,
          /*.enumerator=*/
          {
              /*.fn=*/EnumerateParameterRequest,
              /*.user_data=*/&second_request,
          },
          /*.wait_semaphore_list=*/
          {
              /*.count=*/1,
              /*.semaphores=*/&second_wait_ptr,
              /*.payload_values=*/&second_wait_value,
          },
          /*.signal_semaphore_list=*/
          {
              /*.count=*/1,
              /*.semaphores=*/&second_signal_ptr,
              /*.payload_values=*/&second_signal_value,
          },
      },
  };

  const uint8_t first_source[4] = {0x11, 0x11, 0x11, 0x11};
  const uint8_t second_source[4] = {0x22, 0x22, 0x22, 0x22};
  WriteBufferBytes(first_buffer.get(), iree_make_const_byte_span(
                                           first_source, sizeof(first_source)));
  WriteBufferBytes(
      second_buffer.get(),
      iree_make_const_byte_span(second_source, sizeof(second_source)));

  std::atomic<bool> scatter_started = false;
  std::thread scatter_thread([&]() {
    scatter_started.store(true, std::memory_order_release);
    IREE_EXPECT_OK(iree_io_parameter_provider_scatter_batch(
        provider.get(), device, IREE_HAL_QUEUE_AFFINITY_ANY,
        IREE_ARRAYSIZE(scatters), scatters));
  });
  while (!scatter_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  IREE_EXPECT_OK(iree_hal_semaphore_signal(first_wait.get(), first_wait_value,
                                           /*frontier=*/nullptr));
  IREE_EXPECT_OK(iree_hal_semaphore_signal(second_wait.get(), second_wait_value,
                                           /*frontier=*/nullptr));
  scatter_thread.join();

  IREE_ASSERT_OK(iree_hal_semaphore_wait(first_signal.get(), first_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      second_signal.get(), second_signal_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));

  const uint8_t expected[8] = {0x11, 0x11, 0x11, 0x11, 0x22, 0x22, 0x22, 0x22};
  EXPECT_EQ(0, std::memcmp(target_data, expected, sizeof(expected)));
}

TEST_F(ParameterIndexProviderTest, ScatterBatchWritesAdjacentFileSpans) {
  iree_hal_device_t* device = this->device();

  uint8_t target_data[12] = {0};
  Ref<iree_io_file_handle_t, iree_io_file_handle_release> target_handle;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(target_data, sizeof(target_data)),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      target_handle.out()));

  Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), index.out()));
  IREE_ASSERT_OK(
      AddFileEntry(index.get(), IREE_SV("first"), 0, 4, target_handle.get()));
  IREE_ASSERT_OK(
      AddFileEntry(index.get(), IREE_SV("second"), 4, 4, target_handle.get()));
  IREE_ASSERT_OK(
      AddFileEntry(index.get(), IREE_SV("third"), 8, 4, target_handle.get()));

  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
      IREE_SV("model"), index.get(),
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      iree_allocator_system(), provider.out()));

  const uint8_t source_data[12] = {
      0x10, 0x11, 0x12, 0x13, 0x20, 0x21, 0x22, 0x23, 0x30, 0x31, 0x32, 0x33,
  };
  Ref<iree_hal_buffer_t, iree_hal_buffer_release> source_buffer;
  source_buffer.reset(AllocateTransferBuffer(device, sizeof(source_data)));
  WriteBufferBytes(source_buffer.get(),
                   iree_make_const_byte_span(source_data, sizeof(source_data)));

  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> signal;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, signal.out()));

  ParameterRequest requests[3] = {
      {
          /*.key=*/IREE_SV("first"),
          /*.span=*/
          {/*.parameter_offset=*/0, /*.buffer_offset=*/0, /*.length=*/4},
      },
      {
          /*.key=*/IREE_SV("second"),
          /*.span=*/
          {/*.parameter_offset=*/0, /*.buffer_offset=*/4, /*.length=*/4},
      },
      {
          /*.key=*/IREE_SV("third"),
          /*.span=*/
          {/*.parameter_offset=*/0, /*.buffer_offset=*/8, /*.length=*/4},
      },
  };

  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_ptr = signal.get();
  iree_io_parameter_scatter_t scatter = {
      /*.target_scope=*/IREE_SV("model"),
      /*.source_buffer=*/source_buffer.get(),
      /*.count=*/IREE_ARRAYSIZE(requests),
      /*.enumerator=*/
      {
          /*.fn=*/EnumerateParameterRequest,
          /*.user_data=*/requests,
      },
      /*.wait_semaphore_list=*/iree_hal_semaphore_list_empty(),
      /*.signal_semaphore_list=*/
      {
          /*.count=*/1,
          /*.semaphores=*/&signal_ptr,
          /*.payload_values=*/&signal_value,
      },
  };

  IREE_ASSERT_OK(iree_io_parameter_provider_scatter_batch(
      provider.get(), device, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*scatter_count=*/1, &scatter));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(signal.get(), signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  EXPECT_EQ(0, std::memcmp(target_data, source_data, sizeof(source_data)));
}

}  // namespace
