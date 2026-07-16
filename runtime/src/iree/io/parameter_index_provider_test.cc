// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/parameter_index_provider.h"

#include <cstdint>
#include <cstring>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
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

static iree_hal_device_group_t* CreateLocalSyncDeviceGroup() {
  Ref<iree_async_proactor_pool_t, iree_async_proactor_pool_release>
      proactor_pool;
  IREE_CHECK_OK(iree_async_proactor_pool_create(
      /*node_count=*/1, /*node_ids=*/nullptr,
      iree_async_proactor_pool_options_default(), iree_allocator_system(),
      proactor_pool.out()));

  Ref<iree_hal_allocator_t, iree_hal_allocator_release> device_allocator;
  IREE_CHECK_OK(iree_hal_allocator_create_heap(
      IREE_SV("parameter-index-provider-test"), iree_allocator_system(),
      iree_allocator_system(), device_allocator.out()));

  iree_hal_sync_device_params_t sync_params;
  iree_hal_sync_device_params_initialize(&sync_params);
  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool.get();

  iree_hal_device_t* device = nullptr;
  IREE_CHECK_OK(iree_hal_sync_device_create(
      IREE_SV("parameter-index-provider-test"), &sync_params, &create_params,
      /*loader_count=*/0, /*loaders=*/nullptr, device_allocator.get(),
      iree_allocator_system(), &device));

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

typedef struct ParameterRequest {
  // Source parameter key.
  iree_string_view_t key;
  // Source and target span.
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

static iree_hal_buffer_t* AllocateTargetBuffer(iree_hal_device_t* device,
                                               iree_device_size_t byte_length) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_MAPPING | IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_CHECK_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), params, byte_length, &buffer));
  return buffer;
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

TEST(ParameterIndexProviderTest, GatherBatchPreservesGroupSignals) {
  Ref<iree_hal_device_group_t, iree_hal_device_group_release> device_group;
  device_group.reset(CreateLocalSyncDeviceGroup());
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group.get(), 0);

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
  first_buffer.reset(AllocateTargetBuffer(device, 4));
  Ref<iree_hal_buffer_t, iree_hal_buffer_release> second_buffer;
  second_buffer.reset(AllocateTargetBuffer(device, 4));

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

}  // namespace
