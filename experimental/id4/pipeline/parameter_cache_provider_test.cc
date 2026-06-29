// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/parameter_cache_provider.h"

#include <cstdint>
#include <cstring>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

template <typename T, void (*Release)(T*)>
class OwningRef {
 public:
  OwningRef() = default;
  OwningRef(const OwningRef&) = delete;
  OwningRef& operator=(const OwningRef&) = delete;

  ~OwningRef() { reset(); }

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
  // Owned reference released by this wrapper.
  T* value_ = nullptr;
};

using HalBufferRef = OwningRef<iree_hal_buffer_t, iree_hal_buffer_release>;
using HalDeviceGroupRef =
    OwningRef<iree_hal_device_group_t, iree_hal_device_group_release>;
using HalSemaphoreRef =
    OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>;
using ParameterProviderRef =
    OwningRef<iree_io_parameter_provider_t, iree_io_parameter_provider_release>;

typedef struct CountingSourceProvider {
  // Base IREE provider interface.
  iree_io_parameter_provider_t base;
  // Parameter bytes served by gather calls.
  const uint8_t* data = nullptr;
  // Number of bytes in |data|.
  iree_host_size_t data_length = 0;
  // Number of upstream gather calls observed.
  iree_host_size_t gather_count = 0;
  // Number of upstream notify calls observed.
  iree_host_size_t notify_count = 0;
} CountingSourceProvider;

static void CountingSourceProviderDestroy(
    iree_io_parameter_provider_t* provider) {
  delete reinterpret_cast<CountingSourceProvider*>(provider);
}

static iree_status_t CountingSourceProviderNotify(
    iree_io_parameter_provider_t* provider,
    iree_io_parameter_provider_signal_t signal) {
  (void)signal;
  auto* source = reinterpret_cast<CountingSourceProvider*>(provider);
  ++source->notify_count;
  return iree_ok_status();
}

static bool CountingSourceProviderQuerySupport(
    iree_io_parameter_provider_t* provider, iree_string_view_t scope) {
  (void)provider;
  return iree_string_view_equal(scope, IREE_SV("scope"));
}

static iree_status_t CountingSourceProviderLoad(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_hal_buffer_params_t target_params,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator,
    iree_io_parameter_emitter_t emitter) {
  (void)provider;
  (void)device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)source_scope;
  (void)target_params;
  (void)count;
  (void)enumerator;
  (void)emitter;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "counting source provider load is not implemented");
}

static iree_status_t CountingSourceProviderGather(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_hal_buffer_t* target_buffer,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator) {
  auto* source = reinterpret_cast<CountingSourceProvider*>(provider);
  if (!iree_string_view_equal(source_scope, IREE_SV("scope"))) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "counting source provider scope not found");
  }
  if (count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "counting source provider expects one request");
  }
  iree_string_view_t key = iree_string_view_empty();
  iree_io_parameter_span_t span = {};
  IREE_RETURN_IF_ERROR(
      enumerator.fn(enumerator.user_data, /*i=*/0, &key, &span));
  if (!iree_string_view_equal(key, IREE_SV("weight"))) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "counting source provider key not found");
  }
  if (span.parameter_offset > source->data_length ||
      span.length > source->data_length - span.parameter_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "counting source provider span is out of range");
  }
  ++source->gather_count;
  return iree_hal_device_queue_update(
      device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source->data + span.parameter_offset, /*source_offset=*/0, target_buffer,
      span.buffer_offset, span.length, IREE_HAL_UPDATE_FLAG_NONE);
}

static iree_status_t CountingSourceProviderScatter(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_string_view_t target_scope,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator) {
  (void)provider;
  (void)device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)source_buffer;
  (void)target_scope;
  (void)count;
  (void)enumerator;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "counting source provider scatter is not supported");
}

static const iree_io_parameter_provider_vtable_t kCountingSourceProviderVTable =
    {
        /*.destroy=*/CountingSourceProviderDestroy,
        /*.notify=*/CountingSourceProviderNotify,
        /*.query_support=*/CountingSourceProviderQuerySupport,
        /*.load=*/CountingSourceProviderLoad,
        /*.gather=*/CountingSourceProviderGather,
        /*.scatter=*/CountingSourceProviderScatter,
};

static CountingSourceProvider* CreateCountingSourceProvider(
    const uint8_t* data, iree_host_size_t data_length) {
  auto* provider = new CountingSourceProvider();
  iree_atomic_ref_count_init(&provider->base.ref_count);
  provider->base.vtable = &kCountingSourceProviderVTable;
  provider->data = data;
  provider->data_length = data_length;
  return provider;
}

static iree_status_t CreateLocalSyncDeviceGroup(
    iree_hal_device_group_t** out_device_group) {
  IREE_ASSERT_ARGUMENT(out_device_group);
  *out_device_group = nullptr;

  iree_async_proactor_pool_t* proactor_pool = nullptr;
  iree_status_t status = iree_async_proactor_pool_create(
      /*node_count=*/1, /*node_ids=*/nullptr,
      iree_async_proactor_pool_options_default(), iree_allocator_system(),
      &proactor_pool);

  iree_hal_allocator_t* device_allocator = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_create_heap(
        IREE_SV("id4-parameter-cache-local-sync"), iree_allocator_system(),
        iree_allocator_system(), &device_allocator);
  }

  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    iree_hal_sync_device_params_t sync_params;
    iree_hal_sync_device_params_initialize(&sync_params);
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    status = iree_hal_sync_device_create(
        IREE_SV("id4-parameter-cache-local-sync"), &sync_params, &create_params,
        /*loader_count=*/0, /*loaders=*/nullptr, device_allocator,
        iree_allocator_system(), &device);
  }

  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  if (iree_status_is_ok(status)) {
    iree_async_frontier_tracker_options_t tracker_options =
        iree_async_frontier_tracker_options_default();
    status = iree_async_frontier_tracker_create(
        tracker_options, iree_allocator_system(), &frontier_tracker);
  }

  iree_hal_device_group_t* device_group = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_group_create_from_device(
        device, frontier_tracker, iree_allocator_system(), &device_group);
  }

  iree_async_frontier_tracker_release(frontier_tracker);
  iree_hal_device_release(device);
  iree_hal_allocator_release(device_allocator);
  iree_async_proactor_pool_release(proactor_pool);
  if (iree_status_is_ok(status)) {
    *out_device_group = device_group;
  } else {
    iree_hal_device_group_release(device_group);
  }
  return status;
}

static iree_hal_buffer_params_t MakeTransferBufferParams() {
  iree_hal_buffer_params_t params = {};
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  return params;
}

static iree_status_t CreateParameterCacheProvider(
    CountingSourceProvider* source, iree_device_size_t maximum_cached_bytes,
    ParameterProviderRef* out_cache_provider) {
  id4_pipeline_parameter_cache_provider_options_t options = {};
  options.structure_size = sizeof(options);
  options.source_provider = &source->base;
  options.cache_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  options.cache_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  options.cache_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  options.cache_params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  options.maximum_cached_byte_length = maximum_cached_bytes;
  return id4_pipeline_parameter_cache_provider_create(
      &options, iree_allocator_system(), out_cache_provider->out());
}

static iree_io_parameter_span_t MakeSpan(uint64_t parameter_offset,
                                         iree_device_size_t buffer_offset,
                                         iree_device_size_t length) {
  iree_io_parameter_span_t span = {};
  span.parameter_offset = parameter_offset;
  span.buffer_offset = buffer_offset;
  span.length = length;
  return span;
}

typedef struct SingleRequestEnumerator {
  // Source key returned to the provider.
  iree_string_view_t key;
  // Span returned to the provider.
  iree_io_parameter_span_t span;
} SingleRequestEnumerator;

static iree_status_t EnumerateSingleRequest(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  if (i != 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "single request index is out of range");
  }
  auto* state = static_cast<SingleRequestEnumerator*>(user_data);
  *out_key = state->key;
  *out_span = state->span;
  return iree_ok_status();
}

static iree_io_parameter_enumerator_t MakeSingleRequestEnumerator(
    SingleRequestEnumerator* state) {
  iree_io_parameter_enumerator_t enumerator = {
      /*.fn=*/EnumerateSingleRequest,
      /*.user_data=*/state,
  };
  return enumerator;
}

static iree_status_t GatherAndWaitStatus(iree_io_parameter_provider_t* provider,
                                         iree_hal_device_t* device,
                                         iree_hal_buffer_t* target_buffer,
                                         iree_io_parameter_span_t span) {
  SingleRequestEnumerator request = {
      /*.key=*/IREE_SV("weight"),
      /*.span=*/span,
  };
  HalSemaphoreRef done_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, done_semaphore.out()));
  iree_hal_semaphore_t* done_semaphore_ptr = done_semaphore.get();
  uint64_t done_payload_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&done_semaphore_ptr,
      /*.payload_values=*/&done_payload_value,
  };
  iree_status_t status = iree_io_parameter_provider_gather(
      provider, device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, IREE_SV("scope"),
      target_buffer, /*count=*/1, MakeSingleRequestEnumerator(&request));
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_wait_semaphores(
        device, IREE_ASYNC_WAIT_MODE_ALL, signal_list, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE);
  }
  return status;
}

static void GatherAndWait(iree_io_parameter_provider_t* provider,
                          iree_hal_device_t* device,
                          iree_hal_buffer_t* target_buffer,
                          iree_io_parameter_span_t span) {
  IREE_ASSERT_OK(GatherAndWaitStatus(provider, device, target_buffer, span));
}

static void ExpectCacheStatistics(
    iree_io_parameter_provider_t* provider, iree_host_size_t entry_count,
    iree_device_size_t cached_byte_length,
    iree_device_size_t peak_cached_byte_length,
    iree_host_size_t source_gather_count, iree_host_size_t cache_reuse_count,
    iree_host_size_t evicted_entry_count,
    iree_device_size_t maximum_cached_byte_length = 0) {
  id4_pipeline_parameter_cache_provider_statistics_t statistics = {};
  IREE_ASSERT_OK(id4_pipeline_parameter_cache_provider_query_statistics(
      provider, &statistics));
  EXPECT_EQ(statistics.entry_count, entry_count);
  EXPECT_EQ(statistics.cached_byte_length, cached_byte_length);
  EXPECT_EQ(statistics.peak_cached_byte_length, peak_cached_byte_length);
  EXPECT_EQ(statistics.maximum_cached_byte_length, maximum_cached_byte_length);
  EXPECT_EQ(statistics.source_gather_count, source_gather_count);
  EXPECT_EQ(statistics.cache_reuse_count, cache_reuse_count);
  EXPECT_EQ(statistics.evicted_entry_count, evicted_entry_count);
}

TEST(ParameterCacheProviderTest, ReusesExactSourceSpanAcrossTargetOffsets) {
  HalDeviceGroupRef device_group;
  IREE_ASSERT_OK(CreateLocalSyncDeviceGroup(device_group.out()));
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group.get(), /*device_index=*/0);

  uint8_t source_data[32] = {};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(source_data); ++i) {
    source_data[i] = static_cast<uint8_t>(i + 1);
  }
  CountingSourceProvider* source =
      CreateCountingSourceProvider(source_data, sizeof(source_data));
  ParameterProviderRef source_ref;
  source_ref.reset(&source->base);

  ParameterProviderRef cache_provider;
  IREE_ASSERT_OK(CreateParameterCacheProvider(
      source, /*maximum_cached_bytes=*/0, &cache_provider));
  ExpectCacheStatistics(cache_provider.get(), /*entry_count=*/0,
                        /*cached_byte_length=*/0,
                        /*peak_cached_byte_length=*/0,
                        /*source_gather_count=*/0,
                        /*cache_reuse_count=*/0,
                        /*evicted_entry_count=*/0);

  HalBufferRef target_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), MakeTransferBufferParams(),
      /*allocation_size=*/32, target_buffer.out()));

  GatherAndWait(cache_provider.get(), device, target_buffer.get(),
                MakeSpan(/*parameter_offset=*/4, /*buffer_offset=*/8,
                         /*length=*/8));
  EXPECT_EQ(source->gather_count, 1u);
  ExpectCacheStatistics(cache_provider.get(), /*entry_count=*/1,
                        /*cached_byte_length=*/8,
                        /*peak_cached_byte_length=*/8,
                        /*source_gather_count=*/1,
                        /*cache_reuse_count=*/0,
                        /*evicted_entry_count=*/0);
  uint8_t first_readback[8] = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(target_buffer.get(),
                                          /*source_offset=*/8, first_readback,
                                          sizeof(first_readback)));
  EXPECT_EQ(
      std::memcmp(first_readback, source_data + 4, sizeof(first_readback)), 0);

  GatherAndWait(cache_provider.get(), device, target_buffer.get(),
                MakeSpan(/*parameter_offset=*/4, /*buffer_offset=*/16,
                         /*length=*/8));
  EXPECT_EQ(source->gather_count, 1u);
  ExpectCacheStatistics(cache_provider.get(), /*entry_count=*/1,
                        /*cached_byte_length=*/8,
                        /*peak_cached_byte_length=*/8,
                        /*source_gather_count=*/1,
                        /*cache_reuse_count=*/1,
                        /*evicted_entry_count=*/0);
  uint8_t second_readback[8] = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(target_buffer.get(),
                                          /*source_offset=*/16, second_readback,
                                          sizeof(second_readback)));
  EXPECT_EQ(
      std::memcmp(second_readback, source_data + 4, sizeof(second_readback)),
      0);

  GatherAndWait(cache_provider.get(), device, target_buffer.get(),
                MakeSpan(/*parameter_offset=*/5, /*buffer_offset=*/0,
                         /*length=*/8));
  EXPECT_EQ(source->gather_count, 2u);
  ExpectCacheStatistics(cache_provider.get(), /*entry_count=*/2,
                        /*cached_byte_length=*/16,
                        /*peak_cached_byte_length=*/16,
                        /*source_gather_count=*/2,
                        /*cache_reuse_count=*/1,
                        /*evicted_entry_count=*/0);

  IREE_ASSERT_OK(iree_io_parameter_provider_notify(
      cache_provider.get(), IREE_IO_PARAMETER_PROVIDER_SIGNAL_LOW_MEMORY));
  EXPECT_EQ(source->notify_count, 1u);
  ExpectCacheStatistics(cache_provider.get(), /*entry_count=*/0,
                        /*cached_byte_length=*/0,
                        /*peak_cached_byte_length=*/16,
                        /*source_gather_count=*/2,
                        /*cache_reuse_count=*/1,
                        /*evicted_entry_count=*/2);
  GatherAndWait(cache_provider.get(), device, target_buffer.get(),
                MakeSpan(/*parameter_offset=*/4, /*buffer_offset=*/24,
                         /*length=*/8));
  EXPECT_EQ(source->gather_count, 3u);
  ExpectCacheStatistics(cache_provider.get(), /*entry_count=*/1,
                        /*cached_byte_length=*/8,
                        /*peak_cached_byte_length=*/16,
                        /*source_gather_count=*/3,
                        /*cache_reuse_count=*/1,
                        /*evicted_entry_count=*/2);
}

TEST(ParameterCacheProviderTest, EvictsOldestEntriesToHonorBudget) {
  HalDeviceGroupRef device_group;
  IREE_ASSERT_OK(CreateLocalSyncDeviceGroup(device_group.out()));
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group.get(), /*device_index=*/0);

  uint8_t source_data[32] = {};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(source_data); ++i) {
    source_data[i] = static_cast<uint8_t>(i + 1);
  }
  CountingSourceProvider* source =
      CreateCountingSourceProvider(source_data, sizeof(source_data));
  ParameterProviderRef source_ref;
  source_ref.reset(&source->base);

  ParameterProviderRef cache_provider;
  IREE_ASSERT_OK(CreateParameterCacheProvider(
      source, /*maximum_cached_bytes=*/12, &cache_provider));
  ExpectCacheStatistics(cache_provider.get(), /*entry_count=*/0,
                        /*cached_byte_length=*/0,
                        /*peak_cached_byte_length=*/0,
                        /*source_gather_count=*/0,
                        /*cache_reuse_count=*/0,
                        /*evicted_entry_count=*/0,
                        /*maximum_cached_byte_length=*/12);

  HalBufferRef target_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), MakeTransferBufferParams(),
      /*allocation_size=*/32, target_buffer.out()));

  GatherAndWait(cache_provider.get(), device, target_buffer.get(),
                MakeSpan(/*parameter_offset=*/0, /*buffer_offset=*/0,
                         /*length=*/8));
  ExpectCacheStatistics(cache_provider.get(), /*entry_count=*/1,
                        /*cached_byte_length=*/8,
                        /*peak_cached_byte_length=*/8,
                        /*source_gather_count=*/1,
                        /*cache_reuse_count=*/0,
                        /*evicted_entry_count=*/0,
                        /*maximum_cached_byte_length=*/12);

  GatherAndWait(cache_provider.get(), device, target_buffer.get(),
                MakeSpan(/*parameter_offset=*/8, /*buffer_offset=*/8,
                         /*length=*/8));
  ExpectCacheStatistics(cache_provider.get(), /*entry_count=*/1,
                        /*cached_byte_length=*/8,
                        /*peak_cached_byte_length=*/8,
                        /*source_gather_count=*/2,
                        /*cache_reuse_count=*/0,
                        /*evicted_entry_count=*/1,
                        /*maximum_cached_byte_length=*/12);

  GatherAndWait(cache_provider.get(), device, target_buffer.get(),
                MakeSpan(/*parameter_offset=*/0, /*buffer_offset=*/16,
                         /*length=*/8));
  ExpectCacheStatistics(cache_provider.get(), /*entry_count=*/1,
                        /*cached_byte_length=*/8,
                        /*peak_cached_byte_length=*/8,
                        /*source_gather_count=*/3,
                        /*cache_reuse_count=*/0,
                        /*evicted_entry_count=*/2,
                        /*maximum_cached_byte_length=*/12);
  EXPECT_EQ(source->gather_count, 3u);

  uint8_t readback[8] = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(target_buffer.get(),
                                          /*source_offset=*/16, readback,
                                          sizeof(readback)));
  EXPECT_EQ(std::memcmp(readback, source_data, sizeof(readback)), 0);
}

TEST(ParameterCacheProviderTest, RejectsSpanLargerThanBudget) {
  HalDeviceGroupRef device_group;
  IREE_ASSERT_OK(CreateLocalSyncDeviceGroup(device_group.out()));
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group.get(), /*device_index=*/0);

  uint8_t source_data[32] = {};
  CountingSourceProvider* source =
      CreateCountingSourceProvider(source_data, sizeof(source_data));
  ParameterProviderRef source_ref;
  source_ref.reset(&source->base);

  ParameterProviderRef cache_provider;
  IREE_ASSERT_OK(CreateParameterCacheProvider(
      source, /*maximum_cached_bytes=*/4, &cache_provider));

  HalBufferRef target_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), MakeTransferBufferParams(),
      /*allocation_size=*/8, target_buffer.out()));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      GatherAndWaitStatus(cache_provider.get(), device, target_buffer.get(),
                          MakeSpan(/*parameter_offset=*/0,
                                   /*buffer_offset=*/0, /*length=*/8)));
  EXPECT_EQ(source->gather_count, 0u);
  ExpectCacheStatistics(cache_provider.get(), /*entry_count=*/0,
                        /*cached_byte_length=*/0,
                        /*peak_cached_byte_length=*/0,
                        /*source_gather_count=*/0,
                        /*cache_reuse_count=*/0,
                        /*evicted_entry_count=*/0,
                        /*maximum_cached_byte_length=*/4);
}

TEST(ParameterCacheProviderTest, RejectsMutableScatter) {
  HalDeviceGroupRef device_group;
  IREE_ASSERT_OK(CreateLocalSyncDeviceGroup(device_group.out()));
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group.get(), /*device_index=*/0);

  uint8_t source_data[8] = {};
  CountingSourceProvider* source =
      CreateCountingSourceProvider(source_data, sizeof(source_data));
  ParameterProviderRef source_ref;
  source_ref.reset(&source->base);

  ParameterProviderRef cache_provider;
  IREE_ASSERT_OK(CreateParameterCacheProvider(
      source, /*maximum_cached_bytes=*/0, &cache_provider));

  HalBufferRef source_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), MakeTransferBufferParams(),
      /*allocation_size=*/8, source_buffer.out()));
  iree_io_parameter_span_t scatter_span = MakeSpan(
      /*parameter_offset=*/0, /*buffer_offset=*/0, /*length=*/4);
  SingleRequestEnumerator request = {
      /*.key=*/IREE_SV("weight"),
      /*.span=*/scatter_span,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_io_parameter_provider_scatter(
          cache_provider.get(), device, IREE_HAL_QUEUE_AFFINITY_ANY,
          iree_hal_semaphore_list_empty(), iree_hal_semaphore_list_empty(),
          source_buffer.get(), IREE_SV("scope"), /*count=*/1,
          MakeSingleRequestEnumerator(&request)));
}

}  // namespace
