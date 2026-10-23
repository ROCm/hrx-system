// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <thread>
#include <vector>

#include "iree/async/notification.h"
#include "iree/hal/cts/util/pool_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/hal/slab_pool.h"

namespace iree::hal::cts {

namespace {

iree_hal_queue_family_affinity_t QueueFamilyAffinity(iree_hal_queue_t* queue) {
  return iree_hal_make_queue_family_affinity(
      iree_hal_queue_family_ordinal(iree_hal_queue_family(queue)));
}

constexpr iree_hal_buffer_usage_t kQueueAllocationUsage =
    IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE;

iree_hal_pool_reservation_request_t MakeRequest(iree_device_size_t size) {
  return iree_hal_make_pool_reservation_request(size, /*min_alignment=*/0);
}

}  // namespace

class QueueAllocaTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase<>::SetUp();
    if (this->IsSkipped()) {
      return;
    }
    if (!transfer_queue_) {
      GTEST_SKIP() << "device has no provisioned transfer-capable queue";
    }
  }

  iree_hal_asan_pool_options_t AsanOptions() const {
    return iree_hal_device_spec_sanitizer(iree_hal_device_spec(device_))
        ->asan.pool_options;
  }

  iree_status_t CreateSlabPool(iree_hal_pool_t** out_pool) {
    return CreateQueueSlabPool(device_group_,
                               iree_hal_queue_family(transfer_queue_),
                               kQueueAllocationUsage, /*host_access=*/{},
                               iree_allocator_system(), out_pool);
  }

  iree_status_t CreateTLSFPool(iree_device_size_t range_length,
                               iree_hal_pool_t** out_pool) {
    iree_hal_tlsf_pool_options_t options = {};
    options.tlsf_options.range_length = range_length;
    options.tlsf_options.alignment = IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
    options.tlsf_options.frontier_capacity = 2;
    options.asan = AsanOptions();
    iree_hal_pool_t* backing_pool = nullptr;
    IREE_RETURN_IF_ERROR(CreateSlabPool(&backing_pool));
    iree_status_t status = iree_hal_tlsf_pool_create(
        backing_pool, &options, iree_allocator_system(), out_pool);
    iree_hal_pool_release(backing_pool);
    return status;
  }

  iree_status_t CreateHostExportableTLSFPool(iree_device_size_t range_length,
                                             iree_hal_pool_t** out_pool) {
    const iree_hal_pool_family_access_t family_access = {
        .family = iree_hal_queue_family(transfer_queue_),
        .usage = kQueueAllocationUsage,
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = 1,
        .families = &family_access,
        .host =
            {
                .access = IREE_HAL_MEMORY_ACCESS_ALL,
                .modes = IREE_HAL_MAPPING_MODE_SCOPED |
                         IREE_HAL_MAPPING_MODE_PERSISTENT,
            },
    };
    const iree_hal_external_storage_class_t export_class =
        iree_hal_make_external_storage_class(
            IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER);
    iree_hal_slab_pool_options_t backing_options;
    iree_hal_slab_pool_options_initialize(&backing_options);
    backing_options.owned.exports.class_count = 1;
    backing_options.owned.exports.classes = &export_class;
    backing_options.trace_name = IREE_SV("cts.queue.exportable");
    iree_hal_pool_t* backing_pool = nullptr;
    IREE_RETURN_IF_ERROR(
        iree_hal_slab_pool_create(device_group_, scope, &backing_options,
                                  iree_allocator_system(), &backing_pool));
    iree_hal_tlsf_pool_options_t options = {};
    options.tlsf_options.range_length = range_length;
    options.tlsf_options.alignment = IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
    options.tlsf_options.frontier_capacity = 2;
    options.asan = AsanOptions();
    iree_status_t status = iree_hal_tlsf_pool_create(
        backing_pool, &options, iree_allocator_system(), out_pool);
    iree_hal_pool_release(backing_pool);
    return status;
  }

  iree_status_t CreateFixedBlockPool(iree_device_size_t block_size,
                                     uint32_t block_count,
                                     iree_hal_pool_t** out_backing_pool,
                                     iree_hal_pool_t** out_pool) {
    *out_backing_pool = nullptr;
    *out_pool = nullptr;
    iree_hal_fixed_block_pool_options_t options = {
        .block_size = block_size,
        .blocks_per_slab = block_count,
        .frontier_capacity = 2,
        .asan = AsanOptions(),
    };
    IREE_RETURN_IF_ERROR(CreateSlabPool(out_backing_pool));
    iree_status_t status = CreateFiniteBlockPool(
        *out_backing_pool, options, iree_allocator_system(), out_pool);
    if (!iree_status_is_ok(status)) {
      iree_hal_pool_release(*out_backing_pool);
      *out_backing_pool = nullptr;
    }
    return status;
  }

  iree_hal_queue_t* FindSiblingQueue(iree_hal_queue_t* queue) {
    const iree_hal_queue_family_ordinal_t family_ordinal =
        iree_hal_queue_family_ordinal(iree_hal_queue_family(queue));
    const iree_hal_device_queue_spec_t* queue_spec =
        iree_hal_device_spec_queues(iree_hal_device_spec(device_));
    if (!queue_spec || family_ordinal >= queue_spec->family_count) {
      return NULL;
    }
    const uint32_t queue_count =
        queue_spec->families[family_ordinal].provisioned_queue_count;
    for (iree_hal_queue_ordinal_t queue_ordinal = 0;
         queue_ordinal < queue_count; ++queue_ordinal) {
      iree_hal_queue_t* candidate =
          iree_hal_device_queue(device_, family_ordinal, queue_ordinal);
      if (candidate != queue) {
        return candidate;
      }
    }
    return NULL;
  }

  void Wait(iree_hal_semaphore_list_t semaphore_list) {
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        semaphore_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  void FillAndWait(iree_hal_queue_t* queue, iree_hal_buffer_t* buffer,
                   uint32_t pattern) {
    SemaphoreList empty_wait;
    SemaphoreList fill_signal(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_fill(
        queue, empty_wait, fill_signal, buffer, /*target_offset=*/0,
        iree_hal_buffer_byte_length(buffer), &pattern, sizeof(pattern),
        /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
    Wait(fill_signal);
  }

  void DeallocaAndWait(iree_hal_queue_t* queue, iree_host_size_t buffer_count,
                       iree_hal_buffer_t* const* buffers) {
    SemaphoreList empty_wait;
    SemaphoreList dealloca_signal(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_dealloca(queue, empty_wait, dealloca_signal,
                                           buffer_count, buffers));
    Wait(dealloca_signal);
  }
};

TEST_P(QueueAllocaTest, ExplicitPoolScope) {
  constexpr iree_device_size_t kAllocationSize = 1024;
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateSlabPool(pool.out()));

  // Exercise an alignment accepted by this pool. The device's queue backend
  // may expose a different native memory class than the one selected for the
  // explicit scope above, so its provider properties are not authoritative
  // for this allocation.
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool, &capabilities);
  ASSERT_GT(capabilities.max_allocation_alignment, 0u);
  const iree_device_size_t requested_alignment =
      iree_min((iree_device_size_t)256, capabilities.max_allocation_alignment);
  iree_hal_pool_reservation_request_t request = MakeRequest(kAllocationSize);
  request.min_alignment = requested_alignment;
  Ref<iree_hal_buffer_t> buffer;
  SemaphoreList empty_wait;
  SemaphoreList alloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(
      iree_hal_queue_alloca(transfer_queue_, empty_wait, alloca_signal, pool,
                            /*request_count=*/1, &request, buffer.out()));
  ASSERT_NE(nullptr, buffer.get());
  Wait(alloca_signal);

  const auto memory = iree_hal_buffer_memory_view(buffer);
  ASSERT_NE(memory.backing, nullptr);
  EXPECT_GE(memory.backing->allocation_alignment, requested_alignment);
  EXPECT_EQ(memory.offset % requested_alignment, 0u);
  EXPECT_GE(iree_hal_buffer_byte_length(buffer), kAllocationSize);
  const iree_hal_buffer_usage_t expected_usage = kQueueAllocationUsage;
  EXPECT_EQ(expected_usage, iree_hal_buffer_allowed_usage(buffer));
  EXPECT_EQ(expected_usage,
            iree_hal_buffer_family_usage(
                buffer, iree_hal_queue_family(transfer_queue_)));
  IREE_EXPECT_OK(iree_hal_buffer_validate_family_usage(
      buffer, iree_hal_queue_family(transfer_queue_), expected_usage));

  constexpr uint32_t kPattern = 0xA11CA7EDu;
  FillAndWait(transfer_queue_, buffer, kPattern);
  EXPECT_EQ(kPattern, ReadBufferData<uint32_t>(buffer)[0]);

  iree_hal_buffer_t* buffers[] = {buffer.get()};
  DeallocaAndWait(transfer_queue_, IREE_ARRAYSIZE(buffers), buffers);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(0u, stats.reservation_count);
}

TEST_P(QueueAllocaTest, IndependentAllocatorsShareCachedNativeBacking) {
  // Both allocation policies can give and receive the same cached range.
  for (uint32_t policies = 0; policies < 4; ++policies) {
    SCOPED_TRACE(policies);
    Ref<iree_hal_pool_t> native;
    IREE_ASSERT_OK(CreateSlabPool(native.out()));
    iree_hal_pool_capabilities_t native_capabilities;
    iree_hal_pool_query_capabilities(native, &native_capabilities);
    iree_hal_slab_cache_options_t cache_options;
    iree_hal_slab_cache_options_initialize(&cache_options);
    cache_options.slab = MakeRequest(65536);
    // Share one class satisfying the 256-byte fixed-block stride when the
    // selected native source supports it.
    cache_options.slab.min_alignment =
        iree_min(256, native_capabilities.max_allocation_alignment);
    Ref<iree_hal_pool_t> cache;
    IREE_ASSERT_OK(iree_hal_slab_cache_create(
        native, &cache_options, iree_allocator_system(), cache.out()));
    const auto request = MakeRequest(256);
    const iree_hal_buffer_backing_facts_t* first_backing = nullptr;
    Ref<iree_hal_pool_t> children[2];
    for (uint32_t iteration = 0; iteration < 2; ++iteration) {
      auto& child = children[iteration];
      if (policies & (1u << iteration)) {
        iree_hal_fixed_block_pool_options_t options = {
            .block_size = 256,
            .blocks_per_slab = 16,
            .frontier_capacity = 2,
            .asan = AsanOptions(),
        };
        IREE_ASSERT_OK(iree_hal_fixed_block_pool_create(
            cache, &options, iree_allocator_system(), child.out()));
      } else {
        iree_hal_tlsf_pool_options_t options = {};
        options.tlsf_options.range_length = 4096;
        options.tlsf_options.alignment = IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
        options.tlsf_options.frontier_capacity = 2;
        options.asan = AsanOptions();
        IREE_ASSERT_OK(iree_hal_tlsf_pool_create(
            cache, &options, iree_allocator_system(), child.out()));
      }
      std::array<Ref<iree_hal_buffer_t>, 2> buffers;
      for (auto& buffer : buffers) {
        SemaphoreList empty_wait;
        SemaphoreList allocated(device_, {0}, {1});
        IREE_ASSERT_OK(iree_hal_queue_alloca(transfer_queue_, empty_wait,
                                             allocated, child, 1, &request,
                                             buffer.out()));
        Wait(allocated);
        const auto memory = iree_hal_buffer_memory_view(buffer);
        ASSERT_NE(memory.backing, nullptr);
        if (!first_backing) {
          first_backing = memory.backing;
        }
        EXPECT_EQ(memory.backing, first_backing);
      }
      for (size_t i = 0; i < buffers.size(); ++i) {
        FillAndWait(transfer_queue_, buffers[i],
                    0xCA000000u + iteration * 2 + i);
      }
      for (size_t i = 0; i < buffers.size(); ++i) {
        for (uint32_t value : ReadBufferData<uint32_t>(buffers[i])) {
          EXPECT_EQ(value, 0xCA000000u + iteration * 2 + i);
        }
        iree_hal_buffer_t* buffer = buffers[i];
        DeallocaAndWait(transfer_queue_, 1, &buffer);
        buffers[i].reset();
      }
      // Destruction joins this child's automatic whole-range return before the
      // other independently created policy asks the cache to reuse it.
      child.reset();
    }
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(native, &stats);
    EXPECT_EQ(stats.reserve_count, 1u);
    iree_hal_slab_cache_stats_t cache_stats;
    IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache, &cache_stats));
    EXPECT_EQ(cache_stats.hit_count, 1u);
  }
}

TEST_P(QueueAllocaTest, BlockedSiblingResumesFromSharedCache) {
  for (uint32_t policies = 0; policies < 4; ++policies) {
    SCOPED_TRACE(policies);
    // A finite ordinary backing pool bounds the entire composition to one
    // real native range, regardless of the children's allocation policies.
    Ref<iree_hal_pool_t> native;
    Ref<iree_hal_pool_t> source;
    IREE_ASSERT_OK(CreateFixedBlockPool(65536, 1, native.out(), source.out()));
    iree_hal_slab_cache_options_t cache_options;
    iree_hal_slab_cache_options_initialize(&cache_options);
    cache_options.slab = MakeRequest(65536);
    iree_hal_pool_capabilities_t capabilities;
    iree_hal_pool_query_capabilities(source, &capabilities);
    cache_options.slab.min_alignment =
        iree_min(4096, capabilities.max_allocation_alignment);
    Ref<iree_hal_pool_t> cache;
    IREE_ASSERT_OK(iree_hal_slab_cache_create(
        source, &cache_options, iree_allocator_system(), cache.out()));
    Ref<iree_hal_pool_t> children[2];
    for (uint32_t i = 0; i < 2; ++i) {
      if (policies & (1u << i)) {
        iree_hal_fixed_block_pool_options_t options = {
            .block_size = 256,
            .blocks_per_slab = 16,
            .frontier_capacity = 2,
            .asan = AsanOptions(),
        };
        IREE_ASSERT_OK(iree_hal_fixed_block_pool_create(
            cache, &options, iree_allocator_system(), children[i].out()));
      } else {
        iree_hal_tlsf_pool_options_t options = {};
        options.tlsf_options.range_length = 4096;
        options.tlsf_options.alignment = IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
        options.tlsf_options.frontier_capacity = 2;
        options.asan = AsanOptions();
        IREE_ASSERT_OK(iree_hal_tlsf_pool_create(
            cache, &options, iree_allocator_system(), children[i].out()));
      }
    }
    ASSERT_NE(iree_hal_pool_notification(children[0]),
              iree_hal_pool_notification(children[1]));
    auto* cache_notification = iree_hal_pool_notification(cache);
    ASSERT_NE(cache_notification, iree_hal_pool_notification(children[1]));
    const auto request = MakeRequest(256);
    Ref<iree_hal_buffer_t> first;
    SemaphoreList no_waits;
    SemaphoreList first_allocated(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_alloca(transfer_queue_, no_waits,
                                         first_allocated, children[0], 1,
                                         &request, first.out()));
    Wait(first_allocated);
    const auto* first_backing = iree_hal_buffer_memory_view(first).backing;
    Ref<iree_hal_buffer_t> second;
    SemaphoreList second_allocated(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_alloca(transfer_queue_, no_waits,
                                         second_allocated, children[1], 1,
                                         &request, second.out()));
    // Observation proves that the blocked child captured the backing source.
    // Returning capacity here exercises either the retry bridge or admitted
    // wait; both must preserve the notification without submission ordering.
    while (iree_atomic_load(&cache_notification->observer_count,
                            iree_memory_order_acquire) == 0) {
      uint64_t value = 0;
      IREE_ASSERT_OK(
          iree_hal_semaphore_query(second_allocated.semaphores[0], &value));
      ASSERT_EQ(value, 0u);
      std::this_thread::yield();
    }
    EXPECT_FALSE(iree_hal_semaphore_list_poll(second_allocated));
    iree_hal_buffer_t* first_buffer = first;
    SemaphoreList first_released(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_dealloca(transfer_queue_, first_allocated,
                                           first_released, 1, &first_buffer));
    Wait(first_released);
    first.reset();
    Wait(second_allocated);
    EXPECT_EQ(iree_hal_buffer_memory_view(second).backing, first_backing);
    constexpr uint32_t kPattern = 0xCAFE4321u;
    FillAndWait(transfer_queue_, second, kPattern);
    for (uint32_t value : ReadBufferData<uint32_t>(second)) {
      EXPECT_EQ(value, kPattern);
    }
    iree_hal_buffer_t* second_buffer = second;
    DeallocaAndWait(transfer_queue_, 1, &second_buffer);
    second.reset();
    children[1].reset();
    children[0].reset();
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(native, &stats);
    EXPECT_EQ(stats.reserve_count, 1u);
    iree_hal_slab_cache_stats_t cache_stats;
    IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache, &cache_stats));
    EXPECT_EQ(cache_stats.hit_count, 1u);
  }
}

TEST_P(QueueAllocaTest, MapsSubspanBackingInNativeCoordinates) {
  const iree_hal_pool_family_access_t family_access = {
      .family = iree_hal_queue_family(transfer_queue_),
      .usage = kQueueAllocationUsage,
  };
  const iree_hal_pool_scope_t scope = {
      .family_count = 1,
      .families = &family_access,
      .host =
          {
              .access = IREE_HAL_MEMORY_ACCESS_ALL,
              .modes = IREE_HAL_MAPPING_MODE_SCOPED |
                       IREE_HAL_MAPPING_MODE_PERSISTENT,
          },
  };
  iree_hal_slab_pool_options_t source_options;
  iree_hal_slab_pool_options_initialize(&source_options);
  Ref<iree_hal_pool_t> source_pool;
  iree_status_t status =
      iree_hal_slab_pool_create(device_group_, scope, &source_options,
                                iree_allocator_system(), source_pool.out());
  if (iree_status_is_unavailable(status)) {
    iree::Status unavailable_status(std::move(status));
    GTEST_SKIP() << "device has no host-mappable queue storage: "
                 << FormatSkipStatus(unavailable_status);
  }
  IREE_ASSERT_OK(status);

  Ref<iree_hal_buffer_t> source;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source_pool, MakeRequest(1024), iree_infinite_timeout(), source.out()));
  Ref<iree_hal_buffer_t> source_view;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(
      source, 128, 512, iree_allocator_system(), source_view.out()));

  iree_hal_fixed_block_pool_options_t options = {
      .block_size = 256,
      .frontier_capacity = 2,
      .asan = AsanOptions(),
  };
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
      source_view, 0, IREE_HAL_WHOLE_BUFFER, &options, iree_allocator_system(),
      pool.out()));

  std::array<Ref<iree_hal_buffer_t>, 2> buffers;
  for (auto& buffer : buffers) {
    auto request = MakeRequest(256);
    SemaphoreList signal(device_, {0}, {1});
    IREE_ASSERT_OK(
        iree_hal_queue_alloca(transfer_queue_, iree_hal_semaphore_list_empty(),
                              signal, pool, 1, &request, buffer.out()));
    Wait(signal);
  }
  const auto source_memory = iree_hal_buffer_memory_view(source_view);
  const auto first_memory = iree_hal_buffer_memory_view(buffers[0]);
  const auto second_memory = iree_hal_buffer_memory_view(buffers[1]);
  EXPECT_EQ(first_memory.backing, source_memory.backing);
  EXPECT_EQ(second_memory.backing, source_memory.backing);
  EXPECT_NE(first_memory.offset, second_memory.offset);

  for (size_t i = 0; i < buffers.size(); ++i) {
    FillAndWait(transfer_queue_, buffers[i], 0xA0A0A0A0u + i * 0x01010101u);
  }
  for (auto mode :
       {IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MAPPING_MODE_PERSISTENT}) {
    for (size_t i = 0; i < buffers.size(); ++i) {
      Ref<iree_hal_buffer_t> subspan;
      IREE_ASSERT_OK(iree_hal_buffer_subspan(
          buffers[i], 32, 128, iree_allocator_system(), subspan.out()));
      iree_hal_buffer_mapping_t mapping = {};
      IREE_ASSERT_OK(iree_hal_buffer_map_range(
          subspan, mode, IREE_HAL_MEMORY_ACCESS_ALL,
          IREE_HAL_BUFFER_MAP_FLAG_NONE, 16, 64, &mapping));
      IREE_ASSERT_OK(iree_hal_buffer_mapping_invalidate_range(&mapping, 0, 64));
      for (size_t j = 0; j < 64; ++j) {
        EXPECT_EQ(0xA0u + i, mapping.contents.data[j]);
      }
      memset(mapping.contents.data, 0xB0u + i, 64);
      IREE_ASSERT_OK(iree_hal_buffer_mapping_flush_range(&mapping, 0, 64));
      IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

      const auto memory = iree_hal_buffer_memory_view(buffers[i]);
      auto values = ReadBufferBytes(source_view,
                                    memory.offset - source_memory.offset, 256);
      for (size_t j = 0; j < values.size(); ++j) {
        EXPECT_EQ((j >= 48 && j < 112 ? 0xB0u : 0xA0u) + i, values[j]);
      }
      FillAndWait(transfer_queue_, buffers[i], 0xA0A0A0A0u + i * 0x01010101u);
    }
  }
  iree_hal_buffer_t* raw_buffers[] = {buffers[0], buffers[1]};
  DeallocaAndWait(transfer_queue_, IREE_ARRAYSIZE(raw_buffers), raw_buffers);
}

TEST_P(QueueAllocaTest, NestedBufferPoolsPreserveRangesAndOwnership) {
  Ref<iree_hal_pool_t> native_pool;
  IREE_ASSERT_OK(CreateTLSFPool(16384, native_pool.out()));
  auto request = MakeRequest(8192);
  Ref<iree_hal_buffer_t> source;
  SemaphoreList admission(device_, {0}, {1});
  SemaphoreList allocated(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_alloca(transfer_queue_, admission, allocated,
                                       native_pool, 1, &request, source.out()));

  // Views may be created before commitment; child allocator construction waits
  // for actual prepared storage, without caching an empty pre-commit snapshot.
  Ref<iree_hal_buffer_t> source_view;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(
      source, 256, 4096, iree_allocator_system(), source_view.out()));
  iree_hal_fixed_block_pool_options_t block_options = {
      .block_size = 512,
      .frontier_capacity = 2,
  };
  Ref<iree_hal_pool_t> blocks;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_fixed_block_pool_create_from_buffer(
                            source_view, 256, 2048, &block_options,
                            iree_allocator_system(), blocks.out()));
  EXPECT_EQ(nullptr, blocks.get());
  IREE_ASSERT_OK(iree_hal_semaphore_list_signal(admission, nullptr));
  Wait(allocated);
  FillAndWait(transfer_queue_, source, 0x11111111u);
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
      source_view, 256, 2048, &block_options, iree_allocator_system(),
      blocks.out()));
  source_view.reset();

  Ref<iree_hal_buffer_t> neighbor;
  Ref<iree_hal_buffer_t> arena_backing;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      blocks,
      iree_hal_make_pool_reservation_request(512, request.min_alignment),
      iree_immediate_timeout(), neighbor.out()));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      blocks,
      iree_hal_make_pool_reservation_request(512, request.min_alignment),
      iree_immediate_timeout(), arena_backing.out()));
  FillAndWait(transfer_queue_, neighbor, 0x22222222u);
  FillAndWait(transfer_queue_, arena_backing, 0x33333333u);

  iree_hal_tlsf_pool_options_t tlsf_options = {};
  tlsf_options.tlsf_options.alignment = 16;
  tlsf_options.tlsf_options.frontier_capacity = 2;
  Ref<iree_hal_pool_t> arena;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create_from_buffer(
      arena_backing, 32, 448, &tlsf_options, iree_allocator_system(),
      arena.out()));
  // The finite allocator rounds inward to independently maintainable ranges.
  // Private CPU transfers can require this granularity without public maps.
  const auto source_memory = iree_hal_buffer_memory_view(source);
  const iree_device_size_t alignment = std::max<iree_device_size_t>(
      16, source_memory.backing->maintenance_alignment);
  const iree_device_size_t arena_begin =
      iree_device_align(source_memory.offset + 1056, alignment) -
      source_memory.offset;
  const iree_device_size_t arena_end =
      ((source_memory.offset + 1504) & ~(alignment - 1)) - source_memory.offset;
  const iree_device_size_t second_begin =
      arena_begin + iree_device_align(64, alignment);
  arena_backing.reset();
  Ref<iree_hal_buffer_t> first;
  Ref<iree_hal_buffer_t> second;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      arena, iree_hal_make_pool_reservation_request(64, request.min_alignment),
      iree_immediate_timeout(), first.out()));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      arena, iree_hal_make_pool_reservation_request(64, request.min_alignment),
      iree_immediate_timeout(), second.out()));
  FillAndWait(transfer_queue_, first, 0xABABABABu);
  FillAndWait(transfer_queue_, second, 0xCDCDCDCDu);

  const auto values = ReadBufferData<uint8_t>(source);
  ASSERT_EQ(8192u, values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    uint8_t expected = 0x11;
    if (i >= 512 && i < 1024) {
      expected = 0x22;
    }
    if (i >= 1024 && i < 1536) {
      expected = 0x33;
    }
    if (i >= arena_begin && i < arena_begin + 64) {
      expected = 0xAB;
    }
    if (i >= second_begin && i < second_begin + 64) {
      expected = 0xCD;
    }
    ASSERT_EQ(expected, values[i]) << "byte " << i;
  }

  first.reset();
  second.reset();
  iree_hal_pool_trim(arena, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(arena, &stats);
  EXPECT_EQ(0u, stats.reservation_count);
  EXPECT_EQ(arena_end - arena_begin, stats.bytes_committed);
  iree_hal_pool_query_stats(blocks, &stats);
  EXPECT_EQ(2u, stats.reservation_count);
  arena.reset();
  iree_hal_pool_query_stats(blocks, &stats);
  EXPECT_EQ(1u, stats.reservation_count);
  neighbor.reset();
  iree_hal_pool_query_stats(blocks, &stats);
  EXPECT_EQ(0u, stats.reservation_count);
  EXPECT_EQ(2u, stats.release_count);
  blocks.reset();

  iree_hal_buffer_t* buffers[] = {source.get()};
  DeallocaAndWait(transfer_queue_, IREE_ARRAYSIZE(buffers), buffers);
  iree_hal_pool_query_stats(native_pool, &stats);
  EXPECT_EQ(0u, stats.reservation_count);
}

TEST_P(QueueAllocaTest, PluralSmallLargeAndAlignedTransaction) {
  constexpr std::array<iree_device_size_t, 3> kAllocationSizes = {256, 8192,
                                                                  1024};
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateTLSFPool(/*range_length=*/4096, pool.out()));

  std::array<iree_hal_pool_reservation_request_t, kAllocationSizes.size()>
      requests;
  for (size_t i = 0; i < requests.size(); ++i) {
    requests[i] = MakeRequest(kAllocationSizes[i]);
  }
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool, &capabilities);
  requests.back().min_alignment =
      iree_min(4096, capabilities.max_allocation_alignment);
  std::array<iree_hal_buffer_t*, kAllocationSizes.size()> raw_buffers = {};
  SemaphoreList empty_wait;
  SemaphoreList alloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_alloca(transfer_queue_, empty_wait,
                                       alloca_signal, pool, requests.size(),
                                       requests.data(), raw_buffers.data()));
  for (iree_hal_buffer_t* buffer : raw_buffers) {
    ASSERT_NE(nullptr, buffer);
  }

  std::array<Ref<iree_hal_buffer_t>, kAllocationSizes.size()> buffers = {
      Ref<iree_hal_buffer_t>(raw_buffers[0]),
      Ref<iree_hal_buffer_t>(raw_buffers[1]),
      Ref<iree_hal_buffer_t>(raw_buffers[2]),
  };
  Wait(alloca_signal);
  for (size_t i = 0; i < buffers.size(); ++i) {
    const uint32_t pattern = 0x11000000u + static_cast<uint32_t>(i);
    FillAndWait(transfer_queue_, buffers[i], pattern);
    const auto actual = ReadBufferData<uint32_t>(buffers[i]);
    ASSERT_EQ(actual.size() * sizeof(uint32_t), kAllocationSizes[i]);
    for (uint32_t value : actual) {
      EXPECT_EQ(value, pattern);
    }
    if (requests[i].min_alignment) {
      const auto memory = iree_hal_buffer_memory_view(buffers[i]);
      EXPECT_EQ(memory.offset % requests[i].min_alignment, 0u);
    }
  }

  DeallocaAndWait(transfer_queue_, raw_buffers.size(), raw_buffers.data());
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(0u, stats.reservation_count);
}

TEST_P(QueueAllocaTest, ManyBuffersOneTransaction) {
  // Model activation can commit thousands of slabs in one queue operation.
  // Small buffers exercise the same plural ownership without large backing.
  constexpr size_t kBufferCount = 2048;
  constexpr iree_device_size_t kBufferSize = 256;
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateTLSFPool(/*range_length=*/2 * 1024 * 1024, pool.out()));
  std::vector<iree_hal_pool_reservation_request_t> requests(
      kBufferCount, MakeRequest(kBufferSize));
  std::vector<iree_hal_buffer_t*> raw_buffers(kBufferCount, nullptr);
  SemaphoreList allocated(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      transfer_queue_, iree_hal_semaphore_list_empty(), allocated, pool,
      requests.size(), requests.data(), raw_buffers.data()));
  std::vector<Ref<iree_hal_buffer_t>> buffers;
  buffers.reserve(kBufferCount);
  for (iree_hal_buffer_t* buffer : raw_buffers) {
    buffers.emplace_back(buffer);
  }
  Wait(allocated);
  for (const auto& buffer : buffers) {
    ASSERT_NE(buffer.get(), nullptr);
    EXPECT_NE(iree_hal_buffer_memory_view(buffer).backing, nullptr);
    EXPECT_EQ(iree_hal_buffer_byte_length(buffer), kBufferSize);
  }
  for (size_t index : {size_t{0}, kBufferCount / 2, kBufferCount - 1}) {
    const uint32_t pattern = 0x11000000u + static_cast<uint32_t>(index);
    FillAndWait(transfer_queue_, buffers[index], pattern);
    for (uint32_t value : ReadBufferData<uint32_t>(buffers[index])) {
      EXPECT_EQ(value, pattern);
    }
  }
  DeallocaAndWait(transfer_queue_, raw_buffers.size(), raw_buffers.data());
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
}

TEST_P(QueueAllocaTest, WaitDependencyControlsReadiness) {
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateSlabPool(pool.out()));

  const iree_hal_pool_reservation_request_t request = MakeRequest(/*size=*/512);
  Ref<iree_hal_buffer_t> buffer;
  SemaphoreList alloca_wait(device_, {0}, {1});
  SemaphoreList alloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(
      iree_hal_queue_alloca(transfer_queue_, alloca_wait, alloca_signal, pool,
                            /*request_count=*/1, &request, buffer.out()));
  ASSERT_NE(nullptr, buffer.get());
  EXPECT_FALSE(iree_hal_semaphore_list_poll(alloca_signal));

  IREE_ASSERT_OK(iree_hal_semaphore_list_signal(alloca_wait,
                                                /*frontier=*/nullptr));
  Wait(alloca_signal);
  constexpr uint32_t kPattern = 0xC001CAFEu;
  FillAndWait(transfer_queue_, buffer, kPattern);
  EXPECT_EQ(kPattern, ReadBufferData<uint32_t>(buffer)[0]);

  iree_hal_buffer_t* buffers[] = {buffer.get()};
  DeallocaAndWait(transfer_queue_, IREE_ARRAYSIZE(buffers), buffers);
}

TEST_P(QueueAllocaTest, ExportFollowsAllocationLifetime) {
  Ref<iree_hal_pool_t> pool;
  iree_status_t status =
      CreateHostExportableTLSFPool(/*range_length=*/4096, pool.out());
  if (iree_status_is_unavailable(status)) {
    iree::Status unavailable_status(std::move(status));
    GTEST_SKIP() << "device has no host-exportable queue storage: "
                 << FormatSkipStatus(unavailable_status);
  }
  IREE_ASSERT_OK(status);
  auto request = MakeRequest(/*size=*/256);

  Ref<iree_hal_buffer_t> buffer;
  SemaphoreList alloca_wait(device_, {0}, {1});
  SemaphoreList alloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(
      iree_hal_queue_alloca(transfer_queue_, alloca_wait, alloca_signal, pool,
                            /*request_count=*/1, &request, buffer.out()));
  Ref<iree_hal_buffer_t> subspan;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(
      buffer, 64, 64, iree_allocator_system(), subspan.out()));

  iree_hal_external_storage_t external_buffer = {};
  iree_hal_buffer_range_t logical_range = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_buffer_export(
          subspan, iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
          iree_hal_make_external_storage_class(
              IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
          &external_buffer, &logical_range));
  EXPECT_EQ(IREE_HAL_EXTERNAL_STORAGE_TYPE_NONE,
            external_buffer.storage_class.type);
  IREE_ASSERT_OK(
      iree_hal_semaphore_list_signal(alloca_wait, /*frontier=*/nullptr));
  Wait(alloca_signal);

  constexpr uint32_t kPattern = 0xA11CA7EDu;
  FillAndWait(transfer_queue_, buffer, kPattern);
  IREE_ASSERT_OK(iree_hal_buffer_export(
      subspan, iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
      iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
      &external_buffer, &logical_range));
  EXPECT_EQ(IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER,
            external_buffer.storage_class.type);
  EXPECT_EQ(64u, logical_range.byte_length);
  iree_hal_external_storage_t root_export = {};
  iree_hal_buffer_range_t root_range = {};
  IREE_ASSERT_OK(iree_hal_buffer_export(
      buffer, iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
      iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
      &root_export, &root_range));
  EXPECT_EQ(root_export.payload.host_pointer,
            external_buffer.payload.host_pointer);
  EXPECT_EQ(root_range.byte_offset + 64, logical_range.byte_offset);
  EXPECT_EQ(root_export.byte_length, external_buffer.byte_length);
  for (uint32_t value : ReadBufferData<uint32_t>(subspan)) {
    EXPECT_EQ(kPattern, value);
  }
  iree_hal_external_storage_release(&root_export);
  iree_hal_external_storage_release(&external_buffer);

  iree_hal_buffer_t* buffers[] = {buffer.get()};
  DeallocaAndWait(transfer_queue_, IREE_ARRAYSIZE(buffers), buffers);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(0u, stats.reservation_count);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_buffer_export(
          subspan, iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
          iree_hal_make_external_storage_class(
              IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
          &external_buffer, &logical_range));
  EXPECT_EQ(IREE_HAL_EXTERNAL_STORAGE_TYPE_NONE,
            external_buffer.storage_class.type);
  EXPECT_EQ(0u, external_buffer.byte_length);
  EXPECT_EQ(nullptr, external_buffer.payload.host_pointer);
  EXPECT_EQ(0u, logical_range.byte_length);
}

TEST_P(QueueAllocaTest, ValidationFailureLeavesOutputsUntouched) {
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateSlabPool(pool.out()));

  std::array<iree_hal_pool_reservation_request_t, 2> requests = {
      MakeRequest(/*size=*/256),
      MakeRequest(/*size=*/0),
  };
  auto* const sentinel0 = reinterpret_cast<iree_hal_buffer_t*>(uintptr_t{0x1});
  auto* const sentinel1 = reinterpret_cast<iree_hal_buffer_t*>(uintptr_t{0x2});
  std::array<iree_hal_buffer_t*, 2> outputs = {sentinel0, sentinel1};
  SemaphoreList empty_wait;
  SemaphoreList signal(device_, {0}, {1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_queue_alloca(transfer_queue_, empty_wait, signal, pool,
                            requests.size(), requests.data(), outputs.data()));
  EXPECT_EQ(sentinel0, outputs[0]);
  EXPECT_EQ(sentinel1, outputs[1]);
  EXPECT_FALSE(iree_hal_semaphore_list_poll(signal));

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(0u, stats.reservation_count);
}

TEST_P(QueueAllocaTest, PoolValidationIsAllOrNothing) {
  constexpr iree_device_size_t kBlockSize = 512;
  Ref<iree_hal_pool_t> backing_pool;
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateFixedBlockPool(kBlockSize, /*block_count=*/2,
                                      backing_pool.out(), pool.out()));

  std::array<iree_hal_pool_reservation_request_t, 2> invalid_requests = {
      MakeRequest(kBlockSize),
      MakeRequest(kBlockSize + 1),
  };
  auto* const sentinel0 = reinterpret_cast<iree_hal_buffer_t*>(uintptr_t{0x1});
  auto* const sentinel1 = reinterpret_cast<iree_hal_buffer_t*>(uintptr_t{0x2});
  std::array<iree_hal_buffer_t*, 2> outputs = {sentinel0, sentinel1};
  SemaphoreList empty_wait;
  SemaphoreList failed_signal(device_, {0}, {1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_queue_alloca(transfer_queue_, empty_wait, failed_signal, pool,
                            invalid_requests.size(), invalid_requests.data(),
                            outputs.data()));
  EXPECT_EQ(sentinel0, outputs[0]);
  EXPECT_EQ(sentinel1, outputs[1]);
  EXPECT_FALSE(iree_hal_semaphore_list_poll(failed_signal));

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(0u, stats.reservation_count);

  std::array<iree_hal_pool_reservation_request_t, 2> valid_requests = {
      MakeRequest(kBlockSize),
      MakeRequest(kBlockSize),
  };
  outputs = {};
  SemaphoreList valid_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      transfer_queue_, empty_wait, valid_signal, pool, valid_requests.size(),
      valid_requests.data(), outputs.data()));
  Wait(valid_signal);
  DeallocaAndWait(transfer_queue_, outputs.size(), outputs.data());
  for (iree_hal_buffer_t* buffer : outputs) {
    iree_hal_buffer_release(buffer);
  }
}

TEST_P(QueueAllocaTest, DuplicateDeallocaLeavesEpochLive) {
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateSlabPool(pool.out()));

  const iree_hal_pool_reservation_request_t request = MakeRequest(/*size=*/256);
  Ref<iree_hal_buffer_t> buffer;
  SemaphoreList empty_wait;
  SemaphoreList alloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(
      iree_hal_queue_alloca(transfer_queue_, empty_wait, alloca_signal, pool,
                            /*request_count=*/1, &request, buffer.out()));
  Wait(alloca_signal);

  iree_hal_buffer_t* duplicate_buffers[] = {buffer.get(), buffer.get()};
  SemaphoreList failed_signal(device_, {0}, {1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_queue_dealloca(transfer_queue_, empty_wait, failed_signal,
                              IREE_ARRAYSIZE(duplicate_buffers),
                              duplicate_buffers));
  EXPECT_FALSE(iree_hal_semaphore_list_poll(failed_signal));

  constexpr uint32_t kPattern = 0xD0011CA7u;
  FillAndWait(transfer_queue_, buffer, kPattern);
  EXPECT_EQ(kPattern, ReadBufferData<uint32_t>(buffer)[0]);
  iree_hal_buffer_t* buffers[] = {buffer.get()};
  DeallocaAndWait(transfer_queue_, IREE_ARRAYSIZE(buffers), buffers);
}

TEST_P(QueueAllocaTest, MixedPoolDeallocaLeavesEpochsLive) {
  Ref<iree_hal_pool_t> pool0;
  Ref<iree_hal_pool_t> pool1;
  IREE_ASSERT_OK(CreateSlabPool(pool0.out()));
  IREE_ASSERT_OK(CreateSlabPool(pool1.out()));

  const iree_hal_pool_reservation_request_t request = MakeRequest(/*size=*/256);
  Ref<iree_hal_buffer_t> buffer0;
  Ref<iree_hal_buffer_t> buffer1;
  SemaphoreList empty_wait;
  SemaphoreList signal0(device_, {0}, {1});
  SemaphoreList signal1(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_alloca(transfer_queue_, empty_wait, signal0,
                                       pool0, /*request_count=*/1, &request,
                                       buffer0.out()));
  IREE_ASSERT_OK(iree_hal_queue_alloca(transfer_queue_, empty_wait, signal1,
                                       pool1, /*request_count=*/1, &request,
                                       buffer1.out()));
  Wait(signal0);
  Wait(signal1);

  iree_hal_buffer_t* mixed_buffers[] = {buffer0.get(), buffer1.get()};
  SemaphoreList failed_signal(device_, {0}, {1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_queue_dealloca(transfer_queue_, empty_wait, failed_signal,
                              IREE_ARRAYSIZE(mixed_buffers), mixed_buffers));
  EXPECT_FALSE(iree_hal_semaphore_list_poll(failed_signal));

  constexpr uint32_t kPattern0 = 0x0011CA70u;
  constexpr uint32_t kPattern1 = 0x0011CA71u;
  FillAndWait(transfer_queue_, buffer0, kPattern0);
  FillAndWait(transfer_queue_, buffer1, kPattern1);
  EXPECT_EQ(kPattern0, ReadBufferData<uint32_t>(buffer0)[0]);
  EXPECT_EQ(kPattern1, ReadBufferData<uint32_t>(buffer1)[0]);
  iree_hal_buffer_t* buffers0[] = {buffer0.get()};
  iree_hal_buffer_t* buffers1[] = {buffer1.get()};
  DeallocaAndWait(transfer_queue_, IREE_ARRAYSIZE(buffers0), buffers0);
  DeallocaAndWait(transfer_queue_, IREE_ARRAYSIZE(buffers1), buffers1);
}

TEST_P(QueueAllocaTest, ExhaustionRetriesAfterPluralRelease) {
  constexpr iree_device_size_t kBlockSize = 512;
  Ref<iree_hal_pool_t> backing_pool;
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateFixedBlockPool(kBlockSize, /*block_count=*/2,
                                      backing_pool.out(), pool.out()));

  std::array<iree_hal_pool_reservation_request_t, 2> requests = {
      MakeRequest(kBlockSize),
      MakeRequest(kBlockSize),
  };
  std::array<iree_hal_buffer_t*, 2> first_buffers = {};
  SemaphoreList empty_wait;
  SemaphoreList first_alloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      transfer_queue_, empty_wait, first_alloca_signal, pool, requests.size(),
      requests.data(), first_buffers.data()));
  Wait(first_alloca_signal);

  std::array<iree_hal_buffer_t*, 2> second_buffers = {};
  SemaphoreList second_alloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      transfer_queue_, empty_wait, second_alloca_signal, pool, requests.size(),
      requests.data(), second_buffers.data()));
  EXPECT_FALSE(iree_hal_semaphore_list_poll(second_alloca_signal));

  SemaphoreList first_dealloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_dealloca(
      transfer_queue_, first_alloca_signal, first_dealloca_signal,
      first_buffers.size(), first_buffers.data()));
  Wait(second_alloca_signal);
  Wait(first_dealloca_signal);
  for (iree_hal_buffer_t* buffer : first_buffers) {
    iree_hal_buffer_release(buffer);
  }

  constexpr uint32_t kPattern = 0xB10CA110u;
  FillAndWait(transfer_queue_, second_buffers[0], kPattern);
  EXPECT_EQ(kPattern, ReadBufferData<uint32_t>(second_buffers[0])[0]);
  DeallocaAndWait(transfer_queue_, second_buffers.size(),
                  second_buffers.data());
  for (iree_hal_buffer_t* buffer : second_buffers) {
    iree_hal_buffer_release(buffer);
  }
}

TEST_P(QueueAllocaTest, CrossQueueReusePreservesDeathFrontier) {
  iree_hal_queue_t* sibling_queue = FindSiblingQueue(transfer_queue_);
  if (!sibling_queue) {
    GTEST_SKIP() << "queue family has only one provisioned queue";
  }

  constexpr iree_device_size_t kBlockSize = 512;
  Ref<iree_hal_pool_t> backing_pool;
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateFixedBlockPool(kBlockSize, /*block_count=*/1,
                                      backing_pool.out(), pool.out()));
  const iree_hal_pool_reservation_request_t request = MakeRequest(kBlockSize);

  Ref<iree_hal_buffer_t> first_buffer;
  SemaphoreList empty_wait;
  SemaphoreList first_alloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      transfer_queue_, empty_wait, first_alloca_signal, pool,
      /*request_count=*/1, &request, first_buffer.out()));
  Wait(first_alloca_signal);

  iree_hal_buffer_t* first_buffers[] = {first_buffer.get()};
  SemaphoreList first_dealloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_dealloca(
      transfer_queue_, first_alloca_signal, first_dealloca_signal,
      IREE_ARRAYSIZE(first_buffers), first_buffers));

  Ref<iree_hal_buffer_t> second_buffer;
  SemaphoreList second_alloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      sibling_queue, empty_wait, second_alloca_signal, pool,
      /*request_count=*/1, &request, second_buffer.out()));
  Wait(second_alloca_signal);
  Wait(first_dealloca_signal);

  constexpr uint32_t kPattern = 0xC2055A1Eu;
  FillAndWait(sibling_queue, second_buffer, kPattern);
  EXPECT_EQ(kPattern, ReadBufferData<uint32_t>(second_buffer)[0]);
  iree_hal_buffer_t* second_buffers[] = {second_buffer.get()};
  DeallocaAndWait(sibling_queue, IREE_ARRAYSIZE(second_buffers),
                  second_buffers);
}

CTS_REGISTER_TEST_SUITE(QueueAllocaTest);

}  // namespace iree::hal::cts
