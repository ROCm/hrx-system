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
#include "iree/hal/memory/maintenance.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf_pool.h"

namespace iree::hal::cts {

namespace {

iree_hal_queue_family_affinity_t QueueFamilyAffinity(iree_hal_queue_t* queue) {
  return iree_hal_make_queue_family_affinity(
      iree_hal_queue_family_ordinal(iree_hal_queue_family(queue)));
}

iree_hal_buffer_params_t MakeAllocationParams(iree_hal_queue_t* queue) {
  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE;
  params.queue_family_affinity = QueueFamilyAffinity(queue);
  return params;
}

iree_hal_pool_reservation_request_t MakeRequest(iree_hal_queue_t* queue,
                                                iree_device_size_t size) {
  return {
      .params = MakeAllocationParams(queue),
      .allocation_size = size,
  };
}

// Keeps native slab allocation and advice intact while materializing ordinary
// subspans. Queue allocation must accept these just like native buffer views.
class SubspanSlabProvider : public iree_hal_slab_provider_t {
 public:
  explicit SubspanSlabProvider(iree_hal_slab_provider_t* inner)
      : inner_(inner) {
    iree_hal_slab_provider_retain(inner_);
    iree_hal_slab_provider_initialize(&vtable_, this);
  }

  ~SubspanSlabProvider() { iree_hal_slab_provider_release(inner_); }

  // Borrowed materializations, valid until their queue allocation is released.
  std::vector<iree_hal_buffer_t*> views;

 private:
  static SubspanSlabProvider* Cast(iree_hal_slab_provider_t* provider) {
    return static_cast<SubspanSlabProvider*>(provider);
  }
  static const SubspanSlabProvider* Cast(
      const iree_hal_slab_provider_t* provider) {
    return static_cast<const SubspanSlabProvider*>(provider);
  }

  // Native provider retained for the adapter's lifetime.
  iree_hal_slab_provider_t* inner_;
  // Delegates native operations and changes only buffer materialization.
  static const iree_hal_slab_provider_vtable_t vtable_;
};

const iree_hal_slab_provider_vtable_t SubspanSlabProvider::vtable_ = {
    .destroy =
        [](iree_hal_slab_provider_t* provider) { delete Cast(provider); },
    .acquire_slab =
        [](iree_hal_slab_provider_t* provider, iree_device_size_t min_length,
           iree_device_size_t min_alignment, iree_hal_slab_t* out_slab) {
          return iree_hal_slab_provider_acquire_slab(
              Cast(provider)->inner_, min_length, min_alignment, out_slab);
        },
    .release_slab =
        [](iree_hal_slab_provider_t* provider, const iree_hal_slab_t* slab) {
          iree_hal_slab_provider_release_slab(Cast(provider)->inner_, slab);
        },
    .wrap_buffer =
        [](iree_hal_slab_provider_t* provider, const iree_hal_slab_t* slab,
           iree_device_size_t slab_offset, iree_device_size_t allocation_size,
           iree_hal_buffer_params_t params,
           iree_hal_buffer_release_callback_t release_callback,
           iree_hal_buffer_t** out_buffer) {
          auto* self = Cast(provider);
          Ref<iree_hal_buffer_t> backing;
          IREE_RETURN_IF_ERROR(iree_hal_slab_provider_wrap_buffer(
              self->inner_, slab, 0, slab->length, params,
              iree_hal_buffer_release_callback_null(), backing.out()));
          IREE_RETURN_IF_ERROR(iree_hal_subspan_buffer_create_with_callback(
              backing, iree_hal_buffer_byte_offset(backing) + slab_offset,
              allocation_size, release_callback, iree_allocator_system(),
              out_buffer));
          self->views.push_back(*out_buffer);
          return iree_ok_status();
        },
    .validate_asan_options =
        [](const iree_hal_slab_provider_t* provider,
           const iree_hal_asan_pool_options_t* options) {
          return iree_hal_slab_provider_validate_asan_options(
              Cast(provider)->inner_, options);
        },
    .advise_asan_range =
        [](iree_hal_slab_provider_t* provider, const iree_hal_slab_t* slab,
           iree_device_size_t backing_offset,
           iree_hal_asan_range_advice_flags_t flags,
           const iree_hal_asan_allocation_layout_t* layout) {
          iree_hal_slab_provider_advise_asan_range(
              Cast(provider)->inner_, slab, backing_offset, flags, layout);
        },
    .prefault =
        [](iree_hal_slab_provider_t* provider, const iree_hal_slab_t* slab,
           iree_device_size_t offset, iree_device_size_t length) {
          iree_hal_slab_provider_prefault(Cast(provider)->inner_, slab, offset,
                                          length);
        },
    .trim =
        [](iree_hal_slab_provider_t* provider,
           iree_hal_pool_trim_flags_t flags) {
          iree_hal_slab_provider_trim(Cast(provider)->inner_, flags);
        },
    .query_stats =
        [](const iree_hal_slab_provider_t* provider,
           iree_hal_slab_provider_visited_set_t* visited,
           iree_hal_slab_provider_stats_t* out_stats) {
          iree_hal_slab_provider_query_stats(Cast(provider)->inner_, visited,
                                             out_stats);
        },
    .query_properties =
        [](const iree_hal_slab_provider_t* provider,
           iree_hal_slab_provider_properties_t* properties) {
          iree_hal_slab_provider_query_properties(Cast(provider)->inner_,
                                                  properties);
        },
};

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

  iree_status_t QueryPoolBackend(iree_hal_queue_pool_backend_t* out_backend) {
    return iree_hal_device_query_queue_pool_backend(
        device_, iree_hal_queue_family(transfer_queue_), out_backend);
  }

  iree_status_t CreatePassthroughPool(iree_hal_pool_t** out_pool) {
    iree_hal_queue_pool_backend_t backend = {};
    IREE_RETURN_IF_ERROR(QueryPoolBackend(&backend));
    if (!backend.slab_provider || !backend.notification) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "queue pool backend query returned an incomplete bundle");
    }
    iree_hal_passthrough_pool_options_t options = {};
    options.asan = backend.asan;
    return iree_hal_passthrough_pool_create(
        options, backend.slab_provider, backend.notification,
        backend.frontier_tracker, backend.maintenance, iree_allocator_system(),
        out_pool);
  }

  iree_status_t CreateTLSFPool(iree_device_size_t range_length,
                               iree_hal_pool_t** out_pool) {
    iree_hal_queue_pool_backend_t backend = {};
    IREE_RETURN_IF_ERROR(QueryPoolBackend(&backend));
    if (!backend.slab_provider || !backend.notification) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "queue pool backend query returned an incomplete bundle");
    }
    iree_hal_tlsf_pool_options_t options = {};
    options.tlsf_options.range_length = range_length;
    options.tlsf_options.alignment = IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
    options.tlsf_options.frontier_capacity = 2;
    options.asan = backend.asan;
    iree_hal_passthrough_pool_options_t backing_options = {};
    backing_options.epoch_query = backend.epoch_query;
    iree_hal_pool_t* backing_pool = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_passthrough_pool_create(
        backing_options, backend.slab_provider, backend.notification,
        backend.frontier_tracker, backend.maintenance, iree_allocator_system(),
        &backing_pool));
    iree_status_t status = iree_hal_tlsf_pool_create(
        backing_pool, &options, iree_allocator_system(), out_pool);
    iree_hal_pool_release(backing_pool);
    return status;
  }

  iree_status_t CreateFixedBlockPool(iree_device_size_t block_size,
                                     uint32_t block_count,
                                     iree_hal_pool_t** out_backing_pool,
                                     iree_hal_pool_t** out_pool) {
    iree_hal_queue_pool_backend_t backend = {};
    IREE_RETURN_IF_ERROR(QueryPoolBackend(&backend));
    if (!backend.slab_provider || !backend.notification) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "queue pool backend query returned an incomplete bundle");
    }
    iree_hal_fixed_block_pool_options_t options = {};
    options.block_size = block_size;
    options.blocks_per_slab = block_count;
    options.frontier_capacity = 2;
    options.asan = backend.asan;
    return CreateFiniteBlockPool(backend, options, iree_allocator_system(),
                                 out_backing_pool, out_pool);
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

TEST_P(QueueAllocaTest, ExactQueueAndExplicitPool) {
  constexpr iree_device_size_t kAllocationSize = 1024;
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreatePassthroughPool(pool.out()));

  // Use the native guarantee, which may exceed the CPU heap alignment.
  iree_hal_queue_pool_backend_t backend = {};
  IREE_ASSERT_OK(QueryPoolBackend(&backend));
  iree_hal_slab_provider_properties_t properties;
  iree_hal_slab_provider_query_properties(backend.slab_provider, &properties);
  iree_hal_pool_reservation_request_t request =
      MakeRequest(transfer_queue_, kAllocationSize);
  request.params.min_alignment = properties.allocation_alignment;
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
  EXPECT_EQ(memory.backing->allocation_alignment,
            properties.allocation_alignment);
  EXPECT_EQ(memory.offset % properties.allocation_alignment, 0u);
  EXPECT_GE(iree_hal_buffer_byte_length(buffer), kAllocationSize);
  const iree_hal_buffer_placement_t placement =
      iree_hal_buffer_allocation_placement(buffer);
  EXPECT_EQ(device_, placement.device);
  EXPECT_EQ(QueueFamilyAffinity(transfer_queue_),
            placement.queue_family_affinity);

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
  iree_hal_queue_pool_backend_t backend = {};
  IREE_ASSERT_OK(QueryPoolBackend(&backend));
  iree_hal_slab_provider_properties_t properties;
  iree_hal_slab_provider_query_properties(backend.slab_provider, &properties);
  // Both allocation policies can give and receive the same cached range.
  for (uint32_t policies = 0; policies < 4; ++policies) {
    SCOPED_TRACE(policies);
    iree_hal_passthrough_pool_options_t native_options = {};
    native_options.epoch_query = backend.epoch_query;
    Ref<iree_hal_pool_t> native;
    IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
        native_options, backend.slab_provider, backend.notification,
        backend.frontier_tracker, backend.maintenance, iree_allocator_system(),
        native.out()));
    iree_hal_slab_cache_options_t cache_options;
    iree_hal_slab_cache_options_initialize(&cache_options);
    cache_options.slab = MakeRequest(transfer_queue_, 65536);
    // Share one class satisfying both the native guarantee and the 256-byte
    // fixed-block stride, including sources that strengthen alignment on
    // demand.
    cache_options.slab.params.min_alignment =
        iree_max(properties.allocation_alignment,
                 iree_min(256, properties.max_allocation_alignment));
    Ref<iree_hal_pool_t> cache;
    IREE_ASSERT_OK(iree_hal_slab_cache_create(
        native, &cache_options, iree_allocator_system(), cache.out()));
    const auto request = MakeRequest(transfer_queue_, 256);
    const iree_hal_buffer_backing_facts_t* first_backing = nullptr;
    Ref<iree_hal_pool_t> children[2];
    for (uint32_t iteration = 0; iteration < 2; ++iteration) {
      auto& child = children[iteration];
      if (policies & (1u << iteration)) {
        iree_hal_fixed_block_pool_options_t options = {};
        options.block_size = 256;
        options.blocks_per_slab = 16;
        options.frontier_capacity = 2;
        options.asan = backend.asan;
        IREE_ASSERT_OK(iree_hal_fixed_block_pool_create(
            cache, &options, iree_allocator_system(), child.out()));
      } else {
        iree_hal_tlsf_pool_options_t options = {};
        options.tlsf_options.range_length = 4096;
        options.tlsf_options.alignment = IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
        options.tlsf_options.frontier_capacity = 2;
        options.asan = backend.asan;
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
      // Both children remain alive. Observe the automatic whole-range return
      // before asking the other policy to reuse it, without explicit trimming.
      iree_hal_memory_maintenance_call(
          backend.maintenance, [](void*) {}, nullptr);
      iree_hal_pool_stats_t stats;
      iree_hal_pool_query_stats(child, &stats);
      EXPECT_EQ(stats.bytes_committed, 0u);
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
  iree_hal_queue_pool_backend_t backend = {};
  IREE_ASSERT_OK(QueryPoolBackend(&backend));
  for (uint32_t policies = 0; policies < 4; ++policies) {
    SCOPED_TRACE(policies);
    // A finite ordinary backing pool bounds the entire composition to one
    // real native range, regardless of the children's allocation policies.
    Ref<iree_hal_pool_t> native;
    Ref<iree_hal_pool_t> source;
    IREE_ASSERT_OK(CreateFixedBlockPool(65536, 1, native.out(), source.out()));
    iree_hal_slab_cache_options_t cache_options;
    iree_hal_slab_cache_options_initialize(&cache_options);
    cache_options.slab = MakeRequest(transfer_queue_, 65536);
    iree_hal_pool_capabilities_t capabilities;
    iree_hal_pool_query_capabilities(source, &capabilities);
    cache_options.slab.params.min_alignment =
        iree_min(4096, capabilities.max_allocation_alignment);
    Ref<iree_hal_pool_t> cache;
    IREE_ASSERT_OK(iree_hal_slab_cache_create(
        source, &cache_options, iree_allocator_system(), cache.out()));
    Ref<iree_hal_pool_t> children[2];
    for (uint32_t i = 0; i < 2; ++i) {
      if (policies & (1u << i)) {
        iree_hal_fixed_block_pool_options_t options = {};
        options.block_size = 256;
        options.blocks_per_slab = 16;
        options.frontier_capacity = 2;
        options.asan = backend.asan;
        IREE_ASSERT_OK(iree_hal_fixed_block_pool_create(
            cache, &options, iree_allocator_system(), children[i].out()));
      } else {
        iree_hal_tlsf_pool_options_t options = {};
        options.tlsf_options.range_length = 4096;
        options.tlsf_options.alignment = IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
        options.tlsf_options.frontier_capacity = 2;
        options.asan = backend.asan;
        IREE_ASSERT_OK(iree_hal_tlsf_pool_create(
            cache, &options, iree_allocator_system(), children[i].out()));
      }
    }
    ASSERT_NE(iree_hal_pool_notification(children[0]),
              iree_hal_pool_notification(children[1]));
    auto* cache_notification = iree_hal_pool_notification(cache);
    ASSERT_NE(cache_notification, iree_hal_pool_notification(children[1]));
    const auto request = MakeRequest(transfer_queue_, 256);
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
    iree_hal_memory_maintenance_call(
        backend.maintenance, [](void*) {}, nullptr);
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(native, &stats);
    EXPECT_EQ(stats.reserve_count, 1u);
    iree_hal_slab_cache_stats_t cache_stats;
    IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache, &cache_stats));
    EXPECT_EQ(cache_stats.hit_count, 1u);
  }
}

TEST_P(QueueAllocaTest, MapsSubspanBackingInNativeCoordinates) {
  iree_hal_queue_pool_backend_t backend = {};
  IREE_ASSERT_OK(QueryPoolBackend(&backend));
  auto provider = std::unique_ptr<SubspanSlabProvider,
                                  decltype(&iree_hal_slab_provider_release)>(
      new SubspanSlabProvider(backend.slab_provider),
      iree_hal_slab_provider_release);
  iree_hal_slab_provider_properties_t properties;
  iree_hal_slab_provider_query_properties(provider.get(), &properties);
  if (!iree_all_bits_set(properties.memory_type,
                         IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    GTEST_SKIP() << "queue pool backing is not host visible";
  }

  iree_hal_fixed_block_pool_options_t options = {};
  options.block_size = 256;
  options.blocks_per_slab = 2;
  options.frontier_capacity = 2;
  options.asan = backend.asan;
  backend.slab_provider = provider.get();
  Ref<iree_hal_pool_t> backing_pool;
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateFiniteBlockPool(backend, options,
                                       iree_allocator_system(),
                                       backing_pool.out(), pool.out()));

  std::array<Ref<iree_hal_buffer_t>, 2> buffers;
  for (auto& buffer : buffers) {
    auto request = MakeRequest(transfer_queue_, 256);
    request.params.usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                            IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
    SemaphoreList signal(device_, {0}, {1});
    IREE_ASSERT_OK(
        iree_hal_queue_alloca(transfer_queue_, iree_hal_semaphore_list_empty(),
                              signal, pool, 1, &request, buffer.out()));
    Wait(signal);
  }
  ASSERT_EQ(1u, provider->views.size());
  const auto source_memory = iree_hal_buffer_memory_view(provider->views[0]);
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
      auto values = ReadBufferBytes(provider->views[0],
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
  auto request = MakeRequest(transfer_queue_, 8192);
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
  iree_hal_fixed_block_pool_options_t block_options = {};
  block_options.block_size = 512;
  block_options.frontier_capacity = 2;
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
      blocks, request.params, 512, iree_immediate_timeout(), neighbor.out()));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(blocks, request.params, 512,
                                               iree_immediate_timeout(),
                                               arena_backing.out()));
  FillAndWait(transfer_queue_, neighbor, 0x22222222u);
  FillAndWait(transfer_queue_, arena_backing, 0x33333333u);

  iree_hal_tlsf_pool_options_t tlsf_options = {};
  tlsf_options.tlsf_options.alignment = 16;
  tlsf_options.tlsf_options.frontier_capacity = 2;
  Ref<iree_hal_pool_t> arena;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create_from_buffer(
      arena_backing, 32, 448, &tlsf_options, iree_allocator_system(),
      arena.out()));
  arena_backing.reset();
  Ref<iree_hal_buffer_t> first;
  Ref<iree_hal_buffer_t> second;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      arena, request.params, 64, iree_immediate_timeout(), first.out()));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      arena, request.params, 64, iree_immediate_timeout(), second.out()));
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
    if (i >= 1056 && i < 1120) {
      expected = 0xAB;
    }
    if (i >= 1120 && i < 1184) {
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
  EXPECT_EQ(448u, stats.bytes_committed);
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
    requests[i] = MakeRequest(transfer_queue_, kAllocationSizes[i]);
  }
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool, &capabilities);
  requests.back().params.min_alignment =
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
    if (requests[i].params.min_alignment) {
      const auto memory = iree_hal_buffer_memory_view(buffers[i]);
      EXPECT_EQ(memory.offset % requests[i].params.min_alignment, 0u);
    }
  }

  DeallocaAndWait(transfer_queue_, raw_buffers.size(), raw_buffers.data());
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(0u, stats.reservation_count);
}

TEST_P(QueueAllocaTest, WaitDependencyControlsReadiness) {
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreatePassthroughPool(pool.out()));

  const iree_hal_pool_reservation_request_t request =
      MakeRequest(transfer_queue_, /*size=*/512);
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
  IREE_ASSERT_OK(CreateTLSFPool(/*range_length=*/4096, pool.out()));
  auto request = MakeRequest(transfer_queue_, /*size=*/256);
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool, &capabilities);
  request.params.usage |=
      capabilities.supported_usage & IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;

  Ref<iree_hal_buffer_t> buffer;
  SemaphoreList alloca_wait(device_, {0}, {1});
  SemaphoreList alloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(
      iree_hal_queue_alloca(transfer_queue_, alloca_wait, alloca_signal, pool,
                            /*request_count=*/1, &request, buffer.out()));
  Ref<iree_hal_buffer_t> subspan;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(
      buffer, 64, 64, iree_allocator_system(), subspan.out()));

  iree_hal_external_buffer_t external_buffer = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_buffer_export(
          subspan, IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
          IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer));
  EXPECT_EQ(IREE_HAL_EXTERNAL_BUFFER_TYPE_NONE, external_buffer.type);
  IREE_ASSERT_OK(
      iree_hal_semaphore_list_signal(alloca_wait, /*frontier=*/nullptr));
  Wait(alloca_signal);

  constexpr uint32_t kPattern = 0xA11CA7EDu;
  FillAndWait(transfer_queue_, buffer, kPattern);
  iree_status_t export_status = iree_hal_buffer_export(
      subspan, IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer);
  if (iree_status_is_ok(export_status)) {
    EXPECT_EQ(IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
              external_buffer.type);
    EXPECT_EQ(64u, external_buffer.size);
    iree_hal_external_buffer_t root_export = {};
    IREE_ASSERT_OK(iree_hal_buffer_export(
        buffer, IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
        IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &root_export));
    EXPECT_EQ(root_export.handle.device_allocation.ptr + 64,
              external_buffer.handle.device_allocation.ptr);
    for (uint32_t value : ReadBufferData<uint32_t>(subspan)) {
      EXPECT_EQ(kPattern, value);
    }
  } else {
    // External pointer support is optional; allocation readiness is not.
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, export_status);
    EXPECT_EQ(IREE_HAL_EXTERNAL_BUFFER_TYPE_NONE, external_buffer.type);
  }

  iree_hal_buffer_t* buffers[] = {buffer.get()};
  DeallocaAndWait(transfer_queue_, IREE_ARRAYSIZE(buffers), buffers);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(0u, stats.reservation_count);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_buffer_export(
          subspan, IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
          IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer));
  EXPECT_EQ(IREE_HAL_EXTERNAL_BUFFER_TYPE_NONE, external_buffer.type);
  EXPECT_EQ(0u, external_buffer.size);
  EXPECT_EQ(nullptr, external_buffer.handle.host_allocation.ptr);
}

TEST_P(QueueAllocaTest, ValidationFailureLeavesOutputsUntouched) {
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreatePassthroughPool(pool.out()));

  std::array<iree_hal_pool_reservation_request_t, 2> requests = {
      MakeRequest(transfer_queue_, /*size=*/256),
      MakeRequest(transfer_queue_, /*size=*/0),
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
      MakeRequest(transfer_queue_, kBlockSize),
      MakeRequest(transfer_queue_, kBlockSize + 1),
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
      MakeRequest(transfer_queue_, kBlockSize),
      MakeRequest(transfer_queue_, kBlockSize),
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
  IREE_ASSERT_OK(CreatePassthroughPool(pool.out()));

  const iree_hal_pool_reservation_request_t request =
      MakeRequest(transfer_queue_, /*size=*/256);
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
  IREE_ASSERT_OK(CreatePassthroughPool(pool0.out()));
  IREE_ASSERT_OK(CreatePassthroughPool(pool1.out()));

  const iree_hal_pool_reservation_request_t request =
      MakeRequest(transfer_queue_, /*size=*/256);
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
      MakeRequest(transfer_queue_, kBlockSize),
      MakeRequest(transfer_queue_, kBlockSize),
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
  const iree_hal_pool_reservation_request_t request =
      MakeRequest(transfer_queue_, kBlockSize);

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
