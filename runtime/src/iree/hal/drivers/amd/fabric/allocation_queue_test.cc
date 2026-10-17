// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/allocation_queue.h"

#include <array>
#include <future>
#include <thread>
#include <utility>
#include <vector>

#include "iree/hal/drivers/amd/fabric/slab_provider.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct NativeMemory;

// The native API is the dependency. Requests, worker ownership, and callbacks
// below use the production allocation queue and native memory preparation.
struct NativeAllocator {
  NativeAllocator() {
    api.memory_create = Allocate;
    api.memory_query_info = Query;
    api.memory_destroy = Free;
    options.scope = reinterpret_cast<amdf_memory_scope_t*>(this);
    options.profile.ordinal = 7;
    options.profile.roles = AMDF_MEMORY_PROFILE_ROLE_CREATE;
    options.profile.allocation.byte_length_granularity = 64;
    options.profile.allocation.minimum_alignment = 64;
    options.profile.allocation.maximum_byte_length = 4096;
  }

  static amdf_status_t AMDF_CALL Allocate(amdf_memory_scope_t* scope,
                                          const amdf_memory_create_info_t* info,
                                          amdf_memory_t** out_memory);
  static amdf_status_t AMDF_CALL Query(amdf_memory_t* memory,
                                       amdf_memory_info_t* out_info);
  static amdf_status_t AMDF_CALL Free(amdf_memory_t* memory);

  // Native functions consumed by the production backing owner.
  amdf_api_t api = {};
  // Captured native policy borrowed by acquisition requests.
  iree_hal_amd_fabric_memory_options_t options = {};
  // Native create invocation on which to report resource exhaustion, or zero.
  uint32_t fail_on_allocate = 0;
  // Native preparation fails after a handle has been allocated.
  bool fail_query = false;
  // Number of native allocation attempts on the worker.
  uint32_t allocate_count = 0;
  // Number of native owners consumed by release or rollback.
  uint32_t free_count = 0;
  // Worker thread observed by each native call.
  std::vector<std::thread::id> threads;
  // Optional deterministic gate at the native allocation boundary.
  std::promise<void>* entered = nullptr;
  // Explicit permission to complete the held native allocation.
  std::shared_future<void> resume;
};

struct NativeMemory {
  // Dependency owner, borrowed through native destruction.
  NativeAllocator* allocator;
  // Achieved geometry returned by native preparation.
  uint64_t byte_length;
};

amdf_status_t AMDF_CALL NativeAllocator::Allocate(
    amdf_memory_scope_t* scope, const amdf_memory_create_info_t* info,
    amdf_memory_t** out_memory) {
  auto* self = reinterpret_cast<NativeAllocator*>(scope);
  self->threads.push_back(std::this_thread::get_id());
  ++self->allocate_count;
  if (self->entered) {
    self->entered->set_value();
    self->resume.wait();
    self->entered = nullptr;
  }
  if (self->allocate_count == self->fail_on_allocate) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  EXPECT_EQ(info->memory_profile_ordinal, 7u);
  EXPECT_EQ(info->byte_length % 64, 0u);
  EXPECT_GE(info->minimum_alignment, 64u);
  *out_memory = reinterpret_cast<amdf_memory_t*>(
      new NativeMemory{self, info->byte_length});
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL NativeAllocator::Query(amdf_memory_t* handle,
                                               amdf_memory_info_t* out_info) {
  auto* memory = reinterpret_cast<NativeMemory*>(handle);
  if (memory->allocator->fail_query) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  out_info->byte_length = memory->byte_length;
  out_info->alignment = 64;
  out_info->native_allocation_byte_length = memory->byte_length;
  out_info->native_allocation_granularity = 64;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL NativeAllocator::Free(amdf_memory_t* handle) {
  auto* memory = reinterpret_cast<NativeMemory*>(handle);
  ++memory->allocator->free_count;
  memory->allocator->threads.push_back(std::this_thread::get_id());
  delete memory;
  return AMDF_STATUS_OK;
}

struct Acquire {
  explicit Acquire(NativeAllocator* native) {
    for (size_t i = 0; i < geometries.size(); ++i) {
      geometries[i] = {&native->options, 65 + 64 * i, 16};
    }
    request.operation.api = &native->api;
    request.operation.event_sink = iree_hal_device_event_sink_discard();
    request.count = geometries.size();
    request.requests = geometries.data();
    request.memories = memories.data();
    request.completion = {
        .fn =
            [](void* user_data, iree_status_t status) {
              auto* self = static_cast<Acquire*>(user_data);
              self->completed.set_value(iree::Status(std::move(status)));
            },
        .user_data = this,
    };
  }

  // Captured native geometry, borrowed until completion.
  std::array<iree_hal_amd_fabric_memory_request_t, 3> geometries;
  // Native outputs become owned only after the batch succeeds.
  std::array<iree_hal_amd_fabric_memory_t, 3> memories = {};
  // Production descriptor, submitted without any callback executor.
  iree_hal_amd_fabric_memory_acquire_request_t request = {};
  // Explicit host observation of terminal acquisition.
  std::promise<iree::Status> completed;
};

class AllocationQueueTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_amd_fabric_allocation_queue_create(
        UINT32_MAX, iree_allocator_system(), &queue_));
  }

  void TearDown() override {
    iree_hal_amd_fabric_allocation_queue_release(queue_);
  }

  void Release(iree_hal_amd_fabric_memory_t* memories, size_t count) {
    std::promise<void> completed;
    iree_hal_amd_fabric_memory_release_request_t request = {
        .operation =
            {
                .api = &native_.api,
                .event_sink = iree_hal_device_event_sink_discard(),
            },
        .count = count,
        .memories = memories,
        .completion =
            {
                .fn =
                    [](void* user_data) {
                      static_cast<std::promise<void>*>(user_data)->set_value();
                    },
                .user_data = &completed,
            },
    };
    iree_hal_amd_fabric_allocation_queue_release_memory(queue_, &request);
    completed.get_future().get();
  }

  // Native dependency outlives queue teardown and all callbacks.
  NativeAllocator native_;
  // Owned production worker under test.
  iree_hal_amd_fabric_allocation_queue_t* queue_ = nullptr;
};

TEST_F(AllocationQueueTest, AcquireReleaseBatchUsesNativeWorker) {
  Acquire acquire(&native_);
  const auto caller = std::this_thread::get_id();
  iree_hal_amd_fabric_allocation_queue_acquire_memory(queue_, &acquire.request);
  IREE_ASSERT_OK(acquire.completed.get_future().get());
  for (size_t i = 0; i < acquire.memories.size(); ++i) {
    EXPECT_NE(acquire.memories[i].handle, nullptr);
    EXPECT_EQ(acquire.memories[i].info.byte_length, 128 + 64 * i);
  }
  Release(acquire.memories.data(), acquire.memories.size());
  EXPECT_EQ(native_.allocate_count, 3u);
  EXPECT_EQ(native_.free_count, 3u);
  for (auto& memory : acquire.memories) {
    EXPECT_EQ(memory.handle, nullptr);
  }
  ASSERT_EQ(native_.threads.size(), 6u);
  for (auto thread : native_.threads) {
    EXPECT_NE(thread, caller);
    EXPECT_EQ(thread, native_.threads.front());
  }
}

TEST_F(AllocationQueueTest, FailedBatchRollsBackItsAcquiredPrefix) {
  native_.fail_on_allocate = 2;
  Acquire acquire(&native_);
  iree_hal_amd_fabric_allocation_queue_acquire_memory(queue_, &acquire.request);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        acquire.completed.get_future().get());
  EXPECT_EQ(native_.allocate_count, 2u);
  EXPECT_EQ(native_.free_count, 1u);
  for (auto& memory : acquire.memories) {
    EXPECT_EQ(memory.handle, nullptr);
  }
}

TEST_F(AllocationQueueTest, FailedPreparationReleasesItsNativeHandle) {
  native_.fail_query = true;
  Acquire acquire(&native_);
  iree_hal_amd_fabric_allocation_queue_acquire_memory(queue_, &acquire.request);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        acquire.completed.get_future().get());
  EXPECT_EQ(native_.allocate_count, 1u);
  EXPECT_EQ(native_.free_count, 1u);
  for (auto& memory : acquire.memories) {
    EXPECT_EQ(memory.handle, nullptr);
  }
}

TEST_F(AllocationQueueTest, HeldNativeAllocationDoesNotBlockSubmission) {
  std::promise<void> entered;
  std::promise<void> resume;
  native_.entered = &entered;
  native_.resume = resume.get_future().share();
  Acquire first(&native_);
  Acquire second(&native_);
  iree_hal_amd_fabric_allocation_queue_acquire_memory(queue_, &first.request);
  entered.get_future().get();
  // If submission enters the native API or joins the active worker, this
  // cannot return to release the explicit gate. There is no timed readiness.
  iree_hal_amd_fabric_allocation_queue_acquire_memory(queue_, &second.request);
  resume.set_value();
  IREE_ASSERT_OK(first.completed.get_future().get());
  IREE_ASSERT_OK(second.completed.get_future().get());
  Release(first.memories.data(), first.memories.size());
  Release(second.memories.data(), second.memories.size());
  EXPECT_EQ(native_.allocate_count, 6u);
  EXPECT_EQ(native_.free_count, 6u);
}

TEST_F(AllocationQueueTest, SlabsAndPrivateBackingShareOneNativeWorker) {
  iree_hal_mock_device_options_t device_options;
  iree_hal_mock_device_options_initialize(&device_options);
  iree_hal_device_t* device = nullptr;
  IREE_ASSERT_OK(iree_hal_mock_device_create(&device_options,
                                             iree_allocator_system(), &device));
  const iree_hal_amd_fabric_slab_provider_options_t provider_options = {
      .memory = native_.options,
      .host_binding_index = IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE,
      .memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .maintenance_alignment = 1,
  };
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_fabric_slab_provider_create(
      device, &native_.api, iree_hal_device_event_sink_discard(), queue_,
      &provider_options, iree_allocator_system(), &provider));
  iree_hal_device_release(device);

  std::promise<void> entered;
  std::promise<void> resume;
  native_.entered = &entered;
  native_.resume = resume.get_future().share();
  Acquire first(&native_);
  Acquire second(&native_);
  iree_hal_amd_fabric_allocation_queue_acquire_memory(queue_, &first.request);
  entered.get_future().get();
  iree_hal_amd_fabric_allocation_queue_acquire_memory(queue_, &second.request);
  iree_hal_slab_t slab = {};
  const iree_hal_slab_request_t geometry = {256, 64};
  std::promise<iree::Status> acquired;
  iree_hal_slab_acquire_request_t acquire = {
      .count = 1,
      .requests = &geometry,
      .slabs = &slab,
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
  iree_hal_slab_provider_acquire_slabs(provider, &acquire);
  resume.set_value();
  IREE_ASSERT_OK(first.completed.get_future().get());
  IREE_ASSERT_OK(second.completed.get_future().get());
  IREE_ASSERT_OK(acquired.get_future().get());
  Release(first.memories.data(), first.memories.size());
  Release(second.memories.data(), second.memories.size());
  std::promise<void> released;
  iree_hal_slab_release_request_t release = {
      .count = 1,
      .slabs = &slab,
      .completion =
          {
              .fn =
                  [](void* user_data) {
                    static_cast<std::promise<void>*>(user_data)->set_value();
                  },
              .user_data = &released,
          },
  };
  iree_hal_slab_provider_release_slabs(provider, &release);
  released.get_future().get();
  iree_hal_slab_provider_release(provider);
  EXPECT_EQ(native_.allocate_count, 7u);
  EXPECT_EQ(native_.free_count, 7u);
  for (auto thread : native_.threads) {
    EXPECT_NE(thread, std::this_thread::get_id());
    EXPECT_EQ(thread, native_.threads.front());
  }
}

TEST_F(AllocationQueueTest, CompletionReusesDescriptorAndReleasesFinalOwner) {
  struct Cycle {
    // Queue's external reference transfers into this callback-owned state.
    iree_hal_amd_fabric_allocation_queue_t* queue;
    // Borrowed API and policy, live through the final callback.
    NativeAllocator* native;
    // One native generation per acquisition/release cycle.
    iree_hal_amd_fabric_memory_t memory = {};
    // Native geometry borrowed by acquisition.
    iree_hal_amd_fabric_memory_request_t geometry = {};
    // Exactly one request is outstanding; the callback reuses its storage.
    union {
      iree_hal_amd_fabric_memory_acquire_request_t acquire;
      iree_hal_amd_fabric_memory_release_request_t release;
    } request = {};
    // Remaining native generations to acquire and release.
    uint32_t remaining = 32;
    // Final callback transfers completion before deleting its owner.
    std::promise<void> completed;

    void AcquireNext() {
      geometry = {&native->options, 65, 16};
      request.acquire = {
          .operation =
              {
                  .api = &native->api,
                  .event_sink = iree_hal_device_event_sink_discard(),
              },
          .count = 1,
          .requests = &geometry,
          .memories = &memory,
          .completion = {.fn = Acquired, .user_data = this},
      };
      iree_hal_amd_fabric_allocation_queue_acquire_memory(queue,
                                                          &request.acquire);
    }

    static void Acquired(void* user_data, iree_status_t status) {
      auto* self = static_cast<Cycle*>(user_data);
      IREE_EXPECT_OK(status);
      self->request.release = {
          .operation =
              {
                  .api = &self->native->api,
                  .event_sink = iree_hal_device_event_sink_discard(),
              },
          .count = 1,
          .memories = &self->memory,
          .completion = {.fn = Released, .user_data = self},
      };
      iree_hal_amd_fabric_allocation_queue_release_memory(
          self->queue, &self->request.release);
    }

    static void Released(void* user_data) {
      auto* self = static_cast<Cycle*>(user_data);
      if (--self->remaining) {
        self->AcquireNext();
      } else {
        auto completed = std::move(self->completed);
        iree_hal_amd_fabric_allocation_queue_release(self->queue);
        delete self;
        completed.set_value();
      }
    }
  };
  auto* cycle = new Cycle{queue_, &native_};
  queue_ = nullptr;
  auto completed = cycle->completed.get_future();
  cycle->AcquireNext();
  completed.get();
  EXPECT_EQ(native_.allocate_count, 32u);
  EXPECT_EQ(native_.free_count, 32u);
}

}  // namespace
