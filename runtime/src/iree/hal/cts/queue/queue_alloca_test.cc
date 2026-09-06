// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <vector>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/passthrough_pool.h"
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
      /*.params=*/MakeAllocationParams(queue),
      /*.allocation_size=*/size,
  };
}

}  // namespace

class QueueAllocaTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase<>::SetUp();
    if (this->IsSkipped()) return;
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
    return iree_hal_passthrough_pool_create(options, backend.slab_provider,
                                            backend.notification,
                                            iree_allocator_system(), out_pool);
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
    return iree_hal_tlsf_pool_create(options, backend.slab_provider,
                                     backend.notification, backend.epoch_query,
                                     iree_allocator_system(), out_pool);
  }

  iree_status_t CreateFixedBlockPool(iree_device_size_t block_size,
                                     uint32_t block_count,
                                     iree_hal_pool_epoch_query_t epoch_query,
                                     iree_hal_pool_t** out_pool) {
    iree_hal_queue_pool_backend_t backend = {};
    IREE_RETURN_IF_ERROR(QueryPoolBackend(&backend));
    if (!backend.slab_provider || !backend.notification) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "queue pool backend query returned an incomplete bundle");
    }
    iree_hal_fixed_block_pool_options_t options = {};
    options.block_allocator_options.block_size = block_size;
    options.block_allocator_options.block_count = block_count;
    options.block_allocator_options.frontier_capacity = 2;
    options.asan = backend.asan;
    return iree_hal_fixed_block_pool_create(options, backend.slab_provider,
                                            backend.notification, epoch_query,
                                            iree_allocator_system(), out_pool);
  }

  iree_hal_queue_t* FindSiblingQueue(iree_hal_queue_t* queue) {
    const iree_hal_queue_family_ordinal_t family_ordinal =
        iree_hal_queue_family_ordinal(iree_hal_queue_family(queue));
    const iree_hal_device_queue_spec_t* queue_spec =
        iree_hal_device_spec_queues(iree_hal_device_spec(device_));
    if (!queue_spec || family_ordinal >= queue_spec->family_count) return NULL;
    const uint32_t queue_count =
        queue_spec->families[family_ordinal].provisioned_queue_count;
    for (iree_hal_queue_ordinal_t queue_ordinal = 0;
         queue_ordinal < queue_count; ++queue_ordinal) {
      iree_hal_queue_t* candidate =
          iree_hal_device_queue(device_, family_ordinal, queue_ordinal);
      if (candidate != queue) return candidate;
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
        IREE_HAL_FILL_FLAG_NONE));
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

  const iree_hal_pool_reservation_request_t request =
      MakeRequest(transfer_queue_, kAllocationSize);
  Ref<iree_hal_buffer_t> buffer;
  SemaphoreList empty_wait;
  SemaphoreList alloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(
      iree_hal_queue_alloca(transfer_queue_, empty_wait, alloca_signal, pool,
                            /*request_count=*/1, &request, buffer.out()));
  ASSERT_NE(nullptr, buffer.get());
  Wait(alloca_signal);

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

TEST_P(QueueAllocaTest, PluralTransaction) {
  constexpr std::array<iree_device_size_t, 3> kAllocationSizes = {256, 512,
                                                                  1024};
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateTLSFPool(/*range_length=*/4096, pool.out()));

  std::array<iree_hal_pool_reservation_request_t, kAllocationSizes.size()>
      requests;
  for (size_t i = 0; i < requests.size(); ++i) {
    requests[i] = MakeRequest(transfer_queue_, kAllocationSizes[i]);
  }
  std::array<iree_hal_buffer_t*, kAllocationSizes.size()> raw_buffers = {};
  SemaphoreList empty_wait;
  SemaphoreList alloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_alloca(transfer_queue_, empty_wait,
                                       alloca_signal, pool, requests.size(),
                                       requests.data(), raw_buffers.data()));
  for (iree_hal_buffer_t* buffer : raw_buffers) ASSERT_NE(nullptr, buffer);

  std::array<Ref<iree_hal_buffer_t>, kAllocationSizes.size()> buffers = {
      Ref<iree_hal_buffer_t>(raw_buffers[0]),
      Ref<iree_hal_buffer_t>(raw_buffers[1]),
      Ref<iree_hal_buffer_t>(raw_buffers[2]),
  };
  Wait(alloca_signal);
  for (size_t i = 0; i < buffers.size(); ++i) {
    const uint32_t pattern = 0x11000000u + static_cast<uint32_t>(i);
    FillAndWait(transfer_queue_, buffers[i], pattern);
    EXPECT_EQ(pattern, ReadBufferData<uint32_t>(buffers[i])[0]);
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
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateFixedBlockPool(kBlockSize, /*block_count=*/2,
                                      iree_hal_pool_epoch_query_null(),
                                      pool.out()));

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
  for (iree_hal_buffer_t* buffer : outputs) iree_hal_buffer_release(buffer);
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
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateFixedBlockPool(kBlockSize, /*block_count=*/2,
                                      iree_hal_pool_epoch_query_null(),
                                      pool.out()));

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
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateFixedBlockPool(kBlockSize, /*block_count=*/1,
                                      iree_hal_pool_epoch_query_null(),
                                      pool.out()));
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
