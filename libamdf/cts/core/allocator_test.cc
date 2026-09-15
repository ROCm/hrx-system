// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#include <malloc.h>
#endif

#include "amdf/amdf.h"
#include "gtest/gtest.h"
#include "util/device_cache.h"
#include "util/provider.h"

namespace {

struct RecordingAllocator {
  std::atomic<uint64_t> allocation_attempt_count{0};
  std::atomic<uint64_t> successful_allocation_count{0};
  std::atomic<uint64_t> free_count{0};
  std::atomic<uint64_t> live_allocation_count{0};
  std::atomic<uint64_t> failed_ordinal{UINT64_MAX};

  static void* AMDF_CALL Allocate(void* user_data, uint64_t byte_length,
                                  uint64_t minimum_alignment) {
    auto* self = static_cast<RecordingAllocator*>(user_data);
    const uint64_t ordinal =
        self->allocation_attempt_count.fetch_add(1, std::memory_order_relaxed);
    if (ordinal == self->failed_ordinal.load(std::memory_order_relaxed)) {
      return nullptr;
    }
#if defined(_WIN32)
    void* pointer = _aligned_malloc(static_cast<size_t>(byte_length),
                                    static_cast<size_t>(minimum_alignment));
#else
    void* pointer = nullptr;
    if (posix_memalign(&pointer, static_cast<size_t>(minimum_alignment),
                       static_cast<size_t>(byte_length)) != 0) {
      pointer = nullptr;
    }
#endif
    if (pointer != nullptr) {
      self->successful_allocation_count.fetch_add(1, std::memory_order_relaxed);
      self->live_allocation_count.fetch_add(1, std::memory_order_relaxed);
    }
    return pointer;
  }

  static void AMDF_CALL Free(void* user_data, void* pointer) {
    if (pointer == nullptr) return;
    auto* self = static_cast<RecordingAllocator*>(user_data);
    self->free_count.fetch_add(1, std::memory_order_relaxed);
    self->live_allocation_count.fetch_sub(1, std::memory_order_relaxed);
#if defined(_WIN32)
    _aligned_free(pointer);
#else
    std::free(pointer);
#endif
  }

  void FailNextAllocation() {
    failed_ordinal.store(
        allocation_attempt_count.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
  }

  amdf_allocator_t MakeAllocator() {
    return {
        .user_data = this,
        .allocate = Allocate,
        .free = Free,
    };
  }
};

const amdf_api_t* QueryApi() {
  const amdf_api_t* api = nullptr;
  EXPECT_TRUE(amdf_status_is_ok(amdf_cts_provider_query_api()(
      AMDF_ABI_VERSION_1, AMDF_ABI_VERSION_LATEST, &api)));
  return api;
}

amdf_instance_create_info_t MakeInstanceCreateInfo(
    RecordingAllocator* allocator) {
  amdf_instance_create_info_t create_info = {};
  create_info.type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.structure_size = sizeof(create_info);
  create_info.host_allocator = allocator->MakeAllocator();
  create_info.native_lifetime = GetCtsDeviceCache().native_lifetime();
  return create_info;
}

TEST(AllocatorTest, RejectsMalformedConfigurationWithoutPublishingInstance) {
  const amdf_api_t* api = QueryApi();
  ASSERT_NE(api, nullptr);
  RecordingAllocator allocator;
  amdf_instance_create_info_t create_info = MakeInstanceCreateInfo(&allocator);
  auto* const sentinel = reinterpret_cast<amdf_instance_t*>(uintptr_t{1});
  amdf_instance_t* instance = sentinel;

  create_info.host_allocator.free = nullptr;
  EXPECT_EQ(amdf_status_code(api->instance_create(&create_info, &instance)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(instance, sentinel);

  create_info = MakeInstanceCreateInfo(&allocator);
  create_info.host_allocator.allocate = nullptr;
  EXPECT_EQ(amdf_status_code(api->instance_create(&create_info, &instance)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(instance, sentinel);

  create_info = {};
  create_info.type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.structure_size = sizeof(create_info);
  create_info.host_allocator.user_data = &allocator;
  EXPECT_EQ(amdf_status_code(api->instance_create(&create_info, &instance)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(instance, sentinel);
  EXPECT_EQ(allocator.allocation_attempt_count.load(), 0u);

  create_info = MakeInstanceCreateInfo(&allocator);
  create_info.native_lifetime = UINT32_MAX;
  EXPECT_EQ(amdf_status_code(api->instance_create(&create_info, &instance)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(instance, sentinel);
  EXPECT_EQ(allocator.allocation_attempt_count.load(), 0u);
}

TEST(AllocatorTest, FirstAllocationFailureDoesNotPublishOrRequireCleanup) {
  const amdf_api_t* api = QueryApi();
  ASSERT_NE(api, nullptr);
  RecordingAllocator allocator;
  allocator.FailNextAllocation();
  const amdf_instance_create_info_t create_info =
      MakeInstanceCreateInfo(&allocator);
  auto* const sentinel = reinterpret_cast<amdf_instance_t*>(uintptr_t{1});
  amdf_instance_t* instance = sentinel;

  EXPECT_EQ(amdf_status_code(api->instance_create(&create_info, &instance)),
            AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  EXPECT_EQ(instance, sentinel);
  EXPECT_EQ(allocator.allocation_attempt_count.load(), 1u);
  EXPECT_EQ(allocator.successful_allocation_count.load(), 0u);
  EXPECT_EQ(allocator.free_count.load(), 0u);
  EXPECT_EQ(allocator.live_allocation_count.load(), 0u);
}

TEST(AllocatorTest, PlatformAllocationFailureReleasesPrivateConstruction) {
  const amdf_api_t* api = QueryApi();
  ASSERT_NE(api, nullptr);
  RecordingAllocator allocator;
  allocator.failed_ordinal.store(1);
  const amdf_instance_create_info_t create_info =
      MakeInstanceCreateInfo(&allocator);
  auto* const sentinel = reinterpret_cast<amdf_instance_t*>(uintptr_t{1});
  amdf_instance_t* instance = sentinel;

  EXPECT_EQ(amdf_status_code(api->instance_create(&create_info, &instance)),
            AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  EXPECT_EQ(instance, sentinel);
  EXPECT_EQ(allocator.allocation_attempt_count.load(), 2u);
  EXPECT_EQ(allocator.successful_allocation_count.load(), 1u);
  EXPECT_EQ(allocator.free_count.load(), 1u);
  EXPECT_EQ(allocator.live_allocation_count.load(), 0u);
}

TEST(AllocatorTest, IndependentInstancesRetainTheirOwnAllocator) {
  const amdf_api_t* api = QueryApi();
  ASSERT_NE(api, nullptr);
  RecordingAllocator first_allocator;
  RecordingAllocator second_allocator;
  const amdf_instance_create_info_t first_create_info =
      MakeInstanceCreateInfo(&first_allocator);
  const amdf_instance_create_info_t second_create_info =
      MakeInstanceCreateInfo(&second_allocator);
  amdf_instance_t* first_instance = nullptr;
  amdf_instance_t* second_instance = nullptr;

  ASSERT_EQ(api->instance_create(&first_create_info, &first_instance),
            AMDF_STATUS_OK);
  ASSERT_EQ(api->instance_create(&second_create_info, &second_instance),
            AMDF_STATUS_OK);
  ASSERT_NE(first_instance, nullptr);
  ASSERT_NE(second_instance, nullptr);
  EXPECT_NE(first_instance, second_instance);
  EXPECT_EQ(first_allocator.live_allocation_count.load(), 2u);
  EXPECT_EQ(second_allocator.live_allocation_count.load(), 2u);

  EXPECT_EQ(api->instance_destroy(second_instance), AMDF_STATUS_OK);
  EXPECT_EQ(second_allocator.live_allocation_count.load(), 0u);
  EXPECT_EQ(second_allocator.free_count.load(), 2u);
  EXPECT_EQ(first_allocator.live_allocation_count.load(), 2u);

  EXPECT_EQ(api->instance_destroy(first_instance), AMDF_STATUS_OK);
  EXPECT_EQ(first_allocator.live_allocation_count.load(), 0u);
  EXPECT_EQ(first_allocator.free_count.load(), 2u);
}

TEST(AllocatorTest, EnumerationAllocationFailurePreservesOutputs) {
  const amdf_api_t* api = QueryApi();
  ASSERT_NE(api, nullptr);
  RecordingAllocator allocator;
  const amdf_instance_create_info_t create_info =
      MakeInstanceCreateInfo(&allocator);
  amdf_instance_t* instance = nullptr;
  ASSERT_EQ(api->instance_create(&create_info, &instance), AMDF_STATUS_OK);
  ASSERT_NE(instance, nullptr);
  const uint64_t live_allocation_count = allocator.live_allocation_count.load();

  amdf_endpoint_summary_t summary;
  std::memset(&summary, 0xA5, sizeof(summary));
  const amdf_endpoint_summary_t original_summary = summary;
  uint32_t endpoint_count = 123;
  allocator.FailNextAllocation();
  const amdf_status_t status =
      api->endpoint_enumerate(instance, 1, &summary, &endpoint_count);
  if (amdf_status_code(status) == AMDF_STATUS_CODE_UNSUPPORTED) {
    EXPECT_EQ(std::memcmp(&summary, &original_summary, sizeof(summary)), 0);
    EXPECT_EQ(endpoint_count, 123u);
    EXPECT_EQ(api->instance_destroy(instance), AMDF_STATUS_OK);
    GTEST_SKIP() << "native endpoint discovery is unavailable";
  }
  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_API);
  EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  EXPECT_EQ(std::memcmp(&summary, &original_summary, sizeof(summary)), 0);
  EXPECT_EQ(endpoint_count, 123u);
  EXPECT_EQ(allocator.live_allocation_count.load(), live_allocation_count);
  EXPECT_EQ(api->instance_destroy(instance), AMDF_STATUS_OK);
  EXPECT_EQ(allocator.live_allocation_count.load(), 0u);
}

}  // namespace
