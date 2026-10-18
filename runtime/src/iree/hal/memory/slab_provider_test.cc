// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/slab_provider.h"

#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_hal_asan_pool_options_t ShadowOptions() {
  iree_hal_asan_pool_options_t options = {
      .mode = IREE_HAL_ASAN_POOL_MODE_SHADOW,
      .shadow_granule_size = 8,
      .redzone_size = 16,
  };
  return options;
}

TEST(SlabProviderTest, CPUPropertiesDescribeAtomicStorage) {
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(
      /*min_alignment=*/0, iree_allocator_system(), &provider));

  iree_hal_slab_provider_properties_t properties;
  iree_hal_slab_provider_query_properties(provider, &properties);
  EXPECT_EQ(properties.memory_type, IREE_HAL_CPU_SLAB_PROVIDER_MEMORY_TYPE);
  EXPECT_TRUE(iree_all_bits_set(properties.memory_type,
                                IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                                    IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
                                    IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));
  EXPECT_TRUE(iree_all_bits_set(properties.supported_usage,
                                IREE_HAL_BUFFER_USAGE_STORAGE));

  const iree_hal_atomic_operation_flags_t expected_32 =
      iree_atomic_int32_is_lock_free() ? IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL
                                       : 0;
  EXPECT_EQ(properties.atomic_operations.device_scope_32, expected_32);
  EXPECT_EQ(properties.atomic_operations.system_scope_32, expected_32);

  const iree_hal_atomic_operation_flags_t expected_64 =
      iree_atomic_int64_is_lock_free() ? IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL
                                       : 0;
  EXPECT_EQ(properties.atomic_operations.device_scope_64, expected_64);
  EXPECT_EQ(properties.atomic_operations.system_scope_64, expected_64);

  iree_hal_slab_provider_release(provider);
}

TEST(SlabProviderTest, CPUAlignmentMatchesActualStorage) {
  const iree_host_size_t alignments[] = {0, 1, 64, 256, 4096};
  for (iree_host_size_t alignment : alignments) {
    SCOPED_TRACE(alignment);
    iree_hal_slab_provider_t* provider = nullptr;
    IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(
        alignment, iree_allocator_system(), &provider));
    iree_hal_slab_provider_properties_t properties;
    iree_hal_slab_provider_query_properties(provider, &properties);
    EXPECT_EQ(properties.allocation_alignment,
              iree_max(alignment, IREE_HAL_HEAP_BUFFER_ALIGNMENT));

    for (iree_host_size_t requested_alignment : {1, 64, 256, 4096}) {
      SCOPED_TRACE(requested_alignment);
      EXPECT_GE(properties.max_allocation_alignment, requested_alignment);
      iree_hal_slab_t slab = {};
      IREE_ASSERT_OK(iree_hal_slab_provider_acquire_slab_sync(
          provider, 127, requested_alignment, &slab));
      EXPECT_EQ(
          reinterpret_cast<uintptr_t>(slab.base_ptr) %
              iree_max(requested_alignment, properties.allocation_alignment),
          0u);
      EXPECT_GE(slab.length, 127u);
      EXPECT_EQ(slab.native_allocation.byte_length, slab.length);
      EXPECT_EQ(slab.native_allocation.byte_offset, 0u);
      EXPECT_EQ(slab.native_allocation.granularity, 1u);
      EXPECT_EQ(slab.allocation_alignment,
                iree_max(requested_alignment, properties.allocation_alignment));
      EXPECT_EQ(slab.host_cacheability, IREE_HAL_HOST_CACHEABILITY_WRITE_BACK);
      memset(slab.base_ptr, 0x6B, 127);
      for (iree_host_size_t i = 0; i < 127; ++i) {
        EXPECT_EQ(slab.base_ptr[i], 0x6B);
      }
      iree_hal_slab_provider_release_slab_sync(provider, &slab);
    }
    iree_hal_slab_provider_release(provider);
  }
}

TEST(SlabProviderTest, CPUImportPreservesSourceFactsWithoutInventingGeometry) {
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(
      /*min_alignment=*/0, iree_allocator_system(), &provider));
  alignas(128) uint8_t bytes[256] = {};
  for (iree_device_size_t maintenance_length : {0, 64}) {
    iree_hal_external_storage_t storage = {
        .storage_class = iree_hal_make_external_storage_class(
            IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
    };
    storage.payload.host_pointer = bytes + 64;
    storage.byte_length = 128;
    storage.host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK;
    storage.maintenance_range =
        iree_hal_make_buffer_range(32, maintenance_length);
    storage.physical_backing_id = {{0xC, 0x1234}};
    iree_hal_imported_slab_t imported;
    IREE_ASSERT_OK(
        iree_hal_slab_provider_import_slab(provider, &storage, &imported));
    EXPECT_EQ(imported.slab.base_ptr, bytes + 64);
    EXPECT_EQ(imported.slab.native_allocation.byte_length, 0u);
    EXPECT_EQ(imported.slab.native_allocation.byte_offset, 0u);
    EXPECT_EQ(imported.slab.native_allocation.granularity, 0u);
    EXPECT_EQ(imported.slab.allocation_alignment, 64u);
    EXPECT_EQ(imported.slab.host_cacheability,
              IREE_HAL_HOST_CACHEABILITY_WRITE_BACK);
    EXPECT_EQ(imported.slab.maintenance_range.byte_offset, 32u);
    EXPECT_EQ(imported.slab.maintenance_range.byte_length, maintenance_length);
    EXPECT_EQ(imported.slab.physical_backing_id.words[0], 0xCu);
    EXPECT_EQ(imported.slab.physical_backing_id.words[1], 0x1234u);
    iree_hal_slab_provider_release_slab_sync(provider, &imported.slab);
    EXPECT_EQ(storage.payload.host_pointer, bytes + 64);
    EXPECT_EQ(storage.byte_length, 128u);
  }
  iree_hal_slab_provider_release(provider);
}

TEST(SlabProviderTest, CPUAlignmentMustBeZeroOrPowerOfTwo) {
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_cpu_slab_provider_create(3, iree_allocator_system(), &provider));
  EXPECT_EQ(provider, nullptr);
}

// Counts real provider allocation ownership and rejects one selected native
// allocation. The provider implementation remains the subject under test.
class NativeAllocator {
 public:
  iree_allocator_t allocator() { return {this, Control}; }

  // Successful allocations allowed before the next forced failure.
  int allocations_remaining = -1;
  // Number of allocations still owned by the provider and its slabs.
  size_t live_count = 0;

 private:
  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_pointer) {
    auto* allocator = static_cast<NativeAllocator*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE) {
      if (allocator->allocations_remaining == 0) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "native allocation unavailable");
      }
      if (allocator->allocations_remaining > 0) {
        --allocator->allocations_remaining;
      }
    }
    const bool had_pointer = *inout_pointer != nullptr;
    auto system = iree_allocator_system();
    iree_status_t status =
        system.ctl(system.self, command, params, inout_pointer);
    if (iree_status_is_ok(status)) {
      if (command == IREE_ALLOCATOR_COMMAND_FREE && had_pointer) {
        --allocator->live_count;
      } else if (command != IREE_ALLOCATOR_COMMAND_FREE && !had_pointer) {
        ++allocator->live_count;
      }
    }
    return status;
  }
};

class SlabProviderBatchTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    if (GetParam()) {
      IREE_ASSERT_OK(iree_hal_shared_slab_provider_create(
          allocator_.allocator(), &provider_));
    } else {
      IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(
          64, allocator_.allocator(), &provider_));
    }
  }
  void TearDown() override {
    iree_hal_slab_provider_release(provider_);
    EXPECT_EQ(allocator_.live_count, 0u);
  }

  // Real allocator dependency used for both host provider families.
  NativeAllocator allocator_;
  // Owned CPU or shared-section provider.
  iree_hal_slab_provider_t* provider_ = nullptr;
};

TEST_P(SlabProviderBatchTest,
       CompletesWholeBatchesAndReleasesWithoutAllocation) {
  const iree_hal_slab_request_t geometry[] = {
      {127, 64}, {8192, 4096}, {1025, 256}};
  iree_hal_slab_t slabs[3];
  int acquired = 0;
  iree_hal_slab_acquire_request_t acquire = {
      .count = 3,
      .requests = geometry,
      .slabs = slabs,
      .completion =
          {
              .fn =
                  [](void* user_data, iree_status_t status) {
                    IREE_EXPECT_OK(status);
                    ++*static_cast<int*>(user_data);
                  },
              .user_data = &acquired,
          },
  };
  iree_hal_slab_provider_acquire_slabs(provider_, &acquire);
  ASSERT_EQ(acquired, 1);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_GE(slabs[i].length, geometry[i].min_length);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(slabs[i].base_ptr) %
                  geometry[i].min_alignment,
              0u);
    memset(slabs[i].base_ptr, 0xA7, geometry[i].min_length);
    for (size_t j = 0; j < geometry[i].min_length; ++j) {
      EXPECT_EQ(slabs[i].base_ptr[j], 0xA7);
    }
  }
  allocator_.allocations_remaining = 0;
  int released = 0;
  iree_hal_slab_release_request_t release = {
      .count = 3,
      .slabs = slabs,
      .completion =
          {
              .fn = [](void* user_data) { ++*static_cast<int*>(user_data); },
              .user_data = &released,
          },
  };
  iree_hal_slab_provider_release_slabs(provider_, &release);
  EXPECT_EQ(released, 1);
  EXPECT_EQ(allocator_.live_count, 1u);
}

TEST_P(SlabProviderBatchTest, FailureReturnsNativePrefixBeforeCompletion) {
  const iree_hal_slab_request_t geometry[] = {
      {4096, 64}, {8192, 4096}, {1025, 256}};
  iree_hal_slab_t slabs[3];
  memset(slabs, 0xA5, sizeof(slabs));
  allocator_.allocations_remaining = 2;
  int completions = 0;
  iree_hal_slab_acquire_request_t acquire = {
      .count = 3,
      .requests = geometry,
      .slabs = slabs,
      .completion =
          {
              .fn =
                  [](void* user_data, iree_status_t status) {
                    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                                          status);
                    ++*static_cast<int*>(user_data);
                  },
              .user_data = &completions,
          },
  };
  iree_hal_slab_provider_acquire_slabs(provider_, &acquire);
  EXPECT_EQ(completions, 1);
  EXPECT_EQ(allocator_.live_count, 1u);
  iree_hal_slab_t zeroed[3] = {};
  EXPECT_EQ(memcmp(slabs, zeroed, sizeof(slabs)), 0);

  allocator_.allocations_remaining = -1;
  acquire.completion.fn = [](void* user_data, iree_status_t status) {
    IREE_EXPECT_OK(status);
    ++*static_cast<int*>(user_data);
  };
  iree_hal_slab_provider_acquire_slabs(provider_, &acquire);
  EXPECT_EQ(completions, 2);
  int released = 0;
  iree_hal_slab_release_request_t release = {
      .count = 3,
      .slabs = slabs,
      .completion =
          {
              .fn = [](void* user_data) { ++*static_cast<int*>(user_data); },
              .user_data = &released,
          },
  };
  iree_hal_slab_provider_release_slabs(provider_, &release);
  EXPECT_EQ(released, 1);
  EXPECT_EQ(allocator_.live_count, 1u);
}

INSTANTIATE_TEST_SUITE_P(CPUAndShared, SlabProviderBatchTest,
                         ::testing::Bool());

TEST(SlabProviderASANTest, DisabledOptionsAreAlwaysValid) {
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(
      /*min_alignment=*/0, iree_allocator_system(), &provider));

  iree_hal_asan_pool_options_t options = {};
  IREE_EXPECT_OK(
      iree_hal_slab_provider_validate_asan_options(provider, &options));

  iree_hal_slab_provider_release(provider);
}

TEST(SlabProviderASANTest, EnabledOptionsRequireProviderSupport) {
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(
      /*min_alignment=*/0, iree_allocator_system(), &provider));

  iree_hal_asan_pool_options_t options = ShadowOptions();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_slab_provider_validate_asan_options(provider, &options));

  iree_hal_slab_provider_release(provider);
}

}  // namespace
