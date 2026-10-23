// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

namespace {
constexpr iree_device_size_t kAllocationSize = 1024;

// Release callback that counts how many times it has been invoked.
static void CountingReleaseCallback(
    void* user_data, iree_hal_external_storage_class_t storage_class,
    iree_hal_external_storage_payload_t payload) {
  (void)storage_class;
  (void)payload;
  ++*static_cast<int*>(user_data);
}
}  // namespace

class PoolTest : public CtsTestBase<> {
 protected:
  iree_status_t CreatePool(iree_hal_buffer_usage_t usage,
                           iree_hal_pool_host_access_t host_access,
                           const iree_hal_slab_pool_options_t* options,
                           iree_hal_pool_t** out_pool) {
    *out_pool = nullptr;
    const iree_hal_device_queue_spec_t* queue_spec =
        iree_hal_device_spec_queues(iree_hal_device_spec(device_));
    std::vector<iree_hal_pool_family_access_t> families;
    families.reserve(queue_spec->family_count);
    for (iree_host_size_t i = 0; i < queue_spec->family_count; ++i) {
      if (queue_spec->families[i].provisioned_queue_count == 0) {
        continue;
      }
      families.push_back(iree_hal_pool_family_access_t{
          .family = iree_hal_device_queue_family(device_, i),
          .usage = usage,
      });
    }
    const iree_hal_pool_scope_t scope = {
        .family_count = families.size(),
        .families = families.data(),
        .host = host_access,
    };
    return iree_hal_slab_pool_create(device_group_, scope, options,
                                     iree_allocator_system(), out_pool);
  }

  iree_status_t CreateOwnedPool(iree_hal_buffer_usage_t usage,
                                iree_hal_pool_t** out_pool) {
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    options.trace_name = IREE_SV("cts.pool");
    return CreatePool(usage, /*host_access=*/{}, &options, out_pool);
  }

  iree_status_t CreateHostImportPool(iree_hal_pool_t** out_pool) {
    const iree_hal_pool_import_class_t import_class = {
        .storage_class = iree_hal_make_external_storage_class(
            IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
    };
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    options.owned.mode = IREE_HAL_SLAB_POOL_IMPORT_ONLY;
    options.imports.class_count = 1;
    options.imports.classes = &import_class;
    options.trace_name = IREE_SV("cts.pool.import");
    return CreatePool(IREE_HAL_BUFFER_USAGE_TRANSFER, /*host_access=*/{},
                      &options, out_pool);
  }

  struct HostImportGeometry {
    iree_device_size_t byte_length;
    iree_host_size_t alignment;
  };

  static HostImportGeometry QueryHostImportGeometry(iree_hal_pool_t* pool) {
    iree_hal_pool_import_requirements_t requirements;
    const bool supported = iree_hal_pool_query_import_requirements(
        pool,
        iree_hal_make_external_storage_class(
            IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
        &requirements);
    EXPECT_TRUE(supported);
    EXPECT_TRUE(iree_device_size_is_power_of_two(requirements.base_alignment));
    EXPECT_NE(requirements.byte_length_granularity, 0u);
    const iree_device_size_t byte_length =
        ((std::max(kAllocationSize, requirements.min_byte_length) - 1) /
             requirements.byte_length_granularity +
         1) *
        requirements.byte_length_granularity;
    EXPECT_TRUE(requirements.max_byte_length == 0 ||
                byte_length <= requirements.max_byte_length);
    const iree_device_size_t alignment =
        std::max<iree_device_size_t>(requirements.base_alignment, 64);
    EXPECT_LE(alignment, static_cast<iree_device_size_t>(IREE_HOST_SIZE_MAX));
    return {byte_length, static_cast<iree_host_size_t>(alignment)};
  }
};

TEST_P(PoolTest, BaselineConstruction) {
  for (iree_hal_buffer_usage_t usage :
       {IREE_HAL_BUFFER_USAGE_TRANSFER, IREE_HAL_BUFFER_USAGE_STORAGE}) {
    Ref<iree_hal_pool_t> pool;
    IREE_ASSERT_OK(CreateOwnedPool(usage, pool.out()));

    iree_hal_pool_capabilities_t capabilities;
    iree_hal_pool_query_capabilities(pool, &capabilities);
    EXPECT_TRUE(iree_all_bits_set(capabilities.supported_usage, usage));
    EXPECT_TRUE(capabilities.max_allocation_size == 0 ||
                capabilities.max_allocation_size >= kAllocationSize);
    EXPECT_GT(capabilities.max_allocation_alignment, 0u);

    Ref<iree_hal_buffer_t> buffer;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool, iree_hal_make_pool_reservation_request(kAllocationSize, 1),
        iree_infinite_timeout(), buffer.out()));
    EXPECT_TRUE(
        iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer), usage));
    EXPECT_GE(iree_hal_buffer_allocation_size(buffer), kAllocationSize);
    const auto memory = iree_hal_buffer_memory_view(buffer);
    ASSERT_NE(memory.backing, nullptr);
    EXPECT_GE(memory.backing->native_allocation.byte_length,
              memory.backing->native_allocation.byte_offset + kAllocationSize);
    EXPECT_TRUE(
        iree_device_size_is_power_of_two(memory.backing->allocation_alignment));
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(pool, &stats);
    EXPECT_EQ(stats.bytes_reserved,
              memory.backing->native_allocation.byte_length);
    EXPECT_EQ(stats.bytes_committed,
              memory.backing->native_allocation.byte_length);
  }
}

TEST_P(PoolTest, RejectsEmptyAllocation) {
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateOwnedPool(IREE_HAL_BUFFER_USAGE_TRANSFER, pool.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_pool_allocate_buffer(
                            pool, iree_hal_make_pool_reservation_request(0, 1),
                            iree_immediate_timeout(), buffer.out()));
  EXPECT_FALSE(buffer);
}

// Native registration is owned independently of the caller's release
// callback. Re-importing the same allocation verifies cleanup on backends that
// reject duplicate registration.
TEST_P(PoolTest, ImportHostAllocationNullCallback) {
  Ref<iree_hal_pool_t> pool;
  iree_status_t create_status = CreateHostImportPool(pool.out());
  if (iree_status_is_unavailable(create_status)) {
    iree::Status unavailable_status(std::move(create_status));
    GTEST_SKIP() << FormatSkipStatus(unavailable_status);
  }
  IREE_ASSERT_OK(create_status);
  const iree_hal_external_storage_class_t storage_class =
      iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER);
  iree_hal_pool_import_requirements_t requirements;
  ASSERT_TRUE(iree_hal_pool_query_import_requirements(pool, storage_class,
                                                      &requirements));

  const HostImportGeometry geometry = QueryHostImportGeometry(pool);
  const iree_device_size_t allocation_size = geometry.byte_length;
  ASSERT_LE(allocation_size,
            static_cast<iree_device_size_t>(IREE_HOST_SIZE_MAX));
  void* host_ptr = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc_aligned(
      iree_allocator_system(), static_cast<iree_host_size_t>(allocation_size),
      geometry.alignment, /*offset=*/0, &host_ptr));

  auto make_external_storage = [&]() {
    iree_hal_external_storage_t storage = {
        .storage_class = storage_class,
    };
    storage.payload.host_pointer = host_ptr;
    storage.byte_length = allocation_size;
    storage.host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK;
    storage.maintenance_range = iree_hal_make_buffer_range(0, allocation_size);
    return storage;
  };

  iree_hal_external_storage_t storage = make_external_storage();
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_import_buffer(
      pool, &storage, iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
      buffer.out()));
  EXPECT_EQ(IREE_HAL_EXTERNAL_STORAGE_TYPE_NONE, storage.storage_class.type);
  buffer.reset();
  pool.reset();  // Joins native attachment retirement.

  IREE_ASSERT_OK(CreateHostImportPool(pool.out()));
  storage = make_external_storage();
  IREE_ASSERT_OK(iree_hal_pool_import_buffer(
      pool, &storage, iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
      buffer.out()));
  buffer.reset();
  pool.reset();
  iree_allocator_free_aligned(iree_allocator_system(), host_ptr);
}

// Native registration cleanup and the caller's release callback are separate
// ownership obligations. The callback runs exactly once after native
// attachment retirement.
TEST_P(PoolTest, ImportHostAllocationWithCallback) {
  Ref<iree_hal_pool_t> pool;
  iree_status_t create_status = CreateHostImportPool(pool.out());
  if (iree_status_is_unavailable(create_status)) {
    iree::Status unavailable_status(std::move(create_status));
    GTEST_SKIP() << FormatSkipStatus(unavailable_status);
  }
  IREE_ASSERT_OK(create_status);
  const iree_hal_external_storage_class_t storage_class =
      iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER);
  const HostImportGeometry geometry = QueryHostImportGeometry(pool);
  const iree_device_size_t allocation_size = geometry.byte_length;
  ASSERT_LE(allocation_size,
            static_cast<iree_device_size_t>(IREE_HOST_SIZE_MAX));
  void* host_ptr = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc_aligned(
      iree_allocator_system(), static_cast<iree_host_size_t>(allocation_size),
      geometry.alignment, /*offset=*/0, &host_ptr));

  int release_count = 0;
  iree_hal_external_storage_t storage = {
      .storage_class = storage_class,
  };
  storage.payload.host_pointer = host_ptr;
  storage.byte_length = allocation_size;
  storage.host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK;
  storage.maintenance_range = iree_hal_make_buffer_range(0, allocation_size);
  storage.release = CountingReleaseCallback;
  storage.release_user_data = &release_count;

  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_import_buffer(
      pool, &storage, iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
      buffer.out()));
  EXPECT_EQ(0, release_count);
  buffer.reset();
  pool.reset();
  EXPECT_EQ(1, release_count);

  // The same native storage can be attached again after retirement. This
  // import has no callback because the first callback did not own host_ptr.
  IREE_ASSERT_OK(CreateHostImportPool(pool.out()));
  storage = {};
  storage.storage_class = storage_class;
  storage.payload.host_pointer = host_ptr;
  storage.byte_length = allocation_size;
  storage.host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK;
  storage.maintenance_range = iree_hal_make_buffer_range(0, allocation_size);
  IREE_ASSERT_OK(iree_hal_pool_import_buffer(
      pool, &storage, iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
      buffer.out()));
  buffer.reset();
  pool.reset();
  EXPECT_EQ(1, release_count);
  iree_allocator_free_aligned(iree_allocator_system(), host_ptr);
}

CTS_REGISTER_TEST_SUITE(PoolTest);

}  // namespace iree::hal::cts
